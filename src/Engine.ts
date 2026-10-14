/** Find, start, and talk to the local speech engine over its Unix socket. */
import { Context, Duration, Effect, Layer, Option } from "effect"
import { FetchHttpClient, HttpClient, HttpClientError } from "effect/http"
import { HttpApiClient } from "effect/http-api"
import { ChildProcess, ChildProcessSpawner } from "effect/process"
import { spawn } from "node:child_process"
import { closeSync, openSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { ApiVersion, EngineApi, EngineGroup, type Status } from "./Api.ts"
import { fail, IdiolectError } from "./Errors.ts"
import { Paths, type PathsShape } from "./Paths.ts"

type Client = HttpApiClient.Client<typeof EngineGroup>

/** The engine executable, found on PATH. */
export const EngineCommand = "idiolect-engine"

export interface EngineShape {
  /** The typed engine API. Wrap calls with `call` to explain failures. */
  readonly api: Client["engine"]
  readonly call: <A, E>(effect: Effect.Effect<A, E>) => Effect.Effect<A, IdiolectError>
  /** The engine's status, or None when no engine is running. */
  readonly probe: Effect.Effect<Option.Option<Status>, IdiolectError>
  /** Start the engine in the background unless a compatible one is already running. */
  readonly ensure: Effect.Effect<Status, IdiolectError>
  /** Run `idiolect-engine download|doctor` attached to this terminal. */
  readonly foreground: (args: ReadonlyArray<string>) => Effect.Effect<number, IdiolectError>
}

/** The model IDIOLECT_MODEL asks for, when set; otherwise the engine chooses for the hardware. */
export const requestedModel = (env: Readonly<Record<string, string | undefined>>) =>
  env.IDIOLECT_MODEL?.trim().toLowerCase() || undefined

/** IDIOLECT_MODEL set: the engine must use that model. Unset: one chosen for the hardware. */
export const usesWantedModel = (status: Pick<Status, "model" | "model_requested">, wanted: string | undefined) =>
  wanted === undefined ? !status.model_requested : status.model === wanted

export const socketPath = (paths: PathsShape) => join(paths.data, "engine.sock")

const notInstalled = fail(
  `The speech engine (${EngineCommand}) is not on your PATH. Install Idiolect with Nix, ` +
    "or install the engine with `uv tool install ./engine` from a checkout."
)

export const engineEnvironment = (paths: PathsShape): Record<string, string> => ({
  IDIOLECT_DATA: paths.data,
  HF_HUB_DISABLE_TELEMETRY: "1",
  DO_NOT_TRACK: "1",
  TOKENIZERS_PARALLELISM: "false",
  OMP_NUM_THREADS: "8"
})

/** The engine's final error from its log, e.g. another engine already owning the data. */
const lastError = (log: string) => {
  try {
    const line = readFileSync(log, "utf8").split("\n").findLast((line) => /^\w*(Error|Exception): /.test(line))
    return line ? `: ${line.replace(/^\w+: /, "").replace(/\.$/, "")}` : ""
  } catch {
    return ""
  }
}

const isRecord = (u: unknown): u is Record<string, unknown> => typeof u === "object" && u !== null

/** Turn transport, decoding, and engine errors into one message. */
export const explain = (error: unknown): IdiolectError => {
  if (error instanceof IdiolectError) return error
  if (isRecord(error) && typeof error.detail === "string") return fail(error.detail)
  if (HttpClientError.isHttpClientError(error)) return fail(`The local engine did not answer: ${error.message}`)
  return fail(`Unexpected response from the local engine: ${String(error)}`)
}

const poll = <A>(attempts: number, check: Effect.Effect<Option.Option<A>, IdiolectError>) =>
  Effect.gen(function*() {
    for (let attempt = 0; attempt < attempts; attempt++) {
      const result = yield* check
      if (Option.isSome(result)) return result
      yield* Effect.sleep(Duration.millis(100))
    }
    return Option.none<A>()
  })

export class Engine extends Context.Service<Engine, EngineShape>()("idiolect/Engine") {
  static readonly layer = Layer.effect(
    Engine,
    Effect.gen(function*() {
      const paths = yield* Paths
      const spawner = yield* ChildProcessSpawner.ChildProcessSpawner
      const socket = socketPath(paths)
      const client = yield* HttpApiClient.make(EngineApi, {
        baseUrl: "http://localhost",
        // Bun's fetch connects to a Unix socket with its `unix` option.
        transformClient: HttpClient.transform((request) =>
          Effect.provideService(request, FetchHttpClient.RequestInit, { unix: socket } as RequestInit)
        )
      })
      const api = client.engine
      const call = <A, E>(effect: Effect.Effect<A, E>) => Effect.mapError(effect, explain)
      const environment = engineEnvironment(paths)
      const log = join(paths.data, "engine.log")

      const status = api.status().pipe(
        Effect.timeout(Duration.seconds(2)),
        Effect.map(Option.some),
        Effect.catch((error): Effect.Effect<Option.Option<Status>, IdiolectError> =>
          HttpClientError.isHttpClientError(error) && error.reason._tag === "TransportError"
            ? Effect.succeedNone
            : Effect.fail(explain(error))
        )
      )

      /** A running engine that is not the one wanted (say, before an upgrade) is replaced once idle. */
      const retire = (why: string) =>
        Effect.gen(function*() {
          yield* call(api.shutdown()).pipe(
            Effect.mapError(() => fail(`The engine ${why} is still rendering. Try again when it finishes.`))
          )
          const stopped = yield* poll(
            100,
            Effect.map(status, (running) => Option.isNone(running) ? Option.some(true) : Option.none())
          )
          if (Option.isNone(stopped)) return yield* fail(`The engine ${why} did not stop. See ${log}`)
        })

      const probe = Effect.gen(function*() {
        const running = yield* status
        if (Option.isNone(running)) return running
        if (running.value.api !== ApiVersion) {
          yield* retire("from another Idiolect version")
          return Option.none<Status>()
        }
        if (!usesWantedModel(running.value, requestedModel(process.env))) {
          yield* retire(`using ${running.value.model}`)
          return Option.none<Status>()
        }
        return running
      })

      const installed = Effect.sync(() => Bun.which(EngineCommand) !== null)

      const start = Effect.try({
        try: () => {
          const output = openSync(log, "a")
          try {
            // Detached so the engine keeps the model loaded after this command exits.
            // It exits by itself once idle.
            const child = spawn(EngineCommand, ["serve"], {
              detached: true,
              stdio: ["ignore", output, output],
              cwd: paths.data,
              env: { ...process.env, ...environment }
            })
            const state = { exited: false }
            child.on("exit", () => {
              state.exited = true
            })
            child.on("error", () => {
              state.exited = true
            })
            child.unref()
            return state
          } finally {
            closeSync(output)
          }
        },
        catch: (error) => fail(`The local engine could not start: ${String(error)}`)
      })

      const ensure = Effect.gen(function*() {
        const running = yield* probe
        if (Option.isSome(running)) return running.value
        if (!(yield* installed)) return yield* notInstalled
        process.stderr.write("Starting the local voice engine…\n")
        const child = yield* start
        // Another command may start an engine at the same moment; either one will do.
        for (let attempt = 0; attempt < 300; attempt++) {
          const started = yield* status
          if (Option.isSome(started)) return started.value
          if (child.exited && attempt > 20) break
          yield* Effect.sleep(Duration.millis(100))
        }
        return yield* fail(`The local engine could not start${lastError(log)}. See ${log}.`)
      })

      const foreground = (args: ReadonlyArray<string>) =>
        Effect.gen(function*() {
          if (!(yield* installed)) return yield* notInstalled
          const handle = yield* spawner.spawn(
            ChildProcess.make(EngineCommand, args, {
              env: environment,
              extendEnv: true,
              stdin: "inherit",
              stdout: "inherit",
              stderr: "inherit"
            })
          )
          return Number(yield* handle.exitCode)
        }).pipe(
          Effect.scoped,
          Effect.mapError((error) =>
            error instanceof IdiolectError ? error : fail(`The engine command failed: ${error.message}`)
          )
        )

      return Engine.of({ api, call, probe, ensure, foreground })
    })
  ).pipe(Layer.provide(FetchHttpClient.layer))
}
