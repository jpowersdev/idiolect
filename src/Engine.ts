/** Find, start, stop, and talk to the local speech engine. */
import { Context, Duration, Effect, Layer, Option } from "effect"
import { FetchHttpClient, HttpClientError } from "effect/http"
import { HttpApiClient } from "effect/http-api"
import { ChildProcess, ChildProcessSpawner } from "effect/process"
import { spawn } from "node:child_process"
import { closeSync, existsSync, mkdirSync, openSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { ApiVersion, EngineApi, EngineGroup, type Status } from "./Api.ts"
import { fail, IdiolectError } from "./Errors.ts"
import { Paths, type PathsShape } from "./Paths.ts"

type Client = HttpApiClient.Client<typeof EngineGroup>

export const DefaultPort = 7860

export interface EngineShape {
  readonly port: number
  /** The typed engine API. Wrap calls with `call` to explain failures. */
  readonly api: Client["engine"]
  readonly call: <A, E>(effect: Effect.Effect<A, E>) => Effect.Effect<A, IdiolectError>
  /** The engine's status, or None when nothing is listening on the port. */
  readonly probe: Effect.Effect<Option.Option<Status>, IdiolectError>
  /** Start the engine in the background unless it is already running. */
  readonly ensure: Effect.Effect<Status, IdiolectError>
  /** Ask an idle engine to exit. False when it was not running. */
  readonly stop: Effect.Effect<boolean, IdiolectError>
  /** Run `idiolect-engine serve|download|doctor` attached to this terminal. */
  readonly foreground: (args: ReadonlyArray<string>) => Effect.Effect<number, IdiolectError>
}

/** The engine executable: IDIOLECT_ENGINE, or idiolect-engine on PATH. */
export const engineCommand = (env: Readonly<Record<string, string | undefined>>) =>
  env.IDIOLECT_ENGINE || "idiolect-engine"

const notInstalled = (command: string) =>
  fail(
    `The speech engine (${command}) is not installed. Install it with ` +
      "`uv tool install ./engine` from the Idiolect checkout, or set IDIOLECT_ENGINE to its path."
  )

export const engineEnvironment = (paths: PathsShape, env = process.env): Record<string, string> => ({
  IDIOLECT_DATA: paths.data,
  IDIOLECT_MODELS: paths.models,
  HF_HUB_DISABLE_TELEMETRY: "1",
  DO_NOT_TRACK: "1",
  TOKENIZERS_PARALLELISM: "false",
  OMP_NUM_THREADS: env.OMP_NUM_THREADS || "8"
})

/** The engine's final error from its log, e.g. another engine already owning the library. */
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
export const explain = (port: number) => (error: unknown): IdiolectError => {
  if (error instanceof IdiolectError) return error
  if (isRecord(error) && typeof error.detail === "string") return fail(`Engine error: ${error.detail}`)
  if (HttpClientError.isHttpClientError(error)) {
    return fail(`The local engine on port ${port} did not answer: ${error.message}`)
  }
  return fail(`Unexpected response from the engine on port ${port}: ${String(error)}`)
}

export class Engine extends Context.Service<Engine, EngineShape>()("idiolect/Engine") {
  static readonly layer = (port: number) =>
    Layer.effect(
      Engine,
      Effect.gen(function*() {
        const paths = yield* Paths
        const spawner = yield* ChildProcessSpawner.ChildProcessSpawner
        const client = yield* HttpApiClient.make(EngineApi, { baseUrl: `http://127.0.0.1:${port}` })
        const api = client.engine
        const explainError = explain(port)
        const call = <A, E>(effect: Effect.Effect<A, E>) => Effect.mapError(effect, explainError)
        const command = engineCommand(process.env)
        const environment = engineEnvironment(paths)

        const probe = api.status().pipe(
          Effect.timeout(Duration.seconds(2)),
          Effect.map(Option.some),
          Effect.catch((error): Effect.Effect<Option.Option<Status>, IdiolectError> =>
            HttpClientError.isHttpClientError(error) && error.reason._tag === "TransportError"
              ? Effect.succeedNone
              : Effect.fail(fail(`Port ${port} is occupied by a different service. Choose another --port.`))
          ),
          Effect.flatMap((status): Effect.Effect<Option.Option<Status>, IdiolectError> => {
            if (Option.isNone(status)) return Effect.succeed(status)
            if (status.value.api !== ApiVersion) {
              return Effect.fail(fail(
                `The engine on port ${port} speaks API version ${status.value.api}, ` +
                  `but this CLI needs version ${ApiVersion}. Update both together.`
              ))
            }
            if (status.value.data_directory !== paths.data) {
              return Effect.fail(fail(
                `The engine on port ${port} uses a different voice library ` +
                  `(${status.value.data_directory}). Choose another --port or use the same IDIOLECT_DATA.`
              ))
            }
            return Effect.succeed(status)
          })
        )

        const installed = Effect.sync(() => command.includes("/") ? existsSync(command) : Bun.which(command) !== null)

        const start = Effect.try({
          try: () => {
            mkdirSync(paths.data, { recursive: true })
            const log = openSync(join(paths.data, "engine.log"), "a")
            try {
              // Detached so the engine keeps the model loaded after this command exits.
              const child = spawn(command, ["serve", "--port", String(port)], {
                detached: true,
                stdio: ["ignore", log, log],
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
              closeSync(log)
            }
          },
          catch: (error) => fail(`The local engine could not start: ${String(error)}`)
        })

        const ensure = Effect.gen(function*() {
          const running = yield* probe
          if (Option.isSome(running)) return running.value
          if (!(yield* installed)) return yield* notInstalled(command)
          yield* Effect.logDebug("starting engine")
          process.stderr.write("Starting the local voice engine…\n")
          const child = yield* start
          // Another command may start an engine at the same moment; either one will do.
          for (let attempt = 0; attempt < 300; attempt++) {
            const status = yield* probe
            if (Option.isSome(status)) return status.value
            if (child.exited && attempt > 20) break
            yield* Effect.sleep(Duration.millis(100))
          }
          const log = join(paths.data, "engine.log")
          return yield* fail(`The local engine could not start${lastError(log)}. See ${log}.`)
        })

        const stop = Effect.gen(function*() {
          if (Option.isNone(yield* probe)) return false
          yield* call(api.shutdown())
          for (let attempt = 0; attempt < 100; attempt++) {
            if (Option.isNone(yield* probe)) return true
            yield* Effect.sleep(Duration.millis(100))
          }
          return yield* fail("The engine accepted the request to stop but is still running.")
        })

        const foreground = (args: ReadonlyArray<string>) =>
          Effect.gen(function*() {
            if (!(yield* installed)) return yield* notInstalled(command)
            const handle = yield* spawner.spawn(
              ChildProcess.make(command, args, {
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

        return Engine.of({ port, api, call, probe, ensure, stop, foreground })
      })
    ).pipe(Layer.provide(FetchHttpClient.layer))
}
