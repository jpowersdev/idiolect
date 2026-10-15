/** Find, start, and talk to the local speech engine over its Unix socket. */
import { Context, Duration, Effect, Layer, Option } from "effect"
import { FetchHttpClient, HttpClient, HttpClientError } from "effect/http"
import { HttpApiClient } from "effect/http-api"
import { spawn } from "node:child_process"
import { closeSync, openSync, readFileSync } from "node:fs"
import { join, resolve } from "node:path"
import { ApiVersion, EngineApi, EngineGroup, type Status } from "./Api.ts"
import { fail, IdiolectError } from "./Errors.ts"
import { Paths, type PathsShape } from "./Paths.ts"
import packageJson from "../package.json" with { type: "json" }

export const Version = packageJson.version
/** The engine runs from this same package, with the Bun running the CLI. */
export const EngineEntry = resolve(import.meta.dir, "engine", "main.ts")

type Client = HttpApiClient.Client<typeof EngineGroup>

export interface EngineShape {
  /** The typed engine API. Wrap calls with `call` to explain failures. */
  readonly api: Client["engine"]
  readonly call: <A, E>(effect: Effect.Effect<A, E>) => Effect.Effect<A, IdiolectError>
  /** The engine's status, or None when no engine is running. */
  readonly probe: Effect.Effect<Option.Option<Status>, IdiolectError>
  /** Start the engine in the background unless a compatible one is already running. */
  readonly ensure: Effect.Effect<Status, IdiolectError>
}

export const socketPath = (paths: PathsShape) => join(paths.data, "engine.sock")

/** The engine's final error from its log, e.g. another engine already owning the data. */
const lastError = (log: string) => {
  try {
    const line = readFileSync(log, "utf8").split("\n").findLast((line) => /^Error: /.test(line))
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

      /** A running engine from another release (say, before an upgrade) is replaced once idle. */
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
        if (running.value.api !== ApiVersion || running.value.version !== Version) {
          yield* retire("from another Idiolect version")
          return Option.none<Status>()
        }
        return running
      })

      const start = Effect.try({
        try: () => {
          const output = openSync(log, "a")
          try {
            // Detached so the engine keeps the model loaded after this command exits.
            // It exits by itself once idle.
            const child = spawn(process.execPath, [EngineEntry], {
              detached: true,
              stdio: ["ignore", output, output],
              cwd: paths.data,
              env: { ...process.env, IDIOLECT_DATA: paths.data }
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

      return Engine.of({ api, call, probe, ensure })
    })
  ).pipe(Layer.provide(FetchHttpClient.layer))
}
