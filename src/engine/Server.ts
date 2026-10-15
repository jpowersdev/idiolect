/**
 * The engine's HTTP API (src/Api.ts), served on the data directory's Unix socket. One engine runs
 * per data directory; it exits after ten idle minutes, releasing the GPU memory the models hold.
 */
import { BunHttpServer } from "@effect/platform-bun"
import { Deferred, Effect, Layer } from "effect"
import { HttpRouter } from "effect/http"
import { HttpApiBuilder } from "effect/http-api"
import { chmodSync, existsSync, openSync, readFileSync, rmSync, statSync, writeSync, closeSync } from "node:fs"
import { join } from "node:path"
import { ApiVersion, Busy, Conflict, EngineApi, Invalid, MaxScriptLength, NotFound, TooLarge } from "../Api.ts"
import { SpeechModel } from "../Models.ts"
import type { Local } from "./Local.ts"
import * as Renders from "./Renders.ts"
import * as Voice from "./Voice.ts"
import packageJson from "../../package.json" with { type: "json" }

// A 90-second take at 48 kHz 16-bit mono is about 8.6 MB.
const MAX_TRANSCRIBE_BYTES = 12 * 1024 * 1024
const MAX_UPLOAD_BYTES = 32 * 1024 * 1024
export const IDLE_TIMEOUT = 10 * 60 * 1000

export const socketPath = (data: string) => join(data, "engine.sock")

/**
 * Hold the data directory's engine lock: a file naming the process that owns it. A lock left by
 * an engine that has exited is taken over.
 */
export const lock = (data: string) =>
  Effect.acquireRelease(
    Effect.try({
      try: () => {
        const path = join(data, "engine.lock")
        for (let attempt = 0; attempt < 2; attempt++) {
          try {
            const file = openSync(path, "wx", 0o600)
            writeSync(file, `${process.pid}\n`)
            closeSync(file)
            return path
          } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error
            const owner = Number.parseInt(readFileSync(path, "utf8"), 10)
            if (Number.isInteger(owner) && owner > 0 && alive(owner)) {
              throw new Error("Idiolect is already running with this data directory.")
            }
            rmSync(path, { force: true })
          }
        }
        throw new Error("Could not take the engine lock.")
      },
      catch: (error) => error instanceof Error ? error : new Error(String(error))
    }),
    (path) => Effect.sync(() => rmSync(path, { force: true }))
  )

const alive = (pid: number) => {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM"
  }
}

export interface Options {
  readonly data: string
  readonly local: Local
  readonly renders: Renders.Renders
  /** Completed to stop the engine. */
  readonly stop: Deferred.Deferred<void>
  readonly idleTimeout?: number
}

const handlers = (options: Options, touch: Effect.Effect<void>) => {
  const { data, local, renders, stop } = options
  const saved = Effect.suspend(() => {
    const voice = Voice.read(data)
    return voice ? Effect.succeed(voice) : Effect.fail(new NotFound({ detail: "Record your voice first with idiolect setup." }))
  })
  const job = (id: string) => {
    const render = renders.get(id)
    return render ? Effect.succeed(render) : Effect.fail(new NotFound({ detail: "Render not found" }))
  }
  return HttpApiBuilder.group(EngineApi, "engine", (group) =>
    group
      .handle("status", () =>
        Effect.as(touch, {
          service: "idiolect" as const,
          api: ApiVersion,
          version: packageJson.version,
          pid: process.pid,
          model: SpeechModel.folder,
          ...local.status()
        }))
      .handle("shutdown", () =>
        Effect.gen(function*() {
          if (renders.active() > 0) return yield* new Conflict({ detail: "The engine has an active render." })
          // Answer first; the engine stops just after.
          yield* Effect.forkDetach(Effect.andThen(Effect.sleep("50 millis"), Deferred.succeed(stop, undefined)))
          return { stopping: true }
        }))
      .handle("transcribe", ({ payload, query }) =>
        Effect.gen(function*() {
          yield* touch
          if (payload.length > MAX_TRANSCRIBE_BYTES) return yield* new TooLarge({ detail: "Send at most 12 MB of audio per transcription." })
          if (query.sample_rate < 8000 || query.sample_rate > 96000) return yield* new Invalid({ detail: "sample_rate: use a rate between 8000 and 96000." })
          const pcm = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
          const audio = Float32Array.from({ length: Math.floor(payload.length / 2) }, (_, i) => pcm.getInt16(i * 2, true) / 32768)
          const words = yield* local.transcribe(audio, query.sample_rate).pipe(
            Effect.catch((error) => error instanceof Conflict ? Effect.fail(error) : Effect.die(error))
          )
          return { words }
        }))
      .handle("voice", () => Effect.andThen(touch, Effect.map(saved, Voice.summary)))
      .handleRaw("replaceVoice", ({ request }) =>
        Effect.gen(function*() {
          yield* touch
          const parts = yield* Effect.mapError(request.multipart, () => new Invalid({ detail: "Send the takes and their recordings as a form." }))
          const takes = parts.takes
          if (typeof takes !== "string") return yield* new Invalid({ detail: "Invalid takes: send them as one JSON field." })
          const files = Array.isArray(parts.audio) ? parts.audio : []
          const recordings: Array<Uint8Array> = []
          for (const file of files) {
            if (typeof file === "string") return yield* new Invalid({ detail: "Send each recording as a file." })
            if (statSync(file.path).size > MAX_UPLOAD_BYTES) return yield* new Invalid({ detail: "Each recording must be smaller than 32 MB." })
            recordings.push(readFileSync(file.path))
          }
          return Voice.summary(yield* Voice.replace(data, yield* Voice.parseTakes(takes), recordings))
        }).pipe(Effect.scoped))
      .handle("generate", ({ payload }) =>
        Effect.gen(function*() {
          yield* touch
          const text = payload.text.trim()
          if (text.length === 0) return yield* new Invalid({ detail: "text: Write a script first" })
          if (text.length > MaxScriptLength) return yield* new Invalid({ detail: `text: Use at most ${MaxScriptLength} characters.` })
          yield* saved
          if (!local.status().model_downloaded) return yield* new Conflict({ detail: "The speech model is missing. Run idiolect setup first." })
          if (renders.active() >= Renders.MAX_ACTIVE) return yield* new Busy({ detail: "The queue is full. Wait for a render to finish." })
          return yield* renders.submit(text)
        }))
      .handle("render", ({ params }) => Effect.andThen(touch, job(params.id)))
      .handle("cancel", ({ params }) => Effect.andThen(touch, Effect.map(job(params.id), () => renders.cancel(params.id)!)))
      .handle("audio", ({ params }) =>
        Effect.gen(function*() {
          yield* touch
          const render = yield* job(params.id)
          const audio = renders.audio(params.id)
          if (render.status !== "ready" || audio === undefined) return yield* new Conflict({ detail: "This render is not ready yet" })
          return audio
        }))
  )
}

/** Serve until stopped, or until idle: no requests and no renders for `idleTimeout`. */
export const serve = (options: Options) =>
  Effect.gen(function*() {
    const socket = socketPath(options.data)
    // The lock is held, so a socket left behind is from an engine that has exited.
    rmSync(socket, { force: true })
    let lastRequest = Date.now()
    const touch = Effect.sync(() => {
      lastRequest = Date.now()
    })
    const api = HttpApiBuilder.layer(EngineApi).pipe(Layer.provide(handlers(options, touch)))
    yield* Layer.build(
      HttpRouter.serve(api, { disableLogger: true, disableListenLog: true }).pipe(
        Layer.provide(BunHttpServer.layer({ unix: socket }))
      )
    )
    chmodSync(socket, 0o600)
    yield* Effect.addFinalizer(() => Effect.sync(() => existsSync(socket) && rmSync(socket)))
    const idleTimeout = options.idleTimeout ?? IDLE_TIMEOUT
    yield* Effect.forkScoped(Effect.forever(Effect.gen(function*() {
      yield* Effect.sleep(Math.min(30_000, idleTimeout))
      if (Date.now() - lastRequest >= idleTimeout && options.renders.active() === 0) yield* Deferred.succeed(options.stop, undefined)
    })))
    yield* Deferred.await(options.stop)
  })
