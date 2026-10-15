import { Deferred, Effect, Exit, Fiber, Layer, Option } from "effect"
import { mkdtempSync } from "node:fs"
import { realpathSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { Render, Word } from "../src/Api.ts"
import { Audio, type AudioShape } from "../src/Audio.ts"
import { Engine, type EngineShape, explain } from "../src/Engine.ts"
import { Paths } from "../src/Paths.ts"
import type { Local } from "../src/engine/Local.ts"
import * as Renders from "../src/engine/Renders.ts"
import * as Server from "../src/engine/Server.ts"

export const temporaryDirectory = () => realpathSync(mkdtempSync(join(tmpdir(), "idiolect-test-")))

export const render = (changes: Partial<Render> = {}): Render => ({
  id: "take123",
  completed: 0,
  total: 1,
  status: "queued",
  message: "Waiting for the speech engine",
  duration: null,
  ...changes
})

/** Words spoken half a second apart, starting at `start` seconds. */
export const words = (text: string, start = 1): Array<Word> =>
  text.split(" ").map((word, index) => ({ text: word, start: start + index * 0.5, end: start + (index + 1) * 0.5 }))

/** An engine whose API methods are the given stand-ins. */
export const fakeEngine = (api: Partial<Record<keyof EngineShape["api"], (...args: Array<any>) => Effect.Effect<any, any>>>) =>
  Layer.succeed(
    Engine,
    Engine.of({
      api: api as unknown as EngineShape["api"],
      call: (effect) => Effect.mapError(effect, explain),
      probe: Effect.succeed(Option.none()),
      ensure: Effect.die("not used")
    })
  )

export const fakeAudio = (audio: Partial<AudioShape>) =>
  Layer.succeed(Audio, Audio.of({ check: Effect.void, play: () => Effect.void, record: () => Effect.die("not used"), ...audio }))

export const fakePaths = (data: string) => Layer.succeed(Paths, Paths.of({ data }))

/** Models that hear "Hello" in anything, and speak silence until their render is cancelled. */
export const standInModels = (changes: Partial<Local> = {}): Local => ({
  rate: 24000,
  load: Effect.void,
  speak: (_text, _seed, cancelled) =>
    Effect.gen(function*() {
      while (!cancelled()) yield* Effect.sleep("10 millis")
      return new Float32Array(2400)
    }),
  transcribe: () => Effect.succeed([{ text: "Hello", start: 0.5, end: 0.9 }]),
  status: () => ({ model_downloaded: true, recognizer_downloaded: true, model_status: "idle" }),
  ...changes
})

/** Serve the real engine API on `data`'s socket, in this process, with stand-in models. */
export const startEngine = async (data: string, options: { readonly local?: Local; readonly idleTimeout?: number } = {}) => {
  const local = options.local ?? standInModels()
  const stop = Effect.runSync(Deferred.make<void>())
  const fiber = Effect.runFork(Effect.scoped(Effect.gen(function*() {
    yield* Server.lock(data)
    const renders = yield* Renders.make(local)
    yield* Server.serve({ data, local, renders, stop, ...(options.idleTimeout ? { idleTimeout: options.idleTimeout } : {}) })
  })))
  const socket = Server.socketPath(data)
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await fetch("http://localhost/api/status", { unix: socket } as RequestInit).then((r) => r.ok, () => false)) break
    if (fiber.pollUnsafe() !== undefined) break
    await Bun.sleep(20)
  }
  return {
    socket,
    /** The engine's exit, once it stops by itself or is stopped. */
    exited: Effect.runPromise(Fiber.await(fiber)).then(Exit.isSuccess),
    stop: () => Effect.runPromise(Deferred.succeed(stop, undefined)).then(() => Effect.runPromise(Fiber.await(fiber)))
  }
}
