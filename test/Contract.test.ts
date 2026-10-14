/**
 * Drive the real Python HTTP API (with model stand-ins) through the typed client,
 * so the Effect contract in src/Api.ts and engine/ cannot drift apart.
 */
import { afterAll, beforeAll, expect, test } from "bun:test"
import { BunServices } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { type ChildProcess, spawn } from "node:child_process"
import { existsSync } from "node:fs"
import { join, resolve } from "node:path"
import { Engine } from "../src/Engine.ts"
import { encodeWav } from "../src/Reading.ts"
import { fakePaths, temporaryDirectory } from "./support.ts"

const engineRoot = resolve(import.meta.dir, "..", "engine")
const python = join(engineRoot, ".venv", "bin", "python")
const available = existsSync(python)
const data = temporaryDirectory()
let server: ChildProcess | undefined

const run = <A, E>(effect: Effect.Effect<A, E, Engine>) =>
  Effect.runPromise(effect.pipe(
    Effect.provide(Engine.layer.pipe(Layer.provide(fakePaths(data)), Layer.provide(BunServices.layer)))
  ))

beforeAll(async () => {
  if (!available) return
  server = spawn(python, ["tests/contract_server.py", data], { cwd: engineRoot, stdio: "inherit" })
  for (let attempt = 0; attempt < 100; attempt++) {
    if (Option.isSome(await run(Effect.flatMap(Engine, (engine) => engine.probe)))) return
    await Bun.sleep(100)
  }
  throw new Error("The contract server did not start")
})

afterAll(() => server?.kill())

const sine = (seconds: number) =>
  encodeWav(Int16Array.from({ length: seconds * 48000 }, (_, i) => Math.round(6000 * Math.sin(i / 20))), 48000)

const words = (text: string) =>
  text.split(" ").map((word, index) => ({ text: word, start: index * 0.5, end: index * 0.5 + 0.4 }))

const recording = (takes: ReadonlyArray<{ script: string; text: string; seconds: number }>) => {
  const form = new FormData()
  form.append("takes", JSON.stringify(takes.map(({ script, text }) => ({ script, transcript: text, words: words(text) }))))
  for (const { script, seconds } of takes) {
    form.append("audio", new Blob([sine(seconds) as Uint8Array<ArrayBuffer>], { type: "audio/wav" }), `${script}.wav`)
  }
  return form
}

test.skipIf(!available)("the engine implements the CLI's contract", async () => {
  await run(Effect.gen(function*() {
    const engine = yield* Engine
    const status = yield* engine.ensure
    expect(status).toMatchObject({ service: "idiolect", api: 1, model: "voxcpm2" })

    const noVoice = yield* engine.call(engine.api.voice()).pipe(Effect.flip)
    expect(noVoice.message).toBe("Record your voice first with idiolect setup.")
    const voice = yield* engine.call(engine.api.replaceVoice({
      payload: recording([
        { script: "comma-1", text: "A short reference recording, read aloud slowly and carefully by me.", seconds: 6 },
        { script: "technical", text: "Run the build, check the tests, then merge it into main.", seconds: 5 }
      ])
    }))
    expect(voice.takes).toEqual([{ script: "comma-1", duration: 6 }, { script: "technical", duration: 5 }])
    expect(yield* engine.call(engine.api.voice())).toEqual(voice)
    const tooShort = yield* engine.call(engine.api.replaceVoice({
      payload: recording([{ script: "comma-1", text: "Too short.", seconds: 1 }])
    })).pipe(Effect.flip)
    expect(tooShort.message).toContain("between 3 and 90 seconds")

    const job = yield* engine.call(engine.api.generate({ payload: { text: "A proper script." } }))
    expect(job).toMatchObject({ status: "queued", total: 1 })
    expect((yield* engine.call(engine.api.render({ params: { id: job.id } }))).id).toBe(job.id)
    const notReady = yield* engine.call(engine.api.audio({ params: { id: job.id } })).pipe(Effect.flip)
    expect(notReady.message).toBe("This render is not ready yet")
    const blank = yield* engine.call(engine.api.generate({ payload: { text: "   " } })).pipe(Effect.flip)
    expect(blank.message).toContain("text")

    const busy = yield* engine.call(engine.api.shutdown()).pipe(Effect.flip)
    expect(busy.message).toContain("active render")
    expect((yield* engine.call(engine.api.cancel({ params: { id: job.id } }))).status).toBe("cancelled")

    const { words } = yield* engine.call(engine.api.transcribe({
      query: { sample_rate: 48000 },
      payload: new Uint8Array(48000 * 2)
    }))
    expect(words).toEqual([{ text: " Hello", start: 0.5, end: 0.9 }])

    expect(yield* engine.call(engine.api.shutdown())).toEqual({ stopping: true })
    for (let attempt = 0; attempt < 50 && Option.isSome(yield* engine.probe); attempt++) {
      yield* Effect.sleep("100 millis")
    }
    expect(Option.isNone(yield* engine.probe)).toBe(true)
  }))
}, 30_000)
