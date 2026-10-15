/**
 * Drive the real engine API (with stand-in models) through the CLI's typed client, so the
 * client and server sides of src/Api.ts cannot drift apart.
 */
import { expect, test } from "bun:test"
import { BunServices } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { Engine } from "../src/Engine.ts"
import { encodeWav } from "../src/Reading.ts"
import { fakePaths, startEngine, temporaryDirectory } from "./support.ts"

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

test("the engine implements the CLI's contract", async () => {
  const data = temporaryDirectory()
  const served = await startEngine(data)
  await Effect.runPromise(Effect.gen(function*() {
    const engine = yield* Engine
    const status = yield* engine.ensure
    expect(status).toMatchObject({ service: "idiolect", api: 2, model: "voxcpm2", model_downloaded: true })

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
    yield* engine.call(engine.api.cancel({ params: { id: job.id } }))
    let cancelled = yield* engine.call(engine.api.render({ params: { id: job.id } }))
    for (let attempt = 0; attempt < 100 && cancelled.status !== "cancelled"; attempt++) {
      yield* Effect.sleep("20 millis")
      cancelled = yield* engine.call(engine.api.render({ params: { id: job.id } }))
    }
    expect(cancelled.status).toBe("cancelled")

    const { words } = yield* engine.call(engine.api.transcribe({
      query: { sample_rate: 48000 },
      payload: new Uint8Array(48000 * 2)
    }))
    expect(words).toEqual([{ text: "Hello", start: 0.5, end: 0.9 }])

    expect(yield* engine.call(engine.api.shutdown())).toEqual({ stopping: true })
    for (let attempt = 0; attempt < 50 && Option.isSome(yield* engine.probe); attempt++) {
      yield* Effect.sleep("100 millis")
    }
    expect(Option.isNone(yield* engine.probe)).toBe(true)
  }).pipe(Effect.provide(Engine.layer.pipe(Layer.provide(fakePaths(data)), Layer.provide(BunServices.layer)))))
  expect(await served.exited).toBe(true)
}, 30_000)
