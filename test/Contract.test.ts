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
const python = process.env.IDIOLECT_ENGINE_PYTHON || join(engineRoot, ".venv", "bin", "python")
const available = existsSync(python)
const port = 7900 + Math.floor(Math.random() * 90)
const data = temporaryDirectory()
let server: ChildProcess | undefined

const run = <A, E>(effect: Effect.Effect<A, E, Engine>) =>
  Effect.runPromise(effect.pipe(
    Effect.provide(Engine.layer(port).pipe(Layer.provide(fakePaths(data)), Layer.provide(BunServices.layer)))
  ))

beforeAll(async () => {
  if (!available) return
  server = spawn(python, ["tests/contract_server.py", String(port), data], { cwd: engineRoot, stdio: "inherit" })
  for (let attempt = 0; attempt < 100; attempt++) {
    if (Option.isSome(await run(Effect.flatMap(Engine, (engine) => engine.probe)))) return
    await Bun.sleep(100)
  }
  throw new Error("The contract server did not start")
})

afterAll(() => server?.kill())

const sine = (seconds: number) =>
  encodeWav(Int16Array.from({ length: seconds * 48000 }, (_, i) => Math.round(6000 * Math.sin(i / 20))), 48000)

test.skipIf(!available)("the engine implements the CLI's contract", async () => {
  await run(Effect.gen(function*() {
    const engine = yield* Engine
    const status = yield* engine.ensure
    expect(status).toMatchObject({ service: "idiolect", api: 1, data_directory: data })
    expect(yield* engine.call(engine.api.voices())).toEqual([])

    const form = new FormData()
    form.append("name", "Contract")
    form.append("transcript", "A short reference recording.")
    form.append("audio", new Blob([sine(4) as Uint8Array<ArrayBuffer>], { type: "audio/wav" }), "reference.wav")
    const voice = yield* engine.call(engine.api.addVoice({ payload: form }))
    expect(voice).toMatchObject({ name: "Contract", duration: 4 })

    const rejected = new FormData()
    rejected.append("name", "Short")
    rejected.append("transcript", "Too short.")
    rejected.append("audio", new Blob([sine(1) as Uint8Array<ArrayBuffer>], { type: "audio/wav" }), "reference.wav")
    const tooShort = yield* engine.call(engine.api.addVoice({ payload: rejected })).pipe(Effect.flip)
    expect(tooShort.message).toContain("between 3 and 30 seconds")

    const job = yield* engine.call(engine.api.generate({
      payload: { voice_id: voice.id, text: "A proper script.", language: "English", seed: 12 }
    }))
    expect(job).toMatchObject({ status: "queued", text: "A proper script.", total: 1 })
    expect((yield* engine.call(engine.api.render({ params: { id: job.id } }))).id).toBe(job.id)
    expect((yield* engine.call(engine.api.renders())).map((render) => render.id)).toEqual([job.id])
    const notReady = yield* engine.call(engine.api.audio({ params: { id: job.id } })).pipe(Effect.flip)
    expect(notReady.message).toBe("Engine error: This render is not ready yet")
    const missing = yield* engine.call(engine.api.generate({
      payload: { voice_id: "0".repeat(32), text: "Hello.", language: "English", seed: 12 }
    })).pipe(Effect.flip)
    expect(missing.message).toBe("Engine error: Choose a saved voice first")
    const invalid = yield* engine.call(engine.api.generate({
      payload: { voice_id: voice.id, text: "Hello.", language: "English", seed: -1 }
    })).pipe(Effect.flip)
    expect(invalid.message).toContain("seed")

    const busy = yield* engine.call(engine.api.shutdown()).pipe(Effect.flip)
    expect(busy.message).toContain("active render")
    expect((yield* engine.call(engine.api.cancel({ params: { id: job.id } }))).status).toBe("cancelled")

    const { words } = yield* engine.call(engine.api.transcribe({
      query: { sample_rate: 48000 },
      payload: new Uint8Array(48000 * 2)
    }))
    expect(words).toEqual([{ text: " Hello", start: 0.5, end: 0.9 }])

    expect(yield* engine.stop).toBe(true)
    expect(Option.isNone(yield* engine.probe)).toBe(true)
  }))
}, 30_000)
