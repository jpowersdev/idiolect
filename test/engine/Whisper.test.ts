/**
 * The effect-torch Whisper recognizer against Hugging Face Transformers' transcript and the
 * previous recognizer's word timings, on one of your recorded takes. Needs the model and the
 * reference, so it runs only with IDIOLECT_MODEL_TESTS=1.
 */
import { expect, test } from "bun:test"
import * as BackendApple from "@effect-torch/backend-apple-native"
import { Effect } from "effect"
import { existsSync, readFileSync } from "node:fs"
import { load, transcribe, windows } from "../../src/engine/Whisper.ts"

const home = process.env.HOME!
const model = `${home}/.idiolect/experiments/whisper-base.en`
const reference = `${home}/.idiolect/experiments/whisper-reference`
const enabled = process.env.IDIOLECT_MODEL_TESTS === "1" && existsSync(`${reference}/reference.json`)

test("audio over 30 seconds splits at its quietest moment between 20 and 30 seconds in", () => {
  const rate = 16000
  const audio = new Float32Array(70 * rate).fill(0.1)
  audio.fill(0, 24 * rate, 24.4 * rate)
  const [first, second, third] = windows(audio)
  expect(first!.start).toBe(0)
  expect(Math.abs(first!.end / rate - 24.2)).toBeLessThan(0.15)
  expect(second!.start).toBe(first!.end)
  expect(third!.end).toBe(audio.length)
  expect(windows(new Float32Array(10 * rate))).toEqual([{ start: 0, end: 10 * rate }])
})

test.skipIf(!enabled)("transcription matches the reference, with word timings close to the previous recognizer's", async () => {
  const bytes = readFileSync(`${reference}/audio-16k.f32`)
  const audio = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
  const expected = JSON.parse(readFileSync(`${reference}/faster-whisper-words.json`, "utf8")) as Array<{ text: string; start: number; end: number }>
  const words = await Effect.runPromise(Effect.gen(function*() {
    const recognizer = yield* load(model)
    const started = performance.now()
    const result = yield* transcribe(recognizer, audio)
    console.log(`transcribed ${(audio.length / 16000).toFixed(1)} s in ${Math.round(performance.now() - started)} ms`)
    const silent = yield* transcribe(recognizer, new Float32Array(16000 * 5))
    expect(silent).toEqual([])
    return result
  }).pipe(Effect.provide(BackendApple.layer())))
  expect(words.map((w) => w.text)).toEqual(expected.map((w) => w.text.trim()))
  const differences = words.map((w, i) => Math.max(Math.abs(w.start - expected[i]!.start), Math.abs(w.end - expected[i]!.end))).sort((a, b) => a - b)
  expect(differences[Math.floor(differences.length / 2)]!).toBeLessThan(0.05)
}, 300_000)
