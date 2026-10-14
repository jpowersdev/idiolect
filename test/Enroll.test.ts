import { expect, test } from "bun:test"
import { Duration, Effect, Layer, Stream } from "effect"
import { recordReading } from "../src/Enroll.ts"
import { SampleRate } from "../src/Reading.ts"
import { fakeAudio, fakeEngine, words } from "./support.ts"

const script = "A quiet morning beside the blue river."
const tenthOfASecond = SampleRate * 2 / 10

/** A microphone delivering `seconds` of audio in real time, then optionally stopping. */
const microphone = (seconds: number, value: number, stops = false) =>
  fakeAudio({
    record: () =>
      Effect.succeed({
        pcm: Stream.range(1, seconds * 10).pipe(
          Stream.mapEffect(() =>
            Effect.as(Effect.sleep(Duration.millis(stops ? 1 : 100)), new Uint8Array(tenthOfASecond).fill(value))
          ),
          Stream.concat(stops ? Stream.empty : Stream.never)
        ),
        diagnostics: Effect.succeed(stops ? "Device disconnected" : "")
      })
  })

test("stops once the ending and a pause are heard, then saves the whole take", async () => {
  const payloads: Array<number> = []
  const engine = fakeEngine({
    transcribe: ({ payload }: { payload: Uint8Array }) =>
      Effect.sync(() => {
        payloads.push(payload.length)
        return { words: words(script, 0.5) }
      })
  })
  const reading = await Effect.runPromise(
    recordReading({ script, device: undefined, progress: () => Effect.void, heardEnd: Effect.void }).pipe(
      Effect.provide(Layer.mergeAll(engine, microphone(30, 1)))
    )
  )
  // Listening passes until the ending and a pause after it, then one over the whole take.
  expect(payloads.length).toBeGreaterThanOrEqual(2)
  expect(payloads.at(-1)).toBeGreaterThanOrEqual(payloads.at(-2)!)
  expect(reading.transcript).toBe(script)
  expect(reading.duration).toBe(4.1)
}, 20_000)

test("listening only sends the most recent seconds", async () => {
  const payloads: Array<number> = []
  const engine = fakeEngine({
    transcribe: ({ payload }: { payload: Uint8Array }) =>
      Effect.sync(() => {
        payloads.push(payload.length)
        // The ending arrives after more than 8 seconds of recording.
        return { words: payloads.length > 16 ? words(script, 1) : words("A quiet morning", 1) }
      })
  })
  await Effect.runPromise(
    recordReading({ script, device: undefined, progress: () => Effect.void, heardEnd: Effect.void }).pipe(
      Effect.provide(Layer.mergeAll(engine, microphone(30, 1)))
    )
  )
  expect(Math.max(...payloads.slice(0, -1))).toBeLessThanOrEqual(8 * SampleRate * 2)
  expect(payloads.at(-1)).toBeGreaterThan(8 * SampleRate * 2)
}, 30_000)

test("a silent microphone is reported quickly, with how to fix it", async () => {
  const engine = fakeEngine({ transcribe: () => Effect.succeed({ words: [] }) })
  const started = Date.now()
  const error = await Effect.runPromise(
    recordReading({ script, device: undefined, progress: () => Effect.void, heardEnd: Effect.void }).pipe(
      Effect.flip,
      Effect.provide(Layer.mergeAll(engine, microphone(30, 0)))
    )
  )
  expect(error.message).toContain("only returned silence")
  expect(Date.now() - started).toBeLessThan(8000)
}, 20_000)

test("a recorder that stops is reported with its diagnostics", async () => {
  const engine = fakeEngine({ transcribe: () => Effect.succeed({ words: [] }) })
  const error = await Effect.runPromise(
    recordReading({ script, device: undefined, progress: () => Effect.void, heardEnd: Effect.void }).pipe(
      Effect.flip,
      Effect.provide(Layer.mergeAll(engine, microphone(1, 1, true)))
    )
  )
  expect(error.message).toBe("Microphone recording stopped. Device disconnected")
}, 20_000)
