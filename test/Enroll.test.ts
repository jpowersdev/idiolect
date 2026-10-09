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

test("stops after two complete passes and saves the trimmed reading", async () => {
  let passes = 0
  const engine = fakeEngine({
    transcribe: () => Effect.sync(() => ({ words: ++passes >= 1 ? words(script, 0.5) : [] }))
  })
  const reading = await Effect.runPromise(
    recordReading({ script, device: undefined, progress: () => Effect.void }).pipe(
      Effect.provide(Layer.mergeAll(engine, microphone(30, 1)))
    )
  )
  expect(passes).toBe(2)
  expect(reading.transcript).toBe(script)
  expect(reading.duration).toBe(4.1)
}, 20_000)

test("a silent microphone is reported quickly, with how to fix it", async () => {
  const engine = fakeEngine({ transcribe: () => Effect.succeed({ words: [] }) })
  const started = Date.now()
  const error = await Effect.runPromise(
    recordReading({ script, device: undefined, progress: () => Effect.void }).pipe(
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
    recordReading({ script, device: undefined, progress: () => Effect.void }).pipe(
      Effect.flip,
      Effect.provide(Layer.mergeAll(engine, microphone(1, 1, true)))
    )
  )
  expect(error.message).toBe("Microphone recording stopped. Device disconnected")
}, 20_000)
