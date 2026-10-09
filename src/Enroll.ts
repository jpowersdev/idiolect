/** Record a reading, following along through the engine's recognizer, until it is complete. */
import { Duration, Effect, Fiber, Stream } from "effect"
import type { Word } from "./Api.ts"
import { Audio } from "./Audio.ts"
import { Engine } from "./Engine.ts"
import { fail } from "./Errors.ts"
import {
  matchReading,
  MaxRecordingSeconds,
  pcmSamples,
  type ReadingMatch,
  SampleRate,
  type SavedReading,
  saveReading
} from "./Reading.ts"

const silentMicrophone = () =>
  fail(
    "The microphone only returned silence." + (process.platform === "darwin"
      ? " On macOS, allow your terminal app in System Settings > Privacy & Security > Microphone, then restart it."
      : " Check that the right input device is selected and unmuted.")
  )

const concat = (chunks: ReadonlyArray<Uint8Array>, length: number) => {
  const bytes = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.length
  }
  return bytes
}

export const recordReading = Effect.fn("recordReading")(function*(options: {
  readonly script: string
  readonly device: string | undefined
  readonly progress: (match: ReadingMatch) => Effect.Effect<void>
}) {
  const audio = yield* Audio
  const engine = yield* Engine
  const recording = yield* audio.record({ sampleRate: SampleRate, device: options.device })
  const chunks: Array<Uint8Array> = []
  let length = 0
  let heardSignal = false
  const recorder = yield* recording.pcm.pipe(
    Stream.runForEach((chunk) =>
      Effect.sync(() => {
        chunks.push(chunk)
        length += chunk.length
        heardSignal ||= chunk.some((byte) => byte !== 0)
      })
    ),
    Effect.forkScoped
  )
  const recorderStopped = Effect.gen(function*() {
    const exit = yield* Fiber.await(recorder)
    const detail = yield* recording.diagnostics
    return yield* fail(
      `Microphone recording stopped. ${detail || (exit._tag === "Failure" ? String(exit.cause) : "")}`.trim()
    )
  })

  const follow = Effect.gen(function*() {
    const started = Date.now()
    let completePasses = 0
    let last = matchReading(options.script, [])
    while (Date.now() - started < MaxRecordingSeconds * 1000) {
      // A blocked microphone delivers exact zeros rather than an error.
      if (!heardSignal && length >= SampleRate * 2 * 5) return yield* silentMicrophone()
      if (length < SampleRate * 2 * 3) {
        yield* Effect.sleep(Duration.millis(100))
        continue
      }
      const submitted = Date.now()
      const snapshot = concat(chunks, length & ~1)
      const snapshotSeconds = snapshot.length / 2 / SampleRate
      const { words } = yield* engine.call(
        engine.api.transcribe({ query: { sample_rate: SampleRate }, payload: snapshot })
      )
      last = matchReading(options.script, words)
      yield* options.progress(last)
      completePasses = last.complete ? completePasses + 1 : 0
      // Two agreeing passes, and a short pause after the final word, end the reading.
      if (completePasses >= 2 && snapshotSeconds - words[last.lastWord!]!.end >= 0.45) {
        return { pcm: pcmSamples(concat(chunks, length & ~1)), words, match: last }
      }
      yield* Effect.sleep(Duration.millis(Math.max(0, 1500 - (Date.now() - submitted))))
    }
    return yield* fail(
      `Recognized ${last.matched}/${last.total} words before the time limit. ` +
        "Check your microphone and read the whole script again."
    )
  })

  const result: { pcm: Int16Array; words: ReadonlyArray<Word>; match: ReadingMatch } = yield* Effect.raceFirst(
    follow,
    recorderStopped
  )
  return yield* Effect.try({
    try: (): SavedReading => saveReading(result.pcm, result.words, result.match),
    catch: (error) => fail(error instanceof Error ? error.message : String(error))
  })
}, Effect.scoped)
