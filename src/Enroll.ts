/** Record a reading until the engine's recognizer hears the end of the script. */
import { Duration, Effect, Fiber, Stream } from "effect"
import { Audio } from "./Audio.ts"
import { Engine } from "./Engine.ts"
import { fail } from "./Errors.ts"
import { findEnding, MaxRecordingSeconds, pcmSamples, SampleRate, type SavedReading, saveReading } from "./Reading.ts"

/** Recent speech to check for the end of the script: comfortably more than its final words. */
const ListeningSeconds = 8
/** Silence after the final words that ends the recording. */
const PauseSeconds = 0.6

const silentMicrophone = () =>
  fail(
    "The microphone only returned silence." + (process.platform === "darwin"
      ? " On macOS, allow your terminal app in System Settings > Privacy & Security > Microphone, then restart it."
      : " Check that the right input device is selected and unmuted.")
  )

/** The last `size` bytes (whole samples) of the first `length` recorded bytes. */
const tail = (chunks: ReadonlyArray<Uint8Array>, length: number, size: number) => {
  const start = length - (Math.min(length, size) & ~1)
  const bytes = new Uint8Array(length - start)
  let chunkEnd = chunks.reduce((total, chunk) => total + chunk.length, 0)
  for (let index = chunks.length - 1; index >= 0 && chunkEnd > start; index--) {
    const chunk = chunks[index]!
    const chunkStart = chunkEnd - chunk.length
    const from = Math.max(chunkStart, start)
    const to = Math.min(chunkEnd, length)
    if (to > from) bytes.set(chunk.subarray(from - chunkStart, to - chunkStart), from - start)
    chunkEnd = chunkStart
  }
  return bytes
}

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
  readonly progress: (heard: { readonly seconds: number; readonly heard: string }) => Effect.Effect<void>
  /** Runs as soon as the end is heard, before the whole take is transcribed. */
  readonly heardEnd: Effect.Effect<void>
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

  const transcribe = (pcm: Uint8Array) =>
    engine.call(engine.api.transcribe({ query: { sample_rate: SampleRate }, payload: pcm }))
      .pipe(Effect.map(({ words }) => words))

  // Listen to the last few seconds for the script's final words, then a pause.
  const listen = Effect.gen(function*() {
    const started = Date.now()
    while (Date.now() - started < MaxRecordingSeconds * 1000) {
      // A blocked microphone delivers exact zeros rather than an error.
      if (!heardSignal && length >= SampleRate * 2 * 5) return yield* silentMicrophone()
      if (length < SampleRate * 2 * 3) {
        yield* Effect.sleep(Duration.millis(100))
        continue
      }
      const submitted = Date.now()
      const window = tail(chunks, length & ~1, ListeningSeconds * SampleRate * 2)
      const words = yield* transcribe(window)
      yield* options.progress({ seconds: (Date.now() - started) / 1000, heard: words.map((word) => word.text.trim()).join(" ") })
      const ending = findEnding(options.script, words)
      if (ending.done && window.length / 2 / SampleRate - words[ending.lastWord!]!.end >= PauseSeconds) return
      yield* Effect.sleep(Duration.millis(Math.max(0, 500 - (Date.now() - submitted))))
    }
    return yield* fail(
      "I did not hear the end of the script before the time limit. Check your microphone and read it again."
    )
  })

  yield* Effect.raceFirst(listen, recorderStopped)
  yield* options.heardEnd
  // Then transcribe the whole take once, for its exact words and timings.
  const pcm = concat(chunks, length & ~1)
  const words = yield* transcribe(pcm)
  const ending = findEnding(options.script, words)
  return yield* Effect.try({
    try: (): SavedReading => saveReading(pcmSamples(pcm), words, ending.lastWord ?? words.length - 1),
    catch: (error) => fail(error instanceof Error ? error.message : String(error))
  })
}, Effect.scoped)
