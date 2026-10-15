/**
 * The models on this machine's GPU: each loaded the first time it is needed, then kept for the
 * rest of the command.
 */
import type { Runtime } from "@effect-torch/core"
import { Effect } from "effect"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { downloaded, modelDirectory, ReadingModel, SpeechModel } from "../Models.ts"
import type { Word } from "../Reading.ts"
import type { Voices } from "./Render.ts"
import { endsSentence } from "./Sections.ts"
import { readWav, resample } from "./Signal.ts"
import * as Speech from "./Speech.ts"
import * as Voice from "./Voice.ts"
import * as Whisper from "./Whisper.ts"

const missing = (what: string) => new Error(`The ${what} model is missing. Run idiolect setup first.`)
const asError = (error: unknown) => error instanceof Error ? error : new Error(String(error))

/** Load once; a failed load is tried again next time. */
const lazily = <A, E, R>(load: Effect.Effect<A, E, R>) => {
  let value: A | undefined
  return Effect.suspend(() => value !== undefined ? Effect.succeed(value) : Effect.tap(load, (loaded) => Effect.sync(() => (value = loaded))))
}

export const make = (data: string) =>
  Effect.gen(function*() {
    // The tensor runtime, for work started later.
    const runtime = yield* Effect.context<Runtime.Runtime>()
    const onGpu = <A, E>(work: Effect.Effect<A, E, Runtime.Runtime>) => Effect.provideContext(work, runtime)

    const recognizer = lazily<Whisper.Recognizer, Error, Runtime.Runtime>(Effect.suspend(() =>
      downloaded(data, ReadingModel) ? Effect.mapError(Whisper.load(modelDirectory(data, ReadingModel)), asError) : Effect.fail(missing("reading"))
    ))
    const speech = lazily<Speech.Speech, Error, Runtime.Runtime>(Effect.suspend(() =>
      downloaded(data, SpeechModel) ? Effect.mapError(Speech.load(modelDirectory(data, SpeechModel), "bf16"), asError) : Effect.fail(missing("speech"))
    ))

    // Setup replaces the voice, then previews it; its creation time identifies the recordings.
    let prompt: { readonly createdAt: string; readonly voice: Speech.Voice } | undefined
    const voicePrompt = (loaded: Speech.Speech) =>
      Effect.gen(function*() {
        const saved = Voice.read(data)
        if (saved === undefined) return yield* Effect.fail(new Error("Record your voice first with idiolect setup."))
        if (prompt?.createdAt !== saved.created_at) {
          // A clip ending mid-sentence would make new text sound like the rest of it.
          const clip = saved.clips.find((c) => endsSentence(c.text)) ?? saved.clips[0]!
          const audio = readWav(readFileSync(join(Voice.voiceFolder(data), clip.file)))
          prompt = { createdAt: saved.created_at, voice: yield* Effect.mapError(Speech.voice(loaded, audio.samples, audio.rate, clip.text), asError) }
        }
        return prompt.voice
      })

    return {
      rate: Speech.OUTPUT_RATE,
      load: onGpu(Effect.asVoid(Effect.all([speech, recognizer]))),
      speak: (text, seed) =>
        onGpu(Effect.gen(function*() {
          const loaded = yield* speech
          return yield* Effect.mapError(Speech.speak(loaded, yield* voicePrompt(loaded), text, { seed }), asError)
        })),
      transcribe: (audio: Float32Array, rate: number): Effect.Effect<ReadonlyArray<Word>, Error> =>
        onGpu(Effect.gen(function*() {
          const loaded = yield* recognizer
          return yield* Effect.mapError(Whisper.transcribe(loaded, resample(audio, rate, Whisper.SAMPLE_RATE)), asError)
        }))
    } satisfies Voices
  })
