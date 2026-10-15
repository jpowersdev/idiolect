/**
 * The models on this machine's GPU: each loaded the first time it is needed, then kept until the
 * engine exits. One piece of GPU work runs at a time; a transcription waits for at most one
 * section of a render.
 */
import type { Runtime } from "@effect-torch/core"
import { Effect, Semaphore } from "effect"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { Conflict, type Word } from "../Api.ts"
import { downloaded, modelDirectory, ReadingModel, SpeechModel } from "../Models.ts"
import type { Voices } from "./Renders.ts"
import { endsSentence } from "./Sections.ts"
import { readWav, resample } from "./Signal.ts"
import * as Speech from "./Speech.ts"
import * as Voice from "./Voice.ts"
import * as Whisper from "./Whisper.ts"

export type ModelStatus = "idle" | "loading" | "ready" | "error"

export interface Local extends Voices {
  readonly status: () => {
    readonly model_downloaded: boolean
    readonly recognizer_downloaded: boolean
    readonly model_status: ModelStatus
  }
}

const missing = (what: string) => new Conflict({ detail: `The ${what} model is missing. Run idiolect setup first.` })
const asError = (error: unknown) => error instanceof Error ? error : new Error(String(error))

/** Load once; a failed load is tried again next time. */
const lazily = <A, E, R>(load: Effect.Effect<A, E, R>) => {
  let value: A | undefined
  return Effect.suspend(() => value !== undefined ? Effect.succeed(value) : Effect.tap(load, (loaded) => Effect.sync(() => (value = loaded))))
}

export const make = (data: string) =>
  Effect.gen(function*() {
    const gpu = yield* Semaphore.make(1)
    // The tensor runtime this engine was started with, for work started from requests.
    const runtime = yield* Effect.context<Runtime.Runtime>()
    const onGpu = <A, E>(work: Effect.Effect<A, E, Runtime.Runtime>) => gpu.withPermit(Effect.provideContext(work, runtime))
    let modelStatus: ModelStatus = "idle"

    const recognizer = lazily<Whisper.Recognizer, Error, Runtime.Runtime>(Effect.suspend(() =>
      downloaded(data, ReadingModel) ? Effect.mapError(Whisper.load(modelDirectory(data, ReadingModel)), asError) : Effect.fail(missing("reading"))
    ))

    const speech = lazily<Speech.Speech, Error, Runtime.Runtime>(Effect.suspend(() => {
      if (!downloaded(data, SpeechModel)) return Effect.fail(missing("speech"))
      modelStatus = "loading"
      return Speech.load(modelDirectory(data, SpeechModel), "bf16").pipe(
        Effect.mapError(asError),
        Effect.tap(() => Effect.sync(() => (modelStatus = "ready"))),
        Effect.tapError(() => Effect.sync(() => (modelStatus = "error")))
      )
    }))

    // Setup replaces the voice; its creation time identifies the recordings.
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

    const transcribe = (audio: Float32Array, rate: number): Effect.Effect<ReadonlyArray<Word>, Error> =>
      onGpu(Effect.gen(function*() {
        const loaded = yield* recognizer
        return yield* Effect.mapError(Whisper.transcribe(loaded, resample(audio, rate, Whisper.SAMPLE_RATE)), asError)
      }))

    return {
      rate: Speech.OUTPUT_RATE,
      load: onGpu(Effect.asVoid(Effect.all([speech, recognizer]))),
      speak: (text, seed, cancelled) =>
        onGpu(Effect.gen(function*() {
          const loaded = yield* speech
          return yield* Effect.mapError(Speech.speak(loaded, yield* voicePrompt(loaded), text, { seed, cancelled }), asError)
        })),
      transcribe,
      status: () => ({
        model_downloaded: downloaded(data, SpeechModel),
        recognizer_downloaded: downloaded(data, ReadingModel),
        model_status: modelStatus
      })
    } satisfies Local
  })
