/**
 * Speaking a script: split into sections, each rendered, checked by ear with the recognizer,
 * and rendered again if garbled or faded, then joined with its pauses.
 */
import { Effect } from "effect"
import type { Word } from "../Reading.ts"
import { coverage, speechEnd, splitScript } from "./Sections.ts"
import { cutAfter, endingFade, joinSections, writeWav } from "./Signal.ts"

/** Speaking and listening, as the models (or stand-ins for them) do it. */
export interface Voices {
  /** Say a section in your voice: mono audio at `rate`. */
  readonly speak: (text: string, seed: number) => Effect.Effect<Float32Array, Error>
  readonly rate: number
  readonly transcribe: (audio: Float32Array, rate: number) => Effect.Effect<ReadonlyArray<Word>, Error>
  /** Load the models ahead of the first section, so loading shows as its own step. */
  readonly load: Effect.Effect<void, Error>
}

// A section is rendered again, with another seed, when it sounds garbled or fades out.
export const ATTEMPTS = 3
export const REQUIRED_COVERAGE = 0.9
// Final words this many decibels quieter than the section's typical word: it faded out.
export const FADE_DB = 12
const SEED = 12

/**
 * Render one section and check it with the recognizer: keep the first attempt that says every
 * word through the end without fading out, cut where its final words end, or else the best one.
 */
export const renderSection = (
  voices: Voices,
  text: string,
  index: number,
  onAttempt: (attempt: number) => Effect.Effect<void> = () => Effect.void
) =>
  Effect.gen(function*() {
    let best: { score: number; samples: Float32Array } | undefined
    for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
      yield* onAttempt(attempt)
      let samples = yield* voices.speak(text, SEED + index + 1000 * attempt)
      const words = yield* voices.transcribe(samples, voices.rate)
      const end = speechEnd(text, words)
      const faded = endingFade(samples, voices.rate, words) > FADE_DB
      const score = coverage(text, words) + (end !== undefined ? 1 : 0) - (faded ? 1 : 0)
      // Cut the syllables the model sometimes adds after the text.
      if (end !== undefined) samples = cutAfter(samples, voices.rate, end)
      if (best === undefined || score > best.score) best = { score, samples }
      if (end !== undefined && !faded && score - 1 >= REQUIRED_COVERAGE) break
    }
    return best!.samples
  })

export interface Spoken {
  /** 16-bit WAV. */
  readonly wav: Uint8Array
  readonly duration: number
}

/** Speak a whole script, reporting each step as it starts. */
export const render = (voices: Voices, text: string, progress: (message: string) => Effect.Effect<void> = () => Effect.void) =>
  Effect.gen(function*() {
    const sections = splitScript(text)
    yield* progress("Loading the voice model…")
    yield* voices.load
    const rendered: Array<{ samples: Float32Array; pause: number }> = []
    for (const [index, section] of sections.entries()) {
      const samples = yield* renderSection(voices, section.text, index, (attempt) =>
        progress(`Rendering section ${index + 1} of ${sections.length}${attempt ? " again" : ""}…`))
      rendered.push({ samples, pause: section.pause })
    }
    const joined = yield* Effect.try({ try: () => joinSections(rendered, voices.rate), catch: (error) => error as Error })
    return {
      wav: writeWav({ samples: joined, rate: voices.rate }),
      duration: Math.round((joined.length / voices.rate) * 100) / 100
    } satisfies Spoken
  })
