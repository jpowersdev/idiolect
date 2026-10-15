/**
 * Your voice's folder: voice.json, the whole recorded takes at their recorded rate, and the short
 * 24 kHz clips cut from them for cloning. Replaced only as a whole, once every take is saved.
 */
import { Effect } from "effect"
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import type { Word } from "../Reading.ts"
import { clipRanges } from "./Sections.ts"
import { readWav, resample, writeWav } from "./Signal.ts"

export const CLIP_RATE = 24000
const MAX_TAKE_SECONDS = 90

/** One reading of a setup script: its recording as a WAV, with what was heard. */
export interface Take {
  readonly script: string
  readonly transcript: string
  readonly words: ReadonlyArray<Word>
  /** Some scripts' licenses ask that recordings carry their notice. */
  readonly notice?: string
  readonly wav: Uint8Array
}

export interface Saved {
  readonly created_at: string
  readonly duration: number
  readonly takes: ReadonlyArray<Omit<Take, "wav"> & { readonly duration: number; readonly file: string }>
  readonly clips: ReadonlyArray<{ readonly script: string; readonly text: string; readonly duration: number; readonly file: string }>
}

const invalid = (message: string) => Effect.fail(new Error(message))
const round = (x: number) => Math.round(x * 100) / 100

/** One recorded script as mono, at its recorded rate. */
const readTake = (bytes: Uint8Array) =>
  Effect.gen(function*() {
    const audio = yield* Effect.try({ try: () => readWav(bytes), catch: () => new Error("A recording could not be read as audio.") })
    const seconds = audio.samples.length / audio.rate
    if (!(seconds >= 3 && seconds <= MAX_TAKE_SECONDS)) {
      return yield* invalid(`Each recording must last between 3 and ${MAX_TAKE_SECONDS} seconds.`)
    }
    let sum = 0
    for (const x of audio.samples) sum += x * x
    if (!Number.isFinite(sum) || Math.sqrt(sum / audio.samples.length) < 1e-4) {
      return yield* invalid("A recording is silent or too quiet. Try recording closer to the microphone.")
    }
    return audio
  })

export const voiceFolder = (data: string) => join(data, "voice")

export const read = (data: string): Saved | undefined => {
  const path = join(voiceFolder(data), "voice.json")
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : undefined
}

/** Save each take at its recorded rate, cut its clips, and only then replace the voice. */
export const replace = (data: string, takes: ReadonlyArray<Take>) =>
  Effect.gen(function*() {
    const staging = join(data, "voice.new")
    rmSync(staging, { recursive: true, force: true })
    mkdirSync(join(staging, "takes"), { recursive: true })
    mkdirSync(join(staging, "clips"))
    const saved: Array<Saved["takes"][number]> = []
    const clips: Array<Saved["clips"][number]> = []
    for (const { wav, ...take } of takes) {
      const audio = yield* readTake(wav)
      const file = `takes/${take.script}.wav`
      writeFileSync(join(staging, file), writeWav(audio))
      saved.push({ ...take, duration: round(audio.samples.length / audio.rate), file })
      for (const [n, range] of clipRanges(take.words, audio.rate).entries()) {
        const clip = resample(audio.samples.subarray(range.start, range.end), audio.rate, CLIP_RATE)
        const name = `clips/${take.script}-${String(n + 1).padStart(2, "0")}.wav`
        writeFileSync(join(staging, name), writeWav({ samples: clip, rate: CLIP_RATE }))
        clips.push({ script: take.script, text: range.text, duration: round(clip.length / CLIP_RATE), file: name })
      }
    }
    if (clips.length === 0) return yield* invalid("The recordings are too short to clone from.")
    const voice: Saved = {
      created_at: new Date().toISOString(),
      duration: round(saved.reduce((sum, take) => sum + take.duration, 0)),
      takes: saved,
      clips
    }
    writeFileSync(join(staging, "voice.json"), JSON.stringify(voice, null, 2))
    const folder = voiceFolder(data)
    const previous = join(data, "voice.old")
    rmSync(previous, { recursive: true, force: true })
    if (existsSync(folder)) renameSync(folder, previous)
    renameSync(staging, folder)
    rmSync(previous, { recursive: true, force: true })
    return voice
  }).pipe(Effect.ensuring(Effect.sync(() => rmSync(join(data, "voice.new"), { recursive: true, force: true }))))
