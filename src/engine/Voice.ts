/**
 * Your voice's folder: voice.json, the whole recorded takes at their recorded rate, and the short
 * 24 kHz clips cut from them for cloning. Replaced only as a whole, once every take is saved.
 */
import { Effect, Schema } from "effect"
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { Invalid, type Voice as VoiceSummary } from "../Api.ts"
import { clipRanges } from "./Sections.ts"
import { readWav, resample, writeWav } from "./Signal.ts"

export const CLIP_RATE = 24000
const MAX_TAKE_SECONDS = 90

const Word = Schema.Struct({
  text: Schema.Trim.check(Schema.isMinLength(1), Schema.isMaxLength(100)),
  start: Schema.Finite.check(Schema.isGreaterThanOrEqualTo(0)),
  end: Schema.Finite.check(Schema.isGreaterThanOrEqualTo(0))
})

const Take = Schema.Struct({
  script: Schema.String.check(Schema.isPattern(/^[a-z0-9-]{1,40}$/)),
  transcript: Schema.Trim.check(Schema.isMinLength(1), Schema.isMaxLength(5000)),
  words: Schema.Array(Word).check(Schema.isMinLength(1)),
  // Some scripts' licenses ask that recordings carry their notice.
  notice: Schema.optional(Schema.String.check(Schema.isMaxLength(1000)))
})
type Take = typeof Take.Type

const Takes = Schema.fromJsonString(Schema.Array(Take).check(Schema.isMinLength(1), Schema.isMaxLength(24)))

export interface Saved {
  readonly created_at: string
  readonly duration: number
  readonly takes: ReadonlyArray<Omit<Take, "words"> & { readonly duration: number; readonly words: ReadonlyArray<typeof Word.Type>; readonly file: string }>
  readonly clips: ReadonlyArray<{ readonly script: string; readonly text: string; readonly duration: number; readonly file: string }>
}

const invalid = (detail: string) => new Invalid({ detail })
const round = (x: number) => Math.round(x * 100) / 100

export const parseTakes = (json: string) =>
  Effect.gen(function*() {
    const takes = yield* Schema.decodeUnknownEffect(Takes)(json).pipe(
      Effect.mapError((error) => invalid(`Invalid takes: ${error.message.split("\n")[0]}`))
    )
    if (new Set(takes.map((take) => take.script)).size !== takes.length) return yield* invalid("Invalid takes: Record each script once")
    return takes
  })

/** One recorded script as mono, at its recorded rate. */
const readTake = (bytes: Uint8Array) =>
  Effect.gen(function*() {
    const audio = yield* Effect.try({ try: () => readWav(bytes), catch: () => invalid("A recording could not be read as audio.") })
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

export const summary = (voice: Saved): VoiceSummary => ({
  created_at: voice.created_at,
  duration: voice.duration,
  takes: voice.takes.map((take) => ({ script: take.script, duration: take.duration }))
})

export const voiceFolder = (data: string) => join(data, "voice")

export const read = (data: string): Saved | undefined => {
  const path = join(voiceFolder(data), "voice.json")
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : undefined
}

/** Save each take at its recorded rate, cut its clips, and only then replace the voice. */
export const replace = (data: string, takes: ReadonlyArray<Take>, recordings: ReadonlyArray<Uint8Array>) =>
  Effect.gen(function*() {
    if (recordings.length !== takes.length) return yield* invalid("Send one recording for each take.")
    const staging = join(data, "voice.new")
    rmSync(staging, { recursive: true, force: true })
    mkdirSync(join(staging, "takes"), { recursive: true })
    mkdirSync(join(staging, "clips"))
    const saved: Array<Saved["takes"][number]> = []
    const clips: Array<Saved["clips"][number]> = []
    for (const [index, take] of takes.entries()) {
      const audio = yield* readTake(recordings[index]!)
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
