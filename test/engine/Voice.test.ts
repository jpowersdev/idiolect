import { expect, test } from "bun:test"
import { Effect } from "effect"
import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import type { Word } from "../../src/Reading.ts"
import { readWav, writeWav } from "../../src/engine/Signal.ts"
import * as Voice from "../../src/engine/Voice.ts"
import { temporaryDirectory } from "../support.ts"

const wave = (seconds: number, options: { silent?: boolean; rate?: number } = {}) => {
  const rate = options.rate ?? 48000
  const samples = Float32Array.from({ length: Math.round(seconds * rate) }, (_, i) => options.silent ? 0 : 0.2 * Math.sin((2 * Math.PI * 220 * i) / rate))
  return writeWav({ samples, rate })
}

/** A word every half second; every `sentence`th word, and the last, ends a sentence. */
const timedWords = (seconds: number, sentence = 6): Array<Word> => {
  const count = seconds * 2
  return Array.from({ length: count }, (_, index) => ({
    text: `word${index}${index % sentence === sentence - 1 || index === count - 1 ? "." : ""}`,
    start: index * 0.5,
    end: index * 0.5 + 0.4
  }))
}

const take = (script = "comma-1", seconds = 20, changes: Partial<Voice.Take> = {}): Voice.Take => {
  const words = timedWords(seconds)
  return { script, transcript: words.map((w) => w.text).join(" "), words, wav: wave(seconds), ...changes }
}

const save = (data: string, takes: ReadonlyArray<Voice.Take>) => Effect.runPromise(Effect.flip(Voice.replace(data, takes)).pipe(Effect.option))

test("takes are kept at their recorded rate and cut into 24 kHz clips that end at sentences", async () => {
  const data = temporaryDirectory()
  expect(Voice.read(data)).toBeUndefined()
  const takes = [take(), take("technical", 30, { notice: "Copyright notice." })]
  await Effect.runPromise(Voice.replace(data, takes))
  const saved = Voice.read(data)!
  expect(saved.takes.map((t) => [t.script, t.duration])).toEqual([["comma-1", 20], ["technical", 30]])
  expect(saved.takes[1]!.notice).toBe("Copyright notice.")
  expect(saved.takes[0]!.words).toEqual(takes[0]!.words)
  const folder = Voice.voiceFolder(data)
  const whole = readWav(readFileSync(join(folder, "takes/comma-1.wav")))
  expect(whole.rate).toBe(48000)
  expect(whole.samples.length).toBe(20 * 48000)
  expect(new Set(saved.clips.map((clip) => clip.script))).toEqual(new Set(["comma-1", "technical"]))
  for (const clip of saved.clips) {
    const audio = readWav(readFileSync(join(folder, clip.file)))
    expect(audio.rate).toBe(Voice.CLIP_RATE)
    expect(audio.samples.length / audio.rate).toBeGreaterThanOrEqual(3)
    expect(audio.samples.length / audio.rate).toBeLessThanOrEqual(15.5)
    expect(clip.text.endsWith(".")).toBe(true)
  }
  expect(readdirSync(folder).sort()).toEqual(["clips", "takes", "voice.json"])
})

test("a new recording replaces the voice, leaving nothing else behind", async () => {
  const data = temporaryDirectory()
  await Effect.runPromise(Voice.replace(data, [take("comma-1")]))
  await Effect.runPromise(Voice.replace(data, [take("technical")]))
  expect(Voice.read(data)!.takes.map((t) => t.script)).toEqual(["technical"])
  expect(readdirSync(data)).toEqual(["voice"])
})

test("bad recordings keep the previous voice", async () => {
  const data = temporaryDirectory()
  await Effect.runPromise(Voice.replace(data, [take()]))
  const previous = Voice.read(data)
  const bad: Array<[Uint8Array, string]> = [
    [wave(1), "between 3 and 90 seconds"],
    [wave(91), "between 3 and 90 seconds"],
    [wave(20, { silent: true }), "silent or too quiet"],
    [new TextEncoder().encode("not audio"), "could not be read"]
  ]
  for (const [wav, message] of bad) {
    const error = await save(data, [take(), take("technical", 20, { wav })])
    expect(error._tag === "Some" && error.value.message).toContain(message)
    expect(Voice.read(data)).toEqual(previous)
  }
  expect(readdirSync(data)).toEqual(["voice"])
})
