/** Follow a spoken enrollment script. Pure logic; recording lives in Enroll.ts. */
import type { Word } from "./Api.ts"

// Keep the full recording for cloning; the engine makes its own recognition copy.
export const SampleRate = 48000
export const MaxRecordingSeconds = 45

export const ReadingScript = "On Thursday, three friends chose a quiet path through the village. " +
  "A small boy in a red jacket watched a brown dog cross the green fields. " +
  "Have you packed a good book, some fresh bread, and a cup of tea? " +
  "We usually enjoy these walks, even when the weather turns cold and the wind gets strong."

/** Ignore punctuation and case while retaining word order and the actual spoken words. */
export const tokens = (text: string): Array<string> =>
  text.toLowerCase().replace(/[’']/g, "").match(/[a-z0-9]+/g) ?? []

export interface ReadingMatch {
  readonly matched: number
  readonly total: number
  readonly nextWords: string
  /** Index into the heard words of the script's first and last words, when heard. */
  readonly firstWord: number | undefined
  readonly lastWord: number | undefined
  readonly complete: boolean
}

/** Align the script with the heard words in order (longest common subsequence). */
export const matchReading = (script: string, words: ReadonlyArray<Word>): ReadingMatch => {
  const expected = tokens(script)
  const heard: Array<string> = []
  const owners: Array<number> = []
  words.forEach((word, index) => {
    for (const part of tokens(word.text)) {
      heard.push(part)
      owners.push(index)
    }
  })
  const rows = expected.length + 1
  const columns = heard.length + 1
  const lengths = new Uint16Array(rows * columns)
  for (let i = expected.length - 1; i >= 0; i--) {
    for (let j = heard.length - 1; j >= 0; j--) {
      lengths[i * columns + j] = expected[i] === heard[j]
        ? lengths[(i + 1) * columns + j + 1]! + 1
        : Math.max(lengths[(i + 1) * columns + j]!, lengths[i * columns + j + 1]!)
    }
  }
  const positions = new Map<number, number>()
  for (let i = 0, j = 0; i < expected.length && j < heard.length;) {
    if (expected[i] === heard[j] && lengths[i * columns + j] === lengths[(i + 1) * columns + j + 1]! + 1) {
      positions.set(i++, owners[j++]!)
    } else if (lengths[(i + 1) * columns + j]! >= lengths[i * columns + j + 1]!) {
      i++
    } else {
      j++
    }
  }
  let next = 0
  while (next < expected.length && positions.has(next)) next++
  return {
    matched: positions.size,
    total: expected.length,
    nextWords: expected.slice(next, next + 6).join(" "),
    firstWord: positions.get(0),
    lastWord: positions.get(expected.length - 1),
    complete: expected.length > 0 && positions.size === expected.length
  }
}

export interface SavedReading {
  readonly wav: Uint8Array
  readonly transcript: string
  readonly duration: number
}

/** Trim around the reading and keep fillers actually heard between its first and last words. */
export const saveReading = (
  pcm: Int16Array,
  words: ReadonlyArray<Word>,
  match: ReadingMatch
): SavedReading => {
  if (!match.complete || match.firstWord === undefined || match.lastWord === undefined) {
    throw new Error("The full script was not recognized. Please try the reading again.")
  }
  const first = words[match.firstWord]!
  const last = words[match.lastWord]!
  const start = Math.max(0, Math.trunc((first.start - 0.25) * SampleRate))
  const end = Math.min(pcm.length, Math.trunc((last.end + 0.35) * SampleRate))
  const clip = pcm.subarray(start, end)
  const duration = clip.length / SampleRate
  if (duration < 3 || duration > 30) {
    throw new Error("Read at a natural pace so the complete script fits in 3–30 seconds.")
  }
  const transcript = words.slice(match.firstWord, match.lastWord + 1).map((word) => word.text.trim()).join(" ")
  return { wav: encodeWav(clip, SampleRate), transcript, duration: Math.round(duration * 100) / 100 }
}

/** Little-endian 16-bit PCM bytes as samples. Ignores a trailing odd byte. */
export const pcmSamples = (bytes: Uint8Array): Int16Array => {
  const samples = new Int16Array(bytes.length >> 1)
  const view = new DataView(bytes.buffer, bytes.byteOffset, samples.length * 2)
  for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true)
  return samples
}

export const encodeWav = (samples: Int16Array, sampleRate: number): Uint8Array => {
  const bytes = new Uint8Array(44 + samples.length * 2)
  const view = new DataView(bytes.buffer)
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i))
  }
  text(0, "RIFF")
  view.setUint32(4, 36 + samples.length * 2, true)
  text(8, "WAVE")
  text(12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  text(36, "data")
  view.setUint32(40, samples.length * 2, true)
  samples.forEach((sample, i) => view.setInt16(44 + i * 2, sample, true))
  return bytes
}
