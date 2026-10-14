/** Detect the end of a spoken setup script, and save it. Pure logic; recording lives in Enroll.ts. */
import type { Word } from "./Api.ts"

// Keep the full recording for cloning; the engine makes its own recognition copy.
export const SampleRate = 48000
// Long enough for a slow reading of the longest script.
export const MaxRecordingSeconds = 90

/** Ignore punctuation and case while retaining word order and the actual spoken words. */
export const tokens = (text: string): Array<string> =>
  text.toLowerCase().replace(/[’']/g, "").match(/[a-z0-9]+/g) ?? []

/** Letters two spellings differ by: insertions, deletions, and substitutions. */
const editDistance = (a: string, b: string) => {
  let previous = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    const current = [i]
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(previous[j]! + 1, current[j - 1]! + 1, previous[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1))
    }
    previous = current
  }
  return previous[b.length]!
}

/**
 * The recognizer spells some words its own way: Sara for Sarah, gray for grey.
 * Short words must match exactly, since one letter changes them (walk, work).
 */
export const sameWord = (expected: string, heard: string) => {
  if (expected === heard) return true
  const allowed = expected.length >= 7 ? 2 : expected.length >= 4 ? 1 : 0
  return allowed > 0 && Math.abs(expected.length - heard.length) <= allowed && editDistance(expected, heard) <= allowed
}

/** How many of the script's final words mark the end of a reading. */
export const EndingWords = 5

export interface Ending {
  /** Whether the heard words end with the script's final words. */
  readonly done: boolean
  /** Index into the heard words of the reading's last word, once done. */
  readonly lastWord: number | undefined
}

/**
 * Detect the end of a reading: the first place the heard words contain the script's final words,
 * in order. One may be misheard; a misheard final word counts if something was heard in its place.
 * The first place, because the recognizer sometimes repeats the last words into the silence after.
 */
export const findEnding = (script: string, words: ReadonlyArray<Word>): Ending => {
  const ending = tokens(script).slice(-EndingWords)
  const heard: Array<{ token: string; owner: number }> = []
  words.forEach((word, owner) => {
    for (const token of tokens(word.text)) heard.push({ token, owner })
  })
  if (ending.length === 0) return { done: false, lastWord: undefined }
  // How many of ending[0..last] are heard in order, ending exactly at heard[at], within a few words.
  const matchedBefore = (last: number, at: number) => {
    let matched = 0
    let j = at
    const earliest = Math.max(0, at - ending.length - 2)
    for (let i = last; i >= 0 && j >= earliest; i--) {
      let k = j
      while (k >= earliest && !sameWord(ending[i]!, heard[k]!.token)) k--
      if (i === last && k !== at) return 0
      if (k < earliest) continue
      matched++
      j = k - 1
    }
    return matched
  }
  const final = ending.length - 1
  for (let at = 0; at < heard.length; at++) {
    if (matchedBefore(final, at) >= ending.length - 1) return { done: true, lastWord: heard[at]!.owner }
  }
  // The final word was misheard: accept what was heard right after the rest of the ending.
  for (let at = 1; at < heard.length; at++) {
    if (final > 0 && matchedBefore(final - 1, at - 1) === final) return { done: true, lastWord: heard[at]!.owner }
  }
  return { done: false, lastWord: undefined }
}

export interface SavedReading {
  readonly wav: Uint8Array
  readonly transcript: string
  readonly duration: number
  /** Heard words, timed from the start of the saved clip. */
  readonly words: ReadonlyArray<Word>
}

/** Trim from the first heard word to the reading's last, keeping everything heard between. */
export const saveReading = (pcm: Int16Array, words: ReadonlyArray<Word>, lastWord: number): SavedReading => {
  const first = words[0]
  const last = words[lastWord]
  if (first === undefined || last === undefined) throw new Error("Nothing was recognized. Please try the reading again.")
  const start = Math.max(0, Math.trunc((first.start - 0.25) * SampleRate))
  const end = Math.min(pcm.length, Math.trunc((last.end + 0.35) * SampleRate))
  const clip = pcm.subarray(start, end)
  const duration = clip.length / SampleRate
  if (duration < 3 || duration > MaxRecordingSeconds) {
    throw new Error(`Read at a natural pace so the complete script fits in 3–${MaxRecordingSeconds} seconds.`)
  }
  const offset = start / SampleRate
  const heard = words.slice(0, lastWord + 1).map((word) => ({
    text: word.text.trim(),
    start: round(Math.max(0, word.start - offset)),
    end: round(Math.min(duration, word.end - offset))
  }))
  return {
    wav: encodeWav(clip, SampleRate),
    transcript: heard.map((word) => word.text).join(" "),
    duration: round(duration),
    words: heard
  }
}

const round = (seconds: number) => Math.round(seconds * 100) / 100

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
