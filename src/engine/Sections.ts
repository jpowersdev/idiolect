/**
 * Text and timing logic for rendering and for voices: planning the sections a script renders
 * in, checking a rendered section against the recognizer, and planning the clips a take is cut
 * into. Words are compared with the same rules setup uses to hear the end of a reading.
 */
import { sameWord, tokens, type Word } from "../Reading.ts"

// Longer sections tend to fade out toward the end.
export const SECTION_CHARACTERS = 200
// Silence after a section: within a paragraph, and after one.
export const SENTENCE_PAUSE = 0.3
export const PARAGRAPH_PAUSE = 0.75
// Clip lengths in seconds: long enough to carry a speaking style, short enough to stay a quick
// lead-in; stretched, rarely, to reach the end of a sentence.
export const CLIP_SECONDS = [8, 15, 20] as const

export interface Section {
  readonly text: string
  /** Seconds of silence after it. */
  readonly pause: number
}

/** Python's str.rfind(mark, start, end): the last place mark lies wholly within [start, end). */
const findLast = (text: string, mark: string, start: number, end: number) => {
  const at = text.lastIndexOf(mark, end - mark.length)
  return at >= start ? at : -1
}

/** Keep sentence boundaries where possible, and bound unpunctuated input too. */
export const splitParagraph = (text: string, limit: number): Array<string> => {
  const pieces: Array<string> = []
  for (let sentence of text.split(/(?<=[.!?])\s+|(?<=[。！？])\s*/)) {
    sentence = sentence.trim()
    while (sentence.length > limit) {
      // Break a long sentence after a clause where possible, else between words.
      const clause = Math.max(...[",", ";", ":", "，", " –", " —"].map((mark) => findLast(sentence, mark, Math.floor(limit / 2), limit)))
      let boundary: number
      if (clause >= 0) {
        boundary = clause + 1
      } else {
        boundary = findLast(sentence, " ", 0, limit + 1)
        if (boundary < Math.floor(limit / 2)) boundary = limit
      }
      pieces.push(sentence.slice(0, boundary).trim())
      sentence = sentence.slice(boundary).trim()
    }
    if (sentence) pieces.push(sentence)
  }
  const chunks: Array<string> = []
  let current = ""
  for (const piece of pieces) {
    if (current && current.length + piece.length + 1 > limit) {
      chunks.push(current)
      current = ""
    }
    current = `${current} ${piece}`.trim()
  }
  if (current) chunks.push(current)
  return chunks
}

/**
 * Plan the sections to render, each with the pause that follows it. Paragraphs never share a
 * section, so the model never reads across a paragraph break or a change of speaker; long ones
 * split at sentences. Blank lines separate paragraphs; without any, each line is one.
 */
export const splitScript = (text: string, limit = SECTION_CHARACTERS): Array<Section> => {
  const trimmed = text.trim()
  const paragraphs = /\n\s*\n/.test(trimmed) ? trimmed.split(/\n\s*\n/) : trimmed.split("\n")
  const sections: Array<{ text: string; pause: number }> = []
  for (const paragraph of paragraphs) {
    for (const chunk of splitParagraph(paragraph.replace(/\s+/g, " ").trim(), limit)) sections.push({ text: chunk, pause: SENTENCE_PAUSE })
    if (sections.length) sections.at(-1)!.pause = PARAGRAPH_PAUSE
  }
  if (sections.length) sections.at(-1)!.pause = 0
  return sections
}

/**
 * When the text's final words end in recognized speech, or undefined. The first place they
 * appear, since the model sometimes trails off into extra syllables after the text.
 */
export const speechEnd = (text: string, words: ReadonlyArray<Word>, size = 3): number | undefined => {
  const ending = tokens(text).slice(-size)
  const heard = words.flatMap((word, owner) => tokens(word.text).map((token) => ({ token, owner })))
  if (ending.length === 0) return undefined
  for (let at = 0; at < heard.length; at++) {
    if (!sameWord(ending.at(-1)!, heard[at]!.token)) continue
    let matched = 0
    let j = at
    const earliest = Math.max(0, at - ending.length - 2)
    for (const expected of [...ending].reverse()) {
      let k = j
      while (k >= earliest && !sameWord(expected, heard[k]!.token)) k--
      if (k >= earliest) {
        matched++
        j = k - 1
      }
    }
    if (matched >= ending.length - 1) return words[heard[at]!.owner]!.end
  }
  return undefined
}

/** The share of the text's words heard in order (longest common subsequence). */
export const coverage = (text: string, words: ReadonlyArray<Word>) => {
  const expected = tokens(text)
  const heard = words.flatMap((word) => tokens(word.text))
  if (expected.length === 0) return 1
  let previous = new Array<number>(heard.length + 1).fill(0)
  for (const word of expected) {
    const current = [0]
    heard.forEach((other, j) => current.push(sameWord(word, other) ? previous[j]! + 1 : Math.max(previous[j + 1]!, current[j]!)))
    previous = current
  }
  return previous.at(-1)! / expected.length
}

// Titles that end with a full stop but not a sentence.
const ABBREVIATIONS = new Set(["mr.", "mrs.", "ms.", "dr.", "st.", "mt.", "jr.", "sr.", "vs.", "etc."])
const CLOSING = /["')”’]+$/

export const endsSentence = (text: string) => {
  const stripped = text.replace(CLOSING, "")
  return /[.!?]$/.test(stripped) && !ABBREVIATIONS.has(stripped.split(/\s+/).at(-1)!.toLowerCase())
}

/** How natural a place to end a clip: after a clause, then a pause. */
const boundary = (words: ReadonlyArray<Word>, index: number) => {
  const text = words[index]!.text.replace(CLOSING, "")
  const punctuation = /[,;:–—]$/.test(text) ? 1 : 0
  const gap = index + 1 < words.length ? words[index + 1]!.start - words[index]!.end : 1
  return punctuation + Math.min(gap, 1)
}

/**
 * Split a take's timed words into clips that end at the end of a sentence, as [first, last]
 * word indexes. The model continues from a clip, so one ending mid-sentence would make it read
 * new text as the rest of that sentence.
 */
export const planClips = (words: ReadonlyArray<Word>, [shortest, longest, stretch] = CLIP_SECONDS) => {
  const spans: Array<readonly [number, number]> = []
  let first = 0
  while (first < words.length) {
    const length = (index: number) => words[index]!.end - words[first]!.start
    let reachable: Array<number> = []
    for (let i = first; i < words.length; i++) if (length(i) <= stretch) reachable.push(i)
    if (reachable.length === 0) reachable = [first]
    const sentences = reachable.filter((i) => endsSentence(words[i]!.text) && length(i) >= 3)
    const natural = sentences.filter((i) => length(i) >= shortest && length(i) <= longest)
    const short = sentences.filter((i) => length(i) < shortest)
    const long = sentences.filter((i) => length(i) > longest)
    let end: number
    if (natural.length || short.length || long.length) {
      end = natural.length ? Math.max(...natural) : short.length ? Math.max(...short) : Math.min(...long)
    } else {
      // No sentence ends in reach: end at the most natural pause instead.
      const within = reachable.filter((i) => length(i) <= longest)
      end = (within.length ? within : [first]).reduce((best, i) => {
        const [a, b] = [boundary(words, i), boundary(words, best)]
        return a > b || (a === b && i > best) ? i : best
      })
    }
    if (length(end) >= 3) spans.push([first, end])
    first = end + 1
  }
  return spans
}

/** Each planned clip's sample range in a take and its exact words. */
export const clipRanges = (words: ReadonlyArray<Word>, rate: number) =>
  planClips(words).map(([first, last]) => {
    let start = Math.max(0, words[first]!.start - 0.1)
    if (first > 0) start = Math.max(start, words[first - 1]!.end)
    let end = words[last]!.end + 0.25
    if (last + 1 < words.length) end = Math.min(end, words[last + 1]!.start)
    return {
      start: Math.floor(start * rate),
      end: Math.floor(end * rate),
      text: words.slice(first, last + 1).map((word) => word.text).join(" ")
    }
  })
