import { describe, expect, test } from "bun:test"
import type { Word } from "../../src/Api.ts"
import { clipRanges, coverage, endsSentence, PARAGRAPH_PAUSE, planClips, SENTENCE_PAUSE, speechEnd, splitScript } from "../../src/engine/Sections.ts"

/** A word every half second; every `sentence`th word, and the last, ends a sentence. */
const timedWords = (seconds: number, sentence = 6): Array<Word> => {
  const count = seconds * 2
  return Array.from({ length: count }, (_, i) => ({
    text: `word${i}${i % sentence === sentence - 1 || i === count - 1 ? "." : ""}`,
    start: i * 0.5,
    end: i * 0.5 + 0.4
  }))
}

describe("sections", () => {
  for (const script of [
    "The value is 3.14. That's useful! ".repeat(35),
    "A sentence without punctuation ".repeat(100),
    "你好世界。这里是一个声音测试！".repeat(100),
    "a".repeat(950),
    "First paragraph.\n\nSecond paragraph.\nThird paragraph?"
  ]) {
    test(`keep all of the script, in bounded sections: ${JSON.stringify(script.slice(0, 24))}`, () => {
      const sections = splitScript(script)
      const texts = sections.map((s) => s.text)
      expect(texts.length).toBeGreaterThan(0)
      expect(texts.every((t) => t.length > 0 && t.length <= 300)).toBe(true)
      expect(texts.join("").replace(/\s+/g, "")).toBe(script.replace(/\s+/g, ""))
      expect(sections.at(-1)!.pause).toBe(0)
      if (script.includes("3.14")) expect(texts.every((t) => !t.includes("3. 14"))).toBe(true)
    })
  }

  test("paragraphs get their own sections and longer pauses", () => {
    const story = "It never did.\n\n\"You're up early,\" said a voice beside her.\n\nShe turned. " +
      "An older man was holding a newspaper he had no intention of reading. ".repeat(5)
    const sections = splitScript(story)
    expect(sections.slice(0, 2).map((s) => s.text.slice(0, 12))).toEqual(["It never did", "\"You're up e"])
    expect(sections.slice(0, 2).map((s) => s.pause)).toEqual([PARAGRAPH_PAUSE, PARAGRAPH_PAUSE])
    // A long paragraph splits at sentences, with the shorter pause between its sections.
    expect(sections.length).toBeGreaterThan(3)
    expect(sections[2]!.pause).toBe(SENTENCE_PAUSE)
    expect(sections.at(-1)!.pause).toBe(0)
    // A long sentence breaks after a clause, not between two words of it.
    const long = "Back then, her grandfather would point out every landmark along the coast: the red " +
      "lighthouse, the cliffs where the seals slept, the crooked dock that everyone swore " +
      "would fall into the sea by next winter."
    const texts = splitScript(long).map((s) => s.text)
    expect(texts.length).toBe(2)
    expect(texts[0]!.endsWith(",")).toBe(true)
    expect(texts[1]!.endsWith("next winter.")).toBe(true)
    // Without blank lines, each line is a paragraph; with them, wrapped lines rejoin.
    expect(splitScript("One line.\nAnother line.").length).toBe(2)
    expect(splitScript("A paragraph that\nwraps.\n\nAnother.").length).toBe(2)
  })
})

describe("checking a rendered section", () => {
  const heard: Array<Word> = [
    { text: " A", start: 0, end: 0.2 },
    { text: " completely", start: 0.2, end: 0.8 },
    { text: " new", start: 0.8, end: 1 },
    { text: " story.", start: 1, end: 1.4 },
    { text: " Sorry.", start: 1.4, end: 1.6 },
    { text: " Sorry.", start: 1.6, end: 1.8 }
  ]

  test("the end of speech ignores trailing babble", () => {
    expect(speechEnd("A familiar voice can tell a completely new story.", heard)).toBe(1.4)
    // The recognizer's spelling of a name still counts.
    expect(speechEnd("She walked with Sarah.", [{ text: "walked with Sara.", start: 0, end: 2 }])).toBe(2)
    // Garbled before the end: no end found.
    expect(speechEnd("A completely new story.", heard.slice(0, 2))).toBeUndefined()
  })

  test("coverage is the share of words heard in order", () => {
    expect(coverage("A completely new story.", heard)).toBe(1)
    expect(coverage("A completely new story.", heard.slice(0, 2))).toBe(0.5)
  })
})

describe("clips", () => {
  test("end at sentences, within natural lengths", () => {
    const words = timedWords(40)
    const spans = planClips(words)
    expect(spans[0]![0]).toBe(0)
    expect(spans.at(-1)![1]).toBe(words.length - 1)
    spans.slice(1).forEach(([first], i) => expect(first).toBe(spans[i]![1] + 1))
    for (const [first, last] of spans) {
      const length = words[last]!.end - words[first]!.start
      expect(length).toBeGreaterThanOrEqual(3)
      expect(length).toBeLessThanOrEqual(15)
      expect(words[last]!.text.endsWith(".")).toBe(true)
    }
  })

  test("stretch to reach a sentence end rather than stop at a comma", () => {
    const longSentence = Array.from({ length: 36 }, (_, i) => ({
      text: `word${i}${i === 25 ? "," : i === 35 ? "." : ""}`,
      start: i * 0.5,
      end: i * 0.5 + 0.4
    }))
    expect(planClips(longSentence)).toEqual([[0, 35]])
    // A single word longer than a clip still moves the plan forward.
    expect(planClips([{ text: "Hmm.", start: 0, end: 20 }])).toEqual([[0, 0]])
  })

  test("sentence ends exclude titles, and allow closing quotes", () => {
    expect(endsSentence("bird.")).toBe(true)
    expect(endsSentence("\"Comma?\"")).toBe(true)
    expect(endsSentence("Mrs.")).toBe(false)
  })

  test("cut between words, with a short lead-in", () => {
    const [range] = clipRanges(timedWords(20), 24000)
    expect(range!.start).toBe(0)
    expect(range!.end).toBe(Math.floor(14.9 * 24000) + Math.floor(0.1 * 24000))
    expect(range!.text.endsWith("word29.")).toBe(true)
  })
})
