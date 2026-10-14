import { describe, expect, test } from "bun:test"
import { encodeWav, findEnding, pcmSamples, sameWord, SampleRate, saveReading } from "../src/Reading.ts"
import { words } from "./support.ts"

const script = "Then she put on a plain yellow dress, picked up her kit and headed for work."

describe("hearing the end of a reading", () => {
  test("the script's final words, at the end of what was heard, finish it", () => {
    expect(findEnding(script, words("up her kit and headed for work."))).toEqual({ done: true, lastWord: 6 })
    expect(findEnding(script, words("UP HER KIT AND HEADED FOR WORK"))).toMatchObject({ done: true })
    // The recognizer repeating the last words into the silence afterwards is kept out too.
    expect(findEnding(script, words("and headed for work. work."))).toEqual({ done: true, lastWord: 3 })
    expect(findEnding(script, words("kit and headed for work. headed for work."))).toEqual({ done: true, lastWord: 4 })
    // Words said after the ending are kept out of the reading.
    expect(findEnding(script, words("and headed for work. Okay, done."))).toEqual({ done: true, lastWord: 3 })
  })

  test("anything short of the ending keeps listening", () => {
    for (const heard of ["", "Then she put on a plain yellow dress", "picked up her kit and headed for", "work for headed and kit"]) {
      expect(findEnding(script, heard ? words(heard) : []).done).toBe(false)
    }
  })

  test("one misheard word in the ending is fine", () => {
    expect(findEnding(script, words("her kit and hoarded for work."))).toEqual({ done: true, lastWord: 5 })
    // A misheard final word counts once something was heard in its place.
    expect(findEnding(script, words("her kit and headed for walk."))).toEqual({ done: true, lastWord: 5 })
    expect(findEnding(script, words("her kit and headed for")).done).toBe(false)
  })

  test("the recognizer's own spellings count as the script's words", () => {
    expect(sameWord("sarah", "sara")).toBe(true)
    expect(sameWord("grey", "gray")).toBe(true)
    expect(sameWord("harrison", "harrisson")).toBe(true)
    expect(sameWord("work", "walk")).toBe(false)
    expect(sameWord("cat", "cut")).toBe(false)
  })
})

describe("saving a reading", () => {
  test("trims from the first heard word through the ending, keeping fillers", () => {
    const heard = words("A quiet morning um beside the blue river. Thanks.")
    const saved = saveReading(new Int16Array(SampleRate * 9).fill(3277), heard, 7)
    expect(saved.transcript).toBe("A quiet morning um beside the blue river.")
    expect(saved.duration).toBe(4.6)
    expect(Math.abs(pcmSamples(saved.wav.subarray(44)).length - 4.6 * SampleRate)).toBeLessThanOrEqual(1)
    // Word timings are relative to the trimmed clip, which starts 0.25s before "A".
    expect(saved.words[0]).toEqual({ text: "A", start: 0.25, end: 0.75 })
    expect(saved.words.at(-1)).toEqual({ text: "river.", start: 3.75, end: 4.25 })
  })

  test("nothing heard is never saved", () => {
    expect(() => saveReading(new Int16Array(SampleRate * 5), [], 0)).toThrow("Nothing was recognized")
  })
})

test("WAV encoding preserves samples at 48 kHz", () => {
  const samples = Int16Array.from([0, 1, -1, 32767, -32768])
  const wav = encodeWav(samples, SampleRate)
  const view = new DataView(wav.buffer)
  expect(new TextDecoder().decode(wav.subarray(0, 4))).toBe("RIFF")
  expect(view.getUint32(24, true)).toBe(SampleRate)
  expect(view.getUint16(34, true)).toBe(16)
  expect(Array.from(pcmSamples(wav.subarray(44)))).toEqual(Array.from(samples))
})
