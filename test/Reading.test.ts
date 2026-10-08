import { describe, expect, test } from "bun:test"
import { encodeWav, matchReading, pcmSamples, SampleRate, saveReading } from "../src/Reading.ts"
import { words } from "./support.ts"

describe("following a reading", () => {
  test("requires all words in order", () => {
    const script = "A warm cup of coffee waited beside a blue notebook."
    expect(matchReading(script, words(script.toUpperCase())).complete).toBe(true)
    for (
      const incomplete of [
        "",
        "A warm cup of coffee",
        "beside a blue notebook",
        "A warm cup of coffee waited beside a notebook",
        "A warm cup of coffee waited beside a red notebook",
        "notebook blue a beside waited coffee of cup warm A"
      ]
    ) {
      expect(matchReading(script, incomplete ? words(incomplete) : []).complete).toBe(false)
    }
  })

  test("reports a missing word even when the ending was spoken", () => {
    const match = matchReading("A quiet morning by the river.", words("A morning by the river."))
    expect(match).toMatchObject({ matched: 5, total: 6, complete: false })
    expect(match.nextWords.startsWith("quiet")).toBe(true)
  })

  test("keeps heard fillers and trims outside the reading", () => {
    const script = "A quiet morning beside the blue river."
    const heard = words("Ready. A quiet morning um beside the blue river. Thanks.")
    const match = matchReading(script, heard)
    expect(match.complete).toBe(true)
    const saved = saveReading(new Int16Array(SampleRate * 9).fill(3277), heard, match)
    expect(saved.transcript).toBe("A quiet morning um beside the blue river.")
    expect(saved.duration).toBe(4.6)
    expect(pcmSamples(saved.wav.subarray(44)).length).toBe(Math.round(4.6 * SampleRate))
  })

  test("never saves an incomplete reading", () => {
    const heard = words("A quiet morning")
    expect(() =>
      saveReading(new Int16Array(SampleRate * 5), heard, matchReading("A quiet morning beside the river.", heard))
    ).toThrow("full script")
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
