import { expect, test } from "bun:test"
import { cutAfter, endingFade, GAIN_LIMITS_DB, joinSections, matchLevel, readWav, resample, speechLevel, TARGET_DB, writeWav } from "../../src/engine/Signal.ts"

const full = (length: number, value: number) => new Float32Array(length).fill(value)
const concat = (...parts: Array<Float32Array>) => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0))
  let at = 0
  for (const p of parts) {
    out.set(p, at)
    at += p.length
  }
  return out
}

test("WAV files round-trip at 16 bits", () => {
  const samples = Float32Array.from([0, 0.5, -0.5, 1, -1])
  const { rate, samples: read } = readWav(writeWav({ samples, rate: 24000 }))
  expect(rate).toBe(24000)
  read.forEach((x, i) => expect(Math.abs(x - samples[i]!)).toBeLessThan(1 / 32000))
})

test("a recognition copy is anti-aliased, not folded", () => {
  const rate = 48000
  const audio = Float32Array.from({ length: rate * 6 }, (_, i) => 0.2 * Math.sin((2 * Math.PI * 500 * i) / rate) + 0.1 * Math.sin((2 * Math.PI * 10000 * i) / rate))
  const copy = resample(audio, rate, 16000)
  expect(copy.length).toBe(16000 * 6)
  let error = 0
  for (let i = 100; i < copy.length - 100; i++) error += (copy[i]! - 0.2 * Math.sin((2 * Math.PI * 500 * i) / 16000)) ** 2
  // The 10 kHz tone is removed rather than folded into the recognizer's band.
  expect(Math.sqrt(error / (copy.length - 200))).toBeLessThan(0.001)
})

test("joining trims each section and adds its pause", () => {
  const silence = full(12000, 0)
  const joined = joinSections([{ samples: concat(silence, full(24000, 0.2), silence), pause: 0.75 }, { samples: full(24000, -0.2), pause: 0 }], 24000)
  const seconds = joined.length / 24000
  expect(seconds).toBeGreaterThanOrEqual(2.9)
  expect(seconds).toBeLessThanOrEqual(3)
  expect(joined.subarray(30000, 46000).every((x) => x === 0)).toBe(true)
})

test("cutting after speech keeps a soft final consonant", () => {
  const rate = 24000
  // A word, then a quiet "s" running past the recognizer's estimated end, then silence.
  const [word, hiss] = [full(9600, 0.5), full(3600, 0.02)]
  const cut = cutAfter(concat(word, hiss, full(9600, 0)), rate, 0.4)
  expect(cut.length).toBeGreaterThanOrEqual(word.length + hiss.length)
  expect(cut.length).toBeLessThanOrEqual(word.length + hiss.length + 0.1 * rate)
})

test("a fading ending is detected", () => {
  const rate = 24000
  const levels = [0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.02, 0.005]
  const audio = concat(...levels.map((level) => full(rate / 4, level)))
  const words = levels.map((_, i) => ({ start: i * 0.25, end: (i + 1) * 0.25 }))
  expect(endingFade(audio, rate, words)).toBeGreaterThan(12)
  expect(Math.abs(endingFade(full(rate * 2, 0.3), rate, words))).toBeLessThan(1)
})

test("sections are brought to a common level, within limits", () => {
  const rate = 24000
  expect(Math.abs(speechLevel(matchLevel(full(rate, 10 ** (-36 / 20)), rate), rate) - TARGET_DB)).toBeLessThan(0.5)
  // A nearly silent section is lifted by at most the limit, not blown up.
  expect(Math.abs(speechLevel(matchLevel(full(rate, 10 ** (-60 / 20)), rate), rate) - (-60 + GAIN_LIMITS_DB[1]))).toBeLessThan(0.5)
})
