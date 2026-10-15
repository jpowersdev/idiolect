import { expect, test } from "bun:test"
import { Effect } from "effect"
import { render, renderSection, type Voices } from "../../src/engine/Render.ts"
import { readWav } from "../../src/engine/Signal.ts"

const rate = 24000

const voices = (changes: Partial<Voices>): Voices => ({
  rate,
  load: Effect.void,
  speak: () => Effect.succeed(new Float32Array(rate).fill(0.2)),
  transcribe: () => Effect.succeed([]),
  ...changes
})

test("garbled sections are rendered again, and cut after their final words", async () => {
  let spoken = 0
  let heard = 0
  const stubs = voices({
    // Speech, a short gap, then a stray syllable.
    speak: () =>
      Effect.sync(() => {
        const level = 0.1 * ++spoken
        const audio = new Float32Array(14400 + 4800 + 4800).fill(level)
        audio.fill(0, 14400, 19200)
        return audio
      }),
    // Garbling on the first attempt, then the whole text and a trailing syllable.
    transcribe: () =>
      Effect.sync(() =>
        ++heard === 1
          ? [{ text: "A completely nyuh", start: 0, end: 0.8 }]
          : [{ text: "A completely new story.", start: 0, end: 0.6 }, { text: "Sorry.", start: 0.6, end: 0.9 }]
      )
  })
  const audio = await Effect.runPromise(renderSection(stubs, "A completely new story.", 0))
  expect(spoken).toBe(2)
  expect(audio.subarray(0, 14400).every((x) => Math.abs(x - 0.2) < 1e-6)).toBe(true)
  expect(audio.length).toBe(Math.floor(0.65 * rate))
})

test("a script is every section joined, with its pauses, showing each step", async () => {
  const shown: Array<string> = []
  const spoken = await Effect.runPromise(render(
    voices({ transcribe: () => Effect.succeed([{ text: "Hi.", start: 0, end: 1 }]) }),
    "Hi.\n\nHi.",
    (message) => Effect.sync(() => shown.push(message))
  ))
  expect(shown).toEqual(["Loading the voice model…", "Rendering section 1 of 2…", "Rendering section 2 of 2…"])
  const audio = readWav(spoken.wav)
  expect(audio.rate).toBe(rate)
  // Two seconds of speech and the pause between the paragraphs.
  expect(audio.samples.length / rate).toBeCloseTo(2.75, 1)
  expect(spoken.duration).toBeCloseTo(2.75, 1)
})
