import { describe, expect, test } from "bun:test"
import { Console, Effect, Exit, Layer } from "effect"
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { IdiolectError } from "../src/Errors.ts"
import { checkOutput, inputText, playRender, publish, renderVoice } from "../src/Library.ts"
import { fakeAudio, fakeEngine, temporaryDirectory } from "./support.ts"

const text = (options: Partial<Parameters<typeof inputText>[0]>) =>
  inputText({ words: [], stdinIsTTY: true, readStdin: () => "", ...options })

test("scripts come from words or a pipe", () => {
  expect(text({ words: ["Hello", "there"] })).toBe("Hello there")
  expect(text({ stdinIsTTY: false, readStdin: () => "A piped script.\n" })).toBe("A piped script.")
  expect(() => text({ words: ["x".repeat(12001)] })).toThrow("12,000")
  expect(() => text({})).toThrow("Give me words")
})

describe("saving audio", () => {
  test("the output path is checked before any work starts", () => {
    const folder = temporaryDirectory()
    expect(() => checkOutput(join(folder, "speech.mp3"))).toThrow(".wav")
    expect(() => checkOutput(join(folder, "missing", "speech.wav"))).toThrow("does not exist")
    expect(() => checkOutput(join(folder, "speech.wav"))).not.toThrow()
  })

  test("publishing replaces the file atomically", () => {
    const folder = temporaryDirectory()
    const output = join(folder, "speech.wav")
    writeFileSync(output, "old audio")
    publish(new TextEncoder().encode("new audio"), output)
    expect(readFileSync(output, "utf8")).toBe("new audio")
    expect(readdirSync(folder)).toEqual(["speech.wav"])
  })
})

test("speaking shows each step, and its failures in words", async () => {
  const shown: Array<string> = []
  const engine = fakeEngine({
    speak: (_text, progress) => Effect.andThen(progress("Rendering section 1 of 1…"), Effect.fail(new IdiolectError({ message: "The speech model is missing. Run idiolect setup first." })))
  })
  const error = await Effect.runPromise(renderVoice("Hello").pipe(
    Effect.flip,
    Effect.provide(engine),
    Effect.provideService(Console.Console, { ...globalThis.console, error: (message: string) => shown.push(message) } as never)
  ))
  expect(shown).toEqual(["Rendering section 1 of 1…"])
  expect(error.message).toContain("Run idiolect setup")
})

describe("playback", () => {
  for (const failure of [false, true]) {
    test(failure ? "a player error still removes the temporary audio" : "plays and removes the temporary audio", async () => {
      const played: Array<string> = []
      const layer = Layer.mergeAll(
        fakeAudio({
          play: (path) =>
            Effect.suspend(() => {
              played.push(path)
              expect(readFileSync(path, "utf8")).toBe("generated audio")
              return failure ? Effect.fail(new IdiolectError({ message: "Playback failed: No audio device" })) : Effect.void
            })
        })
      )
      const exit = await Effect.runPromiseExit(playRender({ wav: new TextEncoder().encode("generated audio"), duration: 1.5 }).pipe(Effect.provide(layer)))
      expect(Exit.isFailure(exit)).toBe(failure)
      expect(played.length).toBe(1)
      expect(existsSync(played[0]!)).toBe(false)
    })
  }
})
