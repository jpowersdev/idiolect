import { describe, expect, test } from "bun:test"
import { Effect, Exit, Fiber, Layer } from "effect"
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { IdiolectError } from "../src/Errors.ts"
import { checkOutput, inputText, playRender, publish, renderVoice } from "../src/Library.ts"
import { fakeAudio, fakeEngine, render, temporaryDirectory } from "./support.ts"

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

test("interrupting a render cancels that render", async () => {
  const cancelled: Array<string> = []
  const engine = fakeEngine({
    generate: () => Effect.succeed(render()),
    render: () => Effect.succeed(render({ status: "generating", message: "Rendering section 1 of 2…" })),
    cancel: ({ params }: { params: { id: string } }) =>
      Effect.sync(() => {
        cancelled.push(params.id)
        return render({ status: "cancelled" })
      })
  })
  const exit = await Effect.runPromise(
    Effect.gen(function*() {
      const fiber = yield* Effect.forkChild(renderVoice("Hello"))
      yield* Effect.sleep("700 millis")
      return yield* Fiber.interrupt(fiber).pipe(Effect.andThen(Fiber.await(fiber)))
    }).pipe(Effect.provide(engine))
  )
  expect(Exit.hasInterrupts(exit)).toBe(true)
  expect(cancelled).toEqual(["take123"])
})

test("a failed render reports the engine's message", async () => {
  const engine = fakeEngine({
    generate: () => Effect.succeed(render({ status: "failed", message: "The GPU ran out of memory." }))
  })
  const error = await Effect.runPromise(renderVoice("Hello").pipe(Effect.flip, Effect.provide(engine)))
  expect(error.message).toBe("The GPU ran out of memory.")
})

describe("playback", () => {
  for (const failure of [false, true]) {
    test(failure ? "a player error still removes the temporary audio" : "plays and removes the temporary audio", async () => {
      const played: Array<string> = []
      const layer = Layer.mergeAll(
        fakeEngine({ audio: () => Effect.succeed(new TextEncoder().encode("generated audio")) }),
        fakeAudio({
          play: (path) =>
            Effect.suspend(() => {
              played.push(path)
              expect(readFileSync(path, "utf8")).toBe("generated audio")
              return failure ? Effect.fail(new IdiolectError({ message: "Playback failed: No audio device" })) : Effect.void
            })
        })
      )
      const exit = await Effect.runPromiseExit(playRender(render({ status: "ready", duration: 1.5 })).pipe(Effect.provide(layer)))
      expect(Exit.isFailure(exit)).toBe(failure)
      expect(played.length).toBe(1)
      expect(existsSync(played[0]!)).toBe(false)
    })
  }
})
