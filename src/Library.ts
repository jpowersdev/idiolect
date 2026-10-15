/** Scripts, renders, and saved audio. */
import { Console, Effect } from "effect"
import { existsSync, mkdtempSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { Audio } from "./Audio.ts"
import { Engine, type Spoken } from "./Engine.ts"
import { fail } from "./Errors.ts"

export const MaxScriptLength = 12000

/** Words on the command line, or piped input. */
export const inputText = (options: {
  readonly words: ReadonlyArray<string>
  readonly stdinIsTTY: boolean
  readonly readStdin: () => string
}) => {
  let text: string
  if (options.words.length > 0) {
    text = options.words.join(" ")
  } else if (!options.stdinIsTTY) {
    text = options.readStdin()
  } else {
    throw new Error("Give me words: idiolect say \"Hello there\"")
  }
  text = text.trim()
  if (text.length < 1 || text.length > MaxScriptLength) {
    throw new Error("Use a script between 1 and 12,000 characters.")
  }
  return text
}

/** Validate an --output path before doing any work. */
export const checkOutput = (output: string) => {
  if (!output.toLowerCase().endsWith(".wav")) throw new Error("Choose a .wav output file.")
  const parent = dirname(output)
  if (!existsSync(parent) || !statSync(parent).isDirectory()) {
    throw new Error(`The output directory does not exist: ${parent}`)
  }
}

/** Write a complete file atomically, so a failure never leaves a partial WAV. */
export const publish = (bytes: Uint8Array, output: string) => {
  const temporary = join(dirname(output), `.idiolect-${process.pid}-${Date.now()}.wav`)
  try {
    writeFileSync(temporary, bytes)
    renameSync(temporary, output)
  } finally {
    if (existsSync(temporary)) unlinkSync(temporary)
  }
}

export const attempt = <A>(f: () => A) =>
  Effect.try({ try: f, catch: (error) => fail(error instanceof Error ? error.message : String(error)) })

/** Speak a script in your voice, showing each step. Ctrl+C stops it at once. */
export const renderVoice = Effect.fn("renderVoice")(function*(text: string) {
  const engine = yield* Engine
  return yield* engine.speak(text, (message) => Console.error(message))
})

/** Play a finished render from a private temporary file that is always removed. */
export const playRender = Effect.fn("playRender")(function*(spoken: Spoken) {
  const audio = yield* Audio
  yield* Effect.acquireUseRelease(
    attempt(() => mkdtempSync(join(tmpdir(), "idiolect-playback-"))),
    (folder) =>
      Effect.gen(function*() {
        const path = join(folder, "voiceover.wav")
        yield* attempt(() => writeFileSync(path, spoken.wav))
        yield* Console.error(`Playing ${spoken.duration}s of audio…`)
        yield* audio.play(path)
      }),
    (folder) => Effect.sync(() => rmSync(folder, { recursive: true, force: true }))
  )
})
