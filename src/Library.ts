/** Voice selection, scripts, and saved audio: the CLI's side of the voice library. */
import { Console, Duration, Effect } from "effect"
import { existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { activeStatuses, type Language, type Render, type Voice } from "./Api.ts"
import { Audio } from "./Audio.ts"
import { Engine } from "./Engine.ts"
import { fail } from "./Errors.ts"

export const MaxScriptLength = 12000

export const settingsPath = (data: string) => join(data, "cli.json")

export const defaultVoice = (data: string): string | undefined => {
  const path = settingsPath(data)
  if (!existsSync(path)) return undefined
  try {
    const value = JSON.parse(readFileSync(path, "utf8")) as { default_voice?: unknown }
    return typeof value.default_voice === "string" ? value.default_voice : undefined
  } catch (error) {
    throw new Error(`Cannot read ${path}: ${error}`)
  }
}

export const setDefaultVoice = (data: string, voice: Voice) => {
  mkdirSync(data, { recursive: true })
  const path = settingsPath(data)
  writeFileSync(`${path}.tmp`, JSON.stringify({ default_voice: voice.id }, null, 2))
  renameSync(`${path}.tmp`, path)
}

/** Match a saved name, full ID, or unambiguous ID prefix; otherwise the default voice. */
export const chooseVoice = (
  voices: ReadonlyArray<Voice>,
  selected: string | undefined,
  fallback: string | undefined
): Voice => {
  const wanted = selected ?? fallback
  if (!wanted) throw new Error("Set up your voice with idiolect setup, or choose one with --voice.")
  let matches = voices.filter((voice) => voice.id === wanted)
  if (matches.length === 0) {
    const folded = wanted.toLowerCase()
    matches = voices.filter((voice) => voice.name.toLowerCase() === folded || voice.id.startsWith(wanted))
  }
  if (matches.length !== 1) throw new Error("Voice not found or ambiguous. Run idiolect voices and use its ID.")
  return matches[0]!
}

/** Words on the command line, a file (or - for stdin), or piped input. */
export const inputText = (options: {
  readonly words: ReadonlyArray<string>
  readonly file: string | undefined
  readonly stdinIsTTY: boolean
  readonly readStdin: () => string
  readonly readFile: (path: string) => string
}) => {
  if (options.words.length > 0 && options.file !== undefined) {
    throw new Error("Use words on the command line or --file, rather than both.")
  }
  let text: string
  if (options.file !== undefined) {
    text = options.file === "-" ? options.readStdin() : options.readFile(options.file)
  } else if (options.words.length > 0) {
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

/** Validate a --output path before doing any work, so existing files are never at risk. */
export const checkOutput = (output: string, force: boolean) => {
  if (!output.toLowerCase().endsWith(".wav")) throw new Error("Choose a .wav output file.")
  if (existsSync(output) && !force) {
    throw new Error(`${output} already exists. Choose another path or add --force.`)
  }
  const parent = dirname(output)
  if (!existsSync(parent) || !statSync(parent).isDirectory()) {
    throw new Error(`The output directory does not exist: ${parent}`)
  }
}

/** Publish a complete file atomically, preserving an existing file unless forced. */
export const publish = (bytes: Uint8Array, output: string, force: boolean) => {
  const temporary = join(dirname(output), `.idiolect-${process.pid}-${Date.now()}.wav`)
  try {
    writeFileSync(temporary, bytes)
    if (force) {
      renameSync(temporary, output)
    } else {
      linkSync(temporary, output)
    }
  } catch (error) {
    if (!force && (error as NodeJS.ErrnoException).code === "EEXIST") {
      throw new Error(`${output} already exists. Choose another path or add --force.`)
    }
    throw error
  } finally {
    if (existsSync(temporary)) unlinkSync(temporary)
  }
}

export const attempt = <A>(f: () => A) =>
  Effect.try({ try: f, catch: (error) => fail(error instanceof Error ? error.message : String(error)) })

/** Submit a render and follow its progress. Ctrl+C cancels it after the current section. */
export const renderVoice = Effect.fn("renderVoice")(function*(
  voice: Voice,
  text: string,
  language: Language = "English",
  seed = 12
) {
  const engine = yield* Engine
  yield* Console.error(`Rendering as "${voice.name}"…`)
  let job: Render = yield* engine.call(
    engine.api.generate({ payload: { voice_id: voice.id, text, language, seed } })
  )
  let previous: string | undefined
  const follow = Effect.gen(function*() {
    while (activeStatuses.has(job.status)) {
      if (job.message !== previous) {
        yield* Console.error(job.message)
        previous = job.message
      }
      yield* Effect.sleep(Duration.millis(500))
      job = yield* engine.call(engine.api.render({ params: { id: job.id } }))
    }
    return job
  }).pipe(
    Effect.onInterrupt(() =>
      engine.call(engine.api.cancel({ params: { id: job.id } })).pipe(
        Effect.andThen(Console.error("\nStopping this take after the current section.")),
        Effect.ignore
      )
    )
  )
  const finished = yield* follow
  if (finished.status !== "ready") return yield* fail(finished.message)
  return finished
})

export const renderAudio = Effect.fn("renderAudio")(function*(job: Render) {
  const engine = yield* Engine
  return yield* engine.call(engine.api.audio({ params: { id: job.id } }))
})

/** Play a finished render from a private temporary file that is always removed. */
export const playRender = Effect.fn("playRender")(function*(job: Render) {
  const audio = yield* Audio
  const bytes = yield* renderAudio(job)
  yield* Effect.acquireUseRelease(
    attempt(() => mkdtempSync(join(tmpdir(), "idiolect-playback-"))),
    (folder) =>
      Effect.gen(function*() {
        const path = join(folder, "voiceover.wav")
        yield* attempt(() => writeFileSync(path, bytes))
        yield* Console.error(`Playing ${job.duration ?? 0}s of audio…`)
        yield* audio.play(path)
      }),
    (folder) => Effect.sync(() => rmSync(folder, { recursive: true, force: true }))
  )
})
