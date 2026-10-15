/** Scripts, renders, and saved audio. */
import { Console, Duration, Effect } from "effect"
import { existsSync, mkdtempSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { activeStatuses, MaxScriptLength, type Render } from "./Api.ts"
import { Audio } from "./Audio.ts"
import { Engine } from "./Engine.ts"
import { fail } from "./Errors.ts"

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

/** Submit a render and follow its progress. Ctrl+C cancels it after the current section. */
export const renderVoice = Effect.fn("renderVoice")(function*(text: string) {
  const engine = yield* Engine
  let job: Render = yield* engine.call(engine.api.generate({ payload: { text } }))
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
