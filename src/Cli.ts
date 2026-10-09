/** Guided voice enrollment and local text-to-speech commands. */
import { Console, Effect, Option } from "effect"
import { Argument, Command, Flag } from "effect/cli"
import { readFileSync } from "node:fs"
import { createInterface } from "node:readline"
import { resolve } from "node:path"
import { Languages, type Voice } from "./Api.ts"
import { Audio } from "./Audio.ts"
import { DefaultPort, Engine } from "./Engine.ts"
import { recordReading } from "./Enroll.ts"
import { fail, IdiolectError } from "./Errors.ts"
import {
  attempt,
  checkOutput,
  chooseVoice,
  defaultVoice,
  inputText,
  playRender,
  publish,
  renderAudio,
  renderVoice,
  setDefaultVoice
} from "./Library.ts"
import { Paths } from "./Paths.ts"
import { ReadingScript, type ReadingMatch } from "./Reading.ts"

export const PreviewText = "Here is something I never read aloud. Does this sound like me? " +
  "A familiar voice can tell a completely new story."

const root = Command.make("idiolect").pipe(
  Command.withSharedFlags({
    port: Flag.Int("port").pipe(
      Flag.withDescription(`Local engine port (default: ${DefaultPort})`),
      Flag.withDefault(DefaultPort)
    )
  }),
  Command.withDescription("Your words. Your voice.")
)

const onPort = <A, E, R>(port: number, body: Effect.Effect<A, E, R>) =>
  Effect.gen(function*() {
    if (port < 1 || port > 65535) return yield* fail("Choose a port between 1 and 65535.")
    return yield* Effect.provide(body, Engine.layer(port))
  })

/** Run a subcommand body with the engine on the chosen --port. */
const withEngine = <A, E, R>(body: Effect.Effect<A, E, R>) =>
  Effect.flatMap(root, ({ port }) => onPort(port, body))

const wrap = (text: string, width = 72) =>
  text.split(" ").reduce<Array<string>>((lines, word) => {
    const last = lines.at(-1)
    if (last !== undefined && last.length + word.length + 1 <= width) {
      lines[lines.length - 1] = `${last} ${word}`
    } else {
      lines.push(word)
    }
    return lines
  }, []).join("\n")

/**
 * Read one line in the terminal's normal (cooked) mode, so Ctrl+C stays an
 * ordinary SIGINT that interrupts the program, and the reader always closes.
 */
const ask = (prompt: string) =>
  Effect.callback<string>((resume) => {
    const lines = createInterface({ input: process.stdin, terminal: false })
    process.stdout.write(prompt)
    const ended = () => resume(Effect.interrupt)
    lines.once("close", ended)
    lines.once("line", (line) => {
      lines.off("close", ended)
      lines.close()
      resume(Effect.succeed(line))
    })
    return Effect.sync(() => {
      lines.off("close", ended)
      lines.close()
    })
  })

const enroll = Effect.fn("enroll")(function*(options: { readonly name: string; readonly device: Option.Option<string> }) {
  const name = options.name.trim()
  if (name.length < 1 || name.length > 60) return yield* fail("Use a voice name between 1 and 60 characters.")
  if (!process.stdin.isTTY) return yield* fail("Run idiolect setup in an interactive terminal to record your microphone.")
  const engine = yield* Engine
  const paths = yield* Paths
  const audio = yield* Audio
  yield* audio.check
  const status = yield* engine.ensure
  if (!status.recognizer_downloaded || !status.model_downloaded) {
    return yield* fail("Download the speech and reading models first with idiolect download.")
  }
  const device = Option.getOrUndefined(options.device)
  yield* Console.log(`\nRead this aloud in your natural voice:\n\n${wrap(ReadingScript)}\n`)
  yield* Console.log("Allow about 20–25 seconds at your usual pace, with natural pauses.")
  yield* Console.log("Use your everyday accent and speaking voice.")
  yield* Console.log("Follow the whole script. I will stop after recognizing the final words.")
  yield* Console.log(device ? `Microphone: ${device}` : "Using your default microphone. Use --device to choose another.")

  const progress = (match: ReadingMatch) =>
    Effect.sync(() => {
      const next = match.nextWords ? `  Next: "${match.nextWords}"` : "  Finishing…"
      process.stdout.write(`\r\x1b[KFollowing ${match.matched}/${match.total} words.${next}`)
    })

  let reading: Option.Option<{ wav: Uint8Array; transcript: string; duration: number }> = Option.none()
  while (Option.isNone(reading)) {
    yield* ask("\nPress Enter when you are ready to start. ")
    yield* Console.log("Reading along… (Ctrl+C to cancel)")
    const attemptReading = yield* recordReading({ script: ReadingScript, device, progress }).pipe(Effect.result)
    if (attemptReading._tag === "Success") {
      reading = Option.some(attemptReading.success)
    } else {
      yield* Console.log(`\n${attemptReading.failure.message}`)
      const answer = yield* ask("Press Enter to try again, or type q to quit: ")
      if (answer.trim().toLowerCase() === "q") return
    }
  }
  const { duration, transcript, wav } = reading.value
  yield* Console.log(`\nGot it. Complete script, ${duration}s.`)
  const form = new FormData()
  form.append("name", name)
  form.append("transcript", transcript)
  form.append("audio", new Blob([wav as Uint8Array<ArrayBuffer>], { type: "audio/wav" }), "reference.wav")
  const voice: Voice = yield* engine.call(engine.api.addVoice({ payload: form }))
  yield* attempt(() => setDefaultVoice(paths.data, voice))
  yield* Console.log(`\nSaved "${voice.name}" as your default voice.`)
  yield* Console.log("Now listen to a few new sentences in your cloned voice.")
  yield* renderVoice(voice, PreviewText).pipe(
    Effect.flatMap(playRender),
    Effect.catch((error: IdiolectError) =>
      Console.error(`Your voice is saved, but the preview could not finish: ${error.message}`)
    )
  )
  yield* Console.log("Try: idiolect say \"This is my voice, reading something new.\"")
})

const setupFlags = {
  name: Flag.String("name").pipe(Flag.withDescription("Name for this voice (default: Me)"), Flag.withDefault("Me")),
  device: Flag.String("device").pipe(
    Flag.withDescription("Microphone for SoX (sets AUDIODEV): a CoreAudio device name on macOS, a PulseAudio source or ALSA device on Linux"),
    Flag.optional
  )
}

const setup = Command.make("setup", setupFlags, (options) => withEngine(enroll(options))).pipe(
  Command.withDescription("Read a script, save your voice, and hear a preview")
)

const say = Command.make("say", {
  text: Argument.String("text").pipe(Argument.variadic()),
  file: Flag.String("file").pipe(Flag.withAlias("f"), Flag.withDescription("Read a script from a text file, or - for stdin"), Flag.optional),
  output: Flag.String("output").pipe(Flag.withAlias("o"), Flag.withDescription("Save a WAV instead of playing it aloud"), Flag.optional),
  voice: Flag.String("voice").pipe(Flag.withDescription("Saved voice name or ID (otherwise your default)"), Flag.optional),
  language: Flag.Literals("language", Languages).pipe(Flag.withDefault("English")),
  seed: Flag.Int("seed").pipe(Flag.withDefault(12)),
  force: Flag.Boolean("force").pipe(Flag.withDescription("Replace an existing output file"), Flag.withDefault(false))
}, (options) =>
  withEngine(Effect.gen(function*() {
    if (options.seed < 0 || options.seed > 2 ** 31 - 1) return yield* fail("Choose a seed between 0 and 2147483647.")
    const text = yield* attempt(() =>
      inputText({
        words: options.text,
        file: Option.getOrUndefined(options.file),
        stdinIsTTY: process.stdin.isTTY === true,
        readStdin: () => readFileSync(0, "utf8"),
        readFile: (path) => readFileSync(path, "utf8")
      })
    )
    const output = Option.map(options.output, (path) => resolve(path))
    // Check the output or the player before starting the engine or rendering.
    if (Option.isSome(output)) {
      yield* attempt(() => checkOutput(output.value, options.force))
    } else {
      yield* (yield* Audio).check
    }
    const engine = yield* Engine
    const paths = yield* Paths
    yield* engine.ensure
    const voices = yield* engine.call(engine.api.voices())
    const voice = yield* attempt(() => chooseVoice(voices, Option.getOrUndefined(options.voice), defaultVoice(paths.data)))
    const job = yield* renderVoice(voice, text, options.language, options.seed)
    if (Option.isSome(output)) {
      const bytes = yield* renderAudio(job)
      yield* attempt(() => publish(bytes, output.value, options.force))
      yield* Console.error(`Saved ${job.duration ?? 0}s of audio.`)
      yield* Console.log(output.value)
    } else {
      yield* playRender(job)
    }
  }))).pipe(Command.withDescription("Speak text aloud, or save a WAV with --output"))

const voices = Command.make("voices", {
  use: Flag.String("use").pipe(Flag.withDescription("Make a voice name or ID the default"), Flag.optional)
}, (options) =>
  withEngine(Effect.gen(function*() {
    const engine = yield* Engine
    const paths = yield* Paths
    yield* engine.ensure
    const list = yield* engine.call(engine.api.voices())
    if (Option.isSome(options.use)) {
      const chosen = yield* attempt(() => chooseVoice(list, options.use.pipe(Option.getOrUndefined), undefined))
      yield* attempt(() => setDefaultVoice(paths.data, chosen))
    }
    const selected = yield* attempt(() => defaultVoice(paths.data))
    for (const voice of list) {
      yield* Console.log(`${voice.id === selected ? "*" : " "} ${voice.id.slice(0, 8)}  ${voice.name}  (${voice.duration}s)`)
    }
    if (list.length === 0) yield* Console.log("No voices yet. Run idiolect setup.")
  }))).pipe(Command.withDescription("List saved voices"))

const status = Command.make("status", {}, () =>
  withEngine(Effect.gen(function*() {
    const engine = yield* Engine
    const paths = yield* Paths
    const running = yield* engine.probe
    yield* Console.log(Option.match(running, {
      onNone: () => "Engine stopped",
      onSome: (status) => `Engine ${status.model_status} · ${paths.data}`
    }))
  }))).pipe(Command.withDescription("Show the local engine status"))

const stop = Command.make("stop", {}, () =>
  withEngine(Effect.gen(function*() {
    const engine = yield* Engine
    yield* Console.log((yield* engine.stop) ? "Engine stopped." : "Engine already stopped.")
  }))).pipe(Command.withDescription("Stop the idle engine and release its GPU memory"))

/** Exit codes from foreground engine commands become this process's exit code. */
const exitWith = (code: number) => Effect.sync(() => {
  if (code !== 0) process.exitCode = code
})

const serve = Command.make("serve", {}, () =>
  withEngine(Effect.gen(function*() {
    const engine = yield* Engine
    yield* exitWith(yield* engine.foreground(["serve", "--port", String(engine.port)]))
  }))).pipe(Command.withDescription("Run the local speech engine in the foreground"))

const download = Command.make("download", {}, () =>
  withEngine(Effect.gen(function*() {
    const engine = yield* Engine
    yield* exitWith(yield* engine.foreground(["download"]))
  }))).pipe(Command.withDescription("Download speech and reading models for offline use"))

const doctor = Command.make("doctor", {}, () =>
  withEngine(Effect.gen(function*() {
    const engine = yield* Engine
    yield* exitWith(yield* engine.foreground(["doctor"]))
  }))).pipe(Command.withDescription("Check the GPU and speech models"))

export const idiolect = root.pipe(
  // With no subcommand, start guided setup.
  Command.withHandler(({ port }) => onPort(port, enroll({ name: "Me", device: Option.none() }))),
  Command.withSubcommands([setup, say, voices, status, stop, serve, download, doctor])
)
