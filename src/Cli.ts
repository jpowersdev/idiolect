/** Record your voice, then speak text in it. */
import { Console, Effect, Option } from "effect"
import { Argument, Command, Flag } from "effect/cli"
import { existsSync, readFileSync } from "node:fs"
import { createInterface } from "node:readline"
import { resolve } from "node:path"
import { Audio } from "./Audio.ts"
import { Engine } from "./Engine.ts"
import { recordReading } from "./Enroll.ts"
import { fail, IdiolectError } from "./Errors.ts"
import { install, locate, notInstalled, Version } from "./Install.ts"
import { attempt, checkOutput, inputText, playRender, publish, renderAudio, renderVoice } from "./Library.ts"
import { Paths } from "./Paths.ts"
import type { SavedReading } from "./Reading.ts"
import { type Script, Scripts } from "./Scripts.ts"

export const PreviewText = "Here is something I never read aloud. Does this sound like me? " +
  "A familiar voice can tell a completely new story."

/** Exit codes from foreground engine commands become this process's exit code. */
const exitWith = (code: number) => Effect.sync(() => {
  if (code !== 0) process.exitCode = code
})

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
 * Wait for a single key while recording. The terminal is in raw mode meanwhile, so Ctrl+C
 * arrives as a key too: pass it on as the usual interrupt.
 */
const keypress = (key: string) =>
  Effect.callback<void>((resume) => {
    const input = process.stdin
    const listener = (data: Buffer) => {
      const text = data.toString()
      if (text.includes("\x03")) process.kill(process.pid, "SIGINT")
      else if (text.toLowerCase().includes(key)) resume(Effect.void)
    }
    input.setRawMode(true)
    input.resume()
    input.on("data", listener)
    return Effect.sync(() => {
      input.off("data", listener)
      input.setRawMode(false)
      input.pause()
    })
  })

/** A yes-or-no question; Enter takes the default. */
const confirm = (question: string, preferred = true) =>
  Effect.map(ask(`${question} ${preferred ? "[Y/n]" : "[y/N]"} `), (answer) => {
    const reply = answer.trim().toLowerCase()
    return reply === "" ? preferred : reply === "y" || reply === "yes"
  })

const bold = (text: string) => `\x1b[1m${text}\x1b[22m`
const dim = (text: string) => `\x1b[2m${text}\x1b[22m`

/** One script on a fresh screen, wrapped to read comfortably in any terminal. */
const page = (script: Script, index: number) => {
  const width = Math.min(72, (process.stdout.columns || 80) - 2)
  const notice = script.notice ? `\n\n${dim(wrap(script.notice, width))}` : ""
  return `\x1b[2J\x1b[H${bold(`${index + 1} of ${Scripts.length} · ${script.title}`)}\n\n` +
    `${wrap(script.text, width)}${notice}\n`
}

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

const enroll = Effect.fn("enroll")(function*(options: { readonly device: Option.Option<string> }) {
  if (!process.stdin.isTTY) return yield* fail("Run idiolect setup in an interactive terminal to record your microphone.")
  const engine = yield* Engine
  const audio = yield* Audio
  const paths = yield* Paths
  yield* audio.check

  // Install only what is missing, and only once you agree.
  const location = locate(paths)
  if (location._tag === "Missing") return yield* notInstalled
  if (location._tag === "Managed" && !location.current) {
    const updating = existsSync(location.command)
    const agreed = yield* confirm(
      updating
        ? `\nThe speech engine needs updating for Idiolect ${Version}. Update it now?`
        : `\nIdiolect needs to install its speech engine (Python and PyTorch, about 1 GB) in ${paths.data}. Install it now?`
    )
    if (!agreed) return yield* Console.log("Nothing installed. Run idiolect setup when you are ready.")
    yield* install(paths, updating)
  }
  const status = yield* engine.ensure
  if (!status.recognizer_downloaded || !status.model_downloaded) {
    const agreed = yield* confirm(
      `\nIdiolect needs to download its speech and reading models (about 5 GB) to ${paths.data}. Download them now?`
    )
    if (!agreed) return yield* Console.log("Nothing downloaded. Run idiolect setup when you are ready.")
    const code = yield* engine.foreground(["download"])
    if (code !== 0) return yield* fail("The models could not be downloaded. Run idiolect setup to try again.")
  }
  const previous = yield* engine.call(engine.api.voice()).pipe(Effect.option)
  if (Option.isSome(previous) && !(yield* confirm("\nYou already have a voice. Record a new one?", false))) {
    return yield* Console.log("Everything is ready. Try: idiolect say \"This is my voice, reading something new.\"")
  }
  const device = Option.getOrUndefined(options.device)
  yield* Console.log(`\nYou will read ${Scripts.length} short scripts aloud, about four minutes in all.`)
  yield* Console.log("Use your everyday accent and speaking voice, at your usual pace, with natural pauses.")
  yield* Console.log("Let each script carry you: tell the story, explain the steps, chat with a friend.")
  yield* Console.log("Each recording stops by itself shortly after you read the script's final words.")
  yield* Console.log(device ? `Microphone: ${device}` : "Using your default microphone. Use --device to choose another.")
  if (Option.isSome(previous)) yield* Console.log("Your current voice stays until you finish every script.")

  const progress = ({ heard, seconds }: { readonly heard: string; readonly seconds: number }) =>
    Effect.sync(() => {
      // One line, rewritten in place; a wrapped line would leave copies behind.
      const width = (process.stdout.columns || 80) - 1
      const prefix = `${Math.floor(seconds)}s  `
      const recent = heard.length > width - prefix.length ? `…${heard.slice(-(width - prefix.length - 1))}` : heard
      process.stdout.write(`\r\x1b[K${prefix}${recent}`)
    })

  const heardEnd = Console.log("\nGot it.")

  const takes: Array<{ readonly script: Script; readonly reading: SavedReading }> = []
  yield* ask("\nPress Enter to see the first script. ")
  for (const [index, script] of Scripts.entries()) {
    let reading: Option.Option<SavedReading> = Option.none()
    let restarting = false
    while (Option.isNone(reading)) {
      yield* Console.log(page(script, index))
      // Restarting goes straight back to listening; otherwise wait until the reader is ready.
      if (!restarting) yield* ask("Press Enter when you are ready to start. ")
      yield* Console.log("Listening… (r to restart this script, Ctrl+C to cancel)")
      const attempt = yield* Effect.raceFirst(
        recordReading({ script: script.text, device, progress, heardEnd }).pipe(Effect.result),
        keypress("r").pipe(Effect.as("restart" as const))
      )
      restarting = attempt === "restart"
      if (attempt === "restart") continue
      if (attempt._tag === "Success") {
        reading = Option.some(attempt.success)
      } else {
        yield* Console.log(`\n${attempt.failure.message}`)
        const answer = yield* ask("Press Enter to try again, or type q to quit: ")
        if (answer.trim().toLowerCase() === "q") return
      }
    }
    yield* Console.log(`${reading.value.duration}s.`)
    takes.push({ script, reading: reading.value })
  }

  const form = new FormData()
  form.append("takes", JSON.stringify(takes.map(({ reading, script }) => ({
    script: script.id,
    transcript: reading.transcript,
    words: reading.words,
    ...(script.notice ? { notice: script.notice } : {})
  }))))
  for (const { reading, script } of takes) {
    form.append("audio", new Blob([reading.wav as Uint8Array<ArrayBuffer>], { type: "audio/wav" }), `${script.id}.wav`)
  }
  yield* engine.call(engine.api.replaceVoice({ payload: form }))
  yield* Console.log("\nSaved your voice. Now listen to a few new sentences in it.")
  yield* renderVoice(PreviewText).pipe(
    Effect.flatMap(playRender),
    Effect.catch((error: IdiolectError) =>
      Console.error(`Your voice is saved, but the preview could not finish: ${error.message}`)
    )
  )
  yield* Console.log("Try: idiolect say \"This is my voice, reading something new.\"")
})

const setup = Command.make("setup", {
  device: Flag.String("device").pipe(
    Flag.withDescription("Microphone for SoX (sets AUDIODEV): a CoreAudio device name on macOS, a PulseAudio source or ALSA device on Linux"),
    Flag.optional
  )
}, (options) => enroll(options)).pipe(
  Command.withDescription("Record your voice by reading a few short scripts aloud")
)

const say = Command.make("say", {
  text: Argument.String("text").pipe(Argument.withDescription("Words to speak, or pipe them in"), Argument.variadic()),
  output: Flag.String("output").pipe(Flag.withAlias("o"), Flag.withDescription("Save a WAV instead of playing it aloud"), Flag.optional)
}, (options) =>
  Effect.gen(function*() {
    const text = yield* attempt(() =>
      inputText({
        words: options.text,
        stdinIsTTY: process.stdin.isTTY === true,
        readStdin: () => readFileSync(0, "utf8")
      })
    )
    const output = Option.map(options.output, (path) => resolve(path))
    // Check the output or the player before starting the engine or rendering.
    if (Option.isSome(output)) {
      yield* attempt(() => checkOutput(output.value))
    } else {
      yield* (yield* Audio).check
    }
    yield* (yield* Engine).ensure
    const job = yield* renderVoice(text)
    if (Option.isSome(output)) {
      const bytes = yield* renderAudio(job)
      yield* attempt(() => publish(bytes, output.value))
      yield* Console.error(`Saved ${job.duration ?? 0}s of audio.`)
      yield* Console.log(output.value)
    } else {
      yield* playRender(job)
    }
  })).pipe(Command.withDescription("Speak text in your voice, or save it with --output"))

const doctor = Command.make("doctor", {}, () =>
  Effect.gen(function*() {
    const sox = yield* (yield* Audio).check.pipe(Effect.result)
    yield* Console.log(sox._tag === "Success" ? "SoX: ready" : `SoX: ${sox.failure.message}`)
    // Report the engine without installing it; setup does that.
    const location = locate(yield* Paths)
    if (location._tag === "Missing") {
      yield* Console.log(`Engine: ${notInstalled.message}`)
      return yield* exitWith(1)
    }
    if (location._tag === "OnPath") {
      yield* Console.log(`Engine: ${location.command}`)
    } else {
      yield* Console.log(Bun.which("uv") === null ? "uv: missing (brew install uv)" : "uv: ready")
      if (!location.current) {
        yield* Console.log(
          existsSync(location.command)
            ? `Engine: needs updating for ${Version} (idiolect setup updates it)`
            : "Engine: not installed yet (idiolect setup installs it)"
        )
        return yield* exitWith(1)
      }
      yield* Console.log(`Engine: ${Version}`)
    }
    const code = yield* (yield* Engine).foreground(["doctor"])
    yield* exitWith(sox._tag === "Success" ? code : 1)
  })).pipe(Command.withDescription("Check the GPU, models, engine, and audio tools"))

export const idiolect = Command.make("idiolect").pipe(
  Command.withDescription("Local voice cloning from the terminal"),
  Command.withSubcommands([setup, say, doctor])
)
