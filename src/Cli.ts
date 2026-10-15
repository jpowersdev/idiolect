/** Record your voice, then speak text in it. */
import { Console, Effect, Option } from "effect"
import { Argument, Command, Flag } from "effect/cli"
import { existsSync, readFileSync } from "node:fs"
import { totalmem } from "node:os"
import { createInterface } from "node:readline"
import { join, resolve } from "node:path"
import { Audio } from "./Audio.ts"
import { Engine } from "./Engine.ts"
import { recordReading } from "./Enroll.ts"
import { fail, IdiolectError } from "./Errors.ts"
import { attempt, checkOutput, inputText, playRender, publish, renderAudio, renderVoice } from "./Library.ts"
import { download, downloaded, gigabytes, missingBytes, Models } from "./Models.ts"
import { Paths } from "./Paths.ts"
import type { SavedReading } from "./Reading.ts"
import { type Script, Scripts } from "./Scripts.ts"

export const PreviewText = "Here is something I never read aloud. Does this sound like me? " +
  "A familiar voice can tell a completely new story."

/** Idiolect runs its models with effect-torch's Metal backend, so on Apple silicon for now. */
const supportedHardware = process.platform === "darwin" && process.arch === "arm64"
/** VoxCPM2 takes about 5 GB in bfloat16; the rest is for everything else on the machine. */
const RecommendedMemory = 16 * 2 ** 30

/** Download whatever models are missing, showing each file's progress on one line. */
const downloadModels = (data: string) =>
  Effect.gen(function*() {
    for (const model of Models) {
      let shown = ""
      yield* download(data, model, ({ file, received }) => {
        const line = `Downloading ${model.name}: ${file.name} ${Math.floor((received / file.size) * 100)}% of ${gigabytes(file.size)}`
        if (line !== shown) process.stderr.write(`\r\x1b[K${line}`)
        shown = line
      })
      if (shown) process.stderr.write("\n")
    }
    yield* Console.error("Models ready.")
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
  if (!supportedHardware) return yield* fail("Idiolect needs an Apple silicon Mac for now.")

  // Download only what is missing, and only once you agree.
  const missing = missingBytes(paths.data)
  if (missing > 0) {
    const agreed = yield* confirm(
      `\nIdiolect needs to download its speech and reading models (${gigabytes(missing)}) to ${join(paths.data, "models")}. Download them now?`
    )
    if (!agreed) return yield* Console.log("Nothing downloaded. Run idiolect setup when you are ready.")
    yield* downloadModels(paths.data)
  }
  yield* engine.ensure
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
    // Report only; setup downloads what is missing.
    const { data } = yield* Paths
    let ready = true
    const report = (name: string, ok: boolean, detail: string) => {
      if (!ok) ready = false
      return Console.log(`${name}: ${detail}`)
    }
    const sox = yield* (yield* Audio).check.pipe(Effect.result)
    yield* report("SoX", sox._tag === "Success", sox._tag === "Success" ? "ready" : sox.failure.message)
    yield* report("GPU", supportedHardware, supportedHardware ? "Apple silicon (Metal)" : "Idiolect needs an Apple silicon Mac for now")
    const backend = yield* Effect.tryPromise(() => import("@effect-torch/backend-apple-native")).pipe(Effect.result)
    yield* report(
      "effect-torch",
      backend._tag === "Success",
      backend._tag === "Success" ? "ready" : `the Metal backend did not load (in a checkout, run bun run build-native): ${backend.failure.message}`
    )
    const memory = totalmem()
    yield* Console.log(`Memory: ${Math.round(memory / 2 ** 30)} GB${memory < RecommendedMemory ? " (16 GB or more recommended; speaking may be slow)" : ""}`)
    for (const model of Models) {
      const present = downloaded(data, model)
      yield* report(model.name, present, present ? "ready" : "not downloaded (idiolect setup downloads it)")
    }
    const voice = existsSync(join(data, "voice", "voice.json"))
    yield* Console.log(`Voice: ${voice ? "recorded" : "not recorded yet (idiolect setup records it)"}`)
    yield* Console.log(`Data: ${data}`)
    if (!ready) process.exitCode = 1
  })).pipe(Command.withDescription("Check the GPU, models, and audio tools"))

export const idiolect = Command.make("idiolect").pipe(
  Command.withDescription("Local voice cloning from the terminal"),
  Command.withSubcommands([setup, say, doctor])
)
