/**
 * Microphone and speaker access through SoX, which uses each platform's default
 * device: CoreAudio on macOS, PulseAudio or ALSA on Linux. AUDIODEV selects
 * another device. Swap this layer for a native one without touching callers.
 */
import { Context, Effect, Layer, Scope, Stream } from "effect"
import { ChildProcess, ChildProcessSpawner } from "effect/process"
import { fail, IdiolectError } from "./Errors.ts"

export interface Recording {
  /** 16-bit little-endian mono PCM at the requested rate. Ends if the recorder stops. */
  readonly pcm: Stream.Stream<Uint8Array, IdiolectError>
  /** Whatever the recorder printed to stderr, for explaining failures. */
  readonly diagnostics: Effect.Effect<string>
}

export interface AudioShape {
  /** Record until the surrounding scope closes. */
  readonly record: (options: {
    readonly sampleRate: number
    readonly device: string | undefined
  }) => Effect.Effect<Recording, IdiolectError, Scope.Scope>
  /** Play a WAV file and wait for it to finish. */
  readonly play: (path: string) => Effect.Effect<void, IdiolectError>
  /** Fail early, before starting work, when no audio tool is available. */
  readonly check: Effect.Effect<void, IdiolectError>
}

export class Audio extends Context.Service<Audio, AudioShape>()("idiolect/Audio") {
  static readonly sox = Layer.effect(
    Audio,
    Effect.gen(function*() {
      const spawner = yield* ChildProcessSpawner.ChildProcessSpawner
      const executable = "sox"

      const check = Effect.suspend(() =>
        Bun.which(executable) === null
          ? Effect.fail(
            fail("Recording and playback need SoX. Install it (brew install sox, apt install sox), " +
              "or pass -o filename.wav to save audio instead.")
          )
          : Effect.void
      )

      const record: AudioShape["record"] = ({ device, sampleRate }) =>
        Effect.gen(function*() {
          yield* check
          const handle = yield* spawner.spawn(
            ChildProcess.make(executable, [
              "-q",
              "-d",
              "-t",
              "raw",
              "-r",
              String(sampleRate),
              "-c",
              "1",
              "-b",
              "16",
              "-e",
              "signed-integer",
              "-L",
              "-"
            ], {
              env: device ? { AUDIODEV: device } : {},
              extendEnv: true,
              stdin: "ignore",
              // SIGINT lets SoX close the device cleanly.
              killSignal: "SIGINT",
              forceKillAfter: "3 seconds"
            })
          ).pipe(Effect.mapError((error) => fail(`The microphone could not be opened: ${error.message}`)))
          const errors: Array<Uint8Array> = []
          yield* handle.stderr.pipe(
            Stream.runForEach((chunk) => Effect.sync(() => errors.push(chunk))),
            Effect.ignore,
            Effect.forkScoped
          )
          return {
            pcm: handle.stdout.pipe(
              Stream.mapError((error) => fail(`Microphone recording stopped: ${error.message}`))
            ),
            diagnostics: Effect.sync(() => Buffer.concat(errors).toString().trim())
          }
        })

      const play = (path: string) =>
        Effect.gen(function*() {
          yield* check
          const handle = yield* spawner.spawn(
            ChildProcess.make(executable, ["-q", path, "-d"], { stdin: "ignore", stdout: "ignore" })
          )
          const errors = yield* handle.stderr.pipe(Stream.decodeText(), Stream.mkString)
          const code = yield* handle.exitCode
          if (code !== 0) {
            return yield* fail(`Playback failed: ${errors.trim() || `sox exited with status ${code}`}`)
          }
        }).pipe(
          Effect.scoped,
          Effect.mapError((error) =>
            error instanceof IdiolectError ? error : fail(`Playback failed: ${error.message}`)
          )
        )

      return Audio.of({ record, play, check })
    })
  )
}
