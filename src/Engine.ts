/**
 * The speech engine, in this process: transcription for setup, your saved voice, and speaking.
 * The models load the first time a command needs them, and effect-torch's native backend with
 * them, so commands that never speak or listen never load it.
 */
import { Context, Effect, Layer } from "effect"
import { existsSync } from "node:fs"
import { join } from "node:path"
import { render, type Spoken, type Voices } from "./engine/Render.ts"
import * as Voice from "./engine/Voice.ts"
import { fail, type IdiolectError } from "./Errors.ts"
import { Paths } from "./Paths.ts"
import type { Word } from "./Reading.ts"

export type { Spoken }
export type Take = Voice.Take

export interface EngineShape {
  /** Recognize 16-bit mono PCM, with each word's timing. */
  readonly transcribe: (pcm: Int16Array, rate: number) => Effect.Effect<ReadonlyArray<Word>, IdiolectError>
  readonly hasVoice: Effect.Effect<boolean>
  /** Replace your voice with a complete set of takes. */
  readonly saveVoice: (takes: ReadonlyArray<Take>) => Effect.Effect<void, IdiolectError>
  /** Speak a script in your voice, reporting each step. Interrupting it stops at once. */
  readonly speak: (text: string, progress: (message: string) => Effect.Effect<void>) => Effect.Effect<Spoken, IdiolectError>
}

const explain = (error: unknown) => fail(error instanceof Error ? error.message : String(error))

export class Engine extends Context.Service<Engine, EngineShape>()("idiolect/Engine") {
  static readonly layer = Layer.effect(
    Engine,
    Effect.gen(function*() {
      const { data } = yield* Paths
      const scope = yield* Effect.scope
      // Loaded on first use; the Metal runtime then lasts as long as the command.
      const voices = yield* Effect.cached(
        Effect.gen(function*() {
          const [BackendApple, Local] = yield* Effect.tryPromise({
            try: () => Promise.all([import("@effect-torch/backend-apple-native"), import("./engine/Local.ts")]),
            catch: (error) => fail(`The speech engine could not load: ${error instanceof Error ? error.message : String(error)}`)
          })
          const runtime = yield* Layer.buildWithScope(BackendApple.layer(), scope)
          return yield* Effect.provideContext(Local.make(data), runtime)
        })
      )
      const withVoices = <A>(use: (voices: Voices) => Effect.Effect<A, Error>) =>
        Effect.flatMap(voices, (loaded) => Effect.mapError(use(loaded), explain))

      return Engine.of({
        transcribe: (pcm, rate) => withVoices((loaded) => loaded.transcribe(Float32Array.from(pcm, (x) => x / 32768), rate)),
        hasVoice: Effect.sync(() => existsSync(join(Voice.voiceFolder(data), "voice.json"))),
        saveVoice: (takes) => Effect.asVoid(Effect.mapError(Voice.replace(data, takes), explain)),
        speak: (text, progress) => withVoices((loaded) => render(loaded, text, progress))
      })
    })
  )
}
