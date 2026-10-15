import { Effect, Layer } from "effect"
import { mkdtempSync } from "node:fs"
import { realpathSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Audio, type AudioShape } from "../src/Audio.ts"
import { Engine, type EngineShape } from "../src/Engine.ts"
import { Paths } from "../src/Paths.ts"
import type { Word } from "../src/Reading.ts"

export const temporaryDirectory = () => realpathSync(mkdtempSync(join(tmpdir(), "idiolect-test-")))

/** Words spoken half a second apart, starting at `start` seconds. */
export const words = (text: string, start = 1): Array<Word> =>
  text.split(" ").map((word, index) => ({ text: word, start: start + index * 0.5, end: start + (index + 1) * 0.5 }))

/** An engine made of the given stand-ins. */
export const fakeEngine = (engine: Partial<EngineShape>) =>
  Layer.succeed(
    Engine,
    Engine.of({
      transcribe: () => Effect.die("not used"),
      hasVoice: Effect.succeed(false),
      saveVoice: () => Effect.die("not used"),
      speak: () => Effect.die("not used"),
      ...engine
    })
  )

export const fakeAudio = (audio: Partial<AudioShape>) =>
  Layer.succeed(Audio, Audio.of({ check: Effect.void, play: () => Effect.void, record: () => Effect.die("not used"), ...audio }))

export const fakePaths = (data: string) => Layer.succeed(Paths, Paths.of({ data }))
