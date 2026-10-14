import { Effect, Layer, Option } from "effect"
import { mkdtempSync } from "node:fs"
import { realpathSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { Render, Word } from "../src/Api.ts"
import { Audio, type AudioShape } from "../src/Audio.ts"
import { Engine, type EngineShape, explain } from "../src/Engine.ts"
import { Paths } from "../src/Paths.ts"

export const temporaryDirectory = () => realpathSync(mkdtempSync(join(tmpdir(), "idiolect-test-")))

export const render = (changes: Partial<Render> = {}): Render => ({
  id: "take123",
  completed: 0,
  total: 1,
  status: "queued",
  message: "Waiting for the speech engine",
  duration: null,
  ...changes
})

/** Words spoken half a second apart, starting at `start` seconds. */
export const words = (text: string, start = 1): Array<Word> =>
  text.split(" ").map((word, index) => ({ text: word, start: start + index * 0.5, end: start + (index + 1) * 0.5 }))

/** An engine whose API methods are the given stand-ins. */
export const fakeEngine = (api: Partial<Record<keyof EngineShape["api"], (...args: Array<any>) => Effect.Effect<any, any>>>) =>
  Layer.succeed(
    Engine,
    Engine.of({
      api: api as unknown as EngineShape["api"],
      call: (effect) => Effect.mapError(effect, explain),
      probe: Effect.succeed(Option.none()),
      ensure: Effect.die("not used"),
      foreground: () => Effect.succeed(0)
    })
  )

export const fakeAudio = (audio: Partial<AudioShape>) =>
  Layer.succeed(Audio, Audio.of({ check: Effect.void, play: () => Effect.void, record: () => Effect.die("not used"), ...audio }))

export const fakePaths = (data: string) => Layer.succeed(Paths, Paths.of({ data }))
