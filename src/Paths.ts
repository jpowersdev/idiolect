import { Context, Effect, Layer } from "effect"
import { mkdirSync, realpathSync } from "node:fs"
import { homedir } from "node:os"
import { join, resolve } from "node:path"

export interface PathsShape {
  /** Your voice, the models, the engine's socket, and its log. */
  readonly data: string
}

type Env = Readonly<Record<string, string | undefined>>

/** ~/.idiolect, or IDIOLECT_DATA. */
export const resolvePaths = (env: Env, home: string): PathsShape => ({
  data: resolve(env.IDIOLECT_DATA || join(home, ".idiolect"))
})

export class Paths extends Context.Service<Paths, PathsShape>()("idiolect/Paths") {
  static readonly layer = Layer.effect(
    Paths,
    Effect.sync(() => {
      const paths = resolvePaths(process.env, homedir())
      mkdirSync(paths.data, { recursive: true })
      return Paths.of({ data: realpathSync(paths.data) })
    })
  )
}
