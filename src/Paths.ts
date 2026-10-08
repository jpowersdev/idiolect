import { Context, Effect, Layer } from "effect"
import { existsSync, mkdirSync, readdirSync, realpathSync, renameSync } from "node:fs"
import { homedir } from "node:os"
import { join, resolve } from "node:path"

export interface PathsShape {
  /** Voices, renders, the default-voice setting, and the engine log. */
  readonly data: string
  /** Speech and reading model weights. */
  readonly models: string
}

type Env = Readonly<Record<string, string | undefined>>

/** Standard per-user data locations, overridable with IDIOLECT_DATA and IDIOLECT_MODELS. */
export const resolvePaths = (env: Env, platform: string, home: string): PathsShape => {
  const base = platform === "darwin"
    ? join(home, "Library", "Application Support", "idiolect")
    : join(env.XDG_DATA_HOME || join(home, ".local", "share"), "idiolect")
  const data = resolve(env.IDIOLECT_DATA || base)
  return { data, models: resolve(env.IDIOLECT_MODELS || join(data, "models")) }
}

/**
 * Earlier versions kept data/ and models/ inside the source checkout. Move them
 * into place once, only when the new location does not exist yet.
 */
export const migrateCheckout = (checkout: string, paths: PathsShape): ReadonlyArray<string> => {
  const moved: Array<string> = []
  const move = (from: string, to: string) => {
    if (!existsSync(from) || existsSync(to)) return
    mkdirSync(resolve(to, ".."), { recursive: true })
    renameSync(from, to)
    moved.push(`${from} -> ${to}`)
  }
  const legacyData = join(checkout, "data")
  if (existsSync(legacyData) && !existsSync(paths.data)) {
    move(legacyData, paths.data)
  } else if (existsSync(legacyData)) {
    for (const entry of readdirSync(legacyData)) move(join(legacyData, entry), join(paths.data, entry))
  }
  const legacyModels = join(checkout, "models")
  if (existsSync(legacyModels)) {
    for (const entry of readdirSync(legacyModels)) {
      move(join(legacyModels, entry), join(paths.models, entry))
    }
  }
  return moved
}

export class Paths extends Context.Service<Paths, PathsShape>()("idiolect/Paths") {
  static readonly layer = Layer.effect(
    Paths,
    Effect.sync(() => {
      const paths = resolvePaths(process.env, process.platform, homedir())
      // The engine reports its data directory with symlinks resolved; compare like with like.
      mkdirSync(paths.data, { recursive: true })
      return Paths.of({ ...paths, data: realpathSync(paths.data) })
    })
  )
}
