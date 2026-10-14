/**
 * Where the speech engine comes from. Nix and the development shell put `idiolect-engine` on
 * PATH. Otherwise the npm package carries the engine, built for exactly this version with the
 * lockfile's pinned dependencies, and Idiolect installs it with uv into its own folder.
 */
import { Console, Effect } from "effect"
import { ChildProcess, ChildProcessSpawner } from "effect/process"
import { existsSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { join, resolve } from "node:path"
import { fail, IdiolectError } from "./Errors.ts"
import type { PathsShape } from "./Paths.ts"
import packageJson from "../package.json" with { type: "json" }

export const EngineCommand = "idiolect-engine"
export const Version = packageJson.version

/** Built by `bun run bundle-engine` and published with the CLI; absent in a checkout. */
export const BundleDirectory = resolve(import.meta.dir, "..", "dist", "engine")

export type Location =
  /** On PATH, from Nix or the development shell. */
  | { readonly _tag: "OnPath"; readonly command: string }
  /** Installed by Idiolect, and current when it matches this version. */
  | { readonly _tag: "Managed"; readonly command: string; readonly current: boolean }
  | { readonly _tag: "Missing" }

export const managedDirectory = (paths: PathsShape) => join(paths.data, "engine")
const versionFile = (directory: string) => join(directory, "idiolect-version")

export const locate = (
  paths: PathsShape,
  options: { readonly which?: (command: string) => string | null; readonly bundle?: string } = {}
): Location => {
  const onPath = (options.which ?? Bun.which)(EngineCommand)
  if (onPath !== null) return { _tag: "OnPath", command: onPath }
  if (!existsSync(options.bundle ?? BundleDirectory)) return { _tag: "Missing" }
  const directory = managedDirectory(paths)
  const command = join(directory, "bin", EngineCommand)
  const installed = existsSync(command) && existsSync(versionFile(directory))
    ? readFileSync(versionFile(directory), "utf8").trim()
    : undefined
  return { _tag: "Managed", command, current: installed === Version }
}

export const notInstalled = fail(
  `The speech engine (${EngineCommand}) is missing. Install Idiolect with ` +
    "`bun add -g @jpowersdev/idiolect` or Nix, or in a checkout run `uv sync` in engine/."
)

/** Install or update the bundled engine in Idiolect's own folder, with uv. Setup asks first. */
export const install = (paths: PathsShape, updating: boolean) =>
  Effect.gen(function*() {
    const uv = Bun.which("uv")
    if (uv === null) {
      return yield* fail("Idiolect installs its speech engine with uv. Install it (brew install uv), then try again.")
    }
    const wheel = readdirSync(BundleDirectory).find((name) => name.endsWith(".whl"))
    if (wheel === undefined) return yield* fail(`The bundled engine is missing from ${BundleDirectory}.`)
    const directory = managedDirectory(paths)
    const python = join(directory, "bin", "python")
    yield* Console.error(updating ? `Updating the speech engine to ${Version}…` : "Installing the speech engine…")
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner
    const run = (args: ReadonlyArray<string>) =>
      Effect.gen(function*() {
        const handle = yield* spawner.spawn(
          ChildProcess.make(uv, args, { stdin: "ignore", stdout: "inherit", stderr: "inherit" })
        )
        const code = Number(yield* handle.exitCode)
        if (code !== 0) return yield* fail(`Installing the speech engine failed (uv ${args[0]} ${args[1] ?? ""}).`)
      }).pipe(Effect.scoped)
    // Forget the old version first, so an interrupted update is retried rather than trusted.
    rmSync(versionFile(directory), { force: true })
    yield* run(["venv", "--quiet", "--allow-existing", "--python", "3.12", directory])
    yield* run(["pip", "sync", "--quiet", "--python", python, join(BundleDirectory, "requirements.txt")])
    yield* run(["pip", "install", "--quiet", "--no-deps", "--reinstall", "--python", python, join(BundleDirectory, wheel)])
    writeFileSync(versionFile(directory), `${Version}\n`)
    yield* Console.error("Speech engine ready.")
  }).pipe(
    Effect.mapError((error) =>
      error instanceof IdiolectError ? error : fail(`Installing the speech engine failed: ${error.message}`)
    )
  )

/** The engine's executable. Only setup installs or updates it. */
export const engineExecutable = (paths: PathsShape) =>
  Effect.gen(function*() {
    const location = locate(paths)
    if (location._tag === "Missing") return yield* notInstalled
    if (location._tag === "Managed" && !location.current) {
      return yield* fail(
        existsSync(location.command)
          ? "The speech engine needs updating for this version of Idiolect. Run idiolect setup."
          : "The speech engine is not installed yet. Run idiolect setup."
      )
    }
    return location.command
  })
