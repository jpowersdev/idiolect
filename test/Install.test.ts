import { expect, test } from "bun:test"
import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join, resolve } from "node:path"
import { locate, managedDirectory, Version } from "../src/Install.ts"
import { temporaryDirectory } from "./support.ts"

const nowhere = () => null

test("an engine on PATH, from Nix or the development shell, is used as it is", () => {
  const paths = { data: temporaryDirectory() }
  expect(locate(paths, { which: () => "/nix/store/x/bin/idiolect-engine" }))
    .toEqual({ _tag: "OnPath", command: "/nix/store/x/bin/idiolect-engine" })
})

test("without one, the bundled engine is installed, and reinstalled for a new version", () => {
  const paths = { data: temporaryDirectory() }
  const bundle = temporaryDirectory()
  const command = join(managedDirectory(paths), "bin", "idiolect-engine")
  expect(locate(paths, { which: nowhere, bundle })).toEqual({ _tag: "Managed", command, current: false })
  mkdirSync(join(managedDirectory(paths), "bin"), { recursive: true })
  writeFileSync(command, "")
  writeFileSync(join(managedDirectory(paths), "idiolect-version"), "0.0.1\n")
  expect(locate(paths, { which: nowhere, bundle })).toMatchObject({ current: false })
  writeFileSync(join(managedDirectory(paths), "idiolect-version"), `${Version}\n`)
  expect(locate(paths, { which: nowhere, bundle })).toMatchObject({ current: true })
})

test("in a checkout with neither, the engine is missing", () => {
  expect(locate({ data: temporaryDirectory() }, { which: nowhere, bundle: "/no/such/bundle" }))
    .toEqual({ _tag: "Missing" })
})

test("the CLI and the engine are released together, at one version", () => {
  const pyproject = readFileSync(resolve(import.meta.dir, "..", "engine", "pyproject.toml"), "utf8")
  expect(pyproject.match(/^version = "(.+)"$/m)?.[1]).toBe(Version)
})
