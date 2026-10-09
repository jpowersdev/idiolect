import { expect, test } from "bun:test"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { migrateCheckout, resolvePaths } from "../src/Paths.ts"
import { temporaryDirectory } from "./support.ts"

test("data lives in each platform's per-user location unless overridden", () => {
  expect(resolvePaths({}, "darwin", "/Users/me")).toEqual({
    data: "/Users/me/Library/Application Support/idiolect",
    models: "/Users/me/Library/Application Support/idiolect/models"
  })
  expect(resolvePaths({}, "linux", "/home/me").data).toBe("/home/me/.local/share/idiolect")
  expect(resolvePaths({ XDG_DATA_HOME: "/xdg" }, "linux", "/home/me").data).toBe("/xdg/idiolect")
  expect(resolvePaths({ IDIOLECT_DATA: "/d", IDIOLECT_MODELS: "/m" }, "linux", "/home/me")).toEqual({
    data: "/d",
    models: "/m"
  })
})

test("an old checkout's voices and models move once, never over existing files", () => {
  const root = temporaryDirectory()
  const checkout = join(root, "checkout")
  mkdirSync(join(checkout, "data", "voices", "abc"), { recursive: true })
  writeFileSync(join(checkout, "data", "cli.json"), "old")
  mkdirSync(join(checkout, "models", "whisper-base.en"), { recursive: true })
  const paths = { data: join(root, "home"), models: join(root, "home", "models") }
  mkdirSync(paths.data)
  writeFileSync(join(paths.data, "cli.json"), "new")
  const moved = migrateCheckout(checkout, paths)
  expect(existsSync(join(paths.data, "voices", "abc"))).toBe(true)
  expect(existsSync(join(paths.models, "whisper-base.en"))).toBe(true)
  expect(readFileSync(join(paths.data, "cli.json"), "utf8")).toBe("new")
  expect(existsSync(join(checkout, "data", "cli.json"))).toBe(true)
  expect(moved.length).toBe(2)
  expect(migrateCheckout(checkout, paths)).toEqual([])
})
