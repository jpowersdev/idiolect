import { expect, test } from "bun:test"
import { resolvePaths } from "../src/Paths.ts"

test("data lives in ~/.idiolect unless overridden", () => {
  expect(resolvePaths({}, "/Users/me").data).toBe("/Users/me/.idiolect")
  expect(resolvePaths({ IDIOLECT_DATA: "/d" }, "/Users/me").data).toBe("/d")
})
