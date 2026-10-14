import { expect, test } from "bun:test"
import { requestedModel, usesWantedModel } from "../src/Engine.ts"

test("a running engine is kept only when it uses the model wanted", () => {
  const auto = { model: "voxcpm2", model_requested: false }
  const forced = { model: "qwen3", model_requested: true }
  expect(usesWantedModel(auto, undefined)).toBe(true)
  expect(usesWantedModel(forced, undefined)).toBe(false)
  expect(usesWantedModel(forced, "qwen3")).toBe(true)
  expect(usesWantedModel(auto, "qwen3")).toBe(false)
  expect(requestedModel({ IDIOLECT_MODEL: " Qwen3 " })).toBe("qwen3")
  expect(requestedModel({ IDIOLECT_MODEL: "" })).toBeUndefined()
})
