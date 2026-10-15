/**
 * The effect-torch VoxCPM2 pipeline against one generation captured from PyTorch, with the same
 * noise. Needs the model and the capture, so it runs only with IDIOLECT_MODEL_TESTS=1.
 */
import { expect, test } from "bun:test"
import * as BackendApple from "@effect-torch/backend-apple-native"
import { Safetensors } from "@effect-torch/core"
import { Effect } from "effect"
import { existsSync, readFileSync } from "node:fs"
import { generate, load } from "../../src/engine/VoxCpm.ts"

const home = process.env.HOME!
const model = `${home}/.idiolect/models/voxcpm2`
const capture = `${home}/.idiolect/experiments/voxcpm-reference/generation`
const enabled = process.env.IDIOLECT_MODEL_TESTS === "1" && existsSync(`${capture}/meta.json`)
const read = (name: string) => {
  const bytes = readFileSync(`${capture}/${name}.f32`)
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
}
/** PyTorch's noise and patches are [64, 4] channel-major; the pipeline's are [4, 64]. */
const timeMajor = (a: Float32Array) => Float32Array.from({ length: 256 }, (_, i) => a[(i % 64) * 4 + Math.floor(i / 64)]!)
const maxError = (a: Float32Array, b: Float32Array) => a.reduce((m, x, i) => Math.max(m, Math.abs(x - b[i]!)), 0)

test.skipIf(!enabled)("generation matches PyTorch patch for patch, and stops at the same step", async () => {
  const meta = JSON.parse(readFileSync(`${capture}/meta.json`, "utf8"))
  const config = JSON.parse(readFileSync(`${model}/config.json`, "utf8"))
  const started = performance.now()
  const patches = await Effect.runPromise(Effect.gen(function*() {
    const voxcpm = yield* load(yield* Safetensors.load(`${model}/model.safetensors`), {
      dtype: "f32",
      rope: { theta: config.lm_config.rope_theta, factors: config.lm_config.rope_scaling.short_factor }
    })
    const loaded = performance.now()
    const result = yield* generate(voxcpm, {
      text: Uint32Array.from(meta.text),
      textMask: read("text_mask"),
      feat: read("feat"),
      featMask: read("feat_mask")
    }, {
      minLength: meta.options.min_len,
      maxLength: meta.options.max_len,
      seed: 0,
      noise: (i) => timeMajor(read(`step${String(i).padStart(3, "0")}_noise`))
    })
    console.log(`loaded and compiled in ${((loaded - started) / 1000).toFixed(1)} s; generated ${result.length} patches in ${((performance.now() - loaded) / 1000).toFixed(1)} s`)
    return result
  }).pipe(Effect.provide(BackendApple.layer())))
  expect(patches.length).toBe(meta.steps)
  const errors = patches.map((patch, i) => maxError(patch, timeMajor(read(`step${String(i).padStart(3, "0")}_patch`))))
  console.log(`patch errors: ${errors.map((e) => e.toExponential(1)).join(" ")}`)
  // Rounding differences compound as each patch feeds the next; patches range to about ±5.
  expect(Math.max(...errors)).toBeLessThan(0.01)
}, 600_000)
