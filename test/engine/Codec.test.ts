/**
 * The codec, with weights read from audiovae.pth, against PyTorch's encoding and decoding of the
 * same audio. Needs the model and the capture, so it runs only with IDIOLECT_MODEL_TESTS=1.
 */
import { expect, test } from "bun:test"
import * as BackendApple from "@effect-torch/backend-apple-native"
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import { existsSync, readFileSync } from "node:fs"
import * as Codec from "../../src/engine/Codec.ts"
import { foldWeightNorm, readStateDict } from "../../src/engine/Pth.ts"

const home = process.env.HOME!
const model = `${home}/.idiolect/models/voxcpm2/audiovae.pth`
const capture = `${home}/.idiolect/experiments/voxcpm-reference`
const enabled = process.env.IDIOLECT_MODEL_TESTS === "1" && existsSync(`${capture}/vae-latents.f32`)
const read = (name: string) => {
  const bytes = readFileSync(`${capture}/${name}.f32`)
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
}
const maxError = (a: Float32Array, b: Float32Array) => a.reduce((m, x, i) => Math.max(m, Math.abs(x - b[i]!)), 0)

test.skipIf(!enabled)("encoding and decoding match PyTorch", async () => {
  const audio = read("vae-audio-16k")
  const latents = read("vae-latents")
  const [encoded, decoded] = await Effect.runPromise(Effect.gen(function*() {
    const stored: Record<string, Tensor.Any> = {}
    for (const [name, { data, shape }] of foldWeightNorm(readStateDict(readFileSync(model)))) {
      if (data instanceof Float32Array) stored[name] = yield* Tensor.fromTypedArray(data, shape)
    }
    const weights = yield* Codec.prepare(stored)
    const outputs = yield* Tensor.compute([
      yield* Codec.encode(weights, yield* Tensor.fromTypedArray(audio, [1, 1, audio.length])),
      yield* Codec.decode(weights, yield* Tensor.fromTypedArray(latents, [1, 64, latents.length / 64]))
    ])
    return yield* Effect.forEach(outputs, (x) => Tensor.toTypedArray(x) as Effect.Effect<Float32Array, Tensor.TensorError>)
  }).pipe(Effect.provide(BackendApple.layer())))
  // Latents range to about ±5, and audio to ±1.
  expect(maxError(encoded!, latents)).toBeLessThan(1e-3)
  expect(maxError(decoded!, read("vae-decoded-48k"))).toBeLessThan(1e-4)
}, 120_000)
