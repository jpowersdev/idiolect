/**
 * VoxCPM2's audio codec (AudioVAE v2): causal convolutions between 16 kHz audio and 25 Hz latents
 * of 64 channels, and from latents to 48 kHz audio. Weights come from audiovae.pth with weight
 * normalization folded in (see Models.ts).
 */
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import type { T, Weights } from "./MiniCpm.ts"

export const INPUT_RATE = 16000
export const OUTPUT_RATE = 48000
const ENCODER_RATES = [2, 5, 8, 8]
const DECODER_RATES = [8, 6, 5, 2, 2, 2]
/** Input samples per latent frame, and output samples per latent frame. */
export const ENCODE_HOP = ENCODER_RATES.reduce((a, b) => a * b)
export const DECODE_HOP = DECODER_RATES.reduce((a, b) => a * b)
// Output-rate buckets are [20000, 30000, 40000]; 48 kHz is bucket 3.
const OUTPUT_BUCKET = 3

/** x + sin²(αx) / α, per channel. */
const snake = (w: Weights, name: string, x: T) =>
  Effect.gen(function*() {
    const alpha = w[`${name}.alpha`]!
    const sine = yield* Tensor.sin(yield* Tensor.mul(x, alpha))
    const inverse = yield* Tensor.reciprocal(yield* Tensor.add(alpha, yield* Tensor.constantLike(alpha, 1e-9)))
    return yield* Tensor.add(x, yield* Tensor.mul(inverse, yield* Tensor.square(sine)))
  })

/** A causal convolution: padded on the left only, so no output depends on later input. */
const conv = (
  w: Weights,
  name: string,
  x: T,
  options: { readonly stride?: number; readonly dilation?: number; readonly groups?: number; readonly leftPad: number }
) =>
  Effect.gen(function*() {
    const padded = options.leftPad > 0 ? yield* Tensor.pad(x, [[0, 0], [0, 0], [options.leftPad, 0]]) : x
    const y = yield* Tensor.conv1d(padded, w[`${name}.weight`]!, {
      stride: options.stride ?? 1,
      dilation: options.dilation ?? 1,
      groups: options.groups ?? 1
    })
    const bias = w[`${name}.bias`]
    return bias ? yield* Tensor.add(y, yield* Tensor.reshape(bias, [1, bias.shape[0]!, 1])) : y
  })

/** Snake, a depthwise 7-wide dilated convolution, Snake, a pointwise convolution; plus the input. */
const residualUnit = (w: Weights, name: string, x: T, dilation: number) =>
  Effect.gen(function*() {
    let y = yield* snake(w, `${name}.block.0`, x)
    y = yield* conv(w, `${name}.block.1`, y, { dilation, groups: x.shape[1]!, leftPad: 6 * dilation })
    y = yield* snake(w, `${name}.block.2`, y)
    y = yield* conv(w, `${name}.block.3`, y, { leftPad: 0 })
    return yield* Tensor.add(x, y)
  })

/** 16 kHz audio [1, 1, samples], a whole number of hops long, to latents [1, 64, frames]. */
export const encode = (w: Weights, audio: T) =>
  Effect.gen(function*() {
    let x = yield* conv(w, "encoder.block.0", audio, { leftPad: 6 })
    for (const [i, stride] of ENCODER_RATES.entries()) {
      const block = `encoder.block.${i + 1}.block`
      for (const [j, dilation] of [1, 3, 9].entries()) x = yield* residualUnit(w, `${block}.${j}`, x, dilation)
      x = yield* snake(w, `${block}.3`, x)
      x = yield* conv(w, `${block}.4`, x, { stride, leftPad: stride })
    }
    return yield* conv(w, "encoder.fc_mu", x, { leftPad: 2 })
  })

/** Latents [1, 64, frames] to 48 kHz audio [1, 1, frames · 1920]. */
export const decode = (w: Weights, latents: T) =>
  Effect.gen(function*() {
    let x = yield* conv(w, "decoder.model.0", latents, { groups: latents.shape[1]!, leftPad: 6 })
    x = yield* conv(w, "decoder.model.1", x, { leftPad: 0 })
    const bucket = yield* Tensor.fromTypedArray(Uint32Array.of(OUTPUT_BUCKET), [1])
    for (const [i, stride] of DECODER_RATES.entries()) {
      const layer = i + 2
      // Scale and shift for the output sample rate, before each upsampling block.
      const condition = `decoder.sr_cond_model.${layer}`
      const scale = yield* Tensor.embedding(bucket, { weight: w[`${condition}.scale_embed.weight`]! })
      const shift = yield* Tensor.embedding(bucket, { weight: w[`${condition}.bias_embed.weight`]! })
      x = yield* Tensor.add(yield* Tensor.mul(x, yield* Tensor.unsqueeze(scale, -1)), yield* Tensor.unsqueeze(shift, -1))
      const block = `decoder.model.${layer}.block`
      x = yield* snake(w, `${block}.0`, x)
      // A causal transposed convolution: trim the right edge by the stride.
      const up = yield* Tensor.convTranspose1d(x, w[`${block}.1.weight`]!, { stride })
      const trimmed = yield* Tensor.slice(up, { start: [0, 0, 0], end: [1, up.shape[1]!, up.shape[2]! - stride] })
      const bias = w[`${block}.1.bias`]!
      x = yield* Tensor.add(trimmed, yield* Tensor.reshape(bias, [1, bias.shape[0]!, 1]))
      for (const [j, dilation] of [1, 3, 9].entries()) x = yield* residualUnit(w, `${block}.${j + 2}`, x, dilation)
    }
    x = yield* snake(w, "decoder.model.8", x)
    x = yield* conv(w, "decoder.model.9", x, { leftPad: 6 })
    return yield* Tensor.tanh(x)
  })
