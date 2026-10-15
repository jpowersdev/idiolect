/**
 * VoxCPM2's audio codec (AudioVAE v2): causal convolutions between 16 kHz audio and 25 Hz latents
 * of 64 channels, and from latents to 48 kHz audio. Weights come from audiovae.pth with weight
 * normalization folded in (see Pth.ts).
 *
 * Signals are [channels, samples]. Dense convolutions run as one matrix multiply over strided
 * slices of the input, and transposed ones as one multiply and an overlap-add: far faster on the
 * GPU than direct convolution at these sizes.
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

const size = (x: T) => x.shape.reduce((a, b) => a * b, 1)

/**
 * Weights rearranged for matrix multiplies: a convolution's [out, in, kernel] as [out, kernel · in],
 * a transposed convolution's [in, out, kernel] as [kernel · out, in], biases and Snake's α as
 * columns. Depthwise convolutions keep their shape.
 */
export const prepare = (stored: Weights) =>
  Effect.gen(function*() {
    const names = Object.keys(stored)
    const prepared = yield* Effect.forEach(names, (name) =>
      Effect.gen(function*() {
        const w = stored[name]!
        if (name.endsWith(".alpha") || name.endsWith(".bias")) return yield* Tensor.reshape(w, [size(w), 1])
        // Depthwise convolutions: in each residual unit, and the decoder's first layer.
        if (!name.endsWith(".weight") || w.shape.length !== 3 || /\.block\.\d+\.block\.1\.weight$|^decoder\.model\.0\./.test(name)) return w
        // Transposed convolutions are the decoder blocks' first layer.
        if (/^decoder\.model\.\d+\.block\.1\.weight$/.test(name)) {
          const [input, output, kernel] = w.shape as [number, number, number]
          return yield* Tensor.reshape(yield* Tensor.transpose(w, [2, 1, 0]), [kernel * output, input])
        }
        const [output, input, kernel] = w.shape as [number, number, number]
        return yield* Tensor.reshape(yield* Tensor.transpose(w, [0, 2, 1]), [output, kernel * input])
      }))
    const computed = yield* Tensor.compute(prepared)
    return Object.fromEntries(names.map((name, i) => [name, computed[i]!])) as Weights
  })

/** x + sin²(αx) / α, per channel. */
const snake = (w: Weights, name: string, x: T) =>
  Effect.gen(function*() {
    const alpha = w[`${name}.alpha`]!
    const sine = yield* Tensor.sin(yield* Tensor.mul(x, alpha))
    const inverse = yield* Tensor.reciprocal(yield* Tensor.add(alpha, yield* Tensor.constantLike(alpha, 1e-9)))
    return yield* Tensor.add(x, yield* Tensor.mul(inverse, yield* Tensor.square(sine)))
  })

interface ConvOptions {
  readonly stride?: number
  readonly dilation?: number
  readonly leftPad: number
}

const withBias = (w: Weights, name: string, y: T) => {
  const bias = w[`${name}.bias`]
  return bias ? Tensor.add(y, bias) : Effect.succeed(y)
}

/** A causal convolution: padded on the left only, so no output depends on later input. */
const conv = (w: Weights, name: string, x: T, options: ConvOptions) =>
  Effect.gen(function*() {
    const stride = options.stride ?? 1
    const dilation = options.dilation ?? 1
    const padded = options.leftPad > 0 ? yield* Tensor.pad(x, [[0, 0], [options.leftPad, 0]]) : x
    const [channels, length] = padded.shape as [number, number]
    const weight = w[`${name}.weight`]!
    const kernel = weight.shape[1]! / channels
    const outputs = Math.floor((length - dilation * (kernel - 1) - 1) / stride) + 1
    // Column k holds the input each output sees through kernel tap k.
    const taps = yield* Effect.forEach(Array.from({ length: kernel }, (_, k) => k), (k) =>
      kernel === 1 && stride === 1
        ? Effect.succeed(padded)
        : Tensor.slice(padded, { start: [0, k * dilation], end: [channels, k * dilation + (outputs - 1) * stride + 1], stride: [1, stride] }))
    const columns = taps.length === 1 ? taps[0]! : yield* Tensor.concat(taps as [T, T, ...Array<T>], { dim: 0 })
    return yield* withBias(w, name, yield* Tensor.matmul(weight, columns))
  })

/** A causal depthwise convolution: each channel filtered on its own. */
const depthwise = (w: Weights, name: string, x: T, options: ConvOptions) =>
  Effect.gen(function*() {
    const [channels, length] = x.shape as [number, number]
    const padded = yield* Tensor.pad(yield* Tensor.reshape(x, [1, channels, length]), [[0, 0], [0, 0], [options.leftPad, 0]])
    const y = yield* Tensor.conv1d(padded, w[`${name}.weight`]!, { dilation: options.dilation ?? 1, groups: channels })
    return yield* withBias(w, name, yield* Tensor.reshape(y, [channels, y.shape[2]!]))
  })

/**
 * A causal transposed convolution with a kernel twice its stride: each input frame makes two
 * strides of output, overlapping the next frame's first; the trailing stride is dropped.
 */
const upsample = (w: Weights, name: string, x: T, stride: number) =>
  Effect.gen(function*() {
    const frames = x.shape[1]!
    const weight = w[`${name}.weight`]!
    const output = weight.shape[0]! / (2 * stride)
    const spread = yield* Tensor.reshape(yield* Tensor.matmul(weight, x), [2, stride, output, frames])
    const own = yield* Tensor.slice(spread, { start: [0, 0, 0, 0], end: [1, stride, output, frames] })
    const tail = yield* Tensor.slice(spread, { start: [1, 0, 0, 0], end: [2, stride, output, frames - 1] })
    const previous = yield* Tensor.pad(tail, [[0, 0], [0, 0], [0, 0], [1, 0]])
    const phases = yield* Tensor.reshape(yield* Tensor.add(own, previous), [stride, output, frames])
    const interleaved = yield* Tensor.reshape(yield* Tensor.transpose(phases, [1, 2, 0]), [output, frames * stride])
    return yield* withBias(w, name, interleaved)
  })

/** Snake, a depthwise 7-wide dilated convolution, Snake, a pointwise convolution; plus the input. */
const residualUnit = (w: Weights, name: string, x: T, dilation: number) =>
  Effect.gen(function*() {
    let y: T = yield* snake(w, `${name}.block.0`, x)
    y = yield* depthwise(w, `${name}.block.1`, y, { dilation, leftPad: 6 * dilation })
    y = yield* snake(w, `${name}.block.2`, y)
    y = yield* conv(w, `${name}.block.3`, y, { leftPad: 0 })
    return yield* Tensor.add(x, y)
  })

/** 16 kHz audio [1, 1, samples], a whole number of hops long, to latents [1, 64, frames]. */
export const encode = (w: Weights, audio: T) =>
  Effect.gen(function*() {
    let x: T = yield* conv(w, "encoder.block.0", yield* Tensor.reshape(audio, [1, size(audio)]), { leftPad: 6 })
    for (const [i, stride] of ENCODER_RATES.entries()) {
      const block = `encoder.block.${i + 1}.block`
      for (const [j, dilation] of [1, 3, 9].entries()) x = yield* residualUnit(w, `${block}.${j}`, x, dilation)
      x = yield* snake(w, `${block}.3`, x)
      x = yield* conv(w, `${block}.4`, x, { stride, leftPad: stride })
    }
    const latents = yield* conv(w, "encoder.fc_mu", x, { leftPad: 2 })
    return yield* Tensor.reshape(latents, [1, ...latents.shape])
  })

/** Latents [1, 64, frames] to 48 kHz audio [1, 1, frames · 1920]. */
export const decode = (w: Weights, latents: T) =>
  Effect.gen(function*() {
    let x: T = yield* depthwise(w, "decoder.model.0", yield* Tensor.reshape(latents, latents.shape.slice(1)), { leftPad: 6 })
    x = yield* conv(w, "decoder.model.1", x, { leftPad: 0 })
    const bucket = yield* Tensor.fromTypedArray(Uint32Array.of(OUTPUT_BUCKET), [1])
    for (const [i, stride] of DECODER_RATES.entries()) {
      const layer = i + 2
      // Scale and shift for the output sample rate, before each upsampling block.
      const condition = `decoder.sr_cond_model.${layer}`
      const column = (name: string) =>
        Effect.flatMap(Tensor.embedding(bucket, { weight: w[`${condition}.${name}.weight`]! }), (e) => Tensor.reshape(e, [size(e), 1]))
      x = yield* Tensor.add(yield* Tensor.mul(x, yield* column("scale_embed")), yield* column("bias_embed"))
      const block = `decoder.model.${layer}.block`
      x = yield* snake(w, `${block}.0`, x)
      x = yield* upsample(w, `${block}.1`, x, stride)
      for (const [j, dilation] of [1, 3, 9].entries()) x = yield* residualUnit(w, `${block}.${j + 2}`, x, dilation)
    }
    x = yield* snake(w, "decoder.model.8", x)
    x = yield* conv(w, "decoder.model.9", x, { leftPad: 6 })
    return yield* Tensor.reshape(yield* Tensor.tanh(x), [1, 1, size(x)])
  })
