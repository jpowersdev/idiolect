/**
 * VoxCPM2 speech generation in effect-torch. A prompt of text tokens and audio latent patches runs
 * through the base and residual language models once; then each step makes one patch of four
 * latent frames: the flow-matching sampler over the local diffusion transformer, the local
 * encoder, and one step of each language model, until the stop head fires.
 *
 * Everything that repeats per patch is a program compiled once at fixed shapes: one sampler
 * step, the local encoder for one patch, and one step of each language model against a
 * fixed-size key-value cache.
 */
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import {
  forward,
  HEAD_DIM,
  KV_HEADS,
  linear,
  release,
  ropeTables,
  ropeValues,
  seedCache,
  slotInputs,
  step,
  transposeProjections,
  type RopeConfig,
  type T,
  type Weights
} from "./MiniCpm.ts"

const PATCH = 4
const LATENT = 64
const LM = 2048
const DIT = 1024
const BASE_LAYERS = 28
const RESIDUAL_LAYERS = 8
const LOCAL_LAYERS = 12
const TIMESTEPS = 10
const GUIDANCE = 2.0
/** Positions in the language models' caches: the prompt and every generated patch. */
export const CAPACITY = 1536

// Special text tokens around the reference clip and before the audio.
export const AUDIO_START = 101
export const REFERENCE_START = 103
export const REFERENCE_END = 104

export interface Prompt {
  readonly text: Uint32Array
  readonly textMask: Float32Array
  /** [length, 4, 64]: each position's audio patch, zero for text. */
  readonly feat: Float32Array
  readonly featMask: Float32Array
}

/**
 * The prompt for continuing from a clip of your voice that is also the cloning reference:
 * [reference start] clip [reference end] transcript and text [audio start] clip.
 * reference and continuation: [patches, 4, 64]; the continuation is encoded left-padded.
 */
export const continuationPrompt = (tokens: ReadonlyArray<number>, reference: Float32Array, continuation: Float32Array): Prompt => {
  const patch = PATCH * LATENT
  const referencePatches = reference.length / patch
  const continuationPatches = continuation.length / patch
  const textLength = tokens.length + 1
  const length = 1 + referencePatches + 1 + textLength + continuationPatches
  const text = new Uint32Array(length)
  const textMask = new Float32Array(length)
  const feat = new Float32Array(length * patch)
  const featMask = new Float32Array(length)
  let at = 0
  text[at] = REFERENCE_START
  textMask[at++] = 1
  feat.set(reference, at * patch)
  featMask.fill(1, at, at + referencePatches)
  at += referencePatches
  text[at] = REFERENCE_END
  textMask[at++] = 1
  text.set(tokens, at)
  text[at + tokens.length] = AUDIO_START
  textMask.fill(1, at, at + textLength)
  at += textLength
  feat.set(continuation, at * patch)
  featMask.fill(1, at, at + continuationPatches)
  return { text, textMask, feat, featMask }
}

/** Sinusoidal embedding of a diffusion time, with scale 1000: [sin, cos] over 1024 channels. */
const timeEmbedding = (t: number) => {
  const half = DIT / 2
  const step = Math.log(10000) / (half - 1)
  const out = new Float32Array(DIT)
  for (let i = 0; i < half; i++) {
    const angle = 1000 * t * Math.exp(-i * step)
    out[i] = Math.sin(angle)
    out[half + i] = Math.cos(angle)
  }
  return out
}

/** The sampler's schedule from 1 to 0, warped by sway sampling (coefficient 1). */
const schedule = Array.from({ length: TIMESTEPS + 1 }, (_, i) => {
  const t = 1 - i / TIMESTEPS
  return t + (Math.cos((Math.PI / 2) * t) - 1 + t)
})

/** A seeded normal generator (mulberry32 and Box–Muller), so renders are reproducible. */
export const normals = (seed: number) => {
  let state = seed >>> 0
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0
    let x = state
    x = Math.imul(x ^ (x >>> 15), x | 1)
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61)
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
  return (count: number) => {
    const out = new Float32Array(count)
    for (let i = 0; i < count; i += 2) {
      const r = Math.sqrt(-2 * Math.log(1 - uniform()))
      const theta = 2 * Math.PI * uniform()
      out[i] = r * Math.cos(theta)
      if (i + 1 < count) out[i + 1] = r * Math.sin(theta)
    }
    return out
  }
}

export interface Model {
  readonly weights: Weights
  readonly dtype: Tensor.DType
  readonly rope: RopeConfig
  readonly programs: {
    readonly denoise: Tensor.CompiledProgram
    readonly encodePatch: Tensor.CompiledProgram
    readonly baseStep: Tensor.CompiledProgram
    readonly residualStep: Tensor.CompiledProgram
  }
}

const quantize = (w: Weights, x: T) =>
  Effect.gen(function*() {
    const squashed = yield* Tensor.tanh(yield* linear(w, "fsq_layer.in_proj", x))
    const nine = yield* Tensor.constantLike(squashed, 9)
    return yield* linear(w, "fsq_layer.out_proj", yield* Tensor.div(yield* Tensor.round(yield* Tensor.mul(squashed, nine)), nine))
  })

const timestepMlp = (w: Weights, name: string, embedding: T) =>
  Effect.gen(function*() {
    const hidden = yield* Tensor.silu(yield* linear(w, `feat_decoder.estimator.${name}.linear_1`, embedding))
    return yield* linear(w, `feat_decoder.estimator.${name}.linear_2`, hidden)
  })

/** The local encoder over patches [B, 4, 64]: a learned summary token, 12 bidirectional layers, projected to the language model. */
const encodePatches = (w: Weights, patches: T, rope: { cos: T; sin: T }) =>
  Effect.gen(function*() {
    const b = patches.shape[0]!
    const projected = yield* linear(w, "feat_encoder.in_proj", patches)
    const special = yield* Tensor.broadcastTo(yield* Tensor.reshape(w["feat_encoder.special_token"]!, [1, 1, DIT]), [b, 1, DIT])
    const { hidden } = yield* forward(w, "feat_encoder.encoder", yield* Tensor.concat([special, projected], { dim: 1 }), {
      layers: LOCAL_LAYERS,
      causal: false,
      rope
    })
    const summary = yield* Tensor.slice(hidden, { start: [0, 0, 0], end: [b, 1, DIT] })
    return yield* linear(w, "enc_to_lm_proj", yield* Tensor.reshape(summary, [1, b, DIT]))
  })

/**
 * One Euler step of the flow-matching sampler, guidance included: the diffusion transformer's
 * velocity for the conditioned and unconditioned halves, blended with the CFG-zero* scale.
 */
const denoiseStep = (w: Weights, x: T, mu: T, cond: T, time: T, deltaTime: T, dt: T, rope: { cos: T; sin: T }) =>
  Effect.gen(function*() {
    const e = "feat_decoder.estimator"
    const both = (a: T, b: T) => Tensor.concat([a, b], { dim: 0 })
    const inputs = yield* linear(w, `${e}.in_proj`, yield* both(x, x))
    const conditions = yield* linear(w, `${e}.cond_proj`, yield* both(cond, cond))
    const timeHidden = yield* Tensor.add(yield* timestepMlp(w, "time_mlp", time), deltaTime)
    const sequence = yield* Tensor.concat([
      yield* Tensor.reshape(yield* both(mu, yield* Tensor.zerosLike(mu)), [2, 2, DIT]),
      yield* Tensor.broadcastTo(yield* Tensor.reshape(timeHidden, [1, 1, DIT]), [2, 1, DIT]),
      conditions,
      inputs
    ], { dim: 1 })
    const { hidden } = yield* forward(w, `${e}.decoder`, sequence, { layers: LOCAL_LAYERS, causal: false, rope })
    const patch = yield* Tensor.slice(hidden, { start: [0, 3 + PATCH, 0], end: [2, 3 + 2 * PATCH, DIT] })
    const v = yield* linear(w, `${e}.out_proj`, patch)
    const positive = yield* Tensor.slice(v, { start: [0, 0, 0], end: [1, PATCH, LATENT] })
    const negative = yield* Tensor.slice(v, { start: [1, 0, 0], end: [2, PATCH, LATENT] })
    const scale = yield* Tensor.div(
      yield* Tensor.sum(yield* Tensor.mul(positive, negative)),
      yield* Tensor.add(yield* Tensor.sum(yield* Tensor.square(negative)), yield* Tensor.constantLike(dt, 1e-8))
    )
    const scaled = yield* Tensor.mul(negative, scale)
    const guided = yield* Tensor.add(scaled, yield* Tensor.mul(yield* Tensor.sub(positive, scaled), yield* Tensor.constantLike(dt, GUIDANCE)))
    return yield* Tensor.sub(x, yield* Tensor.mul(guided, dt))
  })

const placeholder = (slot: number, shape: ReadonlyArray<number>, dtype: Tensor.DType) =>
  Effect.flatMap(Effect.flatMap(Tensor.zeros(shape), (z) => Tensor.cast(z, dtype)), (z) => Tensor.makeInput(slot, z))

/** Compile the per-patch programs. */
const compile = (w: Weights, dtype: Tensor.DType, rope: RopeConfig) =>
  Effect.gen(function*() {
    const fixed = { constantWeights: true }
    const ditRope = yield* ropeTables(rope, 3 + 2 * PATCH, dtype)
    const patchRope = yield* ropeTables(rope, 1 + PATCH, dtype)
    const deltaTime = yield* timestepMlp(w, "delta_time_mlp", yield* Tensor.cast(yield* Tensor.fromTypedArray(timeEmbedding(0), [1, DIT]), dtype))
    const [deltaTimeComputed] = yield* Tensor.compute([deltaTime])

    const denoise = yield* Tensor.freezeProgram([
      yield* denoiseStep(
        w,
        yield* placeholder(0, [1, PATCH, LATENT], dtype),
        yield* placeholder(1, [1, LM], dtype),
        yield* placeholder(2, [1, PATCH, LATENT], dtype),
        yield* placeholder(3, [1, DIT], dtype),
        deltaTimeComputed!,
        yield* placeholder(4, [1, 1, 1], dtype),
        ditRope
      )
    ], fixed)

    const encodePatch = yield* Tensor.freezeProgram([
      yield* encodePatches(w, yield* placeholder(0, [1, PATCH, LATENT], dtype), patchRope)
    ], fixed)

    // Inputs shared by both language-model steps: this position's slot and visibility.
    const stepCaches = (first: number, layers: number) =>
      Effect.forEach(Array.from({ length: layers }), (_, i) =>
        Effect.gen(function*() {
          return [
            yield* placeholder(first + 2 * i, [1, KV_HEADS, CAPACITY, HEAD_DIM], dtype),
            yield* placeholder(first + 2 * i + 1, [1, KV_HEADS, CAPACITY, HEAD_DIM], dtype)
          ] as const
        }))

    // Base step: the new patch's embedding at this position → quantized hidden state, the
    // residual model's input, and the stop head's two logits.
    const embed = yield* placeholder(0, [1, 1, LM], dtype)
    const base = yield* step(w, "base_lm", BASE_LAYERS, {
      input: embed,
      rope: { cos: yield* placeholder(1, [1, 1, 1, HEAD_DIM], dtype), sin: yield* placeholder(2, [1, 1, 1, HEAD_DIM], dtype) },
      slot: yield* placeholder(3, [1, 1, CAPACITY, 1], dtype),
      visible: yield* placeholder(4, [1, 1, 1, CAPACITY], dtype),
      caches: yield* stepCaches(5, BASE_LAYERS)
    })
    const quantized = yield* quantize(w, base.hidden)
    const fused = yield* linear(w, "fusion_concat_proj", yield* Tensor.concat([quantized, embed], { dim: -1 }))
    const stop = yield* linear(w, "stop_head", yield* Tensor.silu(yield* linear(w, "stop_proj", quantized)))
    const baseStep = yield* Tensor.freezeProgram([quantized, fused, stop, ...base.caches.flat()], fixed)

    // Residual step: the fused input → the sampler's next conditioning.
    const lmHidden = yield* placeholder(3, [1, 1, LM], dtype)
    const residual = yield* step(w, "residual_lm", RESIDUAL_LAYERS, {
      input: yield* placeholder(0, [1, 1, LM], dtype),
      slot: yield* placeholder(1, [1, 1, CAPACITY, 1], dtype),
      visible: yield* placeholder(2, [1, 1, 1, CAPACITY], dtype),
      caches: yield* stepCaches(4, RESIDUAL_LAYERS)
    })
    const mu = yield* Tensor.concat([
      yield* linear(w, "lm_to_dit_proj", yield* Tensor.reshape(lmHidden, [1, LM])),
      yield* linear(w, "res_to_dit_proj", yield* Tensor.reshape(residual.hidden, [1, LM]))
    ], { dim: -1 })
    const residualStep = yield* Tensor.freezeProgram([mu, ...residual.caches.flat()], fixed)

    return { denoise, encodePatch, baseStep, residualStep }
  })

/** Load VoxCPM2 from its checkpoint (stored in bfloat16) at `dtype`, and compile its programs. */
export const load = (stored: Readonly<Record<string, T>>, options: { readonly dtype: Tensor.DType; readonly rope: RopeConfig }) =>
  Effect.gen(function*() {
    const cast: Record<string, T> = { ...stored }
    for (const name of Object.keys(stored)) {
      if (stored[name]!.dtype === options.dtype) continue
      cast[name] = (yield* Tensor.compute([yield* Tensor.cast(stored[name]!, options.dtype)]))[0]!
      yield* release([stored[name]!])
    }
    const weights = yield* transposeProjections(cast, new Set(["base_lm.embed_tokens.weight"]))
    const programs = yield* compile(weights, options.dtype, options.rope)
    return { weights, dtype: options.dtype, rope: options.rope, programs } satisfies Model
  })

/** A tensor's values as float32, read back to the host. */
const asF32 = (x: T) =>
  Effect.gen(function*() {
    const [y] = yield* Tensor.compute([yield* Tensor.cast(x, "f32")])
    const values = (yield* Tensor.toTypedArray(y!)) as Float32Array
    yield* release([y])
    return values
  })

export interface GenerateOptions {
  readonly minLength: number
  readonly maxLength: number
  readonly seed: number
  /** Each patch's starting noise, [4, 64] time-major; otherwise drawn from `seed`. */
  readonly noise?: (patch: number) => Float32Array
}

/** Generate latent patches, each [4, 64] time-major, from a prompt. */
export const generate = (model: Model, prompt: Prompt, options: GenerateOptions) =>
  Effect.gen(function*() {
    const { dtype, programs, weights: w } = model
    const length = prompt.text.length
    if (length + options.maxLength > CAPACITY) {
      return yield* new Tensor.TensorError({ op: "generate", message: `The prompt and output need ${length + options.maxLength} positions; the cache holds ${CAPACITY}.` })
    }
    const typed = (data: Float32Array, shape: ReadonlyArray<number>) => Effect.flatMap(Tensor.fromTypedArray(data, shape), (x) => Tensor.cast(x, dtype))

    // The prompt, once through both language models, seeds their caches.
    const masks = { text: yield* typed(prompt.textMask, [1, length, 1]), feat: yield* typed(prompt.featMask, [1, length, 1]) }
    const featEmbed = yield* encodePatches(w, yield* typed(prompt.feat, [length, PATCH, LATENT]), yield* ropeTables(model.rope, 1 + PATCH, dtype))
    const textEmbed = yield* Tensor.embedding(yield* Tensor.fromTypedArray(prompt.text, [1, length]), { weight: w["base_lm.embed_tokens.weight"]! })
    const combined = yield* Tensor.add(yield* Tensor.mul(textEmbed, masks.text), yield* Tensor.mul(featEmbed, masks.feat))
    const base = yield* forward(w, "base_lm", combined, { layers: BASE_LAYERS, causal: true, rope: yield* ropeTables(model.rope, length, dtype) })
    const mixed = yield* Tensor.add(yield* Tensor.mul(yield* quantize(w, base.hidden), masks.feat), yield* Tensor.mul(base.hidden, masks.text))
    const residualInput = yield* linear(w, "fusion_concat_proj", yield* Tensor.concat([mixed, yield* Tensor.mul(featEmbed, masks.feat)], { dim: -1 }))
    const residual = yield* forward(w, "residual_lm", residualInput, { layers: RESIDUAL_LAYERS, causal: true })
    const last = (x: T) => Tensor.slice(x, { start: [0, length - 1, 0], end: [1, length, x.shape[2]!] })
    const lmHidden = yield* last(mixed)
    const mu = yield* Tensor.concat([
      yield* linear(w, "lm_to_dit_proj", yield* Tensor.reshape(lmHidden, [1, LM])),
      yield* linear(w, "res_to_dit_proj", yield* Tensor.reshape(yield* last(residual.hidden), [1, LM]))
    ], { dim: -1 })
    const firstStop = yield* linear(w, "stop_head", yield* Tensor.silu(yield* linear(w, "stop_proj", lmHidden)))
    const seeded = yield* Tensor.compute([
      lmHidden,
      mu,
      firstStop,
      ...(yield* Effect.forEach([...base.keysValues, ...residual.keysValues].flat(), (kv) => seedCache(kv, CAPACITY)))
    ])
    let [lm, conditioning, stop] = seeded
    let baseCaches = seeded.slice(3, 3 + 2 * BASE_LAYERS)
    let residualCaches = seeded.slice(3 + 2 * BASE_LAYERS)

    // The last prompt patch conditions the first generated one.
    let previous: T = yield* typed(prompt.feat.slice((length - 1) * PATCH * LATENT, length * PATCH * LATENT), [1, PATCH, LATENT])
    const noise = normals(options.seed)
    const times = yield* Effect.forEach(schedule, (t) => typed(timeEmbedding(t), [1, DIT]))
    const patches: Array<Float32Array> = []
    for (let i = 0; i < options.maxLength; i++) {
      // Flow matching: Euler steps from noise at the schedule's times; the first is skipped (CFG-zero*).
      let x: T = yield* typed(options.noise?.(i) ?? noise(PATCH * LATENT), [1, PATCH, LATENT])
      for (let s = 2; s <= TIMESTEPS; s++) {
        const dt = yield* typed(Float32Array.of(schedule[s - 1]! - schedule[s]!), [1, 1, 1])
        const next = (yield* Tensor.runProgram(programs.denoise, [x, conditioning!, previous, times[s - 1]!, dt]))[0]!
        yield* release([x])
        x = next
      }
      patches.push(yield* asF32(x))
      yield* release([previous])
      previous = x

      const [keepGoing, finish] = yield* asF32(stop!)
      yield* release([stop])
      if (i > options.minLength && finish! > keepGoing!) break

      // The new patch through the local encoder, then one step of each language model.
      const [embed] = yield* Tensor.runProgram(programs.encodePatch, [x])
      const position = length + i
      const { slot, visible } = slotInputs(position, CAPACITY)
      const { cos, sin } = ropeValues(model.rope, 1, position)
      const slotTensor = yield* typed(slot, [1, 1, CAPACITY, 1])
      const visibleTensor = yield* typed(visible, [1, 1, 1, CAPACITY])
      const baseOut = yield* Tensor.runProgram(programs.baseStep, [
        embed!,
        yield* typed(cos, [1, 1, 1, HEAD_DIM]),
        yield* typed(sin, [1, 1, 1, HEAD_DIM]),
        slotTensor,
        visibleTensor,
        ...baseCaches
      ])
      yield* release([embed, lm, ...baseCaches])
      ;[lm] = baseOut
      stop = baseOut[2]
      baseCaches = baseOut.slice(3)
      const residualOut = yield* Tensor.runProgram(programs.residualStep, [baseOut[1]!, slotTensor, visibleTensor, lm!, ...residualCaches])
      yield* release([baseOut[1], conditioning, ...residualCaches])
      conditioning = residualOut[0]
      residualCaches = residualOut.slice(1)
    }
    yield* release([previous, lm, conditioning, stop, ...baseCaches, ...residualCaches])
    return patches
  })
