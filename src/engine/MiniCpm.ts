/**
 * The MiniCPM4 transformer that VoxCPM2's base and residual language models, local encoder, and
 * diffusion transformer share: RMSNorm, grouped-query attention (16 query heads over 2 key-value
 * heads, 128 wide), optional rotary positions, and a gated SiLU feed-forward.
 *
 * `forward` runs a whole sequence at once. `step` traces a one-token step against a fixed-size
 * key-value cache, for a program compiled once and run per token.
 *
 * Projection weights are expected pre-transposed, [in, out] (see `transposeProjections`): Metal
 * multiplies a few rows by them markedly faster than by PyTorch's [out, in] layout.
 */
import { Tensor } from "@effect-torch/core"
import { Effect } from "effect"

export type T = Tensor.Any
export type Weights = Readonly<Record<string, T>>

export const HEADS = 16
export const KV_HEADS = 2
export const HEAD_DIM = 128
const EPS = 1e-5

export interface RopeConfig {
  readonly theta: number
  /** LongRoPE's short factors, one per frequency; each divides its frequency. */
  readonly factors: ReadonlyArray<number>
}

/** Rotary cos and sin for `positions` consecutive positions from `start`: two [positions, 128] arrays. */
export const ropeValues = (config: RopeConfig, positions: number, start = 0) => {
  const half = HEAD_DIM / 2
  const cos = new Float32Array(positions * HEAD_DIM)
  const sin = new Float32Array(positions * HEAD_DIM)
  for (let p = 0; p < positions; p++) {
    for (let j = 0; j < half; j++) {
      const angle = ((start + p) / config.factors[j]!) / config.theta ** ((2 * j) / HEAD_DIM)
      for (const k of [j, j + half]) {
        cos[p * HEAD_DIM + k] = Math.cos(angle)
        sin[p * HEAD_DIM + k] = Math.sin(angle)
      }
    }
  }
  return { cos, sin }
}

export interface Rope {
  readonly cos: T
  readonly sin: T
}

export const ropeTables = (config: RopeConfig, positions: number, dtype: Tensor.DType) =>
  Effect.gen(function*() {
    const { cos, sin } = ropeValues(config, positions)
    return {
      cos: yield* Tensor.cast(yield* Tensor.fromTypedArray(cos, [positions, HEAD_DIM]), dtype),
      sin: yield* Tensor.cast(yield* Tensor.fromTypedArray(sin, [positions, HEAD_DIM]), dtype)
    } satisfies Rope
  })

/** x·cos + rotate_half(x)·sin, where rotate_half(x) = [-x₂, x₁]. x: [B, heads, T, 128]. */
const rotate = (x: T, rope: Rope) =>
  Effect.gen(function*() {
    const half = HEAD_DIM / 2
    const [b, h, t] = x.shape as [number, number, number]
    const first = yield* Tensor.slice(x, { start: [0, 0, 0, 0], end: [b, h, t, half] })
    const second = yield* Tensor.slice(x, { start: [0, 0, 0, half], end: [b, h, t, HEAD_DIM] })
    const rotated = yield* Tensor.concat([yield* Tensor.neg(second), first], { dim: -1 })
    return yield* Tensor.add(yield* Tensor.mul(x, rope.cos), yield* Tensor.mul(rotated, rope.sin))
  })

/** [B, T, heads·128] → [B, heads, T, 128]. */
const splitHeads = (x: T, heads: number) =>
  Effect.flatMap(
    Tensor.reshape(x, [x.shape[0]!, x.shape[1]!, heads, HEAD_DIM]),
    (y) => Tensor.transpose(y, [0, 2, 1, 3])
  )

/** Repeat each key-value head for the eight query heads that share it: [B, 2, T, 128] → [B, 16, T, 128]. */
const shareHeads = (x: T) =>
  Effect.gen(function*() {
    const [b, , t, d] = x.shape as [number, number, number, number]
    const expanded = yield* Tensor.broadcastTo(yield* Tensor.unsqueeze(x, 2), [b, KV_HEADS, HEADS / KV_HEADS, t, d])
    return yield* Tensor.reshape(expanded, [b, HEADS, t, d])
  })

/** x · W (+ b), with W pre-transposed to [in, out]. */
export const linear = (w: Weights, name: string, x: T) =>
  Effect.gen(function*() {
    const y = yield* Tensor.matmul(x, w[`${name}.weight`]!)
    const bias = w[`${name}.bias`]
    return bias ? yield* Tensor.add(y, bias) : y
  })

/**
 * Transpose every projection weight, PyTorch's [out, in], to [in, out] once at load. Lookup
 * tables in `keep` stay as they are.
 */
export const transposeProjections = (weights: Record<string, T>, keep: ReadonlySet<string>) =>
  Effect.gen(function*() {
    const out = { ...weights }
    for (const name of Object.keys(weights)) {
      if (!name.endsWith(".weight") || weights[name]!.shape.length !== 2 || keep.has(name)) continue
      // One at a time, freeing each original, so loading never holds two copies of the model.
      out[name] = (yield* Tensor.compute([yield* Tensor.transpose(weights[name]!, [1, 0])]))[0]!
      yield* release([weights[name]!])
    }
    return out as Weights
  })

/**
 * Free tensors' GPU memory for reuse. effect-torch frees only when asked, never on garbage
 * collection; graphs not yet computed hold nothing and are skipped.
 */
export const release = (tensors: Iterable<T | undefined>) =>
  Tensor.clearAll([...tensors].filter((x): x is Tensor.Concrete => x !== undefined && Tensor.isTensor(x)))

const feedForward = (w: Weights, layer: string, x: T) =>
  Effect.gen(function*() {
    const normed = yield* Tensor.rmsNorm(x, w[`${layer}.post_attention_layernorm.weight`]!, EPS)
    const gate = yield* Tensor.silu(yield* linear(w, `${layer}.mlp.gate_proj`, normed))
    const up = yield* linear(w, `${layer}.mlp.up_proj`, normed)
    return yield* Tensor.add(x, yield* linear(w, `${layer}.mlp.down_proj`, yield* Tensor.mul(gate, up)))
  })

const queryKeyValue = (w: Weights, layer: string, normed: T, rope: Rope | undefined) =>
  Effect.gen(function*() {
    let q = yield* splitHeads(yield* linear(w, `${layer}.self_attn.q_proj`, normed), HEADS)
    let k = yield* splitHeads(yield* linear(w, `${layer}.self_attn.k_proj`, normed), KV_HEADS)
    const v = yield* splitHeads(yield* linear(w, `${layer}.self_attn.v_proj`, normed), KV_HEADS)
    if (rope) {
      q = yield* rotate(q, rope)
      k = yield* rotate(k, rope)
    }
    return { q, k, v }
  })

const mergeHeads = (context: T, batch: number, length: number) =>
  Effect.flatMap(Tensor.transpose(context, [0, 2, 1, 3]), (y) => Tensor.reshape(y, [batch, length, HEADS * HEAD_DIM]))

export interface ForwardOptions {
  readonly layers: number
  readonly causal: boolean
  /** Rotary tables for the sequence's positions; absent for models without positions. */
  readonly rope?: Rope | undefined
}

/**
 * The transformer over a whole sequence [B, T, hidden], ending with its final RMSNorm. Also returns
 * each layer's keys and values ([B, 2, T, 128], rotated), to seed a step cache.
 */
export const forward = (w: Weights, prefix: string, input: T, options: ForwardOptions) =>
  Effect.gen(function*() {
    let x = input
    const keysValues: Array<readonly [T, T]> = []
    const [batch, length] = input.shape as [number, number]
    for (let i = 0; i < options.layers; i++) {
      const layer = `${prefix}.layers.${i}`
      const normed = yield* Tensor.rmsNorm(x, w[`${layer}.input_layernorm.weight`]!, EPS)
      const { k, q, v } = yield* queryKeyValue(w, layer, normed, options.rope)
      keysValues.push([k, v])
      const context = yield* Tensor.scaledDotProductAttention(q, yield* shareHeads(k), yield* shareHeads(v), {
        causal: options.causal
      })
      x = yield* Tensor.add(x, yield* linear(w, `${layer}.self_attn.o_proj`, yield* mergeHeads(context, batch, length)))
      x = yield* feedForward(w, layer, x)
    }
    return { hidden: yield* Tensor.rmsNorm(x, w[`${prefix}.norm.weight`]!, EPS), keysValues }
  })

export interface StepInputs {
  /** [1, 1, hidden]. */
  readonly input: T
  /** Rotary cos and sin for this position, [1, 1, 1, 128] each; absent for models without positions. */
  readonly rope?: Rope | undefined
  /** One-hot over cache slots, [1, 1, capacity, 1]: where this token's keys and values go. */
  readonly slot: T
  /** Additive attention mask over cache slots, [1, 1, 1, capacity]: 0 up to this token, -1e9 after. */
  readonly visible: T
  /** Each layer's cached keys and values, [1, 2, capacity, 128]. */
  readonly caches: ReadonlyArray<readonly [T, T]>
}

/**
 * One token through the transformer against a fixed-size cache, as traced graph nodes for a
 * program compiled once: writes the token's keys and values into its slot, and attends to
 * every visible slot. Returns the normalized hidden state and the updated caches.
 */
export const step = (w: Weights, prefix: string, layers: number, inputs: StepInputs) =>
  Effect.gen(function*() {
    let x = inputs.input
    const keep = yield* Tensor.sub(yield* Tensor.onesLike(inputs.slot), inputs.slot)
    const write = (cache: T, fresh: T) =>
      Effect.gen(function*() {
        return yield* Tensor.add(yield* Tensor.mul(cache, keep), yield* Tensor.mul(fresh, inputs.slot))
      })
    const caches: Array<readonly [T, T]> = []
    const scale = yield* Tensor.cast(yield* Tensor.constant(1 / Math.sqrt(HEAD_DIM)), x.dtype)
    for (let i = 0; i < layers; i++) {
      const layer = `${prefix}.layers.${i}`
      const normed = yield* Tensor.rmsNorm(x, w[`${layer}.input_layernorm.weight`]!, EPS)
      const { k: kNew, q, v: vNew } = yield* queryKeyValue(w, layer, normed, inputs.rope)
      const k = yield* write(inputs.caches[i]![0], kNew)
      const v = yield* write(inputs.caches[i]![1], vNew)
      caches.push([k, v])
      const scores = yield* Tensor.add(
        yield* Tensor.mul(yield* Tensor.matmul(q, yield* Tensor.transpose(yield* shareHeads(k), [0, 1, 3, 2])), scale),
        inputs.visible
      )
      const context = yield* Tensor.matmul(yield* Tensor.softmax(scores), yield* shareHeads(v))
      x = yield* Tensor.add(x, yield* linear(w, `${layer}.self_attn.o_proj`, yield* mergeHeads(context, 1, 1)))
      x = yield* feedForward(w, layer, x)
    }
    return { hidden: yield* Tensor.rmsNorm(x, w[`${prefix}.norm.weight`]!, EPS), caches }
  })

/** Place keys or values for the first T positions into a [1, 2, capacity, 128] cache. */
export const seedCache = (keysOrValues: T, capacity: number) =>
  Tensor.pad(keysOrValues, [[0, 0], [0, 0], [0, capacity - keysOrValues.shape[2]!], [0, 0]])

/** The slot and visibility inputs for position `at` of a cache with `capacity` slots. */
export const slotInputs = (at: number, capacity: number) => {
  const slot = new Float32Array(capacity)
  slot[at] = 1
  const visible = new Float32Array(capacity).fill(-1e9)
  visible.fill(0, 0, at + 1)
  return { slot, visible }
}
