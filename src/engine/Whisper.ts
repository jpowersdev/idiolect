/**
 * Whisper base.en in effect-torch: transcribes 16 kHz audio with word timings, for following a
 * reading during setup and for checking each rendered section. Decoding is greedy without
 * timestamps; word timings come from the cross-attention of its alignment heads, aligned by
 * dynamic time warping, as OpenAI's find_alignment computes them.
 */
import { Safetensors, Tensor } from "@effect-torch/core"
import { Effect } from "effect"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import type { Word } from "../Api.ts"
import { release } from "./MiniCpm.ts"

type T = Tensor.Any
type Weights = Readonly<Record<string, T>>

const D = 512
const HEADS = 8
const LAYERS = 6
const N_FFT = 400
const HOP = 160
export const SAMPLE_RATE = 16000
const WINDOW = 30 * SAMPLE_RATE
const FRAMES = 3000
const AUDIO_FRAMES = 1500
const MAX_TOKENS = 448
/** At most half the context for one window's text, as Whisper does. */
const MAX_NEW_TOKENS = 224

interface Generation {
  readonly decoder_start_token_id: number
  readonly no_timestamps_token_id: number
  readonly eos_token_id: number
  readonly suppress_tokens: ReadonlyArray<number>
  readonly begin_suppress_tokens: ReadonlyArray<number>
  readonly alignment_heads: ReadonlyArray<readonly [number, number]>
}

export interface Recognizer {
  readonly weights: Weights
  readonly generation: Generation
  readonly decode: (ids: ReadonlyArray<number>) => string
  readonly filters: T
  readonly kernels: T
  readonly encoder: Tensor.CompiledProgram
  readonly step: Tensor.CompiledProgram
  readonly suppressed: { readonly always: T; readonly atStart: T }
}

const heads = (x: T) =>
  Effect.flatMap(Tensor.reshape(x, [x.shape[0]!, x.shape[1]!, HEADS, D / HEADS]), (y) => Tensor.transpose(y, [0, 2, 1, 3]))

/** x · W (+ b), with W pre-transposed to [in, out]. */
const linear = (w: Weights, name: string, x: T) =>
  Effect.gen(function*() {
    const y = yield* Tensor.matmul(x, w[`${name}.weight`]!)
    const bias = w[`${name}.bias`]
    return bias ? yield* Tensor.add(y, bias) : y
  })

const keysAndValues = (w: Weights, prefix: string, x: T) =>
  Effect.gen(function*() {
    return [yield* heads(yield* linear(w, `${prefix}.k_proj`, x)), yield* heads(yield* linear(w, `${prefix}.v_proj`, x))] as const
  })

const attend = (w: Weights, prefix: string, query: T, k: T, v: T, causal: boolean) =>
  Effect.gen(function*() {
    const q = yield* heads(yield* linear(w, `${prefix}.q_proj`, query))
    const context = yield* Tensor.scaledDotProductAttention(q, k, v, { causal })
    const merged = yield* Tensor.reshape(yield* Tensor.transpose(context, [0, 2, 1, 3]), [1, query.shape[1]!, D])
    return yield* linear(w, `${prefix}.out_proj`, merged)
  })

const feedForward = (w: Weights, layer: string, x: T) =>
  Effect.gen(function*() {
    return yield* linear(w, `${layer}.fc2`, yield* Tensor.gelu(yield* linear(w, `${layer}.fc1`, x)))
  })

const norm = (w: Weights, name: string, x: T) => Tensor.layerNorm(x, w[`${name}.weight`]!, w[`${name}.bias`]!)

/** The STFT as a convolution: a periodic Hann window times cosine and sine bases, one kernel per bin. */
const stftKernels = () => {
  const bins = N_FFT / 2 + 1
  const kernels = new Float32Array(2 * bins * N_FFT)
  for (let f = 0; f < bins; f++) {
    for (let n = 0; n < N_FFT; n++) {
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT)
      const angle = (2 * Math.PI * f * n) / N_FFT
      kernels[f * N_FFT + n] = window * Math.cos(angle)
      kernels[(bins + f) * N_FFT + n] = -window * Math.sin(angle)
    }
  }
  return Tensor.fromTypedArray(kernels, [2 * bins, 1, N_FFT])
}

/** Log-mel features as WhisperFeatureExtractor computes them: [80, 3000] for up to 30 seconds. */
const logMel = (recognizer: Recognizer, audio: Float32Array) =>
  Effect.gen(function*() {
    // Pad to 30 seconds, then reflect by half a window at each end (center=True).
    const padded = new Float32Array(WINDOW)
    padded.set(audio.subarray(0, WINDOW))
    const half = N_FFT / 2
    const signal = new Float32Array(WINDOW + N_FFT)
    signal.set(padded, half)
    for (let i = 0; i < half; i++) {
      signal[half - 1 - i] = padded[i + 1]!
      signal[half + WINDOW + i] = padded[WINDOW - 2 - i]!
    }
    const bins = N_FFT / 2 + 1
    const spectrum = yield* Tensor.conv1d(yield* Tensor.fromTypedArray(signal, [1, 1, signal.length]), recognizer.kernels, { stride: HOP })
    const real = yield* Tensor.slice(spectrum, { start: [0, 0, 0], end: [1, bins, FRAMES] })
    const imaginary = yield* Tensor.slice(spectrum, { start: [0, bins, 0], end: [1, 2 * bins, FRAMES] })
    const power = yield* Tensor.add(yield* Tensor.square(real), yield* Tensor.square(imaginary))
    const mel = yield* Tensor.matmul(recognizer.filters, yield* Tensor.reshape(power, [bins, FRAMES]))
    const log10 = yield* Tensor.div(
      yield* Tensor.log(yield* Tensor.maximum(mel, yield* Tensor.constantLike(mel, 1e-10))),
      yield* Tensor.constantLike(mel, Math.LN10)
    )
    const [peak] = yield* Tensor.compute([yield* Tensor.max(log10)])
    const floor = (yield* Tensor.toNumberArray(peak!))[0]! - 8
    yield* release([peak])
    const clamped = yield* Tensor.maximum(log10, yield* Tensor.constantLike(log10, floor))
    const [features] = yield* Tensor.compute([
      yield* Tensor.div(yield* Tensor.add(clamped, yield* Tensor.constantLike(clamped, 4)), yield* Tensor.constantLike(clamped, 4))
    ])
    return features!
  })

/** The encoder over [80, 3000] features, and each decoder layer's cross-attention keys and values. */
const encoderGraph = (w: Weights, mel: T) =>
  Effect.gen(function*() {
    const e = (name: string) => w[`model.encoder.${name}`]!
    const input = yield* Tensor.reshape(mel, [1, 80, FRAMES])
    let x = yield* Tensor.gelu(yield* Tensor.add(
      yield* Tensor.conv1d(input, e("conv1.weight"), { padding: 1 }),
      yield* Tensor.reshape(e("conv1.bias"), [1, D, 1])
    ))
    x = yield* Tensor.gelu(yield* Tensor.add(
      yield* Tensor.conv1d(x, e("conv2.weight"), { padding: 1, stride: 2 }),
      yield* Tensor.reshape(e("conv2.bias"), [1, D, 1])
    ))
    x = yield* Tensor.add(yield* Tensor.transpose(x, [0, 2, 1]), yield* Tensor.unsqueeze(e("embed_positions.weight"), 0))
    for (let i = 0; i < LAYERS; i++) {
      const layer = `model.encoder.layers.${i}`
      const normed = yield* norm(w, `${layer}.self_attn_layer_norm`, x)
      const [k, v] = yield* keysAndValues(w, `${layer}.self_attn`, normed)
      x = yield* Tensor.add(x, yield* attend(w, `${layer}.self_attn`, normed, k, v, false))
      x = yield* Tensor.add(x, yield* feedForward(w, layer, yield* norm(w, `${layer}.final_layer_norm`, x)))
    }
    const encoded = yield* norm(w, "model.encoder.layer_norm", x)
    const cross: Array<T> = []
    for (let i = 0; i < LAYERS; i++) cross.push(...(yield* keysAndValues(w, `model.decoder.layers.${i}.encoder_attn`, encoded)))
    return cross
  })

/**
 * One decoder step at fixed shapes: a token at a position (one-hot), against preallocated
 * self-attention caches and the audio's cross-attention keys and values. Returns the next
 * token, chosen on the GPU past an additive suppression mask, and the updated caches.
 */
const stepGraph = (w: Weights, vocabulary: number) =>
  Effect.gen(function*() {
    const d = (name: string) => w[`model.decoder.${name}`]!
    const token = yield* Tensor.makeInput(0, yield* Tensor.fromTypedArray(new Uint32Array(1), [1, 1]))
    const position = yield* Tensor.makeInput(1, yield* Tensor.zeros([1, MAX_TOKENS]))
    const visible = yield* Tensor.makeInput(2, yield* Tensor.zeros([1, 1, 1, MAX_TOKENS]))
    const suppression = yield* Tensor.makeInput(3, yield* Tensor.zeros([1, vocabulary]))
    const cache = yield* Tensor.zeros([1, HEADS, MAX_TOKENS, D / HEADS])
    const audio = yield* Tensor.zeros([1, HEADS, AUDIO_FRAMES, D / HEADS])
    const caches: Array<T> = []
    for (let i = 0; i < 2 * LAYERS; i++) caches.push(yield* Tensor.makeInput(4 + i, cache))
    const cross: Array<T> = []
    for (let i = 0; i < 2 * LAYERS; i++) cross.push(yield* Tensor.makeInput(4 + 2 * LAYERS + i, audio))

    const write = yield* Tensor.reshape(position, [1, 1, MAX_TOKENS, 1])
    const keep = yield* Tensor.sub(yield* Tensor.onesLike(write), write)
    const scale = yield* Tensor.constant(1 / Math.sqrt(D / HEADS))
    let x = yield* Tensor.add(
      yield* Tensor.embedding(token, { weight: d("embed_tokens.weight") }),
      yield* Tensor.unsqueeze(yield* Tensor.matmul(position, d("embed_positions.weight")), 0)
    )
    const updated: Array<T> = []
    for (let i = 0; i < LAYERS; i++) {
      const layer = `model.decoder.layers.${i}`
      const normed = yield* norm(w, `${layer}.self_attn_layer_norm`, x)
      const q = yield* heads(yield* linear(w, `${layer}.self_attn.q_proj`, normed))
      const [kNew, vNew] = yield* keysAndValues(w, `${layer}.self_attn`, normed)
      const k = yield* Tensor.add(yield* Tensor.mul(caches[2 * i]!, keep), yield* Tensor.mul(kNew, write))
      const v = yield* Tensor.add(yield* Tensor.mul(caches[2 * i + 1]!, keep), yield* Tensor.mul(vNew, write))
      updated.push(k, v)
      const scores = yield* Tensor.add(yield* Tensor.mul(yield* Tensor.matmul(q, yield* Tensor.transpose(k, [0, 1, 3, 2])), scale), visible)
      const context = yield* Tensor.matmul(yield* Tensor.softmax(scores), v)
      const merged = yield* Tensor.reshape(yield* Tensor.transpose(context, [0, 2, 1, 3]), [1, 1, D])
      x = yield* Tensor.add(x, yield* linear(w, `${layer}.self_attn.out_proj`, merged))
      const crossed = yield* norm(w, `${layer}.encoder_attn_layer_norm`, x)
      x = yield* Tensor.add(x, yield* attend(w, `${layer}.encoder_attn`, crossed, cross[2 * i]!, cross[2 * i + 1]!, false))
      x = yield* Tensor.add(x, yield* feedForward(w, layer, yield* norm(w, `${layer}.final_layer_norm`, x)))
    }
    x = yield* norm(w, "model.decoder.layer_norm", x)
    // Output projection tied to the token embeddings, laid out [D, vocabulary] for speed.
    const logits = yield* Tensor.matmul(yield* Tensor.reshape(x, [1, D]), d("embed_tokens.weight.transposed"))
    const choice = yield* Tensor.argmax(yield* Tensor.add(logits, suppression), -1)
    return [choice, ...updated]
  })

/** GPT-2 byte-level BPE decoding from vocab.json. */
const byteDecoder = (vocab: Record<string, number>) => {
  const bytes: Array<number> = []
  for (let b = 33; b <= 126; b++) bytes.push(b)
  for (let b = 161; b <= 172; b++) bytes.push(b)
  for (let b = 174; b <= 255; b++) bytes.push(b)
  const chars = [...bytes]
  for (let b = 0, n = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b)
      chars.push(256 + n++)
    }
  }
  const byteOf = new Map(chars.map((c, i) => [String.fromCodePoint(c), bytes[i]!]))
  const tokenText = new Map(Object.entries(vocab).map(([text, id]) => [id, text]))
  return (ids: ReadonlyArray<number>) =>
    new TextDecoder().decode(Uint8Array.from([...ids.map((id) => tokenText.get(id) ?? "").join("")].map((c) => byteOf.get(c) ?? 63)))
}

/** Load Whisper from a Hugging Face checkout of openai/whisper-base.en, and compile its programs. */
export const load = (directory: string) =>
  Effect.gen(function*() {
    const json = (name: string) => JSON.parse(readFileSync(join(directory, name), "utf8"))
    const generation = json("generation_config.json") as Generation
    const stored = yield* Safetensors.load(join(directory, "model.safetensors"))
    // Projections transposed to [in, out] once; lookup tables stay as they are.
    const lookups = new Set(["model.decoder.embed_tokens.weight", "model.decoder.embed_positions.weight", "model.encoder.embed_positions.weight"])
    const weights: Record<string, T> = { ...stored }
    for (const name of Object.keys(stored)) {
      if (!name.endsWith(".weight") || stored[name]!.shape.length !== 2 || lookups.has(name)) continue
      weights[name] = (yield* Tensor.compute([yield* Tensor.transpose(stored[name]!, [1, 0])]))[0]!
      yield* release([stored[name]!])
    }
    weights["model.decoder.embed_tokens.weight.transposed"] = (yield* Tensor.compute([
      yield* Tensor.transpose(stored["model.decoder.embed_tokens.weight"]!, [1, 0])
    ]))[0]!
    const vocabulary = stored["model.decoder.embed_tokens.weight"]!.shape[0]!

    const mel = yield* Tensor.makeInput(0, yield* Tensor.zeros([80, FRAMES]))
    const encoder = yield* Tensor.freezeProgram(yield* encoderGraph(weights, mel), { constantWeights: true })
    const step = yield* Tensor.freezeProgram(yield* stepGraph(weights, vocabulary), { constantWeights: true })
    const mask = (extra: ReadonlyArray<number>) =>
      Effect.gen(function*() {
        const bias = new Float32Array(vocabulary)
        for (const id of [...generation.suppress_tokens, ...extra]) bias[id] = -1e9
        const [tensor] = yield* Tensor.compute([yield* Tensor.fromTypedArray(bias, [1, vocabulary])])
        return tensor!
      })
    const filters = json("preprocessor_config.json").mel_filters as Array<Array<number>>
    const [filtersTensor, kernels] = yield* Tensor.compute([
      yield* Tensor.fromTypedArray(Float32Array.from(filters.flat()), [80, filters[0]!.length]),
      yield* stftKernels()
    ])
    return {
      weights,
      generation,
      decode: byteDecoder(json("vocab.json")),
      filters: filtersTensor!,
      kernels: kernels!,
      encoder,
      step,
      suppressed: { always: yield* mask([]), atStart: yield* mask(generation.begin_suppress_tokens) }
    } satisfies Recognizer
  })

/** Greedy decoding of one window's text tokens. */
const decodeWindow = (recognizer: Recognizer, cross: ReadonlyArray<T>) =>
  Effect.gen(function*() {
    const { generation } = recognizer
    let caches: Array<T> = []
    for (let i = 0; i < 2 * LAYERS; i++) caches.push(yield* Tensor.zeros([1, HEADS, MAX_TOKENS, D / HEADS]))
    const run = (id: number, at: number, suppression: T) =>
      Effect.gen(function*() {
        const position = new Float32Array(MAX_TOKENS)
        position[at] = 1
        const visible = new Float32Array(MAX_TOKENS).fill(-1e9)
        visible.fill(0, 0, at + 1)
        const results = yield* Tensor.runProgram(recognizer.step, [
          yield* Tensor.fromTypedArray(Uint32Array.of(id), [1, 1]),
          yield* Tensor.fromTypedArray(position, [1, MAX_TOKENS]),
          yield* Tensor.fromTypedArray(visible, [1, 1, 1, MAX_TOKENS]),
          suppression,
          ...caches,
          ...cross
        ])
        yield* release(caches)
        caches = results.slice(1)
        const token = Number((yield* Tensor.toTypedArray(results[0]!))[0])
        yield* release([results[0]])
        return token
      })
    yield* run(generation.decoder_start_token_id, 0, recognizer.suppressed.always)
    let next = yield* run(generation.no_timestamps_token_id, 1, recognizer.suppressed.atStart)
    const tokens: Array<number> = []
    while (tokens.length < MAX_NEW_TOKENS && next !== generation.eos_token_id) {
      tokens.push(next)
      next = yield* run(next, tokens.length + 1, recognizer.suppressed.always)
    }
    yield* release(caches)
    return tokens
  })

/**
 * Cross-attention probabilities of the alignment heads for every token of the transcript, over
 * the frames that hold audio: [heads][tokens][frames], softmaxed over those frames only.
 */
const alignmentWeights = (recognizer: Recognizer, cross: ReadonlyArray<T>, tokens: ReadonlyArray<number>, frames: number) =>
  Effect.gen(function*() {
    const w = recognizer.weights
    const d = (name: string) => w[`model.decoder.${name}`]!
    const ids = yield* Tensor.fromTypedArray(Uint32Array.from(tokens), [1, tokens.length])
    const positions = yield* Tensor.slice(d("embed_positions.weight"), { start: [0, 0], end: [tokens.length, D] })
    let x = yield* Tensor.add(yield* Tensor.embedding(ids, { weight: d("embed_tokens.weight") }), yield* Tensor.unsqueeze(positions, 0))
    const scale = yield* Tensor.constant(1 / Math.sqrt(D / HEADS))
    const probabilities: Array<T> = []
    for (let i = 0; i < LAYERS; i++) {
      const layer = `model.decoder.layers.${i}`
      const normed = yield* norm(w, `${layer}.self_attn_layer_norm`, x)
      const [k, v] = yield* keysAndValues(w, `${layer}.self_attn`, normed)
      x = yield* Tensor.add(x, yield* attend(w, `${layer}.self_attn`, normed, k, v, true))
      const crossed = yield* norm(w, `${layer}.encoder_attn_layer_norm`, x)
      const q = yield* heads(yield* linear(w, `${layer}.encoder_attn.q_proj`, crossed))
      for (const [l, h] of recognizer.generation.alignment_heads) {
        if (l !== i) continue
        const qHead = yield* Tensor.slice(q, { start: [0, h, 0, 0], end: [1, h + 1, tokens.length, D / HEADS] })
        const kHead = yield* Tensor.slice(cross[2 * i]!, { start: [0, h, 0, 0], end: [1, h + 1, frames, D / HEADS] })
        probabilities.push(yield* Tensor.softmax(yield* Tensor.mul(yield* Tensor.matmul(qHead, yield* Tensor.transpose(kHead, [0, 1, 3, 2])), scale)))
      }
      x = yield* Tensor.add(x, yield* attend(w, `${layer}.encoder_attn`, crossed, cross[2 * i]!, cross[2 * i + 1]!, false))
      x = yield* Tensor.add(x, yield* feedForward(w, layer, yield* norm(w, `${layer}.final_layer_norm`, x)))
    }
    const computed = yield* Tensor.compute(probabilities)
    const result: Array<Array<Float32Array>> = []
    for (const tensor of computed) {
      const flat = (yield* Tensor.toTypedArray(tensor)) as Float32Array
      result.push(Array.from({ length: tokens.length }, (_, t) => flat.subarray(t * frames, (t + 1) * frames)))
    }
    yield* release(computed)
    return result
  })

/** Median of each sliding window of `width`, with reflected edges (scipy's median_filter). */
const medianFilter = (row: Float32Array, width: number) => {
  const half = Math.floor(width / 2)
  const out = new Float32Array(row.length)
  const at = (i: number) => row[i < 0 ? -i - 1 : i >= row.length ? 2 * row.length - i - 1 : i]!
  const window = new Array<number>(width)
  for (let i = 0; i < row.length; i++) {
    for (let j = 0; j < width; j++) window[j] = at(i - half + j)
    window.sort((a, b) => a - b)
    out[i] = window[half]!
  }
  return out
}

/** OpenAI's dynamic time warping: the cheapest monotonic path through `cost`, as index pairs. */
const dtw = (cost: ReadonlyArray<Float32Array>) => {
  const n = cost.length
  const m = cost[0]!.length
  const total = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Infinity))
  const trace = Array.from({ length: n + 1 }, () => new Int8Array(m + 1).fill(-1))
  total[0]![0] = 0
  for (let j = 1; j <= m; j++) {
    for (let i = 1; i <= n; i++) {
      const c0 = total[i - 1]![j - 1]!
      const c1 = total[i - 1]![j]!
      const c2 = total[i]![j - 1]!
      const [c, t] = c0 < c1 && c0 < c2 ? [c0, 0] : c1 < c0 && c1 < c2 ? [c1, 1] : [c2, 2]
      total[i]![j] = cost[i - 1]![j - 1]! + c
      trace[i]![j] = t
    }
  }
  for (let j = 0; j <= m; j++) trace[0]![j] = 2
  for (let i = 0; i <= n; i++) trace[i]![0] = 1
  const text: Array<number> = []
  const time: Array<number> = []
  let i = n
  let j = m
  while (i > 0 || j > 0) {
    text.push(i - 1)
    time.push(j - 1)
    const t = trace[i]![j]
    if (t === 0) {
      i--
      j--
    } else if (t === 1) i--
    else j--
  }
  return { text: text.reverse(), time: time.reverse() }
}

const PREPEND = `"'“¿([{-`
const APPEND = `"'.。,，!！?？:：”)]}、`
const SENTENCE_ENDS = ".。!！?？"

/**
 * Word timings from the alignment heads, following OpenAI's find_alignment and
 * add_word_timestamps: normalize each head over tokens, median-filter over time, average the
 * heads, align with DTW, split tokens into words, attach punctuation to its neighbours, and trim
 * words that run far longer than usual into a pause.
 */
export const wordTimings = (
  heads: ReadonlyArray<ReadonlyArray<Float32Array>>,
  textTokens: ReadonlyArray<number>,
  eot: number,
  decode: (ids: ReadonlyArray<number>) => string
): Array<Word> => {
  const rows = heads[0]!.length
  const frames = heads[0]![0]!.length
  const normalized = heads.map((head) => {
    const out = head.map((row) => Float32Array.from(row))
    for (let f = 0; f < frames; f++) {
      let mean = 0
      for (let t = 0; t < rows; t++) mean += head[t]![f]!
      mean /= rows
      let variance = 0
      for (let t = 0; t < rows; t++) variance += (head[t]![f]! - mean) ** 2
      const std = Math.sqrt(variance / rows)
      for (let t = 0; t < rows; t++) out[t]![f] = (head[t]![f]! - mean) / std
    }
    return out.map((row) => medianFilter(row, 7))
  })
  // Rows: [start, no-timestamps, text…, end]; the no-timestamps row through the last text row
  // each place the following token.
  const matrix: Array<Float32Array> = []
  for (let t = 1; t < rows - 1; t++) {
    const row = new Float32Array(frames)
    for (const head of normalized) for (let f = 0; f < frames; f++) row[f]! += -head[t]![f]! / normalized.length
    matrix.push(row)
  }
  const path = dtw(matrix)
  const jumpTimes: Array<number> = []
  for (let k = 0; k < path.text.length; k++) {
    if (k === 0 || path.text[k] !== path.text[k - 1]) jumpTimes.push(path.time[k]! / 50)
  }
  // A token starting with a space, or punctuation, starts a word.
  const words: Array<{ text: string; tokens: Array<number> }> = []
  for (const token of [...textTokens, eot]) {
    const piece = token >= eot ? "" : decode([token])
    const punctuation = /^[\p{P}]+$/u.test(piece.trim())
    if (token >= eot || piece.startsWith(" ") || punctuation || words.length === 0) words.push({ text: piece, tokens: [token] })
    else {
      words.at(-1)!.text += piece
      words.at(-1)!.tokens.push(token)
    }
  }
  let boundary = 0
  const timed = words.slice(0, -1).map((word) => {
    const start = jumpTimes[boundary]!
    boundary += word.tokens.length
    return { text: word.text, start, end: jumpTimes[boundary]! }
  })
  for (let i = timed.length - 2; i >= 0; i--) {
    const next = timed[i + 1]!
    if (!next.text.startsWith(" ") && next.text.trim().length > 0 && APPEND.includes(next.text.trim())) {
      timed[i]!.text += next.text
      timed[i]!.end = next.end
      timed.splice(i + 1, 1)
    }
  }
  for (let i = 0; i < timed.length - 1; i++) {
    const word = timed[i]!
    if (word.text.trim().length > 0 && PREPEND.includes(word.text.trim())) {
      timed[i + 1]!.text = word.text.trim() + timed[i + 1]!.text.trimStart()
      timed[i + 1]!.start = word.start
      timed.splice(i--, 1)
    }
  }
  const durations = timed.map((w) => w.end - w.start).filter((d) => d > 0).sort((a, b) => a - b)
  const median = Math.min(0.7, durations.length ? durations[Math.floor(durations.length / 2)]! : 0)
  const longest = median * 2
  for (let i = 1; i < timed.length; i++) {
    const word = timed[i]!
    if (word.end - word.start <= longest) continue
    if (SENTENCE_ENDS.includes(word.text.trim().at(-1) ?? "")) word.end = word.start + longest
    else if (SENTENCE_ENDS.includes(timed[i - 1]!.text.trim().at(-1) ?? "")) word.start = word.end - longest
  }
  const [first, second] = timed
  if (first && first.end > median * 4 && (first.end - first.start > longest || (second && second.end - first.start > longest * 2))) {
    if (second && second.end - second.start > longest) first.end = second.start = Math.max(second.end / 2, second.end - longest)
    first.start = Math.max(0, first.end - longest)
  }
  return timed.map((w) => ({ text: w.text.trim(), start: Math.round(w.start * 100) / 100, end: Math.round(w.end * 100) / 100 }))
}

/** Loudest 10 ms frame's RMS in audio, to skip windows with no speech in them. */
const peakLevel = (audio: Float32Array) => {
  const frame = SAMPLE_RATE / 100
  let peak = 0
  for (let start = 0; start + frame <= audio.length; start += frame) {
    let sum = 0
    for (let i = start; i < start + frame; i++) sum += audio[i]! ** 2
    peak = Math.max(peak, Math.sqrt(sum / frame))
  }
  return peak
}
/** About -50 dBFS: quieter than any speech close to a microphone. */
const SILENCE = 0.003

/**
 * Split audio longer than 30 seconds into windows, each cut at the quietest fifth of a second
 * between 20 and 30 seconds into it, so words are rarely cut in two.
 */
export const windows = (audio: Float32Array) => {
  const starts: Array<number> = []
  let start = 0
  while (audio.length - start > WINDOW) {
    starts.push(start)
    const span = SAMPLE_RATE / 5
    let quietest = start + WINDOW - span
    let lowest = Infinity
    for (let at = start + 20 * SAMPLE_RATE; at + span <= start + WINDOW; at += SAMPLE_RATE / 100) {
      let sum = 0
      for (let i = at; i < at + span; i++) sum += audio[i]! ** 2
      if (sum < lowest) {
        lowest = sum
        quietest = at
      }
    }
    start = quietest + span / 2
  }
  starts.push(start)
  return starts.map((from, i) => ({ start: from, end: starts[i + 1] ?? audio.length }))
}

/** Transcribe 16 kHz mono audio of any length, with each word's start and end in seconds. */
export const transcribe = (recognizer: Recognizer, audio: Float32Array) =>
  Effect.gen(function*() {
    const words: Array<Word> = []
    for (const { end, start } of windows(audio)) {
      const window = audio.subarray(start, end)
      if (peakLevel(window) < SILENCE) continue
      const features = yield* logMel(recognizer, window)
      const cross = yield* Tensor.runProgram(recognizer.encoder, [features])
      yield* release([features])
      const tokens = yield* decodeWindow(recognizer, cross)
      if (tokens.length === 0) {
        yield* release(cross)
        continue
      }
      const { generation } = recognizer
      const frames = Math.min(AUDIO_FRAMES, Math.max(1, Math.floor((window.length / SAMPLE_RATE) * 50)))
      const sequence = [generation.decoder_start_token_id, generation.no_timestamps_token_id, ...tokens, generation.eos_token_id]
      const offset = start / SAMPLE_RATE
      const weights = yield* alignmentWeights(recognizer, cross, sequence, frames)
      yield* release(cross)
      for (const word of wordTimings(weights, tokens, generation.eos_token_id, recognizer.decode)) {
        words.push({ text: word.text, start: Math.round((word.start + offset) * 100) / 100, end: Math.round((word.end + offset) * 100) / 100 })
      }
    }
    return words
  })
