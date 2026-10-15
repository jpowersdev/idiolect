/**
 * Speaking as you: VoxCPM2 continues from a clip of your voice, which is also its cloning
 * reference, and its codec turns the generated latents into 48 kHz audio. The same steps as
 * VoxCPM's own generate, with its defaults: ten sampler steps, guidance 2, and a retry when the
 * model runs on far past the text.
 */
import { Safetensors, Tensor } from "@effect-torch/core"
import * as Tokenizers from "@effect-torch/tokenizers"
import { Effect } from "effect"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import * as Codec from "./Codec.ts"
import type { T, Weights } from "./MiniCpm.ts"
import { foldWeightNorm, readStateDict } from "./Pth.ts"
import { resample } from "./Signal.ts"
import * as VoxCpm from "./VoxCpm.ts"

const PATCH = 4
const LATENT = 64
/** Codec input samples per patch, and output samples per patch. */
const PATCH_INPUT = PATCH * Codec.ENCODE_HOP
const PATCH_OUTPUT = PATCH * Codec.DECODE_HOP
/** Prompt patches decoded ahead of the new ones, so the codec starts from your voice. */
const CONTEXT_PATCHES = 3
/** Generated patches per text token beyond which the model has run on: rendered again. */
const RUNAWAY_RATIO = 6
const RUNAWAY_ATTEMPTS = 3

export const OUTPUT_RATE = Codec.OUTPUT_RATE

export interface Speech {
  readonly model: VoxCpm.Model
  readonly codec: Weights
  readonly tokenizer: Tokenizers.Tokenizer
}

/** A clip of your voice, encoded once for every section it starts. */
export interface Voice {
  readonly text: string
  /** [patches, 4, 64], padded at the end: the cloning reference. */
  readonly reference: Float32Array
  /** [patches, 4, 64], padded at the start, so its last patch ends where the clip does. */
  readonly continuation: Float32Array
}

/** Load VoxCPM2 from a checkout of openbmb/VoxCPM2, its language models at `dtype`. */
export const load = (directory: string, dtype: Tensor.DType) =>
  Effect.gen(function*() {
    const config = JSON.parse(readFileSync(join(directory, "config.json"), "utf8"))
    const model = yield* VoxCpm.load(yield* Safetensors.load(join(directory, "model.safetensors")), {
      dtype,
      rope: { theta: config.lm_config.rope_theta, factors: config.lm_config.rope_scaling.short_factor }
    })
    const stored = foldWeightNorm(readStateDict(readFileSync(join(directory, "audiovae.pth"))))
    const tensors: Record<string, T> = {}
    // Its one integer tensor, the output-rate buckets, is fixed in Codec.ts.
    for (const [name, { data, shape }] of stored) if (data instanceof Float32Array) tensors[name] = yield* Tensor.fromTypedArray(data, shape)
    const codec = yield* Codec.prepare(tensors)
    const tokenizer = yield* Tokenizers.fromFile(join(directory, "tokenizer.json"), {
      padding: { _tag: "None" },
      truncation: { _tag: "None" },
      specialTokens: "Always"
    })
    return { model, codec, tokenizer } satisfies Speech
  })

const tokens = (speech: Speech, text: string) =>
  Effect.map(speech.tokenizer.encode(text, { addSpecialTokens: false }), (ids) => Array.from(ids.data))

/** 16 kHz audio, padded to whole patches, to latent patches [patches, 4, 64]. */
const encodeClip = (speech: Speech, audio: Float32Array, padding: "start" | "end") =>
  Effect.gen(function*() {
    const padded = new Float32Array(Math.ceil(audio.length / PATCH_INPUT) * PATCH_INPUT)
    padded.set(audio, padding === "start" ? padded.length - audio.length : 0)
    const [latents] = yield* Tensor.compute([yield* Codec.encode(speech.codec, yield* Tensor.fromTypedArray(padded, [1, 1, padded.length]))])
    const channelMajor = (yield* Tensor.toTypedArray(latents!)) as Float32Array
    const frames = latents!.shape[2]!
    const patches = new Float32Array(frames * LATENT)
    for (let frame = 0; frame < frames; frame++) {
      for (let c = 0; c < LATENT; c++) patches[frame * LATENT + c] = channelMajor[c * frames + frame]!
    }
    return patches
  })

/** Encode a clip of your voice and its exact words. */
export const voice = (speech: Speech, clip: Float32Array, rate: number, text: string) =>
  Effect.gen(function*() {
    const audio = resample(clip, rate, Codec.INPUT_RATE)
    return {
      text,
      reference: yield* encodeClip(speech, audio, "end"),
      continuation: yield* encodeClip(speech, audio, "start")
    } satisfies Voice
  })

/** Latent patches [patches, 4, 64] to 48 kHz audio. */
const decode = (speech: Speech, patches: Float32Array) =>
  Effect.gen(function*() {
    const frames = patches.length / LATENT
    const channelMajor = new Float32Array(patches.length)
    for (let frame = 0; frame < frames; frame++) {
      for (let c = 0; c < LATENT; c++) channelMajor[c * frames + frame] = patches[frame * LATENT + c]!
    }
    const [audio] = yield* Tensor.compute([yield* Codec.decode(speech.codec, yield* Tensor.fromTypedArray(channelMajor, [1, LATENT, frames]))])
    return (yield* Tensor.toTypedArray(audio!)) as Float32Array
  })

export interface SpeakOptions {
  readonly seed: number
  /** Checked between patches; speaking stops early when it returns true. */
  readonly cancelled?: () => boolean
}

/** Say `text` in your voice, continuing from the clip: 48 kHz audio. */
export const speak = (speech: Speech, from: Voice, text: string, options: SpeakOptions) =>
  Effect.gen(function*() {
    const target = (yield* tokens(speech, text)).length
    const prompt = VoxCpm.continuationPrompt(yield* tokens(speech, `${from.text} ${text}`), from.reference, from.continuation)
    const runaway = target * RUNAWAY_RATIO
    let patches: Array<Float32Array> = []
    for (let attempt = 0; attempt < RUNAWAY_ATTEMPTS; attempt++) {
      patches = yield* VoxCpm.generate(speech.model, prompt, {
        minLength: 2,
        maxLength: runaway + 10,
        seed: options.seed + 7919 * attempt,
        ...(options.cancelled ? { cancelled: options.cancelled } : {})
      })
      if (patches.length < runaway || options.cancelled?.()) break
    }
    const context = Math.min(CONTEXT_PATCHES, from.continuation.length / (PATCH * LATENT))
    const latents = new Float32Array((context + patches.length) * PATCH * LATENT)
    latents.set(from.continuation.subarray(from.continuation.length - context * PATCH * LATENT))
    patches.forEach((patch, i) => latents.set(patch, (context + i) * PATCH * LATENT))
    return (yield* decode(speech, latents)).slice(context * PATCH_OUTPUT)
  })
