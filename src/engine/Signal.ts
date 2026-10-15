/**
 * Audio arithmetic for the engine: WAV files, resampling, loudness, and trimming. Plain
 * TypeScript over Float32Array, mono throughout.
 */

export interface Audio {
  readonly samples: Float32Array
  readonly rate: number
}

/** Read a PCM (8, 16, 24, or 32-bit) or 32-bit float WAV, mixed down to mono. */
export const readWav = (bytes: Uint8Array): Audio => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const text = (at: number) => String.fromCharCode(...bytes.subarray(at, at + 4))
  if (bytes.length < 12 || text(0) !== "RIFF" || text(8) !== "WAVE") throw new Error("Not a WAV file.")
  let format: { encoding: number; channels: number; rate: number; bits: number } | undefined
  let at = 12
  while (at + 8 <= bytes.length) {
    const id = text(at)
    const size = view.getUint32(at + 4, true)
    const body = at + 8
    if (id === "fmt ") {
      let encoding = view.getUint16(body, true)
      // WAVE_FORMAT_EXTENSIBLE: the real encoding is the subformat's first two bytes.
      if (encoding === 0xfffe && size >= 26) encoding = view.getUint16(body + 24, true)
      format = { encoding, channels: view.getUint16(body + 2, true), rate: view.getUint32(body + 4, true), bits: view.getUint16(body + 14, true) }
    } else if (id === "data") {
      if (!format) throw new Error("The WAV file has no format chunk before its data.")
      const { bits, channels, encoding, rate } = format
      const width = bits / 8
      const frames = Math.floor(Math.min(size, bytes.length - body) / (width * channels))
      const samples = new Float32Array(frames)
      const read = encoding === 3 && bits === 32
        ? (p: number) => view.getFloat32(p, true)
        : encoding === 1 && bits === 16
        ? (p: number) => view.getInt16(p, true) / 32768
        : encoding === 1 && bits === 24
        ? (p: number) => ((view.getUint8(p) | (view.getUint8(p + 1) << 8) | (view.getInt8(p + 2) << 16)) / 8388608)
        : encoding === 1 && bits === 32
        ? (p: number) => view.getInt32(p, true) / 2147483648
        : encoding === 1 && bits === 8
        ? (p: number) => (view.getUint8(p) - 128) / 128
        : undefined
      if (!read) throw new Error(`Unsupported WAV encoding (format ${encoding}, ${bits} bits).`)
      for (let f = 0; f < frames; f++) {
        let sum = 0
        for (let c = 0; c < channels; c++) sum += read(body + (f * channels + c) * width)
        samples[f] = sum / channels
      }
      return { samples, rate }
    }
    at = body + size + (size % 2)
  }
  throw new Error("The WAV file has no audio data.")
}

/** 16-bit PCM mono WAV. */
export const writeWav = ({ rate, samples }: Audio): Uint8Array => {
  const bytes = new Uint8Array(44 + samples.length * 2)
  const view = new DataView(bytes.buffer)
  const text = (at: number, value: string) => [...value].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)))
  text(0, "RIFF")
  view.setUint32(4, 36 + samples.length * 2, true)
  text(8, "WAVE")
  text(12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  text(36, "data")
  view.setUint32(40, samples.length * 2, true)
  samples.forEach((x, i) => view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, x)) * 32767), true))
  return bytes
}

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))

/** The zeroth-order modified Bessel function, for the Kaiser window. */
const besselI0 = (x: number) => {
  let sum = 1
  let term = 1
  for (let k = 1; k < 50; k++) {
    term *= (x / (2 * k)) ** 2
    sum += term
    if (term < sum * 1e-12) break
  }
  return sum
}

/** SciPy's firwin(taps, cutoff, window=("kaiser", beta)): a lowpass filter, unit gain at DC. */
const lowpass = (taps: number, cutoff: number, beta: number) => {
  const alpha = (taps - 1) / 2
  const h = new Float64Array(taps)
  const scale = besselI0(beta)
  for (let n = 0; n < taps; n++) {
    const x = cutoff * (n - alpha)
    const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)
    const r = (2 * n) / (taps - 1) - 1
    h[n] = cutoff * sinc * (besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / scale)
  }
  const sum = h.reduce((a, b) => a + b, 0)
  return h.map((v) => v / sum)
}

/**
 * Change sample rate as SciPy's resample_poly does by default: upsample, filter with a
 * Kaiser-windowed (β = 5) lowpass of 20·max(up, down) + 1 taps, downsample, aligned so the
 * output starts at the same instant as the input.
 */
export const resample = (samples: Float32Array, from: number, to: number): Float32Array => {
  if (from === to) return samples
  const divisor = gcd(from, to)
  const up = to / divisor
  const down = from / divisor
  const larger = Math.max(up, down)
  const halfLength = 10 * larger
  const filter = lowpass(2 * halfLength + 1, 1 / larger, 5.0).map((v) => v * up)
  const outputs = Math.ceil((samples.length * up) / down)
  const prePad = down - (halfLength % down)
  const preRemove = Math.floor((halfLength + prePad) / down)
  const h = new Float64Array(prePad + filter.length)
  h.set(filter, prePad)
  const out = new Float32Array(outputs)
  for (let k = 0; k < outputs; k++) {
    // Output k sits at position (k + preRemove)·down of the zero-stuffed signal.
    const t = (k + preRemove) * down
    let sum = 0
    for (let j = t % up; j < h.length && j <= t; j += up) {
      const input = (t - j) / up
      if (input < samples.length) sum += h[j]! * samples[input]!
    }
    out[k] = sum
  }
  return out
}

/** Loudness in 10 ms frames, and the frame length in samples. */
export const frameEnergy = (samples: Float32Array, rate: number) => {
  const frame = Math.max(1, Math.floor(rate * 0.01))
  const frames = Math.floor(samples.length / frame)
  const energy = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    let sum = 0
    for (let i = f * frame; i < (f + 1) * frame; i++) sum += samples[i]! ** 2
    energy[f] = Math.sqrt(sum / frame)
  }
  return { energy, frame }
}

const peakOf = (values: Float32Array) => values.reduce((m, v) => Math.max(m, v), 0)

/** Below this, a frame is silence: soft consonants like "s" and "sh" stay above it. */
export const quietLevel = (energy: Float32Array) => (energy.length ? Math.max(peakOf(energy) * 0.01, 1e-4) : 1e-4)

/** Cut leading and trailing silence, keeping a natural margin around soft first and last sounds. */
export const trimSilence = (samples: Float32Array, rate: number, before = 0.12, after = 0.1) => {
  const { energy, frame } = frameEnergy(samples, rate)
  if (energy.length === 0) return samples
  const level = quietLevel(energy)
  const first = energy.findIndex((e) => e > level)
  if (first < 0) return samples.subarray(0, 0)
  const last = energy.findLastIndex((e) => e > level)
  const start = Math.max(0, first * frame - Math.floor(before * rate))
  const end = Math.min(samples.length, (last + 1) * frame + Math.floor(after * rate))
  return samples.subarray(start, end)
}

export const decibels = (samples: Float32Array) => {
  if (samples.length === 0) return -180
  let sum = 0
  for (const x of samples) sum += x * x
  return 20 * Math.log10(Math.sqrt(sum / samples.length) + 1e-9)
}

/** Loudness of the speech in a section, ignoring pauses. */
export const speechLevel = (samples: Float32Array, rate: number) => {
  const { energy } = frameEnergy(samples, rate)
  const peak = peakOf(energy)
  if (energy.length === 0 || peak === 0) return -180
  const speech = energy.filter((e) => e > peak * 0.1)
  return 20 * Math.log10(Math.sqrt(speech.reduce((a, e) => a + e * e, 0) / speech.length))
}

// Each section's speech is brought to this loudness, within limits, so sections match.
export const TARGET_DB = -24
export const GAIN_LIMITS_DB = [-6, 12] as const

/** Bring a section's speech toward the target loudness, within the gain limits. */
export const matchLevel = (samples: Float32Array, rate: number) => {
  const gain = Math.min(GAIN_LIMITS_DB[1], Math.max(GAIN_LIMITS_DB[0], TARGET_DB - speechLevel(samples, rate)))
  const factor = 10 ** (gain / 20)
  return samples.map((x) => x * factor)
}

/** How many decibels quieter a section's final words are than its typical word. */
export const endingFade = (samples: Float32Array, rate: number, words: ReadonlyArray<{ start: number; end: number }>, last = 3) => {
  const levels = words.map((w) => decibels(samples.subarray(Math.floor(w.start * rate), Math.floor(w.end * rate))))
  if (levels.length <= last) return 0
  const sorted = [...levels].sort((a, b) => a - b)
  const middle = sorted.length / 2
  const median = sorted.length % 2 ? sorted[Math.floor(middle)]! : (sorted[middle - 1]! + sorted[middle]!) / 2
  return median - levels.slice(-last).reduce((a, b) => a + b, 0) / last
}

/**
 * Cut after the speech that ends near `end` seconds. Recognized word timings are approximate,
 * so cut at the first real silence after it, never inside a final consonant.
 */
export const cutAfter = (samples: Float32Array, rate: number, end: number, search = 0.6, silence = 0.06) => {
  const { energy, frame } = frameEnergy(samples, rate)
  const level = quietLevel(energy)
  const needed = Math.floor(silence * 100)
  const first = Math.floor(end * 100)
  let run = 0
  for (let index = first; index < Math.min(energy.length, first + Math.floor(search * 100)); index++) {
    run = energy[index]! < level ? run + 1 : 0
    if (run >= needed) return samples.subarray(0, (index - needed + 1) * frame + Math.floor(0.05 * rate))
  }
  // No silence before whatever follows: keep a generous margin instead.
  return samples.subarray(0, Math.min(samples.length, Math.floor((end + 0.3) * rate)))
}

/** Join sections, each trimmed to its speech and level-matched, with its planned pause after it. */
export const joinSections = (sections: ReadonlyArray<{ readonly samples: Float32Array; readonly pause: number }>, rate: number) => {
  const parts = sections.flatMap(({ pause, samples }) => [matchLevel(trimSilence(samples, rate), rate), new Float32Array(Math.floor(rate * pause))])
  const total = parts.reduce((n, p) => n + p.length, 0)
  if (parts.every((p, i) => i % 2 === 1 || p.length === 0)) throw new Error("No audio was generated")
  const joined = new Float32Array(total)
  let at = 0
  for (const part of parts) {
    joined.set(part, at)
    at += part.length
  }
  if (!joined.every(Number.isFinite)) throw new Error("The model produced invalid audio. Please try again.")
  const peak = joined.reduce((m, x) => Math.max(m, Math.abs(x)), 0)
  return peak > 0.98 ? joined.map((x) => x * (0.98 / peak)) : joined
}
