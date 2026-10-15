import { afterEach, expect, test } from "bun:test"
import { Effect } from "effect"
import { existsSync, readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import type { Word } from "../../src/Api.ts"
import * as Renders from "../../src/engine/Renders.ts"
import * as Server from "../../src/engine/Server.ts"
import { readWav, writeWav } from "../../src/engine/Signal.ts"
import { standInModels, startEngine, temporaryDirectory } from "../support.ts"

const running: Array<{ stop: () => Promise<unknown> }> = []
afterEach(async () => {
  await Promise.all(running.splice(0).map((engine) => engine.stop()))
})

const engine = async (options: Parameters<typeof startEngine>[1] = {}) => {
  const data = temporaryDirectory()
  const served = await startEngine(data, options)
  running.push(served)
  const request = (path: string, init: RequestInit = {}) => fetch(`http://localhost${path}`, { ...init, unix: served.socket } as RequestInit)
  const json = (path: string, body: unknown) =>
    request(path, { method: "POST", body: JSON.stringify(body), headers: { "content-type": "application/json" } })
  return { data, served, request, json }
}

const wave = (seconds: number, options: { silent?: boolean; rate?: number } = {}) => {
  const rate = options.rate ?? 24000
  const samples = Float32Array.from({ length: Math.round(seconds * rate) }, (_, i) => options.silent ? 0 : 0.2 * Math.sin((2 * Math.PI * 220 * i) / rate))
  return writeWav({ samples, rate })
}

/** A word every half second; every `sentence`th word, and the last, ends a sentence. */
const timedWords = (seconds: number, sentence = 6): Array<Word> => {
  const count = seconds * 2
  return Array.from({ length: count }, (_, index) => ({
    text: `word${index}${index % sentence === sentence - 1 || index === count - 1 ? "." : ""}`,
    start: index * 0.5,
    end: index * 0.5 + 0.4
  }))
}

const take = (script = "comma-1", seconds = 20, changes: Record<string, unknown> = {}) => {
  const words = timedWords(seconds)
  return { script, transcript: words.map((w) => w.text).join(" "), words, ...changes }
}

const upload = (
  request: (path: string, init?: RequestInit) => Promise<Response>,
  takes: ReadonlyArray<unknown> = [take()],
  audio: ReadonlyArray<Uint8Array> = takes.map(() => wave(20, { rate: 48000 }))
) => {
  const form = new FormData()
  form.append("takes", JSON.stringify(takes))
  audio.forEach((bytes, i) => form.append("audio", new Blob([bytes as Uint8Array<ArrayBuffer>], { type: "audio/wav" }), `take-${i}.wav`))
  return request("/api/voice", { method: "PUT", body: form })
}

test("takes are kept at their recorded rate and cut into 24 kHz clips that end at sentences", async () => {
  const { data, request } = await engine()
  expect((await request("/api/voice")).status).toBe(404)
  const takes = [take(), take("technical", 30, { notice: "Copyright notice." })]
  const response = await upload(request, takes, [wave(20, { rate: 48000 }), wave(30, { rate: 48000 })])
  expect(response.status).toBe(200)
  const summary = await response.json()
  expect(summary.takes).toEqual([{ script: "comma-1", duration: 20 }, { script: "technical", duration: 30 }])
  expect(await (await request("/api/voice")).json()).toEqual(summary)
  const folder = join(data, "voice")
  const saved = JSON.parse(readFileSync(join(folder, "voice.json"), "utf8"))
  const whole = readWav(readFileSync(join(folder, "takes/comma-1.wav")))
  expect(whole.rate).toBe(48000)
  expect(whole.samples.length).toBe(20 * 48000)
  expect(saved.takes[1].notice).toBe("Copyright notice.")
  expect(saved.takes[0].words).toEqual(takes[0]!.words)
  expect(new Set(saved.clips.map((clip: { script: string }) => clip.script))).toEqual(new Set(["comma-1", "technical"]))
  for (const clip of saved.clips) {
    const audio = readWav(readFileSync(join(folder, clip.file)))
    expect(audio.rate).toBe(24000)
    expect(audio.samples.length / audio.rate).toBeGreaterThanOrEqual(3)
    expect(audio.samples.length / audio.rate).toBeLessThanOrEqual(15.5)
    expect(clip.text.endsWith(".")).toBe(true)
  }
  expect(readdirSync(folder).sort()).toEqual(["clips", "takes", "voice.json"])
})

test("a new recording replaces the voice, leaving nothing else behind", async () => {
  const { data, request } = await engine()
  await upload(request, [take("comma-1")])
  await upload(request, [take("technical")])
  expect((await (await request("/api/voice")).json()).takes.map((t: { script: string }) => t.script)).toEqual(["technical"])
  expect(readdirSync(data).filter((name) => name.startsWith("voice"))).toEqual(["voice"])
})

test("bad recordings keep the previous voice", async () => {
  const { data, request } = await engine()
  const previous = await (await upload(request)).json()
  for (const audio of [wave(1), wave(91), wave(20, { silent: true }), new TextEncoder().encode("not audio")]) {
    expect((await upload(request, [take(), take("technical")], [wave(20), audio])).status).toBe(422)
    expect(await (await request("/api/voice")).json()).toEqual(previous)
  }
  expect(readdirSync(data).filter((name) => name.startsWith("voice"))).toEqual(["voice"])
})

test("invalid takes are rejected", async () => {
  const { request } = await engine()
  for (const takes of [[], [take(), take()], [take("comma-1", 20, { transcript: "   " })], [take("../escape")], [take("comma-1", 20, { words: [] })]]) {
    expect((await upload(request, takes, takes.map(() => wave(20)))).status).toBe(422)
  }
  expect((await upload(request, [take(), take("technical")], [wave(20)])).status).toBe(422)
  expect((await request("/api/voice")).status).toBe(404)
})

test("a render needs a voice and a script, and can be followed and cancelled", async () => {
  const { json, request } = await engine()
  expect((await json("/api/renders", { text: "Hello." })).status).toBe(404)
  await upload(request)
  for (const text of ["   ", "x".repeat(12001)]) {
    const response = await json("/api/renders", { text })
    expect(response.status).toBe(422)
    expect((await response.json()).detail).toContain("text")
  }
  const response = await json("/api/renders", { text: "A proper script." })
  expect(response.status).toBe(202)
  const job = await response.json()
  expect(job).toMatchObject({ status: "queued", total: 1 })
  expect((await (await request(`/api/renders/${job.id}`)).json()).id).toBe(job.id)
  expect((await request(`/api/renders/${job.id}/audio`)).status).toBe(409)
  expect((await request("/api/shutdown", { method: "POST" })).status).toBe(409)
  await request(`/api/renders/${job.id}/cancel`, { method: "POST" })
  for (let attempt = 0; attempt < 100; attempt++) {
    if ((await (await request(`/api/renders/${job.id}`)).json()).status === "cancelled") break
    await Bun.sleep(20)
  }
  expect((await (await request(`/api/renders/${job.id}`)).json()).status).toBe("cancelled")
  expect((await request("/api/renders/missing")).status).toBe(404)
  expect((await request("/api/shutdown", { method: "POST" })).status).toBe(202)
})

test("a finished render's audio is every section joined, with its pauses", async () => {
  const { json, request } = await engine({
    local: standInModels({
      speak: () => Effect.succeed(new Float32Array(24000).fill(0.2)),
      transcribe: () => Effect.succeed([{ text: "Hi.", start: 0, end: 1 }])
    })
  })
  await upload(request)
  const job = await (await json("/api/renders", { text: "Hi.\n\nHi." })).json()
  let render = job
  for (let attempt = 0; attempt < 100 && render.status !== "ready"; attempt++) {
    await Bun.sleep(20)
    render = await (await request(`/api/renders/${job.id}`)).json()
  }
  expect(render).toMatchObject({ status: "ready", completed: 2, total: 2 })
  const audio = readWav(new Uint8Array(await (await request(`/api/renders/${job.id}/audio`)).arrayBuffer()))
  expect(audio.rate).toBe(24000)
  // Two seconds of speech and the pause between the paragraphs.
  expect(audio.samples.length / audio.rate).toBeCloseTo(2.75, 1)
  expect(render.duration).toBeCloseTo(2.75, 1)
})

test("transcription takes raw 16-bit PCM at any rate", async () => {
  const heard: Array<{ audio: Float32Array; rate: number }> = []
  const { request } = await engine({
    local: standInModels({ transcribe: (audio, rate) => Effect.sync(() => heard.push({ audio, rate })).pipe(Effect.as([])) })
  })
  const pcm = new Uint8Array(new Int16Array(48000).fill(8192).buffer)
  const response = await request("/api/transcribe?sample_rate=48000", { method: "POST", body: pcm, headers: { "content-type": "application/octet-stream" } })
  expect(await response.json()).toEqual({ words: [] })
  expect(heard[0]!.rate).toBe(48000)
  expect(heard[0]!.audio.length).toBe(48000)
  expect(heard[0]!.audio.every((x) => x === 0.25)).toBe(true)
  expect((await request("/api/transcribe?sample_rate=100", { method: "POST", body: pcm, headers: { "content-type": "application/octet-stream" } })).status).toBe(422)
})

test("errors are a plain message", async () => {
  const { request } = await engine()
  const body = await (await request("/api/renders/missing")).json()
  expect(typeof body.detail).toBe("string")
})

test("the status reports the contract and the models", async () => {
  const { request } = await engine()
  expect(await (await request("/api/status")).json()).toMatchObject({
    service: "idiolect",
    api: 2,
    pid: process.pid,
    model: "voxcpm2",
    model_downloaded: true,
    recognizer_downloaded: true,
    model_status: "idle"
  })
})

test("two engines cannot use the same data, but a stale lock is taken over", async () => {
  const { data } = await engine()
  await expect(Effect.runPromise(Effect.scoped(Server.lock(data)))).rejects.toThrow("already running")
  const other = temporaryDirectory()
  await Bun.write(join(other, "engine.lock"), "999999\n")
  await Effect.runPromise(Effect.scoped(Server.lock(other)))
  expect(existsSync(join(other, "engine.lock"))).toBe(false)
})

test("an idle engine exits and removes its socket", async () => {
  const { data, served } = await engine({ idleTimeout: 50 })
  expect(await served.exited).toBe(true)
  expect(existsSync(Server.socketPath(data))).toBe(false)
  expect(existsSync(join(data, "engine.lock"))).toBe(false)
})

test("garbled sections are rendered again, and cut after their final words", async () => {
  let spoken = 0
  let heard = 0
  const rate = 24000
  const voices = {
    ...standInModels(),
    rate,
    // Speech, a short gap, then a stray syllable.
    speak: () =>
      Effect.sync(() => {
        const level = 0.1 * ++spoken
        const audio = new Float32Array(14400 + 4800 + 4800).fill(level)
        audio.fill(0, 14400, 19200)
        return audio
      }),
    // Garbling on the first attempt, then the whole text and a trailing syllable.
    transcribe: () =>
      Effect.sync(() =>
        ++heard === 1
          ? [{ text: "A completely nyuh", start: 0, end: 0.8 }]
          : [{ text: "A completely new story.", start: 0, end: 0.6 }, { text: "Sorry.", start: 0.6, end: 0.9 }]
      )
  }
  const audio = await Effect.runPromise(
    Renders.renderSection(voices, "A completely new story.", 0, { cancelled: () => false, attempt: () => {} })
  )
  expect(spoken).toBe(2)
  expect(audio.subarray(0, 14400).every((x) => Math.abs(x - 0.2) < 1e-6)).toBe(true)
  expect(audio.length).toBe(Math.floor(0.65 * rate))
})
