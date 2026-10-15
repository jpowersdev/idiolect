/**
 * Renders: a queue of scripts spoken one at a time, each split into sections that are rendered,
 * checked by ear with the recognizer, and joined. They live in memory for one engine run.
 */
import { Effect, Queue } from "effect"
import { activeStatuses, type Render, type RenderStatus, type Word } from "../Api.ts"
import { coverage, type Section, speechEnd, splitScript } from "./Sections.ts"
import { cutAfter, endingFade, joinSections, writeWav } from "./Signal.ts"

/** Speaking and listening, as the models (or stand-ins for them) do it. */
export interface Voices {
  /** Say a section in your voice: mono audio at `rate`. */
  readonly speak: (text: string, seed: number, cancelled: () => boolean) => Effect.Effect<Float32Array, Error>
  readonly rate: number
  readonly transcribe: (audio: Float32Array, rate: number) => Effect.Effect<ReadonlyArray<Word>, Error>
  /** Called before the first section, so loading shows as its own step. */
  readonly load: Effect.Effect<void, Error>
}

// A section is rendered again, with another seed, when it sounds garbled or fades out.
export const ATTEMPTS = 3
export const REQUIRED_COVERAGE = 0.9
// Final words this many decibels quieter than the section's typical word: it faded out.
export const FADE_DB = 12
const SEED = 12
/** Finished renders kept for their audio; older ones are forgotten. */
export const KEEP = 8
export const MAX_ACTIVE = 8

interface Job {
  render: Render
  readonly sections: ReadonlyArray<Section>
  cancelled: boolean
  audio?: Uint8Array
}

/**
 * Render one section and check it with the recognizer: keep the first attempt that says every
 * word through the end without fading out, cut where its final words end, or else the best one.
 */
export const renderSection = (
  voices: Voices,
  text: string,
  index: number,
  options: { readonly cancelled: () => boolean; readonly attempt: (n: number) => void }
) =>
  Effect.gen(function*() {
    let best: { score: number; samples: Float32Array } | undefined
    for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
      options.attempt(attempt)
      let samples = yield* voices.speak(text, SEED + index + 1000 * attempt, options.cancelled)
      const words = yield* voices.transcribe(samples, voices.rate)
      const end = speechEnd(text, words)
      const faded = endingFade(samples, voices.rate, words) > FADE_DB
      const score = coverage(text, words) + (end !== undefined ? 1 : 0) - (faded ? 1 : 0)
      // Cut the syllables the model sometimes adds after the text.
      if (end !== undefined) samples = cutAfter(samples, voices.rate, end)
      if (best === undefined || score > best.score) best = { score, samples }
      if ((end !== undefined && !faded && score - 1 >= REQUIRED_COVERAGE) || options.cancelled()) break
      yield* Effect.logInfo(`Section ${index + 1} ${faded ? "faded out" : "sounded garbled"} (score ${score.toFixed(2)}); rendering again`)
    }
    return best!.samples
  })

export const make = (voices: Voices) =>
  Effect.gen(function*() {
    const jobs = new Map<string, Job>()
    const queue = yield* Queue.unbounded<Job>()
    const active = () => [...jobs.values()].filter((job) => activeStatuses.has(job.render.status)).length

    const update = (job: Job, changes: Partial<Render>) => {
      job.render = { ...job.render, ...changes }
    }
    const finish = (job: Job, status: RenderStatus, message: string, changes: Partial<Render> = {}) =>
      update(job, { status, message, ...changes })

    const prune = () => {
      const finished = [...jobs.values()].filter((job) => !activeStatuses.has(job.render.status))
      for (const job of finished.slice(0, Math.max(0, finished.length - KEEP))) jobs.delete(job.render.id)
    }

    const run = (job: Job) =>
      Effect.gen(function*() {
        if (job.cancelled) return finish(job, "cancelled", "Cancelled")
        update(job, { status: "loading", message: "Loading the voice model…" })
        yield* voices.load
        const sections: Array<{ samples: Float32Array; pause: number }> = []
        for (const [index, section] of job.sections.entries()) {
          if (job.cancelled) break
          const samples = yield* renderSection(voices, section.text, index, {
            cancelled: () => job.cancelled,
            attempt: (attempt) =>
              update(job, {
                status: "generating",
                message: `Rendering section ${index + 1} of ${job.sections.length}${attempt ? " again" : ""}…`
              })
          })
          sections.push({ samples, pause: section.pause })
          update(job, { completed: index + 1 })
        }
        if (job.cancelled) return finish(job, "cancelled", "Cancelled")
        const joined = joinSections(sections, voices.rate)
        job.audio = writeWav({ samples: joined, rate: voices.rate })
        finish(job, "ready", "Your voiceover is ready", { duration: Math.round((joined.length / voices.rate) * 100) / 100 })
      }).pipe(
        Effect.catchCause((cause) =>
          Effect.gen(function*() {
            yield* Effect.logError(`Render ${job.render.id} failed`, cause)
            const error = cause.reasons.find((reason) => reason._tag === "Fail")?.error
            const message = error instanceof Error ? error.message : "The render failed. See engine.log."
            finish(job, "failed", message.slice(0, 700))
          })
        )
      )

    // One render at a time: the GPU holds one model, and sections finish in order.
    yield* Effect.forkScoped(Effect.forever(Effect.flatMap(Queue.take(queue), run)))

    return {
      active,
      submit: (text: string) =>
        Effect.gen(function*() {
          const sections = splitScript(text)
          const job: Job = {
            render: {
              id: crypto.randomUUID().replaceAll("-", ""),
              completed: 0,
              total: sections.length,
              status: "queued",
              message: "Waiting for the speech engine",
              duration: null
            },
            sections,
            cancelled: false
          }
          jobs.set(job.render.id, job)
          prune()
          yield* Queue.offer(queue, job)
          return job.render
        }),
      get: (id: string) => jobs.get(id)?.render,
      audio: (id: string) => jobs.get(id)?.audio,
      cancel: (id: string) => {
        const job = jobs.get(id)
        if (job === undefined) return undefined
        if (activeStatuses.has(job.render.status) && !job.cancelled) {
          job.cancelled = true
          // A queued render never started: it is cancelled at once.
          if (job.render.status === "queued") finish(job, "cancelled", "Cancelled")
          else update(job, { message: "Stopping after the current section…" })
        }
        return job.render
      }
    }
  })

export type Renders = Effect.Success<ReturnType<typeof make>>
