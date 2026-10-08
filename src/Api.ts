/**
 * The HTTP contract between the CLI and a local speech engine.
 *
 * The Python engine in engine/ implements it today. Any engine that serves
 * these endpoints on 127.0.0.1, and reports `api: ApiVersion`, can replace it.
 */
import { Schema } from "effect"
import { HttpApi, HttpApiEndpoint, HttpApiGroup, HttpApiSchema } from "effect/http-api"

export const ApiVersion = 1

export const Languages = [
  "English",
  "Chinese",
  "Japanese",
  "Korean",
  "German",
  "French",
  "Russian",
  "Portuguese",
  "Spanish",
  "Italian"
] as const

export const Language = Schema.Literals(Languages)
export type Language = typeof Language.Type

export const Status = Schema.Struct({
  service: Schema.Literal("idiolect"),
  api: Schema.Int,
  pid: Schema.Int,
  data_directory: Schema.String,
  model: Schema.String,
  model_downloaded: Schema.Boolean,
  recognizer_downloaded: Schema.Boolean,
  model_status: Schema.String,
  languages: Schema.Array(Schema.String)
})
export type Status = typeof Status.Type

export const Voice = Schema.Struct({
  id: Schema.String,
  name: Schema.String,
  transcript: Schema.String,
  duration: Schema.Number,
  created_at: Schema.String
})
export type Voice = typeof Voice.Type

export const RenderStatus = Schema.Literals([
  "queued",
  "loading",
  "generating",
  "ready",
  "failed",
  "cancelled"
])
export type RenderStatus = typeof RenderStatus.Type
export const activeStatuses: ReadonlySet<RenderStatus> = new Set(["queued", "loading", "generating"])

export const Render = Schema.Struct({
  id: Schema.String,
  voice_id: Schema.String,
  voice_name: Schema.String,
  text: Schema.String,
  language: Schema.String,
  seed: Schema.Int,
  completed: Schema.Int,
  total: Schema.Int,
  status: RenderStatus,
  message: Schema.String,
  duration: Schema.NullOr(Schema.Number),
  elapsed: Schema.NullOr(Schema.Number)
})
export type Render = typeof Render.Type

export const Word = Schema.Struct({
  text: Schema.String,
  start: Schema.Number,
  end: Schema.Number
})
export type Word = typeof Word.Type

/** Every engine error is a status code and a message for the person at the terminal. */
const failure = <const Id extends string>(id: Id, status: number) =>
  Schema.Struct({ detail: Schema.String }).pipe(
    HttpApiSchema.status(status),
    (schema) => schema.annotate({ identifier: id })
  )

export const NotFound = failure("NotFound", 404)
export const Conflict = failure("Conflict", 409)
export const TooLarge = failure("TooLarge", 413)
export const Invalid = failure("Invalid", 422)
export const Busy = failure("Busy", 429)
export type EngineFailure = { readonly detail: string }

const RenderId = { id: Schema.String }

export class EngineGroup extends HttpApiGroup.make("engine")
  .add(HttpApiEndpoint.get("status", "/api/status", { success: Status }))
  .add(
    HttpApiEndpoint.post("shutdown", "/api/shutdown", {
      success: Schema.Struct({ stopping: Schema.Boolean }).pipe(HttpApiSchema.status(202)),
      error: Conflict
    })
  )
  .add(
    HttpApiEndpoint.post("transcribe", "/api/transcribe", {
      query: { sample_rate: Schema.Int },
      // 16-bit little-endian mono PCM.
      payload: Schema.Uint8Array.pipe(
        HttpApiSchema.asUint8Array({ contentType: "application/octet-stream" })
      ),
      success: Schema.Struct({ words: Schema.Array(Word) }),
      error: [Conflict, TooLarge, Invalid]
    })
  )
  .add(HttpApiEndpoint.get("voices", "/api/voices", { success: Schema.Array(Voice) }))
  .add(
    HttpApiEndpoint.post("addVoice", "/api/voices", {
      // Fields: name, transcript, and audio (a WAV file).
      payload: Schema.instanceOf(FormData).pipe(HttpApiSchema.asMultipart()),
      success: Voice.pipe(HttpApiSchema.status(201)),
      error: Invalid
    })
  )
  .add(
    HttpApiEndpoint.post("generate", "/api/generate", {
      payload: Schema.Struct({
        voice_id: Schema.String,
        text: Schema.String,
        language: Language,
        seed: Schema.Int
      }),
      success: Render.pipe(HttpApiSchema.status(202)),
      error: [NotFound, Conflict, Invalid, Busy]
    })
  )
  .add(HttpApiEndpoint.get("renders", "/api/renders", { success: Schema.Array(Render) }))
  .add(
    HttpApiEndpoint.get("render", "/api/renders/:id", {
      params: RenderId,
      success: Render,
      error: NotFound
    })
  )
  .add(
    HttpApiEndpoint.post("cancel", "/api/renders/:id/cancel", {
      params: RenderId,
      success: Render,
      error: NotFound
    })
  )
  .add(
    HttpApiEndpoint.get("audio", "/api/renders/:id/audio", {
      params: RenderId,
      success: Schema.Uint8Array.pipe(HttpApiSchema.asUint8Array({ contentType: "audio/wav" })),
      error: [NotFound, Conflict]
    })
  )
{}

export class EngineApi extends HttpApi.make("idiolect").add(EngineGroup) {}
