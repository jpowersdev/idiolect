/**
 * The HTTP contract between the CLI and its local speech engine (src/engine/), served on the
 * data directory's engine.sock. The engine runs in its own process so the models stay loaded
 * between commands.
 */
import { Schema } from "effect"
import { HttpApi, HttpApiEndpoint, HttpApiGroup, HttpApiSchema } from "effect/http-api"

export const ApiVersion = 2
/** The longest script one render takes, in characters. */
export const MaxScriptLength = 12000

export const Status = Schema.Struct({
  service: Schema.Literal("idiolect"),
  api: Schema.Int,
  /** The engine's release, which must match the CLI's. */
  version: Schema.String,
  pid: Schema.Int,
  /** The speech model. */
  model: Schema.String,
  model_downloaded: Schema.Boolean,
  recognizer_downloaded: Schema.Boolean,
  model_status: Schema.String
})
export type Status = typeof Status.Type

/** Your voice: one recorded take per setup script. */
export const Voice = Schema.Struct({
  created_at: Schema.String,
  duration: Schema.Number,
  takes: Schema.Array(Schema.Struct({ script: Schema.String, duration: Schema.Number }))
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
  completed: Schema.Int,
  total: Schema.Int,
  status: RenderStatus,
  message: Schema.String,
  duration: Schema.NullOr(Schema.Number)
})
export type Render = typeof Render.Type

export const Word = Schema.Struct({
  text: Schema.String,
  start: Schema.Number,
  end: Schema.Number
})
export type Word = typeof Word.Type

// Every engine error is a status code and a message for the person at the terminal.
export class NotFound extends Schema.TaggedError<NotFound>()("NotFound", { detail: Schema.String }, { httpApiStatus: 404 }) {}
export class Conflict extends Schema.TaggedError<Conflict>()("Conflict", { detail: Schema.String }, { httpApiStatus: 409 }) {}
export class TooLarge extends Schema.TaggedError<TooLarge>()("TooLarge", { detail: Schema.String }, { httpApiStatus: 413 }) {}
export class Invalid extends Schema.TaggedError<Invalid>()("Invalid", { detail: Schema.String }, { httpApiStatus: 422 }) {}
export class Busy extends Schema.TaggedError<Busy>()("Busy", { detail: Schema.String }, { httpApiStatus: 429 }) {}

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
  .add(HttpApiEndpoint.get("voice", "/api/voice", { success: Voice, error: NotFound }))
  .add(
    HttpApiEndpoint.put("replaceVoice", "/api/voice", {
      // Fields: takes, a JSON array of { script, transcript, words, notice? },
      // and one audio WAV file per take, in the same order.
      payload: Schema.instanceOf(FormData).pipe(HttpApiSchema.asMultipart()),
      success: Voice,
      error: Invalid
    })
  )
  .add(
    HttpApiEndpoint.post("generate", "/api/renders", {
      payload: Schema.Struct({ text: Schema.String }),
      success: Render.pipe(HttpApiSchema.status(202)),
      error: [NotFound, Conflict, Invalid, Busy]
    })
  )
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
