/**
 * The speech engine: `bun src/engine/main.ts`, started in the background by the CLI and given
 * the data directory as IDIOLECT_DATA. It logs to stderr, which the CLI sends to engine.log.
 */
import * as BackendApple from "@effect-torch/backend-apple-native"
import { BunRuntime } from "@effect/platform-bun"
import { Deferred, Effect } from "effect"
import { homedir } from "node:os"
import { mkdirSync, realpathSync } from "node:fs"
import { resolvePaths } from "../Paths.ts"
import * as Local from "./Local.ts"
import * as Renders from "./Renders.ts"
import * as Server from "./Server.ts"

const engine = Effect.gen(function*() {
  const { data: requested } = resolvePaths(process.env, homedir())
  mkdirSync(requested, { recursive: true })
  const data = realpathSync(requested)
  yield* Server.lock(data)
  const local = yield* Local.make(data)
  const renders = yield* Renders.make(local)
  const stop = yield* Deferred.make<void>()
  yield* Effect.logInfo(`Engine ${process.pid} serving ${Server.socketPath(data)}`)
  yield* Server.serve({ data, local, renders, stop })
  yield* Effect.logInfo("Engine stopping")
})

engine.pipe(
  Effect.scoped,
  Effect.provide(BackendApple.layer()),
  // One line the CLI can show, such as another engine already owning the data.
  Effect.catch((error) =>
    Effect.sync(() => {
      process.stderr.write(`Error: ${error instanceof Error ? error.message : String(error)}\n`)
      process.exitCode = 1
    })
  ),
  BunRuntime.runMain
)
