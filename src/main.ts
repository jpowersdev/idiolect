#!/usr/bin/env bun
import { BunRuntime, BunServices } from "@effect/platform-bun"
import { Console, Effect, Layer } from "effect"
import { Command } from "effect/cli"
import { Audio } from "./Audio.ts"
import { idiolect } from "./Cli.ts"
import { Engine } from "./Engine.ts"
import { Paths } from "./Paths.ts"
import packageJson from "../package.json" with { type: "json" }

const layer = Layer.mergeAll(Engine.layer, Audio.sox).pipe(
  Layer.provideMerge(Paths.layer),
  Layer.provideMerge(BunServices.layer)
)

Command.run(idiolect, { version: packageJson.version }).pipe(
  Effect.catchTag("IdiolectError", (error) =>
    Console.error(`idiolect: ${error.message}`).pipe(Effect.andThen(Effect.sync(() => {
      process.exitCode = 1
    })))
  ),
  Effect.provide(layer),
  BunRuntime.runMain({ disableErrorReporting: false })
)
