#!/usr/bin/env bun
import { BunRuntime, BunServices } from "@effect/platform-bun"
import { Console, Effect, Layer } from "effect"
import { Command } from "effect/cli"
import { Audio } from "./Audio.ts"
import { idiolect } from "./Cli.ts"
import { migrateCheckout, Paths } from "./Paths.ts"
import packageJson from "../package.json" with { type: "json" }

/** Move voices and models out of an older source checkout (set by bin/idiolect). */
const migrate = Effect.gen(function*() {
  const checkout = process.env.IDIOLECT_CHECKOUT
  if (!checkout) return
  const paths = yield* Paths
  for (const move of migrateCheckout(checkout, paths)) yield* Console.error(`Moved ${move}`)
})

const layer = Layer.mergeAll(Paths.layer, Audio.sox).pipe(Layer.provideMerge(BunServices.layer))

migrate.pipe(
  Effect.andThen(Command.run(idiolect, { version: packageJson.version })),
  Effect.catchTag("IdiolectError", (error) =>
    Console.error(`idiolect: ${error.message}`).pipe(Effect.andThen(Effect.sync(() => {
      process.exitCode = 1
    })))
  ),
  Effect.provide(layer),
  BunRuntime.runMain({ disableErrorReporting: false })
)
