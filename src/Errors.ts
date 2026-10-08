import { Schema } from "effect"

/** A failure explained in words for the person at the terminal. */
export class IdiolectError extends Schema.TaggedError<IdiolectError>()("IdiolectError", {
  message: Schema.String
}) {}

export const fail = (message: string) => new IdiolectError({ message })
