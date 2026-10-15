/**
 * The models Idiolect runs, pinned to exact revisions on Hugging Face, and only the files it
 * reads. Setup downloads them, after asking; each file is checked against its size and SHA-256.
 */
import { Effect } from "effect"
import { createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from "node:fs"
import { join } from "node:path"
import { fail } from "./Errors.ts"

export interface ModelFile {
  readonly name: string
  readonly size: number
  readonly sha256: string
}

export interface Model {
  readonly name: string
  readonly repository: string
  readonly revision: string
  /** Under the data directory's models folder. */
  readonly folder: string
  readonly files: ReadonlyArray<ModelFile>
}

export const SpeechModel: Model = {
  name: "VoxCPM2",
  repository: "openbmb/VoxCPM2",
  revision: "32279effe8c19989596f05d353d1447f51d9e915",
  folder: "voxcpm2",
  files: [
    { name: "config.json", size: 4336, sha256: "405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c" },
    { name: "tokenizer.json", size: 3676772, sha256: "f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20" },
    { name: "audiovae.pth", size: 376951122, sha256: "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1" },
    { name: "model.safetensors", size: 4580080592, sha256: "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d" }
  ]
}

export const ReadingModel: Model = {
  name: "Whisper base.en",
  repository: "openai/whisper-base.en",
  revision: "911407f4214e0e1d82085af863093ec0b66f9cd6",
  folder: "whisper",
  files: [
    { name: "generation_config.json", size: 1531, sha256: "c5750f05d94777579e00ce26ef65e5d87c108439f90e3ac519df2587b9d5d41f" },
    { name: "preprocessor_config.json", size: 184990, sha256: "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666" },
    { name: "vocab.json", size: 798156, sha256: "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7" },
    { name: "model.safetensors", size: 290401888, sha256: "d4dd5542fd6a1d35639e21384238f3bfe6c557c849d392b5905d33ee29e71db5" }
  ]
}

export const Models = [SpeechModel, ReadingModel] as const

export const modelDirectory = (data: string, model: Model) => join(data, "models", model.folder)

/** Files are written under a temporary name and renamed once verified, so size alone tells. */
const present = (data: string, model: Model, file: ModelFile) => {
  const path = join(modelDirectory(data, model), file.name)
  return existsSync(path) && statSync(path).size === file.size
}

export const downloaded = (data: string, model: Model) => model.files.every((file) => present(data, model, file))

/** Bytes still to download for these models. */
export const missingBytes = (data: string, models: ReadonlyArray<Model> = Models) =>
  models.flatMap((model) => model.files.filter((file) => !present(data, model, file))).reduce((sum, file) => sum + file.size, 0)

export const gigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`

export interface Progress {
  readonly model: Model
  readonly file: ModelFile
  readonly received: number
}

const fetchFile = (url: string, path: string, file: ModelFile, onProgress: (received: number) => void) =>
  Effect.tryPromise({
    try: async (signal) => {
      const response = await fetch(url, { signal, redirect: "follow" })
      if (!response.ok || response.body === null) throw new Error(`HTTP ${response.status}`)
      const hash = new Bun.CryptoHasher("sha256")
      const output = createWriteStream(path)
      const reader = response.body.getReader()
      let received = 0
      try {
        for (let read = await reader.read(); !read.done; read = await reader.read()) {
          const chunk = read.value
          hash.update(chunk)
          received += chunk.byteLength
          if (!output.write(chunk)) await new Promise((resolve) => output.once("drain", resolve))
          onProgress(received)
        }
      } finally {
        await new Promise<void>((resolve, reject) => output.end((error?: Error | null) => error ? reject(error) : resolve()))
      }
      if (received !== file.size || hash.digest("hex") !== file.sha256) throw new Error("the download did not match its checksum")
    },
    catch: (error) => fail(`Downloading ${file.name} failed: ${error instanceof Error ? error.message : String(error)}`)
  })

/** Download whatever is missing of a model, verifying each file before it takes its place. */
export const download = (data: string, model: Model, onProgress: (progress: Progress) => void = () => {}) =>
  Effect.gen(function*() {
    const directory = modelDirectory(data, model)
    mkdirSync(directory, { recursive: true })
    for (const file of model.files) {
      if (present(data, model, file)) continue
      const path = join(directory, file.name)
      const partial = `${path}.download`
      const url = `https://huggingface.co/${model.repository}/resolve/${model.revision}/${file.name}`
      yield* fetchFile(url, partial, file, (received) => onProgress({ model, file, received })).pipe(
        Effect.onError(() => Effect.sync(() => rmSync(partial, { force: true })))
      )
      renameSync(partial, path)
    }
  })
