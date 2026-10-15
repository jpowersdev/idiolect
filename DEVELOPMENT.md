# Developing Idiolect

Idiolect is one TypeScript package, written with [Effect](https://effect.website), in two processes:

- **The CLI** (`src/`): commands, the microphone and speakers, model downloads, and output files.
- **The speech engine** (`src/engine/`): clones your voice with [VoxCPM2](https://github.com/OpenBMB/VoxCPM), and follows your reading and checks what it says with [Whisper base.en](https://huggingface.co/openai/whisper-base.en). Both models run on the GPU with [effect-torch](https://github.com/mikearnaldi/effect-torch)'s Metal backend.

The CLI starts the engine in the background, with the same Bun, and talks to it over a Unix socket in the data folder. The engine keeps the models loaded between commands and exits after 10 idle minutes. Their contract is defined once, as an Effect `HttpApi`, in `src/Api.ts`; the CLI uses its typed client and the engine implements it with `HttpApiBuilder`.

## Setup and tests

effect-torch is not on npm yet, so it is a submodule in `vendor/effect-torch`, built once with its own Nix toolchains:

```sh
git submodule update --init
bun install
bun run build-native       # effect-torch's native addons and type declarations
nix develop                # or install Bun and SoX yourself
bun src/main.ts doctor     # run the CLI from source
bun test                   # everything, with stand-ins for the models
bun run typecheck
IDIOLECT_MODEL_TESTS=1 bun test test/engine   # the models against captured PyTorch output
```

The contract test serves the real engine API, with stand-ins for the models, and exercises every endpoint through the CLI's typed client, so the two sides cannot drift apart. The model tests compare VoxCPM2, its codec, and Whisper with output captured from PyTorch and Hugging Face Transformers, in `~/.idiolect/experiments`; they are skipped without it.

## How speech is made

- **Recording.** Setup records each script in `src/Scripts.ts` as one take, keeps it whole with its word timings (for fine-tuning later), and cuts it into clips that end at sentences. Adding scripts grows the dataset.
- **Cloning.** The engine continues from the first clip that ends a sentence, which is also the cloning reference, as VoxCPM's own generate does with the same settings (`src/engine/Speech.ts`). The language models run in bfloat16, about 5 GB; the codec runs in float32.
- **Rendering.** Text is split into sections by paragraph, then by sentence. Each section is checked with the recognizer, rendered again if garbled or faded, cut where its last words end, and level-matched before joining (`src/engine/Renders.ts`).
- **Models.** `src/Models.ts` pins each model's revision and files, with their sizes and SHA-256; setup downloads them, after asking. The codec's weights are read straight from `audiovae.pth` (`src/engine/Pth.ts`).

## Releasing

Once effect-torch is published:

1. Replace the `workspace:*` effect-torch dependencies in `package.json` with published versions, and remove `workspaces`, the `effect` override if effect-torch then accepts Effect 4.0, `tsconfig.check.json`'s paths, `scripts/build-native.sh`, and the submodule.
2. Update `outputHash` in `flake.nix`: build once with the fake hash and copy the one Nix reports. The same goes whenever `bun.lock` changes.
3. Bump the version in `package.json`, then `bun publish`.

## Upstream models, libraries, and texts

- [VoxCPM2](https://github.com/OpenBMB/VoxCPM), Apache 2.0, pinned to `32279effe8c19989596f05d353d1447f51d9e915`.
- [Whisper base.en](https://huggingface.co/openai/whisper-base.en), MIT, pinned to `911407f4214e0e1d82085af863093ec0b66f9cd6`.
- [effect-torch](https://github.com/mikearnaldi/effect-torch), pinned as a submodule.
- [Comma Gets a Cure](https://www.dialectsarchive.com/comma-gets-a-cure). "Comma Gets a Cure" and derivative works may be used freely for any purpose without special permission provided the present sentence and the following copyright notification accompany the passage in print, if reproduced in print, and in audio format in the case of a sound recording: Copyright 2000 Douglas N. Honorof, Jill McCullough & Barbara Somerville.
