# Developing Idiolect

Idiolect has two parts:

- **The CLI** (`src/`), in TypeScript with [Effect](https://effect.website): commands, the microphone and speakers, and output files.
- **The speech engine** (`engine/`), a Python package that clones your voice with [VoxCPM2](https://github.com/OpenBMB/VoxCPM) or [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), and uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) to follow your reading and check what it says.

The CLI starts the engine in the background and talks to it over a Unix socket in the data folder. The engine keeps the model loaded between commands and exits after 10 idle minutes. Their contract is defined once, as an Effect `HttpApi`, in `src/Api.ts`.

## Setup and tests

```sh
nix develop              # or install Bun, uv, and SoX yourself
(cd engine && uv sync)   # the dev shell puts engine/.venv/bin on PATH
bun src/main.ts doctor   # run the CLI from source
bun test                 # CLI tests, plus the contract test against engine/.venv
bun run typecheck
cd engine && uv run --frozen pytest -q && uv run --frozen ruff check .
```

The contract test starts the real engine HTTP server, with stand-ins for the models, and exercises every endpoint through the typed client, so the two sides cannot drift apart. It is skipped until `engine/.venv` exists.

## How speech is made

- **Recording.** Setup records each script in `src/Scripts.ts` as one take, keeps it whole with its word timings (for fine-tuning later), and cuts it into clips that end at sentences. Adding scripts grows the dataset.
- **Cloning.** The engine clones from the first clip that ends a sentence. `engine/idiolect_engine/models.py` describes each model and chooses one by memory: Apple GPUs run VoxCPM2 in full precision, about 15 GB, and Qwen3-TTS in half precision, about 8 GB.
- **Rendering.** Text is split into sections by paragraph, then by sentence. Each section is checked with the recognizer, rendered again if garbled or faded, cut where its last words end, and level-matched before joining.

## Releasing

The npm package carries the engine: packing builds it as a wheel, with `engine/uv.lock` exported as pinned requirements, into `dist/engine`. Without `idiolect-engine` on PATH, `idiolect setup` asks to install that bundle with uv into `~/.idiolect/engine`, and installs it again whenever the CLI's version changes. Nix and the development shell put `idiolect-engine` on PATH instead.

1. Set the same version in `package.json` and `engine/pyproject.toml` (a test checks they match).
2. `bun publish`. Packing runs `bun run bundle-engine` first.

## Nix

When `bun.lock` changes, update `outputHash` for the CLI's dependencies in `flake.nix`: build once with an empty hash and copy the one Nix reports. The engine's environment comes straight from `engine/uv.lock` through [uv2nix](https://github.com/pyproject-nix/uv2nix). VoxCPM's training, denoising, and web UI dependencies are left out in `engine/pyproject.toml`; generation never imports them.

## Upstream models, libraries, and texts

- [VoxCPM2](https://github.com/OpenBMB/VoxCPM), Apache 2.0, pinned to `32279effe8c19989596f05d353d1447f51d9e915`.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), Apache 2.0. The 1.7B Base model is pinned to `fd4b254389122332181a7c3db7f27e918eec64e3`.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [Whisper base.en](https://huggingface.co/Systran/faster-whisper-base.en), MIT, pinned to `fad79c08a643e3aa623adfae1c1861e50438482d`.
- [Comma Gets a Cure](https://www.dialectsarchive.com/comma-gets-a-cure). "Comma Gets a Cure" and derivative works may be used freely for any purpose without special permission provided the present sentence and the following copyright notification accompany the passage in print, if reproduced in print, and in audio format in the case of a sound recording: Copyright 2000 Douglas N. Honorof, Jill McCullough & Barbara Somerville.
