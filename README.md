# Idiolect

**Your words. Your voice.**

A local voice cloning CLI. Read a passage once, then turn written text into speech in your voice. Guided recording follows your words and stops automatically. Generated speech plays by default, or saves to a WAV when you pass `--output`.

Idiolect has two parts that talk over a private HTTP connection on `127.0.0.1`:

- **The CLI** (`src/`), written in TypeScript with [Effect](https://effect.website). It handles commands, your microphone and speakers, guided recording, and output files.
- **The speech engine** (`engine/`), a Python package that runs [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for synthesis and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for following your reading. It stays running in the background so the model remains loaded between commands.

Their contract is defined once, as an Effect `HttpApi`, in `src/Api.ts`. Any engine that implements it can replace the Python one.

## Requirements

Idiolect does not detect or install system software for you. Install these yourself:

| Requirement | Used for | Notes |
| --- | --- | --- |
| A GPU that PyTorch supports | Synthesis | An NVIDIA GPU with a working driver on Linux (CUDA 12.8 wheels), or an Apple silicon Mac (Metal). Developed on an RTX 4080 SUPER (16 GB) and an M5 Pro. `idiolect doctor` checks yours. |
| [uv](https://docs.astral.sh/uv/) | Installing the engine | uv fetches Python 3.12 if you do not have it. |
| [Bun](https://bun.sh) 1.3+ | Building or running the CLI | Not needed at runtime if you use a compiled binary. |
| [SoX](https://sox.sourceforge.net) | Microphone and playback | `brew install sox`, `apt install sox`, or your distribution's package. It uses your default devices: CoreAudio on macOS, PulseAudio or ALSA on Linux. Not needed if you only save files with `-o`. |

Windows is untested. The model download is about 4.5 GB.

## Install

From a checkout:

```sh
git clone https://github.com/jpowersdev/idiolect.git
cd idiolect

# The engine: puts idiolect-engine on your PATH (~/.local/bin).
uv tool install ./engine

# The CLI: a single executable with no runtime dependencies.
bun install
bun run build
cp dist/idiolect ~/.local/bin/

idiolect download   # speech and reading models, once
idiolect doctor     # check the GPU and models
idiolect setup --name Me
idiolect say "Here is something I have never recorded myself saying."
idiolect say "Save this voiceover to a file." -o hello.wav
```

Set `IDIOLECT_ENGINE` to the engine's path if `idiolect-engine` is not on your `PATH`.

### With Nix

If you use Nix with flakes, `./bin/idiolect` needs nothing else installed. It enters the pinned development shell, which provides Bun, Python, uv, and SoX, then syncs the locked dependencies and runs the CLI from source:

```sh
./bin/idiolect download
./bin/idiolect setup --name Me
```

On NixOS it also selects the NVIDIA userspace libraries that match the running kernel driver, since a `nixos-rebuild switch` can leave them out of step until you reboot. Optionally link it onto your `PATH` with `ln -s "$PWD/bin/idiolect" ~/.local/bin/idiolect`.

## Record your voice

Run `idiolect setup`, or just `idiolect`. It displays an original 56-word passage and waits for Enter to start recording. Read the entire passage in your normal conversational voice, allowing roughly 20–25 seconds. The engine's recognizer follows your words in order, and recording stops after two recognition passes confirm the full reading. You only read the passage once.

Setup trims the recording around your speech and saves the recognized transcript, including fillers. If the reading is incomplete after 45 seconds, it offers a retry without saving a partial voice. If the microphone only delivers silence, setup says so within a few seconds. On macOS this usually means your terminal app needs permission in System Settings → Privacy & Security → Microphone. Setup then generates and plays new sentences so you can judge the likeness. A new setup recording creates another voice and makes it the default.

Recording uses 48 kHz audio. The engine makes its own 16 kHz copy for recognition and converts the saved reference to the synthesis model's 24 kHz format. Setup creates a reference for cloning; it does not fine-tune model weights.

Guided recording currently uses English and your default microphone. To choose another, pass `idiolect setup --device NAME`, which SoX receives as `AUDIODEV`: a CoreAudio device name on macOS, or a PulseAudio source or ALSA device on Linux. Only clone voices you have permission to use.

## Commands

```sh
idiolect setup --name Me
idiolect say "Hello from my voice."                # Play the generated audio
idiolect say "Hello from my voice." -o hello.wav   # Save without playing
idiolect say --file script.txt -o voiceover.wav
cat script.txt | idiolect say
idiolect voices
idiolect voices --use VOICE_ID
idiolect say "Another take." --voice VOICE_ID --seed 42
idiolect status
idiolect stop                                     # Release GPU memory when idle
idiolect serve                                    # Run the engine in the foreground
idiolect download
idiolect doctor
idiolect --completions zsh                        # Shell completions
```

`--voice` accepts a saved name, full ID, or unambiguous ID prefix. Input and output paths are relative to your current directory. Existing output files are protected unless you pass `--force`. Saved paths go to stdout; generation progress goes to stderr.

Without `--output`, playback waits until the audio finishes. During generation, Ctrl+C cancels your render after the current section. Completed audio remains in the local render history. Long text is split at sentence boundaries where possible and joined with short pauses.

## The engine and your files

The CLI starts the engine automatically and reuses it for later commands. The engine listens only on `127.0.0.1:7860` and refuses requests from browsers. `idiolect stop` refuses to interrupt an active render. Use `--port` for a second engine, and pass the same `--port` to later commands. Only one engine may use a given voice library.

Everything lives in one per-user directory: `~/Library/Application Support/idiolect` on macOS, and `$XDG_DATA_HOME/idiolect` (usually `~/.local/share/idiolect`) on Linux.

- `voices/`: recordings and exact transcripts.
- `renders/`: scripts, progress, and generated WAV files.
- `cli.json`: your default voice.
- `engine.log`: background engine output.
- `models/qwen3-tts-1.7b-base/`: synthesis model.
- `models/whisper-base.en/`: reading recognition model.

Back up `voices/` and `cli.json` to preserve your voices; copy them to another computer to use the same voices there. Model and dependency downloads use the network; recordings and text are processed on your own machine.

Optional environment variables: `IDIOLECT_DATA` moves the voice library, `IDIOLECT_MODELS` moves the models, and `IDIOLECT_ENGINE` selects the engine executable. The engine also honors `IDIOLECT_MODEL` and `IDIOLECT_ASR_MODEL` for individual local models. `OMP_NUM_THREADS` controls the runtime thread environment.

Earlier versions kept `data/` and `models/` inside the checkout. `./bin/idiolect` moves them to the per-user directory the first time it runs, without replacing anything already there.

## Development

```sh
bun test                 # CLI tests, plus the contract test against engine/.venv
bun run typecheck
cd engine && uv run --frozen pytest -q && uv run --frozen ruff check .
```

Run these inside `nix develop` if you use Nix. `bun test` includes a contract test: it starts the real engine HTTP server, with stand-ins for the models, and exercises every endpoint through the typed Effect client, so the two sides cannot drift apart. It is skipped when `engine/.venv` does not exist yet; run `uv sync` in `engine/` first.

The engine generates speech through PyTorch with BF16 and SDPA attention, on CUDA or Metal. FlashAttention compilation is not required.

Upstream models and libraries:

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), Apache 2.0. The 1.7B Base model is pinned to `fd4b254389122332181a7c3db7f27e918eec64e3`.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [Whisper base.en](https://huggingface.co/Systran/faster-whisper-base.en), MIT. The reading model is pinned to `fad79c08a643e3aa623adfae1c1861e50438482d`.
- `bun.lock`, `engine/uv.lock`, and `flake.lock` pin the CLI, engine, and system dependencies.
