# Idiolect

**Your words. Your voice.**

A local voice cloning CLI. Read a passage once, then turn written text into speech in your voice. Guided recording follows your words and stops automatically. Generated speech plays by default, or saves to a WAV when you pass `--output`.

Written in Python 3.12 with `argparse`, Qwen3-TTS for synthesis, and faster-whisper for following your recording. A local background engine keeps the speech model loaded between commands. This repository contains the CLI and that engine; it has no browser UI.

## Get started on NixOS

Requires x86_64 Linux, an NVIDIA GPU with a loaded driver, Nix with flakes enabled, and PipeWire for recording and playback. Developed and tested on an RTX 4080 SUPER with 16 GB VRAM.

```sh
git clone git@github.com:jpowersdev/idiolect.git
cd idiolect
./bin/idiolect download
./bin/idiolect doctor
./bin/idiolect setup --name Me
./bin/idiolect say "Here is something I have never recorded myself saying."
./bin/idiolect say "Save this voiceover to a file." -o hello.wav
```

The launcher enters the pinned Nix development shell, installs the locked Python dependencies, and selects NVIDIA libraries matching the running kernel driver. The first setup downloads several GB of dependencies and model weights. Synthesis and speech recognition then run locally.

Optionally put the command on your `PATH`:

```sh
mkdir -p ~/.local/bin
ln -s "$PWD/bin/idiolect" ~/.local/bin/idiolect
```

Ensure `~/.local/bin` is on your `PATH`. You can also keep using `./bin/idiolect` directly. `python -m idiolect` exposes the same CLI inside the development environment.

## Record your voice

Run `idiolect setup`, or just `idiolect`. It displays an original 56-word passage and waits for Enter to start recording. Read the entire passage in your normal conversational voice, allowing roughly 20–25 seconds. The local recognizer follows your words in order and stops after two recognition passes confirm the full reading. You only read the passage once.

Setup trims the recording around your speech and saves the recognized transcript, including fillers. If the reading is incomplete after 45 seconds, it offers a retry without saving a partial voice. The saved reference must be 3–30 seconds long. Setup then generates and plays new sentences so you can judge the likeness. A new setup recording creates another voice and makes it the default.

Recording uses 48 kHz audio. A separate 16 kHz copy goes to recognition; the saved voice reference is converted directly to the synthesis model's 24 kHz format. Broad sound coverage helps provide a representative sample, but does not guarantee accent or delivery accuracy. Setup creates a reference for cloning; it does not fine-tune model weights.

Guided recording currently uses English and your default PipeWire microphone. Use `idiolect setup --device NAME_OR_ID` to choose another microphone; `wpctl status --name` lists devices. Speech recognition runs locally on the CPU.

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
```

`--voice` accepts a saved name, full ID, or unambiguous ID prefix. Input and output paths are relative to your current directory. Existing output files are protected unless you pass `--force`. Saved paths go to stdout; generation progress goes to stderr.

Without `--output`, playback waits until the audio finishes. Ctrl+C stops playback. During generation, Ctrl+C cancels your render after the current section. Completed audio remains in the local render history. Long text is split at sentence boundaries where possible and joined with short pauses.

## Local engine and files

The CLI starts its engine automatically and reuses it for later commands. The engine listens only on `127.0.0.1:7860`; it provides a private local HTTP API for the CLI, with no browser interface. `idiolect stop` refuses to interrupt an active render.

Use a different port when another checkout is running, for example `./bin/idiolect --port 7861 setup`. Pass the same global `--port` option to subsequent commands. Only one engine may use a given data directory.

- `data/voices/`: recordings and exact transcripts.
- `data/renders/`: scripts, progress, and generated WAV files.
- `data/cli.json`: your default voice.
- `data/engine.log`: background engine output.
- `models/qwen3-tts-1.7b-base/`: synthesis model.
- `models/whisper-base.en/`: reading recognition model.

`data/`, `models/`, and `.venv/` are ignored by Git. Back up `data/` to preserve your recordings and renders. Model and dependency downloads use the network; recordings and text are processed on your own machine.

Optional environment variables: `IDIOLECT_DATA` changes the data directory, `IDIOLECT_MODEL` selects a local Qwen3-TTS Base model, and `IDIOLECT_ASR_MODEL` selects a local faster-whisper model. `OMP_NUM_THREADS` controls the runtime thread environment.

## Development

```sh
nix develop --command uv run --frozen pytest -q
nix develop --command uv run --frozen ruff check idiolect tests
```

The tests cover recording completion, audio preservation, playback and file output, storage, API validation, cancellation, and engine ownership without loading the GPU model.

The CLI uses `argparse`; FastAPI and Uvicorn provide the local engine connection. Model generation runs through PyTorch with CUDA, BF16, and SDPA attention. FlashAttention compilation is not required.

Upstream models and libraries:

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), Apache 2.0. The 1.7B Base model is pinned to `fd4b254389122332181a7c3db7f27e918eec64e3`.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [Whisper base.en](https://huggingface.co/Systran/faster-whisper-base.en), MIT. The reading model is pinned to `fad79c08a643e3aa623adfae1c1861e50438482d`.
- `flake.lock` and `uv.lock` pin the system and Python dependencies.
