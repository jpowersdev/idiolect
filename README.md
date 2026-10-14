# Idiolect

Local voice cloning from the terminal. Read a few short scripts once, then have any text spoken in your voice. Everything runs on your Mac.

```sh
idiolect setup                                  # Record your voice
idiolect say "Here is something I never said."  # Hear it in your voice
idiolect say "Save this one." -o hello.wav      # Or save a WAV
idiolect doctor                                 # Check your Mac, models, and audio
```

## Install

You need an Apple silicon Mac with at least 16 GB of memory.

```sh
brew install bun uv sox
bun add -g @jpowersdev/idiolect
idiolect setup
```

Setup installs the speech engine and downloads the models the first time, after asking.

With Nix, run it directly:

```sh
nix run github:jpowersdev/idiolect -- setup
```

Or add `github:jpowersdev/idiolect` as a flake input and put `inputs.idiolect.packages.${system}.default` in your nix-darwin or NixOS packages.

## Record your voice

Run `idiolect setup`. The first time, it asks to install the speech engine (about 1 GB) and download the models (about 5 GB). Then it shows six short scripts, about four minutes of reading in all. For each one, press Enter and read it in your everyday voice. Recording stops by itself after the last words; press `r` to start a script over. Afterwards, setup plays a few sentences in your new voice.

Use a quiet room and sit close to the microphone. To use a microphone other than the default, pass `--device "MacBook Pro Microphone"` (or another input's name). If setup only hears silence, give your terminal app microphone access in System Settings → Privacy & Security → Microphone.

Running setup again installs anything missing, such as an engine update after you upgrade Idiolect, and asks whether to record a new voice. Only clone voices you have permission to use.

## Speak

```sh
idiolect say "Hello from my voice."
idiolect say < story.txt
idiolect say "A voiceover." -o voiceover.wav
```

Without `-o`, Idiolect plays the audio. With `-o`, it saves a WAV file. Blank lines in your text become paragraph breaks.

## Good to know

- Idiolect picks a speech model for your Mac: VoxCPM2 with 24 GB of memory or more, otherwise Qwen3-TTS. `idiolect doctor` says which. To choose yourself, set `IDIOLECT_MODEL=voxcpm2` or `IDIOLECT_MODEL=qwen3`.
- Your voice and the models live in `~/.idiolect`. Set `IDIOLECT_DATA` to use another folder.
- After setup, nothing leaves your Mac.

Working on Idiolect itself? See [DEVELOPMENT.md](DEVELOPMENT.md).
