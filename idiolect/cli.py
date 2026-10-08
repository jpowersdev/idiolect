"""Guided voice enrollment and local text-to-speech commands."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import httpx

from .client import Backend
from .config import ASR_MODEL, DATA, LANGUAGES
from .reading import Reader, download_recognizer, reading_script, record_reading
from .storage import write_json

PREVIEW_TEXT = (
    "Here is something I never read aloud. Does this sound like me? "
    "A familiar voice can tell a completely new story."
)


def default_voice():
    try:
        return json.loads((DATA / "cli.json").read_text()).get("default_voice")
    except FileNotFoundError:
        return None
    except (ValueError, AttributeError) as error:
        raise ValueError(f"Cannot read {DATA / 'cli.json'}: {error}") from error


def set_default(voice):
    DATA.mkdir(parents=True, exist_ok=True)
    write_json(DATA / "cli.json", {"default_voice": voice["id"]})


def choose_voice(voices, selected=None):
    selected = selected or default_voice()
    if not selected:
        raise ValueError("Set up your voice with idiolect setup, or choose one with --voice.")
    matches = [voice for voice in voices if voice["id"] == selected]
    if not matches:
        matches = [
            voice
            for voice in voices
            if voice["name"].casefold() == selected.casefold() or voice["id"].startswith(selected)
        ]
    if len(matches) != 1:
        raise ValueError("Voice not found or ambiguous. Run idiolect voices and use its ID.")
    return matches[0]


def enroll(backend, args):
    if not sys.stdin.isatty():
        raise ValueError("Run idiolect setup in an interactive terminal to record your microphone.")
    if not (ASR_MODEL / "model.bin").is_file():
        print("Downloading the local reading model (about 150 MB, once)…", file=sys.stderr)
        download_recognizer()
    print("Preparing the reading assistant…", file=sys.stderr, flush=True)
    reader = Reader()
    script = reading_script()
    print(f"\nRead this aloud in your natural voice:\n\n{textwrap.fill(script, 72)}\n")
    print("Allow about 20–25 seconds at your usual pace, with natural pauses.")
    print("Use your everyday accent and speaking voice.")
    print("Follow the whole script. I will stop after recognizing the final words.")
    if args.device:
        print(f"Microphone: {args.device}")
    else:
        print("Using your default microphone. Use --device to choose another.")

    def progress(match):
        next_words = f'  Next: "{match.next_words}"' if match.next_words else "  Finishing…"
        print(
            f"\r\033[KFollowing {match.matched}/{match.total} words.{next_words}",
            end="",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="idiolect-reading-") as folder:
        audio = Path(folder) / "reference.wav"
        while True:
            input("\nPress Enter when you are ready to start. ")
            print("Reading along… (Ctrl+C to cancel)", flush=True)
            try:
                transcript, duration = record_reading(script, audio, reader, progress, args.device)
                break
            except ValueError as error:
                print(f"\n{error}")
                if input("Press Enter to try again, or type q to quit: ").strip().lower() == "q":
                    return
        print(f"\nGot it. Complete script, {duration:g} seconds.")
        with audio.open("rb") as recording:
            voice = backend.request(
                "POST",
                "/api/voices",
                data={"name": args.name, "transcript": transcript},
                files={"audio": ("reference.wav", recording, "audio/wav")},
            )
    set_default(voice)
    print(f'\nSaved "{voice["name"]}" as your default voice.')
    print("Now listen to a few new sentences in your cloned voice.", flush=True)
    try:
        player = audio_player()
        job = render_voice(backend, voice, PREVIEW_TEXT)
        play_render(backend, job, player)
    except (ValueError, OSError, httpx.HTTPError) as error:
        print(f"Your voice is saved, but the preview could not finish: {error}", file=sys.stderr)
    print('Try: idiolect say "This is my voice, reading something new."')


def audio_player():
    # macOS ships afplay; Linux plays through PipeWire.
    name = "afplay" if sys.platform == "darwin" else "pw-play"
    player = shutil.which(name)
    if player is None:
        raise ValueError(
            f"Playback needs {name}. Use the idiolect launcher "
            "or pass -o filename.wav to save the audio."
        )
    return player


def render_voice(backend, voice, text, language="English", seed=12):
    print(f'Rendering as "{voice["name"]}"…', file=sys.stderr, flush=True)
    previous = None

    def progress(job):
        nonlocal previous
        message = job["message"]
        if message != previous:
            print(message, file=sys.stderr, flush=True)
            previous = message

    return backend.render(voice["id"], text, language, seed, progress)


def play_render(backend, job, player):
    with tempfile.TemporaryDirectory(prefix="idiolect-playback-") as folder:
        audio = Path(folder) / "voiceover.wav"
        backend.download(job["id"], audio)
        print(f"Playing {job['duration']:g}s of audio…", file=sys.stderr, flush=True)
        result = subprocess.run(
            [player, str(audio)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            name = Path(player).name
            detail = result.stderr.strip() or f"{name} exited with status {result.returncode}"
            raise ValueError(f"Playback failed: {detail}")


def input_text(args):
    if args.text and args.file is not None:
        raise ValueError("Use words on the command line or --file, rather than both.")
    if args.file is not None:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    elif args.text:
        text = " ".join(args.text)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ValueError('Give me words: idiolect say "Hello there"')
    text = text.strip()
    if not 1 <= len(text) <= 12000:
        raise ValueError("Use a script between 1 and 12,000 characters.")
    return text


def say(backend, args):
    text = input_text(args)
    output = Path(args.output).expanduser().absolute() if args.output is not None else None
    player = None
    if output is not None:
        if output.suffix.lower() != ".wav":
            raise ValueError("Choose a .wav output file.")
        if output.exists() and not args.force:
            raise ValueError(f"{output} already exists. Choose another path or add --force.")
        if not output.parent.is_dir():
            raise ValueError(f"The output directory does not exist: {output.parent}")
    else:
        player = audio_player()
    backend.ensure()
    voice = choose_voice(backend.request("GET", "/api/voices"), args.voice)
    job = render_voice(backend, voice, text, args.language, args.seed)
    if output is not None:
        backend.download(job["id"], output, args.force)
        print(f"Saved {job['duration']:g}s of audio.", file=sys.stderr)
        print(output)
    else:
        play_render(backend, job, player)


def parser():
    result = argparse.ArgumentParser(prog="idiolect", description="Your words. Your voice.")
    result.add_argument("--port", type=int, default=7860, help="Local engine port (default: 7860)")
    commands = result.add_subparsers(dest="command")
    setup = commands.add_parser("setup", help="Read a script, save your voice, and hear a preview")
    setup.add_argument("--name", default="Me", help="Name for this voice (default: Me)")
    setup.add_argument(
        "--device",
        help="Microphone name or ID: on Linux see wpctl status --name; on macOS see "
        'ffmpeg -f avfoundation -list_devices true -i ""',
    )
    speech = commands.add_parser("say", help="Speak text aloud, or save a WAV with --output")
    speech.add_argument("text", nargs="*")
    speech.add_argument("-f", "--file", help="Read a script from a text file, or - for stdin")
    speech.add_argument("-o", "--output", help="Save a WAV instead of playing it aloud")
    speech.add_argument("--voice", help="Saved voice name or ID (otherwise your default)")
    speech.add_argument("--language", choices=LANGUAGES, default="English")
    speech.add_argument("--seed", type=int, default=12)
    speech.add_argument("--force", action="store_true", help="Replace an existing output file")
    voices = commands.add_parser("voices", help="List saved voices")
    voices.add_argument("--use", help="Make a voice name or ID the default")
    commands.add_parser("status", help="Show the local engine status")
    commands.add_parser("stop", help="Stop the idle engine and release its GPU memory")
    commands.add_parser("serve", help="Run the local speech engine in the foreground")
    commands.add_parser("download", help="Download speech and reading models for offline use")
    commands.add_parser("doctor", help="Check the GPU and speech model")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.command is None:
        args = arguments.parse_args(["--port", str(args.port), "setup"])
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Choose a port between 1 and 65535.")
        if args.command in {"serve", "download", "doctor"}:
            command = [sys.executable, "-m", "idiolect.runtime", args.command]
            if args.command == "serve":
                command += ["--port", str(args.port)]
                os.execv(sys.executable, command)
            result = subprocess.run(command, check=False)
            if result.returncode:
                return result.returncode
            if args.command == "download":
                download_recognizer()
            return 0
        with Backend(args.port) as backend:
            if args.command == "say":
                if not 0 <= args.seed <= 2**31 - 1:
                    raise ValueError("Choose a seed between 0 and 2147483647.")
                say(backend, args)
            elif args.command == "status":
                status = backend.probe()
                print(f"Engine {status['model_status']} · {DATA}" if status else "Engine stopped")
            elif args.command == "stop":
                print("Engine stopping." if backend.stop() else "Engine already stopped.")
            else:
                backend.ensure()
                if args.command == "setup":
                    if not 1 <= len(args.name.strip()) <= 60:
                        raise ValueError("Use a voice name between 1 and 60 characters.")
                    enroll(backend, args)
                elif args.command == "voices":
                    voices = backend.request("GET", "/api/voices")
                    if args.use:
                        set_default(choose_voice(voices, args.use))
                    selected = default_voice()
                    for voice in voices:
                        mark = "*" if voice["id"] == selected else " "
                        print(
                            f"{mark} {voice['id'][:8]}  {voice['name']}  ({voice['duration']:g}s)"
                        )
                    if not voices:
                        print("No voices yet. Run idiolect setup.")
        return 0
    except (ValueError, OSError, httpx.HTTPError, EOFError) as error:
        print(f"idiolect: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
