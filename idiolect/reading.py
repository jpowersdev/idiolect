"""Follow a spoken enrollment script without sending microphone audio anywhere."""

import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import ASR_MODEL, ASR_MODEL_ID, ASR_MODEL_REVISION

# Keep the full recording for cloning; only the recognizer needs a 16 kHz copy.
SAMPLE_RATE = 48000
ASR_SAMPLE_RATE = 16000
MAX_RECORDING_SECONDS = 45
READING_SCRIPT = (
    "On Thursday, three friends chose a quiet path through the village. "
    "A small boy in a red jacket watched a brown dog cross the green fields. "
    "Have you packed a good book, some fresh bread, and a cup of tea? "
    "We usually enjoy these walks, even when the weather turns cold and the wind gets strong."
)


def reading_script():
    return READING_SCRIPT


def recognition_audio(audio):
    """Make an anti-aliased recognition copy without reducing the saved recording."""
    return resample_poly(audio, ASR_SAMPLE_RATE, SAMPLE_RATE).astype(np.float32)


def tokens(text):
    # Ignore punctuation/case while retaining word order and actual spoken words.
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "").replace("'", ""))


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class ReadingMatch:
    matched: int
    total: int
    next_words: str
    first_word: int | None
    last_word: int | None

    @property
    def complete(self):
        return self.total > 0 and self.matched == self.total


def match_reading(script, words):
    expected = tokens(script)
    heard, owners = [], []
    for index, word in enumerate(words):
        parts = tokens(word.text)
        heard.extend(parts)
        owners.extend([index] * len(parts))
    positions = {}
    for block in SequenceMatcher(None, expected, heard, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            positions[block.a + offset] = owners[block.b + offset]
    next_index = next((i for i in range(len(expected)) if i not in positions), len(expected))
    return ReadingMatch(
        matched=len(positions),
        total=len(expected),
        next_words=" ".join(expected[next_index : next_index + 6]),
        first_word=positions.get(0),
        last_word=positions.get(len(expected) - 1),
    )


def download_recognizer():
    from huggingface_hub import snapshot_download

    snapshot_download(
        ASR_MODEL_ID,
        revision=ASR_MODEL_REVISION,
        local_dir=ASR_MODEL,
        allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
    )


class Reader:
    def __init__(self):
        from faster_whisper import WhisperModel

        if not all((ASR_MODEL / name).is_file() for name in ("model.bin", "tokenizer.json")):
            raise ValueError("The reading model is missing. Run idiolect download first.")
        self.model = WhisperModel(
            str(ASR_MODEL),
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            local_files_only=True,
        )

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=3,
            temperature=0,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        # Never provide the expected script as a prompt: it must be heard in the audio.
        return [
            Word(word.word, word.start, word.end)
            for segment in segments
            for word in (segment.words or [])
        ]


def save_reading(audio, words, match, output):
    if not match.complete:
        raise ValueError("The full script was not recognized. Please try the reading again.")
    first, last = words[match.first_word], words[match.last_word]
    start = max(0, int((first.start - 0.25) * SAMPLE_RATE))
    end = min(len(audio), int((last.end + 0.35) * SAMPLE_RATE))
    clip = audio[start:end]
    duration = len(clip) / SAMPLE_RATE
    if not 3 <= duration <= 30:
        raise ValueError("Read at a natural pace so the complete script fits in 3–30 seconds.")
    # Retain fillers/repetitions actually heard between the first and last script words.
    transcript = " ".join(
        word.text.strip() for word in words[match.first_word : match.last_word + 1]
    )
    sf.write(output, clip, SAMPLE_RATE, subtype="PCM_16")
    return transcript, round(duration, 2)


def stop_recorder(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def record_reading(script, output: Path, reader, progress, device=None):
    executable = shutil.which("pw-record")
    if not executable:
        raise ValueError("PipeWire recording is unavailable. Start with the idiolect launcher.")
    command = [
        executable,
        "--raw",
        "--rate",
        str(SAMPLE_RATE),
        "--channels",
        "1",
        "--format",
        "s16",
    ]
    if device:
        command.extend(["--target", device])
    command.append("-")
    with tempfile.TemporaryFile() as errors, ThreadPoolExecutor(max_workers=1) as pool:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors, bufsize=0
        )
        future = None
        raw = bytearray()
        started = time.monotonic()
        last_submission = 0.0
        snapshot_duration = 0.0
        complete_passes = 0
        last_match = match_reading(script, [])
        try:
            os.set_blocking(process.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while time.monotonic() - started < MAX_RECORDING_SECONDS:
                    for key, _ in selector.select(timeout=0.1):
                        chunk = os.read(key.fd, 65536)
                        if chunk:
                            raw.extend(chunk)
                        else:
                            errors.seek(0)
                            detail = errors.read().decode(errors="replace").strip()
                            raise ValueError(f"Microphone recording stopped. {detail}".strip())
                    if future is not None and future.done():
                        words = future.result()
                        last_match = match_reading(script, words)
                        progress(last_match)
                        complete_passes = complete_passes + 1 if last_match.complete else 0
                        if (
                            complete_passes >= 2
                            and snapshot_duration - words[last_match.last_word].end >= 0.45
                        ):
                            stop_recorder(process)
                            audio = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2")
                            return save_reading(
                                audio.astype(np.float32) / 32768, words, last_match, output
                            )
                        future = None
                    now = time.monotonic()
                    if (
                        future is None
                        and len(raw) >= SAMPLE_RATE * 3
                        and now - last_submission >= 1.5
                    ):
                        snapshot = np.frombuffer(bytes(raw[: len(raw) // 2 * 2]), dtype="<i2")
                        snapshot_duration = len(snapshot) / SAMPLE_RATE
                        future = pool.submit(
                            reader.transcribe,
                            recognition_audio(snapshot.astype(np.float32) / 32768),
                        )
                        last_submission = now
            raise ValueError(
                f"Recognized {last_match.matched}/{last_match.total} words before the time limit. "
                "Check your microphone and read the whole script again."
            )
        finally:
            stop_recorder(process)
            process.stdout.close()
