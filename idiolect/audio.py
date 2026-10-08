import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def split_script(text: str, limit=300):
    """Keep sentence boundaries where possible, and bound unpunctuated input too."""
    if limit < 1:
        raise ValueError("Chunk limit must be positive")
    text = re.sub(r"[ \t]+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n+", text)
    pieces = []
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > limit:
            boundary = max(
                sentence.rfind(" ", 0, limit + 1),
                sentence.rfind("，", 0, limit),
                sentence.rfind(",", 0, limit),
            )
            if boundary < limit // 2:
                boundary = limit
            elif sentence[boundary] in ",，":
                boundary += 1
            pieces.append(sentence[:boundary].strip())
            sentence = sentence[boundary:].strip()
        if sentence:
            pieces.append(sentence)
    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > limit:
            chunks.append(current)
            current = ""
        current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def prepare_reference(source: Path, output: Path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            "31",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        timeout=40,
        check=False,
    )
    if result.returncode:
        raise ValueError("That file could not be read as audio. Try a WAV, MP3, M4A, or WebM file.")
    audio, sr = sf.read(output, dtype="float32")
    duration = len(audio) / sr
    if not 3 <= duration <= 30:
        raise ValueError(
            "Use a reference between 3 and 30 seconds. A clean 10–15 seconds works well."
        )
    if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 0.0001:
        raise ValueError(
            "This recording is silent or too quiet. Try recording closer to the microphone."
        )
    return round(duration, 2)


def join_audio(paths, output: Path):
    segments = []
    sample_rate = None
    for path in paths:
        audio, sr = sf.read(path, dtype="float32")
        if sample_rate is not None and sr != sample_rate:
            raise ValueError("Generated sections have different sample rates")
        sample_rate = sr
        if segments:
            segments.append(np.zeros(int(sr * 0.18), dtype=np.float32))
        segments.append(audio)
    if not segments:
        raise ValueError("No audio was generated")
    combined = np.concatenate(segments)
    if not np.isfinite(combined).all():
        raise ValueError("The model produced invalid audio. Please try again.")
    peak = float(np.max(np.abs(combined)))
    if peak > 0.98:
        combined = combined * (0.98 / peak)
    sf.write(output, combined, sample_rate, subtype="PCM_16")
    return round(len(combined) / sample_rate, 2)
