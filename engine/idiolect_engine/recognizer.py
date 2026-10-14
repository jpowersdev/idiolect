"""Transcribe microphone audio locally, with word timings, for guided enrollment."""

import threading

import numpy as np
from scipy.signal import resample_poly

from .config import ASR_MODEL, ASR_MODEL_ID, ASR_MODEL_REVISION, recognizer_downloaded

ASR_SAMPLE_RATE = 16000


def recognition_audio(audio, sample_rate):
    """Make an anti-aliased recognition copy without reducing the saved recording."""
    return resample_poly(audio, ASR_SAMPLE_RATE, sample_rate).astype(np.float32)


def download_recognizer():
    from huggingface_hub import snapshot_download

    snapshot_download(
        ASR_MODEL_ID,
        revision=ASR_MODEL_REVISION,
        local_dir=ASR_MODEL,
        allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
    )


class Recognizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None

    def load(self):
        with self.lock:
            if self.model is None:
                from faster_whisper import WhisperModel

                if not recognizer_downloaded():
                    raise ValueError("The reading model is missing. Run idiolect setup first.")
                self.model = WhisperModel(
                    str(ASR_MODEL),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    local_files_only=True,
                )
            return self.model

    def transcribe(self, audio, sample_rate):
        segments, _ = self.load().transcribe(
            recognition_audio(audio, sample_rate),
            language="en",
            beam_size=3,
            temperature=0,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        # Never accept the expected script as a prompt: it must be heard in the audio.
        return [
            {"text": word.word, "start": word.start, "end": word.end}
            for segment in segments
            for word in (segment.words or [])
        ]
