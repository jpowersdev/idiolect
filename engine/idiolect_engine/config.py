import os
import sys
from pathlib import Path


def default_home():
    """Match the CLI's per-user data directory when the engine is run on its own."""
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/idiolect"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "idiolect"


DATA = Path(os.environ.get("IDIOLECT_DATA") or default_home()).resolve()
MODELS = Path(os.environ.get("IDIOLECT_MODELS") or DATA / "models").resolve()
MODEL = Path(os.environ.get("IDIOLECT_MODEL") or MODELS / "qwen3-tts-1.7b-base").resolve()
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
ASR_MODEL = Path(os.environ.get("IDIOLECT_ASR_MODEL") or MODELS / "whisper-base.en").resolve()
ASR_MODEL_ID = "Systran/faster-whisper-base.en"
ASR_MODEL_REVISION = "fad79c08a643e3aa623adfae1c1861e50438482d"
# Bumped whenever the HTTP contract in the CLI's src/Api.ts changes incompatibly.
API_VERSION = 1


def model_downloaded():
    return all(
        (MODEL / name).is_file()
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
        )
    )


def recognizer_downloaded():
    return all((ASR_MODEL / name).is_file() for name in ("model.bin", "tokenizer.json"))


LANGUAGES = [
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]
