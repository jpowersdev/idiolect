import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("IDIOLECT_DATA", ROOT / "data")).resolve()
MODEL = Path(os.environ.get("IDIOLECT_MODEL", ROOT / "models/qwen3-tts-1.7b-base")).resolve()
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
ASR_MODEL = Path(os.environ.get("IDIOLECT_ASR_MODEL", ROOT / "models/whisper-base.en")).resolve()
ASR_MODEL_ID = "Systran/faster-whisper-base.en"
ASR_MODEL_REVISION = "fad79c08a643e3aa623adfae1c1861e50438482d"


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
