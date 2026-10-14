import os
from pathlib import Path

# The same directory as the CLI's, so the engine can also run on its own.
DATA = Path(os.environ.get("IDIOLECT_DATA") or Path.home() / ".idiolect").resolve()
MODELS = DATA / "models"
ASR_MODEL = MODELS / "whisper-base.en"
ASR_MODEL_ID = "Systran/faster-whisper-base.en"
ASR_MODEL_REVISION = "fad79c08a643e3aa623adfae1c1861e50438482d"
# Bumped whenever the HTTP contract in the CLI's src/Api.ts changes incompatibly.
API_VERSION = 1
SEED = 12
# An idle engine exits after this long, releasing the GPU memory the model holds.
IDLE_TIMEOUT = 10 * 60


def socket_path(data=DATA):
    return Path(data) / "engine.sock"


def recognizer_downloaded():
    return all((ASR_MODEL / name).is_file() for name in ("model.bin", "tokenizer.json"))
