"""
The speech models Idiolect can clone your voice with, and how it picks one for this machine.

Both clone from the same clips and transcripts, so switching never means recording again.
VoxCPM2 is cleaner, but Apple GPUs must run it in full precision, which takes about 15 GB;
Qwen3-TTS runs in half precision in about 8 GB, but garbles or fades more often.
"""

import os
from dataclasses import dataclass

from .config import MODELS

GIGABYTE = 1024**3


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    repository: str
    revision: str
    folder: str
    files: tuple[str, ...]
    patterns: tuple[str, ...]
    # Memory this machine needs for the model, with room for everything else.
    needs_gb: int

    @property
    def path(self):
        return MODELS / self.folder

    def downloaded(self):
        return all((self.path / name).is_file() for name in self.files)

    def download(self):
        from huggingface_hub import snapshot_download

        snapshot_download(
            self.repository,
            revision=self.revision,
            local_dir=self.path,
            allow_patterns=list(self.patterns),
        )


VOXCPM2 = Model(
    key="voxcpm2",
    name="VoxCPM2",
    repository="openbmb/VoxCPM2",
    revision="32279effe8c19989596f05d353d1447f51d9e915",
    folder="voxcpm2",
    files=("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json"),
    patterns=("*.json", "*.safetensors", "*.pth", "*.py"),
    needs_gb=24,
)
QWEN3 = Model(
    key="qwen3",
    name="Qwen3-TTS",
    repository="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    revision="fd4b254389122332181a7c3db7f27e918eec64e3",
    folder="qwen3-tts-1.7b-base",
    files=(
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
    ),
    patterns=("*.json", "*.txt", "*.safetensors"),
    needs_gb=12,
)
# Best first: the first one this machine can run is the default.
MODELS_BY_PREFERENCE = (VOXCPM2, QWEN3)


def memory_gb():
    """
    Physical memory, which Apple silicon shares between the CPU and GPU. On a Linux machine
    with a separate GPU, its memory is what matters; that needs its own check.
    """
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIGABYTE


class Choice:
    """The model to use and why, for the engine and for idiolect doctor."""

    def __init__(self, model, reason, warning=None, requested=False):
        self.model = model
        self.reason = reason
        self.warning = warning
        # Set by IDIOLECT_MODEL rather than chosen for the hardware.
        self.requested = requested


def choose(env=None, memory=None):
    """IDIOLECT_MODEL when set; otherwise the best model that fits in this machine's memory."""
    env = os.environ if env is None else env
    memory = memory_gb() if memory is None else memory
    requested = (env.get("IDIOLECT_MODEL") or "").strip().lower()
    if requested:
        model = next((m for m in MODELS_BY_PREFERENCE if m.key == requested), None)
        if model is None:
            keys = ", ".join(m.key for m in MODELS_BY_PREFERENCE)
            raise ValueError(f"IDIOLECT_MODEL={requested} is not a model. Choose one of: {keys}.")
        warning = (
            f"{model.name} needs {model.needs_gb} GB of memory; this machine has {memory:.0f} GB."
            if memory < model.needs_gb
            else None
        )
        return Choice(model, "set by IDIOLECT_MODEL", warning, requested=True)
    for model in MODELS_BY_PREFERENCE:
        if memory >= model.needs_gb:
            return Choice(model, f"chosen for {memory:.0f} GB of memory")
    smallest = MODELS_BY_PREFERENCE[-1]
    return Choice(
        smallest,
        f"the smallest model, for {memory:.0f} GB of memory",
        f"Idiolect needs at least {smallest.needs_gb} GB of memory; this machine has "
        f"{memory:.0f} GB, so speaking will be very slow.",
    )


class LoadedVoxCPM2:
    def __init__(self, model):
        from voxcpm import VoxCPM

        # VoxCPM picks the GPU itself, in full precision on Apple silicon.
        self.model = VoxCPM.from_pretrained(str(model.path), load_denoiser=False, optimize=False)

    def prompt(self, clip_path, clip_text):
        # Continue from the clip, and clone its timbre from it too.
        return {
            "prompt_wav_path": str(clip_path),
            "prompt_text": clip_text,
            "reference_wav_path": str(clip_path),
        }

    def generate(self, prompt, text, seed):
        import torch

        torch.manual_seed(seed)
        wav = self.model.generate(text=text, **prompt)
        return wav, self.model.tts_model.sample_rate


class LoadedQwen3:
    # Qwen3-TTS speaks other languages too; the CLI and the reading are English for now.
    LANGUAGE = "English"

    def __init__(self, model, device):
        import torch
        from qwen_tts import Qwen3TTSModel

        torch.set_num_threads(8)
        self.model = Qwen3TTSModel.from_pretrained(
            str(model.path),
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )

    def prompt(self, clip_path, clip_text):
        return self.model.create_voice_clone_prompt(
            ref_audio=str(clip_path), ref_text=clip_text, x_vector_only_mode=False
        )

    def generate(self, prompt, text, seed):
        import torch

        torch.manual_seed(seed)
        wavs, rate = self.model.generate_voice_clone(
            text=text, language=self.LANGUAGE, voice_clone_prompt=prompt, max_new_tokens=2048
        )
        return wavs[0], rate


def load(model, device):
    return LoadedVoxCPM2(model) if model is VOXCPM2 else LoadedQwen3(model, device)
