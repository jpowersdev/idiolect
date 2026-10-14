"""Your voice: whole recorded takes, kept for training, and short clips cut from them for cloning."""

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from .audio import REFERENCE_RATE, cut_clips, read_take
from .storage import timestamp, write_json


class Word(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=100)
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class Take(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    script: str = Field(pattern=r"^[a-z0-9-]{1,40}$")
    transcript: str = Field(min_length=1, max_length=5000)
    words: list[Word] = Field(min_length=1)
    # Some scripts' licenses ask that recordings carry their notice.
    notice: str | None = Field(default=None, max_length=1000)


class Takes(RootModel[list[Take]]):
    root: list[Take] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def distinct_scripts(self):
        if len({take.script for take in self.root}) != len(self.root):
            raise ValueError("Record each script once")
        return self


parse_takes = Takes.model_validate_json


def build(folder, takes: Takes, sources):
    """Save each take at its recorded rate, then cut and save its 24 kHz clips."""
    (folder / "takes").mkdir()
    (folder / "clips").mkdir()
    saved, clips = [], []
    for take, source in zip(takes.root, sources, strict=True):
        audio, rate = read_take(source)
        name = f"takes/{take.script}.wav"
        sf.write(folder / name, audio, rate, subtype="PCM_16")
        words = [word.model_dump() for word in take.words]
        saved.append(
            {
                "script": take.script,
                "transcript": take.transcript,
                "duration": round(len(audio) / rate, 2),
                "words": words,
                "file": name,
            }
            | ({"notice": take.notice} if take.notice else {})
        )
        for index, (clip, text) in enumerate(cut_clips(audio, rate, words), 1):
            name = f"clips/{take.script}-{index:02d}.wav"
            sf.write(folder / name, np.clip(clip, -1, 1), REFERENCE_RATE, subtype="PCM_16")
            clips.append(
                {
                    "script": take.script,
                    "text": text,
                    "duration": round(len(clip) / REFERENCE_RATE, 2),
                    "file": name,
                }
            )
    if not clips:
        raise ValueError("The recordings are too short to clone from.")
    voice = {
        "created_at": timestamp(),
        "duration": round(sum(take["duration"] for take in saved), 2),
        "takes": saved,
        "clips": clips,
    }
    write_json(folder / "voice.json", voice)
    return voice


def summary(voice):
    return {
        "created_at": voice["created_at"],
        "duration": voice["duration"],
        "takes": [
            {"script": take["script"], "duration": take["duration"]} for take in voice["takes"]
        ],
    }
