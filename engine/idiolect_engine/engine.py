import logging
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from .audio import (
    coverage,
    cut_after,
    ending_fade,
    ends_sentence,
    join_audio,
    speech_end,
    split_script,
)
from .config import SEED
from .models import choose
from .models import load as load_model

log = logging.getLogger(__name__)
ACTIVE = {"queued", "loading", "generating"}


def accelerator():
    """Use CUDA on Linux and Metal (MPS) on Apple silicon."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def empty_cache(device):
    import torch

    if device == "mps":
        torch.mps.empty_cache()
    elif device:
        torch.cuda.empty_cache()


class Renders:
    """Renders live in memory, with their audio in a temporary folder, for one engine run."""

    KEEP = 8

    def __init__(self):
        self.folder = Path(tempfile.mkdtemp(prefix="idiolect-renders-"))
        self.lock = threading.RLock()
        self.jobs = {}

    def create(self, text, sections):
        job_id = uuid.uuid4().hex
        (self.folder / job_id).mkdir()
        job = {
            "id": job_id,
            "text": text,
            "sections": sections,
            "completed": 0,
            "total": len(sections),
            "status": "queued",
            "message": "Waiting for the speech engine",
            "duration": None,
        }
        with self.lock:
            self.jobs[job_id] = job
            self._prune()
            return dict(job)

    def _prune(self):
        finished = [job for job in self.jobs.values() if job["status"] not in ACTIVE]
        for job in finished[: max(0, len(finished) - self.KEEP)]:
            del self.jobs[job["id"]]
            shutil.rmtree(self.path(job["id"]), ignore_errors=True)

    def path(self, job_id):
        return self.folder / job_id

    def get(self, job_id):
        with self.lock:
            return dict(self.jobs[job_id])

    def update(self, job_id, **changes):
        with self.lock:
            self.jobs[job_id].update(changes)
            return dict(self.jobs[job_id])

    def active(self):
        with self.lock:
            return sum(job["status"] in ACTIVE for job in self.jobs.values())

    def close(self):
        shutil.rmtree(self.folder, ignore_errors=True)


# A section is re-rendered, with another seed, when it sounds garbled or fades out.
ATTEMPTS = 3
REQUIRED_COVERAGE = 0.9
# Final words this many decibels quieter than the section's typical word: it faded out.
FADE_DB = 12


class Engine:
    def __init__(self, store, recognizer=None, choice=None):
        self.store = store
        self.recognizer = recognizer
        self.choice = choice or choose()
        self.renders = Renders()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-engine")
        self.lock = threading.RLock()
        self.model = None
        self.prompt = None
        self.cancellations = {}
        self.model_status = "idle"

    def status(self):
        return {
            "model": self.choice.model.key,
            "model_requested": self.choice.requested,
            "model_downloaded": self.choice.model.downloaded(),
            "model_status": self.model_status,
        }

    def active(self):
        return self.renders.active() > 0

    def submit(self, voice, text):
        with self.lock:
            if self.renders.active() >= 8:
                raise ValueError("The queue is full. Wait for a render to finish.")
            job = self.renders.create(text, split_script(text))
            self.cancellations[job["id"]] = threading.Event()
            self.pool.submit(self._generate, job["id"], voice)
            return job

    def job(self, job_id):
        return self.renders.get(job_id)

    def audio(self, job_id):
        return self.renders.path(job_id) / "voiceover.wav"

    def cancel(self, job_id):
        with self.lock:
            event = self.cancellations.get(job_id)
            if event:
                event.set()
                if self.renders.get(job_id)["status"] in ACTIVE:
                    self.renders.update(job_id, message="Stopping after the current section…")
            return self.renders.get(job_id)

    def load(self):
        if self.model is None:
            model = self.choice.model
            if not model.downloaded():
                raise RuntimeError("The speech model is missing. Run idiolect setup first.")
            self.model_status = "loading"
            device = accelerator()
            if device is None:
                raise RuntimeError("No supported GPU is available. Run idiolect doctor.")
            log.info("Loading %s (%s)", model.name, self.choice.reason)
            self.model = load_model(model, device)
            self.model_status = "ready"
        return self.model

    def _voice_prompt(self, model, voice):
        # Setup replaces the voice; its creation time identifies the recordings.
        if self.prompt is None or self.prompt[0] != voice["created_at"]:
            # A clip ending mid-sentence would make new text sound like the rest of it.
            clip = next((c for c in voice["clips"] if ends_sentence(c["text"])), voice["clips"][0])
            self.prompt = (
                voice["created_at"],
                model.prompt(self.store.path(clip["file"]), clip["text"]),
            )
        return self.prompt[1]

    def _render_section(self, model, prompt, text, index, total, job_id, cancel):
        """
        Render one section and check it with the recognizer: keep the first attempt that says
        every word through the end without fading out, cut where its final words end, or else
        the best attempt.
        """
        best = None
        for attempt in range(ATTEMPTS):
            again = " again" if attempt else ""
            self.renders.update(
                job_id,
                status="generating",
                message=f"Rendering section {index + 1} of {total}{again}…",
            )
            wav, sr = model.generate(prompt, text, SEED + index + 1000 * attempt)
            if self.recognizer is None:
                return wav, sr
            words = self.recognizer.transcribe(wav, sr)
            end = speech_end(text, words)
            # Every word heard, through the end, without fading out.
            faded = ending_fade(wav, sr, words) > FADE_DB
            score = coverage(text, words) + (1 if end is not None else 0) - (1 if faded else 0)
            if end is not None:
                # Cut the syllables the model sometimes adds after the text.
                wav = cut_after(wav, sr, end)
            if best is None or score > best[0]:
                best = (score, wav, sr)
            if end is not None and not faded and score - 1 >= REQUIRED_COVERAGE or cancel.is_set():
                break
            problem = "faded out" if faded else "sounded garbled"
            log.info("Section %s %s (score %.2f); rendering again", index + 1, problem, score)
        return best[1], best[2]

    def _generate(self, job_id, voice):
        started = time.monotonic()
        cancel = self.cancellations[job_id]
        folder = self.renders.path(job_id)
        try:
            if cancel.is_set():
                self.renders.update(job_id, status="cancelled", message="Cancelled")
                return
            self.renders.update(job_id, status="loading", message="Loading the voice model…")
            model = self.load()
            import torch

            job = self.renders.get(job_id)
            with torch.inference_mode():
                prompt = self._voice_prompt(model, voice)
                for index, section in enumerate(job["sections"]):
                    if cancel.is_set():
                        break
                    wav, sr = self._render_section(
                        model, prompt, section["text"], index, job["total"], job_id, cancel
                    )
                    sf.write(folder / f"section-{index:03d}.wav", wav, sr, subtype="PCM_16")
                    self.renders.update(job_id, completed=index + 1)
            if cancel.is_set():
                self.renders.update(job_id, status="cancelled", message="Cancelled")
            else:
                duration = join_audio(
                    sorted(folder.glob("section-*.wav")),
                    [section["pause"] for section in job["sections"]],
                    folder / "voiceover.wav",
                )
                log.info("Rendered %ss in %.1fs", duration, time.monotonic() - started)
                self.renders.update(
                    job_id, status="ready", message="Your voiceover is ready", duration=duration
                )
        except Exception as error:
            log.exception("Generation failed for %s", job_id)
            if self.model is None:
                self.model_status = "error"
            message = str(error)
            if "out of memory" in message.lower():
                message = "The GPU ran out of memory. Close other GPU apps and try again."
                empty_cache(accelerator())
            self.renders.update(job_id, status="failed", message=message[:700])
        finally:
            with self.lock:
                self.cancellations.pop(job_id, None)

    def shutdown(self):
        with self.lock:
            for event in self.cancellations.values():
                event.set()
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.renders.close()
