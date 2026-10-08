import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

from .audio import join_audio, split_script
from .config import MODEL, MODEL_ID, model_downloaded

log = logging.getLogger(__name__)
ACTIVE = {"queued", "loading", "generating"}


class Engine:
    def __init__(self, store):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-engine")
        self.lock = threading.RLock()
        self.model = None
        self.prompts = {}
        self.cancellations = {}
        self.model_status = "idle"

    def recover(self):
        for job in self.store.list_jobs():
            if job["status"] in ACTIVE:
                self.store.update_job(
                    job["id"],
                    status="failed",
                    message="Interrupted by a server restart. Generate again to retry.",
                )

    def status(self):
        return {
            "model": MODEL_ID,
            "model_downloaded": model_downloaded(),
            "model_status": self.model_status,
            "local": True,
        }

    def submit(self, voice, text, language, seed):
        with self.lock:
            if sum(j["status"] in ACTIVE for j in self.store.list_jobs()) >= 8:
                raise ValueError("The queue is full. Wait for a render to finish.")
            job = self.store.create_job(voice, text, language, seed, split_script(text))
            self.cancellations[job["id"]] = threading.Event()
            self.pool.submit(self._generate, job["id"], voice)
            return job

    def cancel(self, job_id):
        with self.lock:
            event = self.cancellations.get(job_id)
            if event:
                event.set()
                job = self.store.job(job_id)
                if job["status"] in ACTIVE:
                    self.store.update_job(job_id, message="Stopping after the current section…")
            return self.store.job(job_id)

    def load(self):
        if self.model is None:
            if not model_downloaded():
                raise RuntimeError(
                    "Download the model first with idiolect download, then try again."
                )
            self.model_status = "loading"
            import torch
            from qwen_tts import Qwen3TTSModel

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable. Use bin/idiolect so the matching NVIDIA libraries are selected."
                )
            torch.set_num_threads(8)
            self.model = Qwen3TTSModel.from_pretrained(
                str(MODEL),
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            self.model_status = "ready"
        return self.model

    def _generate(self, job_id, voice):
        started = time.monotonic()
        cancel = self.cancellations[job_id]
        folder = self.store.render_path(job_id)
        try:
            if cancel.is_set():
                self.store.update_job(job_id, status="cancelled", message="Cancelled")
                return
            self.store.update_job(job_id, status="loading", message="Loading the voice model…")
            model = self.load()
            import torch

            job = self.store.job(job_id)
            with torch.inference_mode():
                if voice["id"] not in self.prompts:
                    self.prompts[voice["id"]] = model.create_voice_clone_prompt(
                        ref_audio=str(self.store.voice_path(voice["id"]) / "reference.wav"),
                        ref_text=voice["transcript"],
                        x_vector_only_mode=False,
                    )
                    if len(self.prompts) > 16:
                        self.prompts.pop(next(iter(self.prompts)))
                prompt = self.prompts[voice["id"]]
                for index, chunk in enumerate(job["chunks"]):
                    if cancel.is_set():
                        break
                    self.store.update_job(
                        job_id,
                        status="generating",
                        message=f"Rendering section {index + 1} of {job['total']}…",
                    )
                    torch.manual_seed(job["seed"] + index)
                    wavs, sr = model.generate_voice_clone(
                        text=chunk,
                        language=job["language"],
                        voice_clone_prompt=prompt,
                        max_new_tokens=2048,
                    )
                    sf.write(folder / f"section-{index:03d}.wav", wavs[0], sr, subtype="PCM_16")
                    self.store.update_job(job_id, completed=index + 1)
            if cancel.is_set():
                self.store.update_job(
                    job_id,
                    status="cancelled",
                    message="Cancelled. Finished sections are saved locally.",
                )
            else:
                duration = join_audio(
                    sorted(folder.glob("section-*.wav")), folder / "voiceover.wav"
                )
                self.store.update_job(
                    job_id,
                    status="ready",
                    message="Your voiceover is ready",
                    duration=duration,
                    elapsed=round(time.monotonic() - started, 1),
                )
        except Exception as error:
            log.exception("Generation failed for %s", job_id)
            if self.model is None:
                self.model_status = "error"
            message = str(error)
            if "out of memory" in message.lower():
                message = "The GPU ran out of memory. Close other GPU apps and try again."
                import torch

                torch.cuda.empty_cache()
            self.store.update_job(job_id, status="failed", message=message[:700])
        finally:
            with self.lock:
                self.cancellations.pop(job_id, None)

    def shutdown(self):
        with self.lock:
            for event in self.cancellations.values():
                event.set()
        self.pool.shutdown(wait=True, cancel_futures=False)
