"""Stand-ins for the speech models, so the HTTP API runs without a GPU."""

from idiolect_engine.audio import split_script
from idiolect_engine.engine import Renders


class IdleEngine:
    """Accepts renders but never runs them."""

    def __init__(self, store, recognizer=None):
        self.store = store
        self.renders = Renders()

    def status(self):
        return {
            "model": "voxcpm2",
            "model_requested": False,
            "model_downloaded": True,
            "model_status": "idle",
        }

    def active(self):
        return self.renders.active() > 0

    def submit(self, voice, text):
        return self.renders.create(text, split_script(text))

    def job(self, job_id):
        return self.renders.get(job_id)

    def audio(self, job_id):
        return self.renders.path(job_id) / "voiceover.wav"

    def cancel(self, job_id):
        return self.renders.update(job_id, status="cancelled")

    def shutdown(self):
        self.renders.close()


class FakeRecognizer:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        return [{"text": " Hello", "start": 0.5, "end": 0.9}]
