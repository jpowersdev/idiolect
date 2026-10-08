"""Stand-ins for the speech models, so the HTTP API runs without a GPU."""

from idiolect_engine.audio import split_script
from idiolect_engine.config import MODEL_ID


class IdleEngine:
    def __init__(self, store):
        self.store = store

    def status(self):
        return {"model": MODEL_ID, "model_downloaded": True, "model_status": "idle", "local": True}

    def recover(self):
        pass

    def submit(self, voice, text, language, seed):
        return self.store.create_job(voice, text, language, seed, split_script(text))

    def cancel(self, job_id):
        return self.store.update_job(job_id, status="cancelled")

    def shutdown(self):
        pass


class FakeRecognizer:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        return [{"text": " Hello", "start": 0.5, "end": 0.9}]
