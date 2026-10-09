import json
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


def timestamp():
    return datetime.now(UTC).isoformat()


def identifier(value):
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Invalid identifier")
    return value


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.voices = root / "voices"
        self.renders = root / "renders"
        self.lock = threading.RLock()
        self.voices.mkdir(parents=True, exist_ok=True)
        self.renders.mkdir(parents=True, exist_ok=True)

    def voice_path(self, voice_id):
        return self.voices / identifier(voice_id)

    def render_path(self, render_id):
        return self.renders / identifier(render_id)

    def list_voices(self):
        with self.lock:
            return sorted(
                (json.loads(p.read_text()) for p in self.voices.glob("*/voice.json")),
                key=lambda v: v["created_at"],
                reverse=True,
            )

    def voice(self, voice_id):
        return json.loads((self.voice_path(voice_id) / "voice.json").read_text())

    def list_jobs(self):
        with self.lock:
            return sorted(
                (json.loads(p.read_text()) for p in self.renders.glob("*/job.json")),
                key=lambda j: j["created_at"],
                reverse=True,
            )

    def job(self, job_id):
        with self.lock:
            return json.loads((self.render_path(job_id) / "job.json").read_text())

    def update_job(self, job_id, **changes):
        with self.lock:
            job = self.job(job_id)
            job.update(changes, updated_at=timestamp())
            write_json(self.render_path(job_id) / "job.json", job)
            return job

    def create_job(self, voice, text, language, seed, chunks):
        job_id = uuid.uuid4().hex
        folder = self.render_path(job_id)
        folder.mkdir()
        job = {
            "id": job_id,
            "voice_id": voice["id"],
            "voice_name": voice["name"],
            "text": text,
            "language": language,
            "seed": seed,
            "chunks": chunks,
            "completed": 0,
            "total": len(chunks),
            "status": "queued",
            "message": "Waiting for the speech engine",
            "duration": None,
            "created_at": timestamp(),
            "updated_at": timestamp(),
            "elapsed": None,
        }
        with self.lock:
            write_json(folder / "job.json", job)
            (folder / "script.txt").write_text(text, encoding="utf-8")
        return job
