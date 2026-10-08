import fcntl
import os
import shutil
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .audio import prepare_reference
from .config import DATA, LANGUAGES
from .engine import Engine
from .storage import Store, timestamp, write_json


class Generation(BaseModel):
    voice_id: str
    text: str = Field(min_length=1, max_length=12000)
    language: str = "English"
    seed: int = Field(default=12, ge=0, le=2**31 - 1)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Write a script first")
        return value

    @field_validator("language")
    @classmethod
    def language_supported(cls, value):
        if value not in LANGUAGES:
            raise ValueError("Unsupported language")
        return value


def create_app(data=DATA, engine_factory=Engine):
    store = Store(Path(data))
    engine = engine_factory(store)

    @asynccontextmanager
    async def lifespan(app):
        # Keep the library lock compatible with existing Idiolect checkouts.
        with (store.root / "studio.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "Idiolect is already running with this data directory."
                ) from error
            try:
                engine.recover()
                yield
            finally:
                engine.shutdown()
                fcntl.flock(lock, fcntl.LOCK_UN)

    app = FastAPI(
        title="Idiolect engine",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.state.engine = engine

    @app.get("/api/status")
    def status():
        return {
            **engine.status(),
            "languages": LANGUAGES,
            "service": "idiolect",
            "data_directory": str(store.root.resolve()),
            "pid": os.getpid(),
        }

    @app.get("/api/voices")
    def voices():
        return store.list_voices()

    @app.post("/api/voices", status_code=201)
    def add_voice(
        name: Annotated[str, Form(min_length=1, max_length=60)],
        transcript: Annotated[str, Form(min_length=1, max_length=3000)],
        audio: Annotated[UploadFile, File()],
    ):
        name, transcript = name.strip(), transcript.strip()
        if not name or not transcript:
            raise HTTPException(422, "Add a voice name and the exact words in your recording.")
        voice_id = uuid.uuid4().hex
        folder = store.voice_path(voice_id)
        folder.mkdir()
        source = folder / "upload"
        try:
            size = 0
            with source.open("wb") as output:
                while chunk := audio.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 32 * 1024 * 1024:
                        raise ValueError("Reference audio must be smaller than 32 MB.")
                    output.write(chunk)
            duration = prepare_reference(source, folder / "reference.wav")
            voice = {
                "id": voice_id,
                "name": name,
                "transcript": transcript,
                "duration": duration,
                "created_at": timestamp(),
            }
            with store.lock:
                write_json(folder / "voice.json", voice)
            return voice
        except (ValueError, subprocess.TimeoutExpired) as error:
            shutil.rmtree(folder)
            raise HTTPException(422, str(error)) from error
        except Exception:
            shutil.rmtree(folder)
            raise
        finally:
            source.unlink(missing_ok=True)
            audio.file.close()

    @app.post("/api/generate", status_code=202)
    def generate(request: Generation):
        try:
            voice = store.voice(request.voice_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Choose a saved voice first")
        if not engine.status()["model_downloaded"]:
            raise HTTPException(
                409, "The model is not downloaded yet. Run idiolect download first."
            )
        try:
            return engine.submit(voice, request.text, request.language, request.seed)
        except ValueError as error:
            raise HTTPException(429, str(error)) from error

    @app.get("/api/renders")
    def renders():
        return store.list_jobs()

    @app.get("/api/renders/{job_id}")
    def render(job_id: str):
        try:
            return store.job(job_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Render not found")

    @app.post("/api/renders/{job_id}/cancel")
    def cancel(job_id: str):
        render(job_id)
        return engine.cancel(job_id)

    @app.get("/api/renders/{job_id}/audio")
    def render_audio(job_id: str):
        job = render(job_id)
        if job["status"] != "ready":
            raise HTTPException(409, "This render is not ready yet")
        return FileResponse(
            store.render_path(job_id) / "voiceover.wav",
            media_type="audio/wav",
        )

    return app
