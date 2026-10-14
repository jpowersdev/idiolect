import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import API_VERSION, DATA, IDLE_TIMEOUT, VERSION, recognizer_downloaded
from .engine import Engine
from .recognizer import Recognizer
from .storage import Store
from .voice import build, parse_takes, summary

# A 90-second take at 48 kHz 16-bit mono is about 8.6 MB.
MAX_TRANSCRIBE_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
RENDER_FIELDS = ("id", "completed", "total", "status", "message", "duration")


def receive(upload, path):
    size = 0
    with path.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ValueError("Each recording must be smaller than 32 MB.")
            output.write(chunk)


def public(job):
    return {key: job[key] for key in RENDER_FIELDS}


class Generation(BaseModel):
    text: str = Field(min_length=1, max_length=12000)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Write a script first")
        return value


def create_app(
    data=DATA,
    engine_factory=Engine,
    recognizer=None,
    request_shutdown=None,
    idle_timeout=IDLE_TIMEOUT,
):
    store = Store(Path(data))
    recognizer = recognizer or Recognizer()
    engine = engine_factory(store, recognizer)

    async def exit_when_idle():
        while True:
            await asyncio.sleep(min(30, idle_timeout))
            idle = time.monotonic() - app.state.last_request
            if idle >= idle_timeout and not engine.active():
                request_shutdown()
                return

    @asynccontextmanager
    async def lifespan(app):
        watcher = asyncio.create_task(exit_when_idle()) if request_shutdown else None
        try:
            yield
        finally:
            if watcher:
                watcher.cancel()
            engine.shutdown()

    app = FastAPI(
        title="Idiolect engine",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.state.engine = engine
    app.state.recognizer = recognizer
    app.state.last_request = time.monotonic()

    @app.middleware("http")
    async def track_activity(request: Request, call_next):
        app.state.last_request = time.monotonic()
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Keep every error a plain {"detail": string} for the CLI's contract.
        messages = [
            f"{'.'.join(str(part) for part in item['loc'][1:])}: {item['msg']}".lstrip(": ")
            for item in error.errors()
        ]
        return JSONResponse({"detail": "; ".join(messages) or "Invalid request"}, 422)

    @app.get("/api/status")
    def status():
        return {
            **engine.status(),
            "recognizer_downloaded": recognizer_downloaded(),
            "service": "idiolect",
            "api": API_VERSION,
            "version": VERSION,
            "pid": os.getpid(),
        }

    @app.post("/api/shutdown", status_code=202)
    def shutdown():
        if engine.active():
            raise HTTPException(409, "The engine has an active render.")
        if request_shutdown is None:
            raise HTTPException(409, "This engine cannot be stopped remotely.")
        request_shutdown()
        return {"stopping": True}

    @app.post("/api/transcribe")
    async def transcribe(request: Request, sample_rate: Annotated[int, Query(ge=8000, le=96000)]):
        """Recognize 16-bit little-endian mono PCM with word timings."""
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_TRANSCRIBE_BYTES:
                raise HTTPException(413, "Send at most 12 MB of audio per transcription.")
        audio = np.frombuffer(bytes(body[: len(body) // 2 * 2]), dtype="<i2")
        try:
            words = await run_in_threadpool(
                recognizer.transcribe, audio.astype(np.float32) / 32768, sample_rate
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"words": words}

    @app.get("/api/voice")
    def voice():
        return summary(saved_voice())

    def saved_voice():
        saved = store.voice()
        if saved is None:
            raise HTTPException(404, "Record your voice first with idiolect setup.")
        return saved

    @app.put("/api/voice")
    def replace_voice(
        takes: Annotated[str, Form(max_length=200_000)],
        audio: Annotated[list[UploadFile], File()],
    ):
        """Replace your voice with a complete set of takes, one audio file per take."""
        try:
            parsed = parse_takes(takes)
        except ValidationError as error:
            raise HTTPException(422, f"Invalid takes: {error.errors()[0]['msg']}") from error
        if len(audio) != len(parsed.root):
            raise HTTPException(422, "Send one recording for each take.")
        try:
            with store.replace_voice() as folder:
                sources = []
                for index, upload in enumerate(audio):
                    source = folder / f"upload-{index}"
                    receive(upload, source)
                    sources.append(source)
                saved = build(folder, parsed, sources)
                for source in sources:
                    source.unlink()
            return summary(saved)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            for upload in audio:
                upload.file.close()

    @app.post("/api/renders", status_code=202)
    def generate(request: Generation):
        saved = saved_voice()
        if not engine.status()["model_downloaded"]:
            raise HTTPException(409, "The speech model is missing. Run idiolect setup first.")
        try:
            return public(engine.submit(saved, request.text))
        except ValueError as error:
            raise HTTPException(429, str(error)) from error

    def find(job_id):
        try:
            return engine.job(job_id)
        except KeyError:
            raise HTTPException(404, "Render not found")

    @app.get("/api/renders/{job_id}")
    def render(job_id: str):
        return public(find(job_id))

    @app.post("/api/renders/{job_id}/cancel")
    def cancel(job_id: str):
        find(job_id)
        return public(engine.cancel(job_id))

    @app.get("/api/renders/{job_id}/audio")
    def render_audio(job_id: str):
        if find(job_id)["status"] != "ready":
            raise HTTPException(409, "This render is not ready yet")
        return FileResponse(engine.audio(job_id), media_type="audio/wav")

    return app
