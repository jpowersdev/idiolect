import io
import re

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from idiolect.audio import join_audio, split_script
from idiolect.engine import Engine
from idiolect.server import create_app
from idiolect.storage import Store


class IdleEngine:
    """Exercise HTTP validation/storage without running the GPU during unit tests."""

    def __init__(self, store):
        self.store = store

    def status(self):
        return {"model_downloaded": True, "model_status": "idle", "local": True}

    def recover(self):
        pass

    def submit(self, voice, text, language, seed):
        return self.store.create_job(voice, text, language, seed, split_script(text))

    def cancel(self, job_id):
        return self.store.update_job(job_id, status="cancelled")

    def shutdown(self):
        pass


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, IdleEngine)) as client:
        yield client


def wave_bytes(seconds=4, silent=False):
    buffer = io.BytesIO()
    samples = (
        np.zeros(int(seconds * 24000))
        if silent
        else (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * 24000)) / 24000))
    )
    sf.write(buffer, samples, 24000, format="WAV")
    return buffer.getvalue()


def upload(client, audio=None, transcript="A short reference recording."):
    return client.post(
        "/api/voices",
        data={"name": "Test voice", "transcript": transcript},
        files={"audio": ("sample.wav", audio or wave_bytes(), "audio/wav")},
    )


@pytest.mark.parametrize(
    "script",
    [
        "The value is 3.14. That's useful! " * 35,
        "A sentence without punctuation " * 100,
        "你好世界。这里是一个声音测试！" * 100,
        "a" * 950,
        "First paragraph.\n\nSecond paragraph.\nThird paragraph?",
    ],
)
def test_script_chunks_preserve_all_content_and_bound_size(script):
    chunks = split_script(script)
    assert chunks
    assert all(0 < len(chunk) <= 300 for chunk in chunks)
    assert re.sub(r"\s+", "", "".join(chunks)) == re.sub(r"\s+", "", script)
    if "3.14" in script:
        assert all("3. 14" not in chunk for chunk in chunks)


def test_voice_persists_and_normalized_audio_is_readable(client):
    response = upload(client)
    assert response.status_code == 201
    voice = response.json()
    assert client.get("/api/voices").json()[0]["id"] == voice["id"]
    reference = client.app.state.store.voice_path(voice["id"]) / "reference.wav"
    data, sr = sf.read(reference)
    assert sr == 24000 and data.ndim == 1 and len(data) == 4 * sr


@pytest.mark.parametrize(
    "audio",
    [wave_bytes(1), wave_bytes(31), wave_bytes(silent=True), b"not audio"],
    ids=["too-short", "too-long", "silent", "invalid-file"],
)
def test_bad_references_leave_no_saved_voice(client, audio):
    assert upload(client, audio).status_code == 422
    assert client.get("/api/voices").json() == []
    assert not list(client.app.state.store.voices.iterdir())


def test_blank_transcript_rejected(client):
    assert upload(client, transcript="   ").status_code == 422
    assert client.get("/api/voices").json() == []


def test_render_validates_input_and_preserves_script(client):
    voice = upload(client).json()
    request = {
        "voice_id": voice["id"],
        "text": "A proper script.",
        "language": "English",
        "seed": 12,
    }
    for invalid in ({"text": "   "}, {"text": "x" * 12001}, {"language": "Elvish"}, {"seed": -1}):
        assert client.post("/api/generate", json=request | invalid).status_code == 422
    assert client.post("/api/generate", json=request | {"voice_id": "0" * 32}).status_code == 404
    response = client.post("/api/generate", json=request)
    assert response.status_code == 202
    job = response.json()
    assert client.get(f"/api/renders/{job['id']}").json()["text"] == request["text"]
    assert client.get(f"/api/renders/{job['id']}/audio").status_code == 409
    assert client.post(f"/api/renders/{job['id']}/cancel").json()["status"] == "cancelled"


def test_ids_cannot_escape_data_directory(client):
    for voice_id in ("../../etc/passwd", "not-an-id"):
        assert (
            client.post("/api/generate", json={"voice_id": voice_id, "text": "Hello."}).status_code
            == 404
        )
    assert client.get("/api/renders/not-an-id/audio").status_code == 404


def test_restart_marks_unfinished_renders_interrupted(tmp_path):
    store = Store(tmp_path)
    job = store.create_job({"id": "0" * 32, "name": "Test"}, "Hello", "English", 12, ["Hello"])
    engine = Engine(store)
    try:
        engine.recover()
        assert store.job(job["id"])["status"] == "failed"
        assert "restart" in store.job(job["id"])["message"]
    finally:
        engine.shutdown()


def test_join_audio_preserves_sections_and_adds_pause(tmp_path):
    a, b, out = [tmp_path / name for name in ("a.wav", "b.wav", "out.wav")]
    sf.write(a, np.full(24000, 0.2), 24000)
    sf.write(b, np.full(24000, -0.2), 24000)
    assert join_audio([a, b], out) == 2.18
    samples, sr = sf.read(out)
    assert sr == 24000 and len(samples) == 52320
    assert np.all(samples[24000:28320] == 0)
    sf.write(b, np.ones(16000), 16000)
    with pytest.raises(ValueError, match="sample rates"):
        join_audio([a, b], out)


def test_two_servers_cannot_use_the_same_voice_data(tmp_path):
    with (
        TestClient(create_app(tmp_path, IdleEngine)),
        pytest.raises(RuntimeError, match="already running"),
        TestClient(create_app(tmp_path, IdleEngine)),
    ):
        pass
