import io
import re

import numpy as np
import pytest
import soundfile as sf
from fakes import FakeRecognizer, IdleEngine
from fastapi.testclient import TestClient

from idiolect_engine.audio import join_audio, split_script
from idiolect_engine.config import MODEL_ID
from idiolect_engine.engine import Engine
from idiolect_engine.recognizer import ASR_SAMPLE_RATE, recognition_audio
from idiolect_engine.server import create_app
from idiolect_engine.storage import Store

LOCAL = "http://127.0.0.1:7860"


@pytest.fixture
def client(tmp_path):
    shutdowns = []
    app = create_app(tmp_path, IdleEngine, FakeRecognizer(), lambda: shutdowns.append(True))
    app.state.shutdowns = shutdowns
    with TestClient(app, base_url=LOCAL) as client:
        yield client


def wave_bytes(seconds=4, silent=False, rate=24000):
    buffer = io.BytesIO()
    samples = (
        np.zeros(int(seconds * rate))
        if silent
        else (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * rate)) / rate))
    )
    sf.write(buffer, samples, rate, format="WAV")
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


def test_recorded_48k_reference_is_converted_for_the_model(client):
    voice = upload(client, wave_bytes(5, rate=48000)).json()
    data, sr = sf.read(client.app.state.store.voice_path(voice["id"]) / "reference.wav")
    assert sr == 24000 and len(data) == 5 * sr and voice["duration"] == 5


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
        TestClient(create_app(tmp_path, IdleEngine), base_url=LOCAL),
        pytest.raises(RuntimeError, match="already running"),
        TestClient(create_app(tmp_path, IdleEngine), base_url=LOCAL),
    ):
        pass


def test_errors_are_plain_messages(client):
    response = client.post("/api/generate", json={"voice_id": "0" * 32, "text": "Hi", "seed": -1})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str) and "seed" in response.json()["detail"]


def test_status_reports_contract_version(client):
    status = client.get("/api/status").json()
    assert status["service"] == "idiolect" and status["api"] == 1
    assert "recognizer_downloaded" in status


def test_browser_and_rebound_requests_are_refused(tmp_path):
    with TestClient(create_app(tmp_path, IdleEngine), base_url="http://evil.example") as other:
        assert other.get("/api/status").status_code == 403
    with TestClient(create_app(tmp_path, IdleEngine), base_url=LOCAL) as local:
        response = local.post("/api/shutdown", headers={"Origin": "https://evil.example"})
        assert response.status_code == 403


def test_transcribe_accepts_raw_pcm(client):
    pcm = (np.full(48000, 0.25) * 32768).astype("<i2").tobytes()
    response = client.post(
        "/api/transcribe",
        params={"sample_rate": 48000},
        content=pcm,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.json() == {"words": [{"text": " Hello", "start": 0.5, "end": 0.9}]}
    audio, rate = client.app.state.recognizer.calls[-1]
    assert rate == 48000 and len(audio) == 48000 and np.allclose(audio, 0.25)


def test_shutdown_waits_for_active_renders(client):
    voice = upload(client).json()
    job = client.post("/api/generate", json={"voice_id": voice["id"], "text": "Hi."}).json()
    assert client.post("/api/shutdown").status_code == 409
    client.post(f"/api/renders/{job['id']}/cancel")
    assert client.post("/api/shutdown").status_code == 202
    assert client.app.state.shutdowns == [True]


def test_recognition_copy_is_anti_aliased():
    rate = 48000
    time = np.arange(rate * 6) / rate
    audio = (0.2 * np.sin(2 * np.pi * 500 * time) + 0.1 * np.sin(2 * np.pi * 10000 * time)).astype(
        np.float32
    )
    original = audio.copy()
    recognition = recognition_audio(audio, rate)
    assert len(recognition) == ASR_SAMPLE_RATE * 6
    assert np.array_equal(audio, original)
    # A proper downsample removes the 10 kHz tone rather than folding it into ASR's band.
    expected = 0.2 * np.sin(2 * np.pi * 500 * np.arange(len(recognition)) / ASR_SAMPLE_RATE)
    assert np.sqrt(np.mean((recognition[100:-100] - expected[100:-100]) ** 2)) < 0.001


def test_real_engine_status_matches_the_stand_in(tmp_path):
    """The CLI decodes these fields; keep the GPU engine and its stand-in in step."""
    engine = Engine(Store(tmp_path))
    try:
        assert engine.status().keys() == IdleEngine(None).status().keys()
        assert engine.status()["model"] == MODEL_ID
    finally:
        engine.shutdown()
