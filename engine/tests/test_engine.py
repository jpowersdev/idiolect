import io
import json
import re
import time
from itertools import pairwise
from threading import Event

import numpy as np
import pytest
import soundfile as sf
from fakes import FakeRecognizer, IdleEngine
from fastapi.testclient import TestClient

from idiolect_engine.audio import (
    GAIN_LIMITS_DB,
    PARAGRAPH_PAUSE,
    SENTENCE_PAUSE,
    TARGET_DB,
    coverage,
    cut_after,
    ending_fade,
    ends_sentence,
    join_audio,
    match_level,
    plan_clips,
    speech_end,
    speech_level,
    split_script,
)
from idiolect_engine.engine import Engine, Renders
from idiolect_engine.models import QWEN3, VOXCPM2, choose
from idiolect_engine.recognizer import ASR_SAMPLE_RATE, recognition_audio
from idiolect_engine.server import create_app
from idiolect_engine.storage import Store, exclusive


@pytest.fixture
def client(tmp_path):
    shutdowns = []
    app = create_app(tmp_path, IdleEngine, FakeRecognizer(), lambda: shutdowns.append(True))
    app.state.shutdowns = shutdowns
    with TestClient(app) as client:
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


def timed_words(seconds, sentence=6):
    """A word every half second; every `sentence`th word, and the last, ends a sentence."""
    count = int(seconds * 2)
    return [
        {
            "text": f"word{index}{'.' if index % sentence == sentence - 1 or index == count - 1 else ''}",
            "start": index * 0.5,
            "end": index * 0.5 + 0.4,
        }
        for index in range(count)
    ]


def take(script="comma-1", seconds=20, **changes):
    words = timed_words(seconds)
    return {
        "script": script,
        "transcript": " ".join(word["text"] for word in words),
        "words": words,
    } | changes


def upload(client, takes=None, audio=None):
    takes = takes if takes is not None else [take()]
    if audio is None:
        audio = [wave_bytes(20, rate=48000) for _ in takes]
    return client.put(
        "/api/voice",
        data={"takes": json.dumps(takes)},
        files=[
            ("audio", (f"take-{index}.wav", data, "audio/wav")) for index, data in enumerate(audio)
        ],
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
    sections = split_script(script)
    texts = [section["text"] for section in sections]
    assert texts
    assert all(0 < len(text) <= 300 for text in texts)
    assert re.sub(r"\s+", "", "".join(texts)) == re.sub(r"\s+", "", script)
    assert sections[-1]["pause"] == 0
    if "3.14" in script:
        assert all("3. 14" not in text for text in texts)


def test_paragraphs_get_their_own_sections_and_longer_pauses():
    story = (
        "It never did.\n\n"
        '"You\'re up early," said a voice beside her.\n\n'
        "She turned. " + "An older man was holding a newspaper he had no intention of reading. " * 5
    )
    sections = split_script(story)
    assert [s["text"][:12] for s in sections[:2]] == ["It never did", "\"You're up e"]
    assert [s["pause"] for s in sections[:2]] == [PARAGRAPH_PAUSE, PARAGRAPH_PAUSE]
    # A long paragraph splits at sentences, with the shorter pause between its sections.
    assert len(sections) > 3 and sections[2]["pause"] == SENTENCE_PAUSE
    assert sections[-1]["pause"] == 0
    # A long sentence breaks after a clause, not between two words of it.
    long = (
        "Back then, her grandfather would point out every landmark along the coast: the red "
        "lighthouse, the cliffs where the seals slept, the crooked dock that everyone swore "
        "would fall into the sea by next winter."
    )
    texts = [s["text"] for s in split_script(long)]
    assert len(texts) == 2 and texts[0].endswith(",") and texts[1].endswith("next winter.")
    # Without blank lines, each line is a paragraph; with them, wrapped lines rejoin.
    assert len(split_script("One line.\nAnother line.")) == 2
    assert len(split_script("A paragraph that\nwraps.\n\nAnother.")) == 2


def test_the_end_of_speech_ignores_trailing_babble():
    heard = [
        {"text": " A", "start": 0.0, "end": 0.2},
        {"text": " completely", "start": 0.2, "end": 0.8},
        {"text": " new", "start": 0.8, "end": 1.0},
        {"text": " story.", "start": 1.0, "end": 1.4},
        {"text": " Sorry.", "start": 1.4, "end": 1.6},
        {"text": " Sorry.", "start": 1.6, "end": 1.8},
    ]
    assert speech_end("A familiar voice can tell a completely new story.", heard) == 1.4
    # The recognizer's spelling of a name still counts.
    assert speech_end("She walked with Sarah.", [{"text": "walked with Sara.", "end": 2.0}]) == 2.0
    # Garbled before the end: no end found.
    assert speech_end("A completely new story.", heard[:2]) is None
    assert coverage("A completely new story.", heard) == 1.0
    assert coverage("A completely new story.", heard[:2]) == 0.5


def test_takes_are_kept_and_cut_into_clips(client):
    assert client.get("/api/voice").status_code == 404
    takes = [take(), take("technical", 30, notice="Copyright notice.")]
    response = upload(client, takes, [wave_bytes(20, rate=48000), wave_bytes(30, rate=48000)])
    assert response.status_code == 200
    assert response.json()["takes"] == [
        {"script": "comma-1", "duration": 20},
        {"script": "technical", "duration": 30},
    ]
    assert client.get("/api/voice").json() == response.json()
    store = client.app.state.store
    saved = store.voice()
    # Whole takes stay at the recorded rate, for training later.
    data, sr = sf.read(store.path("takes/comma-1.wav"))
    assert sr == 48000 and len(data) == 20 * sr
    assert saved["takes"][1]["notice"] == "Copyright notice."
    assert saved["takes"][0]["words"] == takes[0]["words"]
    # Clips are 24 kHz for the model, end at sentences, and keep their exact words.
    assert {clip["script"] for clip in saved["clips"]} == {"comma-1", "technical"}
    for clip in saved["clips"]:
        data, sr = sf.read(store.path(clip["file"]))
        assert sr == 24000 and 3 <= len(data) / sr <= 15.5
        assert clip["text"].endswith(".")
    assert sorted(path.name for path in store.folder.iterdir()) == ["clips", "takes", "voice.json"]


def test_clips_end_at_natural_boundaries():
    words = timed_words(40)
    spans = plan_clips(words)
    assert spans[0][0] == 0 and spans[-1][1] == len(words) - 1
    assert all(later[0] == earlier[1] + 1 for earlier, later in pairwise(spans))
    for first, last in spans:
        assert 3 <= words[last]["end"] - words[first]["start"] <= 15
        assert words[last]["text"].endswith(".")
    # Without a sentence end in range, a clip stretches to reach one rather than stop at a comma.
    long_sentence = [
        {
            "text": f"word{i}{',' if i == 25 else '.' if i == 35 else ''}",
            "start": i * 0.5,
            "end": i * 0.5 + 0.4,
        }
        for i in range(36)
    ]
    assert plan_clips(long_sentence) == [(0, 35)]
    assert ends_sentence("bird.") and ends_sentence('"Comma?"') and not ends_sentence("Mrs.")
    # A single word longer than a clip still moves the plan forward.
    assert plan_clips([{"text": "Hmm.", "start": 0, "end": 20}]) == [(0, 0)]


def test_a_new_recording_replaces_the_voice(client):
    upload(client, [take("comma-1")])
    upload(client, [take("technical")])
    assert [t["script"] for t in client.get("/api/voice").json()["takes"]] == ["technical"]
    assert sorted(path.name for path in client.app.state.store.root.iterdir()) == ["voice"]


@pytest.mark.parametrize(
    "audio",
    [wave_bytes(1), wave_bytes(91), wave_bytes(20, silent=True), b"not audio"],
    ids=["too-short", "too-long", "silent", "invalid-file"],
)
def test_bad_recordings_keep_the_previous_voice(client, audio):
    previous = upload(client).json()
    assert upload(client, [take(), take("technical")], [wave_bytes(20), audio]).status_code == 422
    assert client.get("/api/voice").json() == previous
    assert sorted(path.name for path in client.app.state.store.root.iterdir()) == ["voice"]


@pytest.mark.parametrize(
    "takes",
    [[], [take(), take()], [take(transcript="   ")], [take(script="../escape")], [take(words=[])]],
    ids=["none", "repeated-script", "blank-transcript", "unsafe-script", "no-words"],
)
def test_invalid_takes_are_rejected(client, takes):
    assert upload(client, takes, [wave_bytes(20)] * len(takes)).status_code == 422
    assert client.get("/api/voice").status_code == 404


def test_every_take_needs_a_recording(client):
    assert upload(client, [take(), take("technical")], [wave_bytes(20)]).status_code == 422


def test_render_needs_a_voice_and_valid_text(client):
    assert client.post("/api/renders", json={"text": "Hello."}).status_code == 404
    upload(client)
    for invalid in ("   ", "x" * 12001):
        assert client.post("/api/renders", json={"text": invalid}).status_code == 422
    response = client.post("/api/renders", json={"text": "A proper script."})
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued" and job["total"] == 1
    assert client.get(f"/api/renders/{job['id']}").json()["id"] == job["id"]
    assert client.get(f"/api/renders/{job['id']}/audio").status_code == 409
    assert client.post(f"/api/renders/{job['id']}/cancel").json()["status"] == "cancelled"
    assert client.get("/api/renders/missing").status_code == 404


def test_finished_renders_are_pruned(tmp_path):
    renders = Renders()
    try:
        jobs = [renders.create("Hi.", ["Hi."]) for _ in range(Renders.KEEP + 3)]
        for job in jobs:
            renders.update(job["id"], status="ready")
        renders.create("One more.", ["One more."])
        assert len(renders.jobs) == Renders.KEEP + 1
        assert not renders.path(jobs[0]["id"]).exists()
    finally:
        renders.close()
    assert not renders.folder.exists()


def test_join_audio_trims_each_section_and_adds_its_pause(tmp_path):
    a, b, out = [tmp_path / name for name in ("a.wav", "b.wav", "out.wav")]
    silence = np.zeros(12000)
    sf.write(a, np.concatenate([silence, np.full(24000, 0.2), silence]), 24000)
    sf.write(b, np.full(24000, -0.2), 24000)
    duration = join_audio([a, b], [0.75, 0.0], out)
    samples, sr = sf.read(out)
    # Each second of speech keeps its margins; the planned pause sits between them.
    assert sr == 24000 and 2.9 <= duration <= 3.0
    assert np.all(samples[30000 : 30000 + 16000] == 0)
    sf.write(b, np.ones(16000), 16000)
    with pytest.raises(ValueError, match="sample rates"):
        join_audio([a, b], [0.3, 0], out)


def test_garbled_sections_are_rendered_again(tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, text, seed):
            """Speech, a short gap, then a stray syllable."""
            self.calls += 1
            level = 0.1 * self.calls
            speech, gap, stray = np.full(14400, level), np.zeros(4800), np.full(4800, level)
            return np.concatenate([speech, gap, stray]).astype(np.float32), 24000

    class Recognizer:
        """Hears garbling on the first attempt, then the whole text and a trailing syllable."""

        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, sample_rate):
            self.calls += 1
            if self.calls == 1:
                return [{"text": "A completely nyuh", "start": 0, "end": 0.8}]
            return [
                {"text": "A completely new story.", "start": 0, "end": 0.6},
                {"text": "Sorry.", "start": 0.6, "end": 0.9},
            ]

    text = "A completely new story."
    engine = Engine(Store(tmp_path), Recognizer())
    job = engine.renders.create(text, split_script(text))
    model = Model()
    try:
        wav, sr = engine._render_section(model, None, text, 0, 1, job["id"], Event())
    finally:
        engine.shutdown()
    assert model.calls == 2
    # The second attempt, cut in the gap after "story.", before the stray syllable.
    assert np.allclose(wav[:14400], 0.2) and len(wav) == int(0.65 * sr)


def test_two_engines_cannot_use_the_same_data(tmp_path):
    with (
        exclusive(tmp_path),
        pytest.raises(RuntimeError, match="already running"),
        exclusive(tmp_path),
    ):
        pass


def test_errors_are_plain_messages(client):
    upload(client)
    response = client.post("/api/renders", json={"text": "   "})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str) and "text" in response.json()["detail"]


def test_status_reports_contract_version(client):
    status = client.get("/api/status").json()
    assert status["service"] == "idiolect" and status["api"] == 1
    assert "recognizer_downloaded" in status


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
    upload(client)
    job = client.post("/api/renders", json={"text": "Hi."}).json()
    assert client.post("/api/shutdown").status_code == 409
    client.post(f"/api/renders/{job['id']}/cancel")
    assert client.post("/api/shutdown").status_code == 202
    assert client.app.state.shutdowns == [True]


def test_an_idle_engine_exits(tmp_path):
    shutdowns = []
    app = create_app(tmp_path, IdleEngine, FakeRecognizer(), lambda: shutdowns.append(True), 0.1)
    with TestClient(app):
        time.sleep(0.5)
    assert shutdowns == [True]


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
    finally:
        engine.shutdown()


def test_cutting_after_speech_keeps_a_soft_final_consonant():
    rate = 24000
    # A word, then a quiet "s" that runs past the recognizer's estimated end, then silence.
    word, hiss, silence = np.full(9600, 0.5), np.full(3600, 0.02), np.zeros(9600)
    audio = np.concatenate([word, hiss, silence]).astype(np.float32)
    cut = cut_after(audio, rate, end=0.4)
    assert len(cut) >= len(word) + len(hiss)
    assert len(cut) <= len(word) + len(hiss) + int(0.1 * rate)


def test_a_fading_ending_is_detected():
    rate = 24000
    levels = [0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.02, 0.005]
    audio = np.concatenate([np.full(rate // 4, level) for level in levels]).astype(np.float32)
    words = [{"text": "w", "start": i * 0.25, "end": (i + 1) * 0.25} for i in range(len(levels))]
    assert ending_fade(audio, rate, words) > 12
    steady = np.full(rate * 2, 0.3, dtype=np.float32)
    assert abs(ending_fade(steady, rate, words)) < 1


def test_sections_are_brought_to_a_common_level_within_limits():
    rate = 24000
    quiet = np.full(rate, 10 ** (-36 / 20), dtype=np.float32)
    assert abs(speech_level(match_level(quiet, rate), rate) - TARGET_DB) < 0.5
    # A nearly silent section is lifted by at most the limit, not blown up.
    whisper = np.full(rate, 10 ** (-60 / 20), dtype=np.float32)
    assert abs(speech_level(match_level(whisper, rate), rate) - (-60 + GAIN_LIMITS_DB[1])) < 0.5


def test_the_best_model_that_fits_is_chosen():
    assert choose({}, memory=48).model is VOXCPM2
    assert choose({}, memory=24).model is VOXCPM2
    assert choose({}, memory=16).model is QWEN3
    small = choose({}, memory=8)
    assert small.model is QWEN3 and "at least 12 GB" in small.warning


def test_a_model_can_be_chosen_by_name():
    forced = choose({"IDIOLECT_MODEL": "voxcpm2"}, memory=16)
    assert forced.model is VOXCPM2 and forced.requested and not choose({}, memory=48).requested
    assert "needs 24 GB" in forced.warning
    assert choose({"IDIOLECT_MODEL": " Qwen3 "}, memory=48).model is QWEN3
    with pytest.raises(ValueError, match="voxcpm2, qwen3"):
        choose({"IDIOLECT_MODEL": "elevenlabs"}, memory=48)
