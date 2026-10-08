import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf

from idiolect import cli
from idiolect.client import Backend
from idiolect.reading import (
    ASR_SAMPLE_RATE,
    SAMPLE_RATE,
    Word,
    match_reading,
    recognition_audio,
    save_reading,
)


def words(text, start=1.0):
    return [
        Word(word, start + index * 0.5, start + (index + 1) * 0.5)
        for index, word in enumerate(text.split())
    ]


def test_reading_requires_all_words_in_order():
    script = "A warm cup of coffee waited beside a blue notebook."
    assert match_reading(script, words(script.upper())).complete
    for incomplete in (
        "",
        "A warm cup of coffee",
        "beside a blue notebook",
        "A warm cup of coffee waited beside a notebook",
        "A warm cup of coffee waited beside a red notebook",
        "notebook blue a beside waited coffee of cup warm A",
    ):
        assert not match_reading(script, words(incomplete)).complete


def test_reading_reports_missing_word_even_when_ending_was_spoken():
    match = match_reading("A quiet morning by the river.", words("A morning by the river."))
    assert match.matched == 5
    assert match.total == 6
    assert match.next_words.startswith("quiet")
    assert not match.complete


def test_saved_reference_uses_heard_fillers_and_trims_outside_speech(tmp_path):
    script = "A quiet morning beside the blue river."
    heard = words("Ready. A quiet morning um beside the blue river. Thanks.")
    match = match_reading(script, heard)
    assert match.complete
    audio = np.full(SAMPLE_RATE * 9, 0.1, dtype=np.float32)
    path = tmp_path / "reference.wav"
    transcript, duration = save_reading(audio, heard, match, path)
    assert transcript == "A quiet morning um beside the blue river."
    assert duration == 4.6
    saved, rate = sf.read(path)
    assert rate == SAMPLE_RATE and len(saved) == round(4.6 * SAMPLE_RATE)


def test_recognition_copy_does_not_discard_detail_from_saved_voice(tmp_path):
    time = np.arange(SAMPLE_RATE * 6) / SAMPLE_RATE
    audio = (0.2 * np.sin(2 * np.pi * 500 * time) + 0.1 * np.sin(2 * np.pi * 10000 * time)).astype(
        np.float32
    )
    original = audio.copy()
    recognition = recognition_audio(audio)
    assert len(recognition) == ASR_SAMPLE_RATE * 6
    assert np.array_equal(audio, original)
    # A proper downsample removes the 10 kHz tone rather than folding it into ASR's band.
    expected = 0.2 * np.sin(2 * np.pi * 500 * np.arange(len(recognition)) / ASR_SAMPLE_RATE)
    assert np.sqrt(np.mean((recognition[100:-100] - expected[100:-100]) ** 2)) < 0.001

    heard = words("A quiet morning beside the blue river.")
    output = tmp_path / "reference.wav"
    save_reading(
        audio, heard, match_reading("A quiet morning beside the blue river.", heard), output
    )
    saved, rate = sf.read(output)
    assert rate == 48000
    spectrum = np.abs(np.fft.rfft(saved))
    frequencies = np.fft.rfftfreq(len(saved), 1 / rate)
    high = spectrum[np.argmin(np.abs(frequencies - 10000))]
    low = spectrum[np.argmin(np.abs(frequencies - 500))]
    assert 0.45 < high / low < 0.55


def test_incomplete_reading_never_writes_a_reference(tmp_path):
    path = tmp_path / "reference.wav"
    heard = words("A quiet morning")
    with pytest.raises(ValueError, match="full script"):
        save_reading(
            np.ones(SAMPLE_RATE * 5),
            heard,
            match_reading("A quiet morning beside the river.", heard),
            path,
        )
    assert not path.exists()


def test_cli_text_sources_and_validation(monkeypatch, tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("A script from a file.\n")
    assert cli.input_text(SimpleNamespace(text=["Hello", "there"], file=None)) == "Hello there"
    assert cli.input_text(SimpleNamespace(text=[], file=str(script))) == "A script from a file."
    monkeypatch.setattr("sys.stdin", io.StringIO("A piped script."))
    assert cli.input_text(SimpleNamespace(text=[], file=None)) == "A piped script."
    with pytest.raises(ValueError, match="rather than both"):
        cli.input_text(SimpleNamespace(text=["Hello"], file=str(script)))
    with pytest.raises(ValueError, match="12,000"):
        cli.input_text(SimpleNamespace(text=["x" * 12001], file=None))


def test_voice_defaults_and_ambiguous_names(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DATA", tmp_path)
    voices = [{"id": "abcd1111", "name": "Me"}, {"id": "efgh2222", "name": "Me"}]
    with pytest.raises(ValueError, match="Set up"):
        cli.choose_voice(voices)
    cli.set_default(voices[1])
    assert cli.choose_voice(voices) == voices[1]
    assert cli.choose_voice(voices, "abcd") == voices[0]
    with pytest.raises(ValueError, match="ambiguous"):
        cli.choose_voice(voices, "Me")


@pytest.mark.parametrize("failure", [None, "generation", "playback"])
def test_enrollment_saves_voice_and_previews_it(monkeypatch, tmp_path, capsys, failure):
    monkeypatch.setattr(cli, "DATA", tmp_path)
    monkeypatch.setattr(cli, "ASR_MODEL", tmp_path)
    (tmp_path / "model.bin").touch()
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(cli, "Reader", lambda: object())
    monkeypatch.setattr(cli, "reading_script", lambda: "A quiet morning beside the blue river.")
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/test/pw-play")
    rendered = []
    played = []

    def record(script, path, reader, progress, device):
        path.write_bytes(b"recorded wav")
        return "A quiet morning um beside the blue river.", 12.0

    class Server:
        def request(self, method, path, *, data, files):
            assert method == "POST" and path == "/api/voices"
            assert data["transcript"] == "A quiet morning um beside the blue river."
            assert files["audio"][1].read() == b"recorded wav"
            return {"id": "abc123", "name": data["name"]}

        def render(self, voice_id, text, language, seed, progress):
            assert json.loads((tmp_path / "cli.json").read_text())["default_voice"] == voice_id
            assert voice_id == "abc123"
            assert text == cli.PREVIEW_TEXT and text != cli.reading_script()
            rendered.append(text)
            if failure == "generation":
                raise ValueError("Model unavailable")
            return {"id": "preview123", "duration": 4.0}

        def download(self, job_id, output):
            assert job_id == "preview123"
            output.write_bytes(b"preview wav")

    def play(command, **kwargs):
        audio = Path(command[1])
        assert audio.read_bytes() == b"preview wav"
        played.append(audio)
        return SimpleNamespace(
            returncode=1 if failure == "playback" else 0,
            stderr="No audio device" if failure == "playback" else "",
        )

    monkeypatch.setattr(cli, "record_reading", record)
    monkeypatch.setattr(cli.subprocess, "run", play)
    cli.enroll(Server(), SimpleNamespace(name="My voice", device=None))
    assert json.loads((tmp_path / "cli.json").read_text())["default_voice"] == "abc123"
    captured = capsys.readouterr()
    assert "Complete script" in captured.out
    assert rendered == [cli.PREVIEW_TEXT]
    assert len(played) == (0 if failure == "generation" else 1)
    assert all(not audio.exists() for audio in played)
    if failure:
        assert "Your voice is saved, but the preview could not finish" in captured.err
    else:
        assert "Playing 4s of audio" in captured.err


def test_say_preserves_existing_output_before_starting_engine(tmp_path):
    target = tmp_path / "existing.wav"
    target.write_bytes(b"original audio")
    args = cli.parser().parse_args(["say", "Hello", "-o", str(target)])
    with pytest.raises(ValueError, match="already exists"):
        cli.say(None, args)
    assert target.read_bytes() == b"original audio"


@pytest.mark.parametrize("save", [False, True], ids=["play-by-default", "save-with-output"])
def test_say_plays_or_saves_audio(monkeypatch, tmp_path, capsys, save):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "speech.wav").write_bytes(b"keep this existing file")
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/test/pw-play")
    played = []

    def play(command, **kwargs):
        assert command[0] == "/test/pw-play"
        audio = Path(command[1])
        assert audio.read_bytes() == b"generated audio"
        played.append(audio)
        return SimpleNamespace(returncode=0, stderr="")

    class Server:
        def ensure(self):
            pass

        def request(self, method, path):
            return [{"id": "voice123", "name": "Me"}]

        def render(self, voice_id, text, language, seed, progress):
            assert voice_id == "voice123" and text == "Hello"
            return {"id": "take123", "duration": 1.5}

        def download(self, job_id, output, force=False):
            assert job_id == "take123"
            output.write_bytes(b"generated audio")

    monkeypatch.setattr(cli.subprocess, "run", play)
    options = ["say", "Hello", "--voice", "voice123"]
    if save:
        options += ["-o", "saved.wav"]
    cli.say(Server(), cli.parser().parse_args(options))
    captured = capsys.readouterr()
    assert (tmp_path / "speech.wav").read_bytes() == b"keep this existing file"
    if save:
        assert not played
        assert (tmp_path / "saved.wav").read_bytes() == b"generated audio"
        assert captured.out.strip() == str(tmp_path / "saved.wav")
    else:
        assert len(played) == 1 and not played[0].exists()
        assert "Playing 1.5s" in captured.err and captured.out == ""
        assert list(tmp_path.iterdir()) == [tmp_path / "speech.wav"]


def test_say_checks_for_player_before_starting_engine(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="Playback needs"):
        cli.say(None, cli.parser().parse_args(["say", "Hello"]))


@pytest.mark.parametrize("interrupted", [False, True], ids=["player-error", "ctrl-c"])
def test_playback_failure_cleans_up_temporary_audio(monkeypatch, interrupted):
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/test/pw-play")
    played = []

    def play(command, **kwargs):
        played.append(Path(command[1]))
        assert played[0].is_file()
        if interrupted:
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=1, stderr="No audio device")

    backend = SimpleNamespace(
        ensure=lambda: None,
        request=lambda *args: [{"id": "voice123", "name": "Me"}],
        render=lambda *args: {"id": "take123", "duration": 1.5},
        download=lambda job_id, output: output.write_bytes(b"generated audio"),
    )
    monkeypatch.setattr(cli.subprocess, "run", play)
    error = KeyboardInterrupt if interrupted else ValueError
    with pytest.raises(error) as raised:
        cli.say(backend, cli.parser().parse_args(["say", "Hello", "--voice", "voice123"]))
    if not interrupted:
        assert "Playback failed: No audio device" in str(raised.value)
    assert len(played) == 1 and not played[0].exists()


def test_download_is_atomic_and_only_overwrites_when_requested(tmp_path):
    output = tmp_path / "speech.wav"
    output.write_bytes(b"old audio")
    with Backend() as backend:
        backend.http.close()
        backend.http = httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"new audio")),
        )
        with pytest.raises(FileExistsError):
            backend.download("123", output)
        assert output.read_bytes() == b"old audio"
        backend.download("123", output, force=True)
        assert output.read_bytes() == b"new audio"
    assert list(tmp_path.iterdir()) == [output]


def test_failed_download_never_leaves_partial_output(tmp_path):
    output = tmp_path / "speech.wav"
    with Backend() as backend:
        backend.http.close()
        backend.http = httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(409, json={"detail": "Render not ready"})
            ),
        )
        with pytest.raises(ValueError, match="Render not ready"):
            backend.download("123", output)
    assert not list(tmp_path.iterdir())


def test_backend_refuses_a_different_service_or_library():
    for status in ({"service": "other"}, {"service": "idiolect", "data_directory": "/other"}):
        with Backend() as backend:
            backend.http.close()
            backend.http = httpx.Client(
                base_url="http://test",
                transport=httpx.MockTransport(
                    lambda _, value=status: httpx.Response(200, json=value)
                ),
            )
            with pytest.raises(ValueError, match="different"):
                backend.probe()


def test_render_interrupt_cancels_its_own_job(monkeypatch):
    requests = []

    def request(method, path, **kwargs):
        requests.append((method, path))
        return {"id": "take123", "status": "queued", "message": "Waiting"}

    def interrupt(_):
        raise KeyboardInterrupt

    with Backend() as backend:
        monkeypatch.setattr(backend, "request", request)
        with pytest.raises(KeyboardInterrupt):
            backend.render("voice", "Hello", "English", 12, interrupt)
    assert requests == [("POST", "/api/generate"), ("POST", "/api/renders/take123/cancel")]
