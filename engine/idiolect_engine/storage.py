import fcntl
import json
import shutil
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def timestamp():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def exclusive(root: Path):
    """Allow one engine per data directory."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "engine.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Idiolect is already running with this data directory.") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class Store:
    """Your voice's folder: voice.json, plus the takes and clips it lists."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.folder = self.root / "voice"
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.folder / name

    def voice(self):
        try:
            return json.loads((self.folder / "voice.json").read_text())
        except FileNotFoundError:
            return None

    @contextmanager
    def replace_voice(self):
        """Yield a staging folder; it becomes the voice only if the block succeeds."""
        staging = self.root / "voice.new"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            yield staging
            with self.lock:
                previous = self.root / "voice.old"
                shutil.rmtree(previous, ignore_errors=True)
                if self.folder.exists():
                    self.folder.rename(previous)
                staging.rename(self.folder)
                shutil.rmtree(previous, ignore_errors=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
