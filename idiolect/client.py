"""Keep a local speech engine warm between CLI commands."""

import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from .config import DATA, ROOT


class Backend:
    def __init__(self, port=7860):
        self.port = port
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    @staticmethod
    def checked(response):
        if not response.is_success:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ValueError(f"Engine error: {detail}")
        return response

    def request(self, method, path, **kwargs):
        return self.checked(self.http.request(method, path, **kwargs)).json()

    def probe(self):
        try:
            response = self.http.get("/api/status", timeout=2)
        except httpx.ConnectError:
            return None
        try:
            status = response.json()
        except ValueError:
            status = {}
        if not response.is_success or status.get("service") != "idiolect":
            raise ValueError(f"Port {self.port} is occupied by a different service.")
        if status.get("data_directory") != str(DATA):
            raise ValueError(
                f"The engine on port {self.port} uses a different voice library. "
                "Choose another --port or use the same IDIOLECT_DATA directory."
            )
        return status

    def ensure(self):
        status = self.probe()
        if status:
            return status
        DATA.mkdir(parents=True, exist_ok=True)
        with (DATA / "startup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            status = self.probe()
            if status:
                return status
            log_path = DATA / "engine.log"
            print("Starting the local voice engine…", file=sys.stderr, flush=True)
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "idiolect.runtime", "serve", "--port", str(self.port)],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            for _ in range(100):
                status = self.probe()
                if status:
                    return status
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            raise ValueError(f"The local engine could not start. See {log_path}.")

    def stop(self):
        status = self.probe()
        if status is None:
            return False
        pid = status["pid"]
        command = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False
        ).stdout.split()
        if "idiolect.runtime" not in command or "serve" not in command:
            raise ValueError("The engine was started by another launcher; stop it there.")
        jobs = self.request("GET", "/api/renders")
        if any(job["status"] in {"queued", "loading", "generating"} for job in jobs):
            raise ValueError(
                "The engine has an active render. Wait for it to finish before stopping."
            )
        os.kill(pid, signal.SIGTERM)
        return True

    def render(self, voice_id, text, language, seed, progress):
        job = self.request(
            "POST",
            "/api/generate",
            json={"voice_id": voice_id, "text": text, "language": language, "seed": seed},
        )
        try:
            while job["status"] in {"queued", "loading", "generating"}:
                progress(job)
                time.sleep(0.5)
                job = self.request("GET", f"/api/renders/{job['id']}")
        except KeyboardInterrupt:
            self.request("POST", f"/api/renders/{job['id']}/cancel")
            print("\nStopping this take after the current section.", file=sys.stderr)
            raise
        if job["status"] != "ready":
            raise ValueError(job["message"])
        return job

    def download(self, job_id, output, force=False):
        """Publish a complete WAV atomically, preserving existing files by default."""
        output = Path(output)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, suffix=".wav", delete=False
            ) as file:
                temporary = Path(file.name)
                with self.http.stream("GET", f"/api/renders/{job_id}/audio") as response:
                    if not response.is_success:
                        response.read()
                        self.checked(response)
                    for chunk in response.iter_bytes():
                        file.write(chunk)
            if force:
                temporary.replace(output)
            else:
                os.link(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
