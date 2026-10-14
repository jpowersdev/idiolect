import argparse
import sys

from .config import DATA, recognizer_downloaded, socket_path


def chosen():
    from .models import choose

    try:
        return choose()
    except ValueError as error:
        sys.exit(str(error))


def download():
    from .recognizer import download_recognizer

    choice = chosen()
    choice.model.download()
    download_recognizer()
    print(f"Models ready ({choice.model.name}).")


def doctor():
    import platform

    import torch

    from .engine import accelerator
    from .models import MODELS_BY_PREFERENCE, memory_gb

    device = accelerator()
    if device == "mps":
        gpu = f"Apple {platform.machine()} (Metal)"
    elif device:
        gpu = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    else:
        gpu = None
    choice = chosen()
    others = ", ".join(f"{m.key} ({m.name})" for m in MODELS_BY_PREFERENCE)

    def missing(ready):
        return "ready" if ready else "not downloaded (idiolect setup downloads it)"

    print(f"GPU: {gpu or 'none found'}")
    print(f"Memory: {memory_gb():.0f} GB")
    print(
        f"Speech model: {choice.model.name}, {choice.reason}; {missing(choice.model.downloaded())}"
    )
    print(f"  Set IDIOLECT_MODEL to choose another: {others}.")
    if choice.warning:
        print(f"  Warning: {choice.warning}")
    print(f"Reading model: {missing(recognizer_downloaded())}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Data: {DATA}")
    if device is None:
        sys.exit("Idiolect needs an Apple silicon Mac or a CUDA GPU.")
    (torch.ones(4, device=device) * 12).tolist()


def serve():
    import logging
    import os
    import socket

    # engine.log records each render, including sections rendered again.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import uvicorn

    from .server import create_app
    from .storage import exclusive

    # Serving never needs the network; setup downloads the models beforehand.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    server = None

    def request_shutdown():
        server.should_exit = True

    with exclusive(DATA):
        # Only one engine holds the lock, so a leftover socket is from one that exited.
        path = socket_path()
        path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        os.chmod(path, 0o600)
        try:
            app = create_app(request_shutdown=request_shutdown)
            server = uvicorn.Server(uvicorn.Config(app, access_log=False))
            server.run(sockets=[listener])
        finally:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        prog="idiolect-engine",
        description="Idiolect's local speech engine. The idiolect CLI runs it for you.",
    )
    parser.add_argument("command", choices=["serve", "download", "doctor"])
    args = parser.parse_args()
    {"serve": serve, "download": download, "doctor": doctor}[args.command]()


if __name__ == "__main__":
    main()
