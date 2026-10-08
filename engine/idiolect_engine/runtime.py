import argparse
import json

from .config import (
    ASR_MODEL,
    MODEL,
    MODEL_ID,
    MODEL_REVISION,
    model_downloaded,
    recognizer_downloaded,
)


def download():
    from huggingface_hub import snapshot_download

    from .recognizer import download_recognizer

    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    print(f"Speech model ready: {MODEL}")
    download_recognizer()
    print(f"Reading model ready: {ASR_MODEL}")


def doctor():
    import platform

    import torch

    from .engine import accelerator

    device = accelerator()
    if device == "mps":
        gpu = f"Apple {platform.machine()} (Metal)"
    else:
        gpu = torch.cuda.get_device_name(0) if device else None
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_available": device is not None,
                "gpu": gpu,
                "model_downloaded": model_downloaded(),
                "model_directory": str(MODEL),
                "recognizer_downloaded": recognizer_downloaded(),
                "recognizer_directory": str(ASR_MODEL),
            },
            indent=2,
        )
    )
    if device is None:
        raise SystemExit(1)
    print("GPU tensor check:", (torch.ones(4, device=device) * 12).tolist())


def serve(port):
    import os

    import uvicorn

    from .server import create_app

    # Serving never needs the network. Downloading is an explicit separate command.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    server = None

    def request_shutdown():
        server.should_exit = True

    app = create_app(request_shutdown=request_shutdown)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False))
    server.run()


def main():
    parser = argparse.ArgumentParser(
        prog="idiolect-engine", description="Idiolect's local speech engine"
    )
    parser.add_argument(
        "command", choices=["serve", "download", "doctor"], nargs="?", default="serve"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "doctor":
        doctor()
    else:
        serve(args.port)


if __name__ == "__main__":
    main()
