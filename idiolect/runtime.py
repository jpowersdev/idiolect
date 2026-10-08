import argparse
import json

from .config import MODEL, MODEL_ID, MODEL_REVISION, model_downloaded


def main():
    parser = argparse.ArgumentParser(description="Idiolect engine management")
    parser.add_argument(
        "command", choices=["serve", "download", "doctor"], nargs="?", default="serve"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=MODEL,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
        print(f"Model ready: {MODEL}")
    elif args.command == "doctor":
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
                },
                indent=2,
            )
        )
        if device is None:
            raise SystemExit(1)
        print("GPU tensor check:", (torch.ones(4, device=device) * 12).tolist())
    else:
        import os

        import uvicorn

        # Serving never needs the network. Downloading is an explicit separate command.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        uvicorn.run(
            "idiolect.server:create_app",
            factory=True,
            host="127.0.0.1",
            port=args.port,
            access_log=False,
        )


if __name__ == "__main__":
    main()
