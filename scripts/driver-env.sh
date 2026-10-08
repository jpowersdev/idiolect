# Source inside the Nix shell before loading PyTorch.
# A nixos-rebuild switch can leave userspace ahead of the running kernel driver.
driver_lib="$(python3 - <<'PY'
from pathlib import Path
import re
import sys

version_file = Path('/proc/driver/nvidia/version')
if not version_file.exists():
    sys.exit('No loaded NVIDIA driver. Load the driver before starting Idiolect.')
match = re.search(r'\b(\d{3}\.\d+\.\d+)\b', version_file.read_text())
if not match:
    sys.exit('Could not read the loaded NVIDIA driver version.')
version = match.group(1)
current = Path('/run/opengl-driver/lib')
candidates = [current, *(p / 'lib' for p in Path('/nix/store').glob(f'*-nvidia-x11-{version}'))]
for candidate in candidates:
    cuda = candidate / 'libcuda.so.1'
    if cuda.exists() and cuda.resolve().name == f'libcuda.so.{version}':
        print(candidate)
        if candidate != current:
            print(f'Using NVIDIA {version} libraries to match the running kernel.', file=sys.stderr)
        break
else:
    sys.exit(f'NVIDIA {version} libraries are missing. Reboot into your current NixOS generation.')
PY
)"
export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
