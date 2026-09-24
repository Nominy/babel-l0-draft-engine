#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Mac environment is missing. Run scripts/Install-Mac.sh first." >&2
    exit 1
fi

exec "$VENV_PYTHON" - "$REPO_ROOT" "$@" <<'PY'
import argparse
import os
import platform
import re
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
parser = argparse.ArgumentParser(description="Run L0 Draft Engine natively on Apple Silicon Metal.")
parser.add_argument("--port", type=int, default=8767)
parser.add_argument("--preflight-only", action="store_true")
args = parser.parse_args(sys.argv[2:])
if not 1 <= args.port <= 65535:
    parser.error("--port must be between 1 and 65535")
if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
    raise SystemExit("Native Apple Silicon Python is required; do not run under Rosetta.")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Python 3.11 is required. Recreate .venv with scripts/Install-Mac.sh.")

os.chdir(root)
env_file = root / ".env"
if env_file.is_file():
    for number, line in enumerate(env_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise SystemExit(f"Invalid .env assignment at line {number}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        # Values are literal: no shell evaluation, expansion, or command substitution.
        os.environ.setdefault(name, value)

os.environ.setdefault("LOCAL_ENGINE_DEVICE", "mps")
if os.environ["LOCAL_ENGINE_DEVICE"].strip().lower() != "mps":
    raise SystemExit("Start-Mac.sh requires LOCAL_ENGINE_DEVICE=mps; it never falls back to CPU or CUDA.")
for flag in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
    if os.environ.get(flag, "0") not in {"", "0"}:
        raise SystemExit(f"{flag} must be unset or 0; CPU fallback and fast math are forbidden.")
    os.environ[flag] = "0"
if shutil.which("ffmpeg") is None:
    raise SystemExit("ffmpeg is required on PATH, including for raw WAV inference. Install a native Apple Silicon ffmpeg build.")

for name, directory in (
    ("HF_HOME", root / ".cache" / "huggingface"),
    ("TMPDIR", root / ".cache" / "tmp"),
    ("TORCH_HOME", root / ".cache" / "torch"),
):
    if not os.environ.get(name):
        os.environ[name] = str(directory)
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

import torch
import fastapi
import gigaam
import transformers

if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
    raise SystemExit("Metal MPS is unavailable. Use native arm64 PyTorch on a supported macOS release.")
# Availability alone does not prove that Metal kernels can execute in this session.
for dtype in (torch.float32, torch.float16):
    sample = torch.ones((2, 2), device="mps", dtype=dtype)
    result = sample @ sample
    torch.mps.synchronize()
    if result.device.type != "mps" or not torch.equal(result.cpu(), torch.full((2, 2), 2.0, dtype=dtype)):
        raise SystemExit(f"Metal execution check failed for {dtype}.")
print(f"Ready: Apple Silicon Metal (MPS); Python {platform.python_version()}; torch {torch.__version__}", flush=True)
print("Precision: GigaAM FP32 weights with FP16 encoder autocast; punctuation FP16; CPU fallback and fast math disabled.", flush=True)
if args.preflight_only:
    raise SystemExit(0)

print(f"Starting L0 Draft Engine at http://127.0.0.1:{args.port}", flush=True)
print("Models load on demand from the configured paths or public model names. Press Ctrl+C to stop.", flush=True)
os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "l0_draft_engine.app:app", "--host", "127.0.0.1", "--port", str(args.port)])
PY
