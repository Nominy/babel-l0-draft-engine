#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="python3.11"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                echo "--python requires a native Python 3.11 executable or path." >&2
                exit 1
            fi
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: scripts/Install-Mac.sh [--python /path/to/python3.11]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done
if ! command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1; then
    echo "Native Python 3.11 was not found. Install Python 3.11 for Apple Silicon from python.org or supply --python /path/to/python3.11." >&2
    exit 1
fi

check_python() {
    "$1" - <<'PY'
import platform
import sys
if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
    raise SystemExit("Native Apple Silicon Python is required; do not use an Intel interpreter or Rosetta.")
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Python 3.11 is required. Supply --python /path/to/python3.11; an existing incompatible .venv must be moved or removed first.")
print(f"Native Python ready: {platform.python_version()} ({platform.machine()})")
PY
}
check_python "$PYTHON_EXECUTABLE"
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required on PATH, including for raw WAV inference. Install a native Apple Silicon ffmpeg build (for example: brew install ffmpeg), then rerun this installer." >&2
    exit 1
fi

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    "$PYTHON_EXECUTABLE" -m venv "$REPO_ROOT/.venv"
fi
check_python "$VENV_PYTHON"
"$VENV_PYTHON" -m ensurepip --upgrade
"$VENV_PYTHON" -m pip install --upgrade pip 'setuptools<82' wheel
"$VENV_PYTHON" -m pip install --index-url https://pypi.org/simple torch torchaudio
"$VENV_PYTHON" -m pip install --editable "$REPO_ROOT"

"$VENV_PYTHON" - "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
env_file = root / ".env"
if not env_file.exists():
    contents = (root / ".env.example").read_text(encoding="utf-8")
    with env_file.open("x", encoding="utf-8") as target:
        target.write(contents.rstrip() + "\n\nLOCAL_ENGINE_DEVICE=mps\n")
PY

"$REPO_ROOT/scripts/Start-Mac.sh" --preflight-only
printf 'Mac environment ready. Start the endpoint with:\n  %q\n' "$REPO_ROOT/scripts/Start-Mac.sh"
