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
import getpass
import json
import os
import plistlib
import shutil
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
parser = argparse.ArgumentParser(description="Install a per-user, self-restarting Metal inference service.")
parser.add_argument("--port", type=int, default=8767)
parser.add_argument("--remove", action="store_true", help="Unload the service and remove its LaunchAgent; keep models, configuration, and logs.")
args = parser.parse_args(sys.argv[2:])
if sys.platform != "darwin" or os.getuid() == 0:
    parser.error("Run as your normal macOS login user, without sudo.")
if not 1 <= args.port <= 65535:
    parser.error("--port must be between 1 and 65535")

label = "com.babel.l0-draft-engine"
domain = f"gui/{os.getuid()}"
service = f"{domain}/{label}"
plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
log_root = root / ".logs"
launchctl = "/bin/launchctl"
loaded = subprocess.run([launchctl, "print", service], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def unload_service():
    state = subprocess.run([launchctl, "print", service], capture_output=True, text=True, check=True)
    match = re.search(r"^\s*pid = (\d+)$", state.stdout, re.MULTILINE)
    pid = int(match.group(1)) if match else None
    subprocess.run([launchctl, "bootout", service], check=True)
    # bootout returns before a running worker has necessarily finished exiting.
    # Wait for both process exit and deregistration before reusing the label.
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        registered = subprocess.run([launchctl, "print", service], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not alive and not registered:
            return
        time.sleep(0.1)
    raise SystemExit(f"Timed out waiting for {service} to stop; no replacement was started.")


if args.remove:
    if loaded:
        unload_service()
    plist_path.unlink(missing_ok=True)
    print(f"Removed {label}. Models, .env, and logs were preserved.")
    raise SystemExit(0)

if subprocess.run([launchctl, "print", domain], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    raise SystemExit("No graphical login session is available. Log into macOS, then rerun this installer.")

ffmpeg = shutil.which("ffmpeg")
if ffmpeg is None:
    raise SystemExit("ffmpeg is required on PATH. Install it before installing the service.")
service_path = os.pathsep.join(dict.fromkeys([str(Path(ffmpeg).parent), "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]))
service_environment = {"PATH": service_path, "PYTHONUNBUFFERED": "1"}
# Match launchd's environment rather than inheriting shell-only model overrides.
preflight_environment = {
    "HOME": str(Path.home()),
    "USER": getpass.getuser(),
    "LOGNAME": getpass.getuser(),
    **service_environment,
}
if not loaded:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as exc:
            raise SystemExit(f"Port {args.port} is already occupied. Stop the foreground engine first; no process was stopped by this installer.") from exc
subprocess.run(["/bin/bash", str(root / "scripts" / "Start-Mac.sh"), "--port", str(args.port), "--preflight-only"], env=preflight_environment, check=True)

plist = {
    "Label": label,
    "ProgramArguments": ["/bin/bash", str(root / "scripts" / "Start-Mac.sh"), "--port", str(args.port)],
    "WorkingDirectory": str(root),
    "EnvironmentVariables": service_environment,
    "RunAtLoad": True,
    "KeepAlive": True,
    "ThrottleInterval": 10,
    "ExitTimeOut": 30,
    "ProcessType": "Interactive",
    "LimitLoadToSessionType": "Aqua",
    "StandardOutPath": str(log_root / "engine.out.log"),
    "StandardErrorPath": str(log_root / "engine.err.log"),
}
plist_path.parent.mkdir(parents=True, exist_ok=True)
log_root.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=plist_path.parent, prefix=f".{label}.", delete=False) as temporary:
    temporary_path = Path(temporary.name)
    plistlib.dump(plist, temporary)
try:
    if loaded:
        unload_service()
    os.replace(temporary_path, plist_path)
finally:
    temporary_path.unlink(missing_ok=True)
subprocess.run([launchctl, "enable", service], check=True)
subprocess.run([launchctl, "bootstrap", domain, str(plist_path)], check=True)

health_url = f"http://127.0.0.1:{args.port}/health"
# Ignore HTTP proxy settings for a loopback readiness check.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + 45
while time.monotonic() < deadline:
    try:
        with opener.open(health_url, timeout=1) as response:
            health = json.load(response)
        if health.get("ok") is True and health.get("device") == "mps":
            print(f"Service ready: http://127.0.0.1:{args.port}")
            print("Starts at login, survives terminal closure, and restarts after exit or crash.")
            print("Model loading remains lazy; /health does not prove checkpoint inference.")
            print(f"LaunchAgent: {plist_path}")
            print(f"Logs: {log_root / 'engine.out.log'} and {log_root / 'engine.err.log'}")
            print(f"Status: launchctl print {service}")
            print(f"Restart: launchctl kickstart -k {service}")
            raise SystemExit(0)
    except (OSError, ValueError, urllib.error.URLError):
        pass
    time.sleep(0.5)
raise SystemExit(f"LaunchAgent installed, but Metal /health did not become ready within 45 seconds. Inspect {log_root / 'engine.err.log'}; launchd will continue restarting the service.")
PY
