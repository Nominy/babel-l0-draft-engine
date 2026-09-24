#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 - "$REPO_ROOT" "$@" <<'PY'
import argparse
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
parser = argparse.ArgumentParser(description="Install the Mac-to-ethernetservers reverse SSH tunnel.")
parser.add_argument("--remove", action="store_true", help="Remove only the tunnel LaunchAgent.")
args = parser.parse_args(sys.argv[2:])
if sys.platform != "darwin" or os.getuid() == 0:
    parser.error("Run as your normal macOS login user, without sudo.")

label = "com.babel.l0-draft-engine-tunnel"
domain = f"gui/{os.getuid()}"
service = f"{domain}/{label}"
plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
launchctl = "/bin/launchctl"
loaded = subprocess.run([launchctl, "print", service], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
if args.remove:
    if loaded:
        subprocess.run([launchctl, "bootout", service], check=True)
    plist_path.unlink(missing_ok=True)
    print("Removed Mac reverse tunnel; the engine and Apache are unchanged.")
    raise SystemExit(0)

if subprocess.run([launchctl, "print", domain], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    raise SystemExit("No graphical login session is available.")

remote_port = 38767
ssh = [
    "/usr/bin/ssh", "-N", "-T",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-R", f"127.0.0.1:{remote_port}:127.0.0.1:8767",
    "root@93.127.223.38",
]
# Check authentication and port binding without conflicting with our own live listener.
preflight = [ssh[0]] + (ssh[3:] if not loaded else ssh[3:-3] + [ssh[-1]]) + ["true"]
subprocess.run(preflight, check=True, timeout=25)

log_root = root / ".logs"
log_root.mkdir(exist_ok=True)
plist_path.parent.mkdir(parents=True, exist_ok=True)
plist = {
    "Label": label,
    "ProgramArguments": ssh,
    "RunAtLoad": True,
    "KeepAlive": True,
    "ThrottleInterval": 10,
    "StandardOutPath": str(log_root / "tunnel.out.log"),
    "StandardErrorPath": str(log_root / "tunnel.err.log"),
}
if loaded and plist_path.exists():
    with plist_path.open("rb") as existing:
        if plistlib.load(existing) == plist:
            health = subprocess.run(
                [ssh[0], "-o", "BatchMode=yes", ssh[-1],
                 f"curl --noproxy '*' --fail --silent --max-time 3 http://127.0.0.1:{remote_port}/health"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if health.returncode == 0:
                print(f"Tunnel already installed and reachable: {service}")
                raise SystemExit(0)
if loaded:
    subprocess.run([launchctl, "bootout", service], check=True)
with plist_path.open("wb") as output:
    plistlib.dump(plist, output)
subprocess.run([launchctl, "enable", service], check=True)
subprocess.run([launchctl, "bootstrap", domain, str(plist_path)], check=True)

# A live reverse listener proves SSH authentication and the remote bind succeeded.
for _ in range(20):
    probe = subprocess.run(
        ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         "root@93.127.223.38", f"curl --noproxy '*' --fail --silent --max-time 2 http://127.0.0.1:{remote_port}/health"],
        capture_output=True, text=True,
    )
    if probe.returncode == 0 and '"ok":true' in probe.stdout and '"device":"mps"' in probe.stdout:
        print(f"Tunnel ready: ethernetservers 127.0.0.1:{remote_port} -> Mac 127.0.0.1:8767")
        print(f"Status: launchctl print {service}")
        raise SystemExit(0)
    time.sleep(1)
raise SystemExit(f"Tunnel installed but health check failed; inspect {log_root / 'tunnel.err.log'}.")
PY
