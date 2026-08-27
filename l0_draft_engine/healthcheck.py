from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    try:
        with urlopen("http://127.0.0.1:8767/health", timeout=5) as response:
            payload = json.load(response)
            healthy = response.status == 200 and payload.get("ok") is True
    except (OSError, URLError, ValueError):
        healthy = False
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
