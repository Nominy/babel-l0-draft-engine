"""Stage-only contract smoke: release gate, real trusted GPU fallback, persisted labels."""
import argparse
import asyncio
import json
import secrets
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import httpx
from l0_draft_engine.app import create_app as create_trusted
from l0_draft_engine.browser_engine import BrowserDraftEngine
from l0_draft_engine.config import Settings
from l0_draft_engine.coordinator import CoordinatorSettings, create_app as create_coordinator
from l0_draft_engine.inference_release import RELEASE_ID, RELEASE_HEADERS

parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()

async def run():
    engine = BrowserDraftEngine(Settings(inference_runtime="c-denoise-webgpu", require_current_release=True))
    trusted = create_trusted(engine.settings, engine)
    with tempfile.TemporaryDirectory(prefix="babel-release-coordinator-") as temporary:
        settings = CoordinatorSettings(backend_urls=("http://trusted",), cache_dir=Path(temporary),
            require_current_release=True, queue_seconds=0.05, lease_seconds=0.1)
        async with trusted.router.lifespan_context(trusted), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=trusted), base_url="http://trusted") as upstream:
            coordinator = create_coordinator(settings, upstream)
            audio = args.audio.read_bytes()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=coordinator), base_url="http://coordinator", headers={"X-Babel-Local-Engine": "1"}) as client:
                rejected = await client.post("/v1/transcribe", content=b"old client must be rejected before upload parsing")
                assert rejected.status_code == 426 and rejected.json()["requiredRelease"] == RELEASE_ID
                payload = {"taskId": "release-coordinator-smoke", "tracks": [
                    {"lane": "speaker-1", "fieldName": "audio:1"}, {"lane": "speaker-2", "fieldName": "audio:2"}]}
                response = await client.post("/v1/transcribe", headers={**RELEASE_HEADERS, "Authorization": f"Bearer {secrets.token_urlsafe(32)}"},
                    data={"payload": json.dumps(payload)}, files={name: ("sample.wav", audio, "audio/wav") for name in ("audio:1", "audio:2")})
                response.raise_for_status()
                timing = response.json()
                headers = {**RELEASE_HEADERS, "Authorization": f"Bearer {timing['accessToken']}"}
                drafted = await client.post("/v1/draft", json={"taskId": payload["taskId"]}, headers=headers)
                drafted.raise_for_status()
            restarted = create_coordinator(settings, upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted), base_url="http://coordinator", headers={"X-Babel-Local-Engine": "1"}) as client:
                repeated = await client.post("/v1/draft", json={"taskId": payload["taskId"]}, headers=headers)
                repeated.raise_for_status()
                assert repeated.json()["rows"] == drafted.json()["rows"]
            report = {"pass": True, "release": RELEASE_ID, "oldClientStatus": rejected.status_code,
                      "trustedFallback": True, "labelsSurviveCoordinatorRestart": True,
                      "words": sum(len(track["tokens"]) for track in timing["tracks"]), "rows": len(drafted.json()["rows"])}
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report))

asyncio.run(run())
