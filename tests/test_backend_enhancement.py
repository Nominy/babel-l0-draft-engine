from __future__ import annotations

import asyncio
from email import policy
from email.parser import BytesParser
import hashlib
import importlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import httpx
import pytest

from l0_draft_engine.app import create_app
from l0_draft_engine.browser_engine import BrowserDraftEngine
from l0_draft_engine.config import Settings
from l0_draft_engine.engine import ModelUnavailableError
from l0_draft_engine.enhancement import EnhancementModel, MODEL_ID, SOURCE_GRAPH_SHA256
from l0_draft_engine.inference_release import RELEASE_HEADERS


app_module = importlib.import_module("l0_draft_engine.app")
MODEL = {"id": MODEL_ID, "sha256": "a" * 64, "sourceGraphSha256": SOURCE_GRAPH_SHA256}
HEADERS = {**RELEASE_HEADERS, "X-Babel-Local-Engine": "1", "X-Babel-Request-Id": "enhancement-test"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def wav(sample: int, *, frames: int = 160, rate: int = 16000):
    pcm = struct.pack("<h", sample) * frames
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


ORIGINALS = (wav(4096), wav(-4096))
# Binary protocol fixtures, deliberately not an inference substitute or source echo.
OUTPUTS = (wav(2048), wav(-2048))


def payload():
    return {"taskId": "recording/pair", "model": dict(MODEL), "tracks": [
        {"trackId": f"track-{index}", "speakerKey": f"speaker-{index}", "trackLabel": f"Speaker {index}",
         "fieldName": f"audio:{index}", "sourceSha256": hashlib.sha256(source).hexdigest(),
         "sampleRate": 16000, "frameCount": 160}
        for index, source in enumerate(ORIGINALS, 1)
    ]}


def parts(body=None, originals=ORIGINALS):
    return [("payload", (None, json.dumps(body or payload()))), *[
        (f"audio:{index}", (f"{index}.wav", source, "audio/wav"))
        for index, source in enumerate(originals, 1)
    ]]


@pytest.fixture
def backend(monkeypatch, tmp_path):
    settings = Settings(require_current_release=True, max_track_bytes=4096, max_request_bytes=12000,
                        max_inflight_requests=1)
    engine = BrowserDraftEngine(settings)
    engine._prepared = True
    engine._enhancement = EnhancementModel.model_validate(MODEL)
    engine._process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(engine, "prepare", engine.health)
    monkeypatch.setattr(engine, "_stop", lambda: None)
    monkeypatch.setattr(app_module.tempfile, "tempdir", str(tmp_path))
    app = create_app(settings, engine)
    return app, engine, tmp_path


@pytest.mark.anyio
@pytest.mark.parametrize("fault", ["digest", "clock", "missing", "duplicate", "extra", "model"])
async def test_invalid_upload_never_reaches_rpc_and_cleans_workspace(backend, monkeypatch, fault):
    app, engine, temporary = backend
    calls = []
    monkeypatch.setattr(engine, "_call", lambda *args, **kwargs: calls.append(args))
    body = payload()
    if fault == "digest":
        body["tracks"][0]["sourceSha256"] = "0" * 64
    if fault == "clock":
        body["tracks"][0]["frameCount"] += 1
    if fault == "model":
        body["model"]["sha256"] = "b" * 64
    upload = parts(body)
    if fault == "missing":
        upload.pop()
    if fault == "duplicate":
        upload.append(upload[-1])
    if fault == "extra":
        upload.append(("unexpected", (None, "value")))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://backend") as client:
        response = await client.post("/v1/enhance", headers=HEADERS, files=upload)
        assert response.status_code in {400, 422}, response.text
        assert not calls
        assert list(temporary.iterdir()) == []
        # Failed admission and upload cleanup must not consume the sole request slot.
        response = await client.post("/v1/enhance", headers=HEADERS, files=parts({**body, "model": {}}))
        assert response.status_code == 422


@pytest.mark.anyio
async def test_backend_requires_proxy_and_release_headers_before_upload(backend):
    app, _, temporary = backend
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://backend") as client:
        assert (await client.post("/v1/enhance", files=parts())).status_code == 426
        assert (await client.post("/v1/enhance", headers=RELEASE_HEADERS, files=parts())).status_code == 403
    assert list(temporary.iterdir()) == []


def binary_rpc_fixture(captured, *, fault=None):
    def rpc(operation, *, on_progress, payload, paths, outputs):
        assert operation == "enhance"
        captured.extend(Path(value) for value in [*paths.values(), *outputs.values()])
        tracks = []
        for index, (source, audio) in enumerate(zip(payload["tracks"], OUTPUTS, strict=True)):
            progress = {"trackId": source["trackId"], "trackIndex": index, "trackCount": 2,
                        "completedChunks": 1, "totalChunks": 1, "phase": "encoding"}
            on_progress(progress)
            Path(outputs[source["fieldName"]]).write_bytes(audio)
            tracks.append({**{key: value for key, value in source.items() if key != "fieldName"},
                           "mimeType": "audio/wav", "wavSha256": hashlib.sha256(audio).hexdigest(),
                           "totalBytes": len(audio), "chunkCount": 1})
        if fault == "digest":
            tracks[0]["wavSha256"] = "0" * 64
        if fault == "clock":
            Path(outputs["audio:1"]).write_bytes(wav(2048, rate=8000))
            tracks[0]["wavSha256"] = hashlib.sha256(Path(outputs["audio:1"]).read_bytes()).hexdigest()
        if fault == "identity":
            tracks.reverse()
        if fault == "failure":
            raise ModelUnavailableError("GPU device lost")
        return {"model": MODEL_ID, "modelSha256": MODEL["sha256"], "tracks": tracks}
    return rpc


@pytest.mark.anyio
async def test_binary_result_is_validated_streamed_and_removed_with_real_progress_fields(backend, monkeypatch):
    app, engine, temporary = backend
    captured = []
    monkeypatch.setattr(engine, "_call", binary_rpc_fixture(captured))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://backend") as client:
        response = await client.post("/v1/enhance", headers=HEADERS, files=parts())
        assert response.status_code == 200, response.text
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {response.headers['content-type']}\r\n\r\n".encode() + response.content)
        fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                  for part in message.iter_parts()}
        report = json.loads(fields["result"])
        assert report["ok"] and report["provider"] == "swarm"
        assert report["taskId"] == payload()["taskId"]
        assert report["modelSha256"] == MODEL["sha256"]
        assert (fields["audio:1"], fields["audio:2"]) == OUTPUTS
        queue = await client.get("/v1/queue/enhancement-test", headers=RELEASE_HEADERS)
        assert queue.json()["status"] == "completed"
        assert queue.json()["progress"] == {"trackId": "track-2", "trackIndex": 1, "trackCount": 2,
                                             "phase": "encoding", "completedChunks": 1, "totalChunks": 1}
    assert captured and all(not path.exists() for path in captured)
    assert list(temporary.iterdir()) == []


@pytest.mark.anyio
@pytest.mark.parametrize("fault", ["digest", "clock", "identity", "failure"])
async def test_invalid_or_failed_gpu_results_never_escape_and_are_removed(backend, monkeypatch, fault):
    app, engine, temporary = backend
    captured = []
    monkeypatch.setattr(engine, "_call", binary_rpc_fixture(captured, fault=fault))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://backend") as client:
        response = await client.post("/v1/enhance", headers=HEADERS, files=parts())
        assert response.status_code == 503, response.text
    assert captured and all(not path.exists() for path in captured)
    assert list(temporary.iterdir()) == []


@pytest.mark.anyio
async def test_cancelled_request_retains_audio_until_worker_stops_then_cleans(backend, monkeypatch):
    import threading

    app, engine, temporary = backend
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    captured = []

    def blocked_rpc(*args, **kwargs):
        captured.extend(Path(path) for path in kwargs["paths"].values())
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=5):
            raise AssertionError("test did not release the worker")
        raise ModelUnavailableError("GPU operation cancelled after settling")

    monkeypatch.setattr(engine, "_call", blocked_rpc)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://backend") as client:
        pending = asyncio.create_task(client.post("/v1/enhance", headers=HEADERS, files=parts()))
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            pending.cancel()
            await asyncio.sleep(0)
            assert all(path.exists() for path in captured)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
    assert captured and all(not path.exists() for path in captured)
    assert list(temporary.iterdir()) == []


@pytest.mark.anyio
async def test_response_disconnect_removes_both_original_and_enhanced_audio(backend, monkeypatch):
    from starlette.requests import ClientDisconnect

    app, engine, temporary = backend
    captured = []
    monkeypatch.setattr(engine, "_call", binary_rpc_fixture(captured))
    request = httpx.Request("POST", "http://backend/v1/enhance", headers=HEADERS, files=parts())
    content = request.read()
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            await asyncio.Event().wait()
        delivered = True
        return {"type": "http.request", "body": content, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        if message["type"] == "http.response.body":
            assert captured and all(path.exists() for path in captured)
            raise OSError("client disconnected during binary response")

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/enhance", "raw_path": b"/v1/enhance", "query_string": b"",
             "root_path": "", "headers": [(name.lower(), value) for name, value in request.headers.raw],
             "server": ("backend", 80), "client": ("127.0.0.1", 1234)}
    with pytest.raises(ClientDisconnect):
        await app(scope, receive, send)
    assert captured and all(not path.exists() for path in captured)
    assert list(temporary.iterdir()) == []
