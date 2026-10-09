from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import struct
import time
import uuid

import httpx
import pytest
from starlette.requests import Request

from l0_draft_engine import coordinator as coordinator_module
from l0_draft_engine.coordinator import BodyLimitMiddleware, BodyTooLarge, CoordinatorSettings, create_app
from l0_draft_engine.enhancement import MODEL_ID, SOURCE_GRAPH_SHA256, read_multipart
from l0_draft_engine.inference_release import RELEASE_HEADERS, RELEASE_ID


OWNER = "a" * 43
OTHER_OWNER = "b" * 43
MODEL = {"id": MODEL_ID, "sha256": "1" * 64, "sourceGraphSha256": SOURCE_GRAPH_SHA256}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def wav(*, floating: bool = False, channels: int = 1, nonfinite: bool = False) -> bytes:
    samples = (struct.pack("<f", float("nan") if nonfinite else 0.125) if floating else b"\x00\x10") * 160 * channels
    width = 4 if floating else 2
    return (b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 3 if floating else 1, channels, 16_000,
                          16_000 * channels * width, channels * width, width * 8)
            + b"data" + struct.pack("<I", len(samples)) + samples)


def payload(originals: tuple[bytes, bytes] | None = None) -> dict:
    originals = originals or (wav(), wav())
    return {"taskId": "original/task", "model": dict(MODEL), "tracks": [
        {"trackId": f"track-{index}", "speakerKey": f"speaker-{index}", "trackLabel": f"Speaker {index}",
         "fieldName": f"audio:{index}", "sourceSha256": hashlib.sha256(source).hexdigest(),
         "sampleRate": 16_000, "frameCount": 160}
        for index, source in enumerate(originals, 1)
    ]}


def result(body: dict, outputs: tuple[bytes, bytes] | None = None) -> dict:
    outputs = outputs or (wav(), wav())
    return {"model": MODEL_ID, "modelSha256": body["model"]["sha256"], "tracks": [
        {**{key: value for key, value in track.items() if key != "fieldName"},
         "mimeType": "audio/wav", "wavSha256": hashlib.sha256(output).hexdigest(),
         "totalBytes": len(output), "chunkCount": 1}
        for track, output in zip(body["tracks"], outputs, strict=True)
    ]}


def owner_headers(request_id: str, token: str = OWNER) -> dict[str, str]:
    return {**RELEASE_HEADERS, "X-Babel-Local-Engine": "1", "X-Babel-Request-Id": request_id,
            "Authorization": f"Bearer {token}"}


def files(pair: tuple[bytes, bytes] | None = None) -> dict:
    return {f"audio:{index}": (f"{index}.wav", content, "audio/wav")
            for index, content in enumerate(pair or (wav(), wav()), 1)}


def settings(tmp_path: Path, **changes) -> CoordinatorSettings:
    values = dict(backend_urls=("http://asr-only",), cache_dir=tmp_path / "cache",
                  max_track_bytes=4096, max_request_bytes=12_000,
                  queue_seconds=5, lease_seconds=5, request_timeout_seconds=15)
    return CoordinatorSettings(**{**values, **changes})


@asynccontextmanager
async def environment(tmp_path: Path, **changes):
    def forbidden_backend(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"enhancement must never reach ASR backend: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden_backend)) as upstream:
        app = create_app(settings(tmp_path, **changes), client=upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
            yield app, client


async def register(client: httpx.AsyncClient, *, model: dict | None = MODEL, operations=None) -> dict:
    body = {"modelBundleSchema": "babel-browser-model-bundle-v3", "modelRelease": RELEASE_ID, "protocolVersion": 3}
    if model is not None:
        body.update(enhancementModel=model, operations=operations or ["enhance"])
    elif operations is not None:
        body["operations"] = operations
    response = await client.post("/v1/workers/register", json=body, headers=RELEASE_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


async def submit(client: httpx.AsyncClient, request_id: str, body: dict | None = None,
                 originals: tuple[bytes, bytes] | None = None) -> httpx.Response:
    return await client.post("/v1/enhance", data={"payload": json.dumps(body or payload(originals))},
                             files=files(originals), headers=owner_headers(request_id))


async def admitted(app, client, *, body=None, originals=None):
    request_id = str(uuid.uuid4())
    event = asyncio.Event()
    enqueue = app.state.coordinator.enqueue

    def notify(job):
        enqueue(job)
        event.set()

    app.state.coordinator.enqueue = notify
    pending = asyncio.create_task(submit(client, request_id, body, originals))
    try:
        await asyncio.wait_for(event.wait(), timeout=3)
    except BaseException:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        raise
    finally:
        app.state.coordinator.enqueue = enqueue
    return request_id, pending, app.state.coordinator.by_request_id[request_id]


async def lease(client, worker):
    response = await client.post("/v1/workers/lease", json=worker, headers=RELEASE_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


async def complete(client, worker, leased, body, *, outputs=None, metadata=None, headers=None):
    return await client.post(f"/v1/jobs/{leased['jobId']}/complete", data={"payload": json.dumps({
        **worker, "leaseToken": leased["leaseToken"], "result": metadata or result(body, outputs),
    })}, files=files(outputs), headers=headers or {
        **RELEASE_HEADERS, "Authorization": f"Bearer {leased['leaseToken']}",
    })


def multipart_result(response: httpx.Response) -> dict:
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {response.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n".encode() + response.content
    )
    return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()}


def assert_clean(app, job) -> None:
    state = app.state.coordinator
    assert state.inflight == 0 and not state.jobs and not state.by_request_id and not state.waiting
    assert all(worker.job_id is None for worker in state.workers.values())
    assert not any(path.exists() for path in (*job.paths.values(), *job.output_paths.values()))
    assert not list(state.cache.directory.iterdir())


@pytest.mark.anyio
@pytest.mark.parametrize("floating,channels", [(False, 1), (False, 2), (True, 1), (True, 2), (False, 40)])
async def test_round_trip_preserves_originals_and_protects_completed_status(tmp_path, floating, channels):
    original = wav(floating=floating, channels=channels)
    # The last case deliberately exceeds the tiny test limit but remains valid WAV.
    async with environment(tmp_path, max_track_bytes=32_000, max_request_bytes=70_000) as (app, client):
        worker = await register(client)
        body = payload((original, wav()))
        request_id, pending, job = await admitted(app, client, body=body, originals=(original, wav()))
        for token in (None, OTHER_OWNER):
            headers = {**RELEASE_HEADERS, **({"Authorization": f"Bearer {token}"} if token else {})}
            assert (await client.get(f"/v1/queue/{request_id}", headers=headers)).status_code == 404
        status = await client.get(f"/v1/queue/{request_id}", headers=owner_headers(request_id))
        assert status.json() == {"requestId": request_id, "status": "queued", "position": 1, "queuedCount": 1}
        leased = await lease(client, worker)
        assert leased["operation"] == "enhance" and leased["payload"] == body
        for item, expected in zip(leased["audio"], (original, wav()), strict=True):
            assert (await client.get(item["url"])).status_code == 401
            assert (await client.get(item["url"], headers={"Authorization": f"Bearer {OTHER_OWNER}"})).status_code == 403
            downloaded = await client.get(item["url"], headers={"Authorization": f"Bearer {leased['leaseToken']}"})
            assert downloaded.content == expected and downloaded.headers["cache-control"] == "no-store"
        assert (await complete(client, worker, leased, body)).status_code == 200
        response = await pending
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        parts = multipart_result(response)
        assert set(parts) == {"result", "audio:1", "audio:2"}
        assert json.loads(parts["result"]) == {"ok": True, "provider": "swarm", "taskId": body["taskId"], **result(body)}
        assert parts["audio:1"] == parts["audio:2"] == wav()
        assert int(response.headers["content-length"]) == len(response.content)
        assert (await client.get(f"/v1/queue/{request_id}", headers=owner_headers(request_id))).json()["status"] == "completed"
        assert (await client.get(f"/v1/queue/{request_id}")).status_code == 404
        assert (await client.get(f"/v1/queue/{request_id}", headers=owner_headers(request_id, OTHER_OWNER))).status_code == 404
        assert (await complete(client, worker, leased, body)).status_code == 404
        assert_clean(app, job)


@pytest.mark.anyio
async def test_exact_model_capability_and_busy_matching_worker_receive_queued_jobs(tmp_path):
    async with environment(tmp_path) as (app, client):
        old = await register(client, model=None)
        wrong = await register(client, model={**MODEL, "sha256": "2" * 64})
        matching = await register(client)
        first_id, first, first_job = await admitted(app, client)
        second_id, second, second_job = await admitted(app, client)
        for worker in (old, wrong):
            assert (await client.post("/v1/workers/lease", json=worker)).status_code == 204
        first_lease = await lease(client, matching)
        assert first_lease["jobId"] == first_job.job_id
        assert (await client.post("/v1/workers/lease", json=matching)).status_code == 204
        assert (await complete(client, matching, first_lease, payload())).status_code == 200
        assert (await first).status_code == 200
        second_lease = await lease(client, matching)
        assert second_lease["jobId"] == second_job.job_id
        assert (await complete(client, matching, second_lease, payload())).status_code == 200
        assert (await second).status_code == 200
        assert_clean(app, second_job)


@pytest.mark.anyio
@pytest.mark.parametrize("changes", [
    {"operations": ["enhance"]}, {"operations": []}, {"operations": ["enhance", "enhance"], "enhancementModel": MODEL},
    {"operations": ["draft"], "enhancementModel": MODEL},
    {"operations": ["enhance"], "enhancementModel": {**MODEL, "sourceGraphSha256": "0" * 64}},
    {"operations": ["enhance"], "enhancementModel": {**MODEL, "id": "other-model"}},
    {"operations": ["enhance"], "enhancementModel": {**MODEL, "sha256": "F" * 64}},
])
async def test_registration_rejects_inconsistent_or_unpinned_capabilities(tmp_path, changes):
    async with environment(tmp_path) as (_, client):
        response = await client.post("/v1/workers/register", json={
            "modelBundleSchema": "babel-browser-model-bundle-v3", "protocolVersion": 3,
            "modelRelease": RELEASE_ID, **changes,
        })
        assert response.status_code == 422


@pytest.mark.anyio
async def test_no_matching_worker_is_bounded_without_any_asr_fallback(tmp_path):
    async with environment(tmp_path, queue_seconds=0.01) as (app, client):
        await register(client, model=None)
        await register(client, model={**MODEL, "sha256": "2" * 64})
        response = await asyncio.wait_for(submit(client, str(uuid.uuid4())), timeout=2)
        assert response.status_code == 503 and "Originals are unchanged" in response.text
        assert app.state.coordinator.inflight == 0 and not app.state.coordinator.jobs
        assert not list(app.state.coordinator.cache.directory.iterdir())


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["expiry", "cancel", "worker-error"])
async def test_lease_expiry_cancel_and_worker_error_cleanup_without_fallback(tmp_path, reason):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        request_id, pending, job = await admitted(app, client)
        leased = await lease(client, worker)
        if reason == "expiry":
            job.lease_deadline = time.monotonic() - 1
            assert (await client.get(leased["audio"][0]["url"], headers={
                "Authorization": f"Bearer {leased['leaseToken']}"
            })).status_code == 404
        elif reason == "cancel":
            pending.cancel()
        else:
            assert (await client.post(f"/v1/jobs/{job.job_id}/complete", json={
                **worker, "leaseToken": leased["leaseToken"], "error": "GPU unavailable",
            })).status_code == 200
        if reason == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert (await pending).status_code == 503
        assert (await client.get(f"/v1/queue/{request_id}", headers=owner_headers(request_id))).status_code == 404
        assert (await complete(client, worker, leased, payload())).status_code == 404
        assert_clean(app, job)


@pytest.mark.anyio
async def test_progress_requires_lease_owner_and_rejects_regression(tmp_path):
    async with environment(tmp_path) as (app, client):
        worker, stranger = await register(client), await register(client)
        request_id, pending, job = await admitted(app, client)
        leased = await lease(client, worker)
        progress = {"phase": "enhancing", "trackId": "track-1", "trackIndex": 0,
                    "trackCount": 2, "completedChunks": 0, "totalChunks": 1}

        async def post(value, credentials=worker, lease_token=leased["leaseToken"]):
            return await client.post(f"/v1/jobs/{job.job_id}/progress", json={
                **credentials, "leaseToken": lease_token, "progress": value,
            })

        assert (await post(progress, {**worker, "token": "wrong"})).status_code == 401
        assert (await post(progress, stranger)).status_code == 403
        assert (await post(progress, lease_token="wrong")).status_code == 403
        assert (await post(progress)).status_code == 200
        for bad in ({"trackId": "foreign"}, {"trackCount": 3}, {"completedChunks": 2}, {"totalChunks": 999999},
                    {"phase": "encoding"}, {"trackIndex": 1, "trackId": "track-2"}):
            assert (await post({**progress, **bad})).status_code == 422
        finished = {**progress, "completedChunks": 1, "phase": "encoding"}
        assert (await post(finished)).status_code == 200
        assert (await post(progress)).status_code == 422
        status = (await client.get(f"/v1/queue/{request_id}", headers=owner_headers(request_id))).json()
        assert status["progress"] == finished
        assert not {"payload", "tracks", "sourceSha256", "model"} & status.keys()
        assert (await post({**progress, "trackIndex": 1, "trackId": "track-2"})).status_code == 200
        assert (await complete(client, worker, leased, payload())).status_code == 200
        assert (await pending).status_code == 200
        assert (await post(finished)).status_code == 404
        assert_clean(app, job)


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["model", "track", "order", "source", "clock", "hash", "bytes", "chunks", "wav-clock", "truncated"])
async def test_malformed_result_cannot_replace_originals(tmp_path, change):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        _, pending, job = await admitted(app, client)
        leased = await lease(client, worker)
        outputs = (wav(), wav())
        metadata = result(payload())
        track = metadata["tracks"][0]
        if change == "model": metadata["modelSha256"] = "0" * 64
        elif change == "track": track["trackLabel"] = "different"
        elif change == "order": metadata["tracks"].reverse()
        elif change == "source": track["sourceSha256"] = "0" * 64
        elif change == "clock": track["sampleRate"] = 8000
        elif change == "hash": track["wavSha256"] = "0" * 64
        elif change == "bytes": track["totalBytes"] += 2
        elif change == "chunks": track["chunkCount"] = 2
        else:
            bad = bytearray(wav())
            if change == "wav-clock": struct.pack_into("<I", bad, 24, 8000)
            else: bad = bad[:-2]
            outputs = (bytes(bad), wav())
            track["wavSha256"] = hashlib.sha256(outputs[0]).hexdigest()
        assert (await complete(client, worker, leased, payload(), outputs=outputs, metadata=metadata)).status_code == 422
        assert (await pending).status_code == 503
        assert_clean(app, job)


@pytest.mark.anyio
async def test_completion_authentication_precedes_upload_and_duplicate_json_is_rejected(tmp_path):
    async with environment(tmp_path) as (app, client):
        worker, stranger = await register(client), await register(client)
        _, pending, job = await admitted(app, client)
        leased = await lease(client, worker)
        assert (await complete(client, worker, leased, payload(), headers={"Authorization": f"Bearer {OTHER_OWNER}"})).status_code == 403
        assert (await complete(client, stranger, leased, payload())).status_code == 403
        assert not job.result.done()
        response = await client.post(f"/v1/jobs/{job.job_id}/complete", json={
            **worker, "leaseToken": leased["leaseToken"], "result": result(payload()),
        })
        assert response.status_code == 422
        assert (await pending).status_code == 503
        assert_clean(app, job)


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["hash", "clock", "duplicate", "extra", "nan", "truncated", "riff", "oversized", "duration"])
async def test_original_validation_rejects_before_enqueue(tmp_path, change):
    originals = (wav(), wav())
    body = payload(originals)
    changes = {}
    if change == "hash": body["tracks"][0]["sourceSha256"] = "0" * 64
    elif change == "clock": body["tracks"][0]["frameCount"] += 1
    elif change == "duplicate": body["tracks"][1]["fieldName"] = "audio:1"
    elif change == "extra": body["transcript"] = "must not be accepted"
    elif change == "nan": originals = (wav(floating=True, nonfinite=True), wav())
    elif change == "truncated": originals = (wav()[:-2], wav())
    elif change == "riff": originals = (wav() + b"extra", wav())
    elif change == "oversized": changes = {"max_track_bytes": 300}
    elif change == "duration": changes = {"max_audio_seconds": 0.001}
    if change in ("nan", "truncated", "riff"):
        body = payload(originals)
    async with environment(tmp_path, **changes) as (app, client):
        response = await submit(client, str(uuid.uuid4()), body, originals)
        assert response.status_code == (413 if change in ("oversized", "duration") else 422)
        assert app.state.coordinator.inflight == 0 and not app.state.coordinator.jobs


@pytest.mark.anyio
async def test_header_admission_and_release_guards(tmp_path):
    async with environment(tmp_path, max_inflight_requests=1, require_current_release=True) as (app, client):
        worker = await register(client)
        request_id, pending, job = await admitted(app, client)
        assert (await submit(client, str(uuid.uuid4()))).status_code == 429
        headers = owner_headers(str(uuid.uuid4()))
        for missing, expected in (("X-Babel-Local-Engine", 403), ("X-Babel-Request-Id", 400),
                                  ("Authorization", 401), ("X-Babel-Inference-Release", 426)):
            reduced = {key: value for key, value in headers.items() if key.lower() != missing.lower()}
            response = await client.post("/v1/enhance", headers=reduced, data={"payload": json.dumps(payload())}, files=files())
            assert response.status_code == expected
        pending.cancel()
        with pytest.raises(asyncio.CancelledError): await pending
        assert_clean(app, job)


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_stream_retains_files_and_admission_until_send_finishes_or_cancels(tmp_path, cancel):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_app(scope, receive, send):
            async def held_send(message):
                if scope["path"] == "/v1/enhance" and message["type"] == "http.response.start":
                    entered.set()
                    await release.wait()
                await send(message)
            await app(scope, receive, held_send)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=held_app), base_url="http://coordinator") as owner:
            _, pending, job = await admitted(app, owner)
            leased = await lease(client, worker)
            assert (await complete(client, worker, leased, payload())).status_code == 200
            await asyncio.wait_for(entered.wait(), timeout=3)
            assert app.state.coordinator.inflight == 1
            assert all(path.exists() for path in (*job.paths.values(), *job.output_paths.values()))
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError): await pending
            else:
                release.set()
                assert (await pending).status_code == 200
            assert_clean(app, job)


@pytest.mark.anyio
async def test_http_disconnect_revokes_lease_and_removes_originals(tmp_path):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        request_id = str(uuid.uuid4())
        request = httpx.Request("POST", "http://coordinator/v1/enhance", headers=owner_headers(request_id),
                                data={"payload": json.dumps(payload())}, files=files())
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "http.request", "body": request.read(), "more_body": False})
        admitted_event = asyncio.Event()
        enqueue = app.state.coordinator.enqueue
        def notify(job):
            enqueue(job)
            admitted_event.set()
        app.state.coordinator.enqueue = notify
        sent = []
        async def send(message): sent.append(message)
        pending = asyncio.create_task(app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
            "scheme": "http", "path": "/v1/enhance", "raw_path": b"/v1/enhance", "query_string": b"",
            # HTTPX preserves caller header casing; ASGI requires lowercase names,
            # just as HTTPX's ASGITransport supplies for the other route tests.
            "headers": [(name.lower(), value) for name, value in request.headers.raw],
            "client": ("127.0.0.1", 1), "server": ("coordinator", 80),
        }, incoming.get, send))
        await asyncio.wait_for(admitted_event.wait(), timeout=3)
        job = app.state.coordinator.by_request_id[request_id]
        leased = await lease(client, worker)
        incoming.put_nowait({"type": "http.disconnect"})
        await asyncio.wait_for(pending, timeout=3)
        assert sent[0]["status"] == 499
        assert (await complete(client, worker, leased, payload())).status_code == 404
        assert_clean(app, job)


@pytest.mark.anyio
async def test_multipart_completion_uses_binary_body_limit_not_json_worker_limit(monkeypatch):
    monkeypatch.setattr(coordinator_module, "WORKER_BODY_BYTES", 16)
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    limited = BodyLimitMiddleware(app, max_bytes=64)
    for path, content_type, size, expected in (
        ("/v1/jobs/job/complete", b"multipart/form-data; boundary=test", 32, 200),
        ("/v1/jobs/job/complete", b"application/json", 32, 413),
        ("/v1/jobs/job/complete", b"multipart/form-data; boundary=test", 65, 413),
        ("/v1/workers/register", b"multipart/form-data", 32, 413),
    ):
        sent = []
        async def receive(): return {"type": "http.request", "body": b"x" * size}
        async def send(message): sent.append(message)
        await limited({"type": "http", "path": path, "headers": [(b"content-type", content_type)]}, receive, send)
        assert sent[0]["status"] == expected


@pytest.mark.anyio
async def test_parser_closes_partial_spools_on_asgi_body_limit(monkeypatch):
    from l0_draft_engine import enhancement
    class Spool:
        closed = False
        def close(self): self.closed = True
    spool = Spool()
    class Parser:
        def __init__(self, *args, **kwargs): self._files_to_close_on_error = [spool]
        async def parse(self): raise BodyTooLarge
    monkeypatch.setattr(enhancement, "MultiPartParser", Parser)
    with pytest.raises(BodyTooLarge):
        await read_multipart(Request({"type": "http", "headers": []}))
    assert spool.closed


@pytest.mark.anyio
async def test_owner_cancel_interrupts_stalled_result_upload_and_deletes_partial_output(tmp_path, monkeypatch):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        _, pending, job = await admitted(app, client)
        leased = await lease(client, worker)
        started = asyncio.Event()
        copy = coordinator_module.copy_enhancement_audio

        async def stalled_copy(upload, path, identity, **kwargs):
            await copy(upload, path, identity, **kwargs)
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(coordinator_module, "copy_enhancement_audio", stalled_copy)
        uploading = asyncio.create_task(complete(client, worker, leased, payload()))
        await asyncio.wait_for(started.wait(), timeout=3)
        directory = next(iter(job.paths.values())).parent
        assert (directory / "enhanced-0.wav").exists()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=3)
        with pytest.raises(asyncio.CancelledError):
            await uploading
        assert not directory.exists()
        assert_clean(app, job)


@pytest.mark.anyio
async def test_duplicate_owner_request_cannot_remove_original_pending_job(tmp_path):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        request_id, pending, job = await admitted(app, client)
        assert (await submit(client, request_id)).status_code == 409
        assert app.state.coordinator.by_request_id[request_id] is job
        assert app.state.coordinator.inflight == 1
        leased = await lease(client, worker)
        assert (await complete(client, worker, leased, payload())).status_code == 200
        assert (await pending).status_code == 200
        assert_clean(app, job)


@pytest.mark.anyio
async def test_queued_owner_abort_removes_unleased_work(tmp_path):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        request_id, pending, job = await admitted(app, client)
        assert job.phase == "queued" and job.worker_id is None
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (await client.post("/v1/workers/lease", json=worker)).status_code == 204
        assert (await client.get(f"/v1/queue/{request_id}", headers=owner_headers(request_id))).status_code == 404
        assert_clean(app, job)


@pytest.mark.anyio
async def test_oversized_result_is_rejected_before_reading_and_worker_can_finish_valid_pair(tmp_path):
    async with environment(tmp_path) as (app, client):
        worker = await register(client)
        _, pending, job = await admitted(app, client)
        leased = await lease(client, worker)
        denied = await client.post(f"/v1/jobs/{job.job_id}/complete", content=b"", headers={
            "Content-Type": "multipart/form-data; boundary=test",
            "Content-Length": "12001", "Authorization": f"Bearer {leased['leaseToken']}",
        })
        assert denied.status_code == 413 and not job.result.done()
        assert (await complete(client, worker, leased, payload())).status_code == 200
        assert (await pending).status_code == 200
        assert_clean(app, job)
