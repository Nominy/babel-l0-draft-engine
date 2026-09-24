from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import wave

import httpx
import pytest

from l0_draft_engine.coordinator import CoordinatorSettings, create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def wav_bytes() -> bytes:
    result = io.BytesIO()
    with wave.open(result, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x01" * 1600)
    return result.getvalue()


def payload(task_id: str = "task-1") -> dict[str, object]:
    return {
        "taskId": task_id,
        "tracks": [
            {"lane": "speaker-1", "fieldName": "audio:1"},
            {"lane": "speaker-2", "fieldName": "audio:2"},
        ],
    }


def draft_result() -> dict[str, object]:
    return {
        "rows": [{"id": "row-1", "lane": "speaker-1", "startSeconds": 0,
                  "endSeconds": 0.1, "text": "Привет."}],
        "summary": {"rowCount": 1}, "models": {"asr": "browser"},
    }


def timing_result() -> dict[str, object]:
    return {
        "taskId": "task-1",
        "tracks": [
            {"lane": "speaker-1", "tokens": [{"id": "token-1", "text": "Привет",
                                                 "startSeconds": 0, "endSeconds": 0.1}]},
            {"lane": "speaker-2", "tokens": []},
        ],
        "summary": {"tokenCount": 1}, "models": {"asr": "browser"},
    }


def settings(**changes: object) -> CoordinatorSettings:
    return CoordinatorSettings(
        backend_urls=("http://trusted",), max_track_bytes=4096,
        max_request_bytes=12_000, **changes,
    )


async def submit(client: httpx.AsyncClient, operation: str, request_id: str,
                 *, task_id: str = "task-1", first: bytes | None = None) -> httpx.Response:
    return await client.post(
        f"/v1/{operation}",
        data={"payload": json.dumps(payload(task_id))},
        files={"audio:1": ("first.wav", first if first is not None else wav_bytes(), "audio/wav"),
               "audio:2": ("second.wav", wav_bytes(), "audio/wav")},
        headers={"X-Babel-Local-Engine": "1", "X-Babel-Request-Id": request_id},
    )


async def wait_for_status(client: httpx.AsyncClient, request_id: str) -> dict[str, object]:
    for _ in range(100):
        response = await client.get(f"/v1/queue/{request_id}")
        if response.status_code == 200:
            return response.json()
        await asyncio.sleep(0.01)
    raise AssertionError(f"request {request_id} was not admitted")


async def register(client: httpx.AsyncClient) -> dict[str, str]:
    response = await client.post(
        "/v1/workers/register", json={"modelBundleSchema": "babel-browser-model-bundle-v2"}
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.anyio
async def test_two_workers_take_distinct_jobs_and_return_draft_and_timing() -> None:
    fallback_calls: list[str] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/queue/"):
            return httpx.Response(404)
        fallback_calls.append(request.url.path)
        return httpx.Response(503)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = create_app(settings(), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
        first_worker = await register(client)
        second_worker = await register(client)
        first = asyncio.create_task(submit(client, "draft", "first"))
        second = asyncio.create_task(submit(client, "transcribe", "second"))
        assert (await wait_for_status(client, "first"))["status"] == "queued"
        assert (await wait_for_status(client, "second"))["status"] == "queued"
        leases = []
        for worker in (first_worker, second_worker):
            lease = await client.post("/v1/workers/lease", json=worker)
            assert lease.status_code == 200, lease.text
            leases.append((worker, lease.json()))
        assert {lease["operation"] for _, lease in leases} == {"draft", "transcribe"}
        assert len({lease["jobId"] for _, lease in leases}) == 2
        for worker, lease in leases:
            assert (await client.post("/v1/workers/lease", json=worker)).status_code == 204
            assert (await client.get(lease["audio"][0]["url"])).status_code == 401
            assert (await client.get(lease["audio"][0]["url"], headers={
                "Authorization": "Bearer incorrect"
            })).status_code == 403
            audio = await client.get(lease["audio"][0]["url"], headers={
                "Authorization": f"Bearer {lease['leaseToken']}"
            })
            assert audio.status_code == 200
            assert audio.content == wav_bytes()
            assert (await client.get(f"/v1/queue/{'first' if lease['operation'] == 'draft' else 'second'}")).json()["status"] == "running"
            result = draft_result() if lease["operation"] == "draft" else timing_result()
            complete = await client.post(f"/v1/jobs/{lease['jobId']}/complete", json={
                **worker, "leaseToken": lease["leaseToken"], "result": result,
            })
            assert complete.status_code == 200, complete.text
        draft, transcription = await asyncio.gather(first, second)
        assert draft.status_code == transcription.status_code == 200
        assert draft.json() == draft_result()
        assert transcription.json() == timing_result()
        assert (await client.get("/v1/queue/first")).json()["status"] == "completed"
        assert (await client.get("/v1/queue/second")).json()["status"] == "completed"
        assert (await client.post(f"/v1/jobs/{leases[0][1]['jobId']}/complete", json={
            **leases[0][0], "leaseToken": leases[0][1]["leaseToken"], "result": draft_result(),
        })).status_code == 404
    assert fallback_calls == []
    await upstream.aclose()


@pytest.mark.anyio
async def test_idle_fallback_preserves_payload_and_backend_queue_status() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    seen: list[tuple[httpx.Request, bytes]] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True, "device": "cuda", "models": {"asr": {}}})
        if request.url.path.startswith("/v1/queue/"):
            return httpx.Response(200, json={"requestId": "fallback", "status": "queued",
                                              "position": 2, "queuedCount": 2})
        seen.append((request, await request.aread()))
        started.set()
        await release.wait()
        return httpx.Response(200, json=draft_result())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = create_app(settings(), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
        task = asyncio.create_task(submit(client, "draft", "fallback", task_id="original-id"))
        await asyncio.wait_for(started.wait(), timeout=2)
        try:
            assert (await client.get("/v1/queue/fallback")).json()["position"] == 2
            health = (await client.get("/health")).json()
            assert health["ok"] is True and health["device"] == "cuda"
            assert health["models"] == {"asr": {}} and health["swarm"]["jobs"] == 1
        finally:
            release.set()
        result = await task
        assert result.status_code == 200
        assert result.json() == draft_result()
    assert len(seen) == 1
    assert seen[0][0].headers["x-babel-local-engine"] == "1"
    assert seen[0][0].headers["x-babel-request-id"] == "fallback"
    assert b'"original-id"' in seen[0][1]
    assert wav_bytes() in seen[0][1]
    await upstream.aclose()


@pytest.mark.anyio
async def test_dead_primary_fails_over_and_preserves_trusted_validation_error() -> None:
    hosts: list[str] = []

    def backend(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "dead":
            raise httpx.ConnectError("connection refused")
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True, "device": "mps", "models": {}})
        return httpx.Response(422, json={"detail": "unsupported audio"})

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = create_app(CoordinatorSettings(
        backend_urls=("http://dead", "http://healthy"), max_track_bytes=4096,
        max_request_bytes=12_000,
    ), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
        health = await client.get("/health")
        assert health.json()["device"] == "mps"
        response = await submit(client, "draft", "trusted-422")
        assert response.status_code == 422
        assert response.json() == {"detail": "unsupported audio"}
    assert hosts == ["dead", "healthy", "dead", "healthy"]
    await upstream.aclose()


@pytest.mark.anyio
async def test_inflight_limit_rejects_concurrent_upload_before_dispatch() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def backend(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path.startswith("/v1/queue/"):
            return httpx.Response(404)
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(200, json=draft_result())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = create_app(settings(max_inflight_requests=1), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
        first = asyncio.create_task(submit(client, "draft", "admitted"))
        await asyncio.wait_for(started.wait(), timeout=2)
        try:
            rejected = await submit(client, "draft", "rejected")
            assert rejected.status_code == 429
            assert rejected.headers["retry-after"] == "5"
            assert (await client.get("/v1/queue/rejected")).status_code == 404
        finally:
            release.set()
        assert (await first).status_code == 200
    assert calls == 1
    await upstream.aclose()


@pytest.mark.anyio
async def test_invalid_result_and_worker_error_fall_back_without_stranding_lease() -> None:
    calls = 0

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path.startswith("/v1/queue/"):
            return httpx.Response(404)
        calls += 1
        return httpx.Response(200, json=timing_result())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = create_app(settings(), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
        worker = await register(client)
        for request_id, body, expected in (
            ("wrong-lanes", {"result": {**timing_result(), "taskId": "wrong"}}, 422),
            ("volunteer-error", {"error": "model unavailable"}, 200),
        ):
            task = asyncio.create_task(submit(client, "transcribe", request_id))
            await wait_for_status(client, request_id)
            lease = (await client.post("/v1/workers/lease", json=worker)).json()
            denied = await client.post(f"/v1/jobs/{lease['jobId']}/complete", json={
                **worker, "leaseToken": "wrong", **body,
            })
            assert denied.status_code == 403
            complete = await client.post(f"/v1/jobs/{lease['jobId']}/complete", json={
                **worker, "leaseToken": lease["leaseToken"], **body,
            })
            assert complete.status_code == expected
            assert (await task).json() == timing_result()
        assert calls == 2
        assert (await client.post("/v1/workers/lease", json=worker)).status_code == 204
    await upstream.aclose()


@pytest.mark.anyio
async def test_lease_expiry_and_cancelled_request_release_audio_and_fall_back() -> None:
    calls = 0

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path.startswith("/v1/queue/"):
            return httpx.Response(404)
        calls += 1
        return httpx.Response(200, json=draft_result())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = create_app(settings(lease_seconds=0.02, queue_seconds=0.1), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
        worker = await register(client)
        task = asyncio.create_task(submit(client, "draft", "expired"))
        await wait_for_status(client, "expired")
        lease = (await client.post("/v1/workers/lease", json=worker)).json()
        assert (await task).json() == draft_result()
        assert calls == 1
        assert (await client.get(lease["audio"][0]["url"], headers={
            "Authorization": f"Bearer {lease['leaseToken']}"
        })).status_code == 404
        assert (await client.post("/v1/workers/lease", json=worker)).status_code == 204
        pending = asyncio.create_task(submit(client, "draft", "cancelled"))
        await wait_for_status(client, "cancelled")
        path: Path = next(iter(app.state.coordinator.by_request_id["cancelled"].paths.values()))
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not path.exists()
        assert (await client.get("/v1/queue/cancelled")).status_code == 404
        assert (await client.post("/v1/workers/lease", json=worker)).status_code == 204
    await upstream.aclose()


@pytest.mark.anyio
async def test_multipart_and_body_limits_prevent_dispatch() -> None:
    calls = 0

    def backend(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=draft_result())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = create_app(settings(), client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as client:
        assert (await submit(client, "draft", "stereo", first=b"not wav")).status_code == 422
        assert (await client.post("/v1/draft", content=b"", headers={
            "X-Babel-Local-Engine": "1", "Content-Length": "12001"
        })).status_code == 413
        assert (await client.post("/v1/draft", content=b"no-header")).status_code == 403
        assert (await client.post("/v1/workers/register", json={
            "modelBundleSchema": "unsupported"
        })).status_code == 422
        assert (await client.options("/v1/workers/register", headers={
            "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type, authorization",
        })).headers["access-control-allow-origin"] == "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    assert calls == 0
    await upstream.aclose()
