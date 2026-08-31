from __future__ import annotations

import asyncio
import io
import json
import threading
import wave
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from l0_draft_engine.app import create_app
from l0_draft_engine.config import Settings
from l0_draft_engine.schemas import (
    DraftPayload,
    DraftResponse,
    DraftRow,
    TrackSpec,
    TranscriptionResponse,
    TranscriptionToken,
    TranscriptionTrack,
)

@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

PROXY_HEADERS = {"X-Babel-Local-Engine": "1"}


def wav_bytes(channels: int = 1, frames: int = 1600) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * frames * channels)
    return output.getvalue()


class FakeEngine:
    def __init__(self, *, hold_first_draft: bool = False) -> None:
        self.draft_calls = 0
        self.transcribe_calls = 0
        self.paths: list[Path] = []
        self.active_drafts = 0
        self.max_active_drafts = 0
        self.first_draft_started = threading.Event()
        self.release_first_draft = threading.Event()
        self._draft_state_lock = threading.Lock()
        if not hold_first_draft:
            self.release_first_draft.set()

    def health(self) -> dict[str, object]:
        return {"ok": True, "device": "cuda", "models": {"loaded": False}}

    def draft(self, payload, paths) -> DraftResponse:
        with self._draft_state_lock:
            self.draft_calls += 1
            call_number = self.draft_calls
            self.active_drafts += 1
            self.max_active_drafts = max(self.max_active_drafts, self.active_drafts)
        try:
            self.paths = list(paths.values())
            assert set(paths) == {"speaker-1", "speaker-2"}
            assert all(path.is_file() for path in paths.values())
            if call_number == 1:
                self.first_draft_started.set()
                if not self.release_first_draft.wait(timeout=5):
                    raise TimeoutError("test did not release the first draft")
            return DraftResponse(
                rows=[
                    DraftRow(
                        id="aab3266d-21b8-5082-814d-b2a5df1e15be",
                        lane="speaker-1",
                        startSeconds=0.0,
                        endSeconds=0.1,
                        text="Мгм.",
                    )
                ],
                summary={"rowCount": 1},
                models={"asr": "mock"},
            )
        finally:
            with self._draft_state_lock:
                self.active_drafts -= 1

    def transcribe(self, payload, paths) -> TranscriptionResponse:
        with self._draft_state_lock:
            self.transcribe_calls += 1
            self.active_drafts += 1
            self.max_active_drafts = max(self.max_active_drafts, self.active_drafts)
        try:
            self.paths = list(paths.values())
            assert set(paths) == {"speaker-1", "speaker-2"}
            assert all(path.is_file() for path in paths.values())
            return TranscriptionResponse(
                taskId=payload.taskId,
                tracks=[
                    TranscriptionTrack(
                        lane="speaker-1",
                        tokens=[
                            TranscriptionToken(
                                id="timing-1",
                                text="Привет",
                                startSeconds=0.01,
                                endSeconds=0.08,
                            )
                        ],
                    ),
                    TranscriptionTrack(lane="speaker-2", tokens=[]),
                ],
                summary={"trackCount": 2, "tokenCount": 1},
                models={"asr": "mock"},
            )
        finally:
            with self._draft_state_lock:
                self.active_drafts -= 1


def payload(tracks: int = 2) -> str:
    values = [
        {"lane": "speaker-1", "fieldName": "audio:1"},
        {"lane": "speaker-2", "fieldName": "audio:2"},
    ][:tracks]
    return json.dumps({"taskId": "task-1", "tracks": values})


def test_track_lanes_accept_human_and_cyrillic_labels() -> None:
    draft_payload = DraftPayload.model_validate(
        {
            "taskId": "task-1",
            "tracks": [
                {"lane": "Speaker 1", "fieldName": "audio:1"},
                {"lane": "Говорящий 2", "fieldName": "audio:2"},
            ],
        }
    )

    assert [track.lane for track in draft_payload.tracks] == [
        "Speaker 1",
        "Говорящий 2",
    ]


def test_track_field_name_remains_a_safe_multipart_identifier() -> None:
    with pytest.raises(ValidationError, match="must contain only letters"):
        TrackSpec.model_validate({"lane": "Speaker 1", "fieldName": "audio field"})


@pytest.mark.parametrize("lane", ["Speaker\n1", "Говорящий\u007f2"])
def test_track_lane_rejects_control_characters(lane: str) -> None:
    with pytest.raises(ValidationError, match="must not contain control characters"):
        TrackSpec.model_validate({"lane": lane, "fieldName": "audio:1"})


def test_track_lane_preserves_length_limit() -> None:
    with pytest.raises(ValidationError):
        TrackSpec.model_validate({"lane": "С" * 129, "fieldName": "audio:1"})


@pytest.mark.anyio
async def test_health_does_not_load_models(tmp_path: Path) -> None:
    called = False

    def forbidden_factory():
        nonlocal called
        called = True
        raise AssertionError("health loaded a model")

    from l0_draft_engine.engine import DraftEngine

    gigaam_path = tmp_path / "gigaam.ckpt"
    punctuation_path = tmp_path / "punctuation"
    gigaam_path.touch()
    punctuation_path.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (punctuation_path / name).touch()
    settings = Settings(
        device="cpu",
        gigaam_model_path=gigaam_path,
        punctuation_model_path=punctuation_path,
    )
    engine = DraftEngine(
        settings,
        asr_factory=forbidden_factory,
        formatter_factory=forbidden_factory,
    )
    app = create_app(settings, engine)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["device"] == "cpu"
    assert response.json()["models"]["asr"]["loaded"] is False
    assert response.json()["models"]["asr"]["cached"] is True
    assert called is False


@pytest.mark.anyio
async def test_draft_accepts_declared_colon_fields_and_cleans_temp_files() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    files = {
        "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
        "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/draft",
            data={"payload": payload()},
            files=files,
            headers=PROXY_HEADERS,
        )
    assert response.status_code == 200, response.text
    assert response.json()["rows"][0]["endSeconds"] > response.json()["rows"][0]["startSeconds"]
    assert engine.draft_calls == 1
    assert engine.paths and all(not path.exists() for path in engine.paths)


@pytest.mark.anyio
async def test_transcribe_uses_draft_multipart_contract_and_cleans_temp_files() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/transcribe",
            data={"payload": payload()},
            files={
                "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
                "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
            },
            headers=PROXY_HEADERS,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"taskId", "tracks", "summary", "models"}
    assert body["taskId"] == "task-1"
    assert [track["lane"] for track in body["tracks"]] == [
        "speaker-1",
        "speaker-2",
    ]
    assert body["tracks"][0]["tokens"] == [
        {
            "id": "timing-1",
            "text": "Привет",
            "startSeconds": 0.01,
            "endSeconds": 0.08,
        }
    ]
    assert body["tracks"][1]["tokens"] == []
    assert engine.transcribe_calls == 1
    assert engine.paths and all(not path.exists() for path in engine.paths)


@pytest.mark.anyio
async def test_draft_rejects_any_count_other_than_two_tracks() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/draft",
            data={"payload": payload(1)},
            files={"audio:1": ("only.wav", wav_bytes(), "audio/wav")},
            headers=PROXY_HEADERS,
        )
    assert response.status_code == 422
    assert engine.draft_calls == 0


@pytest.mark.anyio
async def test_invalid_payload_returns_a_json_serializable_422() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    invalid_payload = json.dumps(
        {
            "taskId": "task-1",
            "tracks": [
                {"lane": "Speaker\n1", "fieldName": "audio:1"},
                {"lane": "Говорящий 2", "fieldName": "audio:2"},
            ],
        }
    )
    files = {
        "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
        "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/draft",
            data={"payload": invalid_payload},
            files=files,
            headers=PROXY_HEADERS,
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["tracks", 0, "lane"]
    assert "ctx" not in detail[0]
    json.dumps(detail)
    assert engine.draft_calls == 0


@pytest.mark.anyio
async def test_draft_rejects_stereo_track_before_inference() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    files = {
        "audio:1": ("first.wav", wav_bytes(channels=2), "audio/wav"),
        "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/draft",
            data={"payload": payload()},
            files=files,
            headers=PROXY_HEADERS,
        )
    assert response.status_code == 422
    assert "mono" in response.json()["detail"]
    assert engine.draft_calls == 0


@pytest.mark.anyio
async def test_request_content_length_limit_is_enforced_before_form_parsing() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/draft",
            content=b"ignored",
            headers={
                **PROXY_HEADERS,
                "Content-Length": str(settings.max_request_bytes + 1),
            },
        )
    assert response.status_code == 413
    assert engine.draft_calls == 0

@pytest.mark.anyio
async def test_chunked_request_body_limit_cannot_be_bypassed() -> None:
    settings = Settings(max_track_bytes=1024, max_request_bytes=4096)
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]

    async def oversized_body():
        yield b"--boundary\r\nContent-Disposition: form-data; name=\"payload\"\r\n\r\n"
        yield b"x" * settings.max_request_bytes
        yield b"\r\n--boundary--\r\n"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/draft",
            content=oversized_body(),
            headers={
                **PROXY_HEADERS,
                "Content-Type": "multipart/form-data; boundary=boundary",
            },
        )
    assert response.status_code == 413
    assert engine.draft_calls == 0


@pytest.mark.anyio
async def test_draft_requires_proxy_header() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        forbidden = await client.post("/v1/draft", content=b"")
    assert forbidden.status_code == 403
    assert engine.draft_calls == 0


@pytest.mark.anyio
async def test_concurrent_draft_requests_wait_and_execute_one_at_a_time() -> None:
    settings = Settings()
    engine = FakeEngine(hold_first_draft=True)
    app = create_app(settings, engine)  # type: ignore[arg-type]

    async def post_draft(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post(
            "/v1/draft",
            data={"payload": payload()},
            files={
                "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
                "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
            },
            headers=PROXY_HEADERS,
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        first_request = asyncio.create_task(post_draft(client))
        assert await asyncio.to_thread(engine.first_draft_started.wait, 2)
        second_request = asyncio.create_task(post_draft(client))
        await asyncio.sleep(0)
        try:
            assert not second_request.done()
            assert engine.draft_calls == 1
            assert engine.max_active_drafts == 1
        finally:
            engine.release_first_draft.set()
        first, second = await asyncio.gather(first_request, second_request)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert engine.draft_calls == 2
    assert engine.max_active_drafts == 1


@pytest.mark.anyio
async def test_admission_rejects_a_fourth_request_before_body_parsing() -> None:
    settings = Settings(max_inflight_requests=3)
    engine = FakeEngine(hold_first_draft=True)
    app = create_app(settings, engine)  # type: ignore[arg-type]

    async def post_draft(
        client: httpx.AsyncClient, request_id: str
    ) -> httpx.Response:
        return await client.post(
            "/v1/draft",
            data={"payload": payload()},
            files={
                "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
                "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
            },
            headers={**PROXY_HEADERS, "X-Babel-Request-Id": request_id},
        )

    async def wait_until_registered(
        client: httpx.AsyncClient, request_id: str
    ) -> None:
        for _ in range(100):
            if (await client.get(f"/v1/queue/{request_id}")).status_code == 200:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"queue status never appeared for {request_id}")

    body_was_read = False

    async def rejected_body():
        nonlocal body_was_read
        body_was_read = True
        yield b"this body must not be parsed"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        first_task = asyncio.create_task(post_draft(client, "admitted-first"))
        assert await asyncio.to_thread(engine.first_draft_started.wait, 2)
        second_task = asyncio.create_task(post_draft(client, "admitted-second"))
        await wait_until_registered(client, "admitted-second")
        third_task = asyncio.create_task(post_draft(client, "admitted-third"))
        await wait_until_registered(client, "admitted-third")
        try:
            rejected = await client.post(
                "/v1/draft",
                content=rejected_body(),
                headers={
                    **PROXY_HEADERS,
                    "Content-Type": "multipart/form-data; boundary=unused",
                    "X-Babel-Request-Id": "rejected-fourth",
                    "Origin": "https://dashboard.babel.audio",
                },
            )
            assert rejected.status_code == 429
            assert rejected.headers["retry-after"] == "5"
            assert rejected.headers["access-control-expose-headers"] == "Retry-After"
            assert rejected.json()["detail"] == "too many in-flight requests"
            assert body_was_read is False
            assert engine.draft_calls == 1
            assert (
                await client.get("/v1/queue/rejected-fourth")
            ).status_code == 404
        finally:
            engine.release_first_draft.set()
            admitted_responses = await asyncio.gather(
                first_task, second_task, third_task
            )

    assert all(response.status_code == 200 for response in admitted_responses)
    assert engine.draft_calls == 3
    assert engine.max_active_drafts == 1


@pytest.mark.anyio
async def test_admission_slot_recovers_after_invalid_and_completed_requests() -> None:
    settings = Settings(max_inflight_requests=1)
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]

    async def post_draft(
        client: httpx.AsyncClient, request_payload: str
    ) -> httpx.Response:
        return await client.post(
            "/v1/draft",
            data={"payload": request_payload},
            files={
                "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
                "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
            },
            headers=PROXY_HEADERS,
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        invalid = await post_draft(client, "{")
        after_invalid = await post_draft(client, payload())
        after_completion = await post_draft(client, payload())

    assert invalid.status_code == 422
    assert after_invalid.status_code == 200, after_invalid.text
    assert after_completion.status_code == 200, after_completion.text
    assert engine.draft_calls == 2


@pytest.mark.anyio
async def test_queue_status_reports_running_position_and_completion() -> None:
    settings = Settings()
    engine = FakeEngine(hold_first_draft=True)
    app = create_app(settings, engine)  # type: ignore[arg-type]

    async def post_draft(client: httpx.AsyncClient, request_id: str) -> httpx.Response:
        return await client.post(
            "/v1/draft",
            data={"payload": payload()},
            files={
                "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
                "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
            },
            headers={**PROXY_HEADERS, "X-Babel-Request-Id": request_id},
        )

    async def wait_for_status(
        client: httpx.AsyncClient, request_id: str
    ) -> httpx.Response:
        for _ in range(100):
            response = await client.get(f"/v1/queue/{request_id}")
            if response.status_code == 200:
                return response
            await asyncio.sleep(0.01)
        raise AssertionError(f"queue status never appeared for {request_id}")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        first_task = asyncio.create_task(post_draft(client, "first-request"))
        assert await asyncio.to_thread(engine.first_draft_started.wait, 2)
        second_task = asyncio.create_task(post_draft(client, "second-request"))
        second_status = await wait_for_status(client, "second-request")
        first_status = await wait_for_status(client, "first-request")

        assert first_status.json() == {
            "requestId": "first-request",
            "status": "running",
            "position": 0,
            "queuedCount": 1,
        }
        assert second_status.json() == {
            "requestId": "second-request",
            "status": "queued",
            "position": 1,
            "queuedCount": 1,
        }
        assert (await client.get("/v1/queue/unknown")).status_code == 404

        engine.release_first_draft.set()
        first_response, second_response = await asyncio.gather(
            first_task, second_task
        )
        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert (await client.get("/v1/queue/first-request")).json()["status"] == "completed"
        assert (await client.get("/v1/queue/second-request")).json()["status"] == "completed"


@pytest.mark.anyio
async def test_draft_and_transcribe_share_one_inference_queue() -> None:
    settings = Settings()
    engine = FakeEngine(hold_first_draft=True)
    app = create_app(settings, engine)  # type: ignore[arg-type]

    async def post(client: httpx.AsyncClient, endpoint: str) -> httpx.Response:
        return await client.post(
            endpoint,
            data={"payload": payload()},
            files={
                "audio:1": ("first.wav", wav_bytes(), "audio/wav"),
                "audio:2": ("second.wav", wav_bytes(), "audio/wav"),
            },
            headers=PROXY_HEADERS,
        )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        draft_request = asyncio.create_task(post(client, "/v1/draft"))
        assert await asyncio.to_thread(engine.first_draft_started.wait, 2)
        transcription_request = asyncio.create_task(
            post(client, "/v1/transcribe")
        )
        await asyncio.sleep(0)
        try:
            assert not transcription_request.done()
            assert engine.draft_calls == 1
            assert engine.transcribe_calls == 0
            assert engine.max_active_drafts == 1
        finally:
            engine.release_first_draft.set()
        draft_response, transcription_response = await asyncio.gather(
            draft_request, transcription_request
        )

    assert draft_response.status_code == 200, draft_response.text
    assert transcription_response.status_code == 200, transcription_response.text
    assert engine.draft_calls == 1
    assert engine.transcribe_calls == 1
    assert engine.max_active_drafts == 1


@pytest.mark.anyio
async def test_cors_allows_dashboard_and_loopback_origins_only() -> None:
    settings = Settings()
    app = create_app(settings, FakeEngine())  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        dashboard = await client.get(
            "/health", headers={"Origin": "https://dashboard.babel.audio"}
        )
        local = await client.get("/health", headers={"Origin": "http://localhost:3000"})
        remote = await client.get("/health", headers={"Origin": "https://example.com"})
    assert dashboard.headers["access-control-allow-origin"] == "https://dashboard.babel.audio"
    assert local.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in remote.headers


@pytest.mark.anyio
async def test_cors_preflight_allows_engine_request_headers() -> None:
    app = create_app(Settings(), FakeEngine())  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.options(
            "/v1/draft",
            headers={
                "Origin": "https://dashboard.babel.audio",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "content-type, x-babel-local-engine, x-babel-request-id"
                ),
            },
        )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://dashboard.babel.audio"
    )
    assert "POST" in response.headers["access-control-allow-methods"].split(", ")
    allowed_headers = {
        header.strip().lower()
        for header in response.headers["access-control-allow-headers"].split(",")
    }
    assert {
        "content-type",
        "x-babel-local-engine",
        "x-babel-request-id",
    } <= allowed_headers
