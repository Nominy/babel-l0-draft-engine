from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import httpx
import pytest

from l0_draft_engine.app import create_app
from l0_draft_engine.config import Settings
from l0_draft_engine.schemas import DraftResponse, DraftRow

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
    def __init__(self) -> None:
        self.draft_calls = 0
        self.paths: list[Path] = []
        self.busy = False

    def try_admit(self) -> bool:
        if self.busy:
            return False
        self.busy = True
        return True

    def release_admission(self) -> None:
        self.busy = False

    def health(self) -> dict[str, object]:
        return {"ok": True, "device": "cuda", "models": {"loaded": False}}

    def draft(self, payload, paths) -> DraftResponse:
        self.draft_calls += 1
        self.paths = list(paths.values())
        assert set(paths) == {"speaker-1", "speaker-2"}
        assert all(path.is_file() for path in paths.values())
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


def payload(tracks: int = 2) -> str:
    values = [
        {"lane": "speaker-1", "fieldName": "audio:1"},
        {"lane": "speaker-2", "fieldName": "audio:2"},
    ][:tracks]
    return json.dumps({"taskId": "task-1", "tracks": values})


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
async def test_draft_requires_proxy_header_and_rejects_busy_requests() -> None:
    settings = Settings()
    engine = FakeEngine()
    app = create_app(settings, engine)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        forbidden = await client.post("/v1/draft", content=b"")
        engine.busy = True
        busy = await client.post("/v1/draft", content=b"", headers=PROXY_HEADERS)
    assert forbidden.status_code == 403
    assert busy.status_code == 429
    assert busy.headers["retry-after"] == "1"
    assert engine.draft_calls == 0


@pytest.mark.anyio
async def test_cors_allows_only_loopback_web_origins() -> None:
    settings = Settings()
    app = create_app(settings, FakeEngine())  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        local = await client.get("/health", headers={"Origin": "http://localhost:3000"})
        remote = await client.get("/health", headers={"Origin": "https://example.com"})
    assert local.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in remote.headers
