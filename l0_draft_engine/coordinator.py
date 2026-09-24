"""Lightweight public coordinator for browser volunteers and trusted engine failover.

Run with ``uvicorn l0_draft_engine.coordinator:app`` after installing
``requirements-coordinator.txt``. Keep one uvicorn process: leases and status
are deliberately in-memory and cannot be shared between processes.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, asynccontextmanager
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
from typing import Literal
from urllib.parse import quote
import uuid
import wave

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .schemas import DraftPayload, DraftResponse, TranscriptionResponse


REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
ALLOWED_ORIGIN_RE = (
    r"^(?:chrome-extension://[a-p]{32}|https://dashboard\.babel\.audio|"
    r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?)$"
)
CHUNK_BYTES = 1024 * 1024
WORKER_BODY_BYTES = 16 * 1024 * 1024


def _positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class CoordinatorSettings:
    backend_urls: tuple[str, ...] = (
        "http://127.0.0.1:18767",
        "http://127.0.0.1:28767",
        "http://127.0.0.1:38767",
    )
    max_track_bytes: int = 240 * 1024 * 1024
    max_request_bytes: int = 500 * 1024 * 1024
    max_inflight_requests: int = 8
    max_audio_seconds: float = 4 * 60 * 60
    queue_seconds: float = 15.0
    lease_seconds: float = 480.0
    worker_idle_seconds: float = 45.0
    backend_timeout_seconds: float = 900.0
    request_timeout_seconds: float = 895.0  # below the public Apache 900-second timeout

    def __post_init__(self) -> None:
        if not self.backend_urls or any(
            httpx.URL(url).scheme not in ("http", "https") or not httpx.URL(url).host
            for url in self.backend_urls
        ):
            raise ValueError("backend_urls must contain HTTP(S) base URLs")
        if self.max_track_bytes <= 0 or self.max_request_bytes <= 2 * self.max_track_bytes:
            raise ValueError("max_request_bytes must exceed two track limits")
        if not 1 <= self.max_inflight_requests <= 64:
            raise ValueError("max_inflight_requests must be between 1 and 64")
        if min(self.max_audio_seconds, self.queue_seconds, self.lease_seconds,
               self.worker_idle_seconds, self.backend_timeout_seconds,
               self.request_timeout_seconds) <= 0:
            raise ValueError("coordinator timeouts must be positive")
        if self.queue_seconds + self.lease_seconds >= self.request_timeout_seconds:
            raise ValueError("volunteer wait must leave time for trusted fallback")

    @classmethod
    def from_env(cls) -> CoordinatorSettings:
        urls = os.environ.get("COORDINATOR_BACKEND_URLS")
        defaults = cls()
        return cls(
            backend_urls=tuple(url.strip().rstrip("/") for url in urls.split(",") if url.strip())
            if urls is not None else defaults.backend_urls,
            max_track_bytes=_positive_env("COORDINATOR_MAX_TRACK_BYTES", defaults.max_track_bytes),
            max_request_bytes=_positive_env("COORDINATOR_MAX_REQUEST_BYTES", defaults.max_request_bytes),
            max_inflight_requests=_positive_env("COORDINATOR_MAX_INFLIGHT_REQUESTS", defaults.max_inflight_requests),
            max_audio_seconds=_positive_env("COORDINATOR_MAX_AUDIO_SECONDS", int(defaults.max_audio_seconds)),
            queue_seconds=_positive_env("COORDINATOR_QUEUE_SECONDS", int(defaults.queue_seconds)),
            lease_seconds=_positive_env("COORDINATOR_LEASE_SECONDS", int(defaults.lease_seconds)),
            worker_idle_seconds=_positive_env("COORDINATOR_WORKER_IDLE_SECONDS", int(defaults.worker_idle_seconds)),
            backend_timeout_seconds=_positive_env("COORDINATOR_BACKEND_TIMEOUT_SECONDS", int(defaults.backend_timeout_seconds)),
            request_timeout_seconds=_positive_env("COORDINATOR_REQUEST_SECONDS", int(defaults.request_timeout_seconds)),
        )


class BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = WORKER_BODY_BYTES if scope["path"].startswith(("/v1/workers/", "/v1/jobs/")) else self.max_bytes
        count = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal count
            message = await receive()
            if message["type"] == "http.request":
                count += len(message.get("body", b""))
                if count > limit:
                    raise BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except BodyTooLarge:
            if started:
                raise
            await JSONResponse({"detail": "request exceeds size limit"}, status_code=413)(scope, receive, send)


def _request_id(request: Request) -> str:
    if "x-babel-request-id" not in request.headers:
        return str(uuid.uuid4())
    value = request.headers["x-babel-request-id"]
    if not value or len(value) > 128 or not REQUEST_ID_RE.fullmatch(value):
        raise HTTPException(400, "X-Babel-Request-Id must be a nonempty safe identifier of at most 128 characters")
    return value


def _check_length(request: Request, maximum: int) -> None:
    raw = request.headers.get("content-length")
    if raw is None:
        return
    try:
        length = int(raw)
    except ValueError as exc:
        raise HTTPException(400, "invalid Content-Length") from exc
    if length < 0:
        raise HTTPException(400, "invalid Content-Length")
    if length > maximum:
        raise HTTPException(413, "request exceeds size limit")


def _parse_form(form: object) -> tuple[DraftPayload, str, dict[str, UploadFile]]:
    raw_payloads: list[str] = []
    files: dict[str, UploadFile] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            if key in files:
                raise HTTPException(422, f"duplicate audio field: {key}")
            files[key] = value
        elif key == "payload" and isinstance(value, str):
            raw_payloads.append(value)
        else:
            raise HTTPException(422, f"unexpected multipart field: {key}")
    if len(raw_payloads) != 1:
        raise HTTPException(422, "multipart request must contain exactly one payload field")
    try:
        payload = DraftPayload.model_validate_json(raw_payloads[0])
    except (ValidationError, json.JSONDecodeError) as exc:
        detail = exc.errors(include_context=False) if isinstance(exc, ValidationError) else "payload must be valid JSON"
        raise HTTPException(422, detail) from exc
    if len(files) != 2 or set(files) != {track.fieldName for track in payload.tracks}:
        raise HTTPException(422, "multipart request must contain exactly the two declared audio fields")
    return payload, raw_payloads[0], files


async def _copy_audio(upload: UploadFile, path: Path, limit: int, max_seconds: float) -> None:
    size = 0
    with path.open("xb") as output:
        while chunk := await upload.read(CHUNK_BYTES):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "audio track exceeds size limit")
            output.write(chunk)
    if not size:
        raise HTTPException(422, "audio track is empty")
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1:
                raise HTTPException(422, "each audio track must be mono")
            if audio.getcomptype() != "NONE":
                raise HTTPException(422, "audio tracks must be uncompressed WAV")
            if audio.getnframes() <= 0 or audio.getframerate() <= 0:
                raise HTTPException(422, "audio tracks must have positive duration")
            if audio.getnframes() / audio.getframerate() > max_seconds:
                raise HTTPException(413, "audio track exceeds duration limit")
    except HTTPException:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise HTTPException(422, "each audio track must be a valid WAV file") from exc


class RegisterBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modelBundleSchema: Literal["babel-browser-model-bundle-v2"]


class WorkerBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workerId: str
    token: str


class CompleteBody(WorkerBody):
    leaseToken: str
    result: dict[str, object] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def one_outcome(self) -> CompleteBody:
        if (self.result is None) == (self.error is None):
            raise ValueError("exactly one of result or error is required")
        return self


@dataclass
class Worker:
    token: str
    last_seen: float
    job_id: str | None = None


@dataclass
class Job:
    request_id: str
    operation: Literal["draft", "transcribe"]
    payload: DraftPayload
    raw_payload: str
    paths: dict[str, Path]
    filenames: dict[str, str]
    types: dict[str, str]
    result: asyncio.Future[DraftResponse | TranscriptionResponse | None]
    leased: asyncio.Event = field(default_factory=asyncio.Event)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str | None = None
    lease_token: str | None = None
    lease_deadline: float | None = None
    backend_url: str | None = None
    phase: Literal["queued", "running"] = "queued"


class Coordinator:
    def __init__(self, settings: CoordinatorSettings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self.workers: dict[str, Worker] = {}
        self.jobs: dict[str, Job] = {}
        self.by_request_id: dict[str, Job] = {}
        self.waiting: deque[str] = deque()
        self.completed: OrderedDict[str, float] = OrderedDict()
        self.inflight = 0

    def prune(self) -> None:
        now = time.monotonic()
        for worker_id, worker in tuple(self.workers.items()):
            if worker.job_id is None and now - worker.last_seen > self.settings.worker_idle_seconds:
                del self.workers[worker_id]
        while self.completed:
            _, expiry = next(iter(self.completed.items()))
            if expiry > now and len(self.completed) <= 1024:
                break
            self.completed.popitem(last=False)

    def idle_capacity(self) -> bool:
        self.prune()
        idle = sum(worker.job_id is None for worker in self.workers.values())
        return idle > len(self.waiting)

    def enqueue(self, job: Job) -> None:
        self.prune()
        if job.request_id in self.by_request_id or job.request_id in self.completed:
            raise HTTPException(409, "request ID is already registered")
        self.jobs[job.job_id] = job
        self.by_request_id[job.request_id] = job
        if self.idle_capacity():
            self.waiting.append(job.job_id)
        else:
            job.phase = "running"  # direct backend failover; never queue without a worker

    def release(self, job: Job) -> None:
        worker = self.workers.get(job.worker_id or "")
        if worker is not None and worker.job_id == job.job_id:
            worker.last_seen = time.monotonic()
            worker.job_id = None
        job.worker_id = None
        job.lease_token = None
        job.lease_deadline = None

    def fail(self, job: Job) -> None:
        self.release(job)
        if not job.result.done():
            job.result.set_result(None)

    def finish(self, job: Job, *, completed: bool) -> None:
        self.release(job)
        self.jobs.pop(job.job_id, None)
        self.by_request_id.pop(job.request_id, None)
        try:
            self.waiting.remove(job.job_id)
        except ValueError:
            pass
        if completed:
            self.completed[job.request_id] = time.monotonic() + 45
        self.prune()

    def authenticate(self, worker_id: str, token: str) -> Worker:
        worker = self.workers.get(worker_id)
        if worker is None or not secrets.compare_digest(worker.token, token):
            raise HTTPException(401, "worker credentials are invalid")
        return worker

    def status(self, request_id: str) -> dict[str, str | int] | None:
        self.prune()
        job = self.by_request_id.get(request_id)
        if job is not None:
            position = 0
            if job.phase == "queued":
                try:
                    position = self.waiting.index(job.job_id) + 1
                except ValueError:
                    pass
            return {"requestId": request_id, "status": job.phase,
                    "position": position, "queuedCount": len(self.waiting)}
        if request_id in self.completed:
            return {"requestId": request_id, "status": "completed", "position": 0,
                    "queuedCount": len(self.waiting)}
        return None


def _validated_result(job: Job, value: dict[str, object]) -> DraftResponse | TranscriptionResponse:
    json.dumps(value, allow_nan=False)
    if job.operation == "draft":
        result = DraftResponse.model_validate(value)
        lanes = {track.lane for track in job.payload.tracks}
        if any(row.lane not in lanes or not math.isfinite(row.startSeconds)
               or not math.isfinite(row.endSeconds) or row.startSeconds < 0
               for row in result.rows):
            raise ValueError("draft rows must have finite nonnegative timestamps and declared lanes")
        if len({row.id for row in result.rows}) != len(result.rows):
            raise ValueError("draft row IDs must be unique")
        return result
    result = TranscriptionResponse.model_validate(value)
    if result.taskId != job.payload.taskId or [track.lane for track in result.tracks] != [
        track.lane for track in job.payload.tracks
    ]:
        raise ValueError("transcription taskId and track lanes must match the request")
    return result


async def _backend_request(coordinator: Coordinator, job: Job, deadline: float) -> Response:
    last_failure: httpx.Response | None = None
    for base in coordinator.settings.backend_urls:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        timeout = min(coordinator.settings.backend_timeout_seconds, remaining)
        job.backend_url = base
        # ExitStack keeps both input streams open until httpx finishes streaming them.
        try:
            with ExitStack() as streams:
                files = {
                    track.fieldName: (
                        job.filenames[track.fieldName],
                        streams.enter_context(job.paths[track.fieldName].open("rb")),
                        job.types[track.fieldName],
                    )
                    for track in job.payload.tracks
                }
                upstream = await asyncio.wait_for(
                    coordinator.client.post(
                        f"{base}/v1/{job.operation}",
                        data={"payload": job.raw_payload},
                        files=files,
                        headers={"X-Babel-Local-Engine": "1", "X-Babel-Request-Id": job.request_id},
                        timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
                    ),
                    timeout=remaining,
                )
        except (httpx.TransportError, asyncio.TimeoutError):
            continue
        if upstream.status_code in (502, 503, 504):
            last_failure = upstream
            continue
        headers = {}
        if "retry-after" in upstream.headers:
            headers["Retry-After"] = upstream.headers["retry-after"]
        return Response(upstream.content, status_code=upstream.status_code,
                        headers=headers, media_type=upstream.headers.get("content-type"))
    if last_failure is not None:
        return Response(last_failure.content, status_code=last_failure.status_code,
                        media_type=last_failure.headers.get("content-type"))
    raise HTTPException(503, "trusted inference backends are unavailable")


def create_app(settings: CoordinatorSettings | None = None, client: httpx.AsyncClient | None = None) -> FastAPI:
    resolved = settings or CoordinatorSettings.from_env()
    owned_client = client is None
    transport = client or httpx.AsyncClient(trust_env=False)
    state = Coordinator(resolved, transport)

    @asynccontextmanager
    async def lifespan(_service: FastAPI):
        try:
            yield
        finally:
            if owned_client:
                await transport.aclose()
    service = FastAPI(title="Babel L0 Inference Coordinator", version="1.0.0", lifespan=lifespan)
    service.add_middleware(BodyLimitMiddleware, max_bytes=resolved.max_request_bytes)
    service.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOWED_ORIGIN_RE,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Babel-Local-Engine", "X-Babel-Request-Id"],
        expose_headers=["Retry-After"],
        max_age=600,
    )
    service.state.coordinator = state


    @service.get("/health")
    async def health() -> dict[str, object]:
        state.prune()
        swarm = {"workers": len(state.workers), "jobs": len(state.jobs),
                 "queued": len(state.waiting)}
        for base in resolved.backend_urls:
            try:
                upstream = await transport.get(f"{base}/health", timeout=2.0)
                if upstream.is_success:
                    report = upstream.json()
                    if isinstance(report, dict) and report.get("ok") is True:
                        return {**report, "swarm": swarm}
            except (httpx.TransportError, ValueError):
                continue
        return {"ok": state.idle_capacity(), "service": "coordinator",
                "swarm": swarm, "backendsAvailable": False}

    @service.get("/v1/queue/{request_id}")
    async def queue_status(request_id: str) -> dict[str, str | int]:
        status = state.status(request_id)
        job = state.by_request_id.get(request_id)
        if job is not None and job.backend_url is not None:
            try:
                upstream = await transport.get(
                    f"{job.backend_url}/v1/queue/{quote(request_id, safe='')}", timeout=5.0,
                    headers={"X-Babel-Local-Engine": "1"},
                )
                if upstream.is_success:
                    return upstream.json()
            except (httpx.TransportError, ValueError):
                pass
        if status is None:
            for base in resolved.backend_urls:
                try:
                    upstream = await transport.get(
                        f"{base}/v1/queue/{quote(request_id, safe='')}", timeout=2.0,
                        headers={"X-Babel-Local-Engine": "1"},
                    )
                    if upstream.is_success:
                        return upstream.json()
                except (httpx.TransportError, ValueError):
                    continue
        if status is None:
            raise HTTPException(404, "request ID not found")
        return status

    @service.post("/v1/workers/register")
    async def register(body: RegisterBody) -> dict[str, str]:
        state.prune()
        worker_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        state.workers[worker_id] = Worker(token, time.monotonic())
        return {"workerId": worker_id, "token": token}

    @service.post("/v1/workers/lease", response_model=None)
    async def lease(body: WorkerBody) -> Response | dict[str, object]:
        worker = state.authenticate(body.workerId, body.token)
        worker.last_seen = time.monotonic()
        if worker.job_id is not None:
            return Response(status_code=204)
        while state.waiting:
            job = state.jobs.get(state.waiting.popleft())
            if job is None or job.result.done():
                continue
            worker.job_id = job.job_id
            job.worker_id = body.workerId
            job.lease_token = secrets.token_urlsafe(32)
            job.lease_deadline = time.monotonic() + resolved.lease_seconds
            job.phase = "running"
            job.leased.set()
            return {
                "jobId": job.job_id, "leaseToken": job.lease_token,
                "operation": job.operation,
                "payload": job.payload.model_dump(exclude_none=True),
                "audio": [
                    {"fieldName": track.fieldName,
                     "url": f"/v1/jobs/{job.job_id}/audio/{quote(track.fieldName, safe='')}"}
                    for track in job.payload.tracks
                ],
            }
        return Response(status_code=204)

    def authorized_job(job_id: str, lease_token: str) -> Job:
        job = state.jobs.get(job_id)
        if job is None or job.worker_id is None or job.lease_token is None:
            raise HTTPException(404, "job lease not found")
        if not secrets.compare_digest(job.lease_token, lease_token):
            raise HTTPException(403, "invalid lease token")
        if job.lease_deadline is None or time.monotonic() >= job.lease_deadline:
            state.fail(job)
            raise HTTPException(404, "job lease expired")
        return job

    @service.get("/v1/jobs/{job_id}/audio/{field_name}")
    async def job_audio(job_id: str, field_name: str, request: Request) -> FileResponse:
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "lease bearer token is required")
        job = authorized_job(job_id, authorization[7:])
        path = job.paths.get(field_name)
        if path is None:
            raise HTTPException(404, "audio field not found")
        return FileResponse(path, media_type="audio/wav", filename=job.filenames[field_name])

    @service.post("/v1/jobs/{job_id}/complete")
    async def complete(job_id: str, body: CompleteBody) -> dict[str, bool]:
        state.authenticate(body.workerId, body.token)
        job = authorized_job(job_id, body.leaseToken)
        if job.worker_id != body.workerId:
            raise HTTPException(403, "lease belongs to another worker")
        if body.error is not None:
            state.fail(job)
            return {"ok": True}
        try:
            result = _validated_result(job, body.result)
        except (ValidationError, ValueError) as exc:
            state.fail(job)
            raise HTTPException(422, "worker result does not match the request") from exc
        state.release(job)
        if not job.result.done():
            job.result.set_result(result)
        return {"ok": True}

    async def infer(request: Request, operation: Literal["draft", "transcribe"]) -> Response | DraftResponse | TranscriptionResponse:
        if request.headers.get("x-babel-local-engine") != "1":
            raise HTTPException(403, "local proxy header is required")
        request_id = _request_id(request)
        _check_length(request, resolved.max_request_bytes)
        if state.inflight >= resolved.max_inflight_requests:
            raise HTTPException(429, "too many in-flight requests", headers={"Retry-After": "5"})
        state.inflight += 1
        deadline = time.monotonic() + resolved.request_timeout_seconds
        uploads: list[UploadFile] = []
        try:
            try:
                form = await request.form()
            except BodyTooLarge:
                raise
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, "invalid multipart request") from exc
            uploads = [value for _, value in form.multi_items() if isinstance(value, UploadFile)]
            payload, raw_payload, files = _parse_form(form)
            with tempfile.TemporaryDirectory(prefix="babel-coordinator-") as temporary:
                paths: dict[str, Path] = {}
                filenames: dict[str, str] = {}
                types: dict[str, str] = {}
                for index, track in enumerate(payload.tracks):
                    upload = files[track.fieldName]
                    path = Path(temporary) / f"track-{index}.wav"
                    await _copy_audio(upload, path, resolved.max_track_bytes, resolved.max_audio_seconds)
                    await upload.close()
                    uploads.remove(upload)
                    paths[track.fieldName] = path
                    filenames[track.fieldName] = upload.filename or f"track-{index}.wav"
                    types[track.fieldName] = upload.content_type or "audio/wav"
                job = Job(request_id, operation, payload, raw_payload, paths, filenames, types,
                          asyncio.get_running_loop().create_future())
                state.enqueue(job)
                responded = False
                try:
                    if job.phase == "queued":
                        try:
                            await asyncio.wait_for(job.leased.wait(), timeout=min(
                                resolved.queue_seconds, max(0, deadline - time.monotonic())
                            ))
                            remaining = min(
                                max(0, (job.lease_deadline or 0) - time.monotonic()),
                                max(0, deadline - time.monotonic()),
                            )
                            result = await asyncio.wait_for(job.result, timeout=remaining)
                        except asyncio.TimeoutError:
                            result = None
                        if result is not None:
                            responded = True
                            return result
                    state.release(job)
                    try:
                        state.waiting.remove(job.job_id)
                    except ValueError:
                        pass
                    job.phase = "running"
                    response = await _backend_request(state, job, deadline)
                    responded = True
                    return response
                finally:
                    state.finish(job, completed=responded)
        finally:
            try:
                await asyncio.gather(*(upload.close() for upload in uploads), return_exceptions=True)
            finally:
                state.inflight -= 1

    @service.post("/v1/draft", response_model=DraftResponse)
    async def draft(request: Request) -> Response | DraftResponse | TranscriptionResponse:
        return await infer(request, "draft")

    @service.post("/v1/transcribe", response_model=TranscriptionResponse)
    async def transcribe(request: Request) -> Response | DraftResponse | TranscriptionResponse:
        return await infer(request, "transcribe")

    return service


app = create_app()
