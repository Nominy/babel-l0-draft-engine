"""Lightweight public coordinator for browser volunteers and trusted engine failover.

Run with ``uvicorn l0_draft_engine.coordinator:app`` after installing
``requirements-coordinator.txt``. Keep one uvicorn process: leases and status
are deliberately in-memory and cannot be shared between processes.
Enhancement uses this same worker registry, but never trusted ASR fallback or the
timing cache. Its owner capability protects status and its temporary WAV pair is
retained only until the multipart response completes or the owner disconnects.
"""

from __future__ import annotations
from .release_middleware import InferenceReleaseMiddleware

import asyncio
from contextlib import ExitStack, asynccontextmanager
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.datastructures import UploadFile
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .schemas import DraftOptions, DraftPayload, DraftResponse, TranscriptionResponse
from .enhancement import (
    EnhancementModel, EnhancementPayload, EnhancementProgress, EnhancementResponse,
    EnhancementResult, copy_audio as copy_enhancement_audio,
    parse_multipart as parse_enhancement_multipart, read_multipart as read_enhancement_multipart,
)
from .inference_release import RELEASE_ID, RELEASE, RELEASE_HEADERS, RELEASE_HEADER, validate_punctuated_timing, upgrade_detail


REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
ALLOWED_ORIGIN_RE = (
    r"^(?:chrome-extension://[a-p]{32}|https://dashboard\.babel\.audio|"
    r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?)$"
)
CHUNK_BYTES = 1024 * 1024
WORKER_BODY_BYTES = 16 * 1024 * 1024
DEFAULT_LEASE_BYTES = 64 * 1024
ACCESS_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43,128}$")


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
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "babel" / "timing")
    cache_max_bytes: int = 4 * 1024**3
    require_current_release: bool = False

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
        if not 0 < self.cache_max_bytes <= 4 * 1024**3:
            raise ValueError("cache_max_bytes must be between 1 byte and 4 GiB")

    @classmethod
    def from_env(cls) -> CoordinatorSettings:
        urls = os.environ.get("COORDINATOR_BACKEND_URLS")
        defaults = cls()
        return cls(
            require_current_release=os.environ.get("COORDINATOR_REQUIRE_CURRENT_RELEASE", "0") == "1",
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
            cache_dir=Path(os.environ.get("COORDINATOR_CACHE_DIR", defaults.cache_dir)),
            cache_max_bytes=_positive_env("COORDINATOR_CACHE_MAX_BYTES", defaults.cache_max_bytes),
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
        multipart_completion = (
            scope["path"].startswith("/v1/jobs/") and scope["path"].endswith("/complete")
            and dict(scope.get("headers", [])).get(b"content-type", b"").lower().startswith(b"multipart/form-data")
        )
        limit = (
            WORKER_BODY_BYTES
            if scope["path"].startswith(("/v1/workers/", "/v1/jobs/")) and not multipart_completion
            else self.max_bytes
        )
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


def _access_token(request: Request, *, required: bool = True) -> str | None:
    authorization = request.headers.get("authorization", "")
    token = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not ACCESS_TOKEN_RE.fullmatch(token):
        if not required:
            return None
        raise HTTPException(401, "timing bearer token is required")
    return token


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


async def _copy_audio(upload: UploadFile, path: Path, limit: int, max_seconds: float) -> str:
    size = 0
    digest = hashlib.sha256()
    with path.open("xb") as output:
        while chunk := await upload.read(CHUNK_BYTES):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "audio track exceeds size limit")
            digest.update(chunk)
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
    return digest.hexdigest()


class TaskBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    taskId: str = Field(min_length=1, max_length=256)
    options: DraftOptions | None = None

    @field_validator("taskId")
    @classmethod
    def valid_task_id(cls, value: str) -> str:
        return DraftPayload.valid_task_id(value)


class RegisterBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modelBundleSchema: Literal["babel-browser-model-bundle-v3"]
    protocolVersion: Literal[3]
    modelRelease: str
    maxLeaseBytes: int = Field(default=DEFAULT_LEASE_BYTES, ge=DEFAULT_LEASE_BYTES,
                               le=WORKER_BODY_BYTES, strict=True)
    operations: list[Literal["transcribe", "draft", "enhance"]] = Field(
        default_factory=lambda: ["transcribe", "draft"], min_length=1, max_length=3,
    )
    enhancementModel: EnhancementModel | None = None

    @model_validator(mode="after")
    def consistent_capabilities(self) -> RegisterBody:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("worker operations must be distinct")
        if ("enhance" in self.operations) != (self.enhancementModel is not None):
            raise ValueError("enhancement operation requires exactly one model capability")
        return self

    @field_validator("modelRelease")
    @classmethod
    def current_model_release(cls, value: str) -> str:
        if value != RELEASE_ID:
            raise ValueError("worker model update is required")
        return value


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



class EnhancementCompleteBody(WorkerBody):
    leaseToken: str
    result: EnhancementResult


class ProgressBody(WorkerBody):
    leaseToken: str
    progress: EnhancementProgress

def _timing_key(task_id: str, token: str) -> tuple[str, str]:
    return task_id, hashlib.sha256(token.encode()).hexdigest()


def _timing_input_digest(payload: DraftPayload, audio_digests: tuple[str, ...]) -> str:
    identity = {
        "tracks": [(track.lane, digest) for track, digest in zip(payload.tracks, audio_digests, strict=True)],
        "preprocessing": payload.options.preprocessing if payload.options is not None else None,
    }
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass
class TimingFlight:
    input_digest: str
    result: asyncio.Future[TranscriptionResponse | None]


class TimingCache:
    """Private, on-disk LRU keyed by task ID and bearer capability digest."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The task-only format cannot prove lanes or preprocessing; never migrate it.
        for path in directory.glob("*.json"):
            if re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                path.unlink(missing_ok=True)
        self._rotate()

    def _path(self, key: tuple[str, str]) -> Path:
        digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
        return self.directory / f"v2-{digest}.json"

    def get(self, task_id: str, token: str, input_digest: str | None = None) -> TranscriptionResponse | None:
        key = _timing_key(task_id, token)
        path = self._path(key)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        try:
            timing = TranscriptionResponse.model_validate(record["timing"])
            if (
                record["version"] != 3 or record.get("releaseId") != RELEASE_ID
                or record["taskId"] != task_id or timing.taskId != task_id
                or not isinstance(record["tokenDigest"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["tokenDigest"])
                or not isinstance(record["inputDigest"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["inputDigest"])
            ):
                raise ValueError("invalid timing cache record")
        except (ValueError, KeyError, TypeError, ValidationError):
            path.unlink(missing_ok=True)
            return None
        if not secrets.compare_digest(record["tokenDigest"], key[1]):
            return None
        if input_digest is not None and record["inputDigest"] != input_digest:
            return None
        try:
            os.utime(path)
        except FileNotFoundError:
            return None
        return timing.model_copy(update={"accessToken": token})

    def put(self, timing: TranscriptionResponse, token: str, input_digest: str) -> None:
        key = _timing_key(timing.taskId, token)
        data = json.dumps({
            "version": 3,
            "releaseId": RELEASE_ID,
            "taskId": timing.taskId,
            "tokenDigest": key[1],
            "inputDigest": input_digest,
            "timing": timing.model_dump(exclude={"accessToken"}),
        }, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(data) > self.max_bytes:
            raise ValueError("timing response exceeds cache limit")
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".timing-", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary_path, self._path(key))
            self._rotate()
        finally:
            temporary_path.unlink(missing_ok=True)

    def exists(self, task_id: str, token: str) -> bool:
        return self._path(_timing_key(task_id, token)).exists()

    def _rotate(self) -> None:
        entries = sorted(
            ((path, path.stat()) for path in self.directory.glob("*.json") if path.is_file()),
            key=lambda entry: entry[1].st_mtime_ns,
        )
        total = sum(stat.st_size for _, stat in entries)
        for path, stat in entries:
            if total <= self.max_bytes:
                break
            path.unlink(missing_ok=True)
            total -= stat.st_size


@dataclass
class Worker:
    token: str
    last_seen: float
    job_id: str | None = None
    operations: tuple[str, ...] = ("transcribe", "draft")
    enhancement_model: EnhancementModel | None = None
    max_lease_bytes: int = DEFAULT_LEASE_BYTES

    def accepts(self, job: Job) -> bool:
        return job.operation in self.operations and (
            job.operation != "enhance" or self.enhancement_model == job.payload.model
        ) and job.lease_response_bytes <= self.max_lease_bytes


@dataclass
class Job:
    request_id: str
    operation: Literal["draft", "transcribe", "enhance"]
    payload: DraftPayload | EnhancementPayload
    raw_payload: str
    paths: dict[str, Path]
    filenames: dict[str, str]
    types: dict[str, str]
    result: asyncio.Future[DraftResponse | TranscriptionResponse | EnhancementResult | None]
    timing: TranscriptionResponse | None = None
    options: DraftOptions | None = None
    leased: asyncio.Event = field(default_factory=asyncio.Event)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str | None = None
    lease_token: str | None = None
    lease_deadline: float | None = None
    backend_url: str | None = None
    phase: Literal["queued", "running"] = "queued"
    owner_digest: str | None = None
    progress: EnhancementProgress | None = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    io_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    io_task: asyncio.Task | None = None

    def lease_payload(self, lease_token: str) -> dict[str, object]:
        return {
            "jobId": self.job_id, "leaseToken": lease_token, "operation": self.operation,
            "payload": (
                {"taskId": self.payload.taskId,
                 "timing": self.timing.model_dump(exclude={"accessToken"}),
                 **({"options": self.options.model_dump(exclude_none=True)} if self.options else {})}
                if self.operation == "draft" and self.timing is not None
                else self.payload.model_dump(exclude_none=True)
            ),
            "audio": [
                {"fieldName": track.fieldName,
                 "url": f"/v1/jobs/{self.job_id}/audio/{quote(track.fieldName, safe='')}"}
                for track in self.payload.tracks
            ] if self.operation in ("transcribe", "enhance") else [],
        }

    @cached_property
    def lease_response_bytes(self) -> int:
        # token_urlsafe(32) is 43 ASCII bytes. Use the actual UTF-8 JSON encoding,
        # including options/envelope, before assigning a worker that must read it.
        return len(JSONResponse(self.lease_payload("")).body) + 43


class Coordinator:
    def __init__(self, settings: CoordinatorSettings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self.cache = TimingCache(settings.cache_dir, settings.cache_max_bytes)
        self.timing_flights: dict[tuple[str, str], TimingFlight] = {}
        self.workers: dict[str, Worker] = {}
        self.jobs: dict[str, Job] = {}
        self.by_request_id: dict[str, Job] = {}
        self.waiting: deque[str] = deque()
        self.completed: OrderedDict[str, tuple[float, str | None]] = OrderedDict()
        self.inflight = 0

    def prune(self) -> None:
        now = time.monotonic()
        for worker_id, worker in tuple(self.workers.items()):
            if worker.job_id is None and now - worker.last_seen > self.settings.worker_idle_seconds:
                del self.workers[worker_id]
        while self.completed:
            _, (expiry, _) = next(iter(self.completed.items()))
            if expiry > now and len(self.completed) <= 1024:
                break
            self.completed.popitem(last=False)

    def idle_capacity(self, job: Job | None = None) -> bool:
        self.prune()
        idle = sum(
            worker.job_id is None and (worker.accepts(job) if job else
                                      bool(set(worker.operations) & {"transcribe", "draft"}))
            for worker in self.workers.values()
        )
        waiting = sum(
            queued_id in self.jobs and self.jobs[queued_id].operation != "enhance"
            for queued_id in self.waiting
        )
        return idle > waiting

    def enqueue(self, job: Job) -> None:
        self.prune()
        if job.request_id in self.by_request_id or job.request_id in self.completed:
            raise HTTPException(409, "request ID is already registered")
        self.jobs[job.job_id] = job
        self.by_request_id[job.request_id] = job
        if job.operation == "enhance" or self.idle_capacity(job):
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
        job.leased.set()  # Wake queued enhancement on cancellation/failure too.

    def finish(self, job: Job, *, completed: bool) -> None:
        self.release(job)
        self.jobs.pop(job.job_id, None)
        self.by_request_id.pop(job.request_id, None)
        try:
            self.waiting.remove(job.job_id)
        except ValueError:
            pass
        if completed:
            self.completed[job.request_id] = (time.monotonic() + 45, job.owner_digest)
        self.prune()

    def authenticate(self, worker_id: str, token: str) -> Worker:
        worker = self.workers.get(worker_id)
        if worker is None or not secrets.compare_digest(worker.token, token):
            raise HTTPException(401, "worker credentials are invalid")
        return worker

    def status(self, request_id: str, token: str | None = None) -> dict[str, object] | None:
        self.prune()
        job = self.by_request_id.get(request_id)
        owner_digest = job.owner_digest if job else self.completed.get(request_id, (0, None))[1]
        if owner_digest is not None and (
            token is None or not secrets.compare_digest(owner_digest, hashlib.sha256(token.encode()).hexdigest())
        ):
            raise HTTPException(404, "request ID not found")
        if job is not None:
            position = 0
            if job.phase == "queued":
                try:
                    position = self.waiting.index(job.job_id) + 1
                except ValueError:
                    pass
            return {"requestId": request_id, "status": job.phase,
                    "position": position, "queuedCount": len(self.waiting),
                    **({"progress": job.progress.model_dump()} if job.progress is not None else {})}
        if request_id in self.completed:
            return {"requestId": request_id, "status": "completed", "position": 0,
                    "queuedCount": len(self.waiting)}
        return None


def _validated_result(job: Job, value: dict[str, object], require_current_release: bool = False) -> DraftResponse | TranscriptionResponse:
    json.dumps(value, allow_nan=False)
    if job.operation == "draft":
        result = DraftResponse.model_validate(value)
        if require_current_release and result.models.get("release") != RELEASE_ID:
            raise ValueError("draft result belongs to an outdated release")
        lanes = {track.lane for track in job.payload.tracks}
        if any(row.lane not in lanes or not math.isfinite(row.startSeconds)
               or not math.isfinite(row.endSeconds) or row.startSeconds < 0
               for row in result.rows):
            raise ValueError("draft rows must have finite nonnegative timestamps and declared lanes")
        if len({row.id for row in result.rows}) != len(result.rows):
            raise ValueError("draft row IDs must be unique")
        return result
    result = TranscriptionResponse.model_validate(value)
    if require_current_release:
        validate_punctuated_timing(result)
    if result.taskId != job.payload.taskId or [track.lane for track in result.tracks] != [
        track.lane for track in job.payload.tracks
    ]:
        raise ValueError("transcription taskId and track lanes must match the request")
    if any(
        track.sampleRate != segment.sampleRate
        for track in result.tracks for segment in track.segments
    ):
        raise ValueError("segment sample rate must match its track")
    segment_ids = [segment.id for track in result.tracks for segment in track.segments]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("segment IDs must be unique")
    return result


async def _backend_request(coordinator: Coordinator, job: Job, deadline: float) -> Response:
    if job.operation == "enhance":
        raise HTTPException(503, "enhancement requires a matching WebGPU volunteer; Originals are unchanged")
    last_failure: httpx.Response | None = None
    for base in coordinator.settings.backend_urls:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        timeout = min(coordinator.settings.backend_timeout_seconds, remaining)
        job.backend_url = base
        try:
            if job.operation == "draft":
                if job.timing is None:
                    raise RuntimeError("draft job is missing timing")
                upstream = await asyncio.wait_for(
                    coordinator.client.post(
                        f"{base}/v1/draft",
                        json={
                            "timing": job.timing.model_dump(exclude={"accessToken"}),
                            **({"options": job.options.model_dump(exclude_none=True)} if job.options else {}),
                        },
                        headers={**RELEASE_HEADERS, "X-Babel-Local-Engine": "1", "X-Babel-Request-Id": job.request_id},
                        timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
                    ),
                    timeout=remaining,
                )
            else:
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
                            f"{base}/v1/transcribe",
                            data={"payload": job.raw_payload},
                            files=files,
                            headers={**RELEASE_HEADERS, "X-Babel-Local-Engine": "1", "X-Babel-Request-Id": job.request_id},
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
    service.add_middleware(InferenceReleaseMiddleware, enforced=resolved.require_current_release)
    service.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOWED_ORIGIN_RE,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Babel-Local-Engine", "X-Babel-Request-Id", RELEASE_HEADER],
        expose_headers=["Retry-After"],
        max_age=600,
    )
    service.state.coordinator = state

    @service.get("/v1/inference-release")
    async def inference_release():
        return {**RELEASE, "enforced": resolved.require_current_release}


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
                        if resolved.require_current_release and report.get("release") != RELEASE_ID:
                            continue
                        return {**report, "swarm": swarm}
            except (httpx.TransportError, ValueError):
                continue
        return {"ok": state.idle_capacity(), "service": "coordinator",
                "swarm": swarm, "backendsAvailable": False}

    @service.get("/v1/queue/{request_id}")
    async def queue_status(request_id: str, request: Request) -> dict[str, object]:
        token = _access_token(request, required=False)
        status = state.status(request_id, token)
        if status is None and request.headers.get("authorization") is not None:
            # Private enhancement polling must not leak an unknown owner request to ASR backends.
            raise HTTPException(404, "request ID not found")
        job = state.by_request_id.get(request_id)
        if job is not None and job.backend_url is not None:
            try:
                upstream = await transport.get(
                    f"{job.backend_url}/v1/queue/{quote(request_id, safe='')}", timeout=5.0,
                    headers={**RELEASE_HEADERS, "X-Babel-Local-Engine": "1"},
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
                        headers={**RELEASE_HEADERS, "X-Babel-Local-Engine": "1"},
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
        state.workers[worker_id] = Worker(
            token, time.monotonic(), operations=tuple(body.operations),
            enhancement_model=body.enhancementModel,
            max_lease_bytes=body.maxLeaseBytes,
        )
        return {"workerId": worker_id, "token": token}

    @service.post("/v1/workers/lease", response_model=None)
    async def lease(body: WorkerBody) -> Response | dict[str, object]:
        worker = state.authenticate(body.workerId, body.token)
        worker.last_seen = time.monotonic()
        if worker.job_id is not None:
            return Response(status_code=204)
        for job_id in tuple(state.waiting):
            job = state.jobs.get(job_id)
            if job is None or job.result.done():
                state.waiting.remove(job_id)
                continue
            if not worker.accepts(job):
                continue
            state.waiting.remove(job_id)
            worker.job_id = job.job_id
            job.worker_id = body.workerId
            job.lease_token = secrets.token_urlsafe(32)
            job.lease_deadline = time.monotonic() + resolved.lease_seconds
            job.phase = "running"
            job.leased.set()
            return JSONResponse(job.lease_payload(job.lease_token))
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

    class EnhancementAudioResponse(FileResponse):
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            # Cleanup must not remove an input while a worker download holds it open.
            async with self.job.io_lock:
                authorized_job(self.job.job_id, self.lease_token)
                self.job.io_task = asyncio.current_task()
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    self.job.io_task = None

    @service.get("/v1/jobs/{job_id}/audio/{field_name}")
    async def job_audio(job_id: str, field_name: str, request: Request) -> FileResponse:
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "lease bearer token is required")
        job = authorized_job(job_id, authorization[7:])
        path = job.paths.get(field_name)
        if path is None:
            raise HTTPException(404, "audio field not found")
        if job.operation == "enhance":
            response = EnhancementAudioResponse(
                path, media_type="audio/wav", filename=job.filenames[field_name],
                headers={"Cache-Control": "no-store"},
            )
            response.job = job
            response.lease_token = authorization[7:]
            return response
        return FileResponse(path, media_type="audio/wav", filename=job.filenames[field_name])

    def worker_job(job_id: str, body: WorkerBody, lease_token: str) -> Job:
        state.authenticate(body.workerId, body.token)
        job = authorized_job(job_id, lease_token)
        if job.worker_id != body.workerId:
            raise HTTPException(403, "lease belongs to another worker")
        return job

    @service.post("/v1/jobs/{job_id}/progress")
    async def progress(job_id: str, body: ProgressBody) -> dict[str, bool]:
        job = worker_job(job_id, body, body.leaseToken)
        if job.operation != "enhance":
            raise HTTPException(422, "only enhancement jobs publish this progress")
        try:
            body.progress.validate_update(job.payload, job.progress)
        except ValueError as exc:
            raise HTTPException(422, "invalid or regressing enhancement progress") from exc
        job.progress = body.progress
        return {"ok": True}

    @service.post("/v1/jobs/{job_id}/complete")
    async def complete(job_id: str, request: Request) -> dict[str, bool]:
        if request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            authorization = request.headers.get("authorization", "")
            if not authorization.startswith("Bearer "):
                raise HTTPException(401, "lease bearer token is required")
            job = authorized_job(job_id, authorization[7:])
            if job.operation != "enhance":
                raise HTTPException(422, "multipart completion is only for enhancement")
            _check_length(request, resolved.max_request_bytes)
            # Serialize result writers with downloads and cancellation cleanup.
            async with job.io_lock:
                authorized_job(job_id, authorization[7:])
                job.io_task = asyncio.current_task()
                try:
                    form = await read_enhancement_multipart(request)
                    try:
                        body, files = parse_enhancement_multipart(form, EnhancementCompleteBody)
                        authenticated = worker_job(job_id, body, body.leaseToken)
                        if authenticated is not job:
                            raise HTTPException(404, "job lease not found")
                        body.result.validate_request(job.payload)
                        for index, (track, metadata) in enumerate(zip(
                            job.payload.tracks, body.result.tracks, strict=True
                        )):
                            path = job.paths[track.fieldName].parent / f"enhanced-{index}.wav"
                            await copy_enhancement_audio(
                                files[track.fieldName], path, metadata,
                                max_bytes=resolved.max_track_bytes,
                                max_seconds=resolved.max_audio_seconds, output=True,
                            )
                            job.output_paths[track.fieldName] = path
                        # The owner may have disconnected or the lease expired during upload.
                        worker_job(job_id, body, body.leaseToken)
                        state.release(job)
                        job.result.set_result(body.result)
                    finally:
                        await form.close()
                except HTTPException as exc:
                    if exc.status_code not in (401, 403, 404):
                        state.fail(job)
                    raise
                except (ValueError, ValidationError) as exc:
                    state.fail(job)
                    raise HTTPException(422, "worker result does not match the request") from exc
                except BaseException:
                    state.fail(job)
                    raise
                finally:
                    job.io_task = None
            return {"ok": True}
        _check_length(request, WORKER_BODY_BYTES)
        try:
            body = CompleteBody.model_validate(await request.json())
        except (ValueError, ValidationError) as exc:
            raise HTTPException(422, "invalid worker completion") from exc
        job = worker_job(job_id, body, body.leaseToken)
        if body.error is not None:
            state.fail(job)
            return {"ok": True}
        if job.operation == "enhance":
            state.fail(job)
            raise HTTPException(422, "enhancement requires multipart WAV completion")
        try:
            result = _validated_result(job, body.result, resolved.require_current_release)
        except (ValidationError, ValueError) as exc:
            state.fail(job)
            raise HTTPException(422, "worker result does not match the request") from exc
        state.release(job)
        if not job.result.done():
            job.result.set_result(result)
        return {"ok": True}

    async def wait_for_enhancement(job: Job, deadline: float) -> EnhancementResult:
        try:
            await asyncio.wait_for(job.leased.wait(), timeout=min(
                resolved.queue_seconds, max(0, deadline - time.monotonic()),
            ))
            remaining = min(
                max(0, (job.lease_deadline or 0) - time.monotonic()),
                max(0, deadline - time.monotonic()),
            )
            result = await asyncio.wait_for(asyncio.shield(job.result), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                503, "No matching WebGPU volunteer completed enhancement in time. Originals are unchanged; try again later.",
            ) from exc
        if not isinstance(result, EnhancementResult):
            raise HTTPException(503, "The WebGPU volunteer could not enhance this pair. Originals are unchanged.")
        return result

    async def owner_disconnect(request: Request) -> None:
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    @service.post("/v1/enhance")
    async def enhance(request: Request) -> Response:
        if request.headers.get("x-babel-local-engine") != "1":
            raise HTTPException(403, "local proxy header is required")
        token = _access_token(request)
        if len(token) != 43:
            raise HTTPException(401, "enhancement requires a 32-byte owner bearer capability")
        request_id = request.headers.get("x-babel-request-id", "")
        try:
            if str(uuid.UUID(request_id)) != request_id:
                raise ValueError("noncanonical UUID")
        except ValueError as exc:
            raise HTTPException(400, "enhancement requires a UUID X-Babel-Request-Id") from exc
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            raise HTTPException(415, "enhancement requires multipart originals")
        _check_length(request, resolved.max_request_bytes)
        if state.inflight >= resolved.max_inflight_requests:
            raise HTTPException(429, "too many in-flight requests", headers={"Retry-After": "5"})
        state.inflight += 1
        temporary = None
        job = None
        cleaned = False
        response_owns_cleanup = False

        async def cleanup(completed: bool = False) -> None:
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            try:
                if job is not None and state.jobs.get(job.job_id) is job:
                    state.fail(job)
                    state.finish(job, completed=completed)
                if job is not None and job.io_task is not None:
                    # A stalled volunteer upload/download must not hold cancelled
                    # owner admission or temporary audio indefinitely.
                    job.io_task.cancel()
                if temporary is not None:
                    if job is None:
                        temporary.cleanup()
                    else:
                        async with job.io_lock:
                            temporary.cleanup()
            finally:
                state.inflight -= 1

        deadline = time.monotonic() + resolved.request_timeout_seconds
        async def prepare_response() -> Response:
            nonlocal temporary, job
            temporary = tempfile.TemporaryDirectory(prefix="babel-enhancement-")
            form = await read_enhancement_multipart(request)
            try:
                payload, files = parse_enhancement_multipart(form, EnhancementPayload)
                paths = {}
                for index, track in enumerate(payload.tracks):
                    path = Path(temporary.name) / f"original-{index}.wav"
                    await copy_enhancement_audio(
                        files[track.fieldName], path, track,
                        max_bytes=resolved.max_track_bytes, max_seconds=resolved.max_audio_seconds,
                    )
                    paths[track.fieldName] = path
                job = Job(
                    request_id, "enhance", payload, "", paths,
                    {track.fieldName: f"original-{index}.wav" for index, track in enumerate(payload.tracks)},
                    {track.fieldName: "audio/wav" for track in payload.tracks},
                    asyncio.get_running_loop().create_future(),
                    owner_digest=hashlib.sha256(token.encode()).hexdigest(),
                )
            finally:
                await form.close()
            state.enqueue(job)
            waiting = asyncio.create_task(wait_for_enhancement(job, deadline))
            disconnected = asyncio.create_task(owner_disconnect(request))
            try:
                done, _ = await asyncio.wait((waiting, disconnected), return_when=asyncio.FIRST_COMPLETED)
                if disconnected in done:
                    raise HTTPException(499, "enhancement owner disconnected")
                result = await waiting
            finally:
                waiting.cancel()
                disconnected.cancel()
                await asyncio.gather(waiting, disconnected, return_exceptions=True)
            return EnhancementResponse(payload, result, job.output_paths, cleanup)

        try:
            response = await asyncio.wait_for(prepare_response(), timeout=resolved.request_timeout_seconds)
            response_owns_cleanup = True
            return response
        except asyncio.TimeoutError as exc:
            raise HTTPException(503, "Enhancement request timed out. Originals are unchanged.") from exc
        finally:
            if not response_owns_cleanup:
                await cleanup()

    async def run_job(job: Job, deadline: float) -> Response | DraftResponse | TranscriptionResponse:
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
                    result = await asyncio.wait_for(asyncio.shield(job.result), timeout=remaining)
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
            if response.status_code != 200:
                responded = True
                return response
            try:
                result = _validated_result(job, json.loads(response.body), resolved.require_current_release)
            except (ValueError, ValidationError, TypeError) as exc:
                raise HTTPException(502, "trusted inference result is invalid") from exc
            responded = True
            return result
        finally:
            state.finish(job, completed=responded)

    async def cached_timing(
        task_id: str, token: str, *, deadline: float | None = None
    ) -> TranscriptionResponse | None:
        cached = state.cache.get(task_id, token)
        if cached is not None:
            if resolved.require_current_release:
                try:
                    validate_punctuated_timing(cached)
                except ValueError:
                    return None
            return cached
        flight = state.timing_flights.get(_timing_key(task_id, token))
        if flight is None:
            return None
        remaining = (
            resolved.request_timeout_seconds if deadline is None
            else max(0, deadline - time.monotonic())
        )
        try:
            result = await asyncio.wait_for(asyncio.shield(flight.result), timeout=remaining)
            return result.model_copy(update={"accessToken": token}) if result else None
        except asyncio.TimeoutError as exc:
            raise HTTPException(503, "timing request timed out") from exc

    @service.post("/v1/timing/lookup", response_model=TranscriptionResponse)
    async def lookup(request: Request, body: TaskBody) -> TranscriptionResponse:
        token = _access_token(request, required=False)
        timing = await cached_timing(body.taskId, token) if token else None
        if timing is None:
            raise HTTPException(404, "timing not found")
        return timing

    @service.post("/v1/draft", response_model=DraftResponse)
    async def draft(request: Request, body: TaskBody) -> Response | DraftResponse:
        token = _access_token(request, required=False)
        if token is None:
            raise HTTPException(404, "timing not found")
        request_id = _request_id(request)
        if state.inflight >= resolved.max_inflight_requests:
            raise HTTPException(429, "too many in-flight requests", headers={"Retry-After": "5"})
        state.inflight += 1
        deadline = time.monotonic() + resolved.request_timeout_seconds
        try:
            timing = await cached_timing(body.taskId, token, deadline=deadline)
            if timing is None:
                raise HTTPException(404, "timing not found")
            if time.monotonic() >= deadline:
                raise HTTPException(503, "draft request timed out")
            try:
                payload = DraftPayload(
                    taskId=timing.taskId,
                    tracks=[
                        {"lane": track.lane, "fieldName": f"audio:{index}"}
                        for index, track in enumerate(timing.tracks)
                    ],
                    options=body.options,
                )
            except ValidationError as exc:
                raise HTTPException(422, exc.errors(include_context=False, include_input=False)) from exc
            job = Job(request_id, "draft", payload, "", {}, {}, {},
                      asyncio.get_running_loop().create_future(),
                      timing=timing, options=body.options)
            return await run_job(job, deadline)
        finally:
            state.inflight -= 1

    @service.post("/v1/transcribe", response_model=TranscriptionResponse)
    async def transcribe(request: Request) -> Response | TranscriptionResponse:
        if request.headers.get("x-babel-local-engine") != "1":
            raise HTTPException(403, "local proxy header is required")
        token = _access_token(request)
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
                audio_digests: list[str] = []
                for index, track in enumerate(payload.tracks):
                    upload = files[track.fieldName]
                    path = Path(temporary) / f"track-{index}.wav"
                    audio_digests.append(await _copy_audio(
                        upload, path, resolved.max_track_bytes, resolved.max_audio_seconds
                    ))
                    await upload.close()
                    uploads.remove(upload)
                    paths[track.fieldName] = path
                    filenames[track.fieldName] = upload.filename or f"track-{index}.wav"
                    types[track.fieldName] = upload.content_type or "audio/wav"
                identity = _timing_input_digest(payload, tuple(audio_digests))
                cached = state.cache.get(payload.taskId, token, identity)
                if cached is not None:
                    return cached
                if state.cache.exists(payload.taskId, token):
                    raise HTTPException(409, "task ID already belongs to different transcription input")
                key = _timing_key(payload.taskId, token)
                flight = state.timing_flights.get(key)
                if flight is not None:
                    if flight.input_digest != identity:
                        raise HTTPException(409, "task ID already belongs to different transcription input")
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(flight.result), timeout=max(0, deadline - time.monotonic())
                        )
                    except asyncio.TimeoutError as exc:
                        raise HTTPException(503, "timing request timed out") from exc
                    if result is None:
                        raise HTTPException(503, "timing request failed")
                    return result.model_copy(update={"accessToken": token})
                flight = TimingFlight(
                    identity,
                    asyncio.get_running_loop().create_future(),
                )
                state.timing_flights[key] = flight
                try:
                    job = Job(request_id, "transcribe", payload, raw_payload,
                              paths, filenames, types, asyncio.get_running_loop().create_future())
                    response = await run_job(job, deadline)
                    if isinstance(response, Response):
                        return response
                    if not isinstance(response, TranscriptionResponse):
                        raise HTTPException(502, "transcription result has the wrong type")
                    try:
                        state.cache.put(response, token, identity)
                    except ValueError as exc:
                        raise HTTPException(503, "timing result exceeds cache capacity") from exc
                    if not flight.result.done():
                        flight.result.set_result(response)
                    return response.model_copy(update={"accessToken": token})
                finally:
                    if not flight.result.done():
                        flight.result.set_result(None)
                    state.timing_flights.pop(key, None)
        finally:
            try:
                await asyncio.gather(*(upload.close() for upload in uploads), return_exceptions=True)
            finally:
                state.inflight -= 1

    return service


app = create_app()
