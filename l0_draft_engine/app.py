from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import re
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings
from .engine import DraftEngine, DraftInputError, ModelUnavailableError
from .inference_queue import DuplicateRequestIdError, InferenceQueue
from .schemas import DraftPayload, DraftResponse, TranscriptionResponse


COPY_CHUNK_BYTES = 1024 * 1024
ALLOWED_ORIGIN_RE = r"^(?:https://dashboard\.babel\.audio|https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?)$"
REQUEST_ID_MAX_LENGTH = 128
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class _EventLoopAdmissionGate:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._inflight = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def try_acquire(self) -> bool:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._inflight:
                raise RuntimeError("admission gate used from multiple event loops")
            self._loop = loop
        if self._inflight >= self._capacity:
            return False
        self._inflight += 1
        return True

    def release(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("admission gate released from a different event loop")
        if self._inflight <= 0:
            raise RuntimeError("admission gate released without an acquired slot")
        self._inflight -= 1


def _request_id(request: Request) -> str:
    if "x-babel-request-id" not in request.headers:
        return str(uuid.uuid4())
    request_id = request.headers["x-babel-request-id"]
    if (
        not request_id
        or len(request_id) > REQUEST_ID_MAX_LENGTH
        or REQUEST_ID_RE.fullmatch(request_id) is None
    ):
        raise HTTPException(
            status_code=400,
            detail="X-Babel-Request-Id must be a nonempty safe identifier of at most 128 characters",
        )
    return request_id


async def _finish_cancelled_worker(task: asyncio.Task[Any]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    if task.done() and not task.cancelled():
        try:
            task.result()
        except BaseException:
            pass


async def _close_uploads(uploads: list[StarletteUploadFile]) -> None:
    await asyncio.gather(
        *(upload.close() for upload in uploads),
        return_exceptions=True,
    )


class RequestBodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            response = JSONResponse(
                {"detail": "request exceeds size limit"}, status_code=413
            )
            await response(scope, receive, send)


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
    if value < 0:
        raise HTTPException(status_code=400, detail="invalid Content-Length")
    return value


async def _copy_upload(upload: UploadFile, destination: Path, limit: int) -> int:
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(COPY_CHUNK_BYTES):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="audio track exceeds size limit")
                output.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="audio track is empty")
    return size


def _validate_mono_wav(path: Path, max_audio_seconds: float) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1:
                raise HTTPException(
                    status_code=422, detail="each audio track must be mono"
                )
            if audio.getcomptype() != "NONE":
                raise HTTPException(
                    status_code=422, detail="audio tracks must be uncompressed WAV"
                )
            frame_count = audio.getnframes()
            sample_rate = audio.getframerate()
            if frame_count <= 0 or sample_rate <= 0:
                raise HTTPException(
                    status_code=422, detail="audio tracks must have positive duration"
                )
            if frame_count / sample_rate > max_audio_seconds:
                raise HTTPException(status_code=413, detail="audio track exceeds duration limit")
    except HTTPException:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise HTTPException(
            status_code=422, detail="each audio track must be a valid WAV file"
        ) from exc


def _parse_form(form: Any) -> tuple[DraftPayload, dict[str, StarletteUploadFile]]:
    payload_values: list[str] = []
    files: dict[str, StarletteUploadFile] = {}
    for key, value in form.multi_items():
        if isinstance(value, StarletteUploadFile):
            if key in files:
                raise HTTPException(status_code=422, detail=f"duplicate audio field: {key}")
            files[key] = value
        elif key == "payload" and isinstance(value, str):
            payload_values.append(value)
        else:
            raise HTTPException(status_code=422, detail=f"unexpected multipart field: {key}")
    if len(payload_values) != 1:
        raise HTTPException(
            status_code=422, detail="multipart request must contain exactly one payload field"
        )
    try:
        payload = DraftPayload.model_validate_json(payload_values[0])
    except (ValidationError, json.JSONDecodeError) as exc:
        detail = (
            exc.errors(include_context=False)
            if isinstance(exc, ValidationError)
            else "payload must be valid JSON"
        )
        raise HTTPException(status_code=422, detail=detail) from exc
    expected_fields = {track.fieldName for track in payload.tracks}
    if len(files) != 2 or set(files) != expected_fields:
        raise HTTPException(
            status_code=422,
            detail="multipart request must contain exactly the two declared audio fields",
        )
    return payload, files


def create_app(
    settings: Settings | None = None, engine: DraftEngine | None = None
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_engine = engine or DraftEngine(resolved_settings)
    inference_queue = InferenceQueue()
    admission_gate = _EventLoopAdmissionGate(
        resolved_settings.max_inflight_requests
    )
    service = FastAPI(title="Babel Local Drafting Engine", version="1.0.0")
    service.add_middleware(
        RequestSizeLimitMiddleware, max_bytes=resolved_settings.max_request_bytes
    )
    service.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOWED_ORIGIN_RE,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Content-Type",
            "X-Babel-Local-Engine",
            "X-Babel-Request-Id",
        ],
        expose_headers=["Retry-After"],
        max_age=600,
    )
    service.state.settings = resolved_settings
    service.state.engine = resolved_engine
    service.state.inference_queue = inference_queue

    @service.get("/health")
    def health() -> dict[str, object]:
        return resolved_engine.health()

    @service.get("/v1/queue/{request_id}")
    async def queue_status(request_id: str) -> dict[str, str | int]:
        status = inference_queue.status(request_id)
        if status is None:
            raise HTTPException(status_code=404, detail="request ID not found")
        return status.as_dict()

    async def run_inference(
        request: Request,
        inference: Callable[
            [DraftPayload, dict[str, Path]],
            DraftResponse | TranscriptionResponse,
        ],
    ) -> DraftResponse | TranscriptionResponse:
        if request.headers.get("x-babel-local-engine") != "1":
            raise HTTPException(status_code=403, detail="local proxy header is required")
        request_id = _request_id(request)
        content_length = _content_length(request)
        if (
            content_length is not None
            and content_length > resolved_settings.max_request_bytes
        ):
            raise HTTPException(status_code=413, detail="request exceeds size limit")
        if not admission_gate.try_acquire():
            raise HTTPException(
                status_code=429,
                detail="too many in-flight requests",
                headers={"Retry-After": "5"},
            )

        uploads: list[StarletteUploadFile] = []
        try:
            try:
                form = await request.form()
            except RequestBodyTooLarge:
                raise
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail="invalid multipart request"
                ) from exc
            uploads = [
                value
                for _, value in form.multi_items()
                if isinstance(value, StarletteUploadFile)
            ]
            payload, files = _parse_form(form)
            with tempfile.TemporaryDirectory(prefix="babel-local-engine-") as temporary:
                directory = Path(temporary)
                paths: dict[str, Path] = {}
                total_file_bytes = 0
                for index, track in enumerate(payload.tracks):
                    destination = directory / f"track-{index}.wav"
                    upload = files[track.fieldName]
                    total_file_bytes += await _copy_upload(
                        upload,
                        destination,
                        resolved_settings.max_track_bytes,
                    )
                    await upload.close()
                    uploads.remove(upload)
                    if total_file_bytes > resolved_settings.max_request_bytes:
                        raise HTTPException(
                            status_code=413, detail="request exceeds size limit"
                        )
                    _validate_mono_wav(
                        destination, resolved_settings.max_audio_seconds
                    )
                    paths[track.lane] = destination

                try:
                    ticket = inference_queue.register(request_id)
                except DuplicateRequestIdError as exc:
                    raise HTTPException(
                        status_code=409, detail="request ID is already registered"
                    ) from exc
                try:
                    await ticket.ready.wait()
                except BaseException:
                    inference_queue.abandon(ticket)
                    raise

                try:
                    worker = asyncio.create_task(
                        run_in_threadpool(inference, payload, paths)
                    )
                except BaseException:
                    inference_queue.abandon(ticket)
                    raise
                try:
                    try:
                        return await asyncio.shield(worker)
                    except asyncio.CancelledError as exc:
                        await _finish_cancelled_worker(worker)
                        raise exc
                except DraftInputError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                except ModelUnavailableError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                finally:
                    if worker.done():
                        inference_queue.complete(ticket)
        finally:
            try:
                cleanup = asyncio.create_task(_close_uploads(uploads))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    await _finish_cancelled_worker(cleanup)
                    raise exc
            finally:
                admission_gate.release()

    @service.post("/v1/draft", response_model=DraftResponse)
    async def draft(request: Request) -> DraftResponse:
        response = await run_inference(request, resolved_engine.draft)
        if not isinstance(response, DraftResponse):
            raise RuntimeError("draft engine returned the wrong response type")
        return response

    @service.post("/v1/transcribe", response_model=TranscriptionResponse)
    async def transcribe(request: Request) -> TranscriptionResponse:
        response = await run_inference(request, resolved_engine.transcribe)
        if not isinstance(response, TranscriptionResponse):
            raise RuntimeError("transcription engine returned the wrong response type")
        return response

    return service


app = create_app()


def main() -> None:
    settings = app.state.settings
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
