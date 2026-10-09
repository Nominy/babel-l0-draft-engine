"""Ephemeral swarm enhancement wire validation; never performs inference or caches audio.

Both uploads are original WAVs. Only the matching model capability may process them;
outputs retain the original clock and identity and are canonical mono PCM16 WAVs.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
import hashlib
import json
import math
from pathlib import Path
import secrets
import struct
from typing import Annotated, Literal, TypeVar

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


MODEL_ID = "zipenhancer-webgpu-2026-10-09-r1"
SOURCE_GRAPH_SHA256 = "2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016"
AUDIO_CHUNK_BYTES = 512 * 1024
COPY_BYTES = 1024 * 1024
PAYLOAD_BYTES = 64 * 1024
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=256)]


class EnhancementSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class EnhancementModel(EnhancementSchema):
    id: Literal["zipenhancer-webgpu-2026-10-09-r1"]
    sha256: Sha256
    sourceGraphSha256: Literal["2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016"]


class TrackIdentity(EnhancementSchema):
    trackId: Identifier
    speakerKey: Identifier
    trackLabel: Identifier
    sourceSha256: Sha256
    sampleRate: int = Field(gt=0, le=0xFFFFFFFF)
    frameCount: int = Field(gt=0, le=0x7FFFFFFF)

    @field_validator("trackId", "speakerKey", "trackLabel")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("identifiers must be nonempty and contain no control characters")
        return value


class EnhancementTrack(TrackIdentity):
    fieldName: Literal["audio:1", "audio:2"]


class EnhancementPayload(EnhancementSchema):
    taskId: Identifier
    model: EnhancementModel
    tracks: list[EnhancementTrack] = Field(min_length=2, max_length=2)

    @field_validator("taskId")
    @classmethod
    def safe_task_id(cls, value: str) -> str:
        return TrackIdentity.safe_identifier(value)

    @model_validator(mode="after")
    def distinct_tracks(self) -> EnhancementPayload:
        if len({track.trackId for track in self.tracks}) != 2:
            raise ValueError("track IDs must be distinct")
        if [track.fieldName for track in self.tracks] != ["audio:1", "audio:2"]:
            raise ValueError("audio fields must follow the declared track order")
        return self


class EnhancementTrackMetadata(TrackIdentity):
    mimeType: Literal["audio/wav"]
    wavSha256: Sha256
    totalBytes: int = Field(gt=44)
    chunkCount: int = Field(gt=0)

    @model_validator(mode="after")
    def canonical_size(self) -> EnhancementTrackMetadata:
        if self.totalBytes != 44 + 2 * self.frameCount:
            raise ValueError("result byte length must match its PCM16 clock")
        if self.chunkCount != (self.totalBytes + AUDIO_CHUNK_BYTES - 1) // AUDIO_CHUNK_BYTES:
            raise ValueError("result chunk count must match its byte length")
        return self


class EnhancementResult(EnhancementSchema):
    model: Literal["zipenhancer-webgpu-2026-10-09-r1"]
    modelSha256: Sha256
    tracks: list[EnhancementTrackMetadata] = Field(min_length=2, max_length=2)

    def validate_request(self, payload: EnhancementPayload) -> None:
        if self.model != payload.model.id or self.modelSha256 != payload.model.sha256:
            raise ValueError("result model must match the requested model capability")
        keys = TrackIdentity.model_fields.keys()
        for source, result in zip(payload.tracks, self.tracks, strict=True):
            if any(getattr(source, key) != getattr(result, key) for key in keys):
                raise ValueError("result tracks must retain original order, identity and clock")


class EnhancementProgress(EnhancementSchema):
    phase: Literal["loading-model", "enhancing", "encoding"]
    trackId: Identifier
    trackIndex: int = Field(ge=0, le=1)
    trackCount: Literal[2]
    completedChunks: int = Field(ge=0)
    totalChunks: int = Field(ge=0)

    def validate_update(self, payload: EnhancementPayload, previous: EnhancementProgress | None) -> None:
        track = payload.tracks[self.trackIndex]
        maximum = math.ceil(track.frameCount / track.sampleRate / 3) + 2
        if self.trackId != track.trackId or not self.completedChunks <= self.totalChunks <= maximum:
            raise ValueError("progress must describe the leased track and bounded chunk counts")
        if self.phase == "loading-model" and (self.completedChunks or self.totalChunks):
            raise ValueError("model loading has no completed inference chunks")
        if self.phase != "loading-model" and self.totalChunks == 0:
            raise ValueError("inference progress needs a positive chunk count")
        if self.phase == "encoding" and self.completedChunks != self.totalChunks:
            raise ValueError("encoding starts only after all inference chunks complete")
        if previous is None:
            if self.trackIndex != 0:
                raise ValueError("progress must start at the first track")
            return
        if self.trackIndex < previous.trackIndex:
            raise ValueError("progress track index must be monotonic")
        if self.trackIndex > previous.trackIndex:
            if previous.phase != "encoding":
                raise ValueError("the preceding track must finish before advancing")
            return
        ranks = {"loading-model": 0, "enhancing": 1, "encoding": 2}
        if (self.completedChunks < previous.completedChunks
                or (previous.totalChunks and self.totalChunks != previous.totalChunks)
                or ranks[self.phase] < ranks[previous.phase]):
            raise ValueError("progress must not regress or change its established chunk count")


Schema = TypeVar("Schema", bound=BaseModel)


def parse_multipart(form: FormData, schema: type[Schema]) -> tuple[Schema, dict[str, UploadFile]]:
    payloads: list[str] = []
    files: dict[str, UploadFile] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile) and key in ("audio:1", "audio:2") and key not in files:
            files[key] = value
        elif key == "payload" and isinstance(value, str):
            payloads.append(value)
        else:
            raise HTTPException(422, "unexpected or duplicate multipart field")
    if len(payloads) != 1 or len(payloads[0].encode("utf-8")) > PAYLOAD_BYTES:
        raise HTTPException(422, "exactly one bounded JSON payload is required")
    if set(files) != {"audio:1", "audio:2"}:
        raise HTTPException(422, "exactly two audio parts are required")
    try:
        return schema.model_validate_json(payloads[0]), files
    except ValidationError as exc:
        raise HTTPException(422, "invalid enhancement payload") from exc


async def read_multipart(request: Request) -> FormData:
    # Starlette closes partial spools for MultiPartException, but not for an ASGI
    # body limit, disconnect or task cancellation. Close every partial spool here.
    parser = MultiPartParser(request.headers, request.stream(), max_files=2, max_fields=1)
    try:
        return await parser.parse()
    except BaseException as exc:
        for spool in parser._files_to_close_on_error:
            spool.close()
        if isinstance(exc, (MultiPartException, ValueError)):
            raise HTTPException(400, "invalid enhancement multipart request") from exc
        raise


def _validate_wav(path: Path, identity: TrackIdentity, *, output: bool) -> None:
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("audio must be a RIFF WAV")
        if struct.unpack_from("<I", header, 4)[0] + 8 != size:
            raise ValueError("WAV RIFF size must match the uploaded bytes")
        fmt = None
        data = None
        while stream.tell() < size:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError("truncated WAV chunk header")
            kind, length = struct.unpack("<4sI", chunk)
            offset = stream.tell()
            end = offset + length + (length & 1)
            if end > size:
                raise ValueError("truncated WAV chunk")
            if kind == b"fmt ":
                if fmt is not None or length < 16:
                    raise ValueError("invalid or duplicate WAV format chunk")
                fmt = struct.unpack("<HHIIHH", stream.read(16))
                if output and (offset != 20 or length != 16):
                    raise ValueError("output must have a canonical PCM16 WAV header")
            elif kind == b"data":
                if data is not None:
                    raise ValueError("duplicate WAV audio chunk")
                data = (offset, length)
            stream.seek(end)
        if fmt is None or data is None:
            raise ValueError("WAV format and audio chunks are required")
        format_code, channels, rate, byte_rate, align, bits = fmt
        width = 2 if (format_code, bits) == (1, 16) else 4 if (format_code, bits) == (3, 32) else 0
        if not width or channels < 1 or align != channels * width or byte_rate != rate * align:
            raise ValueError("source must be PCM16 or float32 WAV with consistent clock")
        offset, length = data
        if rate != identity.sampleRate or length != identity.frameCount * align:
            raise ValueError("WAV clock must match the declared original clock")
        if output and (format_code != 1 or channels != 1 or offset != 44 or size != 44 + length):
            raise ValueError("output must be canonical mono PCM16 WAV")
        if format_code == 3:
            stream.seek(offset)
            remaining = length
            while remaining:
                block = stream.read(min(COPY_BYTES, remaining))
                if not block or any(not math.isfinite(sample[0]) for sample in struct.iter_unpack("<f", block)):
                    raise ValueError("source WAV contains invalid float samples")
                remaining -= len(block)


async def copy_audio(upload: UploadFile, path: Path, identity: TrackIdentity, *,
                     max_bytes: int, max_seconds: float, output: bool = False) -> None:
    if identity.frameCount / identity.sampleRate > max_seconds:
        raise HTTPException(413, "audio track exceeds duration limit")
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as destination:
        while chunk := await upload.read(COPY_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, "audio track exceeds size limit")
            digest.update(chunk)
            destination.write(chunk)
    expected = identity.wavSha256 if isinstance(identity, EnhancementTrackMetadata) else identity.sourceSha256
    if digest.hexdigest() != expected:
        raise HTTPException(422, "audio digest does not match declared identity")
    if isinstance(identity, EnhancementTrackMetadata) and size != identity.totalBytes:
        raise HTTPException(422, "result audio length does not match metadata")
    try:
        _validate_wav(path, identity, output=output)
    except (ValueError, OSError, struct.error) as exc:
        raise HTTPException(422, "invalid enhancement WAV or mismatched original clock") from exc


class EnhancementResponse(StreamingResponse):
    """Hold request admission and temporary audio until send/disconnect completes."""

    def __init__(self, payload: EnhancementPayload, result: EnhancementResult,
                 paths: dict[str, Path], cleanup: Callable[[bool], Awaitable[None]]) -> None:
        boundary = "babel-" + secrets.token_hex(24)
        report = {"ok": True, "provider": "swarm", "taskId": payload.taskId, **result.model_dump()}
        first = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"result\"\r\n"
                 "Content-Type: application/json\r\n\r\n").encode() + json.dumps(
                     report, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                 ).encode("utf-8") + b"\r\n"
        parts = [(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{track.fieldName}\"; "
                  f"filename=\"enhanced-{index}.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
                 for index, track in enumerate(payload.tracks)]
        end = f"--{boundary}--\r\n".encode()
        length = len(first) + len(end) + sum(len(part) + metadata.totalBytes + 2
                                          for part, metadata in zip(parts, result.tracks, strict=True))

        async def stream() -> AsyncIterator[bytes]:
            yield first
            for part, track in zip(parts, payload.tracks, strict=True):
                yield part
                with paths[track.fieldName].open("rb") as source:
                    while block := source.read(COPY_BYTES):
                        yield block
                yield b"\r\n"
            yield end
            self.sent = True

        self.cleanup = cleanup
        self.sent = False
        super().__init__(stream(), media_type=f"multipart/form-data; boundary={boundary}",
                         headers={"Content-Length": str(length), "Cache-Control": "no-store"})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                await self.cleanup(self.sent)
