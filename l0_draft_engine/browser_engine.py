"""Trusted backend using the exact released client runtime over a private pipe."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import threading
import time
import uuid

from .engine import DraftInputError, ModelUnavailableError
from .inference_release import RELEASE_ID, validate_punctuated_timing
from .schemas import DraftPayload, TranscriptionResponse, DraftResponse, DraftOptions
from .enhancement import (
    COPY_BYTES, EnhancementModel, EnhancementPayload, EnhancementProgress,
    EnhancementResult, _validate_wav,
)


class BrowserDraftEngine:
    def __init__(self, settings):
        self.settings = settings
        self._process = None
        self._responses = Queue()
        self._lock = threading.RLock()
        self._prepared = False
        self._enhancement = None
        self._closed = False

    @contextmanager
    def model_session(self):
        if self._closed:
            raise ModelUnavailableError("The browser backend is closed")
        yield

    def _start(self):
        if self._closed:
            raise ModelUnavailableError("The browser backend is closed")
        if self._process is not None and self._process.poll() is None:
            return
        worker = Path(os.environ.get("BABEL_INFERENCE_WORKER", ""))
        if not worker.is_file() or not os.environ.get("BABEL_INFERENCE_PROFILE"):
            raise ModelUnavailableError("Configure BABEL_INFERENCE_WORKER and a dedicated BABEL_INFERENCE_PROFILE")
        self._responses = Queue()
        self._prepared = False
        self._process = subprocess.Popen(
            [os.environ.get("BABEL_INFERENCE_NODE", "node"), str(worker.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, encoding="utf-8", bufsize=1,
        )
        responses, process = self._responses, self._process
        def read_responses():
            try:
                for line in process.stdout:
                    try:
                        responses.put(json.loads(line))
                    except ValueError:
                        responses.put({"error": "Backend emitted invalid RPC data"})
            finally:
                responses.put(None)
        threading.Thread(target=read_responses, daemon=True).start()

    def _call(self, operation, *, on_progress=None, **values):
        with self._lock:
            self._start()
            identity = uuid.uuid4().hex
            deadline = time.monotonic() + float(os.environ.get("BABEL_INFERENCE_RPC_SECONDS", "840"))
            try:
                self._process.stdin.write(json.dumps({"id": identity, "operation": operation, **values}) + "\n")
                self._process.stdin.flush()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise Empty
                    result = self._responses.get(timeout=remaining)
                    if not isinstance(result, dict) or result.get("id") != identity:
                        raise ValueError("Backend RPC lost its request ownership")
                    if "progress" in result:
                        if set(result) != {"id", "progress"} or on_progress is None:
                            raise ValueError("Unexpected backend RPC progress")
                        on_progress(result["progress"])
                        continue
                    if set(result) == {"id", "error"} and isinstance(result["error"], str):
                        raise ModelUnavailableError(result["error"])
                    if set(result) != {"id", "result"} or not isinstance(result["result"], dict):
                        raise ValueError("Invalid backend RPC final response")
                    return result["result"]
            except (BrokenPipeError, OSError, Empty) as exc:
                self._stop()
                raise ModelUnavailableError("Browser inference stopped or timed out") from exc
            except (ValueError, TypeError, KeyError) as exc:
                self._stop()
                raise ModelUnavailableError("Browser backend RPC validation failed") from exc

    def prepare(self):
        with self._lock:
            self._start()
            if not self._prepared:
                result = self._call("prepare")
                if result.get("ok") is not True or result.get("release") != RELEASE_ID or result.get("provider") != "webgpu":
                    self._stop()
                    raise ModelUnavailableError("Backend did not prove the required model release on WebGPU")
                try:
                    capability = dict(result["enhancement"])
                    if capability.pop("provider") != "webgpu":
                        raise ValueError("Enhancement preflight did not use WebGPU")
                    self._enhancement = EnhancementModel.model_validate(capability)
                except (KeyError, TypeError, ValueError) as exc:
                    self._stop()
                    raise ModelUnavailableError("Backend did not prove ZipEnhancer on WebGPU") from exc
                self._prepared = True
        return self.health()

    def transcribe(self, payload: DraftPayload, paths: dict[str, Path]) -> TranscriptionResponse:
        if payload.options and payload.options.preprocessing not in {None, "raw"}:
            raise DraftInputError("C-denoise requires raw audio; denoising is used only for activity segmentation")
        if set(paths) != {track.lane for track in payload.tracks}:
            raise DraftInputError("Audio paths must match the declared lanes")
        with self._lock:
            self.prepare()
            result = TranscriptionResponse.model_validate(self._call("transcribe",
                payload=payload.model_dump(exclude_none=True),
                paths={track.fieldName: str(paths[track.lane].resolve()) for track in payload.tracks}))
            validate_punctuated_timing(result)
            return result

    def draft(self, timing: TranscriptionResponse, options: DraftOptions | None = None) -> DraftResponse:
        try:
            validate_punctuated_timing(timing)
        except ValueError as exc:
            raise DraftInputError(str(exc)) from exc
        with self._lock:
            self.prepare()
            return DraftResponse.model_validate(self._call("draft",
                timing=timing.model_dump(exclude_none=True),
                options=options.model_dump(exclude_none=True) if options else None))

    def enhance(self, payload: EnhancementPayload, paths: dict[str, Path],
                outputs: dict[str, Path], on_progress) -> EnhancementResult:
        with self._lock:
            self.prepare()
            if payload.model != self._enhancement:
                raise DraftInputError("Requested enhancement model does not match the prepared backend")
            expected = {track.fieldName for track in payload.tracks}
            if set(paths) != expected or set(outputs) != expected:
                raise DraftInputError("Enhancement paths must match the declared tracks")
            previous = None

            def progress(value):
                nonlocal previous
                update = EnhancementProgress.model_validate(value)
                update.validate_update(payload, previous)
                previous = update
                on_progress(update.model_dump())

            try:
                result = EnhancementResult.model_validate(self._call(
                    "enhance", on_progress=progress, payload=payload.model_dump(exclude_none=True),
                    paths={key: str(path.resolve()) for key, path in paths.items()},
                    outputs={key: str(path.resolve()) for key, path in outputs.items()},
                ))
                result.validate_request(payload)
                for source, metadata in zip(payload.tracks, result.tracks, strict=True):
                    path = outputs[source.fieldName]
                    if path.stat().st_size != metadata.totalBytes:
                        raise ValueError("Incomplete backend audio")
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        while block := stream.read(COPY_BYTES):
                            digest.update(block)
                    if digest.hexdigest() != metadata.wavSha256:
                        raise ValueError("Backend audio digest mismatch")
                    _validate_wav(path, metadata, output=True)
                return result
            except (ValueError, OSError) as exc:
                self._stop()
                raise ModelUnavailableError("Backend enhancement result validation failed") from exc

    def health(self):
        ready = self._prepared and self._process is not None and self._process.poll() is None
        return {"ok": ready, "service": "c-denoise-webgpu-backend", "release": RELEASE_ID,
                "provider": "webgpu", "neuralCpuFallback": False,
                **({"enhancement": {**self._enhancement.model_dump(), "provider": "webgpu"}}
                   if ready and self._enhancement is not None else {})}

    def _stop(self):
        process, self._process = self._process, None
        self._prepared = False
        self._enhancement = None
        if process is not None:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()

    def close(self):
        with self._lock:
            self._closed = True
            self._stop()
