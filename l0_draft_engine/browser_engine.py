"""Trusted backend using the exact released client runtime over a private pipe."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import threading
import uuid

from .engine import DraftInputError, ModelUnavailableError
from .inference_release import RELEASE_ID, validate_punctuated_timing
from .schemas import DraftPayload, TranscriptionResponse, DraftResponse, DraftOptions


class BrowserDraftEngine:
    def __init__(self, settings):
        self.settings = settings
        self._process = None
        self._responses = Queue()
        self._lock = threading.RLock()
        self._prepared = False
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

    def _call(self, operation, **values):
        with self._lock:
            self._start()
            identity = uuid.uuid4().hex
            try:
                self._process.stdin.write(json.dumps({"id": identity, "operation": operation, **values}) + "\n")
                self._process.stdin.flush()
                result = self._responses.get(timeout=float(os.environ.get("BABEL_INFERENCE_RPC_SECONDS", "840")))
            except (BrokenPipeError, OSError, Empty) as exc:
                self._stop()
                raise ModelUnavailableError("C-denoise browser inference stopped or timed out") from exc
            if result is None or result.get("id") != identity:
                self._stop()
                raise ModelUnavailableError("C-denoise backend RPC lost its session")
            if result.get("error"):
                raise ModelUnavailableError(result["error"])
            return result["result"]

    def prepare(self):
        with self._lock:
            self._start()
            if not self._prepared:
                result = self._call("prepare")
                if result.get("release") != RELEASE_ID or result.get("provider") != "webgpu":
                    self._stop()
                    raise ModelUnavailableError("Backend did not prove the required model release on WebGPU")
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

    def health(self):
        return {"ok": self._prepared and self._process is not None and self._process.poll() is None,
                "service": "c-denoise-webgpu-backend", "release": RELEASE_ID,
                "provider": "webgpu", "neuralCpuFallback": False}

    def _stop(self):
        process, self._process = self._process, None
        self._prepared = False
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
