from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import SegmentationConfig


class SettingsError(ValueError):
    """Raised for unsafe or internally inconsistent service settings."""


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8767
    device: str = "cuda"
    gigaam_model_path: str | Path = "v3_ctc"
    punctuation_model_path: str | Path = "kontur-ai/sbert_punc_case_ru"
    preprocessing: str = "raw"
    beam_size: int = 5
    hotwords: str = "Мгм мгм Угу угу Ага ага"
    cpu_threads: int = 4
    punctuation_chunk_words: int = 60
    max_track_bytes: int = 240 * 1024 * 1024
    max_request_bytes: int = 500 * 1024 * 1024
    max_audio_seconds: float = 4 * 60 * 60
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    max_inflight_requests: int = 3

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise SettingsError("LOCAL_ENGINE_HOST must be a loopback host")
        if self.device not in {"cuda", "cpu"}:
            raise SettingsError("LOCAL_ENGINE_DEVICE must be 'cuda' or 'cpu'")
        if self.preprocessing not in {"raw", "afftdn"}:
            raise SettingsError("LOCAL_ENGINE_PREPROCESSING must be 'raw' or 'afftdn'")
        if not str(self.gigaam_model_path).strip():
            raise SettingsError("LOCAL_ENGINE_GIGAAM_MODEL must not be empty")
        if not str(self.punctuation_model_path).strip():
            raise SettingsError("LOCAL_ENGINE_PUNCTUATION_MODEL must not be empty")
        if not self.hotwords.strip():
            raise SettingsError("LOCAL_ENGINE_HOTWORDS must not be empty")
        if self.max_request_bytes <= self.max_track_bytes * 2:
            raise SettingsError("LOCAL_ENGINE_MAX_REQUEST_BYTES must exceed two track limits")
        if not 1 <= self.max_inflight_requests <= 64:
            raise SettingsError(
                "LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS must be between 1 and 64"
            )

    @property
    def punctuation_dtype(self) -> str:
        return "float16" if self.device == "cuda" else "float32"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.environ.get("LOCAL_ENGINE_HOST", "127.0.0.1").strip(),
            port=_env_int("LOCAL_ENGINE_PORT", 8767, 1, 65535),
            device=os.environ.get("LOCAL_ENGINE_DEVICE", "cuda").strip().lower(),
            gigaam_model_path=os.environ.get("LOCAL_ENGINE_GIGAAM_MODEL", "v3_ctc").strip(),
            punctuation_model_path=os.environ.get(
                "LOCAL_ENGINE_PUNCTUATION_MODEL", "kontur-ai/sbert_punc_case_ru"
            ).strip(),
            preprocessing=os.environ.get("LOCAL_ENGINE_PREPROCESSING", "raw").strip().lower(),
            beam_size=_env_int("LOCAL_ENGINE_BEAM_SIZE", 5, 1, 10),
            hotwords=os.environ.get("LOCAL_ENGINE_HOTWORDS", "Мгм мгм Угу угу Ага ага"),
            cpu_threads=_env_int("LOCAL_ENGINE_CPU_THREADS", 4, 1, 32),
            punctuation_chunk_words=_env_int(
                "LOCAL_ENGINE_PUNCTUATION_CHUNK_WORDS", 60, 16, 300
            ),
            max_track_bytes=_env_int(
                "LOCAL_ENGINE_MAX_TRACK_BYTES", 240 * 1024 * 1024, 1024, 2**31
            ),
            max_request_bytes=_env_int(
                "LOCAL_ENGINE_MAX_REQUEST_BYTES", 500 * 1024 * 1024, 4096, 2**32
            ),
            max_inflight_requests=_env_int(
                "LOCAL_ENGINE_MAX_INFLIGHT_REQUESTS", 3, 1, 64
            ),
            max_audio_seconds=_env_float(
                "LOCAL_ENGINE_MAX_AUDIO_SECONDS", 4 * 60 * 60, 1.0, 24 * 60 * 60
            ),
        )
