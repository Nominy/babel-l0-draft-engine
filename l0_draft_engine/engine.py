from __future__ import annotations

from bisect import bisect_left
from importlib.util import find_spec
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import (
    AudioTrack,
    EngineError as L0EngineError,
    Segment,
    _apply_backchannel_prior,
    prepare_track,
    segment_track,
)

from .config import Settings
from .gigaam_asr import GigaAMRecognizer
from .l2 import PunctuationFormatter
from .schemas import DraftPayload, DraftResponse, DraftRow

ROW_NAMESPACE = uuid.UUID("54057e89-dfb6-5f31-925d-6119e48bdac4")
MARKUP_RE = re.compile(r"\[[^\[\]\r\n]+\]|</?[^<>\r\n]+>|\{[^{}\r\n]+\}")
EDGE_PUNCTUATION_RE = re.compile(r'^["«»„“”(),.!?;:…]+|["«»„“”(),.!?;:…]+$')


class DraftInputError(ValueError):
    """Raised when valid multipart input contains unusable audio."""


class ModelUnavailableError(RuntimeError):
    """Raised when a configured local model cannot be loaded or executed."""


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    surface: str


@dataclass(frozen=True)
class RowCandidate:
    segment: Segment
    words: tuple[str, ...]
    track_sha256: str
    output_id: str | None = None
    output_start_seconds: float | None = None
    output_end_seconds: float | None = None
    fallback_text: str | None = None
    preserve_order: int | None = None


class DraftEngine:
    """Lazy, cached local ASR/L2 engine with one serialized inference lane."""

    def __init__(
        self,
        settings: Settings,
        *,
        asr_factory: Callable[[], Any] | None = None,
        formatter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._asr_factory = asr_factory or self._load_asr
        self._formatter_factory = formatter_factory or self._load_formatter
        self._asr: Any | None = None
        self._formatter: Any | None = None
        self._gpu_lock = threading.Lock()
        self._admission = threading.BoundedSemaphore(1)

    def try_admit(self) -> bool:
        return self._admission.acquire(blocking=False)

    def release_admission(self) -> None:
        self._admission.release()

    def _load_asr(self) -> Any:
        try:
            return GigaAMRecognizer(
                self.settings.gigaam_model_path,
                self.settings.device,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"cannot initialize GigaAM: {exc}") from exc

    def _load_formatter(self) -> PunctuationFormatter:
        return PunctuationFormatter(
            self.settings.punctuation_model_path,
            self.settings.device,
            self.settings.punctuation_chunk_words,
        )

    def _get_asr(self) -> Any:
        if self._asr is None:
            try:
                self._asr = self._asr_factory()
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(f"cannot initialize GigaAM: {exc}") from exc
        return self._asr

    def _get_formatter(self) -> Any:
        if self._formatter is None:
            try:
                self._formatter = self._formatter_factory()
            except Exception as exc:
                raise ModelUnavailableError(f"cannot initialize punctuation model: {exc}") from exc
        return self._formatter

    def _asr_cached(self) -> bool:
        reference = str(self.settings.gigaam_model_path)
        if Path(reference).is_file():
            return True
        return reference == "v3_ctc" and (
            Path.home() / ".cache" / "gigaam" / "v3_ctc.ckpt"
        ).is_file()

    def _punctuation_cached(self) -> bool:
        path = Path(str(self.settings.punctuation_model_path))
        if not path.is_dir():
            return False
        metadata_present = all(
            (path / name).is_file() for name in ("config.json", "tokenizer.json")
        )
        weights_present = (path / "model.safetensors").is_file() or (
            path / "pytorch_model.bin"
        ).is_file()
        return metadata_present and weights_present
    def _asr_configured(self) -> bool:
        reference = str(self.settings.gigaam_model_path)
        path = Path(reference)
        return path.is_file() if path.is_absolute() or path.suffix == ".ckpt" else True

    def _punctuation_configured(self) -> bool:
        reference = str(self.settings.punctuation_model_path)
        path = Path(reference)
        return path.is_dir() if path.is_absolute() or reference.startswith(".") else True


    def model_summary(self) -> dict[str, object]:
        asr_reference = str(self.settings.gigaam_model_path)
        punctuation_reference = str(self.settings.punctuation_model_path)
        return {
            "asr": {
                "name": asr_reference,
                "device": self.settings.device,
                "loaded": self._asr is not None,
                "cached": self._asr_cached(),
                "checkpoint": Path(asr_reference).name,
            },
            "l2": {
                "name": punctuation_reference,
                "device": self.settings.device,
                "dtype": self.settings.punctuation_dtype,
                "loaded": (
                    self._formatter is not None
                    and bool(getattr(self._formatter, "loaded", True))
                ),
                "cached": self._punctuation_cached(),
            },
        }

    def health(self) -> dict[str, object]:
        models = self.model_summary()
        runtimes_available = all(
            find_spec(module) is not None
            for module in ("numpy", "soundfile", "torch", "transformers")
        )
        device_available = False
        if runtimes_available:
            try:
                import torch

                device_available = self.settings.device == "cpu" or torch.cuda.is_available()
            except Exception:
                device_available = False
        return {
            "ok": (
                runtimes_available
                and device_available
                and self._asr_configured()
                and self._punctuation_configured()
            ),
            "device": self.settings.device,
            "models": models,
        }

    def _transcribe_lane(self, model: Any, track: AudioTrack) -> list[Word]:
        try:
            if isinstance(model, GigaAMRecognizer):
                words = [
                    Word(word.start, word.end, word.surface)
                    for word in model.transcribe(Path(track.derived_path))
                ]
            else:
                segments, _ = model.transcribe(
                    track.derived_path,
                    language="ru",
                    beam_size=self.settings.beam_size,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    vad_filter=False,
                    hotwords=self.settings.hotwords,
                )
                words = []
                for result in segments:
                    for raw_word in result.words or ():
                        if raw_word.start is None or raw_word.end is None:
                            continue
                        start = float(raw_word.start)
                        end = float(raw_word.end)
                        if start < 0 or end <= start:
                            continue
                        surface = EDGE_PUNCTUATION_RE.sub(
                            "", str(raw_word.word).strip()
                        )
                        if not surface:
                            continue
                        surface = _apply_backchannel_prior(surface)
                        words.append(Word(start, end, surface))
        except Exception as exc:
            raise ModelUnavailableError(
                f"ASR failed for lane {track.lane}: {exc}"
            ) from exc
        words.sort(
            key=lambda word: (
                (word.start + word.end) / 2,
                word.start,
                word.end,
                word.surface,
            )
        )
        return words

    @staticmethod
    def _group_rows(
        payload: DraftPayload,
        tracks: dict[str, AudioTrack],
        coarse_by_lane: dict[str, Sequence[Segment]],
        words_by_lane: dict[str, Sequence[Word]],
    ) -> list[RowCandidate]:
        midpoints_by_lane = {
            lane: [(word.start + word.end) / 2 for word in words]
            for lane, words in words_by_lane.items()
        }
        preserved = payload.options.preserveRows if payload.options is not None else None
        if preserved is not None:
            candidates: list[RowCandidate] = []
            for row in preserved:
                lane = row.speakerKey
                lane_words = words_by_lane[lane]
                midpoints = midpoints_by_lane[lane]
                left = bisect_left(midpoints, row.startSeconds)
                right = bisect_left(midpoints, row.endSeconds)
                words = tuple(word.surface for word in lane_words[left:right])
                fallback = None
                existing = _apply_backchannel_prior(row.text.strip())
                if MARKUP_RE.search(existing):
                    words = ()
                    fallback = existing
                elif not words:
                    fallback = existing
                    if not fallback:
                        raise DraftInputError(
                            f"preserved row {row.rowId} has no ASR words or existing text"
                        )
                sample_rate = tracks[lane].sample_rate
                start_sample = round(row.startSeconds * sample_rate)
                end_sample = max(start_sample + 1, round(row.endSeconds * sample_rate))
                segment = Segment(
                    id=row.rowId,
                    lane=lane,
                    stage="preserve",
                    start_sample=start_sample,
                    end_sample=end_sample,
                    sample_rate=sample_rate,
                )
                candidates.append(
                    RowCandidate(
                        segment=segment,
                        words=words,
                        track_sha256=tracks[lane].pcm_sha256,
                        output_id=row.rowId,
                        output_start_seconds=row.startSeconds,
                        output_end_seconds=row.endSeconds,
                        fallback_text=fallback,
                        preserve_order=row.index,
                    )
                )
            return candidates

        candidates = []
        for track_spec in payload.tracks:
            lane = track_spec.lane
            lane_words = list(words_by_lane[lane])
            if not lane_words:
                raise DraftInputError(f"ASR produced no words for lane {lane}")
            groups: list[list[Word]] = []
            current: list[Word] = []
            for word in lane_words:
                should_split = bool(current) and (
                    word.start - current[-1].end >= 0.8
                    or word.end - current[0].start > 12.0
                    or len(current) >= 32
                )
                if should_split:
                    groups.append(current)
                    current = []
                current.append(word)
            if current:
                groups.append(current)
            sample_rate = tracks[lane].sample_rate
            for index, group in enumerate(groups):
                start_sample = max(0, round((group[0].start - 0.01) * sample_rate))
                end_sample = min(
                    tracks[lane].frame_count,
                    round((group[-1].end + 0.01) * sample_rate),
                )
                end_sample = max(start_sample + 1, end_sample)
                segment = Segment(
                    id=f"word-group-{lane}-{index:06d}",
                    lane=lane,
                    stage="words",
                    start_sample=start_sample,
                    end_sample=end_sample,
                    sample_rate=sample_rate,
                )
                candidates.append(
                    RowCandidate(
                        segment,
                        tuple(word.surface for word in group),
                        tracks[lane].pcm_sha256,
                    )
                )
        return candidates

    def _format_candidates(self, candidates: Sequence[RowCandidate]) -> dict[str, str]:
        formatted = {
            candidate.segment.id: candidate.fallback_text
            for candidate in candidates
            if candidate.fallback_text is not None
        }
        spoken = [candidate for candidate in candidates if candidate.words]
        if not spoken:
            return formatted
        formatter = self._get_formatter()
        lanes = sorted({candidate.segment.lane for candidate in spoken})
        for lane in lanes:
            lane_candidates = sorted(
                (candidate for candidate in spoken if candidate.segment.lane == lane),
                key=lambda candidate: (
                    candidate.segment.start_sample,
                    candidate.segment.end_sample,
                    candidate.segment.id,
                ),
            )
            try:
                texts = formatter.format_rows([candidate.words for candidate in lane_candidates])
            except Exception as exc:
                raise ModelUnavailableError(
                    f"punctuation inference failed for lane {lane}: {exc}"
                ) from exc
            if len(texts) != len(lane_candidates):
                raise ModelUnavailableError(
                    f"punctuation model returned the wrong row count for lane {lane}"
                )
            for candidate, text in zip(lane_candidates, texts):
                if not str(text).strip():
                    raise ModelUnavailableError(
                        f"punctuation model returned an empty row for lane {lane}"
                    )
                formatted[candidate.segment.id] = str(text).strip()
        return formatted

    def draft(self, payload: DraftPayload, audio_paths: dict[str, Path]) -> DraftResponse:
        started = time.perf_counter()
        expected_lanes = [track.lane for track in payload.tracks]
        if set(audio_paths) != set(expected_lanes) or len(audio_paths) != 2:
            raise DraftInputError("audio paths must match exactly two payload lanes")
        preprocessing = (
            payload.options.preprocessing
            if payload.options is not None and payload.options.preprocessing is not None
            else self.settings.preprocessing
        )
        workspace = next(iter(audio_paths.values())).parent / "prepared"
        tracks: dict[str, AudioTrack] = {}
        coarse_by_lane: dict[str, Sequence[Segment]] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        try:
            for lane in expected_lanes:
                track, pcm = prepare_track(lane, audio_paths[lane], workspace, preprocessing)
                evidence_pcm: bytes | None = None
                if preprocessing != "raw":
                    _, evidence_pcm = prepare_track(lane, audio_paths[lane], workspace, "raw")
                coarse, _, lane_diagnostics = segment_track(
                    track, pcm, self.settings.segmentation, evidence_pcm
                )
                if not coarse:
                    raise DraftInputError(f"segmentation produced no S2 rows for lane {lane}")
                tracks[lane] = track
                coarse_by_lane[lane] = coarse
                diagnostics[lane] = lane_diagnostics
                del pcm, evidence_pcm
        except DraftInputError:
            raise
        except L0EngineError as exc:
            raise DraftInputError(str(exc)) from exc
        preprocess_finished = time.perf_counter()

        wait_started = time.perf_counter()
        with self._gpu_lock:
            inference_started = time.perf_counter()
            model = self._get_asr()
            words_by_lane = {
                lane: self._transcribe_lane(model, tracks[lane]) for lane in expected_lanes
            }
            asr_finished = time.perf_counter()
            candidates = self._group_rows(
                payload, tracks, coarse_by_lane, words_by_lane
            )
            formatted = self._format_candidates(candidates)
            inference_finished = time.perf_counter()

        if payload.options is not None and payload.options.preserveRows is not None:
            candidates = sorted(
                candidates,
                key=lambda candidate: (
                    candidate.preserve_order
                    if candidate.preserve_order is not None
                    else 2**31
                ),
            )
        else:
            lane_order = {lane: index for index, lane in enumerate(expected_lanes)}
            candidates = sorted(
                candidates,
                key=lambda candidate: (
                    candidate.segment.start_sample / candidate.segment.sample_rate,
                    lane_order[candidate.segment.lane],
                    candidate.segment.end_sample,
                ),
            )
        rows: list[DraftRow] = []
        for candidate in candidates:
            segment = candidate.segment
            if candidate.output_id is not None:
                row_id = candidate.output_id
            else:
                row_id = str(
                    uuid.uuid5(
                        ROW_NAMESPACE,
                        "|".join(
                            (
                                payload.taskId,
                                segment.lane,
                                candidate.track_sha256,
                                str(segment.start_sample),
                                str(segment.end_sample),
                            )
                        ),
                    )
                )
            rows.append(
                DraftRow(
                    id=row_id,
                    lane=segment.lane,
                    startSeconds=(
                        candidate.output_start_seconds
                        if candidate.output_start_seconds is not None
                        else round(segment.start_seconds, 6)
                    ),
                    endSeconds=(
                        candidate.output_end_seconds
                        if candidate.output_end_seconds is not None
                        else round(segment.end_seconds, 6)
                    ),
                    text=formatted[segment.id],
                )
            )
        if not rows or any(row.endSeconds <= row.startSeconds for row in rows):
            raise DraftInputError("draft did not produce positive rows")
        finished = time.perf_counter()
        return DraftResponse(
            rows=rows,
            summary={
                "taskId": payload.taskId,
                "trackCount": 2,
                "rowCount": len(rows),
                "wordCount": sum(len(candidate.words) for candidate in candidates),
                "preprocessing": preprocessing,
                "preservedRows": (
                    payload.options is not None
                    and payload.options.preserveRows is not None
                ),
                "latencyMs": {
                    "preparation": round((preprocess_finished - started) * 1000),
                    "gpuQueue": round((inference_started - wait_started) * 1000),
                    "asr": round((asr_finished - inference_started) * 1000),
                    "l2": round((inference_finished - asr_finished) * 1000),
                    "total": round((finished - started) * 1000),
                },
                "segmentation": diagnostics,
            },
            models=self.model_summary(),
        )
