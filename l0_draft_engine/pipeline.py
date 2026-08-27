from __future__ import annotations

import audioop
import hashlib
import math
import os
import re
import subprocess
import uuid
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


ROW_NAMESPACE = uuid.UUID("d2f99f29-0f5b-5d53-b3c8-f30795876b91")

class EngineError(RuntimeError):
    """Raised when the engine cannot produce a trustworthy candidate."""


@dataclass(frozen=True)
class AudioTrack:
    lane: str
    source_path: str
    derived_path: str
    sample_rate: int
    frame_count: int
    source_sha256: str
    pcm_sha256: str


@dataclass(frozen=True)
class Segment:
    id: str
    lane: str
    stage: str
    start_sample: int
    end_sample: int
    sample_rate: int
    parent_id: str | None = None

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass(frozen=True)
class SegmentationConfig:
    frame_ms: int = 10
    threshold_margin_db: float = 8.0
    threshold_min_dbfs: float = -60.0
    threshold_max_dbfs: float = -36.0
    bridge_gap_ms: int = 160
    minimum_activity_ms: int = 120
    coarse_silence_ms: int = 1000
    coarse_padding_ms: int = 0
    coarse_max_seconds: float = 45.0
    fine_silence_ms: int = 250
    fine_min_seconds: float = 2.5
    fine_target_seconds: float = 8.0
    fine_max_seconds: float = 14.0
    fine_hard_max_seconds: float = 24.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _run_ffmpeg(source: Path, destination: Path, mode: str, sample_rate: int = 16_000) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.{os.getpid()}.tmp.wav")
    filters = ["highpass=f=45"]
    if mode == "afftdn":
        filters.append("afftdn=nr=10:nf=-50:tn=1")
    elif mode != "raw":
        raise EngineError(f"unsupported preprocessing mode: {mode}")
    filters.append(f"aresample={sample_rate}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-af",
        ",".join(filters),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError(f"cannot preprocess {source}: {exc}") from exc
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip() or f"ffmpeg exited {completed.returncode}"
        raise EngineError(f"cannot preprocess {source}: {detail}")
    os.replace(temporary, destination)


def _read_pcm16_mono(path: Path) -> tuple[int, int, bytes]:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
                raise EngineError(f"expected mono uncompressed PCM16 WAV: {path}")
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            pcm = audio.readframes(frame_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise EngineError(f"cannot read WAV {path}: {exc}") from exc
    if len(pcm) != frame_count * 2:
        raise EngineError(f"truncated PCM data in {path}")
    return sample_rate, frame_count, pcm


def prepare_track(lane: str, source: Path, derived_dir: Path, mode: str) -> tuple[AudioTrack, bytes]:
    if not source.is_file():
        raise EngineError(f"audio input does not exist: {source}")
    source_hash = _sha256_file(source)
    derived = derived_dir / f"{lane}-{source_hash[:12]}-{mode}-16k.wav"
    if not derived.is_file():
        _run_ffmpeg(source, derived, mode)
    try:
        sample_rate, frame_count, pcm = _read_pcm16_mono(derived)
    except EngineError:
        derived.unlink(missing_ok=True)
        _run_ffmpeg(source, derived, mode)
        sample_rate, frame_count, pcm = _read_pcm16_mono(derived)
    if sample_rate != 16_000:
        raise EngineError(f"preprocessed audio has unexpected sample rate {sample_rate}: {derived}")
    with wave.open(str(source), "rb") as original:
        source_duration = original.getnframes() / original.getframerate()
    derived_duration = frame_count / sample_rate
    if abs(source_duration - derived_duration) > 0.02:
        raise EngineError(
            f"preprocessing changed duration by {derived_duration - source_duration:.6f}s for {source}"
        )
    pcm_hash = hashlib.sha256(pcm).hexdigest()
    return (
        AudioTrack(
            lane=lane,
            source_path=str(source.resolve()),
            derived_path=str(derived.resolve()),
            sample_rate=sample_rate,
            frame_count=frame_count,
            source_sha256=source_hash,
            pcm_sha256=pcm_hash,
        ),
        pcm,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise EngineError("cannot compute percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _frame_dbfs(pcm: bytes, sample_rate: int, frame_ms: int) -> tuple[list[float], int]:
    frame_samples = max(1, round(sample_rate * frame_ms / 1000))
    frame_bytes = frame_samples * 2
    values: list[float] = []
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : min(len(pcm), offset + frame_bytes)]
        if not frame:
            break
        rms = audioop.rms(frame, 2)
        values.append(20.0 * math.log10(max(rms, 1) / 32768.0))
    return values, frame_samples


def _runs(values: Sequence[bool], target: bool) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for index, value in enumerate(values):
        if value == target and start is None:
            start = index
        elif value != target and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(values)


def _smooth_activity(activity: list[bool], bridge_frames: int, minimum_frames: int) -> list[bool]:
    result = list(activity)
    for start, end in list(_runs(result, False)):
        if start > 0 and end < len(result) and end - start <= bridge_frames:
            result[start:end] = [True] * (end - start)
    for start, end in list(_runs(result, True)):
        if end - start < minimum_frames:
            result[start:end] = [False] * (end - start)
    return result


def _segment_id(lane: str, stage: str, start: int, end: int, parent: str | None = None) -> str:
    identity = f"{lane}|{stage}|{start}|{end}|{parent or ''}"
    return str(uuid.uuid5(ROW_NAMESPACE, identity))


def _make_segment(
    lane: str,
    stage: str,
    start: int,
    end: int,
    sample_rate: int,
    parent: str | None = None,
) -> Segment:
    if start < 0 or end <= start:
        raise EngineError(f"invalid {stage} segment {lane} {start}:{end}")
    return Segment(_segment_id(lane, stage, start, end, parent), lane, stage, start, end, sample_rate, parent)


def _split_long_coarse(
    start: int,
    end: int,
    activity: Sequence[bool],
    frame_samples: int,
    config: SegmentationConfig,
) -> list[tuple[int, int]]:
    max_samples = round(config.coarse_max_seconds * 16_000)
    if end - start <= max_samples:
        return [(start, end)]
    fine_frames = max(1, math.ceil(config.fine_silence_ms / config.frame_ms))
    silence_midpoints = [
        ((left + right) * frame_samples) // 2
        for left, right in _runs(activity, False)
        if right - left >= fine_frames and start < left * frame_samples < end
    ]
    pieces: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_samples:
        target = cursor + max_samples
        candidates = [point for point in silence_midpoints if cursor + frame_samples < point <= target]
        split = max(candidates) if candidates else target
        pieces.append((cursor, split))
        cursor = split
    pieces.append((cursor, end))
    return pieces


def segment_track(
    track: AudioTrack,
    pcm: bytes,
    config: SegmentationConfig,
    evidence_pcm: bytes | None = None,
) -> tuple[list[Segment], list[Segment], dict[str, float]]:
    dbfs, frame_samples = _frame_dbfs(pcm, track.sample_rate, config.frame_ms)
    noise_floor = _percentile(dbfs, 20.0)
    threshold = min(
        config.threshold_max_dbfs,
        max(config.threshold_min_dbfs, noise_floor + config.threshold_margin_db),
    )
    activity = _smooth_activity(
        [value >= threshold for value in dbfs],
        max(1, math.ceil(config.bridge_gap_ms / config.frame_ms)),
        max(1, math.ceil(config.minimum_activity_ms / config.frame_ms)),
    )
    evidence_noise_floor = noise_floor
    evidence_threshold = threshold
    if evidence_pcm is not None:
        evidence_dbfs, evidence_frame_samples = _frame_dbfs(
            evidence_pcm, track.sample_rate, config.frame_ms
        )
        if evidence_frame_samples != frame_samples or len(evidence_dbfs) != len(dbfs):
            raise EngineError(f"raw and denoised activity frames differ for {track.lane}")
        evidence_noise_floor = _percentile(evidence_dbfs, 20.0)
        evidence_threshold = min(
            config.threshold_max_dbfs,
            max(
                config.threshold_min_dbfs,
                evidence_noise_floor + config.threshold_margin_db,
            ),
        )
        evidence_activity = _smooth_activity(
            [value >= evidence_threshold for value in evidence_dbfs],
            max(1, math.ceil(config.bridge_gap_ms / config.frame_ms)),
            max(1, math.ceil(config.minimum_activity_ms / config.frame_ms)),
        )
        activity = [left or right for left, right in zip(activity, evidence_activity)]
        dbfs = [max(left, right) for left, right in zip(dbfs, evidence_dbfs)]
    active_runs = list(_runs(activity, True))
    if not active_runs:
        raise EngineError(f"no speech-like activity detected for {track.lane}")

    coarse_silence_frames = max(1, math.ceil(config.coarse_silence_ms / config.frame_ms))
    coarse_padding = round(config.coarse_padding_ms * track.sample_rate / 1000)
    split_frames = [
        (start + end) // 2
        for start, end in _runs(activity, False)
        if end - start >= coarse_silence_frames
        and start > active_runs[0][0]
        and end < active_runs[-1][1]
    ]
    boundaries = [active_runs[0][0] * frame_samples]
    boundaries.extend(frame * frame_samples for frame in split_frames)
    boundaries.append(min(track.frame_count, active_runs[-1][1] * frame_samples))

    coarse_ranges: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        first_active = next(
            (
                index
                for index in range(
                    left // frame_samples,
                    min(len(activity), math.ceil(right / frame_samples)),
                )
                if activity[index]
            ),
            None,
        )
        last_active = next(
            (
                index
                for index in range(
                    min(len(activity), math.ceil(right / frame_samples)) - 1,
                    left // frame_samples - 1,
                    -1,
                )
                if activity[index]
            ),
            None,
        )
        if first_active is None or last_active is None:
            continue
        start = max(left, first_active * frame_samples - coarse_padding)
        end = min(
            right,
            track.frame_count,
            (last_active + 1) * frame_samples + coarse_padding,
        )
        coarse_ranges.extend(
            _split_long_coarse(start, end, activity, frame_samples, config)
        )

    coarse = [
        _make_segment(track.lane, "S2", start, end, track.sample_rate)
        for start, end in coarse_ranges
    ]
    fine: list[Segment] = []
    for parent in coarse:
        fine.extend(_fine_segments(parent, activity, dbfs, frame_samples, config))
    diagnostics = {
        "noise_floor_dbfs": noise_floor,
        "activity_threshold_dbfs": threshold,
        "evidence_noise_floor_dbfs": evidence_noise_floor,
        "evidence_threshold_dbfs": evidence_threshold,
        "active_fraction": sum(activity) / len(activity),
        "coarse_segments": float(len(coarse)),
        "fine_segments": float(len(fine)),
    }
    return coarse, fine, diagnostics


def _fine_segments(
    parent: Segment,
    activity: Sequence[bool],
    dbfs: Sequence[float],
    frame_samples: int,
    config: SegmentationConfig,
) -> list[Segment]:
    start_frame = parent.start_sample // frame_samples
    end_frame = min(len(activity), math.ceil(parent.end_sample / frame_samples))
    minimum_silence_frames = max(1, math.ceil(config.fine_silence_ms / config.frame_ms))
    candidates = [
        ((left + right) * frame_samples) // 2
        for left, right in _runs(activity[start_frame:end_frame], False)
        if right - left >= minimum_silence_frames
    ]
    candidates = [parent.start_sample + point for point in candidates]
    min_length = round(config.fine_min_seconds * parent.sample_rate)
    target_length = round(config.fine_target_seconds * parent.sample_rate)
    max_length = round(config.fine_max_seconds * parent.sample_rate)
    hard_max = round(config.fine_hard_max_seconds * parent.sample_rate)

    boundaries = [parent.start_sample]
    cursor = parent.start_sample
    while parent.end_sample - cursor > max_length:
        eligible = [point for point in candidates if cursor + min_length <= point <= cursor + max_length]
        if eligible:
            target = cursor + target_length
            split = min(eligible, key=lambda point: (abs(point - target), point))
        elif parent.end_sample - cursor <= hard_max:
            break
        else:
            target_frame = (cursor + target_length) // frame_samples
            radius = max(1, round(1.0 * parent.sample_rate / frame_samples))
            low = max(cursor // frame_samples + 1, target_frame - radius)
            high = min(end_frame - 1, target_frame + radius)
            if high <= low:
                split = min(parent.end_sample - 1, cursor + target_length)
            else:
                quietest = min(range(low, high + 1), key=lambda index: (dbfs[index], abs(index - target_frame)))
                split = quietest * frame_samples
        if split <= cursor or split >= parent.end_sample:
            break
        boundaries.append(split)
        cursor = split
    boundaries.append(parent.end_sample)
    return [
        _make_segment(parent.lane, "S3", start, end, parent.sample_rate, parent.id)
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]

def _apply_backchannel_prior(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "мгм" if match.group(0).islower() else "Мгм"

    return re.sub(r"\bугу\b", replace, text, flags=re.IGNORECASE)
