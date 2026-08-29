from __future__ import annotations

from array import array
import math
import wave

import pytest

from l0_draft_engine.pipeline import (
    AudioTrack,
    EngineError,
    SegmentationConfig,
    prepare_track,
    segment_track,
)


SAMPLE_RATE = 1_000
CONFIG = SegmentationConfig(
    frame_ms=10,
    threshold_margin_db=8.0,
    threshold_min_dbfs=-60.0,
    threshold_max_dbfs=-36.0,
    bridge_gap_ms=10,
    minimum_activity_ms=20,
    coarse_silence_ms=1_000,
)


def pcm(*runs: tuple[int, int]) -> bytes:
    samples = array("h")
    for duration_ms, amplitude in runs:
        samples.extend([amplitude] * duration_ms)
    return samples.tobytes()


def track_for(audio: bytes) -> AudioTrack:
    return AudioTrack(
        lane="speaker-1",
        source_path="source.wav",
        derived_path="denoised.wav",
        sample_rate=SAMPLE_RATE,
        frame_count=len(audio) // 2,
        source_sha256="source",
        pcm_sha256="denoised",
    )


def s2_ranges(audio: bytes) -> list[tuple[int, int]]:
    coarse, _, _ = segment_track(track_for(audio), audio, CONFIG)
    return [(segment.start_sample, segment.end_sample) for segment in coarse]


def test_s2_keeps_silence_shorter_than_one_second_inside_segment() -> None:
    audio = pcm((200, 0), (300, 2_000), (990, 0), (300, 2_000), (200, 0))

    assert s2_ranges(audio) == [(200, 1_790)]


def test_s2_splits_on_exactly_one_second_of_silence() -> None:
    audio = pcm((200, 0), (300, 2_000), (1_000, 0), (300, 2_000), (200, 0))

    assert s2_ranges(audio) == [(200, 500), (1_500, 1_800)]


def test_s2_ignores_short_noise_impulses_on_denoised_lane() -> None:
    audio = pcm((200, 0), (10, 2_000), (200, 0))

    with pytest.raises(EngineError, match="no speech-like activity"):
        segment_track(track_for(audio), audio, CONFIG)


@pytest.mark.parametrize("mode", ["raw", "afftdn"])
def test_preprocessing_preserves_exact_resampled_frame_count(tmp_path, mode: str) -> None:
    source_rate = 48_000
    source_frame_count = round(2.035 * source_rate)
    samples = array(
        "h",
        (
            round(4_000 * math.sin(2 * math.pi * 220 * index / source_rate))
            for index in range(source_frame_count)
        ),
    )
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(source_rate)
        audio.writeframes(samples.tobytes())

    track, processed = prepare_track("speaker-1", source, tmp_path / "prepared", mode)

    expected_frames = round(source_frame_count * 16_000 / source_rate)
    assert track.frame_count == expected_frames
    assert len(processed) == expected_frames * 2
