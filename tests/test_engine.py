from __future__ import annotations

import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from l0_draft_engine.config import Settings
from l0_draft_engine.engine import DraftEngine
from l0_draft_engine.schemas import DraftPayload
from l0_draft_engine.pipeline import AudioTrack, Segment


class FakeASR:
    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def transcribe(self, path: str, **kwargs):
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            lane = "speaker-1" if "speaker-1" in path else "speaker-2"
            surface = " Угу," if lane == "speaker-1" else " Привет."
            words = [SimpleNamespace(start=0.5, end=0.8, word=surface)]
            self.calls.append((path, kwargs))
            return iter([SimpleNamespace(words=words)]), SimpleNamespace()
        finally:
            with self._counter_lock:
                self.active -= 1


class FakeFormatter:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, ...]]] = []

    def format_rows(self, rows):
        values = [tuple(row) for row in rows]
        self.calls.append(values)
        return [" ".join(row) + "." for row in values]


def draft_payload(task_id: str = "task-1") -> DraftPayload:
    return DraftPayload.model_validate(
        {
            "taskId": task_id,
            "tracks": [
                {"lane": "speaker-1", "fieldName": "audio:1"},
                {"lane": "speaker-2", "fieldName": "audio:2"},
            ],
        }
    )


def fake_prepare(lane: str, source: Path, derived_dir: Path, mode: str):
    track = AudioTrack(
        lane=lane,
        source_path=str(source),
        derived_path=str(derived_dir / f"{lane}-{mode}-full.wav"),
        sample_rate=16_000,
        frame_count=64_000,
        source_sha256=f"source-{lane}",
        pcm_sha256=f"pcm-{lane}",
    )
    return track, b"\x00\x00" * track.frame_count


def fake_segment(track: AudioTrack, pcm: bytes, config, evidence_pcm=None):
    segment = Segment(
        id=f"{track.lane}-s2",
        lane=track.lane,
        stage="S2",
        start_sample=0,
        end_sample=track.frame_count,
        sample_rate=track.sample_rate,
    )
    diagnostics = {"coarse_segments": 1.0, "active_fraction": 1.0}
    return [segment], [], diagnostics


def audio_paths(directory: Path) -> dict[str, Path]:
    return {
        "speaker-1": directory / "first.wav",
        "speaker-2": directory / "second.wav",
    }


def test_full_lane_asr_s2_grouping_mgm_prior_and_stable_rows() -> None:
    asr = FakeASR()
    formatter = FakeFormatter()
    settings = Settings(preprocessing="raw")
    engine = DraftEngine(
        settings,
        asr_factory=lambda: asr,
        formatter_factory=lambda: formatter,
    )
    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch("l0_draft_engine.engine.segment_track", side_effect=fake_segment), patch(
        "requests.sessions.Session.request",
        side_effect=AssertionError("remote API must never be called"),
    ):
        paths = audio_paths(Path(temporary))
        first = engine.draft(draft_payload(), paths)
        second = engine.draft(draft_payload(), paths)

    assert [row.id for row in first.rows] == [row.id for row in second.rows]
    assert all(uuid.UUID(row.id).version == 5 for row in first.rows)
    assert {row.lane for row in first.rows} == {"speaker-1", "speaker-2"}
    assert all(row.endSeconds > row.startSeconds for row in first.rows)
    assert next(row.text for row in first.rows if row.lane == "speaker-1") == "Мгм."
    assert first.summary["rowCount"] == 2
    assert first.models["asr"]["name"] == "v3_ctc"
    assert len(asr.calls) == 4
    for path, kwargs in asr.calls:
        assert path.endswith("-full.wav")
        assert kwargs["word_timestamps"] is True
        assert kwargs["vad_filter"] is False
        assert kwargs["hotwords"] == settings.hotwords
        assert kwargs["condition_on_previous_text"] is False
    assert formatter.calls[0][0] == ("Мгм",)


def test_preserve_rows_keeps_live_boundaries_ids_and_empty_interval_fallback() -> None:
    asr = FakeASR()
    formatter = FakeFormatter()
    engine = DraftEngine(
        Settings(preprocessing="raw"),
        asr_factory=lambda: asr,
        formatter_factory=lambda: formatter,
    )
    payload = DraftPayload.model_validate(
        {
            "taskId": "preserve-task",
            "tracks": [
                {"lane": "speaker-1", "fieldName": "audio:1"},
                {"lane": "speaker-2", "fieldName": "audio:2"},
            ],
            "options": {
                "preserveRows": [
                    {
                        "rowId": "live-a",
                        "speakerKey": "speaker-1",
                        "startSeconds": 0.0,
                        "endSeconds": 1.0,
                        "text": "старое",
                        "index": 1,
                    },
                    {
                        "rowId": "live-b",
                        "speakerKey": "speaker-1",
                        "startSeconds": 1.0,
                        "endSeconds": 2.0,
                        "text": "Угу.",
                        "index": 0,
                    },
                    {
                        "rowId": "live-c",
                        "speakerKey": "speaker-2",
                        "startSeconds": 0.0,
                        "endSeconds": 1.0,
                        "text": "старое",
                        "index": 2,
                    },
                ]
            },
        }
    )
    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch("l0_draft_engine.engine.segment_track", side_effect=fake_segment):
        response = engine.draft(payload, audio_paths(Path(temporary)))

    assert [row.id for row in response.rows] == ["live-b", "live-a", "live-c"]
    assert [(row.startSeconds, row.endSeconds) for row in response.rows] == [
        (1.0, 2.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    assert response.rows[0].text == "Мгм."
    assert response.rows[1].text == "Мгм."
    assert response.summary["rowCount"] == 3
    assert response.summary["preservedRows"] is True


def test_gigaam_requires_explicit_cpu_setting() -> None:
    assert Settings().device == "cuda"
    assert Settings(device="cpu").device == "cpu"


def test_health_is_not_ready_when_cached_models_are_missing(tmp_path: Path) -> None:
    engine = DraftEngine(
        Settings(
            device="cpu",
            gigaam_model_path=tmp_path / "missing-gigaam",
            punctuation_model_path=tmp_path / "missing-l2",
        )
    )
    health = engine.health()
    assert health["ok"] is False
    assert health["models"]["asr"]["cached"] is False
    assert health["models"]["l2"]["cached"] is False


def test_gpu_inference_is_serialized_across_requests() -> None:
    asr = FakeASR(delay=0.03)
    formatter = FakeFormatter()
    engine = DraftEngine(
        Settings(preprocessing="raw"),
        asr_factory=lambda: asr,
        formatter_factory=lambda: formatter,
    )
    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch("l0_draft_engine.engine.segment_track", side_effect=fake_segment):
        paths = audio_paths(Path(temporary))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(engine.draft, draft_payload(f"task-{index}"), paths)
                for index in range(2)
            ]
            responses = [future.result() for future in futures]
    assert all(len(response.rows) == 2 for response in responses)
    assert asr.max_active == 1
