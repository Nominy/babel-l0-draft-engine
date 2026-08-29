from __future__ import annotations

import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import wave
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


class SpacedWordASR(FakeASR):
    def __init__(self) -> None:
        super().__init__()
        self.input_durations: list[float] = []

    def transcribe(self, path: str, **kwargs):
        with wave.open(path, "rb") as audio:
            self.input_durations.append(audio.getnframes() / audio.getframerate())
        self.calls.append((path, kwargs))
        words = [
            SimpleNamespace(start=0.1, end=0.2, word=" Первый"),
            SimpleNamespace(start=1.0, end=1.1, word=" второй"),
        ]
        return iter([SimpleNamespace(words=words)]), SimpleNamespace()

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
    derived_path = derived_dir / f"{lane}-{mode}-full.wav"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 64_000
    pcm = b"\x00\x00" * frame_count
    with wave.open(str(derived_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(pcm)
    track = AudioTrack(
        lane=lane,
        source_path=str(source),
        derived_path=str(derived_path),
        sample_rate=16_000,
        frame_count=frame_count,
        source_sha256=f"source-{lane}",
        pcm_sha256=f"pcm-{lane}",
    )
    return track, pcm


def fake_segment(track: AudioTrack, pcm: bytes, config):
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


def test_s2_segment_asr_grouping_mgm_prior_and_stable_rows() -> None:
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
    ), patch("l0_draft_engine.engine.segment_track", side_effect=fake_segment):
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
        assert path.endswith(".wav")
        assert "-s2-" in path
        assert kwargs["word_timestamps"] is True
        assert kwargs["vad_filter"] is False
        assert kwargs["hotwords"] == settings.hotwords
        assert kwargs["condition_on_previous_text"] is False
    assert formatter.calls[0][0] == ("Мгм",)


def test_silent_lane_is_omitted_from_replacement_rows() -> None:
    asr = FakeASR()
    formatter = FakeFormatter()
    engine = DraftEngine(
        Settings(preprocessing="raw"),
        asr_factory=lambda: asr,
        formatter_factory=lambda: formatter,
    )

    def segment_only_speaker(track: AudioTrack, pcm: bytes, config):
        if track.lane == "speaker-2":
            return [], [], {"coarse_segments": 0.0, "active_fraction": 0.0}
        return fake_segment(track, pcm, config)

    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch(
        "l0_draft_engine.engine.segment_track", side_effect=segment_only_speaker
    ):
        response = engine.draft(draft_payload(), audio_paths(Path(temporary)))

    assert [row.lane for row in response.rows] == ["speaker-1"]
    assert response.summary["rowCount"] == 1
    assert response.summary["segmentation"]["speaker-2"]["coarse_segments"] == 0.0
    assert len(asr.calls) == 1


def test_s2_range_is_the_model_input_and_one_babel_row() -> None:
    asr = SpacedWordASR()
    formatter = FakeFormatter()
    engine = DraftEngine(
        Settings(preprocessing="raw"),
        asr_factory=lambda: asr,
        formatter_factory=lambda: formatter,
    )

    def segmented_window(track: AudioTrack, pcm: bytes, config):
        segment = Segment(
            id=f"{track.lane}-window",
            lane=track.lane,
            stage="S2",
            start_sample=16_000,
            end_sample=48_000,
            sample_rate=track.sample_rate,
        )
        return [segment], [], {"coarse_segments": 1.0, "active_fraction": 0.5}

    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch(
        "l0_draft_engine.engine.segment_track", side_effect=segmented_window
    ):
        response = engine.draft(draft_payload(), audio_paths(Path(temporary)))

    assert asr.input_durations == [2.0, 2.0]
    assert [(row.startSeconds, row.endSeconds) for row in response.rows] == [
        (1.0, 3.0),
        (1.0, 3.0),
    ]
    assert [row.text for row in response.rows] == [
        "Первый второй.",
        "Первый второй.",
    ]
    assert formatter.calls == [
        [("Первый", "второй")],
        [("Первый", "второй")],
    ]


def test_raw_asr_mode_still_segments_from_afftdn_pcm() -> None:
    observed_segmentation_pcm: list[bytes] = []

    def mode_marked_prepare(lane: str, source: Path, derived_dir: Path, mode: str):
        track, _ = fake_prepare(lane, source, derived_dir, mode)
        marker = b"\x11\x00" if mode == "afftdn" else b"\x22\x00"
        return track, marker * track.frame_count

    def record_segment(track: AudioTrack, pcm: bytes, config):
        observed_segmentation_pcm.append(pcm)
        return fake_segment(track, pcm, config)

    engine = DraftEngine(
        Settings(preprocessing="raw"),
        asr_factory=FakeASR,
        formatter_factory=FakeFormatter,
    )
    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=mode_marked_prepare
    ), patch(
        "l0_draft_engine.engine.segment_track", side_effect=record_segment
    ):
        engine.draft(draft_payload(), audio_paths(Path(temporary)))

    assert len(observed_segmentation_pcm) == 2
    assert all(audio.startswith(b"\x11\x00") for audio in observed_segmentation_pcm)


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


def test_transcribe_returns_stable_ordered_word_timing_without_formatter() -> None:
    asr = FakeASR()
    formatter_factory_calls = 0

    def forbidden_formatter():
        nonlocal formatter_factory_calls
        formatter_factory_calls += 1
        raise AssertionError("transcription must not initialize punctuation")

    engine = DraftEngine(
        Settings(preprocessing="raw"),
        asr_factory=lambda: asr,
        formatter_factory=forbidden_formatter,
    )
    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch("l0_draft_engine.engine.segment_track", side_effect=fake_segment):
        paths = audio_paths(Path(temporary))
        first = engine.transcribe(draft_payload(), paths)
        second = engine.transcribe(draft_payload(), paths)

    assert first.taskId == "task-1"
    assert [track.lane for track in first.tracks] == ["speaker-1", "speaker-2"]
    assert [
        [(token.text, token.startSeconds, token.endSeconds) for token in track.tokens]
        for track in first.tracks
    ] == [
        [("Мгм", 0.5, 0.8)],
        [("Привет", 0.5, 0.8)],
    ]
    assert [
        [token.id for token in track.tokens] for track in first.tracks
    ] == [
        [token.id for token in track.tokens] for track in second.tracks
    ]
    assert all(
        uuid.UUID(token.id).version == 5
        and token.startSeconds >= 0
        and token.endSeconds > token.startSeconds
        for track in first.tracks
        for token in track.tokens
    )
    assert first.summary["tokenCount"] == 2
    assert set(first.models) == {"asr"}
    assert formatter_factory_calls == 0


def test_transcribe_keeps_silent_and_fully_empty_lanes_as_empty_tracks() -> None:
    for silent_lanes in ({"speaker-2"}, {"speaker-1", "speaker-2"}):
        asr = FakeASR()

        def segment_with_silence(track: AudioTrack, pcm: bytes, config):
            if track.lane in silent_lanes:
                return [], [], {"coarse_segments": 0.0, "active_fraction": 0.0}
            return fake_segment(track, pcm, config)

        engine = DraftEngine(
            Settings(preprocessing="raw"),
            asr_factory=lambda: asr,
            formatter_factory=lambda: (_ for _ in ()).throw(
                AssertionError("transcription must not initialize punctuation")
            ),
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
        ), patch(
            "l0_draft_engine.engine.segment_track", side_effect=segment_with_silence
        ):
            response = engine.transcribe(
                draft_payload(), audio_paths(Path(temporary))
            )

        tokens_by_lane = {track.lane: track.tokens for track in response.tracks}
        assert all(tokens_by_lane[lane] == [] for lane in silent_lanes)
        assert response.summary["tokenCount"] == 2 - len(silent_lanes)


def test_gpu_inference_is_serialized_across_draft_and_transcribe() -> None:
    asr = FakeASR(delay=0.03)
    engine = DraftEngine(
        Settings(preprocessing="raw"),
        asr_factory=lambda: asr,
        formatter_factory=FakeFormatter,
    )
    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch("l0_draft_engine.engine.segment_track", side_effect=fake_segment):
        root = Path(temporary)
        first_directory = root / "draft"
        second_directory = root / "transcribe"
        first_directory.mkdir()
        second_directory.mkdir()
        with ThreadPoolExecutor(max_workers=2) as pool:
            draft_future = pool.submit(
                engine.draft, draft_payload("draft-task"), audio_paths(first_directory)
            )
            transcribe_future = pool.submit(
                engine.transcribe,
                draft_payload("transcribe-task"),
                audio_paths(second_directory),
            )
            draft_response = draft_future.result()
            transcribe_response = transcribe_future.result()

    assert len(draft_response.rows) == 2
    assert sum(len(track.tokens) for track in transcribe_response.tracks) == 2
    assert asr.max_active == 1
