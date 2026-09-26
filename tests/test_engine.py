from __future__ import annotations

import asyncio
import builtins
import hashlib
import io
import json
import sys
import tempfile
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import wave
from unittest.mock import patch

import httpx
import pytest

from l0_draft_engine.app import create_app as create_backend_app
from l0_draft_engine.config import Settings
from l0_draft_engine.coordinator import CoordinatorSettings, create_app as create_coordinator_app
from l0_draft_engine.engine import DraftEngine
from l0_draft_engine.gigaam_asr import GigaWord
from l0_draft_engine.schemas import DraftPayload
from l0_draft_engine.pipeline import AudioTrack, Segment


class FakeASR:
    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[Path] = []
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def transcribe(self, path: Path) -> list[GigaWord]:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            lane = "speaker-1" if "speaker-1" in str(path) else "speaker-2"
            surface = "Угу" if lane == "speaker-1" else "Привет"
            self.calls.append(path)
            return [GigaWord(start=0.5, end=0.8, surface=surface)]
        finally:
            with self._counter_lock:
                self.active -= 1


class SpacedWordASR(FakeASR):
    def __init__(self) -> None:
        super().__init__()
        self.input_durations: list[float] = []

    def transcribe(self, path: Path) -> list[GigaWord]:
        with wave.open(str(path), "rb") as audio:
            self.input_durations.append(audio.getnframes() / audio.getframerate())
        self.calls.append(path)
        return [
            GigaWord(start=1.0, end=1.1, surface="второй"),
            GigaWord(start=1.9, end=2.1, surface="за-пределами"),
            GigaWord(start=0.1, end=0.2, surface="Первый"),
        ]

class FakeFormatter:
    def format_rows(self, rows):
        return [" ".join(row) + "." for row in rows]


class ControlledTimer:
    def __init__(self, interval, function, args=None, kwargs=None) -> None:
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.cancelled = False

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        # A canceled timer may already be executing its callback.
        self.function(*self.args, **self.kwargs)


@pytest.fixture(autouse=True)
def idle_timers(monkeypatch):
    timers: list[ControlledTimer] = []

    def create_timer(*args, **kwargs):
        timer = ControlledTimer(*args, **kwargs)
        timers.append(timer)
        return timer

    monkeypatch.setattr("l0_draft_engine.engine.threading.Timer", create_timer)
    return timers


@pytest.fixture
def lifecycle_engine(monkeypatch):
    references: list[weakref.ReferenceType] = []

    def create_asr():
        model = FakeASR()
        model.cycle = model
        references.append(weakref.ref(model))
        return model

    def create_formatter():
        model = FakeFormatter()
        model.cycle = model
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr("l0_draft_engine.engine.prepare_track", fake_prepare)
    monkeypatch.setattr("l0_draft_engine.engine.segment_track", fake_segment)
    monkeypatch.setattr("l0_draft_engine.engine.find_spec", lambda module: None)
    engine = DraftEngine(
        Settings(device="cpu", preprocessing="raw", model_idle_seconds=1),
        asr_factory=create_asr,
        formatter_factory=create_formatter,
    )
    yield engine, references
    engine.close()


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
        pcm_sha256=hashlib.sha256(pcm).hexdigest(),
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
    return [segment], diagnostics


def audio_paths(directory: Path) -> dict[str, Path]:
    return {
        "speaker-1": directory / "first.wav",
        "speaker-2": directory / "second.wav",
    }


def test_s2_segment_asr_preserves_gigaam_surfaces_and_stable_rows() -> None:
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
        timing = engine.transcribe(draft_payload(), paths)
        first = engine.draft(timing)
        second = engine.draft(timing)

    assert [row.id for row in first.rows] == [row.id for row in second.rows]
    assert all(uuid.UUID(row.id).version == 5 for row in first.rows)
    assert {row.lane for row in first.rows} == {"speaker-1", "speaker-2"}
    assert all(row.endSeconds > row.startSeconds for row in first.rows)
    assert next(row.text for row in first.rows if row.lane == "speaker-1") == "Угу."
    assert first.summary["rowCount"] == 2
    assert first.models["asr"]["name"] == "v3_ctc"
    assert len(asr.calls) == 2


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
            return [], {"coarse_segments": 0.0, "active_fraction": 0.0}
        return fake_segment(track, pcm, config)

    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch(
        "l0_draft_engine.engine.segment_track", side_effect=segment_only_speaker
    ):
        timing = engine.transcribe(draft_payload(), audio_paths(Path(temporary)))
        response = engine.draft(timing)

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
        return [segment], {"coarse_segments": 1.0, "active_fraction": 0.5}

    with tempfile.TemporaryDirectory() as temporary, patch(
        "l0_draft_engine.engine.prepare_track", side_effect=fake_prepare
    ), patch(
        "l0_draft_engine.engine.segment_track", side_effect=segmented_window
    ):
        paths = audio_paths(Path(temporary))
        transcription = engine.transcribe(draft_payload(), paths)
        response = engine.draft(transcription)

    assert asr.input_durations == [2.0, 2.0]
    assert [(row.startSeconds, row.endSeconds) for row in response.rows] == [
        (1.0, 3.0),
        (1.0, 3.0),
    ]
    assert [row.text for row in response.rows] == [
        "Первый второй.",
        "Первый второй.",
    ]
    assert [
        [(token.text, token.startSeconds, token.endSeconds) for token in track.tokens]
        for track in transcription.tracks
    ] == [
        [("Первый", 1.1, 1.2), ("второй", 2.0, 2.1)],
        [("Первый", 1.1, 1.2), ("второй", 2.0, 2.1)],
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
        engine.transcribe(draft_payload(), audio_paths(Path(temporary)))

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
        timing = engine.transcribe(payload, audio_paths(Path(temporary)))
        response = engine.draft(timing, payload.options)

    assert [row.id for row in response.rows] == ["live-b", "live-a", "live-c"]
    assert [(row.startSeconds, row.endSeconds) for row in response.rows] == [
        (1.0, 2.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    assert response.rows[0].text == "Мгм."
    assert response.rows[1].text == "Угу."
    assert response.summary["rowCount"] == 3
    assert response.summary["preservedRows"] is True


@pytest.mark.parametrize(
    ("device", "mps_available", "cuda_available", "ready", "l2_dtype"),
    [
        ("mps", True, False, True, "float16"),
        ("mps", False, True, False, "float16"),
        ("cuda", True, False, False, "float16"),
        ("cuda", False, True, True, "float16"),
        ("cpu", False, False, True, "float32"),
    ],
)
def test_health_checks_selected_backend(
    monkeypatch,
    device: str,
    mps_available: bool,
    cuda_available: bool,
    ready: bool,
    l2_dtype: str,
) -> None:
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("l0_draft_engine.engine.find_spec", lambda module: object())
    engine = DraftEngine(Settings(device=device))

    health = engine.health()

    assert health["ok"] is ready
    assert health["models"]["asr"]["dtype"] == "float32"
    assert health["models"]["asr"]["encoder_autocast_dtype"] == (
        None if device == "cpu" else "float16"
    )
    assert health["models"]["l2"]["dtype"] == l2_dtype


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
                pool.submit(engine.transcribe, draft_payload(f"task-{index}"), paths)
                for index in range(2)
            ]
            timings = [future.result() for future in futures]
            responses = [engine.draft(timing) for timing in timings]
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
        [("Угу", 0.5, 0.8)],
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
                return [], {"coarse_segments": 0.0, "active_fraction": 0.0}
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
        timing = engine.transcribe(draft_payload("draft-task"), audio_paths(first_directory))
        with ThreadPoolExecutor(max_workers=2) as pool:
            draft_future = pool.submit(engine.draft, timing)
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


@pytest.mark.parametrize("operation", ["draft", "transcribe"])
def test_idle_expiry_frees_models_and_next_inference_reloads(
    lifecycle_engine, idle_timers, tmp_path: Path, operation: str
) -> None:
    engine, references = lifecycle_engine
    infer = (
        lambda payload, paths: engine.draft(engine.transcribe(payload, paths))
        if operation == "draft" else engine.transcribe(payload, paths)
    )
    first = infer(draft_payload(), audio_paths(tmp_path))
    if operation == "draft":
        assert [row.text for row in first.rows] == ["Угу.", "Привет."]
    else:
        assert [token.text for track in first.tracks for token in track.tokens] == [
            "Угу", "Привет"
        ]
    models = engine.health()["models"]
    assert models["asr"]["loaded"] is True
    assert models["l2"]["loaded"] is (operation == "draft")
    initially_loaded = len(references)

    idle_timers[-1].fire()

    assert all(not model["loaded"] for model in engine.health()["models"].values())
    assert all(reference() is None for reference in references)
    second = infer(draft_payload(), audio_paths(tmp_path))
    if operation == "draft":
        assert second.rows == first.rows
    else:
        assert second.tracks == first.tracks
    assert len(references) == initially_loaded * 2
    assert all(reference() is not None for reference in references[initially_loaded:])


def test_nested_sessions_and_stale_callbacks_cannot_evict_refreshed_models(
    lifecycle_engine, idle_timers, tmp_path: Path
) -> None:
    engine, references = lifecycle_engine
    engine.draft(engine.transcribe(draft_payload(), audio_paths(tmp_path)))
    stale_timer = idle_timers[-1]

    with engine.model_session():
        with engine.model_session():
            stale_timer.fire()
            assert all(reference() is not None for reference in references)
        stale_timer.fire()
        assert all(reference() is not None for reference in references)
        assert stale_timer.cancelled

    current_timer = idle_timers[-1]
    assert current_timer is not stale_timer
    stale_timer.fire()
    assert all(model["loaded"] for model in engine.health()["models"].values())
    current_timer.fire()
    assert all(reference() is None for reference in references)


def test_waiting_session_keeps_models_after_running_session_finishes(
    lifecycle_engine, idle_timers, tmp_path: Path
) -> None:
    engine, references = lifecycle_engine
    engine.draft(engine.transcribe(draft_payload(), audio_paths(tmp_path)))
    stale_timer = idle_timers[-1]
    waiting = threading.Event()
    release = threading.Event()

    def wait_then_transcribe():
        with engine.model_session():
            waiting.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release the waiting session")
            return engine.transcribe(draft_payload(), audio_paths(tmp_path))

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with engine.model_session():
                future = pool.submit(wait_then_transcribe)
                assert waiting.wait(timeout=2)
            stale_timer.fire()
            assert all(reference() is not None for reference in references)
        finally:
            release.set()
        response = future.result(timeout=5)

    assert response.summary["tokenCount"] == 2
    idle_timers[-1].fire()
    assert all(reference() is None for reference in references)


def test_close_defers_release_until_last_session_and_forbids_new_work(
    lifecycle_engine, idle_timers, tmp_path: Path
) -> None:
    engine, references = lifecycle_engine
    engine.draft(engine.transcribe(draft_payload(), audio_paths(tmp_path)))
    stale_timer = idle_timers[-1]

    with engine.model_session():
        with engine.model_session():
            engine.close()
            engine.close()
            stale_timer.fire()
            assert all(reference() is not None for reference in references)
            with pytest.raises(RuntimeError):
                with engine.model_session():
                    pytest.fail("a closed engine accepted new work")
        assert all(reference() is not None for reference in references)

    assert stale_timer.cancelled
    assert all(reference() is None for reference in references)
    assert all(not model["loaded"] for model in engine.health()["models"].values())
    with pytest.raises(RuntimeError):
        engine.transcribe(draft_payload(), audio_paths(tmp_path))


@pytest.mark.parametrize("idle_seconds", [0, 1])
def test_close_releases_idle_models_even_when_automatic_eviction_is_disabled(
    monkeypatch, idle_timers, tmp_path: Path, idle_seconds: int
) -> None:
    monkeypatch.setattr("l0_draft_engine.engine.prepare_track", fake_prepare)
    monkeypatch.setattr("l0_draft_engine.engine.segment_track", fake_segment)
    engine = DraftEngine(
        Settings(device="cpu", model_idle_seconds=idle_seconds),
        asr_factory=FakeASR,
        formatter_factory=FakeFormatter,
    )
    engine.draft(engine.transcribe(draft_payload(), audio_paths(tmp_path)))
    asr = weakref.ref(engine._asr)
    formatter = weakref.ref(engine._formatter)
    if idle_seconds == 0:
        assert idle_timers == []

    engine.close()
    assert asr() is None
    assert formatter() is None
    assert all(timer.cancelled for timer in idle_timers)
    for timer in idle_timers:
        timer.fire()
    with pytest.raises(RuntimeError):
        with engine.model_session():
            pytest.fail("a closed engine accepted new work")


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_idle_eviction_releases_models_before_clearing_selected_allocator(
    lifecycle_engine, idle_timers, monkeypatch, tmp_path: Path, device: str
) -> None:
    engine, references = lifecycle_engine
    engine.settings = Settings(device=device, model_idle_seconds=1)
    cleared = []

    def empty_cache(backend):
        assert all(reference() is None for reference in references)
        cleared.append(backend)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            mps=SimpleNamespace(empty_cache=lambda: empty_cache("mps")),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: empty_cache("cuda"),
            ),
        ),
    )
    engine.draft(engine.transcribe(draft_payload(), audio_paths(tmp_path)))
    idle_timers[-1].fire()
    assert cleared == [device]


def test_evicting_fake_models_does_not_import_torch(
    lifecycle_engine, idle_timers, monkeypatch, tmp_path: Path
) -> None:
    engine, references = lifecycle_engine
    engine.draft(engine.transcribe(draft_payload(), audio_paths(tmp_path)))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    original_import = builtins.__import__
    torch_imports = []

    def track_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            torch_imports.append(name)
            raise AssertionError("eviction must not initialize torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    idle_timers[-1].fire()

    assert all(reference() is None for reference in references)
    assert torch_imports == []


@pytest.mark.anyio
async def test_coordinator_trusted_timing_then_punctuation_never_repeats_asr(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr("l0_draft_engine.engine.prepare_track", fake_prepare)
    monkeypatch.setattr("l0_draft_engine.engine.segment_track", fake_segment)
    asr = FakeASR(delay=0.01)
    engine = DraftEngine(
        Settings(device="cpu", preprocessing="raw"),
        asr_factory=lambda: asr, formatter_factory=FakeFormatter,
    )
    backend = create_backend_app(engine=engine)
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend), base_url="http://trusted")
    coordinator = create_coordinator_app(
        CoordinatorSettings(
            backend_urls=("http://trusted",), max_track_bytes=4096,
            max_request_bytes=12_000, cache_dir=tmp_path / "cache",
        ), client=upstream,
    )
    audio = io.BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 1600)
    task_id = "test/path-with-slashes"
    bearer = {"Authorization": f"Bearer {'x' * 43}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=coordinator), base_url="http://coordinator"
    ) as client:
        timing = await client.post(
            "/v1/transcribe",
            data={"payload": json.dumps(draft_payload(task_id).model_dump())},
            files={
                "audio:1": ("first.wav", audio.getvalue(), "audio/wav"),
                "audio:2": ("second.wav", audio.getvalue(), "audio/wav"),
            },
            headers={**bearer, "X-Babel-Local-Engine": "1"},
        )
        assert timing.status_code == 200, timing.text
        assert len(asr.calls) == 2
        assert [track["segments"][0]["startSample"] for track in timing.json()["tracks"]] == [0, 0]
        assert (await client.post("/v1/timing/lookup", json={"taskId": task_id},
                                  headers=bearer)).json() == timing.json()
        punctuation = await client.post(
            "/v1/draft", json={"taskId": task_id}, headers=bearer,
        )
        assert punctuation.status_code == 200, punctuation.text
        assert [row["text"] for row in punctuation.json()["rows"]] == ["Угу.", "Привет."]
        assert all(row["endSeconds"] > row["startSeconds"] for row in punctuation.json()["rows"])
        assert len(asr.calls) == 2
        repeat = await client.post(
            "/v1/draft", json={"taskId": task_id}, headers=bearer,
        )
        assert repeat.status_code == 200
        assert [row["id"] for row in repeat.json()["rows"]] == [
            row["id"] for row in punctuation.json()["rows"]
        ]
        preserved = {
            "taskId": task_id,
            "options": {"preserveRows": [{
                "rowId": "existing-row", "speakerKey": "speaker-1",
                "startSeconds": 0.0, "endSeconds": 1.0,
                "text": "старое", "index": 0,
            }]},
        }
        changed = await client.post("/v1/draft", json=preserved, headers=bearer)
        assert changed.status_code == 200, changed.text
        assert [(row["id"], row["text"]) for row in changed.json()["rows"]] == [
            ("existing-row", "Угу.")
        ]
        preserved["options"]["preserveRows"][0]["speakerKey"] = "another-lane"
        assert (await client.post("/v1/draft", json=preserved, headers=bearer)).status_code == 422
        assert len(asr.calls) == 2
    await upstream.aclose()
