from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from l0_draft_engine.gigaam_asr import GigaAMRecognizer, silence_boundaries


def _install_fake_torch(monkeypatch):
    state = {"active": False, "entries": 0}

    class FakeInferenceMode:
        def __enter__(self):
            assert state["active"] is False
            state["active"] = True
            state["entries"] += 1

        def __exit__(self, _exc_type, _exc, _traceback):
            state["active"] = False

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=FakeInferenceMode),
    )
    return state


def test_silence_boundaries_are_ordered_and_bounded() -> None:
    sample_rate = 100
    waveform = np.ones(5000, dtype=np.float32)
    waveform[2100:2300] = 0
    boundaries = silence_boundaries(waveform, sample_rate)
    assert boundaries[0] == 0
    assert boundaries[-1] == len(waveform)
    assert boundaries == sorted(set(boundaries))
    assert all(right - left <= 2400 for left, right in zip(boundaries, boundaries[1:]))


def test_recognizer_preserves_model_word_surfaces_and_absolute_times(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "track.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)
    _install_fake_torch(monkeypatch)

    class FakeModel:
        def transcribe(self, _path: str, *, word_timestamps: bool):
            assert word_timestamps is True
            return SimpleNamespace(
                words=[SimpleNamespace(text="э", start=0.1, end=0.2)]
            )

    recognizer = GigaAMRecognizer(tmp_path / "unused.ckpt")
    recognizer._model = FakeModel()
    words = recognizer.transcribe(audio_path)
    assert [(word.surface, word.start, word.end) for word in words] == [
        ("э", 0.1, 0.2)
    ]


def test_recognizer_enters_inference_mode_for_each_chunk(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "long-track.wav"
    sf.write(audio_path, np.zeros(2500, dtype=np.float32), 100)
    inference = _install_fake_torch(monkeypatch)

    class FakeModel:
        calls = 0

        def transcribe(self, _path: str, *, word_timestamps: bool):
            assert word_timestamps is True
            assert inference["active"] is True
            self.calls += 1
            return SimpleNamespace(words=[])

    model = FakeModel()
    recognizer = GigaAMRecognizer(tmp_path / "unused.ckpt")
    recognizer._model = model
    assert recognizer.transcribe(audio_path) == []
    assert model.calls > 1
    assert inference == {"active": False, "entries": model.calls}
