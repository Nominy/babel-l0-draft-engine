from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import soundfile as sf

from l0_draft_engine.gigaam_asr import GigaAMRecognizer, silence_boundaries


def test_silence_boundaries_are_ordered_and_bounded() -> None:
    sample_rate = 100
    waveform = np.ones(5000, dtype=np.float32)
    waveform[2100:2300] = 0
    boundaries = silence_boundaries(waveform, sample_rate)
    assert boundaries[0] == 0
    assert boundaries[-1] == len(waveform)
    assert boundaries == sorted(set(boundaries))
    assert all(right - left <= 2400 for left, right in zip(boundaries, boundaries[1:]))


def test_recognizer_preserves_model_word_surfaces_and_absolute_times(tmp_path) -> None:
    audio_path = tmp_path / "track.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)

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
