from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf



class GigaAMError(RuntimeError):
    """Raised when the accepted GigaAM checkpoint cannot produce valid immutable words."""


@dataclass(frozen=True)
class GigaWord:
    start: float
    end: float
    surface: str


def silence_boundaries(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_seconds: float = 22.0,
    maximum_seconds: float = 24.0,
    search_seconds: float = 2.0,
) -> list[int]:
    boundaries = [0]
    cursor = 0
    target_samples = round(target_seconds * sample_rate)
    maximum_samples = round(maximum_seconds * sample_rate)
    search_radius = round(search_seconds * sample_rate)
    energy_window = max(1, round(0.12 * sample_rate))
    while len(waveform) - cursor > maximum_samples:
        target = cursor + target_samples
        left = max(cursor + round(12.0 * sample_rate), target - search_radius)
        right = min(len(waveform) - energy_window, target + search_radius)
        if right <= left:
            cut = min(len(waveform), cursor + maximum_samples)
        else:
            cut = min(
                range(left, right + 1, max(1, energy_window // 2)),
                key=lambda index: float(
                    np.mean(np.square(waveform[index : index + energy_window]))
                ),
            )
        if cut <= cursor:
            raise GigaAMError("audio windowing failed to advance")
        boundaries.append(cut)
        cursor = cut
    boundaries.append(len(waveform))
    return boundaries


class GigaAMRecognizer:
    def __init__(self, model: str | Path, device: str = "cuda") -> None:
        self.model = str(model)
        self.device = device
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import gigaam

            self._model = gigaam.load_model(
                self.model,
                fp16_encoder=False,
                device=self.device,
            )
        except Exception as exc:
            raise GigaAMError(f"cannot load GigaAM checkpoint: {exc}") from exc
        return self._model

    def transcribe(self, audio_path: Path) -> list[GigaWord]:
        model = self._load()
        try:
            waveform, sample_rate = sf.read(
                audio_path, dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise GigaAMError(f"cannot read audio: {exc}") from exc
        mono = waveform.mean(axis=1)
        boundaries = silence_boundaries(mono, sample_rate)
        words: list[GigaWord] = []
        import tempfile

        with tempfile.TemporaryDirectory(prefix="babel-gigaam-") as temporary:
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                chunk_path = Path(temporary) / f"{index:04d}.wav"
                sf.write(
                    chunk_path,
                    mono[start:end],
                    sample_rate,
                    subtype="PCM_16",
                )
                try:
                    result = model.transcribe(
                        str(chunk_path), word_timestamps=True
                    )
                except Exception as exc:
                    raise GigaAMError(
                        f"GigaAM inference failed for chunk {index}: {exc}"
                    ) from exc
                offset = start / sample_rate
                for raw_word in result.words or ():
                    surface = str(raw_word.text).strip()
                    word_start = offset + float(raw_word.start)
                    word_end = offset + float(raw_word.end)
                    if (
                        not surface
                        or not math.isfinite(word_start)
                        or not math.isfinite(word_end)
                        or word_start < 0
                        or word_end <= word_start
                    ):
                        continue
                    words.append(GigaWord(word_start, word_end, surface))
        words.sort(key=lambda word: (word.start, word.end, word.surface))
        return words
