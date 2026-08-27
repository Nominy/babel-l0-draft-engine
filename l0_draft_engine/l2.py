from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .production_text import PUNCTUATION_BY_LABEL, format_lanes




class L2ModelError(RuntimeError):
    """Raised when the local punctuation model cannot format all input words."""


class PunctuationFormatter:
    """Corpus-trained punctuation inference that preserves every ASR word."""

    def __init__(self, model: str | Path, device: str, chunk_words: int) -> None:
        self.model = str(model)
        self.device = device
        self.chunk_words = chunk_words
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._logit_bias: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._torch, self._tokenizer, self._model
        local_only = Path(self.model).is_dir()
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.model, strip_accents=False, local_files_only=local_only
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.model, local_files_only=local_only
            )
            if self.device == "cuda":
                model = model.half()
            model = model.to(self.device)
            model.eval()
            calibration_path = Path(self.model) / "calibration.json"
            if calibration_path.is_file():
                calibration = json.loads(
                    calibration_path.read_text(encoding="utf-8")
                )
                labels = [
                    str(model.config.id2label[index])
                    for index in range(model.config.num_labels)
                ]
                if calibration.get("labels") != labels:
                    raise L2ModelError(
                        "punctuation calibration labels do not match model"
                    )
                self._logit_bias = torch.tensor(
                    [
                        float(calibration["biases"].get(label, 0.0))
                        for label in labels
                    ],
                    device=self.device,
                    dtype=model.dtype,
                )
        except Exception as exc:
            raise L2ModelError(f"cannot load punctuation model: {exc}") from exc
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        return torch, tokenizer, model

    def _predict_chunk(self, words: Sequence[str]) -> list[str]:
        torch, tokenizer, model = self._load()
        encoded = tokenizer(
            [word.casefold() for word in words],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = encoded.word_ids(batch_index=0)
        represented = {word_id for word_id in word_ids if word_id is not None}
        if represented != set(range(len(words))):
            if len(words) == 1:
                raise L2ModelError("punctuation tokenizer cannot represent an input word")
            middle = len(words) // 2
            return self._predict_chunk(words[:middle]) + self._predict_chunk(words[middle:])
        inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        try:
            with torch.inference_mode():
                logits = model(**inputs).logits[0]
                if self._logit_bias is not None:
                    logits = logits + self._logit_bias
            prediction_ids = logits.argmax(dim=-1).cpu().tolist()
        except Exception as exc:
            raise L2ModelError(f"punctuation inference failed: {exc}") from exc
        labels: list[str] = []
        for word_index in range(len(words)):
            try:
                token_index = word_ids.index(word_index)
            except ValueError as exc:
                raise L2ModelError(
                    f"punctuation tokenizer lost word {word_index}"
                ) from exc
            raw_label = str(
                model.config.id2label[int(prediction_ids[token_index])]
            )
            label = next(
                (
                    candidate
                    for candidate in PUNCTUATION_BY_LABEL
                    if raw_label == candidate
                    or raw_label.endswith(f"_{candidate}")
                ),
                "",
            )
            if not label:
                raise L2ModelError(
                    f"unsupported punctuation label: {raw_label}"
                )
            labels.append(label)
        return labels

    def predict_rows(self, rows: Sequence[Sequence[str]]) -> list[list[str]]:
        if not any(rows):
            raise L2ModelError("cannot format an empty lane")
        result: list[list[str]] = []
        for row in rows:
            row_labels: list[str] = []
            for offset in range(0, len(row), self.chunk_words):
                row_labels.extend(self._predict_chunk(row[offset : offset + self.chunk_words]))
            if len(row_labels) != len(row):
                raise L2ModelError("punctuation label count does not match input words")
            result.append(row_labels)
        return result

    def format_rows(self, rows: Sequence[Sequence[str]]) -> list[str]:
        return format_lanes(rows, self.predict_rows(rows))
