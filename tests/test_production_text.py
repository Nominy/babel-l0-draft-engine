from __future__ import annotations

import pytest

from l0_draft_engine.production_text import (
    ProductionTextError,
    apply_boundary_labels,
    format_lanes,
)


def test_applies_punctuation_and_deterministic_capitalization() -> None:
    text, sentence_start = apply_boundary_labels(
        ["привет", "как", "дела"],
        ["COMMA", "O", "QUESTION"],
    )
    assert text == "Привет, как дела?"
    assert sentence_start is True


def test_preserves_false_starts_and_raw_filler_surface() -> None:
    text, sentence_start = apply_boundary_labels(
        ["а", "я", "ду", "думаю"],
        ["COMMA", "O", "DASH_SINGLE", "PERIOD"],
    )
    assert text == "А, я ду- думаю."
    assert sentence_start is True

def test_join_label_controls_spacing_without_lexical_heuristics() -> None:
    joined, _ = apply_boundary_labels(
        ["как", "то"],
        ["HYPHEN_JOIN", "PERIOD"],
    )
    interrupted, _ = apply_boundary_labels(
        ["ду", "думаю"],
        ["DASH_SINGLE", "PERIOD"],
    )
    assert joined == "Как-то."
    assert interrupted == "Ду- думаю."


def test_sentence_state_crosses_existing_rows_without_changing_words() -> None:
    assert format_lanes(
        [["ну", "да"], ["это", "верно"]],
        [["COMMA", "O"], ["O", "PERIOD"]],
    ) == ["Ну, да", "это верно."]


def test_rejects_unknown_labels_and_count_mismatch() -> None:
    with pytest.raises(ProductionTextError, match="unsupported"):
        apply_boundary_labels(["слово"], ["INVENTED"])
    with pytest.raises(ProductionTextError, match="count"):
        apply_boundary_labels(["два", "слова"], ["PERIOD"])
