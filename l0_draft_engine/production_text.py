from __future__ import annotations

import unicodedata
from collections.abc import Sequence

PUNCTUATION_BY_LABEL = {
    "O": "",
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "COLON": ":",
    "SEMICOLON": ";",
    "HYPHEN_JOIN": "-",
    "DASH_SINGLE": "-",
    "DASH_DOUBLE": "--",
    "ELLIPSIS": "...",
    "DASH": "-",
}
TERMINAL_LABELS = frozenset({"PERIOD", "QUESTION", "EXCLAMATION"})


class ProductionTextError(ValueError):
    """Raised when formatting would violate the immutable-token contract."""


def lexical_signature(words: Sequence[str]) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFKC", word).casefold() for word in words)


def capitalize_token(word: str) -> str:
    for index, character in enumerate(word):
        upper = character.upper()
        if upper != character:
            return word[:index] + upper + word[index + 1 :]
        if character.lower() != character:
            return word
    return word




def apply_boundary_labels(
    words: Sequence[str],
    labels: Sequence[str],
    *,
    capitalize: bool = True,
    sentence_start: bool = True,
) -> tuple[str, bool]:
    """Apply boundary labels without inserting, deleting, reordering, or rewriting words."""
    if len(words) != len(labels):
        raise ProductionTextError("punctuation label count does not match word count")
    if not words:
        return "", sentence_start
    before = lexical_signature(words)
    next_sentence_start = sentence_start
    displayed_words: list[str] = []
    output = ""
    previous_label = ""
    for word, label in zip(words, labels):
        if not isinstance(word, str) or not word or word.isspace():
            raise ProductionTextError("words must be non-empty lexical strings")
        punctuation = PUNCTUATION_BY_LABEL.get(label)
        if punctuation is None:
            raise ProductionTextError(f"unsupported punctuation label: {label}")
        displayed = (
            capitalize_token(word) if capitalize and next_sentence_start else word
        )
        displayed_words.append(displayed)
        separator = ""
        if output:
            separator = "" if previous_label == "HYPHEN_JOIN" else " "
        output += separator + displayed + punctuation
        previous_label = label
        next_sentence_start = label in TERMINAL_LABELS
    if lexical_signature(displayed_words) != before:
        raise ProductionTextError("capitalization changed lexical token identity")
    return output, next_sentence_start


def format_lanes(
    rows: Sequence[Sequence[str]],
    row_labels: Sequence[Sequence[str]],
) -> list[str]:
    if len(rows) != len(row_labels):
        raise ProductionTextError("row label groups do not match rows")
    result: list[str] = []
    sentence_start = True
    for words, labels in zip(rows, row_labels):
        text, sentence_start = apply_boundary_labels(
            words,
            labels,
            sentence_start=sentence_start,
        )
        result.append(text)
    return result
