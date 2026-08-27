from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SAFE_NAME_RE = re.compile(r"^[\w.:-]+$", re.UNICODE)


class TrackSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    lane: str = Field(min_length=1, max_length=128)
    fieldName: str = Field(min_length=1, max_length=128)

    @field_validator("lane", "fieldName")
    @classmethod
    def safe_name(cls, value: str) -> str:
        if not SAFE_NAME_RE.fullmatch(value):
            raise ValueError("must contain only letters, digits, underscore, dot, colon, or hyphen")
        return value


class PreserveRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rowId: str = Field(min_length=1, max_length=256)
    speakerKey: str = Field(min_length=1, max_length=128)
    startSeconds: float = Field(ge=0)
    endSeconds: float = Field(gt=0)
    text: str = Field(max_length=200_000)
    index: int = Field(ge=0)

    @field_validator("rowId", "speakerKey")
    @classmethod
    def clean_identity(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(ord(character) < 32 for character in cleaned):
            raise ValueError("must be a non-empty identity without control characters")
        return cleaned

    @model_validator(mode="after")
    def positive_interval(self) -> "PreserveRow":
        if not math.isfinite(self.startSeconds) or not math.isfinite(self.endSeconds):
            raise ValueError("row timestamps must be finite")
        if self.endSeconds <= self.startSeconds:
            raise ValueError("preserved row duration must be positive")
        return self


class DraftOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing: Literal["raw", "afftdn"] | None = None
    preserveRows: list[PreserveRow] | None = Field(
        default=None, min_length=1, max_length=10_000
    )


class DraftPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    taskId: str = Field(min_length=1, max_length=256)
    tracks: list[TrackSpec] = Field(min_length=2, max_length=2)
    options: DraftOptions | None = None

    @field_validator("taskId")
    @classmethod
    def valid_task_id(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("must not contain control characters")
        return value

    @model_validator(mode="after")
    def unique_tracks(self) -> "DraftPayload":
        lanes = [track.lane for track in self.tracks]
        fields = [track.fieldName for track in self.tracks]
        if len(set(lanes)) != 2:
            raise ValueError("track lanes must be unique")
        if len(set(fields)) != 2:
            raise ValueError("track fieldName values must be unique")
        preserved = self.options.preserveRows if self.options is not None else None
        if preserved is not None:
            row_ids = [row.rowId for row in preserved]
            indexes = [row.index for row in preserved]
            if len(set(row_ids)) != len(row_ids):
                raise ValueError("preserveRows rowId values must be unique")
            if len(set(indexes)) != len(indexes):
                raise ValueError("preserveRows index values must be unique")
            unknown_lanes = sorted({row.speakerKey for row in preserved} - set(lanes))
            if unknown_lanes:
                raise ValueError(
                    f"preserveRows reference unknown track lanes: {', '.join(unknown_lanes)}"
                )
        return self


class DraftRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    lane: str
    startSeconds: float
    endSeconds: float
    text: str

    @model_validator(mode="after")
    def positive_row(self) -> "DraftRow":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("row duration must be positive")
        if not self.text.strip():
            raise ValueError("row text must not be empty")
        return self


class DraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[DraftRow] = Field(min_length=1)
    summary: dict[str, object]
    models: dict[str, object]
