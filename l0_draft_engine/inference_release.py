from __future__ import annotations
import json
from pathlib import Path

RELEASE = json.loads(Path(__file__).with_name("inference_release.json").read_text(encoding="utf-8"))
RELEASE_ID = RELEASE["id"]
RELEASE_HEADER = "X-Babel-Inference-Release"
RELEASE_HEADERS = {RELEASE_HEADER: RELEASE_ID}


def validate_punctuated_timing(timing):
    if timing.models.get("release") != RELEASE_ID:
        raise ValueError("Timing belongs to an outdated inference release; recapture audio with the current client")
    for track in timing.tracks:
        labels = track.punctuationLabels
        if labels is None or len(labels) != len(track.tokens) or any(type(label) is not int or not 0 <= label < 7 for label in labels):
            raise ValueError("Current C-denoise timing must include one completed punctuation label per word")


def upgrade_detail():
    return {"code": "update-required", "detail": "Update Babel Gold Drafting and reload the task before running inference.",
            "requiredRelease": RELEASE_ID, "minimumExtensionVersion": RELEASE["minimumExtensionVersion"]}
