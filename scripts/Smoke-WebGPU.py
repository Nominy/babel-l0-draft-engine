"""Actual two-lane inference plus cached-label rendering after a process restart."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l0_draft_engine.browser_engine import BrowserDraftEngine
from l0_draft_engine.config import Settings
from l0_draft_engine.schemas import DraftPayload

parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
engine = BrowserDraftEngine(Settings.from_env())
try:
    engine.prepare()
    payload = DraftPayload(taskId="release-backend-smoke", tracks=[
        {"lane": "speaker-1", "fieldName": "audio:1"}, {"lane": "speaker-2", "fieldName": "audio:2"}])
    timing = engine.transcribe(payload, {"speaker-1": args.audio, "speaker-2": args.audio})
    draft = engine.draft(timing)
    engine.close()
    engine = BrowserDraftEngine(Settings.from_env())
    restarted = engine.draft(timing)
    if restarted.rows != draft.rows:
        raise RuntimeError("Completed labels did not survive a backend restart")
    report = {"pass": True, "health": engine.health(), "labelsSurviveRestart": True,
              "words": sum(len(track.tokens) for track in timing.tracks), "rows": len(draft.rows),
              "timing": timing.model_dump(exclude_none=True), "draft": draft.model_dump(exclude_none=True)}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("pass", "words", "rows", "labelsSurviveRestart")}))
finally:
    engine.close()
