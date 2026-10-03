"""Run before switching traffic to the prepared WebGPU backend."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l0_draft_engine.browser_engine import BrowserDraftEngine
from l0_draft_engine.config import Settings

engine = BrowserDraftEngine(Settings.from_env())
try:
    print(json.dumps(engine.prepare(), indent=2))
finally:
    engine.close()
