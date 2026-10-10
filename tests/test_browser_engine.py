from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from l0_draft_engine import browser_engine as module
from l0_draft_engine.browser_engine import BrowserDraftEngine
from l0_draft_engine.config import Settings
from l0_draft_engine.engine import ModelUnavailableError
from l0_draft_engine.enhancement import EnhancementModel, MODEL_ID, SOURCE_GRAPH_SHA256
from l0_draft_engine.inference_release import RELEASE_ID


IDENTITY = "1" * 32
MODEL = {"id": MODEL_ID, "sha256": "2" * 64, "sourceGraphSha256": SOURCE_GRAPH_SHA256}


def rpc_engine(monkeypatch, messages):
    engine = BrowserDraftEngine(Settings())
    pipe = io.StringIO()
    engine._process = SimpleNamespace(stdin=pipe, poll=lambda: None)
    monkeypatch.setattr(engine, "_start", lambda: None)
    stopped = []
    monkeypatch.setattr(engine, "_stop", lambda: stopped.append(True))
    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex=IDENTITY))
    for message in messages:
        engine._responses.put(message)
    return engine, pipe, stopped


def test_rpc_consumes_owned_progress_before_owned_final(monkeypatch):
    events = [{"completedChunks": 0}, {"completedChunks": 1}]
    engine, pipe, stopped = rpc_engine(monkeypatch, [
        *({"id": IDENTITY, "progress": event} for event in events),
        {"id": IDENTITY, "result": {"tracks": ["left", "right"]}},
    ])
    received = []
    assert engine._call("enhance", on_progress=received.append) == {"tracks": ["left", "right"]}
    assert received == events
    assert json.loads(pipe.getvalue())["id"] == IDENTITY
    assert not stopped


@pytest.mark.parametrize("message", [
    {"id": "2" * 32, "progress": {}},
    {"id": "2" * 32, "result": {}},
    {"id": IDENTITY, "progress": {}, "result": {}},
    {"id": IDENTITY},
    [],
    None,
])
def test_rpc_rejects_unowned_or_ambiguous_messages(monkeypatch, message):
    engine, _, stopped = rpc_engine(monkeypatch, [message])
    received = []
    with pytest.raises(ModelUnavailableError):
        engine._call("enhance", on_progress=received.append)
    assert received == []
    assert stopped == [True]


def test_rpc_rejects_progress_for_nonprogress_operation(monkeypatch):
    engine, _, stopped = rpc_engine(monkeypatch, [{"id": IDENTITY, "progress": {}}])
    with pytest.raises(ModelUnavailableError):
        engine._call("draft")
    assert stopped == [True]


def test_rpc_progress_does_not_renew_absolute_deadline(monkeypatch):
    engine, _, stopped = rpc_engine(monkeypatch, [])
    monkeypatch.setenv("BABEL_INFERENCE_RPC_SECONDS", "10")
    clock = iter([100.0, 101.0, 109.0, 111.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    timeouts = []

    def receive(*, timeout):
        timeouts.append(timeout)
        return {"id": IDENTITY, "progress": {"completedChunks": len(timeouts)}}

    engine._responses = SimpleNamespace(get=receive)
    with pytest.raises(ModelUnavailableError, match="timed out"):
        engine._call("enhance", on_progress=lambda _: None)
    assert timeouts == [9.0, 1.0]
    assert stopped == [True]


def test_rpc_invalid_progress_stops_before_final(monkeypatch):
    engine, _, stopped = rpc_engine(monkeypatch, [
        {"id": IDENTITY, "progress": {}}, {"id": IDENTITY, "result": {}},
    ])

    def reject(_):
        raise ValueError("wrong track")

    with pytest.raises(ModelUnavailableError):
        engine._call("enhance", on_progress=reject)
    assert stopped == [True]
    assert engine._responses.qsize() == 1


@pytest.mark.parametrize("capability", [None, {**MODEL, "provider": "cpu"},
                                          {**MODEL, "provider": "webgpu", "sha256": "invalid"}])
def test_prepare_does_not_advertise_unproved_enhancement(monkeypatch, capability):
    engine, _, stopped = rpc_engine(monkeypatch, [{"id": IDENTITY, "result": {
        "ok": True, "release": RELEASE_ID, "provider": "webgpu", "enhancement": capability,
    }}])
    with pytest.raises(ModelUnavailableError):
        engine.prepare()
    assert stopped == [True]
    assert not engine.health()["ok"]
    assert "enhancement" not in engine.health()


def test_prepare_advertises_exact_preflight_capability(monkeypatch):
    engine, _, _ = rpc_engine(monkeypatch, [{"id": IDENTITY, "result": {
        "ok": True, "release": RELEASE_ID, "provider": "webgpu", "enhancement": {**MODEL, "provider": "webgpu"},
    }}])
    health = engine.prepare()
    assert health["ok"]
    assert health["enhancement"] == {**MODEL, "provider": "webgpu"}
    assert engine._enhancement == EnhancementModel.model_validate(MODEL)
