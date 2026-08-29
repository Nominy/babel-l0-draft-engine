from __future__ import annotations

import pytest

from l0_draft_engine.inference_queue import DuplicateRequestIdError, InferenceQueue


def test_fifo_positions_advance_and_completed_state_expires() -> None:
    now = 100.0
    queue = InferenceQueue(
        completed_ttl_seconds=10,
        max_completed=2,
        clock=lambda: now,
    )
    first = queue.register("first")
    second = queue.register("second")
    third = queue.register("third")

    assert first.ready.is_set()
    assert not second.ready.is_set()
    assert not third.ready.is_set()
    assert queue.status("first").as_dict() == {
        "requestId": "first",
        "status": "running",
        "position": 0,
        "queuedCount": 2,
    }
    assert queue.status("second").as_dict() == {
        "requestId": "second",
        "status": "queued",
        "position": 1,
        "queuedCount": 2,
    }
    assert queue.status("third").as_dict()["position"] == 2

    queue.complete(first)
    assert second.ready.is_set()
    assert queue.status("first").status == "completed"
    assert queue.status("second").status == "running"
    assert queue.status("third").position == 1

    queue.complete(second)
    queue.complete(third)
    assert queue.status("third").status == "completed"
    now = 111.0
    assert queue.status("first") is None
    assert queue.status("second") is None
    assert queue.status("third") is None


def test_abandon_removes_waiter_without_wedging_successor() -> None:
    queue = InferenceQueue()
    first = queue.register("first")
    abandoned = queue.register("abandoned")
    next_ticket = queue.register("next")

    queue.abandon(abandoned)
    assert queue.status("abandoned") is None
    assert queue.status("next").position == 1
    queue.complete(first)
    assert next_ticket.ready.is_set()
    assert queue.status("next").status == "running"


def test_duplicate_active_or_recent_request_ids_are_rejected() -> None:
    queue = InferenceQueue()
    ticket = queue.register("same")
    with pytest.raises(DuplicateRequestIdError):
        queue.register("same")
    queue.complete(ticket)
    with pytest.raises(DuplicateRequestIdError):
        queue.register("same")
