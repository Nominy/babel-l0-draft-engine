from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
import time
from typing import Callable, Literal


QueueStatusName = Literal["queued", "running", "completed"]


class DuplicateRequestIdError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QueueStatus:
    request_id: str
    status: QueueStatusName
    position: int
    queued_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "requestId": self.request_id,
            "status": self.status,
            "position": self.position,
            "queuedCount": self.queued_count,
        }


@dataclass(eq=False, slots=True)
class QueueTicket:
    request_id: str
    ready: asyncio.Event


class InferenceQueue:
    """Event-loop-local FIFO gate with observable request state."""

    def __init__(
        self,
        *,
        completed_ttl_seconds: float = 45.0,
        max_completed: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if completed_ttl_seconds <= 0:
            raise ValueError("completed_ttl_seconds must be positive")
        if max_completed <= 0:
            raise ValueError("max_completed must be positive")
        self._completed_ttl_seconds = completed_ttl_seconds
        self._max_completed = max_completed
        self._clock = clock
        self._running: QueueTicket | None = None
        self._waiting: deque[QueueTicket] = deque()
        self._active: dict[str, QueueTicket] = {}
        self._completed: OrderedDict[str, float] = OrderedDict()

    def register(self, request_id: str) -> QueueTicket:
        self._prune_completed()
        if request_id in self._active or request_id in self._completed:
            raise DuplicateRequestIdError(request_id)

        ticket = QueueTicket(request_id=request_id, ready=asyncio.Event())
        self._active[request_id] = ticket
        if self._running is None:
            self._running = ticket
            ticket.ready.set()
        else:
            self._waiting.append(ticket)
        return ticket

    def abandon(self, ticket: QueueTicket) -> None:
        """Remove a request that has not started inference."""
        if self._active.get(ticket.request_id) is not ticket:
            return
        if self._running is ticket:
            self._active.pop(ticket.request_id, None)
            self._running = None
            self._promote_next()
            return
        try:
            self._waiting.remove(ticket)
        except ValueError:
            return
        self._active.pop(ticket.request_id, None)

    def complete(self, ticket: QueueTicket) -> None:
        """Release a running request and retain its terminal state briefly."""
        if self._running is not ticket:
            return
        self._active.pop(ticket.request_id, None)
        self._running = None
        self._completed[ticket.request_id] = self._clock() + self._completed_ttl_seconds
        self._completed.move_to_end(ticket.request_id)
        self._prune_completed()
        self._promote_next()

    def status(self, request_id: str) -> QueueStatus | None:
        self._prune_completed()
        queued_count = len(self._waiting)
        if self._running is not None and self._running.request_id == request_id:
            return QueueStatus(request_id, "running", 0, queued_count)
        ticket = self._active.get(request_id)
        if ticket is not None:
            for position, waiting_ticket in enumerate(self._waiting, start=1):
                if waiting_ticket is ticket:
                    return QueueStatus(request_id, "queued", position, queued_count)
            return None
        if request_id in self._completed:
            return QueueStatus(request_id, "completed", 0, queued_count)
        return None

    def _promote_next(self) -> None:
        while self._waiting:
            ticket = self._waiting.popleft()
            if self._active.get(ticket.request_id) is ticket:
                self._running = ticket
                ticket.ready.set()
                return

    def _prune_completed(self) -> None:
        now = self._clock()
        while self._completed:
            _, expires_at = next(iter(self._completed.items()))
            if expires_at > now and len(self._completed) <= self._max_completed:
                break
            self._completed.popitem(last=False)
