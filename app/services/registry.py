"""Central worker registry with metadata, health state and capacity counters.

The registry deliberately has no provider-specific knowledge and no database;
the project only needs a single-process, in-memory view of the configured
workers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

from app.workers.base import BaseWorker, CHAT_CAPABILITY, WorkerMetadata


class WorkerState(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class DuplicateWorkerError(ValueError):
    """A worker with the same identifier is already registered."""


class UnknownWorkerError(KeyError):
    """No worker with the requested identifier exists."""


class WorkerInUseError(RuntimeError):
    """A worker cannot be removed while requests are still in flight."""


@dataclass
class WorkerRecord:
    worker: BaseWorker
    state: WorkerState = WorkerState.UNKNOWN
    available: bool = True
    last_checked_at: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    in_flight: int = 0

    @property
    def metadata(self) -> WorkerMetadata:
        return self.worker.metadata

    @property
    def worker_id(self) -> str:
        return self.worker.metadata.worker_id

    def as_dict(self) -> dict[str, object]:
        metadata = self.metadata.as_dict()
        metadata.update(
            {
                "state": self.state.value,
                "available": self.available and self.worker.is_available,
                "in_flight": self.in_flight,
                "last_checked_at": self.last_checked_at,
                "last_error": self.last_error,
                "consecutive_failures": self.consecutive_failures,
            }
        )
        return metadata


class WorkerRegistry:
    """In-memory registry; mutations are safe from multiple threads.

    Selection and health checks run on the asyncio event loop, but a small lock
    keeps registration/removal safe if an embedding or test thread inspects the
    registry concurrently.
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerRecord] = {}
        self._lock = threading.RLock()

    def register(self, worker: BaseWorker) -> WorkerRecord:
        worker_id = worker.metadata.worker_id
        with self._lock:
            if worker_id in self._workers:
                raise DuplicateWorkerError(
                    f"worker '{worker_id}' is already registered"
                )
            record = WorkerRecord(worker=worker)
            self._workers[worker_id] = record
            return record

    def unregister(self, worker_id: str) -> BaseWorker:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                raise UnknownWorkerError(worker_id)
            if record.in_flight > 0:
                raise WorkerInUseError(
                    f"worker '{worker_id}' still has {record.in_flight} in-flight requests"
                )
            del self._workers[worker_id]
            return record.worker

    def get(self, worker_id: str) -> WorkerRecord | None:
        with self._lock:
            return self._workers.get(worker_id)

    def require(self, worker_id: str) -> WorkerRecord:
        record = self.get(worker_id)
        if record is None:
            raise UnknownWorkerError(worker_id)
        return record

    def workers(self) -> list[BaseWorker]:
        with self._lock:
            return [record.worker for record in self._workers.values()]

    def records(self) -> list[WorkerRecord]:
        with self._lock:
            return list(self._workers.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._workers)

    def eligible(
        self,
        required_capabilities: Iterable[str] = (CHAT_CAPABILITY,),
        exclude: Iterable[str] = (),
    ) -> list[WorkerRecord]:
        required = frozenset(required_capabilities)
        excluded = frozenset(exclude)
        eligible: list[WorkerRecord] = []
        for record in self.records():
            if record.worker_id in excluded:
                continue
            if record.state == WorkerState.UNHEALTHY:
                continue
            if not record.available or not record.worker.is_available:
                continue
            if not required.issubset(record.metadata.capabilities):
                continue
            eligible.append(record)
        return eligible

    def mark_health(
        self,
        worker_id: str,
        healthy: bool,
        *,
        error: str | None = None,
    ) -> WorkerRecord | None:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                return None
            record.state = WorkerState.HEALTHY if healthy else WorkerState.UNHEALTHY
            record.last_checked_at = datetime.now(timezone.utc).isoformat()
            record.last_error = None if healthy else error
            if healthy:
                record.consecutive_failures = 0
            else:
                record.consecutive_failures += 1
            return record

    def set_available(self, worker_id: str, available: bool) -> WorkerRecord:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                raise UnknownWorkerError(worker_id)
            record.available = available
            return record

    def reserve(self, worker_id: str) -> WorkerRecord:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is None:
                raise UnknownWorkerError(worker_id)
            record.in_flight += 1
            return record

    def release(self, worker_id: str) -> None:
        with self._lock:
            record = self._workers.get(worker_id)
            if record is not None and record.in_flight > 0:
                record.in_flight -= 1

    def snapshot(self) -> list[dict[str, object]]:
        return [record.as_dict() for record in self.records()]
