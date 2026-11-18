"""Capability-aware, least-loaded worker selection.

The policy is intentionally simple: filter to healthy, available workers that
support the required capabilities, then prefer the lowest in-flight/capacity
ratio. A rotating tie-break prevents two equally loaded workers from always
receiving requests in the same order.
"""

from __future__ import annotations

import logging
from typing import Iterable

from app.services.registry import WorkerRecord, WorkerRegistry, WorkerState
from app.workers.base import CHAT_CAPABILITY

logger = logging.getLogger(__name__)


class NoEligibleWorkersError(RuntimeError):
    """No registered worker can serve the request."""

    def __init__(
        self,
        *,
        required_capabilities: Iterable[str],
        reason: str,
    ) -> None:
        self.required_capabilities = frozenset(required_capabilities)
        self.reason = reason
        super().__init__(reason)


class WorkerScheduler:
    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry
        self._round_robin = 0

    @property
    def registry(self) -> WorkerRegistry:
        return self._registry

    def _diagnose(
        self,
        required: frozenset[str],
        exclude: frozenset[str],
    ) -> str:
        records = self._registry.records()
        if not records:
            return "no workers are registered"
        if exclude and all(record.worker_id in exclude for record in records):
            return "all workers were excluded after failures"
        capabilities = ", ".join(sorted(required))
        if not any(required.issubset(record.metadata.capabilities) for record in records):
            return f"no worker supports required capabilities: {capabilities}"
        unhealthy = [
            record.worker_id
            for record in records
            if record.state == WorkerState.UNHEALTHY
            or not record.available
            or not record.worker.is_available
        ]
        if unhealthy:
            return (
                "no healthy, available worker supports required capabilities "
                f"{capabilities}; unavailable: {', '.join(sorted(unhealthy))}"
            )
        return f"no eligible worker for capabilities: {capabilities}"

    def select(
        self,
        required_capabilities: Iterable[str] = (CHAT_CAPABILITY,),
        *,
        exclude: Iterable[str] = (),
    ) -> WorkerRecord:
        required = frozenset(required_capabilities)
        excluded = frozenset(exclude)
        eligible = self._registry.eligible(required, excluded)
        if not eligible:
            raise NoEligibleWorkersError(
                required_capabilities=required,
                reason=self._diagnose(required, excluded),
            )

        self._round_robin += 1
        start = self._round_robin % len(eligible)
        rotated = eligible[start:] + eligible[:start]

        def load(record: WorkerRecord) -> tuple[float, int]:
            capacity = max(1, record.metadata.max_concurrency)
            return (record.in_flight / capacity, record.in_flight)

        selected = min(rotated, key=load)
        self._registry.reserve(selected.worker_id)
        logger.debug(
            "Selected worker %s (provider=%s, in_flight=%d/%d, capabilities=%s)",
            selected.worker_id,
            selected.metadata.provider,
            selected.in_flight,
            selected.metadata.max_concurrency,
            sorted(required),
        )
        return selected

    def release(self, record: WorkerRecord) -> None:
        self._registry.release(record.worker_id)
