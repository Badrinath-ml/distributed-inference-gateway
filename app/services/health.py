"""Periodic, timeout-aware health monitoring for registered workers."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.services.registry import WorkerRegistry, WorkerState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthResult:
    worker_id: str
    healthy: bool
    elapsed_ms: float
    previous_state: WorkerState
    error: str | None = None


class HealthMonitor:
    """One background task checks all workers at a bounded interval.

    Each check is wrapped in ``asyncio.wait_for``. Checks for one cycle run
    concurrently, so the number of tasks is bounded by the number of registered
    workers and no task is created per request.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        *,
        interval_seconds: float = 15.0,
        timeout_seconds: float = 3.0,
        enabled: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("health check interval must be positive")
        if timeout_seconds <= 0:
            raise ValueError("health check timeout must be positive")
        self._registry = registry
        self._interval_seconds = interval_seconds
        self._timeout_seconds = timeout_seconds
        self._enabled = enabled
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _check(self, record) -> HealthResult:
        previous = record.state
        started = time.perf_counter()
        error: str | None = None
        try:
            healthy = await asyncio.wait_for(
                record.worker.health_check(),
                timeout=self._timeout_seconds,
            )
            if not healthy:
                error = "health check returned false"
        except TimeoutError:
            healthy = False
            error = f"health check timed out after {self._timeout_seconds:.3f}s"
        except Exception as exc:
            healthy = False
            error = f"{type(exc).__name__}: {exc}"

        elapsed_ms = (time.perf_counter() - started) * 1000
        self._registry.mark_health(
            record.worker_id,
            healthy,
            error=error,
        )

        if healthy and previous != WorkerState.HEALTHY:
            logger.info("Worker %s is healthy", record.worker_id)
        elif not healthy and previous != WorkerState.UNHEALTHY:
            logger.warning(
                "Worker %s is unhealthy: %s",
                record.worker_id,
                error,
            )
        elif not healthy:
            logger.debug(
                "Worker %s remains unhealthy: %s",
                record.worker_id,
                error,
            )

        return HealthResult(
            worker_id=record.worker_id,
            healthy=healthy,
            elapsed_ms=round(elapsed_ms, 4),
            previous_state=previous,
            error=error,
        )

    async def check_once(self) -> list[HealthResult]:
        records = self._registry.records()
        if not records:
            return []
        return list(
            await asyncio.gather(*(self._check(record) for record in records))
        )

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Health check cycle failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_seconds,
                )
            except TimeoutError:
                continue

    def start(self) -> None:
        if not self._enabled:
            logger.info("Worker health monitoring disabled")
            return
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="worker-health-monitor",
        )

    async def stop(self) -> None:
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
