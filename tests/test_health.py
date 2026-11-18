from __future__ import annotations

import asyncio
import unittest

from app.services.health import HealthMonitor
from app.services.registry import WorkerRegistry, WorkerState
from app.workers import MockWorker


class HealthMonitorTestCase(unittest.IsolatedAsyncioTestCase):
    def make_registry(self) -> WorkerRegistry:
        registry = WorkerRegistry()
        registry.register(MockWorker(worker_id="mock-health"))
        return registry

    async def test_healthy_worker(self) -> None:
        registry = self.make_registry()
        monitor = HealthMonitor(registry, interval_seconds=60, timeout_seconds=0.2)

        results = await monitor.check_once()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].healthy)
        self.assertEqual(registry.require("mock-health").state, WorkerState.HEALTHY)

    async def test_unhealthy_worker(self) -> None:
        registry = self.make_registry()
        registry.require("mock-health").worker.set_healthy(False)
        monitor = HealthMonitor(registry, interval_seconds=60, timeout_seconds=0.2)

        results = await monitor.check_once()

        self.assertFalse(results[0].healthy)
        record = registry.require("mock-health")
        self.assertEqual(record.state, WorkerState.UNHEALTHY)
        self.assertEqual(record.consecutive_failures, 1)
        self.assertIn("returned false", record.last_error or "")
        self.assertEqual(registry.eligible(), [])

    async def test_health_check_timeout(self) -> None:
        registry = WorkerRegistry()
        registry.register(
            MockWorker(
                worker_id="slow-health",
                health_latency_ms=300,
            )
        )
        monitor = HealthMonitor(registry, interval_seconds=60, timeout_seconds=0.05)

        results = await monitor.check_once()

        self.assertFalse(results[0].healthy)
        record = registry.require("slow-health")
        self.assertEqual(record.state, WorkerState.UNHEALTHY)
        self.assertIn("timed out", record.last_error or "")

    async def test_recovery(self) -> None:
        registry = self.make_registry()
        worker = registry.require("mock-health").worker
        monitor = HealthMonitor(registry, interval_seconds=60, timeout_seconds=0.2)

        worker.set_healthy(False)
        await monitor.check_once()
        self.assertEqual(registry.require("mock-health").consecutive_failures, 1)

        worker.set_healthy(True)
        results = await monitor.check_once()

        self.assertTrue(results[0].healthy)
        record = registry.require("mock-health")
        self.assertEqual(record.state, WorkerState.HEALTHY)
        self.assertEqual(record.consecutive_failures, 0)
        self.assertIsNone(record.last_error)

    async def test_unhealthy_worker_excluded_from_eligibility(self) -> None:
        registry = WorkerRegistry()
        registry.register(MockWorker(worker_id="one", healthy=False))
        registry.register(MockWorker(worker_id="two"))
        monitor = HealthMonitor(registry, interval_seconds=60, timeout_seconds=0.2)

        await monitor.check_once()

        eligible_ids = [record.worker_id for record in registry.eligible()]
        self.assertEqual(eligible_ids, ["two"])

    async def test_start_and_stop_cleanly(self) -> None:
        registry = self.make_registry()
        monitor = HealthMonitor(registry, interval_seconds=3600, timeout_seconds=0.2)

        monitor.start()
        await asyncio.sleep(0.05)
        self.assertEqual(registry.require("mock-health").state, WorkerState.HEALTHY)
        task = monitor._task
        self.assertIsNotNone(task)

        await monitor.stop()
        self.assertIsNone(monitor._task)
        self.assertTrue(task is None or task.done())

    async def test_disabled_monitor_does_not_start(self) -> None:
        registry = self.make_registry()
        monitor = HealthMonitor(
            registry,
            interval_seconds=60,
            timeout_seconds=0.2,
            enabled=False,
        )
        monitor.start()
        await asyncio.sleep(0.01)
        self.assertIsNone(monitor._task)
        self.assertEqual(registry.require("mock-health").state, WorkerState.UNKNOWN)
