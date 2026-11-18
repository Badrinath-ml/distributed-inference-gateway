from __future__ import annotations

import unittest

from app.services.registry import (
    DuplicateWorkerError,
    UnknownWorkerError,
    WorkerInUseError,
    WorkerRegistry,
    WorkerState,
)
from app.workers import MockWorker


class WorkerRegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = WorkerRegistry()
        self.worker = MockWorker(
            worker_id="worker-a",
            model="mock-a",
            max_concurrency=3,
        )

    def test_register_lookup_and_metadata(self) -> None:
        record = self.registry.register(self.worker)

        self.assertEqual(len(self.registry), 1)
        self.assertIs(self.registry.get("worker-a"), record)
        self.assertIs(self.registry.require("worker-a"), record)
        self.assertEqual(record.metadata.provider, "mock")
        self.assertEqual(record.metadata.model, "mock-a")
        self.assertEqual(record.metadata.max_concurrency, 3)
        self.assertEqual(record.state, WorkerState.UNKNOWN)
        self.assertIn("chat", record.metadata.capabilities)
        self.assertIn("streaming", record.metadata.capabilities)
        self.assertIn("worker-a", [item["worker_id"] for item in self.registry.snapshot()])

    def test_duplicate_registration_is_rejected(self) -> None:
        self.registry.register(self.worker)
        duplicate = MockWorker(worker_id="worker-a")

        with self.assertRaises(DuplicateWorkerError):
            self.registry.register(duplicate)

    def test_unknown_worker_lookup(self) -> None:
        self.assertIsNone(self.registry.get("missing"))
        with self.assertRaises(UnknownWorkerError):
            self.registry.require("missing")

    def test_health_state_and_availability_changes(self) -> None:
        self.registry.register(self.worker)

        record = self.registry.mark_health("worker-a", False, error="boom")
        self.assertEqual(record.state, WorkerState.UNHEALTHY)
        self.assertEqual(record.consecutive_failures, 1)
        self.assertEqual(record.last_error, "boom")
        self.assertIsNotNone(record.last_checked_at)
        self.assertEqual(self.registry.eligible(), [])

        record = self.registry.mark_health("worker-a", True)
        self.assertEqual(record.state, WorkerState.HEALTHY)
        self.assertEqual(record.consecutive_failures, 0)
        self.assertIsNone(record.last_error)
        self.assertEqual(len(self.registry.eligible()), 1)

        self.registry.set_available("worker-a", False)
        self.assertEqual(self.registry.eligible(), [])
        self.assertFalse(self.registry.get("worker-a").as_dict()["available"])

        self.registry.set_available("worker-a", True)
        self.assertEqual(len(self.registry.eligible()), 1)

    def test_worker_exposed_availability_is_respected(self) -> None:
        self.registry.register(self.worker)
        self.worker.set_available(False)
        self.assertEqual(self.registry.eligible(), [])
        self.worker.set_available(True)
        self.assertEqual(len(self.registry.eligible()), 1)

    def test_capability_filtering(self) -> None:
        self.registry.register(self.worker)
        chat_only = MockWorker(
            worker_id="chat-only",
            capabilities=frozenset({"chat"}),
        )
        self.registry.register(chat_only)

        self.assertIn(
            "worker-a",
            [record.worker_id for record in self.registry.eligible({"chat", "streaming"})],
        )
        streaming_ids = [
            record.worker_id
            for record in self.registry.eligible({"chat", "streaming"})
        ]
        self.assertNotIn("chat-only", streaming_ids)
        self.assertIn(
            "chat-only",
            [record.worker_id for record in self.registry.eligible({"chat"})],
        )

    def test_reserve_release_and_unregister(self) -> None:
        self.registry.register(self.worker)
        record = self.registry.reserve("worker-a")
        self.assertEqual(record.in_flight, 1)

        with self.assertRaises(WorkerInUseError):
            self.registry.unregister("worker-a")

        self.registry.release("worker-a")
        self.assertEqual(self.registry.require("worker-a").in_flight, 0)
        self.assertIs(self.registry.unregister("worker-a"), self.worker)
        self.assertEqual(len(self.registry), 0)

    def test_release_never_goes_negative(self) -> None:
        self.registry.register(self.worker)
        self.registry.release("worker-a")
        self.assertEqual(self.registry.require("worker-a").in_flight, 0)
