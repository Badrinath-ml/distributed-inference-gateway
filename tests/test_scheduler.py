from __future__ import annotations

import json
import unittest

from app.services.inference import generate_completion, stream_completion
from app.services.registry import WorkerRegistry
from app.services.scheduler import NoEligibleWorkersError, WorkerScheduler
from app.workers import (
    MockWorker,
    WorkerRateLimitError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)

PAYLOAD = {"messages": [{"role": "user", "content": "hello"}]}


class WorkerSchedulerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = WorkerRegistry()

    def advance_selection(self, scheduler: WorkerScheduler) -> None:
        """Advance round-robin so the next select starts at the first record."""
        record = scheduler.select()
        scheduler.release(record)

    def test_semantic_category_prefers_specialized_worker(self) -> None:
        from app.workers.base import WorkerMetadata
        local = MockWorker(worker_id="local")
        cloud = MockWorker(worker_id="cloud")
        local._metadata = WorkerMetadata(worker_id="local", provider="vllm", model="local-model", capabilities=frozenset({"chat", "streaming"}), max_concurrency=2, routing_categories=frozenset({"general", "reasoning"}))
        cloud._metadata = WorkerMetadata(worker_id="cloud", provider="openrouter", model="gemma", capabilities=frozenset({"chat", "streaming"}), max_concurrency=2, routing_categories=frozenset({"coding"}))
        self.registry.register(local)
        self.registry.register(cloud)
        scheduler = WorkerScheduler(self.registry)
        self.assertEqual(scheduler.select(route_category="coding").worker_id, "cloud")
        self.assertEqual(scheduler.select(route_category="general").worker_id, "local")

    def test_single_worker_selection_and_release(self) -> None:
        self.registry.register(MockWorker(worker_id="only"))
        scheduler = WorkerScheduler(self.registry)

        record = scheduler.select()
        self.assertEqual(record.worker_id, "only")
        self.assertEqual(record.in_flight, 1)
        scheduler.release(record)
        self.assertEqual(record.in_flight, 0)

    def test_multiple_workers_are_balanced(self) -> None:
        self.registry.register(MockWorker(worker_id="a", max_concurrency=4))
        self.registry.register(MockWorker(worker_id="b", max_concurrency=4))
        scheduler = WorkerScheduler(self.registry)

        selected = [scheduler.select().worker_id for _ in range(4)]
        self.assertEqual(selected.count("a"), 2)
        self.assertEqual(selected.count("b"), 2)

    def test_least_loaded_worker_preferred(self) -> None:
        self.registry.register(MockWorker(worker_id="a", max_concurrency=4))
        self.registry.register(MockWorker(worker_id="b", max_concurrency=4))
        scheduler = WorkerScheduler(self.registry)
        self.registry.reserve("a")
        self.registry.reserve("a")

        record = scheduler.select()
        self.assertEqual(record.worker_id, "b")

    def test_capability_mismatch(self) -> None:
        self.registry.register(
            MockWorker(
                worker_id="chat-only",
                capabilities=frozenset({"chat"}),
            )
        )
        scheduler = WorkerScheduler(self.registry)

        with self.assertRaises(NoEligibleWorkersError) as context:
            scheduler.select({"chat", "streaming"})
        self.assertIn("capabilities", str(context.exception))

    def test_unhealthy_worker_excluded(self) -> None:
        self.registry.register(MockWorker(worker_id="sick"))
        self.registry.register(MockWorker(worker_id="healthy"))
        self.registry.mark_health("sick", False, error="test")

        scheduler = WorkerScheduler(self.registry)
        for _ in range(4):
            self.assertEqual(scheduler.select().worker_id, "healthy")

    def test_all_unavailable(self) -> None:
        self.registry.register(MockWorker(worker_id="a"))
        self.registry.set_available("a", False)
        scheduler = WorkerScheduler(self.registry)

        with self.assertRaises(NoEligibleWorkersError) as context:
            scheduler.select()
        self.assertIn("a", str(context.exception))

    def test_no_workers_registered(self) -> None:
        scheduler = WorkerScheduler(self.registry)
        with self.assertRaises(NoEligibleWorkersError):
            scheduler.select()

    def test_unregister_while_in_flight_is_rejected(self) -> None:
        self.registry.register(MockWorker(worker_id="busy"))
        scheduler = WorkerScheduler(self.registry)
        scheduler.select()

        from app.services.registry import WorkerInUseError

        with self.assertRaises(WorkerInUseError):
            self.registry.unregister("busy")


class InferenceFailoverTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.registry = WorkerRegistry()
        self.worker_a = MockWorker(worker_id="a")
        self.worker_b = MockWorker(worker_id="b")
        self.registry.register(self.worker_a)
        self.registry.register(self.worker_b)
        self.scheduler = WorkerScheduler(self.registry)
        # The scheduler's first selection rotates to the second record; advance
        # it so tests that need worker A to fail first are deterministic.
        record = self.scheduler.select()
        self.scheduler.release(record)

    async def test_connection_failure_retries_once_on_another_worker(self) -> None:
        self.worker_a.fail_next(1, mode="unavailable")

        response = await generate_completion(PAYLOAD, "general", self.scheduler)
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-worker-id"], "b")
        self.assertIn("from b", body["choices"][0]["message"]["content"])
        self.assertEqual(self.worker_a.requests_started, 1)
        self.assertEqual(self.worker_b.requests_started, 1)
        self.assertEqual(self.registry.require("a").in_flight, 0)
        self.assertEqual(self.registry.require("b").in_flight, 0)

    async def test_timeout_is_not_retried(self) -> None:
        self.worker_a.fail_next(1, mode="timeout")

        with self.assertRaises(WorkerTimeoutError):
            await generate_completion(PAYLOAD, "general", self.scheduler)

        self.assertEqual(self.worker_b.requests_started, 0)
        self.assertEqual(self.registry.require("a").in_flight, 0)

    async def test_rate_limit_is_not_retried(self) -> None:
        self.worker_a.fail_next(1, mode="rate_limit")

        with self.assertRaises(WorkerRateLimitError):
            await generate_completion(PAYLOAD, "general", self.scheduler)

        self.assertEqual(self.worker_b.requests_started, 0)

    async def test_retry_can_be_disabled(self) -> None:
        self.worker_a.fail_next(1, mode="unavailable")

        with self.assertRaises(WorkerUnavailableError):
            await generate_completion(
                PAYLOAD,
                "general",
                self.scheduler,
                retry_on_connect_error=False,
            )

        self.assertEqual(self.worker_b.requests_started, 0)

    async def test_no_alternative_worker_returns_failure(self) -> None:
        registry = WorkerRegistry()
        worker = MockWorker(worker_id="only")
        registry.register(worker)
        scheduler = WorkerScheduler(registry)
        worker.fail_next(1, mode="unavailable")

        with self.assertRaises(WorkerUnavailableError):
            await generate_completion(PAYLOAD, "general", scheduler)
        self.assertEqual(registry.require("only").in_flight, 0)

    async def test_all_unhealthy_raises_no_eligible(self) -> None:
        self.registry.mark_health("a", False, error="test")
        self.registry.mark_health("b", False, error="test")

        with self.assertRaises(NoEligibleWorkersError):
            await generate_completion(PAYLOAD, "general", self.scheduler)

    async def test_stream_retries_connection_failure_before_open(self) -> None:
        self.worker_a.fail_next(1, mode="unavailable")

        response = await stream_completion(PAYLOAD, "general", self.scheduler)
        payload = b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-worker-id"], "b")
        self.assertIn(b"from b", payload)
        self.assertIn(b"data: [DONE]", payload)
        self.assertEqual(self.registry.require("a").in_flight, 0)
        self.assertEqual(self.registry.require("b").in_flight, 0)

    async def test_stream_timeout_is_not_retried(self) -> None:
        self.worker_a.fail_next(1, mode="timeout")

        with self.assertRaises(WorkerTimeoutError):
            await stream_completion(PAYLOAD, "general", self.scheduler)

        self.assertEqual(self.worker_b.requests_started, 0)
        self.assertEqual(self.registry.require("a").in_flight, 0)

    async def test_stream_releases_reservation_after_completion(self) -> None:
        response = await stream_completion(PAYLOAD, "general", self.scheduler)
        b"".join([chunk async for chunk in response.body_iterator])

        for worker_id in ("a", "b"):
            self.assertEqual(self.registry.require(worker_id).in_flight, 0)
