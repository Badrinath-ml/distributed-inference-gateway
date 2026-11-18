from __future__ import annotations

import asyncio
import logging
import unittest
from unittest import mock

import httpx

from app.main import app
from app.workers import MockWorker


class FakeSemanticRouter:
    """Embedding-free router so integration tests stay fast and offline."""

    def route(self, prompt: str):
        return {
            "category": "general",
            "scores": {"coding": 0.1, "reasoning": 0.2, "general": 0.8},
            "top_score": 0.8,
            "margin": 0.6,
        }


class MultiWorkerGatewayTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._old_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self._router_patch = mock.patch(
            "app.main.SemanticRouter",
            FakeSemanticRouter,
        )
        self._router_patch.start()
        self._build_patch = None
        self._lifespan = None
        self.client: httpx.AsyncClient | None = None
        self.workers: list[MockWorker] = []

    async def asyncTearDown(self) -> None:
        if self.client is not None:
            await self.client.aclose()
        if self._lifespan is not None:
            await self._lifespan.__aexit__(None, None, None)
        if self._build_patch is not None:
            self._build_patch.stop()
        for worker in self.workers:
            await worker.aclose()
        self._router_patch.stop()
        logging.disable(self._old_disable)

    async def start_gateway(self, workers: list[MockWorker]) -> None:
        self.workers = workers
        self._build_patch = mock.patch(
            "app.main.build_workers",
            return_value=workers,
        )
        self._build_patch.start()
        self._lifespan = app.router.lifespan_context(app)
        await self._lifespan.__aenter__()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=10.0,
        )

    async def post_chat(self, prompt: str = "Explain how HTTP works.", **extra):
        payload = {"messages": [{"role": "user", "content": prompt}]}
        payload.update(extra)
        return await self.client.post("/v1/chat/completions", json=payload)

    async def test_routes_across_multiple_workers(self) -> None:
        workers = [
            MockWorker(worker_id="worker-a", latency_ms=15),
            MockWorker(worker_id="worker-b", latency_ms=15),
        ]
        await self.start_gateway(workers)

        responses = await asyncio.gather(
            *(self.post_chat(f"request {index}") for index in range(6))
        )

        self.assertTrue(all(response.status_code == 200 for response in responses))
        selected = {response.headers["x-worker-id"] for response in responses}
        self.assertEqual(selected, {"worker-a", "worker-b"})

    async def test_unhealthy_worker_is_excluded_then_all_unhealthy_returns_503(
        self,
    ) -> None:
        workers = [
            MockWorker(worker_id="worker-a"),
            MockWorker(worker_id="worker-b"),
        ]
        await self.start_gateway(workers)
        workers[0].set_healthy(False)
        app.state.worker_registry.mark_health("worker-a", False, error="test")

        for _ in range(3):
            response = await self.post_chat()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-worker-id"], "worker-b")

        workers[1].set_healthy(False)
        app.state.worker_registry.mark_health("worker-b", False, error="test")
        response = await self.post_chat()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "No eligible inference worker is available.",
        )

    async def test_all_workers_operator_unavailable_returns_503(self) -> None:
        workers = [
            MockWorker(worker_id="worker-a"),
            MockWorker(worker_id="worker-b"),
        ]
        await self.start_gateway(workers)
        app.state.worker_registry.set_available("worker-a", False)
        app.state.worker_registry.set_available("worker-b", False)

        response = await self.post_chat()

        self.assertEqual(response.status_code, 503)

    async def test_worker_failure_during_inference_retries_once(self) -> None:
        workers = [
            MockWorker(worker_id="worker-a"),
            MockWorker(worker_id="worker-b"),
        ]
        await self.start_gateway(workers)

        # Advance round-robin so the failing worker is selected next.
        warmup = await self.post_chat("warmup")
        self.assertEqual(warmup.headers["x-worker-id"], "worker-b")
        workers[0].fail_next(1, mode="unavailable")

        response = await self.post_chat("retry me")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-worker-id"], "worker-b")
        self.assertEqual(workers[0].requests_started, 1)
        self.assertEqual(workers[1].requests_started, 2)

    async def test_rate_limit_maps_to_429_with_retry_after(self) -> None:
        worker = MockWorker(worker_id="limited")
        await self.start_gateway([worker])
        worker.fail_next(1, mode="rate_limit")

        response = await self.post_chat()

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json()["detail"],
            "Inference worker rate limit exceeded.",
        )
        self.assertEqual(response.headers["retry-after"], "1")

    async def test_streaming_requires_streaming_capability(self) -> None:
        worker = MockWorker(
            worker_id="chat-only",
            capabilities=frozenset({"chat"}),
        )
        await self.start_gateway([worker])

        non_stream = await self.post_chat()
        self.assertEqual(non_stream.status_code, 200)

        stream = await self.post_chat(stream=True)
        self.assertEqual(stream.status_code, 503)

    async def test_worker_health_endpoint_reports_metadata(self) -> None:
        workers = [
            MockWorker(worker_id="worker-a"),
            MockWorker(worker_id="worker-b", healthy=False),
        ]
        await self.start_gateway(workers)
        app.state.worker_registry.mark_health("worker-b", False, error="test")

        response = await self.client.get("/health/workers")

        self.assertEqual(response.status_code, 200)
        snapshot = response.json()["workers"]
        self.assertEqual({item["worker_id"] for item in snapshot}, {"worker-a", "worker-b"})
        by_id = {item["worker_id"]: item for item in snapshot}
        self.assertIn(by_id["worker-a"]["state"], {"unknown", "healthy"})
        self.assertEqual(by_id["worker-b"]["state"], "unhealthy")
        self.assertTrue(by_id["worker-a"]["available"])
        self.assertNotIn("api_key", str(snapshot))
