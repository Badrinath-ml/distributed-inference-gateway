from __future__ import annotations

import asyncio
import logging
import threading
import time
import unittest
from unittest import mock

import httpx

from app.config import settings
from app.main import app
from app.workers import WorkerTimeoutError
from tests.stub_upstream import StubServer

ALLOWED_CATEGORIES = {"coding", "reasoning", "general"}


class GatewayTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._old_base_url = settings.vllm_base_url
        self._old_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

        self.stub = StubServer().start()
        settings.vllm_base_url = self.stub.url

        self._lifespan = app.router.lifespan_context(app)
        await self._lifespan.__aenter__()

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=10.0,
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        await self._lifespan.__aexit__(None, None, None)
        self.stub.stop()
        settings.vllm_base_url = self._old_base_url
        logging.disable(self._old_disable)

    async def _post_chat(self, prompt: str = "Implement a binary search tree in Python.", **extra):
        payload = {"messages": [{"role": "user", "content": prompt}]}
        payload.update(extra)
        return await self.client.post("/v1/chat/completions", json=payload)

    async def test_health(self) -> None:
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    async def test_non_streaming_completion_and_header(self) -> None:
        response = await self._post_chat()

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.headers.get("x-semantic-category"),
            ALLOWED_CATEGORIES,
        )
        body = response.json()
        self.assertEqual(body["model"], settings.vllm_model)
        self.assertEqual(
            body["choices"][0]["message"]["content"],
            "Hello from stub.",
        )

    async def test_streaming_sse_and_done(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Implement a binary search tree in Python.",
                }
            ],
            "stream": True,
        }

        async with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(
                response.headers.get("x-semantic-category"),
                ALLOWED_CATEGORIES,
            )
            self.assertTrue(
                response.headers.get("content-type", "").startswith(
                    "text/event-stream"
                )
            )
            lines = [
                line
                async for line in response.aiter_lines()
                if line.startswith("data: ")
            ]

        self.assertIn("data: [DONE]", lines)
        self.assertTrue(any("Hello" in line for line in lines))

    async def test_embedding_runs_in_executor_thread(self) -> None:
        router = app.state.semantic_router
        original_route = router.route
        seen_thread_ids: list[int] = []

        def recording_route(prompt: str):
            seen_thread_ids.append(threading.get_ident())
            return original_route(prompt)

        with mock.patch.object(router, "route", side_effect=recording_route):
            response = await self._post_chat()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(seen_thread_ids)
        self.assertNotEqual(seen_thread_ids[0], threading.get_ident())

    async def test_slow_embedding_does_not_block_health_probe(self) -> None:
        router = app.state.semantic_router

        def slow_route(prompt: str):
            time.sleep(0.25)
            return {
                "category": "general",
                "scores": {"coding": 0.1, "reasoning": 0.2, "general": 0.8},
                "top_score": 0.8,
                "margin": 0.6,
            }

        with mock.patch.object(router, "route", side_effect=slow_route):
            chat_task = asyncio.create_task(self._post_chat())
            await asyncio.sleep(0.05)

            started = time.perf_counter()
            health_response = await self.client.get("/health")
            health_elapsed = time.perf_counter() - started

            chat_response = await chat_task

        self.assertEqual(health_response.status_code, 200)
        self.assertLess(
            health_elapsed,
            0.15,
            f"health probe was blocked for {health_elapsed:.3f}s by embedding",
        )
        self.assertEqual(chat_response.status_code, 200)

    async def test_worker_health_endpoint(self) -> None:
        response = await self.client.get("/health/workers")

        self.assertEqual(response.status_code, 200)
        workers = response.json()["workers"]
        self.assertEqual(len(workers), 1)
        worker = workers[0]
        self.assertEqual(worker["worker_id"], settings.worker1_id)
        self.assertEqual(worker["provider"], "vllm")
        self.assertEqual(worker["model"], settings.vllm_model)
        self.assertIn("chat", worker["capabilities"])
        self.assertIn("streaming", worker["capabilities"])
        self.assertTrue(worker["available"])
        self.assertNotEqual(worker["state"], "unhealthy")

    async def test_empty_messages_returns_400(self) -> None:
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": []},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "No textual user message found.",
        )

    async def test_upstream_error_maps_to_502(self) -> None:
        response = await self._post_chat(simulate_error=True)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Inference gateway request failed.",
        )

    async def test_timeout_error_maps_to_502(self) -> None:
        record = app.state.worker_registry.records()[0]
        with mock.patch.object(
            record.worker,
            "chat_completion",
            side_effect=WorkerTimeoutError(
                "simulated timeout",
                worker_id=record.worker_id,
            ),
        ):
            response = await self._post_chat()

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "Inference gateway request failed.",
        )

    async def test_concurrent_streaming_requests(self) -> None:
        async def one_request(index: int):
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Implement binary search variant {index} in Python.",
                    }
                ],
                "stream": True,
            }
            async with self.client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                lines = [
                    line
                    async for line in response.aiter_lines()
                    if line.startswith("data: ")
                ]
            return {
                "status": response.status_code,
                "category": response.headers.get("x-semantic-category"),
                "lines": lines,
            }

        results = await asyncio.gather(*(one_request(index) for index in range(4)))

        self.assertEqual(len(results), 4)
        for result in results:
            self.assertEqual(result["status"], 200)
            self.assertIn(result["category"], ALLOWED_CATEGORIES)
            self.assertIn("data: [DONE]", result["lines"])


class LifespanTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_router_initialized_once_per_lifespan(self) -> None:
        instances = []

        class CountingRouter:
            def __init__(self) -> None:
                instances.append(self)

            def route(self, prompt: str):
                return {
                    "category": "general",
                    "scores": {
                        "coding": 0.1,
                        "reasoning": 0.2,
                        "general": 0.8,
                    },
                    "top_score": 0.8,
                    "margin": 0.6,
                }

        old_base_url = settings.vllm_base_url
        old_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        stub = StubServer().start()
        settings.vllm_base_url = stub.url

        try:
            with mock.patch("app.main.SemanticRouter", CountingRouter):
                lifespan = app.router.lifespan_context(app)
                await lifespan.__aenter__()
                try:
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                        timeout=10.0,
                    ) as client:
                        for _ in range(3):
                            response = await client.post(
                                "/v1/chat/completions",
                                json={
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "Explain how HTTP works.",
                                        }
                                    ]
                                },
                            )
                            self.assertEqual(response.status_code, 200)
                finally:
                    await lifespan.__aexit__(None, None, None)
        finally:
            stub.stop()
            settings.vllm_base_url = old_base_url
            logging.disable(old_disable)

        self.assertEqual(len(instances), 1)

    async def test_embedding_executor_shutdown_after_lifespan(self) -> None:
        class NoopRouter:
            def route(self, prompt: str):
                return {
                    "category": "general",
                    "scores": {"coding": 0.1, "reasoning": 0.2, "general": 0.8},
                    "top_score": 0.8,
                    "margin": 0.6,
                }

        old_base_url = settings.vllm_base_url
        old_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        stub = StubServer().start()
        settings.vllm_base_url = stub.url
        executor = None

        try:
            with mock.patch("app.main.SemanticRouter", NoopRouter):
                lifespan = app.router.lifespan_context(app)
                await lifespan.__aenter__()
                try:
                    executor = app.state.embedding_executor
                    self.assertFalse(executor.is_shutdown)
                    self.assertEqual(executor.max_workers, 1)
                    await executor.run(lambda: None)
                    self.assertTrue(executor.active_thread_names())
                finally:
                    await lifespan.__aexit__(None, None, None)
        finally:
            stub.stop()
            settings.vllm_base_url = old_base_url
            logging.disable(old_disable)

        self.assertIsNotNone(executor)
        self.assertTrue(executor.is_shutdown)
        self.assertEqual(executor.active_thread_names(), [])
