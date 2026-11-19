from __future__ import annotations

import asyncio
import json
import socket
import time
import unittest

import httpx

from app.workers import (
    LocalVLLMWorker,
    MockWorker,
    OpenRouterWorker,
    WorkerError,
    WorkerProviderError,
    WorkerRateLimitError,
    WorkerResponseError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from tests.stub_upstream import StubServer


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def completion_body(model: str = "qwen/provider-variant") -> dict:
    return {
        "id": "chatcmpl-cloud",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "cloud answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def sse_body(content: str = "cloud stream") -> bytes:
    chunks = [
        {
            "id": "chatcmpl-cloud",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen/provider-variant",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-cloud",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen/provider-variant",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    lines = [
        f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()
        for chunk in chunks
    ]
    lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


class LocalVLLMWorkerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.stub = StubServer().start()
        self.worker = LocalVLLMWorker(
            worker_id="local-test",
            base_url=self.stub.url,
            model="Qwen/Qwen3-1.7B",
            timeout_seconds=2.0,
        )

    async def asyncTearDown(self) -> None:
        await self.worker.aclose()
        self.stub.stop()

    async def test_metadata_and_health(self) -> None:
        self.assertEqual(self.worker.metadata.provider, "vllm")
        self.assertEqual(self.worker.metadata.model, "Qwen/Qwen3-1.7B")
        self.assertTrue(
            {"chat", "streaming"}.issubset(self.worker.metadata.capabilities)
        )
        self.assertTrue(await self.worker.health_check())

    async def test_optional_modal_proxy_token_is_sent_as_bearer(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json=completion_body())

        worker = LocalVLLMWorker(
            worker_id="modal-test",
            base_url="http://modal-test",
            model="Qwen/Qwen3-1.7B",
            api_key="wk-test-id.ws-test-secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            await worker.chat_completion(
                {"messages": [{"role": "user", "content": "hello"}]}
            )
        finally:
            await worker.aclose()

        self.assertEqual(
            seen["authorization"],
            "Bearer wk-test-id.ws-test-secret",
        )

    async def test_disabled_health_probe_does_not_wake_modal_server(self) -> None:
        worker = LocalVLLMWorker(
            worker_id="modal-no-probe",
            base_url="http://modal-test",
            model="Qwen/Qwen3-1.7B",
            health_check_enabled=False,
            transport=httpx.MockTransport(
                lambda request: (_ for _ in ()).throw(AssertionError("unexpected probe"))
            ),
        )
        try:
            self.assertTrue(await worker.health_check())
        finally:
            await worker.aclose()

    async def test_modal_cold_start_503_is_retried_for_chat_and_stream(self) -> None:
        chat_calls = 0

        def chat_handler(request: httpx.Request) -> httpx.Response:
            nonlocal chat_calls
            chat_calls += 1
            if chat_calls == 1:
                return httpx.Response(503, text="container starting")
            return httpx.Response(200, json=completion_body())

        chat_worker = LocalVLLMWorker(
            worker_id="modal-chat-retry",
            base_url="http://modal-test",
            model="Qwen/Qwen3-1.7B",
            cold_start_retries=1,
            cold_start_retry_delay_seconds=0.001,
            transport=httpx.MockTransport(chat_handler),
        )
        try:
            result = await chat_worker.chat_completion(
                {"messages": [{"role": "user", "content": "hello"}]}
            )
            self.assertEqual(result["choices"][0]["message"]["content"], "Hello from stub.")
            self.assertEqual(chat_calls, 2)
        finally:
            await chat_worker.aclose()

        stream_calls = 0

        def stream_handler(request: httpx.Request) -> httpx.Response:
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                return httpx.Response(503, text="container starting")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body(),
            )

        stream_worker = LocalVLLMWorker(
            worker_id="modal-stream-retry",
            base_url="http://modal-test",
            model="Qwen/Qwen3-1.7B",
            cold_start_retries=1,
            cold_start_retry_delay_seconds=0.001,
            transport=httpx.MockTransport(stream_handler),
        )
        try:
            stream = await stream_worker.open_stream(
                {"messages": [{"role": "user", "content": "hello"}], "stream": True}
            )
            try:
                body = b"".join([chunk async for chunk in stream.body_iterator])
            finally:
                await stream.aclose()
            self.assertIn(b"Hello", body)
            self.assertEqual(stream_calls, 2)
        finally:
            await stream_worker.aclose()

    async def test_non_streaming_overrides_model_and_validates_choices(self) -> None:
        body = await self.worker.chat_completion(
            {
                "model": "client-must-not-win",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

        self.assertEqual(body["model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello from stub.")

    async def test_streaming_forwards_sse_and_done(self) -> None:
        stream = await self.worker.open_stream(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        )
        try:
            payload = b"".join([chunk async for chunk in stream.body_iterator])
        finally:
            await stream.aclose()

        self.assertIn(b"Hello", payload)
        self.assertIn(b"data: [DONE]", payload)

    async def test_provider_error_is_normalized(self) -> None:
        with self.assertRaises(WorkerProviderError) as context:
            await self.worker.chat_completion(
                {
                    "messages": [{"role": "user", "content": "fail"}],
                    "simulate_error": True,
                }
            )
        self.assertEqual(context.exception.status_code, 500)

    async def test_timeout_is_normalized(self) -> None:
        worker = LocalVLLMWorker(
            worker_id="slow-local",
            base_url=self.stub.url,
            model="Qwen/Qwen3-1.7B",
            timeout_seconds=0.1,
        )
        try:
            with self.assertRaises(WorkerTimeoutError):
                await worker.chat_completion(
                    {
                        "messages": [{"role": "user", "content": "slow"}],
                        "simulate_slow": True,
                        "slow_seconds": 0.4,
                    }
                )
        finally:
            await worker.aclose()

    async def test_connection_failure_is_retryable_error(self) -> None:
        worker = LocalVLLMWorker(
            worker_id="offline-local",
            base_url=f"http://127.0.0.1:{free_port()}",
            model="Qwen/Qwen3-1.7B",
            timeout_seconds=0.5,
        )
        try:
            with self.assertRaises(WorkerUnavailableError):
                await worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
        finally:
            await worker.aclose()

    async def test_read_failure_is_not_retryable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("connection dropped", request=request)

        worker = LocalVLLMWorker(
            worker_id="dropped-local",
            base_url="http://test",
            model="Qwen/Qwen3-1.7B",
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaises(WorkerError) as context:
                await worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
            self.assertNotIsInstance(context.exception, WorkerUnavailableError)
        finally:
            await worker.aclose()

    async def test_invalid_json_and_rate_limit(self) -> None:
        invalid_worker = LocalVLLMWorker(
            worker_id="invalid-local",
            base_url="http://test",
            model="m",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text="not json")
            ),
        )
        try:
            with self.assertRaises(WorkerResponseError):
                await invalid_worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
        finally:
            await invalid_worker.aclose()

        limited_worker = LocalVLLMWorker(
            worker_id="limited-local",
            base_url="http://test",
            model="m",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    429,
                    headers={"retry-after": "2.5"},
                    text="rate limited",
                )
            ),
        )
        try:
            with self.assertRaises(WorkerRateLimitError) as context:
                await limited_worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
            self.assertEqual(context.exception.retry_after_seconds, 2.5)
        finally:
            await limited_worker.aclose()


class OpenRouterWorkerTestCase(unittest.IsolatedAsyncioTestCase):
    def make_worker(self, handler, **kwargs) -> OpenRouterWorker:
        return OpenRouterWorker(
            worker_id="cloud-test",
            api_key="sk-test-secret",
            model="qwen/qwen3-4b:free",
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    async def test_non_streaming_response_is_normalized(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/v1/chat/completions")
            self.assertEqual(
                request.headers["authorization"],
                "Bearer sk-test-secret",
            )
            return httpx.Response(200, json=completion_body())

        worker = self.make_worker(handler)
        try:
            body = await worker.chat_completion(
                {
                    "model": "client-model",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )
        finally:
            await worker.aclose()

        self.assertEqual(body["model"], "qwen/qwen3-4b:free")
        self.assertEqual(body["choices"][0]["message"]["content"], "cloud answer")

    async def test_streaming_response_is_forwarded(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body(),
            )

        worker = self.make_worker(handler)
        try:
            stream = await worker.open_stream(
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                }
            )
            try:
                payload = b"".join([chunk async for chunk in stream.body_iterator])
            finally:
                await stream.aclose()
        finally:
            await worker.aclose()

        self.assertIn(b"cloud stream", payload)
        self.assertIn(b"data: [DONE]", payload)

    async def test_timeout_rate_limit_provider_and_invalid_response(self) -> None:
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("simulated", request=request)

        worker = self.make_worker(timeout_handler)
        try:
            with self.assertRaises(WorkerTimeoutError):
                await worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
        finally:
            await worker.aclose()

        worker = self.make_worker(
            lambda request: httpx.Response(
                429,
                headers={"retry-after": "1.5"},
                text="slow down",
            )
        )
        try:
            with self.assertRaises(WorkerRateLimitError) as context:
                await worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
            self.assertEqual(context.exception.retry_after_seconds, 1.5)
        finally:
            await worker.aclose()

        worker = self.make_worker(
            lambda request: httpx.Response(503, text="provider unavailable")
        )
        try:
            with self.assertRaises(WorkerProviderError) as context:
                await worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
            self.assertEqual(context.exception.status_code, 503)
            self.assertNotIn("sk-test-secret", str(context.exception))
        finally:
            await worker.aclose()

        worker = self.make_worker(
            lambda request: httpx.Response(200, text="<html>not json</html>")
        )
        try:
            with self.assertRaises(WorkerResponseError):
                await worker.chat_completion(
                    {"messages": [{"role": "user", "content": "hello"}]}
                )
        finally:
            await worker.aclose()

    async def test_non_sse_stream_is_rejected(self) -> None:
        worker = self.make_worker(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json=completion_body(),
            )
        )
        try:
            with self.assertRaises(WorkerResponseError):
                await worker.open_stream(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                )
        finally:
            await worker.aclose()

    async def test_health_check_uses_model_listing(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.method)
            return httpx.Response(200, json={"data": []})

        worker = self.make_worker(handler)
        try:
            self.assertTrue(await worker.health_check())
            self.assertTrue(calls)
        finally:
            await worker.aclose()

        failing = self.make_worker(lambda request: httpx.Response(401, text="no"))
        try:
            self.assertFalse(await failing.health_check())
        finally:
            await failing.aclose()

        def raise_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        unavailable = self.make_worker(raise_handler)
        try:
            self.assertFalse(await unavailable.health_check())
        finally:
            await unavailable.aclose()

    def test_requires_api_key_and_model(self) -> None:
        with self.assertRaises(ValueError):
            OpenRouterWorker(worker_id="x", api_key="", model="qwen/qwen3-4b:free")
        with self.assertRaises(ValueError):
            OpenRouterWorker(worker_id="x", api_key="secret", model="")


class MockWorkerTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_same_payload(self) -> None:
        first = MockWorker(worker_id="mock-a")
        second = MockWorker(worker_id="mock-a")
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        body_one = await first.chat_completion(payload)
        body_two = await second.chat_completion(payload)

        self.assertEqual(body_one, body_two)
        self.assertEqual(body_one["id"], body_two["id"])
        self.assertIn("hello", body_one["choices"][0]["message"]["content"])

    async def test_worker_id_changes_content(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        body_a = await MockWorker(worker_id="mock-a").chat_completion(payload)
        body_b = await MockWorker(worker_id="mock-b").chat_completion(payload)
        self.assertNotEqual(
            body_a["choices"][0]["message"]["content"],
            body_b["choices"][0]["message"]["content"],
        )

    async def test_latency_is_simulated(self) -> None:
        worker = MockWorker(worker_id="slow-mock", latency_ms=60)
        started = time.perf_counter()
        await worker.chat_completion(
            {"messages": [{"role": "user", "content": "hello"}]}
        )
        elapsed = time.perf_counter() - started
        self.assertGreaterEqual(elapsed, 0.05)

    async def test_failure_modes_and_recovery(self) -> None:
        worker = MockWorker(worker_id="failing-mock")
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        for mode, error_type in (
            ("error", WorkerProviderError),
            ("timeout", WorkerTimeoutError),
            ("unavailable", WorkerUnavailableError),
            ("rate_limit", WorkerRateLimitError),
        ):
            worker.fail_next(1, mode=mode)
            with self.subTest(mode=mode):
                with self.assertRaises(error_type):
                    await worker.chat_completion(payload)
                body = await worker.chat_completion(payload)
                self.assertEqual(body["model"], worker.metadata.model)

        worker.set_healthy(False)
        self.assertFalse(await worker.health_check())
        worker.set_healthy(True)
        self.assertTrue(await worker.health_check())

    async def test_streaming_contains_done(self) -> None:
        worker = MockWorker(worker_id="stream-mock")
        stream = await worker.open_stream(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        )
        try:
            payload = b"".join([chunk async for chunk in stream.body_iterator])
        finally:
            await stream.aclose()

        self.assertIn(b"stream-mock", payload)
        self.assertIn(b"data: [DONE]", payload)

    async def test_no_external_dependency(self) -> None:
        worker = MockWorker(worker_id="offline-mock")
        body = await worker.chat_completion(
            {"messages": [{"role": "user", "content": "offline"}]}
        )
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(worker.requests_started, 1)
        self.assertEqual(worker.requests_succeeded, 1)
