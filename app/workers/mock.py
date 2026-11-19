"""Deterministic mock worker for local development and integration tests.

The mock worker implements the same contract as local vLLM and OpenRouter,
requires no GPU, network, model download or API key, and can simulate latency,
unhealthy state and failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, AsyncIterator

from app.workers.base import (
    DEFAULT_CAPABILITIES,
    BaseWorker,
    WorkerMetadata,
    WorkerProviderError,
    WorkerRateLimitError,
    WorkerStream,
    WorkerTimeoutError,
    WorkerUnavailableError,
    WorkerError,
)

FAULT_MODES = {"error", "timeout", "unavailable", "rate_limit"}


def _canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _prompt_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
    return " ".join(texts).strip()


class MockWorker(BaseWorker):
    """Deterministic OpenAI-compatible worker with controllable faults."""

    def __init__(
        self,
        *,
        worker_id: str = "mock-worker",
        model: str = "mock-qwen",
        capabilities: frozenset[str] = DEFAULT_CAPABILITIES,
        max_concurrency: int = 4,
        latency_ms: float = 0.0,
        health_latency_ms: float = 0.0,
        healthy: bool = True,
        available: bool = True,
    ) -> None:
        self._metadata = WorkerMetadata(
            worker_id=worker_id,
            provider="mock",
            model=model,
            capabilities=capabilities,
            max_concurrency=max_concurrency,
            description="Deterministic mock worker; no external dependencies.",
            routing_categories=frozenset({"coding", "reasoning", "general"}),
        )
        self.latency_ms = max(0.0, latency_ms)
        self.health_latency_ms = max(0.0, health_latency_ms)
        self._healthy = healthy
        self._available = available
        self._fault_mode: str | None = None
        self._fault_remaining = 0
        self.requests_started = 0
        self.requests_succeeded = 0
        self.requests_failed = 0

    @property
    def metadata(self) -> WorkerMetadata:
        return self._metadata

    @property
    def is_available(self) -> bool:
        return self._available

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    def set_available(self, available: bool) -> None:
        self._available = available

    def set_latency_ms(self, latency_ms: float) -> None:
        self.latency_ms = max(0.0, latency_ms)

    def fail_next(self, count: int = 1, mode: str = "error") -> None:
        if mode not in FAULT_MODES:
            raise ValueError(f"mode must be one of {sorted(FAULT_MODES)}")
        self._fault_mode = mode
        self._fault_remaining = max(0, count)

    def clear_faults(self) -> None:
        """Cancel any pending injected failures."""
        self._fault_remaining = 0
        self._fault_mode = None

    def _maybe_fail(self) -> None:
        if self._fault_remaining <= 0:
            return
        self._fault_remaining -= 1
        mode = self._fault_mode
        if mode == "timeout":
            raise WorkerTimeoutError(
                f"worker {self._metadata.worker_id} simulated timeout",
                worker_id=self._metadata.worker_id,
            )
        if mode == "unavailable":
            raise WorkerUnavailableError(
                f"worker {self._metadata.worker_id} simulated connection failure",
                worker_id=self._metadata.worker_id,
            )
        if mode == "rate_limit":
            raise WorkerRateLimitError(
                f"worker {self._metadata.worker_id} simulated rate limit",
                worker_id=self._metadata.worker_id,
                retry_after_seconds=1.0,
            )
        raise WorkerProviderError(
            f"worker {self._metadata.worker_id} simulated provider error",
            worker_id=self._metadata.worker_id,
            status_code=500,
        )

    async def health_check(self) -> bool:
        if self.health_latency_ms:
            await asyncio.sleep(self.health_latency_ms / 1000.0)
        return self._healthy

    def _completion_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        canonical = _canonical_payload(payload)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        prompt = _prompt_text(payload)
        content = f"Mock response from {self._metadata.worker_id}: {prompt}"
        return {
            "id": f"chatcmpl-mock-{digest}",
            "object": "chat.completion",
            "created": 0,
            "model": self._metadata.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, len(prompt.split())),
                "completion_tokens": max(1, len(content.split())),
                "total_tokens": max(1, len(prompt.split()))
                + max(1, len(content.split())),
            },
        }

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests_started += 1
        try:
            self._maybe_fail()
            if self.latency_ms:
                await asyncio.sleep(self.latency_ms / 1000.0)
            body = self._completion_body(payload)
        except WorkerError:
            self.requests_failed += 1
            raise
        self.requests_succeeded += 1
        return body

    async def open_stream(self, payload: dict[str, Any]) -> WorkerStream:
        self.requests_started += 1
        body = self._completion_body(payload)
        try:
            self._maybe_fail()
        except WorkerError:
            self.requests_failed += 1
            raise

        async def iterator() -> AsyncIterator[bytes]:
            try:
                if self.latency_ms:
                    await asyncio.sleep(self.latency_ms / 1000.0)
                message = body["choices"][0]["message"]
                content = message["content"]
                midpoint = max(1, len(content) // 2)
                chunks = [
                    {
                        "id": body["id"],
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": self._metadata.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": body["id"],
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": self._metadata.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content[:midpoint]},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": body["id"],
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": self._metadata.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content[midpoint:]},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": body["id"],
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": self._metadata.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                    {
                        "id": body["id"],
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": self._metadata.model,
                        "choices": [],
                        "usage": body["usage"],
                    },
                ]
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode(
                        "utf-8"
                    )
                yield b"data: [DONE]\n\n"
            except asyncio.CancelledError:
                raise

        self.requests_succeeded += 1
        return WorkerStream(
            status_code=200,
            media_type="text/event-stream",
            body_iterator=iterator(),
        )
