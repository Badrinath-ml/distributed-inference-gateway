"""Provider-independent inference worker contract.

Every worker exposes the same small surface to the gateway:

* immutable metadata (identifier, provider, model, capabilities, capacity),
* an asynchronous health check,
* one non-streaming chat completion method returning a normalized
  OpenAI-compatible response body,
* one streaming method returning a :class:`WorkerStream`.

Provider-specific HTTP details stay inside the worker implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

CHAT_CAPABILITY = "chat"
STREAMING_CAPABILITY = "streaming"
DEFAULT_CAPABILITIES = frozenset({CHAT_CAPABILITY, STREAMING_CAPABILITY})


@dataclass(frozen=True, slots=True)
class WorkerMetadata:
    """Static, non-secret worker information used by the registry and router."""

    worker_id: str
    provider: str
    model: str
    capabilities: frozenset[str]
    max_concurrency: int
    description: str = ""
    routing_categories: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if CHAT_CAPABILITY not in self.capabilities:
            raise ValueError(f"worker must support the '{CHAT_CAPABILITY}' capability")

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "provider": self.provider,
            "model": self.model,
            "capabilities": sorted(self.capabilities),
            "max_concurrency": self.max_concurrency,
            "description": self.description,
            "routing_categories": sorted(self.routing_categories) if self.routing_categories else ["*"],
        }


class WorkerStream:
    """An opened upstream response stream plus one-shot close semantics."""

    def __init__(
        self,
        *,
        status_code: int,
        media_type: str,
        body_iterator: AsyncIterator[bytes],
        headers: dict[str, str] | None = None,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.status_code = status_code
        self.media_type = media_type
        self.body_iterator = body_iterator
        self.headers = headers or {}
        self._close = close
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            await self._close()


class BaseWorker(ABC):
    """Common worker interface used by the registry, scheduler and gateway."""

    @property
    @abstractmethod
    def metadata(self) -> WorkerMetadata:
        """Static worker metadata."""

    @property
    def is_available(self) -> bool:
        """Whether the worker is currently accepting new work.

        The registry also keeps an operator-controlled availability flag; both
        must be true for a worker to be selected.
        """

        return True

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` when the worker is healthy.

        Implementations must not raise for ordinary unhealthy conditions. They
        must be safe to cancel when the health monitor enforces a timeout.
        """

    @abstractmethod
    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a normalized OpenAI-compatible non-streaming response body."""

    @abstractmethod
    async def open_stream(self, payload: dict[str, Any]) -> WorkerStream:
        """Open an upstream SSE stream.

        Connection and pre-stream errors must be raised before returning so the
        gateway can still send a predictable HTTP error response.
        """

    async def aclose(self) -> None:
        """Release worker-owned resources. The default implementation is a no-op."""


class WorkerError(Exception):
    """Base class for failures attributable to an inference worker."""

    def __init__(self, message: str, *, worker_id: str | None = None) -> None:
        super().__init__(message)
        self.worker_id = worker_id


class WorkerUnavailableError(WorkerError):
    """The worker could not be reached before processing the request.

    This is the only worker failure considered safe for a single retry on
    another worker: no response bytes and (normally) no duplicate remote work.
    """


class WorkerTimeoutError(WorkerError):
    """The worker did not respond within the configured timeout."""


class WorkerRateLimitError(WorkerError):
    """The provider rejected the request because of rate limiting."""

    def __init__(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message, worker_id=worker_id)
        self.retry_after_seconds = retry_after_seconds


class WorkerProviderError(WorkerError):
    """The provider returned an error status or otherwise rejected the request."""

    def __init__(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, worker_id=worker_id)
        self.status_code = status_code


class WorkerResponseError(WorkerError):
    """The provider response could not be normalized to the internal format."""
