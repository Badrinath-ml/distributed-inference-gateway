"""Cloud inference worker for OpenRouter's OpenAI-compatible API.

The API key is held only in memory and never logged. The worker is registered
only when both an API key and a model identifier are configured.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.workers.base import (
    DEFAULT_CAPABILITIES,
    BaseWorker,
    WorkerMetadata,
    WorkerProviderError,
    WorkerRateLimitError,
    WorkerResponseError,
    WorkerStream,
    WorkerTimeoutError,
    WorkerUnavailableError,
    WorkerError,
)

logger = logging.getLogger(__name__)

PROVIDER = "openrouter"
_MESSAGE_BODY_LIMIT = 300


def _body_excerpt(response: httpx.Response) -> str:
    try:
        return response.text[:_MESSAGE_BODY_LIMIT]
    except Exception:  # pragma: no cover - defensive
        return ""


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _translate_transport_error(exc: Exception, worker_id: str) -> WorkerError:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return WorkerUnavailableError(
            f"worker {worker_id} is unreachable: {type(exc).__name__}",
            worker_id=worker_id,
        )
    if isinstance(exc, httpx.TimeoutException):
        return WorkerTimeoutError(
            f"worker {worker_id} timed out: {type(exc).__name__}",
            worker_id=worker_id,
        )
    # Other transport errors (for example a read failure after the request was
    # sent) are not safe to retry: upstream work may already have happened.
    return WorkerError(
        f"worker {worker_id} failed: {type(exc).__name__}",
        worker_id=worker_id,
    )


def _error_for_status(response: httpx.Response, worker_id: str) -> WorkerError:
    status = response.status_code
    excerpt = _body_excerpt(response)
    message = f"worker {worker_id} returned HTTP {status}"
    if excerpt:
        message = f"{message}: {excerpt}"
    if status == 429:
        return WorkerRateLimitError(
            message,
            worker_id=worker_id,
            retry_after_seconds=_retry_after(response),
        )
    return WorkerProviderError(message, worker_id=worker_id, status_code=status)


def _validate_completion(body: Any, worker_id: str) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise WorkerResponseError(
            f"worker {worker_id} returned a non-object response",
            worker_id=worker_id,
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise WorkerResponseError(
            f"worker {worker_id} response has no choices",
            worker_id=worker_id,
        )
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise WorkerResponseError(
            f"worker {worker_id} response has an invalid first choice",
            worker_id=worker_id,
        )
    return body


class OpenRouterWorker(BaseWorker):
    """OpenRouter cloud worker with explicit timeout/error normalization."""

    def __init__(
        self,
        *,
        worker_id: str,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60.0,
        max_concurrency: int = 4,
        health_timeout_seconds: float = 5.0,
        site_url: str | None = None,
        app_name: str = "distributed-inference-gateway",
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key must not be empty")
        if not model:
            raise ValueError("OpenRouter model identifier must not be empty")

        self._metadata = WorkerMetadata(
            worker_id=worker_id,
            provider=PROVIDER,
            model=model,
            capabilities=DEFAULT_CAPABILITIES,
            max_concurrency=max_concurrency,
            description="OpenRouter cloud worker for coding and complex implementation requests.",
            routing_categories=frozenset({"coding"}),
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._health_timeout_seconds = health_timeout_seconds
        self._site_url = site_url
        self._app_name = app_name
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=None,
                write=30.0,
                pool=10.0,
            ),
            transport=transport,
        )

    @property
    def metadata(self) -> WorkerMetadata:
        return self._metadata

    @property
    def chat_url(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self, *, stream: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        return headers

    def _request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = payload.copy()
        request_payload["model"] = self._metadata.model
        return request_payload

    async def health_check(self) -> bool:
        """Probe the free model-list endpoint; this does not generate tokens."""
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=self._health_timeout_seconds,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                self.chat_url,
                json=self._request_payload(payload),
                headers=self._headers(),
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise _translate_transport_error(exc, self._metadata.worker_id) from exc

        if response.status_code >= 400:
            raise _error_for_status(response, self._metadata.worker_id)

        try:
            body = response.json()
        except ValueError as exc:
            raise WorkerResponseError(
                f"worker {self._metadata.worker_id} returned invalid JSON",
                worker_id=self._metadata.worker_id,
            ) from exc

        body = _validate_completion(body, self._metadata.worker_id)
        # The provider may report the routed variant; expose the configured
        # model so all workers share one internal response shape.
        body["model"] = self._metadata.model
        return body

    async def open_stream(self, payload: dict[str, Any]) -> WorkerStream:
        request = self._client.build_request(
            "POST",
            self.chat_url,
            json=self._request_payload(payload),
            headers=self._headers(stream=True),
        )
        try:
            upstream = await self._client.send(request, stream=True)
        except Exception as exc:
            raise _translate_transport_error(exc, self._metadata.worker_id) from exc

        if upstream.status_code >= 400:
            try:
                await upstream.aread()
            finally:
                await upstream.aclose()
            raise _error_for_status(upstream, self._metadata.worker_id)

        content_type = upstream.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            try:
                body = (await upstream.aread()).decode(errors="replace")[
                    :_MESSAGE_BODY_LIMIT
                ]
            finally:
                await upstream.aclose()
            raise WorkerResponseError(
                f"worker {self._metadata.worker_id} returned non-SSE stream content"
                + (f": {body}" if body else ""),
                worker_id=self._metadata.worker_id,
            )

        async def iterator():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return WorkerStream(
            status_code=upstream.status_code,
            media_type="text/event-stream",
            body_iterator=iterator(),
            close=upstream.aclose,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
