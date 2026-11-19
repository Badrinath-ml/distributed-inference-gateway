"""Worker 1: local vLLM OpenAI-compatible inference worker."""

from __future__ import annotations

import asyncio
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

PROVIDER = "vllm"
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


def _translate_transport_error(
    exc: Exception,
    worker_id: str,
) -> WorkerError:
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


def _error_for_status(
    response: httpx.Response,
    worker_id: str,
) -> WorkerError:
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


class LocalVLLMWorker(BaseWorker):
    """OpenAI-compatible local vLLM worker with health checking."""

    def __init__(
        self,
        *,
        worker_id: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        max_concurrency: int = 2,
        health_timeout_seconds: float = 3.0,
        enable_thinking: bool = False,
        api_key: str | None = None,
        health_check_enabled: bool = True,
        cold_start_retries: int = 0,
        cold_start_retry_delay_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._metadata = WorkerMetadata(
            worker_id=worker_id,
            provider=PROVIDER,
            model=model,
            capabilities=DEFAULT_CAPABILITIES,
            max_concurrency=max_concurrency,
            description="Local vLLM worker for general and reasoning requests.",
            routing_categories=frozenset({"general", "reasoning"}),
        )
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._enable_thinking = enable_thinking
        self._api_key = api_key
        self._health_check_enabled = health_check_enabled
        self._cold_start_retries = max(0, cold_start_retries)
        self._cold_start_retry_delay_seconds = max(0.0, cold_start_retry_delay_seconds)
        self._health_timeout_seconds = health_timeout_seconds
        self._owns_client = client is None
        # Streaming reads have no read timeout: generation may legitimately
        # take a long time after the first upstream byte.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = client or httpx.AsyncClient(
            headers=headers,
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
        return f"{self._base_url}/v1/chat/completions"

    def _request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = payload.copy()
        request_payload["model"] = self._metadata.model
        # Qwen3 chat templates emit an explicit <think> channel by default.
        # Disable it for user-facing completions unless an operator opts in.
        if self._metadata.model.lower().startswith("qwen3") or "qwen3-" in self._metadata.model.lower():
            template_kwargs = dict(request_payload.get("chat_template_kwargs") or {})
            template_kwargs["enable_thinking"] = False
            request_payload["chat_template_kwargs"] = template_kwargs
        return request_payload

    async def health_check(self) -> bool:
        # Modal scales idle servers to zero. A periodic HTTP health probe would
        # keep the GPU container warm and defeat scale-to-zero billing.
        if not self._health_check_enabled:
            return True
        try:
            response = await self._client.get(
                f"{self._base_url}/health",
                timeout=self._health_timeout_seconds,
            )
            if response.status_code == 200:
                return True
            if response.status_code == 404:
                # Some OpenAI-compatible deployments only expose /v1/models.
                models = await self._client.get(
                    f"{self._base_url}/v1/models",
                    timeout=self._health_timeout_seconds,
                )
                return models.status_code == 200
            return False
        except Exception:
            return False

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = None
        for attempt in range(self._cold_start_retries + 1):
            try:
                response = await self._client.post(
                    self.chat_url,
                    json=self._request_payload(payload),
                    timeout=self._timeout_seconds,
                )
            except Exception as exc:
                raise _translate_transport_error(exc, self._metadata.worker_id) from exc
            # Modal Servers return 503 while a scaled-to-zero container starts.
            # This opt-in retry is safe for this known pre-container response.
            if response.status_code != 503 or attempt >= self._cold_start_retries:
                break
            await response.aclose()
            await asyncio.sleep(self._cold_start_retry_delay_seconds)

        if response is None:
            raise WorkerResponseError(
                f"worker {self._metadata.worker_id} returned no response",
                worker_id=self._metadata.worker_id,
            )
        if response.status_code >= 400:
            raise _error_for_status(response, self._metadata.worker_id)

        try:
            body = response.json()
        except ValueError as exc:
            raise WorkerResponseError(
                f"worker {self._metadata.worker_id} returned invalid JSON",
                worker_id=self._metadata.worker_id,
            ) from exc
        return _validate_completion(body, self._metadata.worker_id)

    async def open_stream(self, payload: dict[str, Any]) -> WorkerStream:
        upstream = None
        for attempt in range(self._cold_start_retries + 1):
            request = self._client.build_request(
                "POST",
                self.chat_url,
                json=self._request_payload(payload),
            )
            try:
                upstream = await self._client.send(request, stream=True)
            except Exception as exc:
                raise _translate_transport_error(exc, self._metadata.worker_id) from exc
            if upstream.status_code != 503 or attempt >= self._cold_start_retries:
                break
            await upstream.aread()
            await upstream.aclose()
            await asyncio.sleep(self._cold_start_retry_delay_seconds)

        if upstream is None:
            raise WorkerResponseError(
                f"worker {self._metadata.worker_id} returned no response",
                worker_id=self._metadata.worker_id,
            )
        if upstream.status_code >= 400:
            try:
                await upstream.aread()
            finally:
                await upstream.aclose()
            raise _error_for_status(upstream, self._metadata.worker_id)

        async def iterator():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return WorkerStream(
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            body_iterator=iterator(),
            close=upstream.aclose,
        )

    async def metrics_text(self) -> str:
        """Return the local vLLM Prometheus exposition, if its metrics endpoint is enabled."""
        response = await self._client.get(f"{self._base_url}/metrics", timeout=self._health_timeout_seconds)
        response.raise_for_status()
        return response.text

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
