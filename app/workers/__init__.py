"""Inference worker implementations and the shared worker contract."""

from app.workers.base import (
    CHAT_CAPABILITY,
    STREAMING_CAPABILITY,
    BaseWorker,
    WorkerError,
    WorkerMetadata,
    WorkerProviderError,
    WorkerRateLimitError,
    WorkerResponseError,
    WorkerStream,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from app.workers.local_vllm import LocalVLLMWorker
from app.workers.mock import MockWorker
from app.workers.openrouter import OpenRouterWorker

__all__ = [
    "CHAT_CAPABILITY",
    "STREAMING_CAPABILITY",
    "BaseWorker",
    "LocalVLLMWorker",
    "MockWorker",
    "OpenRouterWorker",
    "WorkerError",
    "WorkerMetadata",
    "WorkerProviderError",
    "WorkerRateLimitError",
    "WorkerResponseError",
    "WorkerStream",
    "WorkerTimeoutError",
    "WorkerUnavailableError",
]
