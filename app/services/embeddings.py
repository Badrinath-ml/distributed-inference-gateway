from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model(model_name: str) -> tuple[Any, Any]:
    """Load the CPU embedding model once per process, guarded by a lock.

    The model is immutable during inference (eval mode, no gradients), so
    sharing it between service instances and executor threads is safe. The lock
    prevents two application lifespans from racing the initial load.
    """

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        logger.info("Loading embedding model %s on CPU...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(torch.device("cpu"))
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
        logger.info("Embedding model loaded successfully.")
        return tokenizer, model


class EmbeddingService:
    def __init__(self) -> None:
        self.model_name = _MODEL_NAME
        self.device = torch.device("cpu")
        self.tokenizer, self.model = _load_model(self.model_name)

    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

            token_embeddings = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            summed_embeddings = torch.sum(
                token_embeddings * mask,
                dim=1,
            )

            token_counts = torch.clamp(
                mask.sum(dim=1),
                min=1e-9,
            )

            sentence_embeddings = (
                summed_embeddings / token_counts
            )

            sentence_embeddings = F.normalize(
                sentence_embeddings,
                p=2,
                dim=1,
            )

        return sentence_embeddings


class EmbeddingExecutor:
    """Bounded, application-scoped executor for CPU embedding inference.

    One executor is created during the FastAPI lifespan and shut down at the
    end of it. Requests submit work through :meth:`run`; the event loop stays
    free to serve health checks and other requests while a worker thread runs
    the CPU-bound tokenizer/model call.
    """

    def __init__(
        self,
        max_workers: int = 1,
        *,
        thread_name_prefix: str = "gateway-embedding",
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._shutdown = False

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    def active_thread_names(self) -> list[str]:
        return sorted(
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith(self._thread_name_prefix)
        )

    async def run(self, function: Callable[..., T], *args: Any) -> T:
        if self._shutdown:
            raise RuntimeError("embedding executor is shut down")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, function, *args)

    def shutdown(self, wait: bool = True) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=True)
