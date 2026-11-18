from __future__ import annotations

import logging
from pathlib import Path

from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request

from app.config import settings
from app.services.embeddings import EmbeddingExecutor
from app.services.health import HealthMonitor
from app.services.inference import generate_completion, stream_completion
from app.services.metrics import MetricsMiddleware, metrics
from app.services.registry import WorkerRegistry
from app.services.router import SemanticRouter
from app.services.scheduler import NoEligibleWorkersError, WorkerScheduler
from app.workers import (
    BaseWorker,
    LocalVLLMWorker,
    MockWorker,
    OpenRouterWorker,
    WorkerError,
    WorkerRateLimitError,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_workers() -> list[BaseWorker]:
    """Build workers from configuration.

    Kept as a module-level function so tests can inject deterministic workers
    without contacting a GPU, cloud provider or model download.
    """
    workers: list[BaseWorker] = []

    if settings.worker1_enabled:
        workers.append(
            LocalVLLMWorker(
                worker_id=settings.worker1_id,
                base_url=settings.vllm_base_url,
                model=settings.vllm_model,
                timeout_seconds=settings.worker1_timeout_seconds,
                max_concurrency=settings.worker1_max_concurrency,
                health_timeout_seconds=settings.health_check_timeout_seconds,
            )
        )

    if settings.openrouter_enabled:
        if settings.openrouter_api_key and settings.openrouter_model:
            workers.append(
                OpenRouterWorker(
                    worker_id="worker2-openrouter",
                    api_key=settings.openrouter_api_key,
                    model=settings.openrouter_model,
                    base_url=settings.openrouter_base_url,
                    timeout_seconds=settings.openrouter_timeout_seconds,
                    max_concurrency=settings.openrouter_max_concurrency,
                    health_timeout_seconds=settings.health_check_timeout_seconds,
                    site_url=settings.openrouter_site_url,
                    app_name=settings.openrouter_app_name,
                )
            )
        elif settings.openrouter_api_key or settings.openrouter_model:
            logger.warning(
                "OpenRouter worker not registered: set both OPENROUTER_API_KEY "
                "and OPENROUTER_MODEL (and keep OPENROUTER_ENABLED=true)."
            )

    if settings.mock_worker_enabled:
        workers.append(
            MockWorker(
                worker_id=settings.mock_worker_id,
                model=settings.mock_worker_model,
                latency_ms=settings.mock_worker_latency_ms,
                max_concurrency=settings.mock_worker_max_concurrency,
            )
        )

    if not workers:
        logger.warning("No inference workers are configured.")
    return workers


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.embedding_torch_threads is not None:
        torch.set_num_threads(settings.embedding_torch_threads)

    logger.info("Initializing semantic router...")
    app.state.semantic_router = SemanticRouter()

    app.state.embedding_executor = EmbeddingExecutor(
        max_workers=settings.embedding_executor_workers,
    )
    logger.info(
        "Embedding executor initialized with %d worker(s).",
        app.state.embedding_executor.max_workers,
    )

    registry = WorkerRegistry()
    workers = build_workers()
    for worker in workers:
        record = registry.register(worker)
        logger.info(
            "Registered worker %s (provider=%s, model=%s, capabilities=%s, "
            "max_concurrency=%d, state=%s)",
            worker.metadata.worker_id,
            worker.metadata.provider,
            worker.metadata.model,
            sorted(worker.metadata.capabilities),
            worker.metadata.max_concurrency,
            record.state.value,
        )

    app.state.worker_registry = registry
    app.state.worker_scheduler = WorkerScheduler(registry)
    app.state.health_monitor = HealthMonitor(
        registry,
        interval_seconds=settings.health_check_interval_seconds,
        timeout_seconds=settings.health_check_timeout_seconds,
        enabled=settings.health_check_enabled,
    )
    app.state.health_monitor.start()

    logger.info("Gateway initialized with %d worker(s).", len(workers))

    try:
        yield
    finally:
        logger.info("Gateway shutting down.")
        await app.state.health_monitor.stop()
        for worker in workers:
            try:
                await worker.aclose()
            except Exception:
                logger.exception("Failed to close worker %s", worker.metadata.worker_id)
        app.state.embedding_executor.shutdown(wait=True)
        logger.info("Gateway shutdown complete.")


app = FastAPI(
    title="Distributed Inference Gateway",
    lifespan=lifespan,
)
app.add_middleware(MetricsMiddleware, store=metrics)


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(Path(__file__).resolve().parent.parent / "dashboard" / "index.html")


@app.get("/api/metrics/overview")
async def metrics_overview():
    return metrics.snapshot()


@app.get("/api/metrics/workers")
async def metrics_workers(request: Request):
    return {"workers": request.app.state.worker_registry.snapshot()}


@app.get("/api/metrics/recent")
async def metrics_recent(limit: int = 100):
    return {"requests": metrics.recent(limit)}


@app.get("/api/testing/capabilities")
async def testing_capabilities(request: Request):
    workers = request.app.state.worker_registry.workers()
    mocks = [worker for worker in workers if worker.metadata.provider == "mock"]
    return {
        "enabled": settings.dashboard_fault_injection_enabled,
        "mock_workers": [worker.metadata.as_dict() for worker in mocks],
        "real_workers_mutable": False,
    }


@app.post("/api/testing/faults")
async def apply_mock_fault(payload: dict, request: Request):
    if not settings.dashboard_fault_injection_enabled:
        raise HTTPException(status_code=403, detail="Fault injection is disabled by configuration.")
    if payload.get("confirm") is not True:
        raise HTTPException(status_code=400, detail="Explicit confirmation is required.")
    worker_id = str(payload.get("worker_id", ""))
    worker = next((w for w in request.app.state.worker_registry.workers() if w.metadata.worker_id == worker_id), None)
    if worker is None or worker.metadata.provider != "mock" or not isinstance(worker, MockWorker):
        raise HTTPException(status_code=404, detail="Only registered mock workers can be targeted.")
    action = payload.get("action")
    registry = request.app.state.worker_registry
    if action == "unhealthy":
        worker.set_healthy(False)
        registry.mark_health(worker_id, False, error="dashboard fault injection")
    elif action == "unavailable":
        worker.set_available(False)
        registry.set_available(worker_id, False)
    elif action == "latency":
        latency_ms = float(payload.get("latency_ms", 500))
        if not 0 <= latency_ms <= 10000:
            raise HTTPException(status_code=400, detail="latency_ms must be between 0 and 10000.")
        worker.set_latency_ms(latency_ms)
    elif action == "fail_next":
        count = int(payload.get("count", 1))
        mode = str(payload.get("mode", "error"))
        if not 1 <= count <= 10 or mode not in {"error", "timeout", "unavailable", "rate_limit"}:
            raise HTTPException(status_code=400, detail="Invalid failure count or mode.")
        worker.fail_next(count=count, mode=mode)
    elif action == "restore":
        worker.set_healthy(True)
        worker.set_available(True)
        worker.set_latency_ms(0)
        registry.set_available(worker_id, True)
        registry.mark_health(worker_id, True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported fault action.")
    return {"message": f"Applied {action} to mock worker {worker_id}.", "worker_id": worker_id, "action": action}


def extract_user_prompt(payload: dict) -> str:
    messages = payload.get("messages", [])

    user_messages = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "user"
    ]

    # Handle ordinary text prompts.
    # For multimodal content, collect textual blocks only.
    text_parts = []

    for content in user_messages:
        if isinstance(content, str):
            text_parts.append(content)

        elif isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    text_parts.append(item.get("text", ""))

    return " ".join(text_parts).strip()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/health/workers")
async def worker_health(request: Request):
    registry = request.app.state.worker_registry
    return {"workers": registry.snapshot()}


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: dict,
    request: Request,
):
    try:
        prompt = extract_user_prompt(payload)

        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="No textual user message found.",
            )

        router = request.app.state.semantic_router
        # CPU embedding inference runs in the application-scoped thread pool;
        # the event loop remains free to serve health checks and other requests.
        routing_result = await request.app.state.embedding_executor.run(
            router.route,
            prompt,
        )

        category = routing_result["category"]

        logger.info(
            "Semantic routing: category=%s, score=%.4f, margin=%.4f",
            category,
            routing_result["top_score"],
            routing_result["margin"],
        )

        scheduler = request.app.state.worker_scheduler

        if payload.get("stream", False):
            return await stream_completion(
                payload,
                route_category=category,
                scheduler=scheduler,
                retry_on_connect_error=settings.routing_retry_on_connect_error,
            )

        return await generate_completion(
            payload,
            route_category=category,
            scheduler=scheduler,
            retry_on_connect_error=settings.routing_retry_on_connect_error,
        )

    except HTTPException:
        raise

    except NoEligibleWorkersError as exc:
        logger.warning("No eligible inference worker: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="No eligible inference worker is available.",
        ) from exc

    except WorkerRateLimitError as exc:
        headers = {}
        if exc.retry_after_seconds is not None:
            headers["Retry-After"] = str(int(exc.retry_after_seconds))
        logger.warning("Inference worker rate limited: %s", exc)
        raise HTTPException(
            status_code=429,
            detail="Inference worker rate limit exceeded.",
            headers=headers or None,
        ) from exc

    except WorkerError:
        logger.exception("Inference worker request failed")

        raise HTTPException(
            status_code=502,
            detail="Inference gateway request failed.",
        )

    except Exception:
        logger.exception("Inference request failed")

        raise HTTPException(
            status_code=502,
            detail="Inference gateway request failed.",
        )
