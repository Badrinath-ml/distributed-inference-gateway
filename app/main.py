from __future__ import annotations

import logging
from pathlib import Path

from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi import Body
import re

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
    title="Distributed Speculative Decoder Inference and Semantic Routing Gateway",
    lifespan=lifespan,
)
app.add_middleware(MetricsMiddleware, store=metrics)


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse(Path(__file__).resolve().parent.parent / "dashboard" / "index.html")


@app.get("/api/metrics/overview")
async def metrics_overview(request: Request):
    snapshot = metrics.snapshot()
    workers = {item["worker_id"]: item for item in request.app.state.worker_registry.snapshot()}
    grouped: dict[str, list[dict]] = {}
    for sample in metrics.recent(200):
        worker_id = sample.get("worker_id") or "unassigned"
        grouped.setdefault(worker_id, []).append(sample)
    def summarize(rows):
        durations = [float(row["gateway_duration_ms"]) for row in rows]
        first = [float(row["gateway_first_body_ms"]) for row in rows if row.get("gateway_first_body_ms") is not None]
        return {"requests": len(rows), "errors": sum(int(row.get("status_code") or 500) >= 400 or row.get("completed") is False for row in rows),
                "gateway_latency_ms": {"p50": _percentile_api(durations, .50), "p95": _percentile_api(durations, .95), "p99": _percentile_api(durations, .99)},
                "first_body_ms": {"p50": _percentile_api(first, .50), "p95": _percentile_api(first, .95)}}
    snapshot["worker_metrics"] = {worker_id: {**summarize(rows), "provider": workers.get(worker_id, {}).get("provider", "unknown"), "model": workers.get(worker_id, {}).get("model", "unknown")} for worker_id, rows in grouped.items()}
    provider_groups: dict[str, list[dict]] = {}
    for worker_id, rows in grouped.items():
        provider = workers.get(worker_id, {}).get("provider", "unknown")
        provider_groups.setdefault(provider, []).extend(rows)
    snapshot["provider_metrics"] = {provider: summarize(rows) for provider, rows in provider_groups.items()}
    return snapshot


def _percentile_api(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo = int(pos); hi = min(lo + 1, len(ordered) - 1); weight = pos - lo
    return round(ordered[lo] * (1 - weight) + ordered[hi] * weight, 3)


@app.get("/api/metrics/vllm")
async def local_vllm_metrics(request: Request):
    local = [worker for worker in request.app.state.worker_registry.workers() if isinstance(worker, LocalVLLMWorker)]
    if not local:
        return {"available": False, "reason": "No local vLLM worker is registered."}
    worker = local[0]
    try:
        raw = await worker.metrics_text()
        values: dict[str, float] = {}
        buckets: dict[str, dict[float, float]] = {}
        for line in raw.splitlines():
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([^\s{]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$", line)
            if not match:
                continue
            name, labels, raw_value = match.groups()
            value = float(raw_value)
            values[name] = values.get(name, 0.0) + value
            if name.endswith("_bucket") and labels:
                le = re.search(r'(?:^|,)le="([^"]+)"', labels)
                if le:
                    bound = float("inf") if le.group(1) == "+Inf" else float(le.group(1))
                    family = name[:-7]
                    family_buckets = buckets.setdefault(family, {})
                    family_buckets[bound] = family_buckets.get(bound, 0.0) + value

        def histogram_quantile(metric_name: str, quantile: float):
            series = sorted(buckets.get(metric_name, {}).items(), key=lambda item: item[0])
            if not series:
                return None
            total = series[-1][1]
            if total <= 0:
                return None
            target = total * quantile
            previous_bound = 0.0
            previous_count = 0.0
            for upper_bound, cumulative in series:
                if cumulative >= target:
                    if upper_bound == float("inf"):
                        estimate = previous_bound
                    elif cumulative <= previous_count:
                        estimate = upper_bound
                    else:
                        fraction = (target - previous_count) / (cumulative - previous_count)
                        estimate = previous_bound + (upper_bound - previous_bound) * fraction
                    return round(estimate * 1000, 3)
                previous_bound, previous_count = upper_bound, cumulative
            return None
        draft = first_metric("vllm:spec_decode_num_draft_tokens_total", "vllm_spec_decode_num_draft_tokens_total")
        accepted = first_metric("vllm:spec_decode_num_accepted_tokens_total", "vllm_spec_decode_num_accepted_tokens_total")
        profile = {"target_model": worker.metadata.model, "draft_model": "Qwen/Qwen3-0.6B", "dtype": "half", "max_model_len": 1024, "max_num_seqs": 2, "max_model_len": 1024, "gpu_memory_utilization": 0.80, "kv_cache_memory_mib": 512, "num_speculative_tokens": 2, "enforce_eager": "1", "flashinfer_sampler": "0"}
        profile_path = Path(__file__).resolve().parent.parent / "scripts" / "worker1.env"
        if profile_path.exists():
            profile_text = profile_path.read_text(encoding="utf-8")
            for key, field in (("VLLM_DRAFT_MODEL", "draft_model"), ("VLLM_DTYPE", "dtype"), ("VLLM_MAX_MODEL_LEN", "max_model_len"), ("VLLM_MAX_NUM_SEQS", "max_num_seqs"), ("VLLM_GPU_MEMORY_UTILIZATION", "gpu_memory_utilization"), ("VLLM_KV_CACHE_MEMORY_BYTES", "kv_cache_memory_bytes"), ("VLLM_NUM_SPECULATIVE_TOKENS", "num_speculative_tokens"), ("VLLM_ENFORCE_EAGER", "enforce_eager"), ("VLLM_USE_FLASHINFER_SAMPLER", "flashinfer_sampler")):
                prefix = 'export ' + key + '="${' + key + ':-'
                for line in profile_text.splitlines():
                    if line.startswith(prefix) and line.endswith('}"'):
                        raw_value = line[len(prefix):-2]
                        profile[field] = int(raw_value) if raw_value.isdigit() else raw_value
                        break
        if "kv_cache_memory_bytes" in profile:
            profile["kv_cache_memory_mib"] = round(profile.pop("kv_cache_memory_bytes") / (1024 * 1024), 1)
        return {"available": True, "worker_id": worker.metadata.worker_id, "model": worker.metadata.model, "launch_profile": profile,
                "speculative_decoding": {"draft_tokens_total": draft, "accepted_tokens_total": accepted,
                    "acceptance_rate": round(accepted / draft, 4) if draft and accepted is not None else None},
                "token_latency_metrics": {name: first_metric(name) for name in ("vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count", "vllm:time_per_output_token_seconds_sum", "vllm:time_per_output_token_seconds_count", "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count")},
                "metrics_endpoint": worker._base_url + "/metrics",
                "histogram_quantiles_ms": {family: {"p50": histogram_quantile(metric, 0.50), "p95": histogram_quantile(metric, 0.95)} for family, metric in {"ttft": "vllm:time_to_first_token_seconds", "tpot": "vllm:time_per_output_token_seconds", "inter_token_latency": "vllm:inter_token_latency_seconds"}.items()},
                "inter_token_jitter_ms": (lambda p50, p95: round(p95 - p50, 3) if p50 is not None and p95 is not None else None)(histogram_quantile("vllm:inter_token_latency_seconds", 0.50), histogram_quantile("vllm:inter_token_latency_seconds", 0.95)),
                "note": "Means and histogram quantiles are process-wide vLLM metrics, not per-request samples. Quantiles are approximate bucket interpolations; ITL spread (p95-p50) is a jitter proxy, not per-token standard deviation."}
    except Exception as exc:
        return {"available": False, "worker_id": worker.metadata.worker_id, "reason": f"vLLM metrics endpoint unavailable: {type(exc).__name__}"}


@app.get("/api/metrics/workers")
async def metrics_workers(request: Request):
    return {"workers": request.app.state.worker_registry.snapshot()}


@app.put("/api/workers/{worker_id}/availability")
async def set_worker_availability(worker_id: str, request: Request, payload: dict = Body(...)):
    available = payload.get("available")
    if not isinstance(available, bool):
        raise HTTPException(status_code=400, detail="'available' must be a boolean.")
    try:
        record = request.app.state.worker_registry.set_available(worker_id, available)
        if isinstance(record.worker, MockWorker):
            record.worker.set_available(available)
        return {"worker_id": worker_id, "available": available, "state": record.state.value}
    except KeyError:
        raise HTTPException(status_code=404, detail="Worker not found.")


@app.post("/api/workers/mock")
async def add_mock_worker(payload: dict, request: Request):
    worker_id = str(payload.get("worker_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,63}", worker_id):
        raise HTTPException(status_code=400, detail="worker_id must be 2-64 safe characters.")
    latency_ms = float(payload.get("latency_ms", 25))
    concurrency = int(payload.get("max_concurrency", 4))
    if not 0 <= latency_ms <= 10000 or not 1 <= concurrency <= 64:
        raise HTTPException(status_code=400, detail="Invalid latency or max_concurrency.")
    worker = MockWorker(worker_id=worker_id, model="mock-simulator", latency_ms=latency_ms, max_concurrency=concurrency)
    try:
        record = request.app.state.worker_registry.register(worker)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    request.app.state.worker_registry.mark_health(worker_id, True)
    return {"worker": request.app.state.worker_registry.require(worker_id).as_dict(), "message": "Mock worker added."}


@app.delete("/api/workers/{worker_id}")
async def remove_mock_worker(worker_id: str, request: Request):
    try:
        record = request.app.state.worker_registry.require(worker_id)
        if not isinstance(record.worker, MockWorker):
            raise HTTPException(status_code=403, detail="Only mock workers can be removed.")
        worker = request.app.state.worker_registry.unregister(worker_id)
        await worker.aclose()
        return {"worker_id": worker_id, "removed": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="Worker not found.")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


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
        worker.clear_faults()
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
