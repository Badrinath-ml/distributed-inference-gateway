"""Worker-independent gateway request execution and failover policy."""

from __future__ import annotations

import logging
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.services.scheduler import NoEligibleWorkersError, WorkerScheduler
from app.services.metrics import metrics
from app.workers.base import (
    CHAT_CAPABILITY,
    STREAMING_CAPABILITY,
    WorkerError,
    WorkerUnavailableError,
)

logger = logging.getLogger(__name__)

CHAT_REQUIRED = frozenset({CHAT_CAPABILITY})
STREAM_REQUIRED = frozenset({CHAT_CAPABILITY, STREAMING_CAPABILITY})


def _response_headers(route_category: str, worker_id: str) -> dict[str, str]:
    return {
        "X-Semantic-Category": route_category,
        "X-Worker-Id": worker_id,
    }


def _select(
    scheduler: WorkerScheduler,
    required_capabilities: frozenset[str],
    exclude: set[str],
    *,
    retrying: bool,
    route_category: str | None = None,
):
    try:
        return scheduler.select(required_capabilities, exclude=exclude, route_category=route_category)
    except NoEligibleWorkersError as exc:
        if retrying:
            metrics.record_failover_result(False)
            raise WorkerUnavailableError(
                "no alternative worker is available after a connection failure"
            ) from exc
        raise


async def generate_completion(
    payload: dict[str, Any],
    route_category: str,
    scheduler: WorkerScheduler,
    *,
    retry_on_connect_error: bool = True,
):
    """Execute a non-streaming completion with at most one safe retry."""
    excluded: set[str] = set()
    attempt = 0

    while True:
        record = _select(
            scheduler,
            CHAT_REQUIRED,
            excluded,
            retrying=attempt > 0,
            route_category=route_category,
        )
        try:
            body = await record.worker.chat_completion(payload)
        except WorkerUnavailableError:
            scheduler.release(record)
            if attempt > 0:
                metrics.record_failover_result(False)
            if not retry_on_connect_error or attempt >= 1:
                raise
            logger.warning(
                "Worker %s was unreachable; retrying on another worker",
                record.worker_id,
            )
            metrics.record_failover()
            excluded.add(record.worker_id)
            attempt += 1
            continue
        except BaseException:
            scheduler.release(record)
            if attempt > 0:
                metrics.record_failover_result(False)
            raise

        scheduler.release(record)
        if attempt > 0:
            metrics.record_failover_result(True)
        logger.info(
            "Routed completion to worker=%s provider=%s category=%s",
            record.worker_id,
            record.metadata.provider,
            route_category,
        )
        return JSONResponse(
            content=body,
            headers={**_response_headers(route_category, record.worker_id), "X-Routing-Decision": "semantic-specialized" if record.metadata.routing_categories and route_category in record.metadata.routing_categories else "load-health-fallback"},
        )


async def stream_completion(
    payload: dict[str, Any],
    route_category: str,
    scheduler: WorkerScheduler,
    *,
    retry_on_connect_error: bool = True,
):
    """Execute a streaming completion.

    A connection failure that occurs before the upstream stream is opened may
    be retried once on another worker; once SSE bytes can flow, no retry is
    attempted. The worker reservation is released exactly once when the stream
    finishes, fails or is cancelled.
    """
    excluded: set[str] = set()
    attempt = 0

    while True:
        record = _select(
            scheduler,
            STREAM_REQUIRED,
            excluded,
            retrying=attempt > 0,
            route_category=route_category,
        )
        try:
            stream = await record.worker.open_stream(payload)
        except WorkerUnavailableError:
            scheduler.release(record)
            if attempt > 0:
                metrics.record_failover_result(False)
            if not retry_on_connect_error or attempt >= 1:
                raise
            logger.warning(
                "Worker %s was unreachable while opening a stream; retrying",
                record.worker_id,
            )
            metrics.record_failover()
            excluded.add(record.worker_id)
            attempt += 1
            continue
        except BaseException:
            scheduler.release(record)
            if attempt > 0:
                metrics.record_failover_result(False)
            raise
        break

    released = False

    async def release_once() -> None:
        nonlocal released
        if released:
            return
        released = True
        scheduler.release(record)

    async def cleanup() -> None:
        await stream.aclose()
        await release_once()

    async def event_stream():
        completed = False
        try:
            async for chunk in stream.body_iterator:
                yield chunk
            completed = True
        finally:
            await cleanup()
            if attempt > 0:
                metrics.record_failover_result(completed)

    logger.info(
        "Routed stream to worker=%s provider=%s category=%s",
        record.worker_id,
        record.metadata.provider,
        route_category,
    )
    headers = _response_headers(route_category, record.worker_id)
    headers["X-Routing-Decision"] = "semantic-specialized" if record.metadata.routing_categories and route_category in record.metadata.routing_categories else "load-health-fallback"
    headers.update(
        {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
        background=BackgroundTask(cleanup),
    )
