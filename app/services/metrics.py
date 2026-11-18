"""Bounded in-process request telemetry for the evaluation dashboard.

Measurements are gateway-observed ASGI timings, not client network timings and
not token-level timings. The dashboard labels these distinctions explicitly.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


class MetricsStore:
    """Thread-safe counters plus a fixed-size ring buffer of request samples."""

    def __init__(self, max_samples: int = 2000) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self._samples: deque[dict[str, Any]] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._started_monotonic = time.monotonic()
        self._requests_total = 0
        self._requests_failed = 0
        self._active_requests = 0
        self._failovers = 0

    def begin(self) -> None:
        with self._lock:
            self._active_requests += 1

    def record_failover(self) -> None:
        with self._lock:
            self._failovers += 1

    def finish(self, sample: dict[str, Any]) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._requests_total += 1
            if int(sample.get("status_code") or 500) >= 400 or sample.get("completed") is False:
                self._requests_failed += 1
            self._samples.append(sample)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            total = self._requests_total
            failed = self._requests_failed
            active = self._active_requests
            failovers = self._failovers
        durations = [float(row["gateway_duration_ms"]) for row in samples]
        first_chunks = [
            float(row["gateway_first_body_ms"])
            for row in samples
            if row.get("gateway_first_body_ms") is not None
        ]
        streaming = [row for row in samples if row.get("streaming")]
        status_counts: dict[str, int] = {}
        worker_counts: dict[str, int] = {}
        for row in samples:
            status = str(row.get("status_code", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            worker = row.get("worker_id") or "unassigned"
            worker_counts[worker] = worker_counts.get(worker, 0) + 1
        return {
            "started_at": self._started_at,
            "uptime_seconds": round(time.monotonic() - self._started_monotonic, 1),
            "average_requests_per_second": round(total / max(1.0, time.monotonic() - self._started_monotonic), 3),
            "requests_total": total,
            "requests_failed": failed,
            "error_rate": round(failed / total, 4) if total else 0.0,
            "active_requests": active,
            "failovers_total": failovers,
            "samples_retained": len(samples),
            "sample_capacity": self._samples.maxlen,
            "gateway_duration_ms": {
                "p50": _percentile(durations, 0.50),
                "p95": _percentile(durations, 0.95),
                "p99": _percentile(durations, 0.99),
            },
            "gateway_first_body_ms": {
                "p50": _percentile(first_chunks, 0.50),
                "p95": _percentile(first_chunks, 0.95),
            },
            "streaming_requests_retained": len(streaming),
            "status_counts_retained": status_counts,
            "worker_distribution_retained": worker_counts,
            "measurement_notes": [
                "Gateway timings are measured at the ASGI boundary.",
                "gateway_first_body_ms is the first non-empty response body emitted by the gateway, not client-observed TTFT.",
                "ASGI body frames and SSE events are not model tokens; TPOT, ITL and token jitter are not inferred.",
                "Percentiles use the bounded recent sample window; request totals and failures are process-lifetime counters.",
            ],
        }

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            return list(self._samples)[-limit:]


metrics = MetricsStore()


class MetricsMiddleware:
    """Collect low-overhead request lifecycle measurements for chat completions."""

    def __init__(self, app, store: MetricsStore | None = None) -> None:
        self.app = app
        self.store = store or metrics

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        status_code = 500
        response_headers: dict[str, str] = {}
        first_body_at: float | None = None
        body_frames = 0
        body_bytes = 0
        completed = False
        self.store.begin()

        async def measured_send(message) -> None:
            nonlocal status_code, response_headers, first_body_at
            nonlocal body_frames, body_bytes, completed
            is_final_body = False
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = {
                    key.decode("latin-1").lower(): value.decode("latin-1")
                    for key, value in message.get("headers", [])
                }
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    if first_body_at is None:
                        first_body_at = time.perf_counter()
                    body_frames += 1
                    body_bytes += len(body)
                is_final_body = not message.get("more_body", False)

            # Send first. If the client disconnects while sending the final
            # frame, the outer exception path records an incomplete request.
            await send(message)
            if is_final_body and not completed:
                completed = True
                finished = time.perf_counter()
                self.store.finish({
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status_code": status_code,
                    "worker_id": response_headers.get("x-worker-id"),
                    "semantic_category": response_headers.get("x-semantic-category"),
                    "streaming": "text/event-stream" in response_headers.get("content-type", ""),
                    "gateway_duration_ms": round((finished - started) * 1000, 3),
                    "gateway_first_body_ms": round((first_body_at - started) * 1000, 3) if first_body_at else None,
                    "asgi_body_frames": body_frames,
                    "response_bytes": body_bytes,
                    "completed": True,
                })

        try:
            await self.app(scope, receive, measured_send)
            if not completed:
                finished = time.perf_counter()
                self.store.finish({
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status_code": status_code,
                    "worker_id": response_headers.get("x-worker-id"),
                    "semantic_category": response_headers.get("x-semantic-category"),
                    "streaming": "text/event-stream" in response_headers.get("content-type", ""),
                    "gateway_duration_ms": round((finished - started) * 1000, 3),
                    "gateway_first_body_ms": round((first_body_at - started) * 1000, 3) if first_body_at else None,
                    "asgi_body_frames": body_frames,
                    "response_bytes": body_bytes,
                    "completed": False,
                })
                completed = True
        except BaseException:
            if not completed:
                finished = time.perf_counter()
                self.store.finish({
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status_code": status_code,
                    "worker_id": response_headers.get("x-worker-id"),
                    "semantic_category": response_headers.get("x-semantic-category"),
                    "streaming": "text/event-stream" in response_headers.get("content-type", ""),
                    "gateway_duration_ms": round((finished - started) * 1000, 3),
                    "gateway_first_body_ms": round((first_body_at - started) * 1000, 3) if first_body_at else None,
                    "asgi_body_frames": body_frames,
                    "response_bytes": body_bytes,
                    "completed": False,
                })
            raise
