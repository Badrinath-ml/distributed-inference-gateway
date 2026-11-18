#!/usr/bin/env python3
"""Stage 4 gateway embedding/event-loop benchmark.

Runs the FastAPI gateway in-process against the deterministic local stub
upstream, so the measurement isolates semantic-routing/embedding behaviour and
gateway overhead from GPU inference.

For every repetition the script records:

* end-to-end non-streaming gateway request latency (embedding + routing +
  local stub upstream),
* concurrent gateway throughput at several concurrency levels,
* event-loop heartbeat lag and ``/health`` probe latency while that load runs,
* process CPU time,
* direct single-prompt CPU embedding latency (outside the concurrent load).

The exact same workload can be run before and after the Stage 4 optimization.
Results never overwrite an existing file unless ``--force`` is passed.

Example::

    PYTHONPATH=. .venv/bin/python scripts/benchmark_stage4_gateway.py \
        --label baseline-sync --repetitions 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import torch

from app.config import settings
from app.main import app
from tests.stub_upstream import StubServer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "stage4"

PROMPTS = [
    "Implement a binary search tree in Python.",
    "Solve this complex mathematical proof.",
    "Explain how HTTP works.",
    "Debug this Python code.",
    "Analyze the logical implications of this statement.",
    "What is cloud computing?",
    "Explain the time complexity of Dijkstra's algorithm.",
    "Write a Python program to calculate statistical probability.",
    "Explain how neural networks make predictions.",
    "Design an algorithm to solve a scheduling problem.",
    "Implement a graph traversal algorithm using BFS.",
    "Analyze this mathematical problem and derive its solution.",
    "Evaluate multiple possibilities and make a logical decision.",
    "Explain what an API is in simple terms.",
    "What is the difference between RAM and ROM?",
    "Give me a brief explanation of cloud computing.",
]

HEARTBEAT_INTERVAL_SECONDS = 0.01


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(percentile(values, 0.95), 4),
        "p99": round(percentile(values, 0.99), 4),
        "max": round(max(values), 4),
        "p99_reliable": len(values) >= 100,
    }


def process_cpu_seconds() -> float:
    return time.process_time()


class Metrics:
    def __init__(self) -> None:
        self.loop_lags_ms: list[float] = []
        self.health_probe_ms: list[float] = []
        self.health_probe_failures = 0
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._probe_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def _heartbeat(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            now = time.perf_counter()
            lag_ms = max(0.0, now - last - HEARTBEAT_INTERVAL_SECONDS) * 1000
            self.loop_lags_ms.append(lag_ms)
            last = now

    async def _probe(self, client: httpx.AsyncClient) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                response = await client.get("/health")
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code == 200:
                    self.health_probe_ms.append(elapsed)
                else:
                    self.health_probe_failures += 1
            except Exception:
                self.health_probe_failures += 1
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    async def start(self, client: httpx.AsyncClient) -> None:
        self._stop = asyncio.Event()
        self.loop_lags_ms = []
        self.health_probe_ms = []
        self.health_probe_failures = 0
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        self._probe_task = asyncio.create_task(self._probe(client))
        await asyncio.sleep(0.05)

    async def stop(self) -> None:
        self._stop.set()
        tasks = [
            task
            for task in (self._heartbeat_task, self._probe_task)
            if task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._heartbeat_task = None
        self._probe_task = None


async def timed_request(
    client: httpx.AsyncClient,
    prompt: str,
) -> tuple[float, dict[str, Any] | None, str | None]:
    started = time.perf_counter()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return elapsed_ms, None, f"HTTP {response.status_code}: {response.text[:160]}"
        body = response.json()
        if not body.get("choices"):
            return elapsed_ms, None, "response missing choices"
        return elapsed_ms, body, None
    except Exception as exc:  # pragma: no cover - defensive benchmark path
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, None, f"{type(exc).__name__}: {exc}"


async def run_batch(
    client: httpx.AsyncClient,
    concurrency: int,
    requests: int,
    metrics_client: httpx.AsyncClient,
) -> dict[str, Any]:
    prompts = [PROMPTS[index % len(PROMPTS)] for index in range(requests)]
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(requests):
        queue.put_nowait(index)

    latencies: list[float] = []
    errors: list[str] = []

    async def worker() -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            elapsed_ms, _body, error = await timed_request(client, prompts[index])
            latencies.append(elapsed_ms)
            if error:
                errors.append(error)

    metrics = Metrics()
    await metrics.start(metrics_client)
    cpu_start = process_cpu_seconds()
    wall_start = time.perf_counter()

    await asyncio.gather(*(worker() for _ in range(concurrency)))

    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = process_cpu_seconds() - cpu_start
    await metrics.stop()

    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_seconds": round(wall_seconds, 6),
        "throughput_requests_per_second": round(requests / wall_seconds, 4)
        if wall_seconds > 0
        else None,
        "latency_ms": summarize(latencies),
        "latencies_ms": [round(value, 4) for value in latencies],
        "errors": errors[:10],
        "error_count": len(errors),
        "error_rate": round(len(errors) / requests, 6) if requests else None,
        "event_loop_lag_ms": summarize(metrics.loop_lags_ms),
        "event_loop_lags_ms": [round(value, 4) for value in metrics.loop_lags_ms],
        "health_probe_ms": summarize(metrics.health_probe_ms),
        "health_probe_latencies_ms": [
            round(value, 4) for value in metrics.health_probe_ms
        ],
        "health_probe_failures": metrics.health_probe_failures,
        "cpu_time_seconds": round(cpu_seconds, 6),
        "cpu_utilization_cores": round(cpu_seconds / wall_seconds, 4)
        if wall_seconds > 0
        else None,
    }


async def measure_serial_embedding(
    encode: Callable[[list[str]], Any],
    samples: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    prompt = PROMPTS[0]
    for _ in range(samples):
        started = time.perf_counter()
        embedding = encode([prompt])
        latencies.append((time.perf_counter() - started) * 1000)
    norms = embedding.norm(dim=1)
    return {
        "samples": samples,
        "latency_ms": summarize(latencies),
        "shape": list(embedding.shape),
        "norm_first": float(norms[0].item()),
        "torch_num_threads": torch.get_num_threads(),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Stage 4 gateway embedding/event-loop benchmark",
        "",
        f"- Label: `{result['label']}`",
        f"- Timestamp (UTC): `{result['timestamp_utc']}`",
        f"- Executor present: `{result['gateway']['embedding_executor_present']}`",
        f"- Executor workers: `{result['gateway'].get('embedding_executor_workers')}`",
        f"- Python: `{result['environment']['python'].split()[0]}`",
        f"- Torch threads: {result['environment']['torch_num_threads']}",
        f"- CPU count: {result['environment']['cpu_count']}",
        "",
        "## Direct embedding latency",
        "",
        "| Measurement | p50 ms | p95 ms | p99 ms | n |",
        "|---|---:|---:|---:|---:|",
    ]
    embed = result["embedding"]["latency_ms"]
    lines.append(
        f"| `EmbeddingService.encode` (1 prompt) | {embed.get('median')} | "
        f"{embed.get('p95')} | {embed.get('p99')} | {embed.get('count')} |"
    )

    lines.extend(
        [
            "",
            "## Per-repetition gateway results",
            "",
            "| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | "
            "Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for repetition in result["repetitions"]:
        for row in repetition["levels"]:
            lat = row["latency_ms"]
            lag = row["event_loop_lag_ms"]
            probe = row["health_probe_ms"]
            lines.append(
                f"| {repetition['repetition']} | {row['concurrency']} | "
                f"{row['requests']} | {row['wall_seconds']} | "
                f"{row['throughput_requests_per_second']} | {lat.get('median')} | "
                f"{lat.get('p95')} | {lat.get('p99')} | {lag.get('p95')} | "
                f"{lag.get('p99')} | {probe.get('p95')} | "
                f"{row['cpu_utilization_cores']} | {row['error_count']} |"
            )

    lines.extend(
        [
            "",
            "## Aggregated by concurrency (all repetitions)",
            "",
            "| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | "
            "Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["aggregate"]:
        lat = row["latency_ms"]
        lag = row["event_loop_lag_ms"]
        probe = row["health_probe_ms"]
        lines.append(
            f"| {row['concurrency']} | {row['requests']} | "
            f"{row['throughput_mean']} | {lat.get('median')} | {lat.get('p95')} | "
            f"{lat.get('p99')} | {lag.get('p95')} | {lag.get('p99')} | "
            f"{probe.get('p95')} | {row['cpu_utilization_cores']} | "
            f"{row['error_count']} |"
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            f"- Warm-up: {result['method']['warmup_requests']} gateway requests before timing.",
            f"- Concurrency levels: {result['method']['concurrency_levels']} "
            f"with {result['method']['requests_per_concurrency_level']} requests each.",
            f"- Repetitions: {result['method']['repetitions']}.",
            "- Event-loop lag is measured with a 10 ms heartbeat; lag is the "
            "scheduling delay beyond the requested sleep.",
            "- Health probe latency is a concurrent `GET /health` every 10 ms.",
            "- The upstream is a deterministic local HTTP stub, so GPU/network "
            "time is excluded and gateway/embedding cost dominates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.disable(logging.CRITICAL)
    stub = StubServer().start()
    old_base_url = settings.vllm_base_url
    settings.vllm_base_url = stub.url
    timestamp = utc_stamp()

    try:
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        try:
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://stage4-benchmark",
                timeout=30.0,
            ) as client:
                for index in range(args.warmup):
                    elapsed, _body, error = await timed_request(
                        client, PROMPTS[index % len(PROMPTS)]
                    )
                    if error:
                        raise SystemExit(f"Warm-up failed: {error} ({elapsed:.1f} ms)")

                service = app.state.semantic_router.embedding_service
                embedding = await measure_serial_embedding(
                    service.encode, args.embedding_samples
                )

                repetitions: list[dict[str, Any]] = []
                for repetition in range(1, args.repetitions + 1):
                    levels = []
                    for concurrency in args.concurrency:
                        row = await run_batch(
                            client,
                            concurrency=concurrency,
                            requests=args.requests_per_level,
                            metrics_client=client,
                        )
                        levels.append(row)
                        print(
                            f"rep {repetition}/{args.repetitions} "
                            f"conc {concurrency}: {row['throughput_requests_per_second']} req/s, "
                            f"p50 {row['latency_ms'].get('median')} ms, "
                            f"loop p95 {row['event_loop_lag_ms'].get('p95')} ms, "
                            f"errors {row['error_count']}"
                        )
                    repetitions.append({"repetition": repetition, "levels": levels})

                aggregate = []
                for index, concurrency in enumerate(args.concurrency):
                    rows = [rep["levels"][index] for rep in repetitions]
                    latencies = [
                        value for row in rows for value in row["latencies_ms"]
                    ]
                    loop_lags = [
                        value for row in rows for value in row["event_loop_lags_ms"]
                    ]
                    probes = [
                        value
                        for row in rows
                        for value in row["health_probe_latencies_ms"]
                    ]
                    aggregate.append(
                        {
                            "concurrency": concurrency,
                            "requests": sum(row["requests"] for row in rows),
                            "throughput_mean": round(
                                statistics.mean(
                                    row["throughput_requests_per_second"] for row in rows
                                ),
                                4,
                            ),
                            "throughput_values": [
                                row["throughput_requests_per_second"] for row in rows
                            ],
                            "latency_ms": summarize(latencies),
                            "event_loop_lag_ms": summarize(loop_lags),
                            "health_probe_ms": summarize(probes),
                            "cpu_utilization_cores": round(
                                statistics.mean(
                                    row["cpu_utilization_cores"] for row in rows
                                ),
                                4,
                            ),
                            "error_count": sum(row["error_count"] for row in rows),
                        }
                    )

                executor = getattr(app.state, "embedding_executor", None)
                result = {
                    "schema_version": 1,
                    "label": args.label,
                    "timestamp_utc": timestamp,
                    "script": "scripts/benchmark_stage4_gateway.py",
                    "command": sys.argv,
                    "environment": {
                        "python": sys.version,
                        "platform": platform.platform(),
                        "cpu_count": os.cpu_count(),
                        "torch_version": torch.__version__,
                        "torch_num_threads": torch.get_num_threads(),
                        "torch_num_interop_threads": torch.get_num_interop_threads(),
                    },
                    "gateway": {
                        "embedding_executor_present": executor is not None,
                        "embedding_executor_workers": getattr(
                            executor, "max_workers", None
                        ),
                        "model": service.model_name,
                        "model_device": str(service.device),
                    },
                    "method": {
                        "warmup_requests": args.warmup,
                        "concurrency_levels": args.concurrency,
                        "requests_per_concurrency_level": args.requests_per_level,
                        "repetitions": args.repetitions,
                        "embedding_samples": args.embedding_samples,
                        "heartbeat_interval_ms": HEARTBEAT_INTERVAL_SECONDS * 1000,
                        "upstream": "local deterministic stub (tests/stub_upstream.py)",
                    },
                    "embedding": embedding,
                    "repetitions": repetitions,
                    "aggregate": aggregate,
                    "notes": [
                        "Non-streaming chat-completions requests include semantic routing.",
                        "The upstream stub adds negligible latency and no GPU work.",
                        "Event-loop lag and /health probe latency are measured under the same load.",
                        "CPU time is process CPU time (time.process_time) and may exceed one core.",
                    ],
                }

                output = (
                    Path(args.output)
                    if args.output
                    else DEFAULT_OUTPUT_DIR
                    / f"embedding_{args.label.replace(' ', '_')}_{timestamp}.json"
                )
                if output.exists() and not args.force:
                    raise SystemExit(f"Refusing to overwrite existing result: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                markdown = output.with_suffix(".md")
                write_markdown(result, markdown)
                print(f"Saved: {output}")
                print(f"Saved: {markdown}")
                return result
        finally:
            await lifespan.__aexit__(None, None, None)
    finally:
        stub.stop()
        settings.vllm_base_url = old_base_url
        logging.disable(logging.NOTSET)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--embedding-samples", type=int, default=100)
    parser.add_argument("--concurrency", default="1,2,4,8")
    args = parser.parse_args(argv)
    args.concurrency = [
        int(value)
        for value in args.concurrency.split(",")
        if value.strip()
    ]
    if not args.concurrency:
        raise SystemExit("--concurrency must contain at least one level")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
