#!/usr/bin/env python3
"""Multi-worker registry/scheduler routing benchmark (no GPU or cloud calls).

Runs the gateway in-process with deterministic mock workers and an
embedding-free semantic router so the measurement isolates worker selection,
load balancing, health state changes and gateway dispatch overhead.

Example::

    PYTHONPATH=. .venv/bin/python scripts/benchmark_multiworker.py \
        --label three-mock-workers
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

import app.main as main_module
from app.main import app
from app.workers import MockWorker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "multiworker"


class FakeSemanticRouter:
    def route(self, prompt: str) -> dict[str, Any]:
        return {
            "category": "general",
            "scores": {"coding": 0.1, "reasoning": 0.2, "general": 0.8},
            "top_score": 0.8,
            "margin": 0.6,
        }


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


async def one_request(
    client: httpx.AsyncClient,
    prompt: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "worker_id": response.headers.get("x-worker-id"),
            "elapsed_ms": elapsed_ms,
            "error": None if response.status_code == 200 else response.text[:200],
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "ok": False,
            "status_code": None,
            "worker_id": None,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_load(
    client: httpx.AsyncClient,
    *,
    concurrency: int,
    requests: int,
    prompt_prefix: str,
) -> dict[str, Any]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(requests):
        queue.put_nowait(index)
    results: list[dict[str, Any]] = []

    async def worker() -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(
                await one_request(
                    client,
                    f"{prompt_prefix} request {index}",
                )
            )

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall_seconds = time.perf_counter() - started

    latencies = [result["elapsed_ms"] for result in results]
    distribution = Counter(
        result["worker_id"] for result in results if result["worker_id"]
    )
    failures = [result for result in results if not result["ok"]]
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_seconds": round(wall_seconds, 6),
        "throughput_requests_per_second": round(requests / wall_seconds, 4)
        if wall_seconds > 0
        else None,
        "latency_ms": summarize(latencies),
        "latencies_ms": [round(value, 4) for value in latencies],
        "worker_distribution": dict(sorted(distribution.items())),
        "error_count": len(failures),
        "error_rate": round(len(failures) / requests, 6),
        "failures": failures[:10],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Multi-worker routing benchmark",
        "",
        f"- Label: `{result['label']}`",
        f"- Timestamp (UTC): `{result['timestamp_utc']}`",
        f"- Workers: {result['configuration']['workers']} mock workers",
        f"- Simulated worker latency: {result['configuration']['latency_ms']} ms",
        f"- Worker max concurrency: {result['configuration']['max_concurrency']}",
        "- Semantic router: embedding-free fake (routing overhead only)",
        "",
        "## Per-repetition results",
        "",
        "| Rep | Conc | Requests | req/s | p50 ms | p95 ms | p99 ms | Distribution | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for repetition in result["repetitions"]:
        for row in repetition["levels"]:
            stats = row["latency_ms"]
            lines.append(
                f"| {repetition['repetition']} | {row['concurrency']} | "
                f"{row['requests']} | {row['throughput_requests_per_second']} | "
                f"{stats.get('median')} | {stats.get('p95')} | {stats.get('p99')} | "
                f"`{row['worker_distribution']}` | {row['error_count']} |"
            )

    lines.extend(
        [
            "",
            "## Aggregated by concurrency",
            "",
            "| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Distribution | Errors |",
            "|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in result["aggregate"]:
        stats = row["latency_ms"]
        lines.append(
            f"| {row['concurrency']} | {row['requests']} | "
            f"{row['throughput_mean']} | {stats.get('median')} | "
            f"{stats.get('p95')} | {stats.get('p99')} | "
            f"`{row['worker_distribution']}` | {row['error_count']} |"
        )

    lines.extend(
        [
            "",
            "## Health and availability scenarios",
            "",
            "| Scenario | Requests | Success | Distribution | Status counts |",
            "|---|---:|---:|---|---|",
        ]
    )
    for scenario in result["scenarios"]:
        lines.append(
            f"| {scenario['name']} | {scenario['requests']} | "
            f"{scenario['successes']} | `{scenario['worker_distribution']}` | "
            f"`{scenario['status_codes']}` |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Mock workers require no GPU, API key, network or model download.",
            "- The embedding-free router isolates scheduler/registry/dispatch cost.",
            "- Worker failure scenarios are covered by the unit/integration suite; "
            "this benchmark reports health-state routing only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.disable(logging.CRITICAL)
    timestamp = utc_stamp()
    workers = [
        MockWorker(
            worker_id=f"mock-{index + 1}",
            model=f"mock-qwen-{index + 1}",
            latency_ms=args.latency_ms,
            max_concurrency=args.max_concurrency,
        )
        for index in range(args.workers)
    ]

    original_build_workers = main_module.build_workers
    original_router = main_module.SemanticRouter
    main_module.build_workers = lambda: workers
    main_module.SemanticRouter = FakeSemanticRouter

    lifespan = None
    try:
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://multiworker-benchmark",
            timeout=30.0,
        ) as client:
            for index in range(args.warmup):
                result = await one_request(client, f"warmup {index}")
                if not result["ok"]:
                    raise SystemExit(f"Warm-up failed: {result}")

            repetitions: list[dict[str, Any]] = []
            for repetition in range(1, args.repetitions + 1):
                levels = []
                for concurrency in args.concurrency:
                    row = await run_load(
                        client,
                        concurrency=concurrency,
                        requests=args.requests_per_level,
                        prompt_prefix=f"rep {repetition} conc {concurrency}",
                    )
                    levels.append(row)
                    print(
                        f"rep {repetition}/{args.repetitions} conc {concurrency}: "
                        f"{row['throughput_requests_per_second']} req/s, "
                        f"p50 {row['latency_ms'].get('median')} ms, "
                        f"dist {row['worker_distribution']}, errors {row['error_count']}"
                    )
                repetitions.append({"repetition": repetition, "levels": levels})

            aggregate = []
            for index, concurrency in enumerate(args.concurrency):
                rows = [rep["levels"][index] for rep in repetitions]
                latencies = [value for row in rows for value in row["latencies_ms"]]
                distribution = Counter()
                for row in rows:
                    distribution.update(row["worker_distribution"])
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
                        "worker_distribution": dict(sorted(distribution.items())),
                        "error_count": sum(row["error_count"] for row in rows),
                    }
                )

            scenarios = []
            registry = app.state.worker_registry

            async def capture_scenario(name: str, requests: int) -> dict[str, Any]:
                row = await run_load(
                    client,
                    concurrency=2,
                    requests=requests,
                    prompt_prefix=name,
                )
                statuses = Counter(
                    result["status_code"]
                    for result in row.get("failures", [])
                )
                successes = requests - row["error_count"]
                if successes:
                    statuses[200] = successes
                return {
                    "name": name,
                    "requests": requests,
                    "successes": successes,
                    "worker_distribution": row["worker_distribution"],
                    "status_codes": dict(sorted(statuses.items())),
                }

            workers[0].set_healthy(False)
            registry.mark_health(workers[0].metadata.worker_id, False, error="benchmark")
            scenarios.append(await capture_scenario("one_worker_unhealthy", 40))

            workers[0].set_healthy(True)
            registry.mark_health(workers[0].metadata.worker_id, True)
            scenarios.append(await capture_scenario("worker_recovered", 40))

            for worker in workers:
                registry.set_available(worker.metadata.worker_id, False)
            scenarios.append(await capture_scenario("all_workers_unavailable", 5))

            for worker in workers:
                registry.set_available(worker.metadata.worker_id, True)

            result = {
                "schema_version": 1,
                "label": args.label,
                "timestamp_utc": timestamp,
                "script": "scripts/benchmark_multiworker.py",
                "command": sys.argv,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                },
                "configuration": {
                    "workers": args.workers,
                    "latency_ms": args.latency_ms,
                    "max_concurrency": args.max_concurrency,
                    "concurrency_levels": args.concurrency,
                    "requests_per_level": args.requests_per_level,
                    "repetitions": args.repetitions,
                    "warmup": args.warmup,
                },
                "worker_metadata": [worker.metadata.as_dict() for worker in workers],
                "repetitions": repetitions,
                "aggregate": aggregate,
                "scenarios": scenarios,
                "notes": [
                    "Mock workers only; no GPU, cloud API key, network or model download.",
                    "Embedding-free router isolates scheduler/registry/dispatch overhead.",
                    "Health monitoring uses the real HealthMonitor unless state is set directly.",
                ],
            }

            output = (
                Path(args.output)
                if args.output
                else DEFAULT_OUTPUT_DIR
                / f"routing_{args.label.replace(' ', '_')}_{timestamp}.json"
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
        if lifespan is not None:
            await lifespan.__aexit__(None, None, None)
        main_module.build_workers = original_build_workers
        main_module.SemanticRouter = original_router
        logging.disable(logging.NOTSET)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--latency-ms", type=float, default=25.0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--concurrency", default="1,2,4,8")
    args = parser.parse_args(argv)
    args.concurrency = [
        int(value) for value in args.concurrency.split(",") if value.strip()
    ]
    if not args.concurrency:
        raise SystemExit("--concurrency must contain at least one level")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
