#!/usr/bin/env python3
"""CPU-only embedding and semantic-routing microbenchmark.

Loads one MiniLM model through SemanticRouter, measures encode/route latency,
and checks that concurrent route calls produce the same results as serial
calls. This does not exercise the FastAPI event loop; gateway event-loop
behaviour is measured by scripts/benchmark_worker1.py with --health-url.

Example::

    PYTHONPATH=. .venv/bin/python scripts/benchmark_embeddings.py \
        --output benchmarks/worker1_embedding.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from app.services.router import SemanticRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks"

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


def timed_call(function: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = function()
    return (time.perf_counter() - start) * 1000, result


def run_concurrency(
    router: SemanticRouter,
    prompts: list[str],
    serial_results: list[dict[str, Any]],
    workers: int,
    rounds: int,
) -> dict[str, Any]:
    calls = prompts * rounds

    def timed_route(prompt: str) -> tuple[float, dict[str, Any]]:
        return timed_call(lambda: router.route(prompt))

    latencies: list[float] = []
    results: list[dict[str, Any]] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for timing, result in executor.map(timed_route, calls):
            latencies.append(timing)
            results.append(result)
    wall_seconds = time.perf_counter() - start

    reference = [
        serial_results[index % len(serial_results)] for index in range(len(calls))
    ]
    categories_match = all(
        actual["category"] == expected["category"]
        for actual, expected in zip(results, reference)
    )
    max_score_difference = max(
        (
            abs(actual["top_score"] - expected["top_score"])
            for actual, expected in zip(results, reference)
        ),
        default=0.0,
    )
    return {
        "workers": workers,
        "rounds": rounds,
        "calls": len(calls),
        "wall_seconds": round(wall_seconds, 6),
        "throughput_calls_per_second": round(len(calls) / wall_seconds, 4)
        if wall_seconds > 0
        else None,
        "latency_ms": summarize(latencies),
        "categories_match_serial": categories_match,
        "max_top_score_difference": round(max_score_difference, 10),
        "scores_match_serial": max_score_difference < 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--encode-runs", type=int, default=50)
    parser.add_argument("--route-runs", type=int, default=50)
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    started = time.perf_counter()
    router = SemanticRouter()
    load_seconds = time.perf_counter() - started
    service = router.embedding_service

    # Warm up lazy tokenizer/model paths before recording.
    for _ in range(5):
        service.encode([PROMPTS[0]])
        router.route(PROMPTS[0])

    encode_latencies: list[float] = []
    for _ in range(args.encode_runs):
        elapsed, _ = timed_call(lambda: service.encode([PROMPTS[0]]))
        encode_latencies.append(elapsed)

    batch_latencies: list[float] = []
    for _ in range(max(10, args.encode_runs // 2)):
        elapsed, _ = timed_call(lambda: service.encode(PROMPTS[:4]))
        batch_latencies.append(elapsed)

    route_latencies: list[float] = []
    serial_results: list[dict[str, Any]] = []
    for index in range(args.route_runs):
        elapsed, result = timed_call(lambda: router.route(PROMPTS[index % len(PROMPTS)]))
        route_latencies.append(elapsed)
        serial_results.append(result)

    workers = [int(value) for value in args.workers.split(",") if value.strip()]
    concurrency = [
        run_concurrency(router, PROMPTS, serial_results, worker, args.rounds)
        for worker in workers
    ]

    # Batch-vs-single consistency for the embedding service.
    with torch.inference_mode():
        batch = service.encode(PROMPTS[:4])
        singles = torch.cat([service.encode([prompt]) for prompt in PROMPTS[:4]], dim=0)
        batch_single_max_abs_diff = float((batch - singles).abs().max().item())

    timestamp = utc_stamp()
    result = {
        "schema_version": 1,
        "timestamp_utc": timestamp,
        "script": "scripts/benchmark_embeddings.py",
        "command": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
        },
        "model": service.model_name,
        "load_seconds": round(load_seconds, 6),
        "embedding": {
            "shape": list(batch.shape),
            "norm_first": float(batch.norm(dim=1)[0].item()),
            "batch_single_max_abs_diff": round(batch_single_max_abs_diff, 10),
            "batch_single_consistent": batch_single_max_abs_diff < 1e-6,
            "encode_ms": summarize(encode_latencies),
            "encode_batch4_ms": summarize(batch_latencies),
        },
        "routing": {
            "route_ms": summarize(route_latencies),
            "categories": sorted(router.categories.keys()),
        },
        "concurrency": concurrency,
        "notes": [
            "CPU-only measurement; vLLM/GPU is not involved.",
            "ThreadPoolExecutor is used to test concurrent correctness and indicative latency.",
            "Production gateway execution uses Starlette/AnyIO's bounded thread pool.",
            "Concurrent results are compared against serial categories and top scores.",
        ],
    }

    output = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR / f"worker1_embedding_{timestamp}.json"
    )
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown = output.with_suffix(".md")
    lines = [
        "# Worker 1 embedding and routing microbenchmark",
        "",
        f"- Timestamp (UTC): `{timestamp}`",
        f"- Model: `{service.model_name}`",
        f"- Torch threads: {torch.get_num_threads()}",
        f"- Load time: {load_seconds:.3f} s",
        "",
        "## Latency",
        "",
        "| Measurement | p50 | p95 | p99 | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, stats in (
        ("Encode (1 prompt)", result["embedding"]["encode_ms"]),
        ("Encode (4 prompts)", result["embedding"]["encode_batch4_ms"]),
        ("Route", result["routing"]["route_ms"]),
    ):
        lines.append(
            f"| {label} | {stats.get('median')} | {stats.get('p95')} | "
            f"{stats.get('p99')} | {stats.get('count')} |"
        )
    lines.extend(
        [
            "",
            "## Concurrent routing",
            "",
            "| Workers | Calls | Wall s | Calls/s | p50 ms | p95 ms | Categories match | Scores match |",
            "|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in concurrency:
        stats = item["latency_ms"]
        lines.append(
            f"| {item['workers']} | {item['calls']} | {item['wall_seconds']} | "
            f"{item['throughput_calls_per_second']} | {stats.get('median')} | "
            f"{stats.get('p95')} | {item['categories_match_serial']} | "
            f"{item['scores_match_serial']} |"
        )
    lines.extend(
        [
            "",
            "## Consistency",
            "",
            f"- Batch vs single max abs diff: `{result['embedding']['batch_single_max_abs_diff']}`",
            f"- Batch vs single consistent: `{result['embedding']['batch_single_consistent']}`",
            "",
        ]
    )
    markdown.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({key: result[key] for key in (
        "load_seconds", "embedding", "routing", "concurrency"
    )}, indent=2))
    print(f"Saved: {output}")
    print(f"Saved: {markdown}")


if __name__ == "__main__":
    main()
