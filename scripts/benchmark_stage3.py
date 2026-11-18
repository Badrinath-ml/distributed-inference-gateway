#!/usr/bin/env python3
"""Concurrent paired benchmark for Worker 1 max-num-seqs experiments.

Each measured unit is a batch of `--concurrency` simultaneous chat requests
sent to one path, followed immediately by a batch of the same size to the other
path. Path order is randomized per batch pair. This keeps the comparison close
in time while avoiding both paths sharing the same vLLM batch.

Warm mode reuses one prompt (prefix cache warm). Cold mode uses a unique prompt
for every request. Prefix caching is never reconfigured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from scripts.benchmark_worker1 import (
    STANDARD_PROMPT,
    build_acceptance,
    build_long_prompt,
    build_payload,
    fetch_text,
    gpu_identity,
    gpu_sample,
    metrics_sampler,
    run_request,
    summarize,
    summarize_gpu,
    summarize_kv,
    summarize_requests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "stage3"
SHORT_PROMPT = "Explain what a CPU cache does and why it matters."

WORKLOADS: dict[str, dict[str, Any]] = {
    "A": {"name": "short", "prompt": SHORT_PROMPT, "max_tokens": 32, "long_tokens": None},
    "B": {"name": "standard", "prompt": STANDARD_PROMPT, "max_tokens": 150, "long_tokens": None},
    "C": {"name": "long", "prompt": None, "max_tokens": 100, "long_tokens": 800},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cold_prompt(workload: str, nonce: str) -> str:
    return f"Request {nonce}. {WORKLOADS[workload]['prompt']}"


def summarize_batch(
    batch_index: int,
    path: str,
    wall_seconds: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    completion_tokens = sum(
        result.get("completion_tokens") or 0 for result in results if result.get("ok")
    )
    return {
        "batch_index": batch_index,
        "path": path,
        "wall_seconds": round(wall_seconds, 6),
        "completion_tokens": completion_tokens,
        "aggregate_tokens_per_second": round(completion_tokens / wall_seconds, 6)
        if wall_seconds > 0
        else None,
        "requests_successful": sum(1 for result in results if result.get("ok")),
        "requests_total": len(results),
        "ttft_ms": summarize([result["ttft_ms"] for result in results if result.get("ok")]),
        "total_ms": summarize([result["total_ms"] for result in results if result.get("ok")]),
        "requests": results,
        "route_categories": sorted(
            {
                result.get("route_category")
                for result in results
                if result.get("route_category")
            }
        ),
    }


def summarize_group(batches: list[dict[str, Any]]) -> dict[str, Any]:
    direct_batches = [batch["direct"] for batch in batches]
    gateway_batches = [batch["gateway"] for batch in batches]
    direct_results = [result for batch in direct_batches for result in batch["requests"]]
    gateway_results = [result for batch in gateway_batches for result in batch["requests"]]
    direct_tokens = sum(batch["completion_tokens"] for batch in direct_batches)
    gateway_tokens = sum(batch["completion_tokens"] for batch in gateway_batches)
    direct_wall = sum(batch["wall_seconds"] for batch in direct_batches)
    gateway_wall = sum(batch["wall_seconds"] for batch in gateway_batches)
    wall_deltas = [
        round((batch["gateway"]["wall_seconds"] - batch["direct"]["wall_seconds"]) * 1000.0, 4)
        for batch in batches
    ]
    throughput_deltas = [
        round(
            (batch["gateway"]["aggregate_tokens_per_second"] or 0.0)
            - (batch["direct"]["aggregate_tokens_per_second"] or 0.0),
            4,
        )
        for batch in batches
    ]
    return {
        "direct": summarize_requests(direct_results),
        "gateway": summarize_requests(gateway_results),
        "aggregate": {
            "direct_completion_tokens": direct_tokens,
            "gateway_completion_tokens": gateway_tokens,
            "direct_batch_seconds": round(direct_wall, 6),
            "gateway_batch_seconds": round(gateway_wall, 6),
            "direct_tokens_per_second": round(direct_tokens / direct_wall, 6)
            if direct_wall > 0
            else None,
            "gateway_tokens_per_second": round(gateway_tokens / gateway_wall, 6)
            if gateway_wall > 0
            else None,
        },
        "batch_wall_ms": {
            "direct": summarize([batch["wall_seconds"] * 1000.0 for batch in direct_batches]),
            "gateway": summarize([batch["wall_seconds"] * 1000.0 for batch in gateway_batches]),
            "paired_delta": summarize(wall_deltas),
        },
        "batch_throughput": {
            "direct": summarize(
                [
                    batch["aggregate_tokens_per_second"]
                    for batch in direct_batches
                    if batch["aggregate_tokens_per_second"] is not None
                ]
            ),
            "gateway": summarize(
                [
                    batch["aggregate_tokens_per_second"]
                    for batch in gateway_batches
                    if batch["aggregate_tokens_per_second"] is not None
                ]
            ),
            "paired_delta": summarize(throughput_deltas),
        },
        "orders": {
            "direct_first": sum(1 for batch in batches if batch["order"][0] == "direct"),
            "gateway_first": sum(1 for batch in batches if batch["order"][0] == "gateway"),
        },
        "route_categories": sorted(
            {
                category
                for batch in gateway_batches
                for category in batch["route_categories"]
            }
        ),
    }


async def run_batch(
    path: str,
    prompts: list[str],
    max_tokens: int,
    args: argparse.Namespace,
    clients: dict[str, httpx.AsyncClient],
    request_counter: list[int],
    batch_index: int,
) -> dict[str, Any]:
    namespace = SimpleNamespace(model=args.model, temperature=args.temperature)
    url = (
        (args.direct_url if path == "direct" else args.gateway_url).rstrip("/")
        + "/v1/chat/completions"
    )
    started = time.perf_counter()
    results = await asyncio.gather(
        *[
            run_request(
                clients[path],
                url,
                build_payload(namespace, prompt, max_tokens),
                request_counter[0] + index + 1,
                capture_headers=path == "gateway",
            )
            for index, prompt in enumerate(prompts)
        ]
    )
    wall = time.perf_counter() - started
    request_counter[0] += len(prompts)
    batch = summarize_batch(batch_index, path, wall, list(results))
    batch["concurrency"] = len(prompts)
    return batch


async def run_group(
    workload: str,
    mode: str,
    args: argparse.Namespace,
    rng: random.Random,
    clients: dict[str, httpx.AsyncClient],
    metrics_client: httpx.AsyncClient,
    request_counter: list[int],
    long_prompt_text: str,
) -> dict[str, Any]:
    config = WORKLOADS[workload]
    base_prompt = config["prompt"] or long_prompt_text
    max_tokens = config["max_tokens"]
    print(
        f"\n=== Workload {workload} ({config['name']}), mode={mode}, "
        f"concurrency={args.concurrency}, batches={args.batches} ===",
        flush=True,
    )

    warmups = []
    for warmup_index in range(args.warmup_batches):
        order = ["direct", "gateway"] if rng.random() < 0.5 else ["gateway", "direct"]
        for path in order:
            prompts = [base_prompt] * args.concurrency
            batch = await run_batch(
                path, prompts, max_tokens, args, clients, request_counter, -1
            )
            batch["phase"] = "warmup"
            batch["warmup_index"] = warmup_index
            warmups.append(batch)
            print(
                f"Warmup {warmup_index + 1}/{args.warmup_batches} {path}: "
                f"{batch['requests_successful']}/{batch['requests_total']} ok, "
                f"wall {batch['wall_seconds']:.3f}s, "
                f"{batch['aggregate_tokens_per_second']} tok/s",
                flush=True,
            )

    before_text = await fetch_text(metrics_client, args.metrics_url)
    stop_event = asyncio.Event()
    metric_samples: list[dict[str, Any]] = []
    sampler = asyncio.create_task(
        metrics_sampler(
            metrics_client,
            args.metrics_url,
            args.metrics_sample_interval,
            stop_event,
            metric_samples,
        )
    )

    batches: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for batch_index in range(args.batches):
            order = ["direct", "gateway"] if rng.random() < 0.5 else ["gateway", "direct"]
            prompts_by_path: dict[str, list[str]] = {}
            for path in ("direct", "gateway"):
                prompts = []
                for _ in range(args.concurrency):
                    if mode == "warm":
                        prompts.append(base_prompt)
                    else:
                        request_counter[0] += 1
                        nonce = (
                            f"{workload}-{mode}-{batch_index:03d}-"
                            f"{request_counter[0]:05d}-{rng.randrange(1_000_000):06d}"
                        )
                        prompts.append(cold_prompt(workload, nonce))
                prompts_by_path[path] = prompts

            pair: dict[str, Any] = {"batch_index": batch_index, "order": order}
            for path in order:
                pair[path] = await run_batch(
                    path,
                    prompts_by_path[path],
                    max_tokens,
                    args,
                    clients,
                    request_counter,
                    batch_index,
                )
            batches.append(pair)
            direct = pair["direct"]
            gateway = pair["gateway"]
            print(
                f"Batch {batch_index + 1:02d}/{args.batches} order={'/'.join(order)} | "
                f"direct {direct['requests_successful']}/{direct['requests_total']} "
                f"wall {direct['wall_seconds']:.3f}s {direct['aggregate_tokens_per_second']} tok/s | "
                f"gateway {gateway['requests_successful']}/{gateway['requests_total']} "
                f"wall {gateway['wall_seconds']:.3f}s {gateway['aggregate_tokens_per_second']} tok/s",
                flush=True,
            )
    finally:
        stop_event.set()
        await asyncio.gather(sampler, return_exceptions=True)

    elapsed = time.perf_counter() - started
    after_text = await fetch_text(metrics_client, args.metrics_url)
    group = summarize_group(batches)
    group.update(
        {
            "workload": workload,
            "workload_name": config["name"],
            "mode": mode,
            "concurrency": args.concurrency,
            "max_tokens": max_tokens,
            "batches": batches,
            "warmups": warmups,
            "timed_wall_seconds": round(elapsed, 6),
            "speculative_decoding": build_acceptance(before_text, after_text),
            "kv_cache": {"summary": summarize_kv(metric_samples), "samples": metric_samples},
        }
    )
    return group


def write_markdown(result: dict[str, Any], path: Path) -> None:
    config = result["vllm_config"]
    lines = [
        "# Worker 1 Stage 3 concurrent paired benchmark",
        "",
        f"- Label: `{result['label']}`",
        f"- Timestamp (UTC): `{result['timestamp_utc']}`",
        f"- KV: {config['kv_mib']} MiB; max-num-seqs: {config['max_num_seqs']}",
        f"- Concurrency per batch: {result['protocol']['concurrency']}",
        f"- Batches per workload/mode: {result['protocol']['batches']}",
        f"- Seed: `{result['protocol']['seed']}`",
        "",
        "Each batch sends the configured number of simultaneous requests to one path;",
        "the paired batch is sent to the other path immediately afterwards.",
        "",
    ]
    for workload in result["protocol"]["workloads"]:
        info = result["workloads"][workload]
        lines.extend([f"## Workload {workload}: {info['name']}", ""])
        for mode in result["protocol"]["modes"]:
            group = info[mode]
            direct = group["direct"]
            gateway = group["gateway"]
            agg = group["aggregate"]
            wall = group["batch_wall_ms"]
            bw = group["batch_throughput"]
            kv = group["kv_cache"]["summary"]
            running = kv.get("num_requests_running") or {}
            waiting = kv.get("num_requests_waiting") or {}

            def stats(summary: dict[str, Any]) -> str:
                if not summary.get("count"):
                    return "n/a"
                return f"{summary['median']:.2f} / {summary['p95']:.2f}"

            lines.extend(
                [
                    f"### {mode.capitalize()} mode",
                    "",
                    "| Metric | Direct | Gateway | Paired delta |",
                    "|---|---:|---:|---:|",
                    f"| Success | {direct['requests_successful']}/{direct['requests_total']} | "
                    f"{gateway['requests_successful']}/{gateway['requests_total']} | — |",
                    f"| Batch wall p50/p95 (ms) | {stats(wall['direct'])} | "
                    f"{stats(wall['gateway'])} | "
                    f"{wall['paired_delta'].get('median', 'n/a')} ms |",
                    f"| Batch tok/s p50/p95 | {stats(bw['direct'])} | {stats(bw['gateway'])} | "
                    f"{bw['paired_delta'].get('median', 'n/a')} |",
                    f"| Aggregate tok/s | {agg['direct_tokens_per_second']} | "
                    f"{agg['gateway_tokens_per_second']} | — |",
                    f"| Per-request TTFT p50/p95 (ms) | {stats(direct['ttft_ms'])} | "
                    f"{stats(gateway['ttft_ms'])} | — |",
                    f"| Per-request total p50/p95 (ms) | {stats(direct['total_ms'])} | "
                    f"{stats(gateway['total_ms'])} | — |",
                    f"| Running requests p50/p95 | {stats(running)} | (shared) | — |",
                    f"| Waiting requests p50/p95 | {stats(waiting)} | (shared) | — |",
                    f"| KV usage p95 (%) | "
                    f"{100.0 * (kv.get('kv_cache_usage_perc') or {}).get('p95', 0.0):.2f} | "
                    f"(shared) | — |",
                    f"| Spec acceptance | {100.0 * (group['speculative_decoding'].get('acceptance_rate') or 0.0):.2f}% | "
                    f"(shared) | — |",
                    f"| Batch order | {group['orders']['direct_first']} direct-first | "
                    f"{group['orders']['gateway_first']} gateway-first | — |",
                    "",
                ]
            )
            failures = direct.get("failures", []) + gateway.get("failures", [])
            if failures:
                lines.append(f"Failures: `{failures}`")
                lines.append("")

    gpu = result["gpu"]["summary"]
    lines.extend(
        [
            "## GPU telemetry",
            "",
            f"- Samples: {gpu.get('sample_count')}",
            f"- Temperature C: `{gpu.get('temperature_c')}`",
            f"- Utilization %: `{gpu.get('utilization_gpu_pct')}`",
            f"- Power W: `{gpu.get('power_w')}`",
            f"- SM clock MHz: `{gpu.get('clock_sm_mhz')}`",
            f"- Memory clock MHz: `{gpu.get('clock_mem_mhz')}`",
            f"- Memory used MiB: `{gpu.get('memory_used_mib')}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in result["notes"])
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kv-mib", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--direct-url", default="http://localhost:8000")
    parser.add_argument("--gateway-url", default="http://localhost:8080")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workloads", default="A,B,C")
    parser.add_argument("--modes", default="warm,cold")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--metrics-sample-interval", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    if args.concurrency < 2:
        raise SystemExit("--concurrency must be >= 2 for the concurrency experiment")
    timestamp = utc_now()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 2**31)
    rng = random.Random(seed)
    workloads = [item.strip().upper() for item in args.workloads.split(",") if item.strip()]
    modes = [item.strip().lower() for item in args.modes.split(",") if item.strip()]
    for workload in workloads:
        if workload not in WORKLOADS:
            raise SystemExit(f"Unknown workload: {workload}")
    for mode in modes:
        if mode not in {"warm", "cold"}:
            raise SystemExit(f"Unknown mode: {mode}")

    long_prompt_text, long_tokens, long_error = build_long_prompt(
        args.model, WORKLOADS["C"]["long_tokens"] or 800
    )
    WORKLOADS["C"]["prompt"] = long_prompt_text

    gpu = await gpu_identity()
    print(f"GPU: {gpu}")
    print(f"Seed: {seed}")
    print(
        f"KV {args.kv_mib} MiB, max-num-seqs {args.max_num_seqs}, "
        f"concurrency {args.concurrency}, batches {args.batches}"
    )

    timeout = httpx.Timeout(connect=10.0, read=args.timeout, write=30.0, pool=10.0)
    clients = {
        "direct": httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max(10, args.concurrency * 2),
                max_keepalive_connections=max(5, args.concurrency),
            ),
        ),
        "gateway": httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max(10, args.concurrency * 2),
                max_keepalive_connections=max(5, args.concurrency),
            ),
        ),
    }
    metrics_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))

    stop_event = asyncio.Event()
    gpu_samples: list[dict[str, Any]] = []
    gpu_task = asyncio.create_task(
        gpu_sample(args.gpu_sample_interval, stop_event, gpu_samples)
    )

    request_counter = [0]
    all_groups: dict[str, Any] = {}
    run_started = time.perf_counter()
    try:
        for workload in workloads:
            all_groups[workload] = {
                "name": WORKLOADS[workload]["name"],
                "max_tokens": WORKLOADS[workload]["max_tokens"],
            }
            for mode in modes:
                all_groups[workload][mode] = await run_group(
                    workload,
                    mode,
                    args,
                    rng,
                    clients,
                    metrics_client,
                    request_counter,
                    long_prompt_text,
                )
    finally:
        stop_event.set()
        await asyncio.gather(gpu_task, return_exceptions=True)
        for client in clients.values():
            await client.aclose()
        await metrics_client.aclose()

    elapsed = time.perf_counter() - run_started
    result: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "timestamp_utc": timestamp,
        "script": "scripts/benchmark_stage3.py",
        "command": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "endpoints": {
            "direct_url": args.direct_url,
            "gateway_url": args.gateway_url,
            "metrics_url": args.metrics_url,
        },
        "model": args.model,
        "vllm_config": {
            "kv_mib": args.kv_mib,
            "kv_bytes": args.kv_mib * 1024 * 1024,
            "max_num_seqs": args.max_num_seqs,
            "num_speculative_tokens": 2,
            "max_model_len": 1024,
            "prefix_caching_changed": False,
        },
        "protocol": {
            "seed": seed,
            "concurrency": args.concurrency,
            "batches": args.batches,
            "warmup_batches": args.warmup_batches,
            "workloads": workloads,
            "modes": modes,
            "long_prompt": {
                "tokenizer_token_count": long_tokens,
                "tokenizer_error": long_error,
            },
            "timed_wall_seconds": round(elapsed, 6),
        },
        "workloads": all_groups,
        "gpu": {"identity": gpu, "summary": summarize_gpu(gpu_samples), "samples": gpu_samples},
        "notes": [
            "Each paired unit is a batch of concurrent requests to one path followed by"
            " an equally sized batch to the other path; batch order is randomized.",
            "With max-num-seqs=1, the second simultaneous request is expected to wait.",
            "Cold mode varies the prompt prefix so prefix caching cannot produce a hit.",
            "GPU telemetry covers warmups and timed batches.",
        ],
    }

    output = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"concurrent_{args.label}_{timestamp}.json"
    )
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    write_markdown(result, markdown_path)
    print(f"\nSaved: {output}")
    print(f"Saved: {markdown_path}")
    return result


def main() -> int:
    args = parse_args()
    asyncio.run(async_main(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
