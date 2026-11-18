#!/usr/bin/env python3
"""Paired direct-vs-gateway benchmark for Worker 1 Stage 2.

Controls implemented here:

* Direct and gateway requests are paired back-to-back for the same workload.
* The order inside each pair is randomized with a recorded seed.
* Warm-prefix-cache and cold/varied-prompt measurements are separate.
* Cold measurements use a unique prompt for every request and for each path,
  so neither path benefits from a previously cached prefix. Prefix caching
  itself is left enabled and is never reconfigured.
* GPU temperature, utilization, power, SM clock and memory clock are sampled
  throughout the run.
* Workload, generation settings and speculative-decoding settings are fixed.

This script only issues HTTP requests and reads metrics. It never starts,
stops or reconfigures a server.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "stage2"
SHORT_PROMPT = "Explain what a CPU cache does and why it matters."

WORKLOADS: dict[str, dict[str, Any]] = {
    "A": {
        "name": "short",
        "prompt": SHORT_PROMPT,
        "max_tokens": 32,
        "long_tokens": None,
    },
    "B": {
        "name": "standard",
        "prompt": STANDARD_PROMPT,
        "max_tokens": 150,
        "long_tokens": None,
    },
    "C": {
        "name": "long",
        "prompt": None,
        "max_tokens": 100,
        "long_tokens": 800,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_cold_prompt(workload: str, nonce: str) -> str:
    base = WORKLOADS[workload]["prompt"]
    return f"Request {nonce}. {base}"


def paired_delta(
    pairs: list[dict[str, Any]],
    direct_key: str,
    gateway_key: str,
    metric: str,
) -> dict[str, Any]:
    deltas = []
    percentages = []
    for pair in pairs:
        direct = pair.get(direct_key) or {}
        gateway = pair.get(gateway_key) or {}
        if not direct.get("ok") or not gateway.get("ok"):
            continue
        direct_value = direct.get(metric)
        gateway_value = gateway.get(metric)
        if direct_value is None or gateway_value is None:
            continue
        delta = gateway_value - direct_value
        deltas.append(round(delta, 4))
        if direct_value != 0:
            percentages.append(round(100.0 * delta / abs(direct_value), 4))
    result = summarize(deltas)
    result["n_pairs"] = len(deltas)
    result["median_percent"] = (
        round(statistics.median(percentages), 4) if percentages else None
    )
    return result


def summarize_pair_group(
    pairs: list[dict[str, Any]],
    direct_key: str,
    gateway_key: str,
) -> dict[str, Any]:
    direct_results = [pair[direct_key] for pair in pairs if pair.get(direct_key)]
    gateway_results = [pair[gateway_key] for pair in pairs if pair.get(gateway_key)]
    direct_tokens = sum(
        result.get("completion_tokens") or 0
        for result in direct_results
        if result.get("ok")
    )
    gateway_tokens = sum(
        result.get("completion_tokens") or 0
        for result in gateway_results
        if result.get("ok")
    )
    direct_seconds = sum(
        (result.get("total_ms") or 0.0) / 1000.0
        for result in direct_results
        if result.get("ok")
    )
    gateway_seconds = sum(
        (result.get("total_ms") or 0.0) / 1000.0
        for result in gateway_results
        if result.get("ok")
    )
    return {
        "direct": summarize_requests(direct_results),
        "gateway": summarize_requests(gateway_results),
        "paired_delta_gateway_minus_direct": {
            metric: paired_delta(pairs, direct_key, gateway_key, metric)
            for metric in ("ttft_ms", "total_ms", "tokens_per_second")
        },
        "aggregate": {
            "direct_completion_tokens": direct_tokens,
            "gateway_completion_tokens": gateway_tokens,
            "direct_active_seconds": round(direct_seconds, 6),
            "gateway_active_seconds": round(gateway_seconds, 6),
            "direct_tokens_per_second": round(direct_tokens / direct_seconds, 6)
            if direct_seconds > 0
            else None,
            "gateway_tokens_per_second": round(gateway_tokens / gateway_seconds, 6)
            if gateway_seconds > 0
            else None,
        },
        "orders": {
            "direct_first": sum(1 for pair in pairs if pair["order"][0] == "direct"),
            "gateway_first": sum(1 for pair in pairs if pair["order"][0] == "gateway"),
        },
    }


async def run_group(
    workload: str,
    mode: str,
    args: argparse.Namespace,
    rng: random.Random,
    direct_client: httpx.AsyncClient,
    gateway_client: httpx.AsyncClient,
    metrics_client: httpx.AsyncClient,
    request_counter: list[int],
    long_prompt_info: dict[str, Any],
) -> dict[str, Any]:
    config = WORKLOADS[workload]
    base_prompt = config["prompt"]
    if base_prompt is None:
        base_prompt = long_prompt_info["text"]
    max_tokens = config["max_tokens"]
    namespace = SimpleNamespace(model=args.model, temperature=args.temperature)

    print(
        f"\n=== Workload {workload} ({config['name']}), mode={mode}, "
        f"max_tokens={max_tokens}, pairs={args.pairs} ===",
        flush=True,
    )

    # Warm up kernels and, for warm mode, the shared prefix cache.
    warmups = []
    for warmup_index in range(args.warmup_pairs):
        order = ["direct", "gateway"] if rng.random() < 0.5 else ["gateway", "direct"]
        for path in order:
            client = direct_client if path == "direct" else gateway_client
            url = (
                (args.direct_url if path == "direct" else args.gateway_url).rstrip("/")
                + "/v1/chat/completions"
            )
            payload = build_payload(namespace, base_prompt, max_tokens)
            request_counter[0] += 1
            result = await run_request(
                client,
                url,
                payload,
                request_counter[0],
                capture_headers=path == "gateway",
            )
            result.update({"path": path, "phase": "warmup", "warmup_index": warmup_index})
            warmups.append(result)
            print(
                f"Warmup {warmup_index + 1}/{args.warmup_pairs} {path}: "
                f"{'ok' if result['ok'] else 'FAILED'} "
                f"TTFT {result.get('ttft_ms')} total {result.get('total_ms')} ms",
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

    pairs: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for pair_index in range(args.pairs):
            order = ["direct", "gateway"] if rng.random() < 0.5 else ["gateway", "direct"]
            prompts: dict[str, str] = {}
            prompt_ids: dict[str, str] = {}
            if mode == "warm":
                prompts["direct"] = base_prompt
                prompts["gateway"] = base_prompt
                prompt_ids["direct"] = "warm-shared"
                prompt_ids["gateway"] = "warm-shared"
            else:
                for path in ("direct", "gateway"):
                    request_counter[0] += 1
                    nonce = (
                        f"{workload}-{mode}-{request_counter[0]:05d}-"
                        f"{rng.randrange(1_000_000):06d}"
                    )
                    prompts[path] = build_cold_prompt(workload, nonce)
                    prompt_ids[path] = nonce

            pair: dict[str, Any] = {
                "pair_index": pair_index,
                "order": order,
                "prompt_ids": prompt_ids,
            }
            for path in order:
                client = direct_client if path == "direct" else gateway_client
                url = (
                    (args.direct_url if path == "direct" else args.gateway_url).rstrip("/")
                    + "/v1/chat/completions"
                )
                payload = build_payload(namespace, prompts[path], max_tokens)
                request_counter[0] += 1
                result = await run_request(
                    client,
                    url,
                    payload,
                    request_counter[0],
                    capture_headers=path == "gateway",
                )
                result.update(
                    {
                        "path": path,
                        "phase": "measured",
                        "pair_index": pair_index,
                        "order_index": order.index(path),
                        "workload": workload,
                        "mode": mode,
                        "prompt_id": prompt_ids[path],
                    }
                )
                pair[path] = result

            pairs.append(pair)
            direct = pair["direct"]
            gateway = pair["gateway"]
            print(
                f"Pair {pair_index + 1:02d}/{args.pairs} order={'/'.join(order)} | "
                f"direct {'ok' if direct['ok'] else 'FAIL'} "
                f"TTFT {direct.get('ttft_ms')} total {direct.get('total_ms')} ms | "
                f"gateway {'ok' if gateway['ok'] else 'FAIL'} "
                f"TTFT {gateway.get('ttft_ms')} total {gateway.get('total_ms')} ms"
                + (
                    f" | route {gateway.get('route_category')}"
                    if gateway.get("route_category")
                    else ""
                ),
                flush=True,
            )
    finally:
        stop_event.set()
        await asyncio.gather(sampler, return_exceptions=True)

    elapsed = time.perf_counter() - started
    after_text = await fetch_text(metrics_client, args.metrics_url)
    group = summarize_pair_group(pairs, "direct", "gateway")
    group.update(
        {
            "workload": workload,
            "workload_name": config["name"],
            "mode": mode,
            "max_tokens": max_tokens,
            "pairs": pairs,
            "warmups": warmups,
            "timed_wall_seconds": round(elapsed, 6),
            "speculative_decoding": build_acceptance(before_text, after_text),
            "kv_cache": {
                "summary": summarize_kv(metric_samples),
                "samples": metric_samples,
            },
            "route_categories": sorted(
                {
                    pair["gateway"].get("route_category")
                    for pair in pairs
                    if pair.get("gateway") and pair["gateway"].get("route_category")
                }
            ),
        }
    )
    return group


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Worker 1 Stage 2 paired benchmark",
        "",
        f"- Label: `{result['label']}`",
        f"- Timestamp (UTC): `{result['timestamp_utc']}`",
        f"- vLLM KV: {result['vllm_config']['kv_mib']} MiB, "
        f"max-num-seqs {result['vllm_config']['max_num_seqs']}",
        f"- Randomization seed: `{result['protocol']['seed']}`",
        f"- Pairs per workload/mode: {result['protocol']['pairs']}",
        f"- Warmup pairs: {result['protocol']['warmup_pairs']}",
        "",
        "Direct and gateway requests are adjacent within each pair. The order is",
        "randomized per pair. Cold mode uses a unique prompt for every request and",
        "for each path; prefix caching remains enabled but cannot reuse a prefix.",
        "",
    ]
    for workload in result["protocol"]["workloads"]:
        lines.append(f"## Workload {workload}: {result['workloads'][workload]['name']}")
        lines.append("")
        for mode in result["protocol"]["modes"]:
            group = result["workloads"][workload][mode]
            direct = group["direct"]
            gateway = group["gateway"]
            delta = group["paired_delta_gateway_minus_direct"]
            spec = group["speculative_decoding"]
            kv = group["kv_cache"]["summary"]
            kv_usage = kv.get("kv_cache_usage_perc") or {}
            kv_median = kv_usage.get("median")
            kv_p95 = kv_usage.get("p95")

            def stat(summary: dict[str, Any], key: str) -> str:
                stats = summary.get(key) or {}
                if not stats.get("count"):
                    return "n/a"
                return f"{stats['median']:.2f} / {stats['p95']:.2f}"

            def percent(value: Any) -> str:
                if value is None:
                    return "n/a"
                return f"{100.0 * value:.2f}"

            lines.extend(
                [
                    f"### {mode.capitalize()} mode",
                    "",
                    "| Metric | Direct | Gateway | Paired delta median |",
                    "|---|---:|---:|---:|",
                    f"| Success | {direct['requests_successful']}/{direct['requests_total']} | "
                    f"{gateway['requests_successful']}/{gateway['requests_total']} | — |",
                    f"| TTFT p50/p95 (ms) | {stat(direct, 'ttft_ms')} | "
                    f"{stat(gateway, 'ttft_ms')} | "
                    f"{delta['ttft_ms'].get('median', 'n/a')} ms |",
                    f"| Total p50/p95 (ms) | {stat(direct, 'total_ms')} | "
                    f"{stat(gateway, 'total_ms')} | "
                    f"{delta['total_ms'].get('median', 'n/a')} ms |",
                    f"| Per-request tok/s p50/p95 | "
                    f"{stat(direct, 'tokens_per_second_per_request')} | "
                    f"{stat(gateway, 'tokens_per_second_per_request')} | "
                    f"{delta['tokens_per_second'].get('median', 'n/a')} |",
                    f"| Aggregate tok/s | {group['aggregate']['direct_tokens_per_second']} | "
                    f"{group['aggregate']['gateway_tokens_per_second']} | — |",
                    f"| Pair order | {group['orders']['direct_first']} direct-first | "
                    f"{group['orders']['gateway_first']} gateway-first | — |",
                    f"| Spec acceptance | {percent(spec.get('acceptance_rate'))}% | "
                    f"(shared counters) | — |",
                    f"| Mean acceptance length | {spec.get('mean_acceptance_length')} | "
                    f"(shared counters) | — |",
                    f"| KV usage p50/p95 (%) | {percent(kv_median)} / {percent(kv_p95)} | "
                    f"(shared) | — |",
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
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--direct-url", default="http://localhost:8000")
    parser.add_argument("--gateway-url", default="http://localhost:8080")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--pairs", type=int, default=15)
    parser.add_argument("--warmup-pairs", type=int, default=1)
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

    long_prompt_text, long_prompt_tokens, long_prompt_error = build_long_prompt(
        args.model, WORKLOADS["C"]["long_tokens"] or 800
    )
    WORKLOADS["C"]["prompt"] = long_prompt_text
    long_prompt_info = {
        "text": long_prompt_text,
        "tokenizer_token_count": long_prompt_tokens,
        "tokenizer_error": long_prompt_error,
    }

    gpu = await gpu_identity()
    print(f"GPU: {gpu}")
    print(f"Seed: {seed}")
    print(f"Workloads: {workloads}; modes: {modes}")
    print(f"KV: {args.kv_mib} MiB; max-num-seqs: {args.max_num_seqs}")

    timeout = httpx.Timeout(connect=10.0, read=args.timeout, write=30.0, pool=10.0)
    direct_client = httpx.AsyncClient(timeout=timeout)
    gateway_client = httpx.AsyncClient(timeout=timeout)
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
                group = await run_group(
                    workload,
                    mode,
                    args,
                    rng,
                    direct_client,
                    gateway_client,
                    metrics_client,
                    request_counter,
                    long_prompt_info,
                )
                all_groups[workload][mode] = group
    finally:
        stop_event.set()
        await asyncio.gather(gpu_task, return_exceptions=True)
        await direct_client.aclose()
        await gateway_client.aclose()
        await metrics_client.aclose()

    elapsed = time.perf_counter() - run_started
    result: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "timestamp_utc": timestamp,
        "script": "scripts/benchmark_stage2.py",
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
            "note": "Prefix caching was left at the server default and was not reconfigured.",
        },
        "protocol": {
            "seed": seed,
            "pairs": args.pairs,
            "warmup_pairs": args.warmup_pairs,
            "workloads": workloads,
            "modes": modes,
            "randomized": "direct/gateway order randomized independently for every pair",
            "warm_prompt": "same prompt for every request, prefix cache intentionally warm",
            "cold_prompt": "unique nonce prefix for every request and every path",
            "temperature": args.temperature,
            "max_tokens": {
                workload: WORKLOADS[workload]["max_tokens"] for workload in workloads
            },
            "long_prompt": long_prompt_info,
            "timed_wall_seconds": round(elapsed, 6),
        },
        "workloads": all_groups,
        "gpu": {
            "identity": gpu,
            "summary": summarize_gpu(gpu_samples),
            "samples": gpu_samples,
        },
        "notes": [
            "Requests inside a pair run back-to-back; order is randomized per pair.",
            "Paired deltas are gateway minus direct for adjacent requests, which reduces"
            " thermal drift compared with two long separate batches.",
            "Aggregate throughput is completion tokens divided by summed successful request"
            " wall time; it is not a same-instant multi-client throughput measurement.",
            "Cold mode varies the prompt prefix so prefix caching cannot produce a hit;"
            " the server's prefix-cache configuration was not changed.",
            "GPU telemetry covers warmups and timed requests.",
        ],
    }

    output = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"paired_{args.label}_{timestamp}.json"
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
