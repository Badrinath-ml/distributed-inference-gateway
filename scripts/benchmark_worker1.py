#!/usr/bin/env python3
"""Worker 1 benchmark harness.

Measures streaming chat-completion latency, throughput, request success,
speculative-decoding counters, KV-cache utilisation and GPU telemetry.

This script only issues HTTP requests. It never starts, stops or reconfigures
an inference server.

Examples
--------
Direct vLLM, single request workload::

    PYTHONPATH=. .venv/bin/python scripts/benchmark_worker1.py \
        --base-url http://localhost:8000 \
        --metrics-url http://localhost:8000/metrics \
        --label baseline-direct

Two concurrent clients, ten repeated batches::

    PYTHONPATH=. .venv/bin/python scripts/benchmark_worker1.py \
        --mode concurrent --concurrency 2 --batches 10 \
        --base-url http://localhost:8080 \
        --metrics-url http://localhost:8000/metrics \
        --label gateway-concurrency

Long-context capacity check::

    PYTHONPATH=. .venv/bin/python scripts/benchmark_worker1.py \
        --mode single --long-context --long-prompt-tokens 800 \
        --long-max-tokens 100 --runs 3 \
        --base-url http://localhost:8000 \
        --metrics-url http://localhost:8000/metrics \
        --label long-context
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks"

STANDARD_PROMPT = (
    "Explain speculative decoding in LLM inference. "
    "Describe the roles of draft and target models."
)

LONG_PROMPT_BASE = (
    "Speculative decoding uses a small draft model to propose several tokens "
    "which the larger target model verifies in a single forward pass. "
    "The method can reduce decode latency when the draft model is accurate, "
    "but it adds memory and scheduling overhead. "
)

# Counter/gauge names used for speculative decoding and capacity reporting.
SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
KV_USAGE = "vllm:kv_cache_usage_perc"
REQUESTS_RUNNING = "vllm:num_requests_running"
REQUESTS_WAITING = "vllm:num_requests_waiting"
GENERATION_TOKENS = "vllm:generation_tokens_total"
PROMPT_TOKENS = "vllm:prompt_tokens_total"
REQUEST_SUCCESS = "vllm:request_success_total"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile; p is in [0, 1]."""
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


def summarize(values: list[float | int]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"count": 0}
    return {
        "count": len(numeric),
        "min": round(min(numeric), 4),
        "mean": round(statistics.mean(numeric), 4),
        "median": round(statistics.median(numeric), 4),
        "p95": round(percentile(numeric, 0.95), 4),
        "p99": round(percentile(numeric, 0.99), 4),
        "max": round(max(numeric), 4),
        "p99_reliable": len(numeric) >= 100,
    }


def parse_prometheus(text: str) -> dict[str, float]:
    """Aggregate Prometheus samples by metric name, ignoring labels."""
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name_part, value_part = line.rsplit(None, 1)
        except ValueError:
            continue
        try:
            value = float(value_part)
        except ValueError:
            continue
        name = name_part.split("{", 1)[0]
        values[name] = values.get(name, 0.0) + value
    return values


def parse_labeled_positions(text: str, metric_name: str) -> dict[str, float]:
    """Extract per-position speculative counters from Prometheus text."""
    positions: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith(metric_name + "{"):
            continue
        try:
            label_part, value_part = line.rsplit(None, 1)
        except ValueError:
            continue
        try:
            value = float(value_part)
        except ValueError:
            continue
        labels_text = label_part[len(metric_name) :].strip().strip("{}")
        for label in labels_text.split(","):
            key, _, raw_value = label.partition("=")
            if key.strip() != "position":
                continue
            position = raw_value.strip().strip('"')
            positions[position] = positions.get(position, 0.0) + value
            break
    return positions


def metric_value(snapshot: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in snapshot:
            return snapshot[name]
    return None


def metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    *names: str,
) -> float:
    old = metric_value(before, *names)
    new = metric_value(after, *names)
    if old is None or new is None:
        return 0.0
    return max(0.0, new - old)


def build_long_prompt(model: str, target_tokens: int) -> tuple[str, int | None, str | None]:
    """Build a prompt of approximately target_tokens using the real tokenizer."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        base_ids = tokenizer.encode(LONG_PROMPT_BASE, add_special_tokens=False)
        if not base_ids:
            raise ValueError("Tokenizer produced no tokens for the base text.")
        repeats = (target_tokens // len(base_ids)) + 1
        ids = (base_ids * repeats)[:target_tokens]
        return tokenizer.decode(ids), len(ids), None
    except Exception as exc:  # pragma: no cover - fallback path is defensive
        # Rough fallback: Qwen tokenizers average around four characters/token.
        repeats = max(1, math.ceil(target_tokens * 4 / len(LONG_PROMPT_BASE)))
        text = (LONG_PROMPT_BASE * repeats).strip()
        return text, None, f"{type(exc).__name__}: {exc}"


def build_payload(args: argparse.Namespace, content: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def run_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    index: int,
    capture_headers: bool = False,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_content: float | None = None
    content_times: list[float] = []
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    status_code: int | None = None
    route_category: str | None = None
    done_seen = False
    error: str | None = None

    try:
        async with client.stream("POST", url, json=payload) as response:
            status_code = response.status_code
            if capture_headers:
                route_category = response.headers.get("x-semantic-category")
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    done_seen = True
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue

                usage = event.get("usage")
                if usage:
                    if usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = int(usage["prompt_tokens"])

                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if not content:
                    continue
                now = time.perf_counter()
                if first_content is None:
                    first_content = now
                content_times.append(now)

    except Exception as exc:
        end = time.perf_counter()
        return {
            "index": index,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "status_code": status_code,
            "route_category": route_category,
            "total_ms": round((end - start) * 1000, 4),
            "ttft_ms": None,
            "generation_ms": None,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tokens_per_second": None,
            "intervals_ms": [],
            "interval_p50_ms": None,
            "interval_p95_ms": None,
            "done_seen": done_seen,
            "finish_reason": finish_reason,
        }

    end = time.perf_counter()
    total_ms = (end - start) * 1000
    intervals_ms = [
        (content_times[i] - content_times[i - 1]) * 1000
        for i in range(1, len(content_times))
    ]

    if first_content is not None:
        generation_ms = (end - first_content) * 1000
        ttft_ms = (first_content - start) * 1000
    else:
        generation_ms = None
        ttft_ms = None

    if completion_tokens is not None and generation_ms and generation_ms > 0:
        tokens_per_second = completion_tokens / (generation_ms / 1000)
    else:
        tokens_per_second = None

    if first_content is None:
        error = "No generated content received."
    elif completion_tokens is None:
        error = "Completion token usage missing."
    elif not done_seen:
        error = "Stream ended without [DONE]."
    elif completion_tokens <= 0:
        error = "Completion token count is zero."

    return {
        "index": index,
        "ok": error is None,
        "error": error,
        "status_code": status_code,
        "route_category": route_category,
        "total_ms": round(total_ms, 4),
        "ttft_ms": round(ttft_ms, 4) if ttft_ms is not None else None,
        "generation_ms": round(generation_ms, 4) if generation_ms is not None else None,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_second": round(tokens_per_second, 4)
        if tokens_per_second is not None
        else None,
        "intervals_ms": [round(value, 4) for value in intervals_ms],
        "interval_p50_ms": round(statistics.median(intervals_ms), 4)
        if intervals_ms
        else None,
        "interval_p95_ms": round(percentile(intervals_ms, 0.95), 4)
        if intervals_ms
        else None,
        "done_seen": done_seen,
        "finish_reason": finish_reason,
    }


async def fetch_text(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


async def gpu_sample(interval: float, stop_event: asyncio.Event, samples: list[dict[str, Any]]) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        started = time.perf_counter()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                line = stdout.decode(errors="replace").strip().splitlines()
                if line:
                    parts = [part.strip() for part in line[0].split(",")]
                    if len(parts) >= 6:
                        def number(value: str) -> float | None:
                            try:
                                return float(value)
                            except ValueError:
                                return None

                        samples.append(
                            {
                                "t": round(started, 4),
                                "memory_used_mib": number(parts[0]),
                                "utilization_gpu_pct": number(parts[1]),
                                "temperature_c": number(parts[2]),
                                "power_w": number(parts[3]),
                                "clock_sm_mhz": number(parts[4]),
                                "clock_mem_mhz": number(parts[5]),
                            }
                        )
            else:
                samples.append(
                    {
                        "t": round(started, 4),
                        "error": stderr.decode(errors="replace").strip()[:200],
                    }
                )
        except Exception as exc:
            samples.append({"t": round(started, 4), "error": f"{type(exc).__name__}: {exc}"})
        elapsed = time.perf_counter() - started
        await asyncio.sleep(max(0.0, interval - elapsed))


async def metrics_sampler(
    client: httpx.AsyncClient,
    url: str,
    interval: float,
    stop_event: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    while not stop_event.is_set():
        started = time.perf_counter()
        text = await fetch_text(client, url)
        if text is not None:
            snapshot = parse_prometheus(text)
            samples.append(
                {
                    "t": round(started, 4),
                    "kv_cache_usage_perc": metric_value(snapshot, KV_USAGE),
                    "num_requests_running": metric_value(snapshot, REQUESTS_RUNNING),
                    "num_requests_waiting": metric_value(snapshot, REQUESTS_WAITING),
                    "generation_tokens_total": metric_value(snapshot, GENERATION_TOKENS),
                }
            )
        await asyncio.sleep(interval)


async def health_sampler(
    url: str,
    interval: float,
    stop_event: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while not stop_event.is_set():
            started = time.perf_counter()
            try:
                response = await client.get(url)
                samples.append(
                    {
                        "t": round(started, 4),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 4),
                        "ok": response.status_code == 200,
                    }
                )
            except Exception as exc:
                samples.append(
                    {
                        "t": round(started, 4),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 4),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            await asyncio.sleep(interval)


async def gpu_identity() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,"
        "clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            return {"error": stderr.decode(errors="replace").strip()[:300]}
        parts = [part.strip() for part in stdout.decode(errors="replace").strip().split(",")]
        if len(parts) < 8:
            return {"error": f"Unexpected nvidia-smi output: {parts}"}
        return {
            "name": parts[0],
            "memory_total_mib": parts[1],
            "memory_used_mib_idle": parts[2],
            "utilization_gpu_pct_idle": parts[3],
            "temperature_c_idle": parts[4],
            "power_w_idle": parts[5],
            "clock_sm_mhz_idle": parts[6],
            "clock_mem_mhz_idle": parts[7],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def summarize_gpu(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if "error" not in sample]
    summary: dict[str, Any] = {"sample_count": len(valid)}
    for key in (
        "memory_used_mib",
        "utilization_gpu_pct",
        "temperature_c",
        "power_w",
        "clock_sm_mhz",
        "clock_mem_mhz",
    ):
        values = [sample[key] for sample in valid if sample.get(key) is not None]
        summary[key] = summarize(values)
    memories = [sample["memory_used_mib"] for sample in valid if sample.get("memory_used_mib") is not None]
    if memories:
        summary["memory_delta_mib"] = round(max(memories) - memories[0], 4)
    errors = [sample["error"] for sample in samples if "error" in sample]
    if errors:
        summary["errors"] = errors[:5]
    return summary


def summarize_kv(samples: list[dict[str, Any]]) -> dict[str, Any]:
    usage = [
        sample["kv_cache_usage_perc"]
        for sample in samples
        if sample.get("kv_cache_usage_perc") is not None
    ]
    running = [
        sample["num_requests_running"]
        for sample in samples
        if sample.get("num_requests_running") is not None
    ]
    waiting = [
        sample["num_requests_waiting"]
        for sample in samples
        if sample.get("num_requests_waiting") is not None
    ]
    return {
        "sample_count": len(samples),
        "kv_cache_usage_perc": summarize(usage),
        "num_requests_running": summarize(running),
        "num_requests_waiting": summarize(waiting),
    }


def build_acceptance(
    before_text: str | None,
    after_text: str | None,
) -> dict[str, Any]:
    if before_text is None or after_text is None:
        return {"available": False, "reason": "metrics endpoint unavailable"}
    before = parse_prometheus(before_text)
    after = parse_prometheus(after_text)
    drafts = metric_delta(before, after, SPEC_DRAFTS, SPEC_DRAFTS.removesuffix("_total"))
    draft_tokens = metric_delta(
        before,
        after,
        SPEC_DRAFT_TOKENS,
        SPEC_DRAFT_TOKENS.removesuffix("_total"),
    )
    accepted = metric_delta(before, after, SPEC_ACCEPTED, SPEC_ACCEPTED.removesuffix("_total"))
    before_pos = parse_labeled_positions(before_text, SPEC_ACCEPTED_PER_POS)
    after_pos = parse_labeled_positions(after_text, SPEC_ACCEPTED_PER_POS)
    per_position = {
        position: round(max(0.0, after_pos.get(position, 0.0) - before_pos.get(position, 0.0)), 4)
        for position in sorted(set(before_pos) | set(after_pos))
    }
    return {
        "available": True,
        "counter_deltas": {
            "drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted,
        },
        "acceptance_rate": round(accepted / draft_tokens, 6) if draft_tokens > 0 else None,
        "mean_acceptance_length": round(1 + accepted / drafts, 6) if drafts > 0 else None,
        "accepted_tokens_per_position": per_position,
        "note": "Global counter deltas for the timed phase; warm-up requests are excluded.",
    }


def summarize_requests(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result["ok"]]
    failed = [result for result in results if not result["ok"]]
    all_intervals = [
        interval
        for result in successful
        for interval in result.get("intervals_ms", [])
    ]
    completion_tokens = [
        result["completion_tokens"]
        for result in successful
        if result.get("completion_tokens") is not None
    ]
    return {
        "requests_total": len(results),
        "requests_successful": len(successful),
        "requests_failed": len(failed),
        "success_rate": round(len(successful) / len(results), 6) if results else None,
        "ttft_ms": summarize([result["ttft_ms"] for result in successful]),
        "total_ms": summarize([result["total_ms"] for result in successful]),
        "tokens_per_second_per_request": summarize(
            [result["tokens_per_second"] for result in successful]
        ),
        "completion_tokens": summarize(completion_tokens),
        "inter_chunk_ms": summarize(all_intervals),
        "failures": [
            {
                "index": result["index"],
                "error": result.get("error"),
                "status_code": result.get("status_code"),
            }
            for result in failed
        ],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    acceptance = result["speculative_decoding"]
    gpu = result["gpu"]["summary"]
    kv = result["kv_cache"]["summary"]

    def row(name: str, stats: dict[str, Any], unit: str) -> str:
        if not stats or stats.get("count", 0) == 0:
            return f"| {name} | n/a | n/a | n/a | n/a | n/a |\n"
        return (
            f"| {name} | {stats['median']:.3f} | {stats['p95']:.3f} | "
            f"{stats['p99']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | "
            f"{stats['count']} | {unit} |\n"
        )

    lines = [
        "# Worker 1 benchmark result",
        "",
        f"- Label: `{result['label']}`",
        f"- Timestamp (UTC): `{result['timestamp_utc']}`",
        f"- Base URL: `{result['endpoint']['base_url']}`",
        f"- Workload: `{result['workload']['mode']}`",
        f"- Requests: {summary['requests_total']} total, "
        f"{summary['requests_successful']} successful, "
        f"{summary['requests_failed']} failed "
        f"(success rate {summary['success_rate']})",
        "",
        "## Latency (successful requests)",
        "",
        "| Metric | p50 | p95 | p99 | min | max | n | unit |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        row("TTFT", summary["ttft_ms"], "ms").rstrip("\n"),
        row("Total latency", summary["total_ms"], "ms").rstrip("\n"),
        row("Inter-chunk", summary["inter_chunk_ms"], "ms").rstrip("\n"),
        row("Per-request throughput", summary["tokens_per_second_per_request"], "tokens/s").rstrip("\n"),
        "",
        f"Aggregate timed throughput: `{result['throughput']['aggregate_tokens_per_second']}` tokens/s",
        f"(timed wall `{result['throughput']['timed_wall_seconds']}` s, "
        f"active batch wall `{result['throughput']['active_batch_seconds']}` s).",
        "",
        "## Speculative decoding",
        "",
        f"- Counter deltas: `{acceptance.get('counter_deltas')}`",
        f"- Acceptance rate: `{acceptance.get('acceptance_rate')}`",
        f"- Mean acceptance length: `{acceptance.get('mean_acceptance_length')}`",
        f"- Accepted tokens per position: `{acceptance.get('accepted_tokens_per_position')}`",
        "",
        "## GPU telemetry",
        "",
        f"- Samples: {gpu.get('sample_count')}",
        f"- Memory used (MiB): `{gpu.get('memory_used_mib')}`",
        f"- Memory delta (MiB): `{gpu.get('memory_delta_mib')}`",
        f"- GPU util (%): `{gpu.get('utilization_gpu_pct')}`",
        f"- Temperature (C): `{gpu.get('temperature_c')}`",
        f"- Power (W): `{gpu.get('power_w')}`",
        "",
        "## KV cache",
        "",
        f"- KV usage (%): `{kv.get('kv_cache_usage_perc')}`",
        f"- Requests running: `{kv.get('num_requests_running')}`",
        f"- Requests waiting: `{kv.get('num_requests_waiting')}`",
        "",
        "## Failures",
        "",
    ]
    if summary["failures"]:
        lines.extend(f"- `{failure}`" for failure in summary["failures"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Measurement notes",
            "",
            "- TTFT is the time from request start to the first content-bearing SSE chunk.",
            "- Total latency ends when the stream closes after `[DONE]`.",
            "- Inter-chunk latency is measured between content-bearing SSE events, not raw tokens.",
            "- GPU samples are periodic `nvidia-smi` polls; short bursts may be missed.",
            "- Speculative counters are global deltas for the timed phase and exclude warm-up.",
            "- p99 is marked unreliable below 100 samples in the JSON payload.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_stamp()
    content = STANDARD_PROMPT
    max_tokens = args.max_tokens
    long_prompt_info: dict[str, Any] = {"enabled": args.long_context}
    if args.long_context:
        content, token_count, error = build_long_prompt(args.model, args.long_prompt_tokens)
        max_tokens = args.long_max_tokens
        long_prompt_info.update(
            {
                "target_tokens": args.long_prompt_tokens,
                "tokenizer_token_count": token_count,
                "tokenizer_error": error,
                "max_tokens": max_tokens,
                "estimated_total_tokens": (token_count + max_tokens)
                if token_count is not None
                else None,
            }
        )

    payload = build_payload(args, content, max_tokens)
    if args.dry_run:
        print(json.dumps({"payload": payload, "long_prompt": long_prompt_info}, indent=2))
        return {}

    if args.mode == "concurrent" and args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.mode == "concurrent" and args.batches < 1:
        raise SystemExit("--batches must be >= 1")

    gpu_id = await gpu_identity() if not args.no_gpu else {"skipped": True}
    print(f"GPU: {gpu_id}")

    print(f"Target: {args.base_url}")
    print(f"Label: {args.label}")
    print(f"Mode: {args.mode}, runs={args.runs}, concurrency={args.concurrency}, batches={args.batches}")
    if args.long_context:
        print(f"Long prompt: {long_prompt_info}")
    print(f"Warm-up requests: {args.warmup}")

    timeout = httpx.Timeout(connect=10.0, read=args.timeout, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as warmup_client:
        for index in range(args.warmup):
            result = await run_request(warmup_client, args.url, payload, index)
            print(
                f"Warm-up {index + 1}/{args.warmup}: "
                f"{'ok' if result['ok'] else 'FAILED'} "
                f"({result.get('total_ms')} ms)"
            )

    metrics_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    before_text = await fetch_text(metrics_client, args.metrics_url) if args.metrics_url else None
    await metrics_client.aclose()

    stop_event = asyncio.Event()
    gpu_samples: list[dict[str, Any]] = []
    metric_samples: list[dict[str, Any]] = []
    health_samples: list[dict[str, Any]] = []
    sampler_tasks = []

    if not args.no_gpu:
        sampler_tasks.append(
            asyncio.create_task(gpu_sample(args.gpu_sample_interval, stop_event, gpu_samples))
        )
    if args.metrics_url:
        metrics_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        sampler_tasks.append(
            asyncio.create_task(
                metrics_sampler(
                    metrics_client,
                    args.metrics_url,
                    args.metrics_sample_interval,
                    stop_event,
                    metric_samples,
                )
            )
        )
    if args.health_url:
        sampler_tasks.append(
            asyncio.create_task(
                health_sampler(args.health_url, args.health_interval, stop_event, health_samples)
            )
        )

    timed_start = time.perf_counter()
    batches: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    if args.mode == "concurrent":
        clients = [httpx.AsyncClient(timeout=timeout) for _ in range(args.concurrency)]
        try:
            for batch_index in range(args.batches):
                batch_start = time.perf_counter()
                tasks = [
                    run_request(
                        clients[worker],
                        args.url,
                        payload,
                        batch_index * args.concurrency + worker,
                    )
                    for worker in range(args.concurrency)
                ]
                batch_results = await asyncio.gather(*tasks)
                batch_end = time.perf_counter()
                all_results.extend(batch_results)
                batch_tokens = sum(
                    result["completion_tokens"] or 0
                    for result in batch_results
                    if result["ok"]
                )
                batches.append(
                    {
                        "batch_index": batch_index,
                        "wall_seconds": round(batch_end - batch_start, 6),
                        "completion_tokens": batch_tokens,
                        "aggregate_tokens_per_second": round(
                            batch_tokens / (batch_end - batch_start), 6
                        )
                        if batch_end > batch_start
                        else None,
                        "requests": batch_results,
                    }
                )
                print(
                    f"Batch {batch_index + 1}/{args.batches}: "
                    f"{sum(1 for result in batch_results if result['ok'])}/{len(batch_results)} ok, "
                    f"{batch_tokens} tokens, wall {batch_end - batch_start:.3f}s"
                )
        finally:
            for client in clients:
                await client.aclose()
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for index in range(args.runs):
                result = await run_request(client, args.url, payload, index)
                all_results.append(result)
                batches.append(
                    {
                        "batch_index": index,
                        "wall_seconds": round(result["total_ms"] / 1000, 6),
                        "completion_tokens": result["completion_tokens"] or 0,
                        "aggregate_tokens_per_second": result["tokens_per_second"],
                        "requests": [result],
                    }
                )
                print(
                    f"Run {index + 1:02d}/{args.runs}: "
                    f"{'ok' if result['ok'] else 'FAILED'} | "
                    f"TTFT {result.get('ttft_ms')} ms | "
                    f"Total {result.get('total_ms')} ms | "
                    f"Tokens {result.get('completion_tokens')} | "
                    f"{result.get('tokens_per_second')} tok/s"
                )

    timed_end = time.perf_counter()
    stop_event.set()
    if sampler_tasks:
        await asyncio.gather(*sampler_tasks, return_exceptions=True)
    if args.metrics_url:
        await metrics_client.aclose()

    after_text = None
    if args.metrics_url:
        after_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        after_text = await fetch_text(after_client, args.metrics_url)
        await after_client.aclose()

    summary = summarize_requests(all_results)
    successful = [result for result in all_results if result["ok"]]
    total_completion_tokens = sum(
        result["completion_tokens"] or 0 for result in successful
    )
    active_batch_seconds = sum(batch["wall_seconds"] for batch in batches)
    timed_wall_seconds = timed_end - timed_start
    aggregate_tokens_per_second = (
        total_completion_tokens / active_batch_seconds
        if active_batch_seconds > 0
        else None
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "timestamp_utc": timestamp,
        "script": "scripts/benchmark_worker1.py",
        "command": sys.argv,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "endpoint": {
            "base_url": args.base_url,
            "url": args.url,
            "metrics_url": args.metrics_url,
            "health_url": args.health_url,
        },
        "model": args.model,
        "workload": {
            "mode": args.mode,
            "runs": args.runs,
            "warmup": args.warmup,
            "concurrency": args.concurrency,
            "batches": args.batches,
            "max_tokens": max_tokens,
            "temperature": args.temperature,
            "long_context": long_prompt_info,
            "standard_prompt": STANDARD_PROMPT,
            "payload": payload,
        },
        "summary": summary,
        "throughput": {
            "aggregate_tokens_per_second": round(aggregate_tokens_per_second, 6)
            if aggregate_tokens_per_second is not None
            else None,
            "total_completion_tokens": total_completion_tokens,
            "active_batch_seconds": round(active_batch_seconds, 6),
            "timed_wall_seconds": round(timed_wall_seconds, 6),
            "note": "Aggregate throughput is total completion tokens divided by active batch wall time; it is not per-request throughput.",
        },
        "batches": batches,
        "requests": all_results,
        "speculative_decoding": build_acceptance(before_text, after_text),
        "kv_cache": {
            "summary": summarize_kv(metric_samples),
            "samples": metric_samples,
        },
        "gpu": {
            "identity": gpu_id,
            "summary": summarize_gpu(gpu_samples) if not args.no_gpu else {"skipped": True},
            "samples": gpu_samples,
        },
        "health_probe": {
            "summary": summarize([sample["elapsed_ms"] for sample in health_samples if sample.get("ok")]),
            "failures": sum(1 for sample in health_samples if not sample.get("ok")),
            "samples": health_samples,
        },
        "metrics_before": parse_prometheus(before_text) if before_text else None,
        "metrics_after": parse_prometheus(after_text) if after_text else None,
    }

    output = Path(args.output) if args.output else (
        DEFAULT_OUTPUT_DIR / f"worker1_{args.label.replace(' ', '_')}_{timestamp}.json"
    )
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    write_markdown(result, markdown_path)

    print("\n========== SUMMARY ==========")
    print(f"Success: {summary['requests_successful']}/{summary['requests_total']} "
          f"({summary['success_rate']})")
    for key, label in (
        ("ttft_ms", "TTFT"),
        ("total_ms", "Total"),
        ("inter_chunk_ms", "Inter-chunk"),
        ("tokens_per_second_per_request", "Per-request throughput"),
    ):
        stats = summary[key]
        if stats.get("count", 0):
            print(
                f"{label}: median={stats['median']:.3f} p95={stats['p95']:.3f} "
                f"p99={stats['p99']:.3f} (n={stats['count']})"
            )
    print(f"Aggregate throughput: {result['throughput']['aggregate_tokens_per_second']} tok/s")
    acceptance = result["speculative_decoding"]
    print(
        f"Spec acceptance: rate={acceptance.get('acceptance_rate')} "
        f"mean_length={acceptance.get('mean_acceptance_length')} "
        f"deltas={acceptance.get('counter_deltas')}"
    )
    print(f"GPU summary: {result['gpu']['summary']}")
    print(f"Saved: {output}")
    print(f"Saved: {markdown_path}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    parser.add_argument("--health-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--label", default="worker1")
    parser.add_argument("--mode", choices=["single", "concurrent"], default="single")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--long-prompt-tokens", type=int, default=800)
    parser.add_argument("--long-max-tokens", type=int, default=100)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--metrics-sample-interval", type=float, default=0.5)
    parser.add_argument("--health-interval", type=float, default=0.05)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.url = args.base_url.rstrip("/") + "/v1/chat/completions"
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        asyncio.run(run(args))
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
