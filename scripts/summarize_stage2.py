#!/usr/bin/env python3
"""Build a cross-configuration Stage 2 comparison from raw paired JSON files.

The script reads only saved measurement files. It refuses to overwrite an
existing report unless --force is supplied, and it leaves every raw JSON file
untouched.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = str(PROJECT_ROOT / "benchmarks" / "stage2" / "paired_kv*.json")
DEFAULT_STARTUP = str(PROJECT_ROOT / "benchmarks" / "stage2" / "startup_kv*.json")


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.{digits}f}"


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_for(result: dict[str, Any], workload: str, mode: str) -> dict[str, Any]:
    group = result["workloads"][workload][mode]
    direct = group["direct"]
    gateway = group["gateway"]
    delta = group["paired_delta_gateway_minus_direct"]
    spec = group["speculative_decoding"]
    kv = group["kv_cache"]["summary"].get("kv_cache_usage_perc") or {}
    return {
        "config": f"{result['vllm_config']['kv_mib']}/{result['vllm_config']['max_num_seqs']}",
        "kv_mib": result["vllm_config"]["kv_mib"],
        "seqs": result["vllm_config"]["max_num_seqs"],
        "workload": workload,
        "mode": mode,
        "direct_success": f"{direct['requests_successful']}/{direct['requests_total']}",
        "gateway_success": f"{gateway['requests_successful']}/{gateway['requests_total']}",
        "direct_ttft": (direct.get("ttft_ms") or {}).get("median"),
        "gateway_ttft": (gateway.get("ttft_ms") or {}).get("median"),
        "ttft_delta": (delta.get("ttft_ms") or {}).get("median"),
        "ttft_delta_p95": (delta.get("ttft_ms") or {}).get("p95"),
        "ttft_delta_pct": (delta.get("ttft_ms") or {}).get("median_percent"),
        "direct_total": (direct.get("total_ms") or {}).get("median"),
        "gateway_total": (gateway.get("total_ms") or {}).get("median"),
        "total_delta": (delta.get("total_ms") or {}).get("median"),
        "total_delta_pct": (delta.get("total_ms") or {}).get("median_percent"),
        "direct_tps": group["aggregate"].get("direct_tokens_per_second"),
        "gateway_tps": group["aggregate"].get("gateway_tokens_per_second"),
        "spec_acceptance": spec.get("acceptance_rate"),
        "mean_acceptance_length": spec.get("mean_acceptance_length"),
        "kv_p95": kv.get("p95"),
        "kv_max": kv.get("max"),
        "orders": group.get("orders"),
        "route_categories": group.get("route_categories"),
    }


def write_series(
    lines: list[str],
    results: list[dict[str, Any]],
    workloads: list[str],
    mode: str,
) -> None:
    lines.extend(
        [
            f"### {mode.capitalize()} mode",
            "",
            "| KV/seqs | Workload | OK direct/gw | TTFT direct/gw (ms) | "
            "Paired TTFT delta ms (p50/p95) | Paired total delta ms | "
            "Aggregate tok/s direct/gw | Spec accept | KV p95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        for workload in workloads:
            if workload not in result["workloads"] or mode not in result["workloads"][workload]:
                continue
            row = row_for(result, workload, mode)
            lines.append(
                f"| {row['kv_mib']}/{row['seqs']} | {workload} | "
                f"{row['direct_success']} / {row['gateway_success']} | "
                f"{fmt(row['direct_ttft'])} / {fmt(row['gateway_ttft'])} | "
                f"{fmt(row['ttft_delta'])} / {fmt(row['ttft_delta_p95'])} "
                f"({fmt(row['ttft_delta_pct'])}%) | "
                f"{fmt(row['total_delta'])} ({fmt(row['total_delta_pct'])}%) | "
                f"{fmt(row['direct_tps'])} / {fmt(row['gateway_tps'])} | "
                f"{pct(row['spec_acceptance'])}% | {pct(row['kv_p95'])}% |"
            )
    lines.append("")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--startup", nargs="*", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--title", default="Worker 1 Stage 2 KV cache sweep")
    args = parser.parse_args()

    input_paths = args.inputs or sorted(glob.glob(DEFAULT_INPUT))
    if not input_paths:
        raise SystemExit("No paired result files found.")

    results = sorted(
        (load(path) for path in input_paths),
        key=lambda item: (
            item["vllm_config"]["max_num_seqs"],
            item["vllm_config"]["kv_mib"],
        ),
    )
    workloads = sorted({workload for result in results for workload in result["workloads"]})
    startup_paths = args.startup or sorted(glob.glob(DEFAULT_STARTUP))
    startups = [load(path) for path in startup_paths if load(path).get("ok")]

    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {args.title}",
        "",
        "Generated from saved raw measurement files. No values are inferred.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{Path(path).name}`" for path in input_paths)
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- Direct and gateway requests are paired back-to-back; order is randomized per pair.",
            "- Warm mode repeats one prompt; cold mode uses a unique prompt for every request and path.",
            "- Prefix caching stays enabled; cold mode defeats it with unique prefixes instead of changing config.",
            "- Workloads: A short 32 tokens, B standard 150 tokens, C long 800 prompt + 100 completion.",
            "- Speculative decoding: `Qwen/Qwen3-0.6B` draft, two speculative tokens, throughout.",
            "",
        ]
    )

    for mode in ("warm", "cold"):
        write_series(lines, results, workloads, mode)

    if startups:
        lines.extend(
            [
                "## Startup facts",
                "",
                "| KV MiB | seqs | OK | KV tokens | Max concurrency | Model GiB | Load s | Engine init s |",
                "|---:|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for startup in sorted(
            startups,
            key=lambda item: (item.get("max_num_seqs", 1), item.get("kv_mib", 0)),
        ):
            facts = startup.get("facts") or {}
            lines.append(
                f"| {startup.get('kv_mib')} | {startup.get('max_num_seqs')} | "
                f"{startup.get('ok')} | {facts.get('kv_cache_tokens', 'n/a')} | "
                f"{facts.get('max_concurrency', 'n/a')} | "
                f"{facts.get('model_memory_gib', 'n/a')} | "
                f"{fmt(facts.get('model_load_seconds'), 3)} | "
                f"{fmt(facts.get('engine_init_seconds'), 3)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## GPU telemetry by configuration",
            "",
            "| KV/seqs | Samples | Temp mean/max C | Power mean W | SM clock mean MHz | Memory clock mean MHz |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        gpu = result["gpu"]["summary"]
        temp = gpu.get("temperature_c") or {}
        power = gpu.get("power_w") or {}
        sm = gpu.get("clock_sm_mhz") or {}
        mem = gpu.get("clock_mem_mhz") or {}
        lines.append(
            f"| {result['vllm_config']['kv_mib']}/{result['vllm_config']['max_num_seqs']} | "
            f"{gpu.get('sample_count')} | "
            f"{fmt(temp.get('mean'))}/{fmt(temp.get('max'))} | "
            f"{fmt(power.get('mean'))} | {fmt(sm.get('mean'))} | {fmt(mem.get('mean'))} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Selection criteria",
            "",
            "The final recommendation must weigh, in this order: request success rate,"
            " memory headroom, context capacity, latency, then aggregate throughput."
            " A configuration that fails requests or leaves unsafe memory headroom is"
            " not viable regardless of latency.",
            "",
        ]
    )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
