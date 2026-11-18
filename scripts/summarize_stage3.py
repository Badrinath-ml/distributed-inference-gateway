#!/usr/bin/env python3
"""Build a max-num-seqs comparison from saved Stage 3 concurrent JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = str(PROJECT_ROOT / "benchmarks" / "stage3" / "concurrent_*.json")
DEFAULT_STARTUP = str(PROJECT_ROOT / "benchmarks" / "stage2" / "startup_kv*seqs*.json")


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def stat(summary: dict[str, Any] | None, key: str = "median") -> Any:
    if not summary:
        return None
    return summary.get(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--startup", nargs="*", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = args.inputs or sorted(glob.glob(DEFAULT_INPUT))
    if not paths:
        raise SystemExit("No Stage 3 result files found.")
    results = sorted(
        (json.loads(Path(path).read_text(encoding="utf-8")) for path in paths),
        key=lambda item: (item["vllm_config"]["max_num_seqs"], item["vllm_config"]["kv_mib"]),
    )
    startup_paths = args.startup or sorted(glob.glob(DEFAULT_STARTUP))
    startups = []
    for path in startup_paths:
        item = json.loads(Path(path).read_text(encoding="utf-8"))
        if item.get("ok"):
            startups.append(item)

    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Worker 1 Stage 3: max-num-seqs 1 versus 2",
        "",
        "Generated from saved raw measurement files. No values are inferred.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{Path(path).name}`" for path in paths)
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- Two simultaneous requests per batch, followed by a paired batch to the other path.",
            "- Batch order is randomized independently for every pair.",
            "- Warm mode repeats one prompt; cold mode uses a unique prompt per request.",
            "- Prefix caching remains enabled and is not reconfigured.",
            "- Workloads: A short 32 tokens, B standard 150 tokens, C long 800 prompt + 100 completion.",
            "",
        ]
    )

    models = sorted({result["vllm_config"]["kv_mib"] for result in results})
    for mode in ("warm", "cold"):
        lines.extend(
            [
                f"### {mode.capitalize()} mode",
                "",
                "| KV/seqs | Workload | OK direct/gw | Batch wall p50 ms direct/gw | "
                "Batch tok/s p50 direct/gw | Aggregate tok/s direct/gw | "
                "Waiting p95 | KV p95 | Spec accept |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in results:
            config = result["vllm_config"]
            if config["kv_mib"] not in models:
                continue
            for workload in result["protocol"]["workloads"]:
                group = result["workloads"][workload].get(mode)
                if not group:
                    continue
                direct = group["direct"]
                gateway = group["gateway"]
                wall = group["batch_wall_ms"]
                bw = group["batch_throughput"]
                agg = group["aggregate"]
                kv = group["kv_cache"]["summary"]
                waiting = kv.get("num_requests_waiting") or {}
                usage = kv.get("kv_cache_usage_perc") or {}
                lines.append(
                    f"| {config['kv_mib']}/{config['max_num_seqs']} | {workload} | "
                    f"{direct['requests_successful']}/{direct['requests_total']} / "
                    f"{gateway['requests_successful']}/{gateway['requests_total']} | "
                    f"{fmt(stat(wall['direct']))} / {fmt(stat(wall['gateway']))} | "
                    f"{fmt(stat(bw['direct']))} / {fmt(stat(bw['gateway']))} | "
                    f"{fmt(agg['direct_tokens_per_second'])} / "
                    f"{fmt(agg['gateway_tokens_per_second'])} | "
                    f"{fmt(waiting.get('p95'))} | "
                    f"{100.0 * usage.get('p95', 0.0):.2f}% | "
                    f"{100.0 * (group['speculative_decoding'].get('acceptance_rate') or 0.0):.2f}% |"
                )
        lines.append("")

    if startups:
        lines.extend(
            [
                "## Startup facts",
                "",
                "| KV MiB | seqs | KV tokens | Max concurrency | Model GiB | Load s | Engine init s |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for startup in sorted(
            startups,
            key=lambda item: (item.get("max_num_seqs", 1), item.get("kv_mib", 0)),
        ):
            facts = startup.get("facts") or {}
            lines.append(
                f"| {startup.get('kv_mib')} | {startup.get('max_num_seqs')} | "
                f"{facts.get('kv_cache_tokens', 'n/a')} | "
                f"{facts.get('max_concurrency', 'n/a')} | "
                f"{facts.get('model_memory_gib', 'n/a')} | "
                f"{fmt(facts.get('model_load_seconds'), 3)} | "
                f"{fmt(facts.get('engine_init_seconds'), 3)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## GPU telemetry",
            "",
            "| KV/seqs | Samples | Temp mean/max C | Power mean W | SM clock mean MHz |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        gpu = result["gpu"]["summary"]
        temp = gpu.get("temperature_c") or {}
        power = gpu.get("power_w") or {}
        sm = gpu.get("clock_sm_mhz") or {}
        lines.append(
            f"| {result['vllm_config']['kv_mib']}/{result['vllm_config']['max_num_seqs']} | "
            f"{gpu.get('sample_count')} | {fmt(temp.get('mean'))}/{fmt(temp.get('max'))} | "
            f"{fmt(power.get('mean'))} | {fmt(sm.get('mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "Prefer a setting only if every request succeeds and KV pressure leaves"
            " no waiting for the supported concurrency. Then compare aggregate"
            " throughput at concurrency two, followed by per-request latency.",
            "",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
