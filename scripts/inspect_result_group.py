#!/usr/bin/env python3
"""Print telemetry details for one workload/mode group in a saved result file.

Read-only: opens a saved Stage 2/3 JSON file and prints the metrics that the
Markdown reports summarize only partially (KV max, waiting p95, GPU peaks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--workload", default=None)
    parser.add_argument("--mode", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = json.loads(args.path.read_text(encoding="utf-8"))
    groups = []
    for workload, workload_info in result.get("workloads", {}).items():
        if not isinstance(workload_info, dict):
            continue
        for mode in ("warm", "cold"):
            group = workload_info.get(mode)
            if not isinstance(group, dict) or "direct" not in group:
                continue
            if args.workload and workload != args.workload:
                continue
            if args.mode and mode != args.mode:
                continue
            groups.append((workload, mode, group))

    if not groups:
        raise SystemExit("No matching groups.")

    for workload, mode, group in groups:
        kv = (group.get("kv_cache") or {}).get("summary") or {}
        usage = kv.get("kv_cache_usage_perc") or {}
        running = kv.get("num_requests_running") or {}
        waiting = kv.get("num_requests_waiting") or {}
        direct = group.get("direct") or {}
        gateway = group.get("gateway") or {}
        print(f"{args.path.name} workload {workload} mode {mode}")
        print(
            f"  success direct {direct.get('requests_successful')}/"
            f"{direct.get('requests_total')}, gateway "
            f"{gateway.get('requests_successful')}/{gateway.get('requests_total')}"
        )
        print(f"  kv usage median/p95/max: {usage.get('median')}/{usage.get('p95')}/{usage.get('max')}")
        print(f"  running median/p95/max: {running.get('median')}/{running.get('p95')}/{running.get('max')}")
        print(f"  waiting median/p95/max: {waiting.get('median')}/{waiting.get('p95')}/{waiting.get('max')}")
        if "batch_wall_ms" in group:
            wall = group["batch_wall_ms"]
            print(
                f"  batch wall p50 direct/gateway: "
                f"{wall['direct'].get('median')}/{wall['gateway'].get('median')}"
            )
        else:
            print(
                f"  total p50 direct/gateway: "
                f"{(direct.get('total_ms') or {}).get('median')}/"
                f"{(gateway.get('total_ms') or {}).get('median')}"
            )
        print()

    gpu = result.get("gpu", {}).get("summary", {})
    temp = gpu.get("temperature_c") or {}
    memory = gpu.get("memory_used_mib") or {}
    sm = gpu.get("clock_sm_mhz") or {}
    print(
        f"GPU: sample_count {gpu.get('sample_count')}, temp mean/max "
        f"{temp.get('mean')}/{temp.get('max')}, memory max {memory.get('max')} MiB, "
        f"SM clock mean {sm.get('mean')} MHz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
