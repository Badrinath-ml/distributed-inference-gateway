#!/usr/bin/env python3
"""Read-only verification of saved Worker 1 Stage 2/3 benchmark files.

Prints per-file request totals, failures, KV/memory peaks and GPU peaks so the
final report can cite only measured values. This script never contacts the
servers and never writes results.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    "benchmarks/stage2/paired_kv*.json",
    "benchmarks/stage3/concurrent_*.json",
]


def mode_groups(result: dict[str, Any]):
    """Yield (workload, mode, group) for every measured workload/mode pair."""
    for workload, workload_info in result.get("workloads", {}).items():
        if not isinstance(workload_info, dict):
            continue
        for mode in ("warm", "cold"):
            group = workload_info.get(mode)
            if isinstance(group, dict) and "direct" in group:
                yield workload, mode, group


def group_totals(group: dict[str, Any]) -> tuple[int, int, int]:
    ok = fail = total = 0
    for side in ("direct", "gateway"):
        summary = group.get(side) or {}
        ok += summary.get("requests_successful", 0)
        fail += summary.get("requests_failed", 0)
        total += summary.get("requests_total", 0)
    return ok, fail, total


def failures(result: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for workload, mode, group in mode_groups(result):
        for side in ("direct", "gateway"):
            summary = group.get(side) or {}
            for failure in summary.get("failures", []):
                found.append(
                    {
                        "workload": workload,
                        "mode": mode,
                        "side": side,
                        **failure,
                    }
                )
    return found


def main() -> int:
    paths: list[str] = []
    for pattern in PATTERNS:
        paths.extend(sorted(glob.glob(str(PROJECT_ROOT / pattern))))
    if not paths:
        raise SystemExit("No result files found.")

    grand_ok = grand_fail = grand_total = 0
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        ok = fail = total = 0
        for _workload, _mode, group in mode_groups(result):
            mode_ok, mode_fail, mode_total = group_totals(group)
            ok += mode_ok
            fail += mode_fail
            total += mode_total
        grand_ok += ok
        grand_fail += fail
        grand_total += total
        gpu = result.get("gpu", {}).get("summary", {})
        temp = gpu.get("temperature_c") or {}
        memory = gpu.get("memory_used_mib") or {}
        kv_p95 = []
        for _workload, _mode, group in mode_groups(result):
            usage = (
                group.get("kv_cache", {})
                .get("summary", {})
                .get("kv_cache_usage_perc")
                or {}
            )
            if usage.get("p95") is not None:
                kv_p95.append(usage["p95"])
        config = result.get("vllm_config", {})
        print(
            f"{Path(path).name}: KV {config.get('kv_mib')} MiB, "
            f"seqs {config.get('max_num_seqs')}, "
            f"{ok}/{total} ok, failed {fail}, "
            f"GPU max {temp.get('max')} C, "
            f"mem max {memory.get('max')} MiB, "
            f"max KV p95 {round(100.0 * max(kv_p95), 2) if kv_p95 else 'n/a'}%"
        )
        for failure in failures(result):
            print(f"  FAILURE: {failure}")

    print(f"TOTAL: {grand_ok}/{grand_total} requests ok, {grand_fail} failed")
    return 0 if grand_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
