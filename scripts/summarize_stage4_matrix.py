#!/usr/bin/env python3
"""Summarize Stage 4 gateway benchmark JSON artifacts.

Reads every ``benchmarks/stage4/embedding_*.json`` file (never modifies them)
and prints a Markdown comparison table plus the selected executor matrix.

Example::

    PYTHONPATH=. .venv/bin/python scripts/summarize_stage4_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE4_DIR = PROJECT_ROOT / "benchmarks" / "stage4"


def load_artifacts() -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(STAGE4_DIR.glob("embedding_*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        data["_path"] = path
        artifacts.append(data)
    return artifacts


def short_label(data: dict[str, Any]) -> str:
    return data["label"]


def format_row(data: dict[str, Any], row: dict[str, Any]) -> str:
    latency = row["latency_ms"]
    lag = row["event_loop_lag_ms"]
    probe = row["health_probe_ms"]
    return (
        f"| `{short_label(data)}` | {row['concurrency']} | {row['requests']} | "
        f"{row['throughput_mean']} | {latency.get('median')} | "
        f"{latency.get('p95')} | {latency.get('p99')} | "
        f"{lag.get('p95')} | {lag.get('p99')} | {probe.get('p95')} | "
        f"{row['cpu_utilization_cores']} | {row['error_count']} |"
    )


def main() -> None:
    artifacts = load_artifacts()
    if not artifacts:
        raise SystemExit("No Stage 4 artifacts found")

    print("# Stage 4 artifact summary")
    print()
    print("| Label | Executor workers | Torch threads | Direct p50 ms | Direct p95 ms |")
    print("|---|---:|---:|---:|---:|")
    for data in artifacts:
        embedding = data["embedding"]["latency_ms"]
        print(
            f"| `{data['label']}` | "
            f"{data['gateway'].get('embedding_executor_workers')} | "
            f"{data['environment']['torch_num_threads']} | "
            f"{embedding.get('median')} | {embedding.get('p95')} |"
        )

    print()
    print("## Aggregate by concurrency")
    print()
    print(
        "| Label | Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | "
        "Loop p95 ms | Loop p99 ms | Health p95 ms | CPU cores | Errors |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for data in artifacts:
        for row in data["aggregate"]:
            print(format_row(data, row))

    print()
    print("## Artifact paths")
    print()
    for data in artifacts:
        print(f"- `{data['_path'].relative_to(PROJECT_ROOT)}`")


if __name__ == "__main__":
    main()
