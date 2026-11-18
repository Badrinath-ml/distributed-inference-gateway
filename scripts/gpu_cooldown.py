#!/usr/bin/env python3
"""GPU cooldown helper for Worker 1 experiments.

This script only reads GPU telemetry with nvidia-smi. It never changes clocks,
power limits, fan curves or any other system setting. It waits until the GPU
temperature falls to the requested band, or until the timeout expires, and
records the full telemetry series for the experiment log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "stage2"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "nvidia-smi failed")
    line = process.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    if len(parts) < 6:
        raise RuntimeError(f"Unexpected nvidia-smi output: {parts}")
    return {
        "memory_used_mib": number(parts[0]),
        "utilization_gpu_pct": number(parts[1]),
        "temperature_c": number(parts[2]),
        "power_w": number(parts[3]),
        "clock_sm_mhz": number(parts[4]),
        "clock_mem_mhz": number(parts[5]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="cooldown")
    parser.add_argument("--target-c", type=float, default=52.0)
    parser.add_argument("--max-c", type=float, default=55.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--confirm-samples", type=int, default=2)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-if-not-reached", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_c < args.target_c:
        raise SystemExit("--max-c must be >= --target-c")
    if args.confirm_samples < 1:
        raise SystemExit("--confirm-samples must be >= 1")

    started = time.perf_counter()
    timestamp = utc_now()
    samples: list[dict] = []
    consecutive = 0
    reached = False
    start_snapshot = gpu_snapshot()

    print(
        f"Cooldown start: {start_snapshot['temperature_c']} C, "
        f"target <= {args.max_c} C (aim {args.target_c} C), timeout {args.timeout:.0f}s",
        flush=True,
    )

    last_report = 0.0
    while time.perf_counter() - started < args.timeout:
        snapshot = gpu_snapshot()
        elapsed = time.perf_counter() - started
        snapshot["elapsed_s"] = round(elapsed, 4)
        samples.append(snapshot)

        temperature = snapshot["temperature_c"]
        if temperature is not None and temperature <= args.max_c:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= args.confirm_samples:
            reached = True
            print(
                f"Cooldown reached {temperature} C after {elapsed:.1f}s",
                flush=True,
            )
            break

        if elapsed - last_report >= 15.0:
            print(
                f"  cooling: {temperature} C, util {snapshot['utilization_gpu_pct']}%, "
                f"power {snapshot['power_w']} W, elapsed {elapsed:.0f}s",
                flush=True,
            )
            last_report = elapsed

        time.sleep(args.interval)

    end_snapshot = gpu_snapshot()
    elapsed = time.perf_counter() - started
    if not reached:
        print(
            f"Cooldown timeout after {elapsed:.1f}s at {end_snapshot['temperature_c']} C",
            flush=True,
        )

    output = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"cooldown_{args.label}_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "label": args.label,
        "timestamp_utc": timestamp,
        "script": "scripts/gpu_cooldown.py",
        "target_c": args.target_c,
        "max_c": args.max_c,
        "timeout_s": args.timeout,
        "interval_s": args.interval,
        "reached_target": reached,
        "elapsed_s": round(elapsed, 4),
        "start": start_snapshot,
        "end": end_snapshot,
        "samples": samples,
        "note": "Read-only nvidia-smi telemetry; no GPU or fan settings were changed.",
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    if args.fail_if_not_reached and not reached:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
