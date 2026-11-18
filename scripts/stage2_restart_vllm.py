#!/usr/bin/env python3
"""Safely restart Worker 1 vLLM with an experimental KV-cache configuration.

Safety rules enforced by this script:

* Port 8000 must be owned by the PID recorded in /tmp/worker1_vllm.pid.
* That PID's command line must contain ``vllm``, ``serve`` and the target model.
* Only that verified process is signalled. No wildcard kills are used.
* Port 8080 is inspected but never modified.
* If the process does not exit after SIGTERM, the script aborts instead of
  escalating to SIGKILL.

The gateway is intentionally left running; only vLLM is restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PID_FILE = Path("/tmp/worker1_vllm.pid")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "stage2"
DEFAULT_LOG_DIR = PROJECT_ROOT / "benchmarks" / "logs"
TARGET_MODEL = "Qwen/Qwen3-1.7B"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def process_matches(pid: int) -> tuple[bool, str]:
    cmdline = read_cmdline(pid)
    if not cmdline:
        return False, "process not found"
    text = " ".join(cmdline)
    if "vllm" not in text or "serve" not in text:
        return False, f"not a vLLM serve process: {text[:200]}"
    if TARGET_MODEL not in cmdline:
        return False, f"target model {TARGET_MODEL} not present: {text[:200]}"
    return True, text


def port_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["ss", "-ltnp"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ss failed")
    pids: list[int] = []
    for line in result.stdout.splitlines():
        if f":{port} " not in line:
            continue
        for match in re.finditer(r"pid=(\d+)", line):
            pid = int(match.group(1))
            if pid not in pids:
                pids.append(pid)
    return pids


def gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    if len(parts) < 6:
        return {"error": f"Unexpected nvidia-smi output: {parts}"}
    return {
        "memory_used_mib": number(parts[0]),
        "utilization_gpu_pct": number(parts[1]),
        "temperature_c": number(parts[2]),
        "power_w": number(parts[3]),
        "clock_sm_mhz": number(parts[4]),
        "clock_mem_mhz": number(parts[5]),
    }


def health_ok(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for(predicate, timeout: float, interval: float, description: str) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    print(f"Timed out waiting for {description}", flush=True)
    return False


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "(log file missing)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def parse_startup_facts(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    facts: dict = {}
    match = re.search(
        r"GPU KV cache size:\s*([\d,]+) tokens,\s*Maximum concurrency for\s*([\d,]+) tokens per request:\s*([\d.]+)x",
        text,
    )
    if match:
        facts["kv_cache_tokens"] = int(match.group(1).replace(",", ""))
        facts["concurrency_tokens_per_request"] = int(match.group(2).replace(",", ""))
        facts["max_concurrency"] = float(match.group(3))
    match = re.search(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", text)
    if match:
        facts["model_memory_gib"] = float(match.group(1))
        facts["model_load_seconds"] = float(match.group(2))
    match = re.search(r"init engine .*? took ([\d.]+) s", text)
    if match:
        facts["engine_init_seconds"] = float(match.group(1))
    match = re.search(r"non-default args: (\{.*\})", text)
    if match:
        facts["non_default_args"] = match.group(1)
    return facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-mib", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--label", default=None)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_FILE)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gateway-port", type=int, default=8080)
    parser.add_argument("--stop-timeout", type=float, default=90.0)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--idle-memory-mib", type=float, default=1200.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--prestart-max-c", type=float, default=60.0)
    parser.add_argument("--benchmark-max-c", type=float, default=55.0)
    parser.add_argument("--cooldown-timeout", type=float, default=900.0)
    parser.add_argument("--skip-cooldown", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_cooldown(label: str, max_c: float, timeout: float) -> int:
    command = [
        sys.executable,
        "scripts/gpu_cooldown.py",
        "--label",
        label,
        "--target-c",
        str(max(45.0, max_c - 3.0)),
        "--max-c",
        str(max_c),
        "--timeout",
        str(timeout),
    ]
    print(f"Cooling before {label} to <= {max_c} C...", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return result.returncode


def main() -> int:
    args = parse_args()
    timestamp = utc_now()
    label = args.label or f"kv{args.kv_mib}mib-seqs{args.max_num_seqs}"
    log_path = DEFAULT_LOG_DIR / f"worker1_stage2_vllm_{label}_{timestamp}.log"
    output_path = DEFAULT_OUTPUT_DIR / f"startup_{label}_{timestamp}.json"

    print(f"Requested configuration: KV {args.kv_mib} MiB, max-num-seqs {args.max_num_seqs}")
    print(f"Log: {log_path}")

    # --- Inspect ports before doing anything destructive ---------------------
    port8000 = port_pids(args.port)
    port8080 = port_pids(args.gateway_port)
    print(f"Port {args.port} owner PIDs: {port8000}")
    print(f"Port {args.gateway_port} owner PIDs: {port8080} (not modified)")

    if not port8080:
        print("WARNING: gateway is not listening on port 8080; gateway benchmarks will fail.")

    expected_pid = None
    if args.pid_file.exists():
        try:
            expected_pid = int(args.pid_file.read_text().strip())
        except ValueError:
            print(f"WARNING: invalid PID file {args.pid_file}")

    start_snapshot = gpu_snapshot()
    print(f"GPU before restart: {start_snapshot}")

    if args.dry_run:
        print("Dry run: no process was stopped or started.")
        return 0

    # --- Stop only the verified vLLM process ---------------------------------
    if port8000:
        if len(port8000) != 1:
            print(f"ABORT: port {args.port} has multiple owners {port8000}; refusing to guess.")
            return 2
        pid = port8000[0]
        if expected_pid is not None and pid != expected_pid:
            print(
                f"ABORT: port {args.port} is owned by PID {pid}, but PID file says {expected_pid}."
            )
            return 2
        matches, details = process_matches(pid)
        if not matches:
            print(f"ABORT: refusing to signal PID {pid}: {details}")
            return 2
        print(f"Stopping verified vLLM PID {pid}")
        os.kill(pid, signal.SIGTERM)
        if not wait_for(
            lambda: not Path(f"/proc/{pid}").exists(),
            args.stop_timeout,
            0.5,
            f"PID {pid} to exit",
        ):
            print(f"ABORT: PID {pid} did not exit after SIGTERM; not escalating.")
            print(tail(log_path, 30))
            return 2
        if not wait_for(
            lambda: not port_pids(args.port),
            30.0,
            0.5,
            f"port {args.port} to be released",
        ):
            print(f"ABORT: port {args.port} is still occupied after PID {pid} exited.")
            return 2
        print("vLLM stopped and port released.")
    else:
        print(f"Port {args.port} is free; no process to stop.")
        if expected_pid is not None and Path(f"/proc/{expected_pid}").exists():
            print(
                f"ABORT: PID file points to live PID {expected_pid}, but it does not own port {args.port}."
            )
            return 2

    if not wait_for(
        lambda: (gpu_snapshot().get("memory_used_mib") or 0) <= args.idle_memory_mib,
        60.0,
        2.0,
        f"GPU memory to fall below {args.idle_memory_mib} MiB",
    ):
        print(f"ABORT: GPU memory did not return to idle: {gpu_snapshot()}")
        return 2

    prestart_cooldown = {"skipped": True}
    if not args.skip_cooldown:
        returncode = run_cooldown(
            f"prestart-{label}", args.prestart_max_c, args.cooldown_timeout
        )
        prestart_cooldown = {
            "skipped": False,
            "returncode": returncode,
            "max_c": args.prestart_max_c,
            "timeout_s": args.cooldown_timeout,
            "note": "returncode 2 means cooldown timed out and the start proceeded",
        }

    # --- Start the requested configuration -----------------------------------
    env = os.environ.copy()
    env["VLLM_KV_CACHE_MEMORY_BYTES"] = str(args.kv_mib * 1024 * 1024)
    env["VLLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "wb")
    started_at = time.monotonic()
    process = subprocess.Popen(
        ["bash", "scripts/start_worker1.sh"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    args.pid_file.write_text(f"{process.pid}\n")
    print(f"Started PID {process.pid}; waiting for health...")

    healthy = False
    while time.monotonic() - started_at < args.startup_timeout:
        if process.poll() is not None:
            break
        if health_ok(args.port):
            healthy = True
            break
        time.sleep(1.0)

    startup_seconds = time.monotonic() - started_at
    if not healthy:
        print(f"STARTUP FAILED after {startup_seconds:.1f}s (exit={process.poll()})")
        print("--- log tail ---")
        print(tail(log_path))
        if process.poll() is None:
            print(f"Stopping failed vLLM PID {process.pid} with SIGTERM")
            os.kill(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("WARNING: failed process did not exit after SIGTERM; not escalating.")
        result = {
            "schema_version": 1,
            "label": label,
            "timestamp_utc": timestamp,
            "ok": False,
            "kv_mib": args.kv_mib,
            "max_num_seqs": args.max_num_seqs,
            "pid": process.pid,
            "log": str(log_path),
            "startup_seconds": round(startup_seconds, 4),
            "gpu_before": start_snapshot,
            "gpu_after": gpu_snapshot(),
            "error": "startup or health check failed",
            "log_tail": tail(log_path),
        }
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved failure record: {output_path}")
        return 2

    time.sleep(max(0.0, args.settle_seconds))
    poststart_cooldown = {"skipped": True}
    if not args.skip_cooldown:
        returncode = run_cooldown(
            f"poststart-{label}", args.benchmark_max_c, args.cooldown_timeout
        )
        poststart_cooldown = {
            "skipped": False,
            "returncode": returncode,
            "max_c": args.benchmark_max_c,
            "timeout_s": args.cooldown_timeout,
            "note": "returncode 2 means cooldown timed out and benchmarking may start warm",
        }

    facts = parse_startup_facts(log_path)
    cmdline = read_cmdline(process.pid)
    result = {
        "schema_version": 1,
        "label": label,
        "timestamp_utc": timestamp,
        "ok": True,
        "kv_mib": args.kv_mib,
        "kv_bytes": args.kv_mib * 1024 * 1024,
        "max_num_seqs": args.max_num_seqs,
        "pid": process.pid,
        "log": str(log_path),
        "cmdline": cmdline,
        "startup_seconds": round(startup_seconds, 4),
        "gpu_before": start_snapshot,
        "gpu_after": gpu_snapshot(),
        "prestart_cooldown": prestart_cooldown,
        "poststart_cooldown": poststart_cooldown,
        "facts": facts,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"vLLM healthy after {startup_seconds:.1f}s")
    print(f"Startup facts: {facts}")
    print(f"GPU after startup: {result['gpu_after']}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
