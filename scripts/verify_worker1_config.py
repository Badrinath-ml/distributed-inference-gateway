#!/usr/bin/env python3
"""Verify the promoted Worker 1 configuration without restarting vLLM.

Three independent checks:

1. `scripts/worker1.env` defaults equal the selected Stage 2/3 configuration.
2. `scripts/start_worker1.sh --print`, run with all VLLM_* variables removed
   from the environment, expands to the expected vLLM arguments. This proves
   the launch script reads the values from the env file.
3. The live vLLM process recorded in the PID file was started with the same
   arguments, so no restart is required to apply the promoted defaults.

The script only reads files and process metadata, prints a health status, and
never starts, stops or reconfigures a server.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "scripts" / "worker1.env"
START_SCRIPT = PROJECT_ROOT / "scripts" / "start_worker1.sh"
PID_FILE = Path("/tmp/worker1_vllm.pid")

EXPECTED_DEFAULTS = {
    "VLLM_TARGET_MODEL": "Qwen/Qwen3-1.7B",
    "VLLM_DRAFT_MODEL": "Qwen/Qwen3-0.6B",
    "VLLM_HOST": "0.0.0.0",
    "VLLM_PORT": "8000",
    "VLLM_DTYPE": "half",
    "VLLM_MAX_MODEL_LEN": "1024",
    "VLLM_MAX_NUM_SEQS": "2",
    "VLLM_GPU_MEMORY_UTILIZATION": "0.80",
    "VLLM_KV_CACHE_MEMORY_BYTES": str(512 * 1024 * 1024),
    "VLLM_NUM_SPECULATIVE_TOKENS": "2",
    "VLLM_ENFORCE_EAGER": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}

EXPECTED_ARGS = {
    "--host": "0.0.0.0",
    "--port": "8000",
    "--dtype": "half",
    "--max-model-len": "1024",
    "--max-num-seqs": "2",
    "--gpu-memory-utilization": "0.80",
    "--kv-cache-memory-bytes": str(512 * 1024 * 1024),
}


def parse_env_defaults(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    defaults: dict[str, str] = {}
    for match in re.finditer(
        r'^export (VLLM_[A-Z0-9_]+)="\$\{\1:-([^}]*)\}"', text, re.MULTILINE
    ):
        defaults[match.group(1)] = match.group(2)
    return defaults


def check_defaults(defaults: dict[str, str]) -> list[str]:
    problems = []
    for key, expected in EXPECTED_DEFAULTS.items():
        actual = defaults.get(key)
        if actual != expected:
            problems.append(f"{key}: expected {expected!r}, found {actual!r}")
    return problems


def expanded_command() -> list[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("VLLM_")}
    result = subprocess.run(
        ["bash", str(START_SCRIPT), "--print"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"start_worker1.sh --print failed: {result.stderr.strip()}")
    tokens = shlex.split(result.stdout.strip())
    # First token is the exported sampler setting, second the vllm binary.
    if "serve" not in tokens:
        raise SystemExit(f"Unexpected --print output: {result.stdout}")
    return tokens[tokens.index("serve") + 1 :]


def arg_value(args: list[str], flag: str) -> str | None:
    if flag not in args:
        return None
    index = args.index(flag)
    return args[index + 1] if index + 1 < len(args) else None


def check_args(args: list[str], prefix: str) -> list[str]:
    problems = []
    if "Qwen/Qwen3-1.7B" not in args:
        problems.append(f"{prefix}: target model Qwen/Qwen3-1.7B missing")
    for flag, expected in EXPECTED_ARGS.items():
        actual = arg_value(args, flag)
        if actual != expected:
            problems.append(f"{prefix}: {flag} expected {expected!r}, found {actual!r}")
    if "--enforce-eager" not in args:
        problems.append(f"{prefix}: --enforce-eager missing")
    spec_raw = arg_value(args, "--speculative-config")
    if spec_raw is None:
        problems.append(f"{prefix}: --speculative-config missing")
    else:
        try:
            spec = json.loads(spec_raw)
        except json.JSONDecodeError as exc:
            problems.append(f"{prefix}: invalid speculative JSON: {exc}")
        else:
            if spec.get("method") != "draft_model":
                problems.append(f"{prefix}: speculative method {spec.get('method')!r}")
            if spec.get("model") != "Qwen/Qwen3-0.6B":
                problems.append(f"{prefix}: draft model {spec.get('model')!r}")
            if spec.get("num_speculative_tokens") != 2:
                problems.append(
                    f"{prefix}: num_speculative_tokens "
                    f"{spec.get('num_speculative_tokens')!r}"
                )
    return problems


def running_args() -> tuple[list[str] | None, str]:
    if not PID_FILE.exists():
        return None, f"PID file {PID_FILE} missing"
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        return None, f"PID file {PID_FILE} invalid"
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return None, f"PID {pid} not running"
    tokens = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
    if "serve" not in tokens:
        return None, f"PID {pid} is not a vLLM serve process"
    return tokens[tokens.index("serve") + 1 :], f"PID {pid}"


def health_status(port: str) -> str:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2.0
        ) as response:
            return f"HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"unreachable ({exc})"


def main() -> int:
    print(f"Env file: {ENV_PATH}")
    defaults = parse_env_defaults(ENV_PATH)
    problems = check_defaults(defaults)
    print(f"  parsed defaults: {len(defaults)}; mismatches: {len(problems)}")
    for key in EXPECTED_DEFAULTS:
        print(f"    {key}={defaults.get(key)}")

    print("Launch script expansion (VLLM_* removed from environment):")
    launch_args = expanded_command()
    launch_problems = check_args(launch_args, "launch script")
    print(f"  args: {' '.join(launch_args)}")
    print(f"  mismatches: {len(launch_problems)}")

    print("Live process:")
    live_args, live_label = running_args()
    if live_args is None:
        live_problems = [f"live process: {live_label}"]
        print(f"  {live_label}")
    else:
        live_problems = check_args(live_args, "live process")
        print(f"  {live_label}; mismatches: {len(live_problems)}")
    print(f"  health: {health_status(EXPECTED_DEFAULTS['VLLM_PORT'])}")

    problems.extend(launch_problems)
    problems.extend(live_problems)
    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nPASS: env defaults, launch expansion and live process all match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
