#!/usr/bin/env bash
# Start Worker 1 with the configuration in scripts/worker1.env.
#
# Foreground use:
#   scripts/start_worker1.sh
#
# Background use with a log and PID file:
#   nohup scripts/start_worker1.sh > benchmarks/logs/worker1.log 2>&1 &
#   echo $! > /tmp/worker1_vllm.pid
#
# Print the exact expanded vLLM command without starting it:
#   scripts/start_worker1.sh --print
#
# Experiment overrides are environment variables, for example:
#   VLLM_KV_CACHE_MEMORY_BYTES=402653184 VLLM_MAX_NUM_SEQS=2 scripts/start_worker1.sh
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
# shellcheck source=/dev/null
source scripts/worker1.env

speculative_config="$(printf '{"method":"draft_model","model":"%s","num_speculative_tokens":%s}' \
  "$VLLM_DRAFT_MODEL" "$VLLM_NUM_SPECULATIVE_TOKENS")"

args=(
  serve "$VLLM_TARGET_MODEL"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --dtype "$VLLM_DTYPE"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES"
  --speculative-config "$speculative_config"
)

if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  args+=(--enforce-eager)
fi

if [[ "${1:-}" == "--print" ]]; then
  printf 'VLLM_USE_FLASHINFER_SAMPLER=%q ' "$VLLM_USE_FLASHINFER_SAMPLER"
  printf '%q ' "$project_root/.venv/bin/vllm" "${args[@]}"
  printf '\n'
  exit 0
fi

exec "$project_root/.venv/bin/vllm" "${args[@]}"
