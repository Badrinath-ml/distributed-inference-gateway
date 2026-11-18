#!/usr/bin/env bash
# Run the Stage 4 executor/thread matrix sequentially so measurements do not
# contend with each other. Each configuration writes a unique artifact pair.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks/stage4/logs

run() {
  local workers="$1"
  local threads="$2"
  local label="optimized-${workers}w-t${threads}"
  echo "[$(date -u +%FT%TZ)] starting ${label}"
  EMBEDDING_EXECUTOR_WORKERS="${workers}" EMBEDDING_TORCH_THREADS="${threads}" \
    PYTHONPATH=. .venv/bin/python scripts/benchmark_stage4_gateway.py \
      --label "${label}" \
      --repetitions 3 \
      --requests-per-level 100 \
      --embedding-samples 100 \
    2>&1 | tee "benchmarks/stage4/logs/${label}.log"
  echo "[$(date -u +%FT%TZ)] finished ${label}"
}

run 2 8
run 2 4
run 4 4
run 4 8

echo "[$(date -u +%FT%TZ)] matrix complete"
