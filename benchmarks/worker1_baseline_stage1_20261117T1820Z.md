# Worker 1 baseline report: Qwen3-1.7B + Qwen3-0.6B speculative decoding

- Captured: 2026-11-17, 18:14–18:19 UTC
- Scope: Stage 0 tooling/tests plus Stage 1 baseline only. No production code was changed.
- Backup before changes: `backups/pre-worker1-20261117T180528Z.tar.gz`
- Repository has no Git metadata. Rollback is the backup tarball.

## Configuration under test

| Item | Value |
|---|---|
| vLLM | 0.33.0, V1 engine |
| Target model | `Qwen/Qwen3-1.7B`, FP16 |
| Draft model | `Qwen/Qwen3-0.6B`, FP16 |
| Speculative tokens | 2 |
| `max-model-len` | 1024 |
| `max-num-seqs` | 1 |
| `gpu-memory-utilization` | 0.80 |
| `kv-cache-memory-bytes` | 268435456 (256 MiB) |
| CUDA graphs | Disabled (`--enforce-eager`) |
| Sampler env | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Gateway | Unchanged `app.main:app`, uvicorn on port 8080 |

Startup facts from `benchmarks/logs/worker1_baseline_vllm_20261117T181157Z.log`:

- Model loading: **4.33 GiB**, **13.84 s**
- Engine init (profile, KV cache, warmup): **2.11 s**
- GPU KV cache: **1,168 tokens**, 73 blocks, **1.14x** maximum concurrency at 1,024 tokens/request
- Server ready: **18:12:38 UTC**

`--kv-cache-memory-bytes` bypasses memory profiling and overrides `gpu-memory-utilization`
(vLLM `vllm/v1/worker/gpu_worker.py:656-676`). The 0.80 value is therefore informational
for this run.

## Test suite

The unchanged application passes the new standard-library test suite:

- **15 tests passed** in 16.3 s
- Coverage: health, non-streaming completion, SSE forwarding and `[DONE]`, response headers,
  400/502 handling, upstream timeout, concurrent requests, lifespan initialization,
  router categories, and embedding consistency

## Stage 0: CPU embedding and routing

Model `sentence-transformers/all-MiniLM-L6-v2`, CPU, 8 torch threads.
Source: `benchmarks/worker1_embedding_stage0_20261117T1810Z.json`.

| Measurement | p50 (ms) | p95 (ms) | p99 (ms) | n |
|---|---:|---:|---:|---:|
| Encode, 1 prompt | 3.4639 | 5.1656 | 5.9027 | 50 |
| Encode, 4 prompts | 7.4064 | 8.7149 | 9.2524 | 25 |
| Route, 1 prompt | 3.7476 | 5.6185 | 7.0396 | 50 |

Concurrent routing on CPU was slower per call than serial routing, not faster:

| Workers | Calls | p50 (ms) | p95 (ms) | Categories match serial | Scores match serial |
|---:|---:|---:|---:|---|---|
| 1 | 48 | 3.9053 | 5.8133 | yes | yes |
| 2 | 48 | 17.2559 | 19.0607 | yes | yes |
| 4 | 48 | 27.0473 | 31.5348 | yes | yes |
| 8 | 48 | 78.8493 | 86.4281 | yes | yes |

This is CPU contention, so Stage 4 must test event-loop blocking and CPU oversubscription,
not just single-request latency.

## Stage 1: Direct vLLM versus gateway

Workload: streaming chat completion, 3 warmups + 30 timed runs, `max_tokens=150`,
temperature 0, one request at a time. Identical prompt for every request.

Artifacts:

- Direct: `benchmarks/worker1_baseline-direct-256mib-seqs1_20261117T181444Z.json`
- Gateway: `benchmarks/worker1_baseline-gateway-sync_20261117T181645Z.json`

| Metric | Direct vLLM | Gateway | Delta |
|---|---:|---:|---:|
| Successful requests | 30/30 | 30/30 | 0 |
| TTFT median (ms) | 102.29 | 149.96 | **+47.67 (+46.6%)** |
| TTFT p95 (ms) | 139.99 | 165.80 | +25.81 |
| TTFT p99 (ms) | 151.24 | 184.19 | +32.94 |
| Total median (ms) | 3,711.89 | 3,770.88 | **+58.99 (+1.59%)** |
| Total p95 (ms) | 4,179.13 | 4,143.91 | −35.22 |
| Total p99 (ms) | 4,240.69 | 4,264.21 | +23.52 |
| Inter-chunk median (ms) | 47.04 | 49.34 | +2.30 |
| Per-request throughput median (tok/s) | 41.58 | 41.36 | −0.22 (−0.52%) |
| Aggregate throughput (tok/s) | 40.07 | 39.33 | −0.73 (−1.83%) |

Interpretation limits:

- Routing costs only ~3.7 ms of CPU per request, so it does not explain the full 47.7 ms TTFT delta.
  The gateway also builds a new `httpx.AsyncClient` per request and re-streams SSE through ASGI.
- The direct run executed first with the GPU at 44 °C idle; the gateway run started at 66 °C idle.
  Run order and GPU thermals are a confound for this single sequential comparison.
- Prefix caching was active because every standard request used the same prompt. vLLM's own log
  showed a 52.5% prefix-cache hit rate by the end of these runs, so absolute TTFT is optimistic for
  both paths.

Gateway routing behavior was preserved: all 33 standard-prompt requests (30 timed + 3 warmup)
were classified `general`, and all 4 long-context requests were classified `coding`.
No gateway errors or non-200 responses were logged.

## Stage 1: Long-context capacity

Workload: one 800-token prompt + 100 completion tokens (estimated 900 total), 1 warmup + 3 runs.

| Metric | Direct vLLM | Gateway |
|---|---:|---:|
| Successful requests | 3/3 | 3/3 |
| TTFT median (ms) | 107.80 | 184.72 |
| TTFT p95 (ms) | 121.33 | 188.54 |
| Total median (ms) | 2,246.61 | 2,287.38 |
| Per-request throughput median (tok/s) | 47.09 | 47.66 |
| Aggregate throughput (tok/s) | 44.45 | 43.88 |
| Peak KV cache usage | 77.8% | 77.8% |
| Requests waiting | 0 | 0 |

Artifacts:

- Direct: `benchmarks/worker1_baseline-long-context-direct-800p_20261117T181847Z.json`
- Gateway: `benchmarks/worker1_baseline-long-context-gateway-800p_20261117T181903Z.json`

The 900-token request fits in the 1,168-token KV cache, leaving roughly 268 tokens of headroom.
This does not establish capacity for two concurrent long requests; 1.14x concurrency already shows
that two 1,024-token requests cannot both be resident. The long-context prompt was also identical
across runs, and the vLLM log reached a 78.3% prefix-cache hit rate, so repeat-run TTFT benefits
from prefix caching.

## Speculative decoding stayed active

Counters are deltas across the 30 timed standard runs. Direct and gateway values are identical,
which shows the gateway does not alter the speculative request path.

| Counter | Standard workload | Long-context workload |
|---|---:|---:|
| Drafts | 2,220 | 132 |
| Draft tokens | 4,440 | 264 |
| Accepted tokens | 2,310 | 171 |
| Acceptance rate | 52.03% | 64.77% |
| Mean acceptance length | 2.0405 | 2.2955 |

This is evidence that speculative decoding ran and how well it was accepted. It is **not** a
target-only A/B and must not be reported as a speculative-decoding speedup.

## GPU and KV telemetry (observed)

| Run | Idle temp | Mean temp | Max temp | Mean power | GPU memory used |
|---|---:|---:|---:|---:|---:|
| Direct standard | 44 °C | 76.4 °C | 85 °C | 53.4 W | 5,009 MiB |
| Gateway standard | 66 °C | 85.1 °C | 87 °C | 55.5 W | 5,009 MiB |
| Direct long-context | 67 °C | 83.1 °C | 84 °C | 55.0 W | 5,009 MiB |
| Gateway long-context | 65 °C | 81.0 °C | 83 °C | 55.7 W | 5,009 MiB |

- GPU memory usage was constant at 5,009 / 6,141 MiB once the model was loaded.
- Utilization during standard generation averaged 75.9–77.6%, median 78–81%.
- The 87 °C peak during the gateway standard run is close to typical laptop-GPU thermal limits.
  Longer sweeps need cooldown or run-order randomization to avoid thermal bias.
- KV cache usage was 8.3–16.7% for standard requests (median 9.7–10.4%) and 72.2–77.8% for
  long-context requests. No request ever waited in either workload.

## Observations

1. Speculative decoding is configured and active, with 52.0% acceptance on the standard workload
   and 64.8% on the 800-token prompt. Mean acceptance length is 2.04 and 2.30 respectively.
2. At concurrency 1, the gateway adds about 48 ms to TTFT but only about 59 ms to total latency
   (+1.6%) and about 0.5% to per-request throughput. The cause is not yet isolated; routing CPU
   time alone is too small to explain it.
3. A 900-token request succeeds with 77.8% KV usage and no waiting, but 256 MiB KV leaves little
   room. The 1.14x maximum concurrency confirms that two full-length requests cannot be resident.
4. The sequential run order and rising GPU temperature confound direct-versus-gateway timing.
   Any production claim needs an interleaved or randomized A/B, plus a concurrency workload.
5. Prefix caching makes repeated-identical-prompt latency optimistic. Workloads with unique
   prompts will show higher TTFT and different KV pressure.

## Open decisions before Stage 2

The next approved step is the KV cache sweep (`256 / 384 / 512 MiB`) followed by
`max-num-seqs 1` versus `2`, using workloads A/B/C. Each KV size requires restarting vLLM.
No restart, production edit, or sweep has been performed yet.

Servers left running after the baseline:

- vLLM: PID 27072 on port 8000, healthy
- Gateway: PID 27404 on port 8080, healthy

Both were left running because stopping inference servers requires approval.
