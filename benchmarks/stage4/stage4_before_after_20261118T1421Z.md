# Stage 4 final report: CPU embedding offload and multi-worker routing

- Date: 2026-11-18 (UTC)
- Scope: gateway CPU embedding/event-loop optimization; executor/thread matrix;
  multi-worker scheduler/routing benchmark with mock workers; live Worker 1 and
  live OpenRouter cloud smoke checks.
- Status: **Stage 4 complete.** One executor worker with eight PyTorch threads
  is the selected configuration.
- Detailed documentation: `docs/benchmarks.md`, `docs/architecture.md`.

## What changed

Before: `SemanticRouter.route()` ran synchronously inside the async request
handler, blocking the event loop for every CPU embedding call.

After: a bounded, application-scoped `EmbeddingExecutor` runs embedding in a
worker thread via `loop.run_in_executor`. The model is loaded once under a
process-level lock and shared read-only. The executor is created in the FastAPI
lifespan and shut down on exit. In parallel, worker selection was generalized
behind a registry/scheduler with health monitoring, typed worker errors and one
safe retry on connection failure.

## Stage 4 before/after

Baseline = synchronous embedding. Optimized = 1 executor worker, 8 PyTorch
intra-op threads. Same workload, 3 repetitions, 100 requests per concurrency
level, in-process gateway against the deterministic stub upstream. GPU and
network time are excluded by design.

| Concurrency | Baseline req/s | Optimized req/s | Speedup | Baseline p50 ms | Optimized p50 ms | Baseline p95 ms | Optimized p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.82 | 55.87 | 4.73x | 78.74 | 16.85 | 129.98 | 24.70 |
| 2 | 11.86 | 49.90 | 4.21x | 164.43 | 35.92 | 246.67 | 57.20 |
| 4 | 12.58 | 46.46 | 3.69x | 303.99 | 74.12 | 455.97 | 152.67 |
| 8 | 13.22 | 49.41 | 3.74x | 588.34 | 146.76 | 795.75 | 237.64 |

| Concurrency | Baseline loop p95 ms | Optimized loop p95 ms | Baseline health p95 ms | Optimized health p95 ms | Baseline CPU cores | Optimized CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 118.48 | 2.67 | 8.03 | 1.29 | 5.89 | 7.40 | 0 |
| 2 | 181.20 | 3.19 | 7.91 | 1.34 | 5.95 | 7.43 | 0 |
| 4 | 303.64 | 3.04 | 8.58 | 1.42 | 5.95 | 7.43 | 0 |
| 8 | 607.87 | 3.51 | 1.46 | 1.68 | 6.06 | 7.40 | 0 |

Event-loop lag p95 improved by 44x at concurrency 1 and 173x at concurrency 8.
Request p50 improved by 4.0–4.7x. All 1,200 measured gateway requests per
configuration succeeded (0 errors). Single-call embedding time did not improve:
baseline p50 7.51 ms versus optimized 8.68 ms, within the 8.12–9.79 ms spread
across matrix runs. The executor fixes scheduling, not arithmetic.

## Executor/thread selection

| Workers × torch threads | Total torch threads | c1 req/s | c1 p50 ms | c8 req/s | c8 p50 ms | c8 p95 ms | c8 loop p95 ms | Max CPU cores |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 × 8 (selected)** | **8** | **55.87** | **16.85** | **49.41** | **146.76** | **237.64** | **3.51** | **7.43** |
| 2 × 4 | 8 | 52.87 | 17.54 | 50.57 | 148.45 | 214.54 | 4.84 | 6.79 |
| 2 × 8 | 16 | 30.61 | 38.20 | 32.02 | 241.67 | 312.34 | 3.29 | 3.49 |
| 4 × 4 | 16 | 47.45 | 18.91 | 30.07 | 251.29 | 401.89 | 12.63 | 12.28 |
| 4 × 8 | 32 | 34.17 | 36.60 | 37.79 | 206.80 | 250.84 | 4.24 | 3.51 |

Selection rationale: 1×8 gives the lowest single-request latency and highest
concurrency-1 throughput, keeps loop lag under 4 ms at every level, and does not
oversubscribe the 16-core host. 2×4 is a close alternative with a slightly
better c8 p95 but slower c1 and higher loop lag; it remains documented but not
selected. Configurations with 16–32 total torch threads degrade under load.

## Multi-worker routing benchmark

Three mock workers, 25 ms simulated latency, `max_concurrency=4`, embedding-free
router, no GPU/network/API key, 3 repetitions × 100 requests per level.

| Concurrency | req/s mean | p50 ms | p95 ms | p99 ms | Distribution | Errors |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 33.23 | 30.06 | 31.04 | 31.92 | 100 / 100 / 100 | 0 |
| 2 | 62.97 | 31.68 | 33.03 | 34.21 | 100 / 100 / 100 | 0 |
| 4 | 118.84 | 33.49 | 35.18 | 39.47 | 100 / 100 / 100 | 0 |
| 8 | 197.46 | 38.56 | 45.66 | 49.46 | 100 / 100 / 100 | 0 |

One unhealthy worker received zero traffic (40/40 succeeded on the remaining
two). Recovery restored it to rotation. With all workers unavailable, all 5
requests returned `503`. Scheduler selection is exactly balanced.

## Live Worker 1 end-to-end check

A short paired streaming check against real vLLM (2 warm-ups + 10 timed
requests, 150 tokens, temperature 0) confirmed the new dispatch path:

| Metric | Direct | Gateway | Delta |
|---|---:|---:|---:|
| Success | 10/10 | 10/10 | 0 |
| TTFT median | 267.79 ms | 391.76 ms | +123.97 ms |
| Total median | 9,319.86 ms | 9,252.24 ms | −67.63 ms |
| Aggregate throughput | 15.94 tok/s | 16.19 tok/s | +1.6% |
| Speculative acceptance | 52.03% | 52.03% | identical |

A non-streaming gateway request returned `X-Worker-Id: worker1-local-vllm` and
`X-Semantic-Category: general`; `/health/workers` returned `in_flight: 0` after
the load. Absolute throughput was ~2.6x lower than the Stage 1 baseline in this
session (GPU ~31 W, ~42% utilization, no active throttle flags), so only the
paired delta is comparable; the gateway/direct TTFT ratio (1.46x) matches the
Stage 1 ratio (1.47x). See `docs/benchmarks.md` for the caveat.

## Live OpenRouter cloud check

A cloud-only gateway (`WORKER1_ENABLED=false`, `MOCK_WORKER_ENABLED=false`) was
started against the real OpenRouter API with a valid credential and the free model
`liquid/lfm-3-1.7b:free`. `/health/workers` reported one healthy
`worker2-openrouter`. One non-streaming and one streaming request both returned
`200` with real content (`OK`), zero cost and the correct routing headers:

| Request | HTTP | Latency | First chunk | Chunks / `[DONE]` | Content | Usage |
|---|---:|---:|---:|---|---|---|
| Non-streaming | 200 | 1,071 ms | — | — | `OK` | 100 tokens |
| Streaming | 200 | 742.26 ms | 595.54 ms | 67 / yes | `OK` | not requested |

The originally configured `qwen/qwen3-4b:free` had been retired by the provider
(`404`, "no provider serves this model anymore"); the key itself was valid. The
gateway health probe still showed that worker healthy because `GET /models`
succeeds even when the configured model is absent. This is documented as a
limitation. `.env` now uses the validated free model. See `docs/benchmarks.md`
for the full live-check section and artifacts.

## Evidence

- Corrected baseline: `benchmarks/stage4/embedding_baseline-sync_20261118T141008Z.json`
- Selected optimized run: `benchmarks/stage4/embedding_optimized-1w-t8_20261118T141548Z.json`
- Matrix: `benchmarks/stage4/embedding_optimized-{2w-t8,2w-t4,4w-t4,4w-t8}_*.json`
- Multi-worker: `benchmarks/multiworker/routing_three-mock-workers_20261118T142056Z.json`
- Live E2E: `benchmarks/worker1_stage4-e2e-direct_20261118T142528Z.json`,
  `benchmarks/worker1_stage4-e2e-gateway_20261118T142724Z.json`
- Live OpenRouter gateway: `benchmarks/cloud/openrouter_gateway_20261118T152817Z.json`
- Live OpenRouter log: `benchmarks/cloud/gateway_live_openrouter_20261118T152817Z.log`
- Preserved superseded baseline: `benchmarks/stage4/embedding_baseline-sync_20261118T140749Z.json`

The first baseline artifact has a flawed aggregate calculation. It is kept for
history only; all numbers here use the corrected raw-sample run.

## Limitations

- The live OpenRouter check is two sequential requests on a free model. It
  confirms the real provider path, but it is not a latency benchmark, concurrency
  test or quota/rate-limit characterization.
- The OpenRouter health probe verifies reachability and credential acceptance,
  not model availability; a retired model can appear healthy until a chat request
  fails. Free model availability and capacity can change without notice.
- Stage 4 excludes GPU inference by using a local stub upstream. It does not
  claim end-to-end vLLM latency improvements.
- Short single-host runs; not a soak test or capacity plan. See
  `docs/limitations.md` for the full list.
