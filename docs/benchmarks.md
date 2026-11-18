# Benchmarks

All benchmark artifacts are append-only and timestamped. Existing files were
never overwritten; `--force` was not used for any reported run.

## Stage 4: CPU embedding and event-loop optimization

### Problem

The original gateway called `EmbeddingService.encode()` directly inside the
async request handler. MiniLM inference is CPU-bound, so each request blocked
the asyncio event loop for the duration of tokenization and model inference.
Under concurrent load this serialized requests and delayed `/health`, streaming
and every other coroutine.

### Change

`EmbeddingExecutor` is a bounded `ThreadPoolExecutor` created once in the
FastAPI lifespan. The handler awaits `loop.run_in_executor(...)`, so embedding
work runs on worker threads while the event loop remains responsive. The model
is loaded once under a process-level lock and shared read-only between threads
(eval mode, `torch.inference_mode()`). The executor is shut down during lifespan
exit and is idempotent.

### Method

`scripts/benchmark_stage4_gateway.py` runs the real FastAPI app in-process
against `tests/stub_upstream.py`, so GPU inference and network time are
excluded. Each repetition measures concurrency levels 1, 2, 4 and 8 with 100
non-streaming chat requests per level, after 20 warm-up requests. A 10 ms
heartbeat records event-loop lag and a concurrent `GET /health` records probe
latency during the same load. Each configuration used 3 repetitions (300
requests per level) and 100 direct embedding samples.

Environment: Python 3.12.3, PyTorch 2.14.0, 16 logical CPUs, CPU-only embedding
`sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional normalized vectors.

### Before and after

Baseline = synchronous embedding on the event loop. Optimized = one executor
worker, eight PyTorch intra-op threads.

| Concurrency | Baseline req/s | Optimized req/s | Speedup | Baseline p50 ms | Optimized p50 ms | Baseline p95 ms | Optimized p95 ms | Baseline p99 ms | Optimized p99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.82 | 55.87 | 4.73x | 78.74 | 16.85 | 129.98 | 24.70 | 147.31 | 45.54 |
| 2 | 11.86 | 49.90 | 4.21x | 164.43 | 35.92 | 246.67 | 57.20 | 302.55 | 118.27 |
| 4 | 12.58 | 46.46 | 3.69x | 303.99 | 74.12 | 455.97 | 152.67 | 497.63 | 236.35 |
| 8 | 13.22 | 49.41 | 3.74x | 588.34 | 146.76 | 795.75 | 237.64 | 869.71 | 254.31 |

| Concurrency | Baseline loop lag p95 ms | Optimized loop lag p95 ms | Baseline health p95 ms | Optimized health p95 ms | Baseline CPU cores | Optimized CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 118.48 | 2.67 | 8.03 | 1.29 | 5.89 | 7.40 | 0 / 300 |
| 2 | 181.20 | 3.19 | 7.91 | 1.34 | 5.95 | 7.43 | 0 / 300 |
| 4 | 303.64 | 3.04 | 8.58 | 1.42 | 5.95 | 7.43 | 0 / 300 |
| 8 | 607.87 | 3.51 | 1.46 | 1.68 | 6.06 | 7.40 | 0 / 300 |

Event-loop lag p95 fell by 44x (concurrency 1) to 173x (concurrency 8). The
gateway used about 1.3–1.5 more CPU cores because work now overlaps instead of
blocking the loop; this is expected and still within the 16-core host.

Direct single-prompt embedding time was **not** improved, which is expected:
the executor changes where the work runs, not how fast the arithmetic is.
Baseline direct embedding was p50 7.51 ms / p95 13.29 ms / p99 18.10 ms;
optimized was p50 8.68 ms / p95 13.49 ms / p99 18.62 ms. Across the five matrix
runs, direct p50 ranged 8.12–9.79 ms, so the difference is run-to-run CPU
variation, not evidence of a faster encoder.

**Baseline artifact note.** The first baseline run
(`embedding_baseline-sync_20261118T140749Z.json`) has a flawed aggregate
calculation and is preserved only for history. The table above uses the
corrected raw-sample run `embedding_baseline-sync_20261118T141008Z.json`.
Do not mix the two.

### Executor/thread matrix

Five configurations were run sequentially to avoid cross-run CPU contention.
All used the same workload. "Total torch threads" is executor workers multiplied
by PyTorch intra-op threads.

| Executor workers | Torch threads | Total torch threads | Direct p50 ms | c1 req/s | c1 p50 ms | c8 req/s | c8 p50 ms | c8 p95 ms | c8 loop p95 ms | Max CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 8.68 | 55.87 | 16.85 | 49.41 | 146.76 | 237.64 | 3.51 | 7.43 | 0 |
| 2 | 4 | 8 | 9.70 | 52.87 | 17.54 | 50.57 | 148.45 | 214.54 | 4.84 | 6.79 | 0 |
| 2 | 8 | 16 | 9.30 | 30.61 | 38.20 | 32.02 | 241.67 | 312.34 | 3.29 | 3.49 | 0 |
| 4 | 4 | 16 | 9.79 | 47.45 | 18.91 | 30.07 | 251.29 | 401.89 | 12.63 | 12.28 | 0 |
| 4 | 8 | 32 | 8.12 | 34.17 | 36.60 | 37.79 | 206.80 | 250.84 | 4.24 | 3.51 | 0 |

Selection: **1 executor worker × 8 PyTorch threads**.

- It has the lowest single-request p50 (16.85 ms) and highest concurrency-1
  throughput, which is the latency-sensitive path that matters when upstream
  inference is the bottleneck.
- Event-loop lag stays at 2.67–3.51 ms p95 across all levels.
- It uses 8 total torch threads, matching the 8-thread baseline without
  oversubscribing the 16-core host.
- 2×4 is a credible alternative: similar latency, slightly better c8 p95 and
  lower CPU use, but slightly slower single-request p50 and higher loop lag.
  It remains documented, not selected, to keep one simple application-scoped
  executor and the best c1 latency.
- 2×8, 4×4 and 4×8 over-subscribe hardware threads once concurrency rises;
  throughput and tails degrade, and 4×4/4×8 show 8–13 ms loop-lag p95.

The selected configuration is 1 worker. `EMBEDDING_TORCH_THREADS=8` reproduces
the measured setup; `.env.example` shows it explicitly. PyTorch's default was
also 8 on this host, so the application default (unset) is equivalent here.

### Artifacts

- Baseline (superseded aggregate): `benchmarks/stage4/embedding_baseline-sync_20261118T140749Z.json`
- Baseline (corrected): `benchmarks/stage4/embedding_baseline-sync_20261118T141008Z.json`
- Optimized 1×8 (selected): `benchmarks/stage4/embedding_optimized-1w-t8_20261118T141548Z.json`
- Matrix 2×8: `benchmarks/stage4/embedding_optimized-2w-t8_20261118T141733Z.json`
- Matrix 2×4: `benchmarks/stage4/embedding_optimized-2w-t4_20261118T141822Z.json`
- Matrix 4×4: `benchmarks/stage4/embedding_optimized-4w-t4_20261118T141857Z.json`
- Matrix 4×8: `benchmarks/stage4/embedding_optimized-4w-t8_20261118T141942Z.json`
- Summary: `benchmarks/stage4/stage4_before_after_20261118T1421Z.md`

Every JSON has a matching Markdown table generated by the same script. Logs are
under `benchmarks/stage4/logs/`.

### Exact commands

```bash
EMBEDDING_EXECUTOR_WORKERS=1 EMBEDDING_TORCH_THREADS=8 \
  PYTHONPATH=. .venv/bin/python scripts/benchmark_stage4_gateway.py \
  --label optimized-1w-t8 --repetitions 3 \
  --requests-per-level 100 --embedding-samples 100

# Remaining matrix entries, sequentially:
scripts/run_stage4_matrix.sh

# Rebuild the comparison table from saved JSON:
PYTHONPATH=. .venv/bin/python scripts/summarize_stage4_matrix.py
```

## Multi-worker routing benchmark

`scripts/benchmark_multiworker.py` runs the gateway in-process with three
deterministic mock workers, 25 ms simulated latency each, `max_concurrency=4`,
and an embedding-free semantic router. This isolates registry, scheduler,
dispatch and failover behavior from CPU embedding and real providers. No GPU,
network, model download or API key is used.

| Concurrency | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Worker distribution | Errors |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 300 | 33.23 | 30.06 | 31.04 | 31.92 | 100 / 100 / 100 | 0 |
| 2 | 300 | 62.97 | 31.68 | 33.03 | 34.21 | 100 / 100 / 100 | 0 |
| 4 | 300 | 118.84 | 33.49 | 35.18 | 39.47 | 100 / 100 / 100 | 0 |
| 8 | 300 | 197.46 | 38.56 | 45.66 | 49.46 | 100 / 100 / 100 | 0 |

The 25 ms mock latency accounts for most of the p50; the additional ~5 ms at
concurrency 1 and ~13 ms at concurrency 8 is gateway/scheduler/ASGI overhead.
Distribution is exactly balanced across the three workers at every level.

Availability scenarios (each sent 40 requests at concurrency 2 unless noted):

| Scenario | Success | Distribution | Status codes |
|---|---:|---|---|
| One worker marked unhealthy | 40/40 | mock-2: 20, mock-3: 20 | 200: 40 |
| Worker recovered | 40/40 | mock-1: 14, mock-2: 13, mock-3: 13 | 200: 40 |
| All workers unavailable | 0/5 | none | 503: 5 |

The unhealthy worker received no traffic. Recovery returned it to rotation.
With every worker unavailable, the gateway returned `503` for every request
instead of attempting a worker call. Failure and retry semantics (429, 502,
timeout, connection retry) are covered by the automated suite rather than this
benchmark.

Artifact:
`benchmarks/multiworker/routing_three-mock-workers_20261118T142056Z.json`
(plus the matching `.md`).

Command:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_multiworker.py \
  --label three-mock-workers
```

## Live Worker 1 end-to-end check

A short live check was run after the Stage 4 and multi-worker work to confirm
that the new dispatch path still works against real vLLM, including semantic
routing, streaming, worker selection and response headers. Worker 1 was started
with the frozen `scripts/worker1.env` configuration (512 MiB KV, seqs2,
speculative decoding); `scripts/verify_worker1_config.py` reported PASS for env
defaults, launch expansion and the live process.

Workload: streaming chat completion, 2 warm-ups + 10 timed requests,
`max_tokens=150`, temperature 0, thinking disabled. Direct vLLM and gateway runs
were adjacent (direct first), so this is a paired check, not a randomized A/B.

| Metric | Direct vLLM | Gateway | Delta |
|---|---:|---:|---:|
| Successful requests | 10/10 | 10/10 | 0 |
| TTFT median | 267.79 ms | 391.76 ms | +123.97 ms |
| TTFT p95 | 336.38 ms | 461.54 ms | +125.16 ms |
| Total median | 9,319.86 ms | 9,252.24 ms | −67.63 ms |
| Total p95 | 9,975.48 ms | 9,657.81 ms | −317.68 ms |
| Inter-chunk median | 120.74 ms | 116.80 ms | −3.94 ms |
| Aggregate throughput | 15.94 tok/s | 16.19 tok/s | +1.6% |
| Speculative acceptance | 52.03% | 52.03% | identical |

A non-streaming request to the gateway also returned `200`, a valid completion,
`X-Semantic-Category: general`, `X-Worker-Id: worker1-local-vllm`, and
`/health/workers` returned `in_flight: 0` after the load.

**Session caveat.** Absolute throughput in this session was much lower than the
Stage 1 baseline (about 16 tok/s versus 41.6 tok/s). The GPU drew only ~31 W and
averaged ~42% utilization at 60–66 °C, with no active power or thermal throttle
flags; the Stage 1 run drew ~55 W and averaged ~76%. The cause was not isolated,
so absolute numbers should not be compared across sessions. The valid comparison
is the paired delta. The gateway/direct TTFT ratio was 1.46x, almost identical to
the Stage 1 ratio of 1.47x; the larger absolute delta reflects the slower
session. Direct ran first and the gateway run was slightly warmer, which would
bias against the gateway.

Artifacts:

- `benchmarks/worker1_stage4-e2e-direct_20261118T142528Z.json`
- `benchmarks/worker1_stage4-e2e-gateway_20261118T142724Z.json`
- `benchmarks/logs/worker1_stage4_e2e_20261118T1423Z.log`
- `benchmarks/logs/gateway_stage4_e2e_20261118T1423Z.log`

The gateway and vLLM processes were stopped after the check, restoring the
pre-check idle state.

## Live OpenRouter cloud check

A short live check confirmed the cloud worker end to end with a real credential,
not a mocked transport. It ran on 2026-11-18 at 15:28 UTC against
`liquid/lfm-3-1.7b:free` with a cloud-only gateway
(`WORKER1_ENABLED=false`, `MOCK_WORKER_ENABLED=false`), so every completion had
to come from OpenRouter. The semantic router and embedding executor still ran
locally, as in normal gateway operation.

Workload: one non-streaming and one streaming request, prompt "Reply with exactly:
OK", `max_tokens=512`, temperature 0. This is a smoke check, not a benchmark.

| Request | HTTP | Gateway latency | First chunk | Chunks / `[DONE]` | Worker header | Category | Content | Usage | Cost |
|---|---:|---:|---:|---|---|---|---|---|---:|
| Non-streaming | 200 | 1,071 ms | — | — | `worker2-openrouter` | `general` | `OK` | 15 prompt + 85 completion = 100 | 0 |
| Streaming | 200 | 742.26 ms | 595.54 ms | 67 / yes | `worker2-openrouter` | `general` | `OK` | not requested | 0 |

`/health/workers` reported exactly one worker, provider `openrouter`, state
`healthy`. The provider reported 83 reasoning tokens and 2 visible tokens for the
non-streaming reply; the free model reasons before answering. The streaming
response passed the terminal `data: [DONE]` through unchanged.

**Retired-model finding.** The model originally configured in `.env` was
`qwen/qwen3-4b:free`. The credential was valid (`GET /models` returned `200` and
443 models), but chat requests returned `404`: "no provider serves this model
anymore." The worker and gateway error paths behaved correctly, but the health
probe still reported the worker healthy because `GET /models` succeeds even when
the configured model is absent from the catalog. After testing alternatives, the
free `liquid/lfm-3-1.7b:free` was selected and persisted in `.env`. A different
free model tested during diagnosis (`google/gemma-4-26b-a4b-it:free`) returned an
upstream shared-pool `429`.

Artifacts:

- Live gateway JSON: `benchmarks/cloud/openrouter_gateway_20261118T152817Z.json`
- Live gateway log: `benchmarks/cloud/gateway_live_openrouter_20261118T152817Z.log`
- Direct retired-model `404`: `benchmarks/cloud/openrouter_direct_20261118T152620Z.json`
- Direct free-model `429`: `benchmarks/cloud/openrouter_direct_20261118T152656Z.json`
- Direct successful checks: `benchmarks/cloud/openrouter_direct_20261118T152703Z.json`,
  `openrouter_direct_20261118T152716Z.json`,
  `openrouter_direct_20261118T152752Z.json`

The gateway was stopped after the check. To reproduce the HTTP path (with a
running gateway and a valid `.env`):

```bash
WORKER1_ENABLED=false MOCK_WORKER_ENABLED=false \
  OPENROUTER_MODEL=liquid/lfm-3-1.7b:free \
  .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8080

curl -sS http://127.0.0.1:8080/health/workers
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":512,"temperature":0}'
```

Free model names and upstream capacity change frequently. If a request returns
`502` while `/health/workers` is green, check the provider error and update
`OPENROUTER_MODEL`; see `docs/limitations.md`.

## Interpretation limits

- Stage 4 excludes real vLLM/GPU time by design. It measures gateway overhead,
  not model latency.
- Multi-worker results use mock workers; they validate selection and balance,
  not OpenRouter latency or quota behavior.
- CPU benchmarks are short and single-host. They are comparative, not capacity
  planning or soak tests.
- The live OpenRouter check is two sequential requests on a free model. It
  confirms registration, routing, streaming and a real provider response, but it
  is not a latency benchmark or a quota/rate-limit characterization. Free model
  availability and capacity are external and can change without notice.
- The OpenRouter health probe does not verify model availability; see
  `docs/limitations.md`.
