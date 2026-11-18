# Limitations and validation status

This document records what was **not** validated, so results are not overstated.

## Validation status

- **Live OpenRouter validation passed on 2026-11-18.** The credential was
  accepted (`GET /models` returned `200`, 443 models) and a cloud-only gateway
  registered `worker2-openrouter` with no local vLLM or mock worker. One
  non-streaming request (`200`, real content `OK`, 100 total tokens, cost `0`)
  and one streaming request (`200`, 67 SSE chunks, terminal `[DONE]`, real
  content `OK`) succeeded. Evidence:
  `benchmarks/cloud/openrouter_gateway_20261118T152817Z.json`.
- **The originally configured model had been retired.** OpenRouter returned
  `404` for `qwen/qwen3-4b:free` with "no provider serves this model anymore".
  The working free model `liquid/lfm-3-1.7b:free` was validated and persisted in
  `.env`. A second free model tested during diagnosis
  (`google/gemma-4-26b-a4b-it:free`) returned upstream `429` from the shared free
  pool. Free model availability and capacity can change without notice; verify
  the configured model before relying on the cloud worker.
- **No Git history exists.** The project directory is not a Git repository and
  was not initialized, reset or rewritten. All changes are working-tree changes;
  rollback relies on the existing `backups/` archives and new timestamped
  artifacts, not on commits.
- **Stage 5–9 work described in earlier planning notes is not implemented here.**
  The multi-worker extension in this repository covers the requested cloud
  worker, mock worker, registry, health monitor, scheduler and failover routing;
  it does not add a distributed control plane, persistence, authentication or a
  deployment orchestrator.

## Benchmark scope

- The Stage 4 gateway benchmark runs the FastAPI app in-process against
  `tests/stub_upstream.py`. It isolates gateway CPU/embedding/event-loop cost but
  excludes real GPU inference, network latency and real vLLM queueing. It must
  not be presented as end-to-end model latency.
- The multi-worker routing benchmark uses three deterministic mock workers with
  simulated latency. It measures scheduler, registry and dispatch overhead, not
  provider performance or real failover timing.
- The live OpenRouter check is two sequential requests on a free model. It
  confirms credentials, registration, routing headers, streaming and error-free
  completion end to end; it is not a latency benchmark, concurrency test or
  quota/rate-limit characterization.
- Stage 4 measurements are single-host and short (3 repetitions × 100 requests
  per concurrency level). They are comparative evidence, not a capacity plan or
  a long-duration soak test.
- The live Worker 1 check is a short, sequential direct-vs-gateway run, not a
  randomized or thermal-controlled A/B. Its absolute throughput was much lower
  than the Stage 1 baseline in the same session (GPU ~31 W / ~42% utilization,
  no active throttle flags); only the paired delta is used.
- The first Stage 4 baseline artifact
  (`embedding_baseline-sync_20261118T140749Z.json`) contains a flawed aggregate
  calculation. It is preserved for history and explicitly superseded by
  `embedding_baseline-sync_20261118T141008Z.json`; do not use the first file for
  aggregate conclusions.
- Worker 1 Stage 2/3 results carry thermal caveats documented in
  `benchmarks/final_stage2_stage3_report_20261118T0000Z.md`: laptop GPU peaks of
  87–90 °C and run-to-run clock/thermal variation. Those benchmarks are not
  repeated here.

## Architecture limitations

- **Single-process state.** Registry, health state, round-robin position and
  in-flight counters live in one process. Multiple gateway replicas do not share
  load or health; each replica routes independently.
- **No admission queue.** `max_concurrency` is a scheduler weight, not a hard
  semaphore. If all workers are at capacity, the scheduler still selects the
  least-loaded worker and the upstream server queues or rejects the request.
- **Health is eventually consistent.** A worker can fail between checks; the
  first affected request sees the error and the monitor updates state on its next
  cycle. There is no circuit breaker, backoff or ejection window beyond the
  health state.
- **OpenRouter health checks reachability, not model availability.** The probe
  is `GET /models` with the configured credential. It confirms the provider is
  reachable and the key is accepted, but it does not confirm that the configured
  model is still served. A retired model can appear healthy while chat requests
  fail with `404` and the gateway returns `502`. Update `OPENROUTER_MODEL` when
  the provider retires a model.
- **Retry is deliberately narrow.** Only one retry, only on connection failure,
  only before response bytes flow. Timeouts, rate limits, provider errors and
  mid-stream failures are returned to the client. This avoids duplicate
  generation but means a transient timeout is not transparently recovered.
- **No response-body rewriting for streaming.** SSE chunks are forwarded as
  received. If a provider emits a nonstandard stream, the gateway does not
  normalize chunk contents.
- **No gateway-level auth, TLS, quotas or per-client rate limiting.** Deploy
  behind a reverse proxy or API gateway if those are required.
- **No persistence or observability backend.** There are no Prometheus metrics,
  traces or audit logs. `/health/workers` is the only structured runtime view.
- **Semantic category does not select a worker.** Every worker advertises the
  same `chat`/`streaming` capabilities, so routing is by health, availability
  and load. The category is returned in `X-Semantic-Category` and logged, but it
  is not a specialization key. Per-category worker pools would be a separate
  feature.
- **Model choice is static per worker.** Capability routing is by `chat` and
  `streaming`; there is no per-request model selection, cost-aware routing or
  token-budget routing.
- **Embedding model is fixed.** `sentence-transformers/all-MiniLM-L6-v2` is
  loaded from the Hugging Face cache; the first run needs network access or a
  pre-populated cache.
- **Mock worker is development-only.** It returns deterministic canned content
  and must not be used as a production fallback.

## Operational caveats

- Worker 1 requires a separate vLLM process with enough GPU memory for the target
  and draft models. `scripts/worker1.env` is the single source of truth; changing
  it changes later launches, not a running process.
- The gateway's `GET /health` is liveness only. A `200` does not mean any
  inference worker is usable; use `/health/workers` for that.
- Tests and benchmarks rely on the local stub upstream. A benign `BrokenPipeError`
  may be logged by the stub when a test intentionally times out; it is not a test
  failure.
- The first embedding request after process start pays model-load time. The
  benchmark warms up before timing, and the application lifespan loads the model
  before accepting traffic.
