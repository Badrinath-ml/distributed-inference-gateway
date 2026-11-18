# Architecture

The gateway is a single FastAPI process. It performs semantic routing locally
and forwards inference to any registered worker that exposes the internal
worker contract. There is no database, message queue or control plane; the
registry is an in-memory index of the workers configured at startup.

## Components

| Component | File | Responsibility |
|---|---|---|
| Application lifespan | `app/main.py` | Loads the router and embedding model, creates the embedding executor, builds and registers workers, starts/stops the health monitor, closes resources. |
| Semantic router | `app/services/router.py` | Classifies a prompt into `coding`, `reasoning` or `general` using MiniLM embeddings. |
| Embedding service | `app/services/embeddings.py` | CPU MiniLM encoder with a process-wide, lock-protected model cache. |
| Embedding executor | `app/services/embeddings.py` | Bounded `ThreadPoolExecutor` that runs CPU tokenization/model inference off the asyncio event loop. |
| Worker contract | `app/workers/base.py` | Metadata, health check, non-streaming completion, streaming handle and typed errors. |
| Local vLLM worker | `app/workers/local_vllm.py` | Calls Worker 1's OpenAI-compatible API and `/health`. |
| OpenRouter worker | `app/workers/openrouter.py` | Calls OpenRouter's OpenAI-compatible API; API key held only in memory. |
| Mock worker | `app/workers/mock.py` | Deterministic offline worker with controllable latency, health and faults. |
| Registry | `app/services/registry.py` | Worker records, health/availability state, in-flight counters, capability filtering. |
| Scheduler | `app/services/scheduler.py` | Filters eligible workers and selects the least loaded one with a rotating tie-break. |
| Health monitor | `app/services/health.py` | One background task per gateway; concurrent per-worker checks with a bounded timeout. |
| Inference dispatch | `app/services/inference.py` | Executes completions, applies the single safe retry, owns stream cleanup and reservation release. |

## Startup

1. `torch.set_num_threads(EMBEDDING_TORCH_THREADS)` is applied when configured,
   before embedding inference begins.
2. `SemanticRouter` loads `sentence-transformers/all-MiniLM-L6-v2` on CPU. The
   model load is guarded by a process-level lock and cache, so repeated lifespans
   in tests or reloads share one immutable eval-mode model.
3. `EmbeddingExecutor` is created with `EMBEDDING_EXECUTOR_WORKERS` threads.
4. `build_workers()` constructs the configured workers. The OpenRouter worker is
   skipped unless both the API key and model are present; a warning is logged if
   only one is set. The mock worker is registered only when explicitly enabled.
5. Each worker is registered, then the scheduler and health monitor are created.
6. The health monitor starts one background asyncio task. It does not create a
   task per request.

## Request flow

```
client
  │  POST /v1/chat/completions
  ▼
FastAPI handler (event loop)
  │  extract textual user messages
  ├─▶ embedding_executor.run(router.route, prompt)
  │      └─ worker thread: MiniLM tokenize + inference + category
  ├─▶ scheduler.select(required capabilities)
  │      └─ capable + healthy + available, min in_flight / max_concurrency
  ├─▶ worker.chat_completion(payload)  ── or ──  worker.open_stream(payload)
  │      └─ worker HTTP client (vLLM / OpenRouter / mock)
  ▼
JSONResponse with X-Semantic-Category + X-Worker-Id
  or StreamingResponse that forwards SSE and releases the reservation on close
```

The `model` field in the incoming body is replaced by the selected worker's
configured model. The caller does not choose a worker directly.

For streaming, the worker's `open_stream()` must finish connecting before it
returns a `WorkerStream`. Connection and pre-stream HTTP errors are therefore
still ordinary gateway errors. Once the stream is open, the registry reservation
is held until the response body finishes, fails or the client disconnects. A
`BackgroundTask` and an idempotent cleanup path guarantee exactly one release.

## Selection policy

`WorkerRegistry.eligible()` keeps a worker only when all of the following hold:

- state is not `UNHEALTHY`,
- the operator-controlled availability flag is true,
- the worker's own `is_available` property is true,
- it supports the required capabilities (`chat`, plus `streaming` for SSE).

The scheduler rotates the eligible list on each call to avoid always ordering
equal candidates the same way, then chooses the minimum
`(in_flight / max_concurrency, in_flight)` tuple. `max_concurrency` is a routing
weight, not a hard admission limit: vLLM and OpenRouter enforce their own
concurrency upstream, and the gateway does not queue requests beyond selecting a
worker. An in-flight counter is incremented under the registry lock on selection
and decremented on completion or error.

If no worker is eligible, the scheduler raises `NoEligibleWorkersError` with a
diagnostic reason (no workers, missing capability, or all unhealthy/unavailable).
The HTTP layer maps it to `503`.

## Health monitoring

The health monitor runs one cycle immediately after startup, then every
`HEALTH_CHECK_INTERVAL_SECONDS`. Each worker check runs concurrently and is
wrapped in `asyncio.wait_for(..., HEALTH_CHECK_TIMEOUT_SECONDS)`. A timeout,
exception or `False` result marks the worker `UNHEALTHY`; a later success marks
it `HEALTHY` again and resets the consecutive-failure count. `UNKNOWN` workers
are still eligible, which avoids a startup deadlock while the first check is
pending.

Health state changes are recorded in the registry and exposed by
`GET /health/workers`. `GET /health` is process liveness only; it deliberately
does not fail when every inference worker is down.

## Failure handling and retry policy

Workers translate transport and provider failures into typed errors:

| Error | Meaning | Gateway behavior |
|---|---|---|
| `WorkerUnavailableError` | Connection failed before processing | One retry on another worker (if enabled), otherwise `502` |
| `WorkerTimeoutError` | Configured timeout elapsed | `502`; never retried |
| `WorkerRateLimitError` | Upstream `429` | `429` with `Retry-After` when provided; never retried |
| `WorkerProviderError` | Other upstream error status | `502`; never retried |
| `WorkerResponseError` | Invalid/unexpected response body | `502`; never retried |

The retry is intentionally limited to one alternate worker and only while no
response bytes can have flowed. A timeout may have consumed upstream work, so it
is not retried; a rate limit is a provider decision, so it is not retried. For
streaming, only failures while opening the stream are retryable.

## Shutdown

On lifespan exit the gateway stops the health monitor, closes every worker's
HTTP client and shuts down the embedding executor with `wait=True`. The executor
rejects new work after shutdown and its `shutdown()` is idempotent. In-flight
requests are allowed to finish; the process does not drain a separate queue
because there is none.

## Scale and process model

This design targets one gateway process with a small number of workers. The
registry, scheduler rotation and in-flight counters are process-local. Running
multiple gateway replicas would give each replica an independent view of load and
health; that is acceptable for stateless HTTP routing but is not a distributed
scheduler. See `docs/limitations.md`.
