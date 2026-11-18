# Configuration

Settings are loaded by `pydantic-settings` from environment variables and an
optional `.env` file in the process working directory. `.env` is not committed;
`.env.example` documents the supported variables.

## Worker 1 — local vLLM

| Variable | Default | Notes |
|---|---|---|
| `VLLM_BASE_URL` | required | Base URL of the OpenAI-compatible vLLM server, e.g. `http://localhost:8000`. |
| `VLLM_MODEL` | required | Model name sent upstream, e.g. `Qwen/Qwen3-1.7B`. |
| `WORKER1_ENABLED` | `true` | Set `false` to skip Worker 1. |
| `WORKER1_ID` | `worker1-local-vllm` | Registry identifier. |
| `WORKER1_TIMEOUT_SECONDS` | `120` | Non-streaming request timeout. |
| `WORKER1_MAX_CONCURRENCY` | `2` | Scheduler weight matching the selected vLLM `max-num-seqs=2`. |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | `3` | Also used for Worker 1's `/health` probe. |

Worker 1's launch configuration is separate and frozen in
`scripts/worker1.env`: `Qwen/Qwen3-1.7B`, draft `Qwen/Qwen3-0.6B`, FP16, two
speculative tokens, 512 MiB KV cache, `max-num-seqs=2`, max model length 1024,
eager mode, `gpu-memory-utilization=0.80`.

## Worker 2 — OpenRouter cloud

The worker is registered only when **both** `OPENROUTER_API_KEY` and
`OPENROUTER_MODEL` are non-empty and `OPENROUTER_ENABLED=true`. If only one of the
two is set, the gateway logs a warning and skips the worker. The API key is held
only in memory and is never written to logs or responses.

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_ENABLED` | `true` | Master switch. |
| `OPENROUTER_API_KEY` | unset | Required to register. Never commit a real value. |
| `OPENROUTER_MODEL` | unset | Required to register. Use a model currently served by OpenRouter, e.g. a free `:free` variant. Free models rotate; check `GET /models` if requests start failing. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. |
| `OPENROUTER_TIMEOUT_SECONDS` | `60` | Non-streaming request timeout. |
| `OPENROUTER_MAX_CONCURRENCY` | `4` | Scheduler weight. |
| `OPENROUTER_SITE_URL` | unset | Optional `HTTP-Referer` attribution header. |
| `OPENROUTER_APP_NAME` | `distributed-inference-gateway` | Optional `X-Title` attribution header. |

Health checks call `GET /models`, which does not generate tokens. This verifies
that the provider is reachable and the credential is accepted; it does **not**
verify that `OPENROUTER_MODEL` is still served. A retired model can therefore
show as healthy in `/health/workers` while chat requests return `502`. Check the
startup log and update the model if OpenRouter retires it. See
`docs/limitations.md`.

No paid call is made by the test suite or by either benchmark script.

## Worker 3 — mock worker

Disabled by default; intended for offline development and tests. It requires no
GPU, network, API key or model download. `MockWorker.fail_next()` and
`set_healthy()` provide deterministic fault injection in tests.

| Variable | Default | Notes |
|---|---|---|
| `MOCK_WORKER_ENABLED` | `false` | Enable to register the mock worker. |
| `MOCK_WORKER_ID` | `worker3-mock` | Registry identifier. |
| `MOCK_WORKER_MODEL` | `mock-qwen` | Model name returned in responses. |
| `MOCK_WORKER_LATENCY_MS` | `0` | Simulated response latency. |
| `MOCK_WORKER_MAX_CONCURRENCY` | `4` | Scheduler weight. |

## Stage 4 — CPU embedding execution

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_EXECUTOR_WORKERS` | `1` | Bounded thread pool size. Selected by the measured matrix in `docs/benchmarks.md`. |
| `EMBEDDING_TORCH_THREADS` | unset | `torch.set_num_threads()` value. Unset keeps PyTorch's default. The selected matrix value is applied only when set. |

The model itself is cached process-wide and shared by all executor threads; it is
immutable eval-mode inference, so no per-request model copy is made.

## Health monitoring and routing

| Variable | Default | Notes |
|---|---|---|
| `HEALTH_CHECK_ENABLED` | `true` | Starts the background monitor. |
| `HEALTH_CHECK_INTERVAL_SECONDS` | `15` | Delay between health cycles. |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | `3` | Per-worker check timeout. |
| `ROUTING_RETRY_ON_CONNECT_ERROR` | `true` | One retry on a different worker only for `WorkerUnavailableError`. |
| `GATEWAY_PORT` | `8080` | Informational; uvicorn's `--port` controls the actual listener. |

## Examples

Local Worker 1 plus gateway:

```dotenv
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=Qwen/Qwen3-1.7B
GATEWAY_PORT=8080
EMBEDDING_EXECUTOR_WORKERS=1
EMBEDDING_TORCH_THREADS=8
```

Offline mock-only gateway:

```dotenv
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=Qwen/Qwen3-1.7B
WORKER1_ENABLED=false
MOCK_WORKER_ENABLED=true
```

Local + cloud:

```dotenv
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=Qwen/Qwen3-1.7B
OPENROUTER_API_KEY=replace-me
OPENROUTER_MODEL=liquid/lfm-3-1.7b:free
```

## Validation and secret hygiene

- `OPENROUTER_API_KEY` has no default and is read only by `OpenRouterWorker`.
  It is never included in `WorkerMetadata.as_dict()` or `/health/workers`.
- Benchmark and test paths use mock transports or the local stub; they do not
  read the key from `.env`.
- Startup logs list worker metadata but not credentials.
- `.env.example` contains placeholders only. Check your local `.env` before
  sharing logs or archives.
