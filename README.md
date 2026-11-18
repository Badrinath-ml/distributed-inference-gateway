# Distributed Inference Gateway

An OpenAI-compatible FastAPI gateway that classifies each prompt with a small
CPU embedding model and routes the request to one of several registered
inference workers:

- **Worker 1** — local vLLM serving `Qwen/Qwen3-1.7B` with `Qwen/Qwen3-0.6B`
  speculative decoding.
- **Worker 2** — optional OpenRouter cloud worker (registered only when both
  `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` are set).
- **Worker 3** — deterministic mock worker for development and tests (disabled
  by default, requires no GPU, network or API key).

The gateway keeps one process-wide view of worker health and load, routes by
capability and least in-flight/capacity ratio, retries once only on connection
failures before any response bytes flow, and maps worker failures to explicit
HTTP status codes.

## Status

| Workstream | State | Evidence |
|---|---|---|
| Stage 0–1: tooling, baseline, direct vs gateway | Complete | `benchmarks/worker1_baseline_stage1_20261117T1820Z.md` |
| Stage 2: KV cache sweep (256/384/512 MiB) | Complete; 512 MiB selected | `benchmarks/final_stage2_stage3_report_20261118T0000Z.md` |
| Stage 3: `max-num-seqs` 1 vs 2 | Complete; 2 selected | `benchmarks/final_stage2_stage3_report_20261118T0000Z.md` |
| Stage 4: CPU embedding/event-loop optimization | Complete | `benchmarks/stage4/stage4_before_after_20261118T1421Z.md` |
| Multi-worker architecture (cloud + mock + registry + health + scheduler) | Implemented and tested with mocks; live Worker 1 and live OpenRouter checks passed | `docs/benchmarks.md`, `docs/limitations.md` |

Stage 4 summary: CPU embedding moved off the event loop into a bounded
application-scoped executor. A dedicated matrix selected one worker thread and
eight PyTorch intra-op threads; see `docs/benchmarks.md` for before/after data.

## Requirements

- Python 3.12 (the tested environment).
- For Worker 1: a CUDA-capable vLLM installation and the Qwen models. vLLM is
  intentionally not listed in `requirements.txt`; install it according to its
  own platform instructions.
- No paid API is needed to run the gateway, tests or mock benchmarks.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# Edit .env if Worker 1 is not at http://localhost:8000.
```

Start Worker 1 (separate terminal, frozen configuration):

```bash
scripts/start_worker1.sh
```

Start the gateway:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Without Worker 1 or cloud credentials, enable the mock worker for a fully
offline run:

```bash
MOCK_WORKER_ENABLED=true WORKER1_ENABLED=false \
  .venv/bin/python -m uvicorn app.main:app --port 8080
```

`VLLM_BASE_URL` and `VLLM_MODEL` are required by the settings model. Keep them
in `.env` even when Worker 1 is disabled.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process liveness (always `200` while the app runs; does not include worker health). |
| `GET` | `/health/workers` | Registry snapshot: metadata, health state, availability, in-flight count, last error. |\n| `GET` | `/dashboard` | Browser-based inference evaluation console. |\n| `GET` | `/api/metrics/overview` | Bounded request telemetry, latency percentiles, errors and failovers. |\n| `GET` | `/api/metrics/workers` | Worker registry snapshot for dashboard. |\n| `GET` | `/api/metrics/recent?limit=100` | Recent request samples (maximum 200). |\n| `GET` | `/api/testing/capabilities` | Reports whether mock-only fault injection is enabled. |\n| `POST` | `/api/testing/faults` | Explicitly confirmed mock-worker fault action; disabled by default. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion; supports `stream: true`. |

Responses include `X-Semantic-Category` and `X-Worker-Id` headers. Streaming
responses pass upstream SSE through, including the terminal `data: [DONE]`.
The gateway overrides the request `model` with the selected worker's configured
model, so clients do not need to know which worker will serve them.

Error mapping:

| Condition | Status |
|---|---|
| No textual user message | `400` |
| No healthy, available, capable worker | `503` |
| Worker/provider rate limit | `429` with `Retry-After` when supplied |
| Worker/provider error, timeout, invalid response | `502` |

## Configuration

All settings are environment variables (or `.env` entries). See
`docs/configuration.md` for the full table and defaults. The most important are:

- `VLLM_BASE_URL`, `VLLM_MODEL` — Worker 1 endpoint and model.
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` — both required to register Worker 2.
- `EMBEDDING_EXECUTOR_WORKERS` (default `1`) and `EMBEDDING_TORCH_THREADS`
  (default unset) — Stage 4 embedding execution.
- `HEALTH_CHECK_INTERVAL_SECONDS`, `HEALTH_CHECK_TIMEOUT_SECONDS` — background
  worker health monitoring.
- `ROUTING_RETRY_ON_CONNECT_ERROR` (default `true`) — one retry on connection
  failure before any response bytes.

## Tests

The suite is standard-library `unittest` (no cloud provider calls, GPU or API
key). It uses a local stub upstream for inference; the embedding model is loaded
from the local Hugging Face cache (the first run may consult the hub if the model
is absent).

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

`pytest tests -q` works too when pytest is installed. The stub upstream used by
the tests is `tests/stub_upstream.py`; cloud responses are always mocked.

## Benchmarks

All benchmark artifacts are append-only and timestamped:

- `scripts/benchmark_stage4_gateway.py` — Stage 4 embedding/event-loop harness
  (in-process gateway + deterministic local stub upstream).
- `scripts/benchmark_multiworker.py` — scheduler/routing benchmark with three
  mock workers (no GPU or cloud call).
- `scripts/run_stage4_matrix.sh` — sequential executor/thread matrix.
- Existing Worker 1 benchmarks and reports remain under `benchmarks/`.

Results and the selected configuration are documented in `docs/benchmarks.md`.
The raw JSON/Markdown files are the source of truth.

## Documentation

- `docs/architecture.md` — request flow, worker contract, registry, scheduler,
  health monitor, failover and shutdown.
- `docs/configuration.md` — every environment variable and default.
- `docs/benchmarks.md` — Stage 4 baseline, matrix, selection and multi-worker
  routing results.
- `docs/limitations.md` — known limitations, validation status and operational
  caveats.
