# Deploy the Qwen3 speculative-decoding worker on Modal

This deployment moves only Worker 1 to Modal. The FastAPI gateway, semantic
router, scheduler, dashboard, metrics store and optional OpenRouter worker stay
where they are. The gateway connects to Modal through the existing OpenAI-
compatible vLLM worker interface.

## Deployment profile

| Setting | Value |
|---|---|
| Target | `Qwen/Qwen3-1.7B` |
| Draft | `Qwen/Qwen3-0.6B` |
| GPU | NVIDIA A10 |
| vLLM | `0.21.0` |
| Precision | FP16 |
| Speculative decoding | draft model, 2 speculative tokens |
| Maximum model length | 1024 |
| Maximum sequences | 2 |
| KV cache | 512 MiB |
| GPU memory utilization | 0.80 |
| Initial startup mode | eager |
| Autoscaling | serverless, scale down after 5 idle minutes |
| Endpoint access | private; Modal Proxy Token required |

The profile deliberately preserves the locally benchmarked model, draft model,
KV cache and sequence count. The Modal GPU is a different device from the
laptop RTX 4060, so rerun the direct-vLLM and gateway benchmarks after deployment.
Eager mode is selected for the first deployment to simplify startup; benchmark
CUDA graphs after confirming memory headroom.

## 1. Install and authenticate Modal

From the repository root, use Python 3.12:

```bash
python -m venv .venv-modal
source .venv-modal/bin/activate
pip install -r requirements-modal.txt
modal setup
```

`modal setup` opens the authentication flow and configures the local Modal
profile. Do not commit Modal credentials.

## 2. Create a private Proxy Token

In the Modal dashboard, create a **Proxy Token**. Keep both the Token ID and
Token Secret private. The combined gateway credential is:

```text
TOKEN_ID.TOKEN_SECRET
```

The deployment uses `@app.server` without `unauthenticated=True`, so Modal
authentication remains enabled. Do not make this inference endpoint public.

## 3. Deploy the worker

```bash
modal deploy modal_vllm.py
```

The first deployment builds the vLLM image and downloads both model weights.
This can take several minutes. Model weights and vLLM compilation artifacts
are stored in named Modal Volumes so subsequent container starts can reuse them.
Copy the private server URL printed by the deployment command.

## 4. Connect the existing gateway

In the gateway's local `.env`, replace the local URL with the deployed Modal
server URL and set the combined Proxy Token:

```dotenv
VLLM_BASE_URL=https://YOUR-MODAL-SERVER-URL
VLLM_MODEL=Qwen/Qwen3-1.7B
VLLM_API_KEY=TOKEN_ID.TOKEN_SECRET
WORKER1_MAX_CONCURRENCY=2
```

Keep `VLLM_API_KEY` out of Git, logs, screenshots and client-side code. The
gateway sends it as `Authorization: Bearer ...`; Modal accepts this combined
Proxy Token format. Restart the gateway after editing `.env`.

## 5. Verify

Check the gateway's worker registry:

```bash
curl -s http://localhost:8080/health/workers
```

Then send a streaming request through the gateway (not directly to Modal) and
confirm that the response identifies `worker1-local-vllm`. Check vLLM metrics:

```bash
curl -s http://localhost:8080/api/metrics/vllm
```

A successful deployment should pass health checks, return a completion, stream
SSE correctly, and expose vLLM Prometheus metrics. Run a paired direct-vs-gateway
benchmark and verify speculative draft/accepted token counters before claiming
performance parity with the local RTX 4060 run.

## Cost and lifecycle controls

Modal Starter currently includes $30/month in compute credits; this is a credit
allowance, not unlimited free GPU use. The published A10 rate is $0.000306 per
GPU-second (about $1.10 per GPU-hour), before CPU, memory, network and other
applicable charges. At that GPU-only rate, $30 corresponds to about 27.2 GPU
hours; actual usage will be lower after accounting for the other resources.

This server scales down after five idle minutes, but each active container is
billable while running. Avoid leaving test traffic or a warm server running
unnecessarily. Monitor the Modal Usage/Billing page before and after each
benchmark. Set any available spend controls in your workspace. The first model
download and persistent volumes also consume storage/network resources.

To remove the deployed app when finished:

```bash
modal app stop distributed-inference-gateway-vllm
```

Stopping the app stops active compute; named volumes remain until deleted
separately from the Modal dashboard/CLI. Do not delete the volumes if you want
to retain cached model weights for a later deployment.

## Troubleshooting

- **401:** check that `VLLM_API_KEY` is exactly `TOKEN_ID.TOKEN_SECRET` and
  the token belongs to the correct Modal workspace/environment.
- **503 during startup:** wait for model download and vLLM initialization; inspect
  the Modal app logs.
- **GPU unavailable:** check A10 availability and workspace GPU concurrency.
- **Out of memory:** retain the 1,024-token context and two-sequence limit; if
  needed, test a larger GPU before changing the validated model profile.
- **No speculative metrics:** check the vLLM logs for draft-model loading and
  inspect `/metrics` on the Modal server through the authenticated gateway.

## References

- [Modal vLLM inference example](https://modal.com/docs/examples/vllm_inference)
- [Modal Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth)
- [Modal pricing](https://modal.com/pricing)
