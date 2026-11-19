"""Deploy the project's Qwen3 speculative-decoding worker on Modal.

Deploy with: modal deploy modal_vllm.py
The Modal Server is private by default. Authenticate with a Modal Proxy Token:
Authorization: Bearer <TOKEN_ID>.<TOKEN_SECRET>
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "distributed-inference-gateway-vllm"
TARGET_MODEL = "Qwen/Qwen3-1.7B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"
PORT = 8000

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)

hf_cache = modal.Volume.from_name(
    "distributed-inference-gateway-hf-cache",
    create_if_missing=True,
)
vllm_cache = modal.Volume.from_name(
    "distributed-inference-gateway-vllm-cache",
    create_if_missing=True,
)

# Private by default. A Proxy Token is required for every request.
@app.server(
    image=vllm_image,
    gpu="A10",
    scaledown_window=5 * 60,
    startup_timeout=15 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    port=PORT,
    target_concurrency=2,
)
class QwenVLLMServer:
    @modal.enter()
    def start(self) -> None:
        speculative_config = {
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 2,
        }
        command = [
            "vllm", "serve", TARGET_MODEL,
            "--served-model-name", TARGET_MODEL,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--dtype", "half",
            "--max-model-len", "1024",
            "--max-num-seqs", "2",
            "--gpu-memory-utilization", "0.80",
            "--kv-cache-memory-bytes", "536870912",
            "--speculative-config", json.dumps(speculative_config),
            # Fast first deployment; CUDA graphs can be benchmarked after
            # confirming memory headroom and stable startup on the Modal GPU.
            "--enforce-eager",
        ]
        print("Starting vLLM:", " ".join(command), flush=True)
        self.process = subprocess.Popen(command)

        deadline = time.monotonic() + 14 * 60
        health_url = f"http://127.0.0.1:{PORT}/health"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {self.process.returncode}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        print("vLLM health check passed.", flush=True)
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(2)
        self.process.terminate()
        raise TimeoutError("vLLM did not become healthy within 14 minutes")

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        # Persist downloaded model weights and compiled artifacts for later starts.
        hf_cache.commit()
        vllm_cache.commit()


@app.local_entrypoint()
def show_configuration() -> None:
    """Print the immutable deployment profile before deploying."""
    print(f"App: {APP_NAME}")
    print(f"Target: {TARGET_MODEL}")
    print(f"Draft: {DRAFT_MODEL}")
    print("GPU: A10G | max sequences: 2 | speculative tokens: 2")
    print("Endpoint authentication: Modal Proxy Token (private by default)")
