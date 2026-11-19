from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway configuration, loaded from environment variables and ``.env``.

    Secrets are never given defaults. The OpenRouter worker is only registered
    when both ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODEL`` are present.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Worker 1: local vLLM (frozen validated launch configuration lives in
    # scripts/worker1.env; these fields only describe the gateway client).
    vllm_base_url: str
    vllm_model: str
    vllm_enable_thinking: bool = False
    worker1_enabled: bool = True
    worker1_id: str = "worker1-local-vllm"
    worker1_timeout_seconds: float = Field(default=120.0, gt=0)
    worker1_max_concurrency: int = Field(default=2, ge=1, le=64)

    # Worker 2: OpenRouter cloud worker (optional).
    openrouter_enabled: bool = True
    openrouter_api_key: str | None = None
    openrouter_model: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: float = Field(default=60.0, gt=0)
    openrouter_max_concurrency: int = Field(default=4, ge=1, le=64)
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "distributed-speculative-decoder-inference-semantic-routing-gateway"

    # Worker 3: deterministic mock worker (development/tests only).
    mock_worker_enabled: bool = False
    mock_worker_id: str = "worker3-mock"
    mock_worker_model: str = "mock-qwen"
    mock_worker_latency_ms: float = Field(default=0.0, ge=0)
    mock_worker_max_concurrency: int = Field(default=4, ge=1, le=64)

    # Stage 4 embedding execution.
    embedding_executor_workers: int = Field(default=1, ge=1, le=32)
    embedding_torch_threads: int | None = Field(default=None, ge=1, le=64)

    # Dashboard and development-only fault injection.
    dashboard_fault_injection_enabled: bool = False

    # Worker health monitoring.
    health_check_enabled: bool = True
    health_check_interval_seconds: float = Field(default=15.0, gt=0)
    health_check_timeout_seconds: float = Field(default=3.0, gt=0)

    # Retry only connection failures where no upstream work is expected.
    routing_retry_on_connect_error: bool = True

    gateway_port: int = 8080


settings = Settings()
