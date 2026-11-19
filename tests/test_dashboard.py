from __future__ import annotations

import logging
import unittest
from unittest import mock

import httpx

from app.config import settings
from app.main import app
from app.services.metrics import metrics
from app.workers import LocalVLLMWorker, MockWorker


class FakeSemanticRouter:
    def route(self, prompt: str):
        return {
            "category": "coding",
            "scores": {"coding": 0.8, "reasoning": 0.2, "general": 0.1},
            "top_score": 0.8,
            "margin": 0.6,
        }


class DashboardApiTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.old_health = settings.health_check_enabled
        self.old_faults = settings.dashboard_fault_injection_enabled
        self.old_disable = logging.root.manager.disable
        settings.health_check_enabled = False
        settings.dashboard_fault_injection_enabled = False
        logging.disable(logging.CRITICAL)

        self.mock_worker = MockWorker(worker_id="dashboard-mock")
        self.real_worker = LocalVLLMWorker(
            worker_id="dashboard-local",
            base_url="http://127.0.0.1:1",
            model="test-model",
        )
        self.router_patch = mock.patch("app.main.SemanticRouter", FakeSemanticRouter)
        self.build_patch = mock.patch(
            "app.main.build_workers",
            return_value=[self.real_worker, self.mock_worker],
        )
        self.router_patch.start()
        self.build_patch.start()
        self.lifespan = app.router.lifespan_context(app)
        await self.lifespan.__aenter__()
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://dashboard-test",
            timeout=10.0,
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.build_patch.stop()
        self.router_patch.stop()
        settings.health_check_enabled = self.old_health
        settings.dashboard_fault_injection_enabled = self.old_faults
        logging.disable(self.old_disable)

    async def test_dashboard_and_metrics_endpoints(self) -> None:
        page = await self.client.get("/dashboard")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers.get("content-type", ""))
        self.assertIn("Distributed Speculative Decoder Inference", page.text)

        before = metrics.snapshot()["requests_total"]
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("x-request-id"))
        self.assertEqual(response.headers.get("x-semantic-category"), "coding")
        self.assertEqual(response.headers.get("x-worker-id"), "dashboard-mock")
        self.assertEqual(response.headers.get("x-routing-decision"), "load-health-fallback")

        overview = await self.client.get("/api/metrics/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertGreaterEqual(overview.json()["requests_total"], before + 1)

        workers = await self.client.get("/api/metrics/workers")
        self.assertEqual(
            {item["worker_id"] for item in workers.json()["workers"]},
            {"dashboard-mock", "dashboard-local"},
        )
        self.assertIn("worker_metrics", overview.json())
        recent = await self.client.get("/api/metrics/recent?limit=10")
        self.assertTrue(recent.json()["requests"])
        vllm = await self.client.get("/api/metrics/vllm")
        self.assertEqual(vllm.status_code, 200)
        self.assertFalse(vllm.json()["available"])

    async def test_vllm_metrics_parser_exposes_speculation_and_token_latency(self) -> None:
        exposition = """# HELP vllm metrics
vllm:spec_decode_num_draft_tokens_total 100
vllm:spec_decode_num_accepted_tokens_total 60
vllm:time_to_first_token_seconds_sum 2
vllm:time_to_first_token_seconds_count 10
vllm:time_per_output_token_seconds_sum 0.5
vllm:time_per_output_token_seconds_count 10
vllm:inter_token_latency_seconds_sum 0.2
vllm:inter_token_latency_seconds_count 10
vllm:time_to_first_token_seconds_bucket{le="0.1"} 5
vllm:time_to_first_token_seconds_bucket{le="0.2"} 9
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10
vllm:time_per_output_token_seconds_bucket{le="0.01"} 5
vllm:time_per_output_token_seconds_bucket{le="0.1"} 9
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 10
vllm:inter_token_latency_seconds_bucket{le="0.01"} 5
vllm:inter_token_latency_seconds_bucket{le="0.02"} 9
vllm:inter_token_latency_seconds_bucket{le="+Inf"} 10
"""
        async def fake_metrics_text(_worker):
            return exposition
        with mock.patch.object(LocalVLLMWorker, "metrics_text", fake_metrics_text):
            response = await self.client.get("/api/metrics/vllm")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["speculative_decoding"]["acceptance_rate"], 0.6)
        self.assertEqual(data["launch_profile"]["draft_model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(data["launch_profile"]["kv_cache_memory_mib"], 512)
        self.assertEqual(data["histogram_quantiles_ms"]["ttft"]["p50"], 100.0)
        self.assertEqual(data["inter_token_jitter_ms"], 10.0)

    async def test_worker_controls_and_dynamic_mock_registration(self) -> None:
        disabled = await self.client.put("/api/workers/dashboard-mock/availability", json={"available": False})
        self.assertEqual(disabled.status_code, 200)
        self.assertFalse(app.state.worker_registry.require("dashboard-mock").available)
        enabled = await self.client.put("/api/workers/dashboard-mock/availability", json={"available": True})
        self.assertEqual(enabled.status_code, 200)
        added = await self.client.post("/api/workers/mock", json={"worker_id": "dashboard-extra", "latency_ms": 5, "max_concurrency": 3})
        self.assertEqual(added.status_code, 200)
        self.assertEqual(app.state.worker_registry.require("dashboard-extra").metadata.provider, "mock")
        removed = await self.client.delete("/api/workers/dashboard-extra")
        self.assertEqual(removed.status_code, 200)
        self.assertIsNone(app.state.worker_registry.get("dashboard-extra"))

    async def test_fault_injection_is_disabled_and_mock_only(self) -> None:
        payload = {
            "worker_id": "dashboard-mock",
            "action": "unhealthy",
            "confirm": True,
        }
        denied = await self.client.post("/api/testing/faults", json=payload)
        self.assertEqual(denied.status_code, 403)

        settings.dashboard_fault_injection_enabled = True
        missing_confirmation = await self.client.post(
            "/api/testing/faults",
            json={**payload, "confirm": False},
        )
        self.assertEqual(missing_confirmation.status_code, 400)

        real_worker_target = await self.client.post(
            "/api/testing/faults",
            json={**payload, "worker_id": "dashboard-local"},
        )
        self.assertEqual(real_worker_target.status_code, 404)

        applied = await self.client.post("/api/testing/faults", json=payload)
        self.assertEqual(applied.status_code, 200)
        self.assertEqual(
            app.state.worker_registry.require("dashboard-mock").state.value,
            "unhealthy",
        )

        restored = await self.client.post(
            "/api/testing/faults",
            json={**payload, "action": "restore"},
        )
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(
            app.state.worker_registry.require("dashboard-mock").state.value,
            "healthy",
        )


if __name__ == "__main__":
    unittest.main()
