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
            "category": "general",
            "scores": {"coding": 0.1, "reasoning": 0.2, "general": 0.8},
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
            return_value=[self.mock_worker, self.real_worker],
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
        self.assertIn("Evaluation Console", page.text)

        before = metrics.snapshot()["requests_total"]
        response = await self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("x-request-id"))

        overview = await self.client.get("/api/metrics/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertGreaterEqual(overview.json()["requests_total"], before + 1)

        workers = await self.client.get("/api/metrics/workers")
        self.assertEqual(
            {item["worker_id"] for item in workers.json()["workers"]},
            {"dashboard-mock", "dashboard-local"},
        )
        recent = await self.client.get("/api/metrics/recent?limit=10")
        self.assertTrue(recent.json()["requests"])

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
