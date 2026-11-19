import unittest

from app.services.metrics import MetricsStore


class MetricsStoreTests(unittest.TestCase):
    def test_bounded_samples_and_lifetime_counters(self):
        store = MetricsStore(max_samples=2)
        for index in range(3):
            store.begin()
            store.finish({
                "request_id": str(index),
                "status_code": 200 if index < 2 else 502,
                "gateway_duration_ms": float(index + 1),
                "gateway_first_body_ms": 0.5,
                "streaming": index == 1,
                "worker_id": "mock-1",
            })
        snapshot = store.snapshot()
        self.assertEqual(snapshot["requests_total"], 3)
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["active_requests"], 0)
        self.assertEqual(snapshot["samples_retained"], 2)
        self.assertEqual(snapshot["gateway_duration_ms"]["p50"], 2.5)
        self.assertEqual(snapshot["streaming_requests_retained"], 1)

    def test_incomplete_response_counts_as_failed(self):
        store = MetricsStore()
        store.begin()
        store.finish({"status_code": 200, "gateway_duration_ms": 10.0, "completed": False})
        snapshot = store.snapshot()
        self.assertEqual(snapshot["requests_failed"], 1)
        self.assertEqual(snapshot["error_rate"], 1.0)

    def test_failover_counter(self):
        store = MetricsStore()
        store.record_failover()
        self.assertEqual(store.snapshot()["failovers_total"], 1)
        store.record_failover_result(True)
        store.record_failover_result(False)
        snapshot = store.snapshot()
        self.assertEqual(snapshot["failover_successes"], 1)
        self.assertEqual(snapshot["failover_failures"], 1)
        self.assertEqual(snapshot["failover_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
