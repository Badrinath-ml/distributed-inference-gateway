from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch

from app.services.embeddings import EmbeddingExecutor, EmbeddingService

PROMPTS = [
    "Implement a binary search tree in Python.",
    "Solve this complex mathematical proof.",
    "Explain how HTTP works.",
    "Debug this Python code.",
    "Analyze the logical implications of this statement.",
    "What is cloud computing?",
    "Explain the time complexity of Dijkstra's algorithm.",
    "Explain how neural networks make predictions.",
]


class EmbeddingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = EmbeddingService()

    def test_shape_and_normalization(self) -> None:
        with torch.inference_mode():
            embeddings = self.service.encode(PROMPTS[:2])

        self.assertEqual(tuple(embeddings.shape), (2, 384))
        norms = embeddings.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_batch_matches_single_encoding(self) -> None:
        with torch.inference_mode():
            batch = self.service.encode(PROMPTS[:4])
            singles = torch.cat(
                [self.service.encode([prompt]) for prompt in PROMPTS[:4]],
                dim=0,
            )

        self.assertTrue(
            torch.allclose(batch, singles, atol=1e-5, rtol=1e-5),
            f"max abs diff={(batch - singles).abs().max().item():.3e}",
        )

    def test_concurrent_encoding_matches_serial(self) -> None:
        with torch.inference_mode():
            serial = self.service.encode(PROMPTS)
            with ThreadPoolExecutor(max_workers=8) as executor:
                concurrent = torch.cat(
                    list(executor.map(lambda prompt: self.service.encode([prompt]), PROMPTS)),
                    dim=0,
                )

        self.assertTrue(
            torch.allclose(serial, concurrent, atol=1e-5, rtol=1e-5),
            f"max abs diff={(serial - concurrent).abs().max().item():.3e}",
        )


class EmbeddingExecutorTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_pool_runs_off_event_loop_thread(self) -> None:
        executor = EmbeddingExecutor(max_workers=2)
        try:
            loop_thread = threading.get_ident()

            def work() -> int:
                time.sleep(0.05)
                return threading.get_ident()

            thread_ids = await asyncio.gather(
                executor.run(work),
                executor.run(work),
            )

            self.assertEqual(executor.max_workers, 2)
            self.assertTrue(all(thread_id != loop_thread for thread_id in thread_ids))
            self.assertEqual(len(set(thread_ids)), 2)
            self.assertEqual(len(thread_ids), 2)
            self.assertEqual(len(executor.active_thread_names()), 2)
        finally:
            executor.shutdown(wait=True)

    async def test_cancellation_does_not_leak_threads(self) -> None:
        executor = EmbeddingExecutor(max_workers=1)
        task = asyncio.create_task(executor.run(time.sleep, 0.2))
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        executor.shutdown(wait=True)
        self.assertEqual(executor.active_thread_names(), [])
        self.assertTrue(executor.is_shutdown)

        with self.assertRaises(RuntimeError):
            await executor.run(lambda: None)

    async def test_shutdown_is_idempotent(self) -> None:
        executor = EmbeddingExecutor(max_workers=1)
        await executor.run(lambda: None)
        executor.shutdown(wait=True)
        executor.shutdown(wait=True)
        self.assertTrue(executor.is_shutdown)
        self.assertEqual(executor.active_thread_names(), [])


if __name__ == "__main__":
    unittest.main()
