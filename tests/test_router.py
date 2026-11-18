from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from app.services.router import SemanticRouter

CASES = [
    ("Implement a binary search tree in Python.", "coding"),
    ("Debug this Python code.", "coding"),
    ("Solve this complex mathematical proof.", "reasoning"),
    ("Analyze the logical implications of this statement.", "reasoning"),
    ("Explain how HTTP works.", "general"),
    ("What is cloud computing?", "general"),
]


class RouterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.router = SemanticRouter()

    def test_known_category_classifications(self) -> None:
        for prompt, expected in CASES:
            with self.subTest(prompt=prompt):
                result = self.router.route(prompt)
                self.assertEqual(result["category"], expected)

    def test_result_structure_and_score_ordering(self) -> None:
        result = self.router.route("Implement a binary search tree in Python.")

        self.assertEqual(
            set(result.keys()),
            {"category", "scores", "top_score", "margin"},
        )
        self.assertEqual(
            set(result["scores"].keys()),
            set(self.router.categories.keys()),
        )
        ranked = sorted(result["scores"].values(), reverse=True)
        self.assertAlmostEqual(result["top_score"], ranked[0], places=6)
        self.assertAlmostEqual(
            result["margin"],
            ranked[0] - ranked[1],
            places=6,
        )
        self.assertIn(result["category"], self.router.categories)

    def test_deterministic_results(self) -> None:
        prompt = "Explain how neural networks make predictions."
        first = self.router.route(prompt)
        second = self.router.route(prompt)

        self.assertEqual(first["category"], second["category"])
        self.assertAlmostEqual(first["top_score"], second["top_score"], places=6)
        self.assertAlmostEqual(first["margin"], second["margin"], places=6)

    def test_concurrent_routing_matches_serial(self) -> None:
        prompts = [prompt for prompt, _ in CASES] * 4
        serial = [self.router.route(prompt) for prompt in prompts]

        with ThreadPoolExecutor(max_workers=8) as executor:
            concurrent = list(executor.map(self.router.route, prompts))

        for expected, actual in zip(serial, concurrent):
            self.assertEqual(actual["category"], expected["category"])
            self.assertAlmostEqual(
                actual["top_score"],
                expected["top_score"],
                places=6,
            )
            self.assertAlmostEqual(
                actual["margin"],
                expected["margin"],
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
