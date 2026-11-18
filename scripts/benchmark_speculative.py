
import asyncio
import json
import statistics
import time

import httpx


URL = "http://localhost:8000/v1/chat/completions"

WARMUP_RUNS = 2
REPETITIONS = 8

PAYLOAD = {
    "model": "Qwen/Qwen3-1.7B",
    "messages": [
        {
            "role": "user",
            "content": (
                "Explain speculative decoding in LLM inference. "
                "Describe the roles of draft and target models."
            ),
        }
    ],
    "max_tokens": 150,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {
        "enable_thinking": False
    },
}


async def run_test(client):
    start = time.perf_counter()

    first_content_time = None
    content_times = []
    completion_tokens = None

    async with client.stream("POST", URL, json=PAYLOAD) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except ValueError:
                continue

            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")

            choices = event.get("choices", [])
            if not choices:
                continue

            content = choices[0].get("delta", {}).get("content", "")

            if content:
                now = time.perf_counter()

                if first_content_time is None:
                    first_content_time = now

                content_times.append(now)

    end = time.perf_counter()

    if first_content_time is None:
        raise RuntimeError("No generated content received.")

    if completion_tokens is None:
        raise RuntimeError(
            "Completion token usage missing. "
            "Check stream_options support."
        )

    intervals = [
        (content_times[i] - content_times[i - 1]) * 1000
        for i in range(1, len(content_times))
    ]

    generation_time = end - first_content_time

    return {
        "ttft_ms": (first_content_time - start) * 1000,
        "total_ms": (end - start) * 1000,
        "completion_tokens": completion_tokens,
        "tokens_per_second": (
            completion_tokens / generation_time
            if generation_time > 0 else 0
        ),
        "interval_p50_ms": (
            statistics.median(intervals) if intervals else 0
        ),
    }


async def main():
    results = []

    async with httpx.AsyncClient(timeout=None) as client:
        print("Running warm-up requests...")

        for _ in range(WARMUP_RUNS):
            await run_test(client)

        print(f"\nRunning {REPETITIONS} benchmark requests...")

        for i in range(REPETITIONS):
            result = await run_test(client)
            results.append(result)

            print(
                f"Run {i + 1:02} | "
                f"TTFT: {result['ttft_ms']:.2f} ms | "
                f"Total: {result['total_ms']:.2f} ms | "
                f"Tokens: {result['completion_tokens']} | "
                f"Throughput: {result['tokens_per_second']:.2f} tok/s"
            )

    print("\n========== SUMMARY ==========")

    for key, label in [
        ("ttft_ms", "TTFT"),
        ("total_ms", "Total latency"),
        ("tokens_per_second", "Output throughput"),
        ("interval_p50_ms", "Inter-chunk p50"),
    ]:
        values = [r[key] for r in results]
        print(f"{label}: {statistics.median(values):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
