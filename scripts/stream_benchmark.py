
import asyncio
import json
import statistics
import time

import httpx

URL = "http://localhost:8080/v1/chat/completions"


def percentile(values, p):
    values = sorted(values)
    index = max(0, int(len(values) * p + 0.999999) - 1)
    return values[min(index, len(values) - 1)]


async def benchmark():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Explain speculative decoding in LLM inference."
            }
        ],
        "max_tokens": 150,
        "stream": True,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    async with httpx.AsyncClient(timeout=None) as client:
        start = time.perf_counter()
        first_token_time = None
        content_times = []

        async with client.stream(
            "POST", URL, json=payload
        ) as response:

            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                try:
                    event = json.loads(data)
                    content = (
                        event["choices"][0]["delta"].get("content", "")
                    )
                except (ValueError, KeyError, IndexError):
                    continue

                if not content:
                    continue

                now = time.perf_counter()
                content_times.append(now)

                if first_token_time is None:
                    first_token_time = now

        end = time.perf_counter()

    if first_token_time is None:
        print("No generated content received.")
        return

    
    intervals = [
        (content_times[i] - content_times[i - 1]) * 1000
        for i in range(1, len(content_times))
    ]

    print("\n--- STREAMING BENCHMARK ---")
    print(f"TTFT: {(first_token_time - start) * 1000:.3f} ms")
    print(f"Total time: {(end - start) * 1000:.3f} ms")
    print(f"Content events: {len(content_times)}")

    if intervals:
        print(f"Inter-chunk min: {min(intervals):.6f} ms")
        print(f"Inter-chunk max: {max(intervals):.6f} ms")
        print(f"Inter-chunk mean: {statistics.mean(intervals):.6f} ms")
        print(f"Inter-chunk p50: {percentile(intervals, 0.50):.6f} ms")
        print(f"Inter-chunk p95: {percentile(intervals, 0.95):.6f} ms")

        near_zero = sum(1 for value in intervals if value < 0.1)
        print(f"Intervals below 0.1 ms: {near_zero}/{len(intervals)}")



if __name__ == "__main__":
    asyncio.run(benchmark())
