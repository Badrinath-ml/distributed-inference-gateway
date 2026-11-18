
import asyncio
import json
import statistics
import time

import httpx

ENDPOINTS = {
    "direct": "http://localhost:8000/v1/chat/completions",
    "gateway": "http://localhost:8080/v1/chat/completions",
}

REPETITIONS = 5

PAYLOAD = {
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
    "stream": True,
    "chat_template_kwargs": {
        "enable_thinking": False
    },
}


async def run_test(client, url):
    start = time.perf_counter()
    first_content_time = None
    content_times = []
    content_events = 0

    async with client.stream("POST", url, json=PAYLOAD) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
                content = event["choices"][0]["delta"].get("content", "")
            except (ValueError, KeyError, IndexError):
                continue

            if not content:
                continue

            now = time.perf_counter()
            content_events += 1
            content_times.append(now)

            if first_content_time is None:
                first_content_time = now

    end = time.perf_counter()

    if first_content_time is None:
        raise RuntimeError("No generated content received.")

    intervals = [
        (content_times[i] - content_times[i - 1]) * 1000
        for i in range(1, len(content_times))
    ]

    return {
        "ttft_ms": (first_content_time - start) * 1000,
        "total_ms": (end - start) * 1000,
        "events": content_events,
        "interval_p50_ms": (
            statistics.median(intervals) if intervals else 0
        ),
        "interval_mean_ms": (
            statistics.mean(intervals) if intervals else 0
        ),
    }


def print_summary(name, results):
    ttft = [r["ttft_ms"] for r in results]
    total = [r["total_ms"] for r in results]
    p50 = [r["interval_p50_ms"] for r in results]

    print(f"\n{name.upper()} SUMMARY")
    print(f"TTFT median: {statistics.median(ttft):.2f} ms")
    print(f"TTFT range: {min(ttft):.2f} - {max(ttft):.2f} ms")
    print(f"Total median: {statistics.median(total):.2f} ms")
    print(f"Total range: {min(total):.2f} - {max(total):.2f} ms")
    print(f"Inter-chunk p50 median: {statistics.median(p50):.2f} ms")


async def main():
    results = {"direct": [], "gateway": []}

    async with httpx.AsyncClient(timeout=None) as client:
        print("Running warm-up requests...")

        for name, url in ENDPOINTS.items():
            await run_test(client, url)

        print(f"\nRunning {REPETITIONS} paired benchmark rounds...")

        for i in range(REPETITIONS):
            # Alternate order to reduce systematic order bias.
            order = ["direct", "gateway"] if i % 2 == 0 else [
                "gateway", "direct"
            ]

            for name in order:
                result = await run_test(client, ENDPOINTS[name])
                results[name].append(result)

                print(
                    f"Round {i + 1} | {name:7} | "
                    f"TTFT: {result['ttft_ms']:.2f} ms | "
                    f"Total: {result['total_ms']:.2f} ms | "
                    f"Events: {result['events']}"
                )

    print("\n========== FINAL COMPARISON ==========")

    print_summary("Direct vLLM", results["direct"])
    print_summary("FastAPI Gateway", results["gateway"])

    ttft_differences = [
        g["ttft_ms"] - d["ttft_ms"]
        for d, g in zip(results["direct"], results["gateway"])
    ]

    total_differences = [
        g["total_ms"] - d["total_ms"]
        for d, g in zip(results["direct"], results["gateway"])
    ]

    print("\nESTIMATED GATEWAY OVERHEAD (PAIRED)")
    print(
        f"TTFT difference median: "
        f"{statistics.median(ttft_differences):.2f} ms"
    )
    print(
        f"Total difference median: "
        f"{statistics.median(total_differences):.2f} ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
