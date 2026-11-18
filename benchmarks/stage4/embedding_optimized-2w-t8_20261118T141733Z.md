# Stage 4 gateway embedding/event-loop benchmark

- Label: `optimized-2w-t8`
- Timestamp (UTC): `20261118T141733Z`
- Executor present: `True`
- Executor workers: `2`
- Python: `3.12.3`
- Torch threads: 8
- CPU count: 16

## Direct embedding latency

| Measurement | p50 ms | p95 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| `EmbeddingService.encode` (1 prompt) | 9.3039 | 21.6914 | 31.7032 | 100 |

## Per-repetition gateway results

| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100 | 2.158488 | 46.3287 | 18.5495 | 38.3281 | 64.0017 | 2.8497 | 5.3279 | 1.4567 | 7.4089 | 0 |
| 1 | 2 | 100 | 2.603238 | 38.4137 | 49.2979 | 67.5844 | 85.8507 | 2.8223 | 5.1703 | 1.4392 | 2.5238 | 0 |
| 1 | 4 | 100 | 2.483964 | 40.2582 | 96.7397 | 115.0828 | 126.1355 | 2.6141 | 2.9816 | 1.0911 | 2.5149 | 0 |
| 1 | 8 | 100 | 3.016877 | 33.1469 | 240.0235 | 286.5886 | 293.1603 | 3.4149 | 5.6298 | 1.2525 | 2.4873 | 0 |
| 2 | 1 | 100 | 4.623439 | 21.6289 | 41.7945 | 66.7809 | 70.4217 | 2.6717 | 4.1076 | 1.4871 | 1.5106 | 0 |
| 2 | 2 | 100 | 3.277528 | 30.5108 | 65.8749 | 76.5987 | 85.6749 | 2.7834 | 3.8612 | 1.4985 | 2.306 | 0 |
| 2 | 4 | 100 | 3.52883 | 28.338 | 135.493 | 172.0914 | 209.139 | 3.8747 | 5.671 | 1.5332 | 2.2708 | 0 |
| 2 | 8 | 100 | 3.53758 | 28.2679 | 283.4705 | 324.5154 | 329.0272 | 3.5296 | 4.6795 | 1.5901 | 2.263 | 0 |
| 3 | 1 | 100 | 4.191476 | 23.8579 | 39.9266 | 52.3413 | 63.1148 | 2.3363 | 2.6581 | 1.2181 | 1.5647 | 0 |
| 3 | 2 | 100 | 2.628831 | 38.0397 | 51.0319 | 65.7331 | 80.2414 | 2.3749 | 3.1399 | 1.184 | 2.4924 | 0 |
| 3 | 4 | 100 | 2.583707 | 38.7041 | 101.2364 | 118.9101 | 126.4075 | 2.476 | 3.0305 | 1.2436 | 2.4944 | 0 |
| 3 | 8 | 100 | 2.886712 | 34.6415 | 225.1681 | 252.7304 | 262.4285 | 3.0322 | 5.078 | 1.1799 | 2.4402 | 0 |

## Aggregated by concurrency (all repetitions)

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 30.6052 | 38.2045 | 61.4861 | 70.1927 | 2.5934 | 4.1023 | 1.3928 | 3.4947 | 0 |
| 2 | 300 | 35.6547 | 53.4883 | 74.1749 | 85.8237 | 2.6718 | 4.0601 | 1.4358 | 2.4407 | 0 |
| 4 | 300 | 35.7668 | 104.0138 | 155.8079 | 180.7502 | 3.03 | 4.7035 | 1.3639 | 2.4267 | 0 |
| 8 | 300 | 32.0188 | 241.6655 | 312.3381 | 327.9994 | 3.2855 | 5.3779 | 1.3504 | 2.3968 | 0 |

## Method

- Warm-up: 20 gateway requests before timing.
- Concurrency levels: [1, 2, 4, 8] with 100 requests each.
- Repetitions: 3.
- Event-loop lag is measured with a 10 ms heartbeat; lag is the scheduling delay beyond the requested sleep.
- Health probe latency is a concurrent `GET /health` every 10 ms.
- The upstream is a deterministic local HTTP stub, so GPU/network time is excluded and gateway/embedding cost dominates.
