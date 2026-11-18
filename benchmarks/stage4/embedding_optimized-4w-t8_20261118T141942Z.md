# Stage 4 gateway embedding/event-loop benchmark

- Label: `optimized-4w-t8`
- Timestamp (UTC): `20261118T141942Z`
- Executor present: `True`
- Executor workers: `4`
- Python: `3.12.3`
- Torch threads: 8
- CPU count: 16

## Direct embedding latency

| Measurement | p50 ms | p95 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| `EmbeddingService.encode` (1 prompt) | 8.123 | 13.113 | 23.6695 | 100 |

## Per-repetition gateway results

| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100 | 1.910439 | 52.344 | 16.1013 | 41.4352 | 64.8867 | 2.6661 | 4.424 | 1.3639 | 7.3844 | 0 |
| 1 | 2 | 100 | 2.64984 | 37.7381 | 50.1385 | 75.6168 | 84.238 | 2.7224 | 6.4399 | 1.2816 | 2.459 | 0 |
| 1 | 4 | 100 | 2.377817 | 42.0554 | 93.3564 | 108.9825 | 121.0243 | 3.0465 | 3.5844 | 1.2364 | 2.9695 | 0 |
| 1 | 8 | 100 | 2.571306 | 38.8907 | 201.6487 | 230.6395 | 245.3584 | 4.5443 | 6.9532 | 1.2723 | 2.8645 | 0 |
| 2 | 1 | 100 | 3.980013 | 25.1255 | 37.9856 | 49.4897 | 59.8562 | 2.5237 | 3.2637 | 1.1591 | 1.5677 | 0 |
| 2 | 2 | 100 | 2.738832 | 36.5119 | 52.9625 | 69.8631 | 75.7602 | 2.6151 | 3.0855 | 1.1906 | 2.4557 | 0 |
| 2 | 4 | 100 | 2.441867 | 40.9523 | 95.6219 | 115.1812 | 124.281 | 3.3797 | 4.2435 | 1.2925 | 2.8945 | 0 |
| 2 | 8 | 100 | 2.694453 | 37.1133 | 209.544 | 250.4837 | 258.577 | 4.1483 | 5.2989 | 1.4573 | 2.7987 | 0 |
| 3 | 1 | 100 | 3.994424 | 25.0349 | 38.4664 | 52.6697 | 61.8972 | 2.3377 | 3.0058 | 1.189 | 1.5784 | 0 |
| 3 | 2 | 100 | 2.380295 | 42.0116 | 46.37 | 57.0733 | 73.0528 | 2.1656 | 2.8865 | 1.1769 | 2.5471 | 0 |
| 3 | 4 | 100 | 2.348155 | 42.5866 | 90.7075 | 111.3145 | 115.8603 | 3.1498 | 4.6846 | 1.2529 | 3.0585 | 0 |
| 3 | 8 | 100 | 2.67658 | 37.3611 | 205.5968 | 262.3641 | 280.4756 | 3.7847 | 6.2524 | 1.4274 | 2.899 | 0 |

## Aggregated by concurrency (all repetitions)

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 34.1681 | 36.5955 | 49.7674 | 64.8443 | 2.4996 | 3.7242 | 1.2088 | 3.5102 | 0 |
| 2 | 300 | 38.7539 | 49.6842 | 69.8632 | 80.9909 | 2.6155 | 3.6856 | 1.2152 | 2.4873 | 0 |
| 4 | 300 | 41.8648 | 93.3044 | 113.1418 | 123.4797 | 3.1803 | 4.2267 | 1.2496 | 2.9742 | 0 |
| 8 | 300 | 37.7884 | 206.7966 | 250.8428 | 272.6625 | 4.2424 | 6.4485 | 1.4287 | 2.8541 | 0 |

## Method

- Warm-up: 20 gateway requests before timing.
- Concurrency levels: [1, 2, 4, 8] with 100 requests each.
- Repetitions: 3.
- Event-loop lag is measured with a 10 ms heartbeat; lag is the scheduling delay beyond the requested sleep.
- Health probe latency is a concurrent `GET /health` every 10 ms.
- The upstream is a deterministic local HTTP stub, so GPU/network time is excluded and gateway/embedding cost dominates.
