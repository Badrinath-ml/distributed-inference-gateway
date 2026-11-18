# Stage 4 gateway embedding/event-loop benchmark

- Label: `optimized-4w-t4`
- Timestamp (UTC): `20261118T141857Z`
- Executor present: `True`
- Executor workers: `4`
- Python: `3.12.3`
- Torch threads: 4
- CPU count: 16

## Direct embedding latency

| Measurement | p50 ms | p95 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| `EmbeddingService.encode` (1 prompt) | 9.7903 | 17.7627 | 33.2598 | 100 |

## Per-repetition gateway results

| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100 | 2.19006 | 45.6608 | 20.15 | 33.9861 | 55.9162 | 2.9461 | 5.2627 | 1.6832 | 3.7116 | 0 |
| 1 | 2 | 100 | 1.958443 | 51.061 | 37.1718 | 52.9106 | 81.1742 | 3.5867 | 6.624 | 1.3431 | 8.2267 | 0 |
| 1 | 4 | 100 | 3.06318 | 32.6458 | 116.1886 | 187.2041 | 217.6622 | 8.8113 | 13.6804 | 1.9596 | 11.7097 | 0 |
| 1 | 8 | 100 | 3.084161 | 32.4237 | 240.917 | 342.8097 | 402.1949 | 11.2589 | 20.4441 | 1.6587 | 12.1878 | 0 |
| 2 | 1 | 100 | 2.16771 | 46.1316 | 19.6952 | 33.2895 | 43.2364 | 3.0448 | 5.1642 | 1.4388 | 6.5299 | 0 |
| 2 | 2 | 100 | 1.861376 | 53.7237 | 33.3277 | 61.5625 | 83.4939 | 4.0867 | 6.4927 | 1.3466 | 8.8213 | 0 |
| 2 | 4 | 100 | 2.898818 | 34.4968 | 110.9985 | 172.3162 | 240.9579 | 7.5376 | 12.9996 | 1.8298 | 11.8492 | 0 |
| 2 | 8 | 100 | 3.028017 | 33.0249 | 235.7717 | 309.4377 | 384.8719 | 10.1336 | 14.0568 | 1.8053 | 12.3582 | 0 |
| 3 | 1 | 100 | 1.978508 | 50.5431 | 17.246 | 39.4312 | 52.124 | 3.7013 | 4.5808 | 2.1686 | 7.0939 | 0 |
| 3 | 2 | 100 | 2.046317 | 48.8683 | 36.9983 | 54.7802 | 124.075 | 3.6581 | 6.0429 | 1.9188 | 8.8384 | 0 |
| 3 | 4 | 100 | 3.263022 | 30.6464 | 118.65 | 208.3768 | 283.1342 | 8.0306 | 16.2962 | 1.6141 | 11.9702 | 0 |
| 3 | 8 | 100 | 4.040484 | 24.7495 | 302.7333 | 454.9433 | 488.4969 | 15.0346 | 28.366 | 1.5551 | 12.303 | 0 |

## Aggregated by concurrency (all repetitions)

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 47.4452 | 18.909 | 34.8357 | 52.0107 | 3.2977 | 5.0956 | 1.7838 | 5.7785 | 0 |
| 2 | 300 | 51.2177 | 35.7451 | 57.349 | 99.6662 | 3.8424 | 6.5495 | 1.6533 | 8.6288 | 0 |
| 4 | 300 | 32.5963 | 115.2849 | 191.0575 | 269.4687 | 8.4577 | 14.6882 | 1.8643 | 11.843 | 0 |
| 8 | 300 | 30.066 | 251.2931 | 401.8926 | 471.4244 | 12.6313 | 22.4155 | 1.6538 | 12.283 | 0 |

## Method

- Warm-up: 20 gateway requests before timing.
- Concurrency levels: [1, 2, 4, 8] with 100 requests each.
- Repetitions: 3.
- Event-loop lag is measured with a 10 ms heartbeat; lag is the scheduling delay beyond the requested sleep.
- Health probe latency is a concurrent `GET /health` every 10 ms.
- The upstream is a deterministic local HTTP stub, so GPU/network time is excluded and gateway/embedding cost dominates.
