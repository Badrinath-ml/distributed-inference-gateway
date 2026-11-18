# Stage 4 gateway embedding/event-loop benchmark

- Label: `optimized-2w-t4`
- Timestamp (UTC): `20261118T141822Z`
- Executor present: `True`
- Executor workers: `2`
- Python: `3.12.3`
- Torch threads: 4
- CPU count: 16

## Direct embedding latency

| Measurement | p50 ms | p95 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| `EmbeddingService.encode` (1 prompt) | 9.7024 | 11.6295 | 17.3019 | 100 |

## Per-repetition gateway results

| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100 | 1.851977 | 53.9964 | 17.4556 | 24.6569 | 37.6996 | 2.9528 | 5.6282 | 1.3849 | 3.7023 | 0 |
| 1 | 2 | 100 | 1.787145 | 55.9552 | 34.906 | 45.1182 | 59.2353 | 3.2649 | 6.0616 | 1.4034 | 6.7804 | 0 |
| 1 | 4 | 100 | 1.817111 | 55.0324 | 71.3941 | 81.4841 | 104.6705 | 3.4542 | 4.9881 | 1.2127 | 6.7864 | 0 |
| 1 | 8 | 100 | 1.944519 | 51.4266 | 140.7431 | 219.4367 | 251.5158 | 6.0193 | 11.1523 | 1.5451 | 6.7791 | 0 |
| 2 | 1 | 100 | 1.895286 | 52.7625 | 17.6683 | 28.2731 | 37.7219 | 2.811 | 3.3409 | 1.4337 | 6.3358 | 0 |
| 2 | 2 | 100 | 1.938105 | 51.5968 | 34.4514 | 54.3051 | 141.0506 | 3.7934 | 8.175 | 1.261 | 6.811 | 0 |
| 2 | 4 | 100 | 2.09123 | 47.8187 | 79.8839 | 109.8543 | 119.5455 | 4.5272 | 7.8263 | 1.4408 | 6.7991 | 0 |
| 2 | 8 | 100 | 1.986536 | 50.3389 | 142.9 | 222.5381 | 236.1149 | 4.2953 | 16.1752 | 1.3566 | 6.7948 | 0 |
| 3 | 1 | 100 | 1.928274 | 51.8599 | 17.577 | 30.7373 | 51.4402 | 2.9309 | 5.9096 | 1.3183 | 6.2811 | 0 |
| 3 | 2 | 100 | 1.838213 | 54.4007 | 35.8274 | 48.9983 | 63.6284 | 3.3223 | 4.1851 | 1.4138 | 6.7764 | 0 |
| 3 | 4 | 100 | 2.095694 | 47.7169 | 76.1593 | 115.0736 | 212.8221 | 3.9582 | 9.0247 | 1.8516 | 6.7949 | 0 |
| 3 | 8 | 100 | 2.002291 | 49.9428 | 156.7191 | 187.7868 | 191.2766 | 4.962 | 7.7994 | 1.2306 | 6.7873 | 0 |

## Aggregated by concurrency (all repetitions)

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 52.8729 | 17.5369 | 28.5956 | 51.4402 | 2.8804 | 5.4927 | 1.3958 | 5.4397 | 0 |
| 2 | 300 | 53.9842 | 35.1471 | 49.3609 | 79.051 | 3.3348 | 6.9599 | 1.3929 | 6.7893 | 0 |
| 4 | 300 | 50.1893 | 75.8272 | 104.6812 | 141.2119 | 3.8685 | 7.6566 | 1.5166 | 6.7935 | 0 |
| 8 | 300 | 50.5694 | 148.4493 | 214.5408 | 239.8263 | 4.8409 | 12.3313 | 1.334 | 6.7871 | 0 |

## Method

- Warm-up: 20 gateway requests before timing.
- Concurrency levels: [1, 2, 4, 8] with 100 requests each.
- Repetitions: 3.
- Event-loop lag is measured with a 10 ms heartbeat; lag is the scheduling delay beyond the requested sleep.
- Health probe latency is a concurrent `GET /health` every 10 ms.
- The upstream is a deterministic local HTTP stub, so GPU/network time is excluded and gateway/embedding cost dominates.
