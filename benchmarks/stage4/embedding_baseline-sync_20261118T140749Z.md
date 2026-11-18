# Stage 4 gateway embedding/event-loop benchmark

- Label: `baseline-sync`
- Timestamp (UTC): `20261118T140749Z`
- Executor present: `False`
- Executor workers: `None`
- Python: `3.12.3`
- Torch threads: 8
- CPU count: 16

## Direct embedding latency

| Measurement | p50 ms | p95 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| `EmbeddingService.encode` (1 prompt) | 9.5299 | 19.3497 | 27.7323 | 100 |

## Per-repetition gateway results

| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100 | 6.079197 | 16.4495 | 54.6316 | 92.7564 | 154.413 | 79.9688 | 146.7084 | 6.3343 | 5.6322 | 0 |
| 1 | 2 | 100 | 7.02248 | 14.24 | 134.1737 | 204.8879 | 262.0423 | 133.6675 | 189.1569 | 7.9843 | 6.0354 | 0 |
| 1 | 4 | 100 | 7.187128 | 13.9138 | 283.1024 | 349.8843 | 374.0407 | 236.4902 | 347.1318 | 6.9067 | 6.0772 | 0 |
| 1 | 8 | 100 | 6.68431 | 14.9604 | 513.5668 | 629.6872 | 716.5534 | 523.4669 | 588.137 | 1.2631 | 6.1765 | 0 |
| 2 | 1 | 100 | 7.714058 | 12.9633 | 66.0928 | 136.0593 | 162.3797 | 126.8108 | 151.2528 | 7.7351 | 6.0325 | 0 |
| 2 | 2 | 100 | 7.371083 | 13.5665 | 141.4553 | 224.3314 | 292.5285 | 150.1246 | 168.0318 | 7.2276 | 5.9439 | 0 |
| 2 | 4 | 100 | 7.851539 | 12.7364 | 293.5647 | 462.4466 | 510.7529 | 309.4901 | 336.2055 | 9.3204 | 6.03 | 0 |
| 2 | 8 | 100 | 6.950607 | 14.3872 | 524.532 | 704.2707 | 752.4246 | 523.9577 | 554.7683 | 1.1222 | 6.1662 | 0 |
| 3 | 1 | 100 | 8.91955 | 11.2113 | 90.4313 | 134.6514 | 142.3183 | 123.9234 | 133.2102 | 10.1918 | 6.0678 | 0 |
| 3 | 2 | 100 | 8.841096 | 11.3108 | 169.1481 | 256.908 | 321.5966 | 162.6497 | 232.0523 | 6.5151 | 5.9748 | 0 |
| 3 | 4 | 100 | 7.744403 | 12.9126 | 286.1576 | 437.1313 | 448.9849 | 238.5058 | 306.1183 | 8.3882 | 6.0185 | 0 |
| 3 | 8 | 100 | 6.952286 | 14.3838 | 522.3304 | 696.4316 | 754.2487 | 538.9209 | 552.5805 | 1.3762 | 6.0574 | 0 |

## Aggregated by concurrency (all repetitions)

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 13.5414 | 66.0928 | 90.4313 | 90.4313 | 126.8108 | 126.8108 | 10.1918 | 5.9108 | 0 |
| 2 | 300 | 13.0391 | 141.4553 | 169.1481 | 169.1481 | 162.6497 | 162.6497 | 7.9843 | 5.9847 | 0 |
| 4 | 300 | 13.1876 | 286.1576 | 293.5647 | 293.5647 | 309.4901 | 309.4901 | 9.3204 | 6.0419 | 0 |
| 8 | 300 | 14.5771 | 522.3304 | 524.532 | 524.532 | 538.9209 | 538.9209 | 1.3762 | 6.1334 | 0 |

## Method

- Warm-up: 20 gateway requests before timing.
- Sequential latency: 100 requests per repetition at concurrency 1.
- Concurrency levels: [1, 2, 4, 8] with 100 requests each.
- Repetitions: 3.
- Event-loop lag is measured with a 10 ms heartbeat; lag is the scheduling delay beyond the requested sleep.
- Health probe latency is a concurrent `GET /health` every 10 ms.
- The upstream is a deterministic local HTTP stub, so GPU/network time is excluded and gateway/embedding cost dominates.
