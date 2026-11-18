# Stage 4 gateway embedding/event-loop benchmark

- Label: `baseline-sync`
- Timestamp (UTC): `20261118T141008Z`
- Executor present: `False`
- Executor workers: `None`
- Python: `3.12.3`
- Torch threads: 8
- CPU count: 16

## Direct embedding latency

| Measurement | p50 ms | p95 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| `EmbeddingService.encode` (1 prompt) | 7.5137 | 13.2908 | 18.0972 | 100 |

## Per-repetition gateway results

| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100 | 8.370789 | 11.9463 | 75.0255 | 128.1456 | 136.066 | 118.2265 | 128.528 | 6.7082 | 5.8864 | 0 |
| 1 | 2 | 100 | 7.728074 | 12.9398 | 144.7887 | 235.3355 | 289.2929 | 179.9042 | 220.7606 | 6.4377 | 5.9024 | 0 |
| 1 | 4 | 100 | 7.685559 | 13.0114 | 283.1488 | 474.9965 | 503.5115 | 249.7293 | 311.8865 | 6.8205 | 5.9426 | 0 |
| 1 | 8 | 100 | 7.957497 | 12.5668 | 605.6329 | 857.4064 | 872.8599 | 651.3112 | 707.4375 | 1.5586 | 6.0666 | 0 |
| 2 | 1 | 100 | 9.044265 | 11.0567 | 89.1392 | 139.1712 | 165.9988 | 124.899 | 149.0859 | 8.289 | 5.8701 | 0 |
| 2 | 2 | 100 | 9.470195 | 10.5594 | 190.7405 | 257.222 | 308.2336 | 182.0541 | 222.728 | 8.7926 | 5.9064 | 0 |
| 2 | 4 | 100 | 8.516482 | 11.7419 | 329.8428 | 459.5329 | 488.0109 | 340.1977 | 380.8242 | 8.8123 | 5.9004 | 0 |
| 2 | 8 | 100 | 7.014393 | 14.2564 | 550.9823 | 722.0091 | 759.7059 | 535.6897 | 604.8543 | 1.3696 | 6.0877 | 0 |
| 3 | 1 | 100 | 8.023252 | 12.4638 | 70.5691 | 127.1694 | 135.6492 | 113.0866 | 123.2191 | 8.2099 | 5.9009 | 0 |
| 3 | 2 | 100 | 8.272543 | 12.0882 | 160.404 | 236.6428 | 270.1157 | 161.9656 | 214.0204 | 8.6745 | 6.0339 | 0 |
| 3 | 4 | 100 | 7.704852 | 12.9788 | 293.9253 | 417.181 | 436.6425 | 258.6113 | 281.7319 | 8.4454 | 6.0102 | 0 |
| 3 | 8 | 100 | 7.796269 | 12.8266 | 600.3511 | 791.1429 | 806.9413 | 546.6902 | 590.5317 | 1.1 | 6.0112 | 0 |

## Aggregated by concurrency (all repetitions)

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 11.8223 | 78.7364 | 129.9765 | 147.3101 | 118.4773 | 138.8053 | 8.0252 | 5.8858 | 0 |
| 2 | 300 | 11.8625 | 164.431 | 246.6672 | 302.5486 | 181.2019 | 224.4337 | 7.9123 | 5.9476 | 0 |
| 4 | 300 | 12.5774 | 303.9872 | 455.9727 | 497.6325 | 303.6423 | 344.065 | 8.5834 | 5.9511 | 0 |
| 8 | 300 | 13.2166 | 588.3384 | 795.7503 | 869.7072 | 607.8657 | 662.0753 | 1.461 | 6.0552 | 0 |

## Method

- Warm-up: 20 gateway requests before timing.
- Concurrency levels: [1, 2, 4, 8] with 100 requests each.
- Repetitions: 3.
- Event-loop lag is measured with a 10 ms heartbeat; lag is the scheduling delay beyond the requested sleep.
- Health probe latency is a concurrent `GET /health` every 10 ms.
- The upstream is a deterministic local HTTP stub, so GPU/network time is excluded and gateway/embedding cost dominates.
