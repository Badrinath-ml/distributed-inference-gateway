# Stage 4 gateway embedding/event-loop benchmark

- Label: `optimized-1w-t8`
- Timestamp (UTC): `20261118T141548Z`
- Executor present: `True`
- Executor workers: `1`
- Python: `3.12.3`
- Torch threads: 8
- CPU count: 16

## Direct embedding latency

| Measurement | p50 ms | p95 ms | p99 ms | n |
|---|---:|---:|---:|---:|
| `EmbeddingService.encode` (1 prompt) | 8.6804 | 13.493 | 18.6245 | 100 |

## Per-repetition gateway results

| Rep | Conc | Requests | Wall s | req/s | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100 | 1.831953 | 54.5866 | 17.2184 | 24.7146 | 36.6832 | 2.5499 | 3.2093 | 1.2977 | 7.3982 | 0 |
| 1 | 2 | 100 | 1.919118 | 52.1073 | 36.215 | 42.1725 | 118.2656 | 2.9925 | 5.1994 | 1.3812 | 7.4316 | 0 |
| 1 | 4 | 100 | 2.106209 | 47.4787 | 71.7196 | 150.4717 | 200.8282 | 3.3673 | 8.5258 | 1.512 | 7.4419 | 0 |
| 1 | 8 | 100 | 2.22352 | 44.9737 | 159.6082 | 253.3887 | 287.8619 | 4.9689 | 14.6975 | 1.8175 | 7.3771 | 0 |
| 2 | 1 | 100 | 1.674783 | 59.7092 | 16.0886 | 20.8278 | 25.4862 | 2.3106 | 3.0039 | 1.1349 | 7.3988 | 0 |
| 2 | 2 | 100 | 2.1199 | 47.172 | 36.4572 | 53.5476 | 209.0202 | 2.9611 | 5.0341 | 1.1668 | 7.4371 | 0 |
| 2 | 4 | 100 | 2.15338 | 46.4386 | 72.2885 | 111.5712 | 279.1348 | 2.7695 | 4.0157 | 1.179 | 7.4167 | 0 |
| 2 | 8 | 100 | 1.906093 | 52.4633 | 140.4991 | 193.2223 | 235.5722 | 2.9116 | 4.8336 | 1.3209 | 7.436 | 0 |
| 3 | 1 | 100 | 1.875564 | 53.3173 | 17.17 | 26.472 | 52.4214 | 2.9512 | 4.2556 | 1.3914 | 7.3941 | 0 |
| 3 | 2 | 100 | 1.983773 | 50.409 | 35.283 | 69.596 | 108.2249 | 3.3981 | 6.9082 | 1.429 | 7.4293 | 0 |
| 3 | 4 | 100 | 2.199044 | 45.4743 | 77.2347 | 159.7857 | 211.8425 | 2.9128 | 4.0433 | 1.4396 | 7.4378 | 0 |
| 3 | 8 | 100 | 1.968259 | 50.8063 | 146.2626 | 191.9153 | 223.1777 | 3.3234 | 7.1391 | 1.721 | 7.3915 | 0 |

## Aggregated by concurrency (all repetitions)

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Loop lag p95 ms | Loop lag p99 ms | Health p95 ms | CPU cores | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 55.871 | 16.8455 | 24.7022 | 45.5376 | 2.6689 | 3.5761 | 1.2921 | 7.397 | 0 |
| 2 | 300 | 49.8961 | 35.9211 | 57.195 | 118.2656 | 3.1944 | 6.2468 | 1.3361 | 7.4327 | 0 |
| 4 | 300 | 46.4639 | 74.1246 | 152.6693 | 236.3494 | 3.0392 | 6.5311 | 1.4158 | 7.4321 | 0 |
| 8 | 300 | 49.4144 | 146.7649 | 237.6398 | 254.3122 | 3.5116 | 7.515 | 1.6752 | 7.4015 | 0 |

## Method

- Warm-up: 20 gateway requests before timing.
- Concurrency levels: [1, 2, 4, 8] with 100 requests each.
- Repetitions: 3.
- Event-loop lag is measured with a 10 ms heartbeat; lag is the scheduling delay beyond the requested sleep.
- Health probe latency is a concurrent `GET /health` every 10 ms.
- The upstream is a deterministic local HTTP stub, so GPU/network time is excluded and gateway/embedding cost dominates.
