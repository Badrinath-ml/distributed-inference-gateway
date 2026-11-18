# Worker 1 Stage 2 paired benchmark

- Label: `smoke`
- Timestamp (UTC): `20261117T182849Z`
- vLLM KV: 256 MiB, max-num-seqs 1
- Randomization seed: `12345`
- Pairs per workload/mode: 1
- Warmup pairs: 1

Direct and gateway requests are adjacent within each pair. The order is
randomized per pair. Cold mode uses a unique prompt for every request and
for each path; prefix caching remains enabled but cannot reuse a prefix.

## Workload A: short

### Warm mode

| Metric | Direct | Gateway | Paired delta median |
|---|---:|---:|---:|
| Success | 1/1 | 1/1 | — |
| TTFT p50/p95 (ms) | 100.91 / 100.91 | 141.89 / 141.89 | 40.9762 ms |
| Total p50/p95 (ms) | 852.74 / 852.74 | 1036.12 / 1036.12 | 183.3808 ms |
| Per-request tok/s p50/p95 | 42.56 / 42.56 | 35.78 / 35.78 | -6.7781 |
| Aggregate tok/s | 37.526295 | 30.884567 | — |
| Pair order | 1 direct-first | 0 gateway-first | — |
| Spec acceptance | 0.411765 | (shared counters) | — |
| Mean acceptance length | 1.823529 | (shared counters) | — |
| KV usage p50/p95 (%) | 0.0347 / 0.0417 | (shared) | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta median |
|---|---:|---:|---:|
| Success | 1/1 | 1/1 | — |
| TTFT p50/p95 (ms) | 202.74 / 202.74 | 151.08 / 151.08 | -51.6551 ms |
| Total p50/p95 (ms) | 849.39 / 849.39 | 725.15 / 725.15 | -124.2427 ms |
| Per-request tok/s p50/p95 | 49.49 / 49.49 | 55.74 / 55.74 | 6.2572 |
| Aggregate tok/s | 37.674122 | 44.129002 | — |
| Pair order | 1 direct-first | 0 gateway-first | — |
| Spec acceptance | 0.78 | (shared counters) | — |
| Mean acceptance length | 2.56 | (shared counters) | — |
| KV usage p50/p95 (%) | 0.0486 / 0.0674 | (shared) | — |

## Workload C: long

### Warm mode

| Metric | Direct | Gateway | Paired delta median |
|---|---:|---:|---:|
| Success | 1/1 | 1/1 | — |
| TTFT p50/p95 (ms) | 98.95 / 98.95 | 185.90 / 185.90 | 86.9509 ms |
| Total p50/p95 (ms) | 2194.55 / 2194.55 | 2207.31 / 2207.31 | 12.7564 ms |
| Per-request tok/s p50/p95 | 47.72 / 47.72 | 49.47 / 49.47 | 1.7515 |
| Aggregate tok/s | 45.56743 | 45.304089 | — |
| Pair order | 0 direct-first | 1 gateway-first | — |
| Spec acceptance | 0.647727 | (shared counters) | — |
| Mean acceptance length | 2.295455 | (shared counters) | — |
| KV usage p50/p95 (%) | 0.7222 / 0.7722 | (shared) | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta median |
|---|---:|---:|---:|
| Success | 1/1 | 1/1 | — |
| TTFT p50/p95 (ms) | 240.48 / 240.48 | 335.86 / 335.86 | 95.3867 ms |
| Total p50/p95 (ms) | 2301.06 / 2301.06 | 2415.56 / 2415.56 | 114.504 ms |
| Per-request tok/s p50/p95 | 48.53 / 48.53 | 48.08 / 48.08 | -0.4461 |
| Aggregate tok/s | 43.458225 | 41.398192 | — |
| Pair order | 1 direct-first | 0 gateway-first | — |
| Spec acceptance | 0.617978 | (shared counters) | — |
| Mean acceptance length | 2.235955 | (shared counters) | — |
| KV usage p50/p95 (%) | 0.7431 / 0.7917 | (shared) | — |

## GPU telemetry

- Samples: 54
- Temperature C: `{'count': 54, 'min': 43.0, 'mean': 61.4444, 'median': 62.5, 'p95': 69.0, 'p99': 69.47, 'max': 70.0, 'p99_reliable': False}`
- Utilization %: `{'count': 54, 'min': 0.0, 'mean': 76.3704, 'median': 84.0, 'p95': 91.0, 'p99': 95.23, 'max': 100.0, 'p99_reliable': False}`
- Power W: `{'count': 54, 'min': 2.45, 'mean': 47.9417, 'median': 51.975, 'p95': 56.045, 'p99': 58.5557, 'max': 59.25, 'p99_reliable': False}`
- SM clock MHz: `{'count': 54, 'min': 210.0, 'mean': 2420.8333, 'median': 2475.0, 'p95': 2490.0, 'p99': 2490.0, 'max': 2490.0, 'p99_reliable': False}`
- Memory clock MHz: `{'count': 54, 'min': 374.0, 'mean': 7840.2593, 'median': 8000.0, 'p95': 8000.0, 'p99': 8000.0, 'max': 8000.0, 'p99_reliable': False}`
- Memory used MiB: `{'count': 54, 'min': 5060.0, 'mean': 5060.0, 'median': 5060.0, 'p95': 5060.0, 'p99': 5060.0, 'max': 5060.0, 'p99_reliable': False}`

## Notes

- Requests inside a pair run back-to-back; order is randomized per pair.
- Paired deltas are gateway minus direct for adjacent requests, which reduces thermal drift compared with two long separate batches.
- Aggregate throughput is completion tokens divided by summed successful request wall time; it is not a same-instant multi-client throughput measurement.
- Cold mode varies the prompt prefix so prefix caching cannot produce a hit; the server's prefix-cache configuration was not changed.
- GPU telemetry covers warmups and timed requests.

