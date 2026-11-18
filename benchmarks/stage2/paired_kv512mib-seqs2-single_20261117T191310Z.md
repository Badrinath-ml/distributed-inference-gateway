# Worker 1 Stage 2 paired benchmark

- Label: `kv512mib-seqs2-single`
- Timestamp (UTC): `20261117T191310Z`
- vLLM KV: 512 MiB, max-num-seqs 2
- Randomization seed: `488280396`
- Pairs per workload/mode: 6
- Warmup pairs: 1

Direct and gateway requests are adjacent within each pair. The order is
randomized per pair. Cold mode uses a unique prompt for every request and
for each path; prefix caching remains enabled but cannot reuse a prefix.

## Workload A: short

### Warm mode

| Metric | Direct | Gateway | Paired delta median |
|---|---:|---:|---:|
| Success | 6/6 | 6/6 | — |
| TTFT p50/p95 (ms) | 102.94 / 105.89 | 139.17 / 144.31 | 35.8225 ms |
| Total p50/p95 (ms) | 861.51 / 890.58 | 901.50 / 908.29 | 32.1234 ms |
| Per-request tok/s p50/p95 | 42.16 / 43.05 | 42.11 / 42.93 | 0.075 |
| Aggregate tok/s | 36.991952 | 35.627465 | — |
| Pair order | 3 direct-first | 3 gateway-first | — |
| Spec acceptance | 41.18% | (shared counters) | — |
| Mean acceptance length | 1.823529 | (shared counters) | — |
| KV usage p50/p95 (%) | 2.07 / 2.76 | (shared) | — |

## Workload B: standard

### Warm mode

| Metric | Direct | Gateway | Paired delta median |
|---|---:|---:|---:|
| Success | 6/6 | 6/6 | — |
| TTFT p50/p95 (ms) | 101.78 / 104.85 | 150.25 / 168.71 | 46.8516 ms |
| Total p50/p95 (ms) | 3549.35 / 3704.69 | 3610.89 / 3754.52 | 55.7757 ms |
| Per-request tok/s p50/p95 | 43.53 / 44.30 | 43.37 / 44.65 | 0.0416 |
| Aggregate tok/s | 41.955937 | 41.357978 | — |
| Pair order | 3 direct-first | 3 gateway-first | — |
| Spec acceptance | 52.03% | (shared counters) | — |
| Mean acceptance length | 2.040541 | (shared counters) | — |
| KV usage p50/p95 (%) | 4.83 / 7.59 | (shared) | — |

## Workload C: long

### Warm mode

| Metric | Direct | Gateway | Paired delta median |
|---|---:|---:|---:|
| Success | 6/6 | 6/6 | — |
| TTFT p50/p95 (ms) | 108.72 / 121.57 | 181.48 / 202.25 | 75.7943 ms |
| Total p50/p95 (ms) | 2213.45 / 2253.41 | 2305.22 / 2399.68 | 94.7288 ms |
| Per-request tok/s p50/p95 | 47.43 / 48.06 | 46.77 / 47.84 | -0.9176 |
| Aggregate tok/s | 45.040301 | 43.019315 | — |
| Pair order | 4 direct-first | 2 gateway-first | — |
| Spec acceptance | 64.77% | (shared counters) | — |
| Mean acceptance length | 2.295455 | (shared counters) | — |
| KV usage p50/p95 (%) | 36.55 / 38.62 | (shared) | — |

## GPU telemetry

- Samples: 192
- Temperature C: `{'count': 192, 'min': 52.0, 'mean': 79.4896, 'median': 81.0, 'p95': 86.0, 'p99': 87.0, 'max': 87.0, 'p99_reliable': True}`
- Utilization %: `{'count': 192, 'min': 0.0, 'mean': 79.8646, 'median': 82.0, 'p95': 86.0, 'p99': 90.9, 'max': 100.0, 'p99_reliable': True}`
- Power W: `{'count': 192, 'min': 2.57, 'mean': 54.4216, 'median': 55.475, 'p95': 59.843, 'p99': 62.9447, 'max': 64.21, 'p99_reliable': True}`
- SM clock MHz: `{'count': 192, 'min': 210.0, 'mean': 2444.1406, 'median': 2460.0, 'p95': 2475.0, 'p99': 2476.35, 'max': 2490.0, 'p99_reliable': True}`
- Memory clock MHz: `{'count': 192, 'min': 374.0, 'mean': 7955.0729, 'median': 8000.0, 'p95': 8000.0, 'p99': 8000.0, 'max': 8000.0, 'p99_reliable': True}`
- Memory used MiB: `{'count': 192, 'min': 5344.0, 'mean': 5344.0, 'median': 5344.0, 'p95': 5344.0, 'p99': 5344.0, 'max': 5344.0, 'p99_reliable': True}`

## Notes

- Requests inside a pair run back-to-back; order is randomized per pair.
- Paired deltas are gateway minus direct for adjacent requests, which reduces thermal drift compared with two long separate batches.
- Aggregate throughput is completion tokens divided by summed successful request wall time; it is not a same-instant multi-client throughput measurement.
- Cold mode varies the prompt prefix so prefix caching cannot produce a hit; the server's prefix-cache configuration was not changed.
- GPU telemetry covers warmups and timed requests.

