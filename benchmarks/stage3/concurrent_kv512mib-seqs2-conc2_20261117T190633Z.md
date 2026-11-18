# Worker 1 Stage 3 concurrent paired benchmark

- Label: `kv512mib-seqs2-conc2`
- Timestamp (UTC): `20261117T190633Z`
- KV: 512 MiB; max-num-seqs: 2
- Concurrency per batch: 2
- Batches per workload/mode: 10
- Seed: `934767670`

Each batch sends the configured number of simultaneous requests to one path;
the paired batch is sent to the other path immediately afterwards.

## Workload A: short

### Warm mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 935.59 / 1030.25 | 1027.80 / 1065.40 | 94.2285 ms |
| Batch tok/s p50/p95 | 68.41 / 70.65 | 62.27 / 64.97 | -6.5015 |
| Aggregate tok/s | 67.333188 | 62.241794 | — |
| Per-request TTFT p50/p95 (ms) | 137.47 / 167.07 | 214.78 / 265.94 | — |
| Per-request total p50/p95 (ms) | 935.08 / 1052.53 | 1027.38 / 1065.71 | — |
| Running requests p50/p95 | 2.00 / 2.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 0.00 | (shared) | — |
| KV usage p95 (%) | 4.83 | (shared) | — |
| Spec acceptance | 41.18% | (shared) | — |
| Batch order | 5 direct-first | 5 gateway-first | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 703.03 / 777.81 | 835.11 / 891.90 | 109.6105 ms |
| Batch tok/s p50/p95 | 91.03 / 92.36 | 76.65 / 82.15 | -12.3248 |
| Aggregate tok/s | 89.544792 | 76.92782 | — |
| Per-request TTFT p50/p95 (ms) | 142.32 / 180.42 | 251.10 / 325.71 | — |
| Per-request total p50/p95 (ms) | 702.51 / 823.26 | 834.63 / 903.67 | — |
| Running requests p50/p95 | 2.00 / 2.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 0.00 | (shared) | — |
| KV usage p95 (%) | 6.55 | (shared) | — |
| Spec acceptance | 76.44% | (shared) | — |
| Batch order | 7 direct-first | 3 gateway-first | — |

## Workload B: standard

### Warm mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 4070.05 / 4430.54 | 4169.25 / 4448.73 | 129.104 ms |
| Batch tok/s p50/p95 | 73.72 / 77.68 | 71.96 / 76.19 | -2.351 |
| Aggregate tok/s | 72.957411 | 71.54621 | — |
| Per-request TTFT p50/p95 (ms) | 148.67 / 204.67 | 236.69 / 300.33 | — |
| Per-request total p50/p95 (ms) | 4069.40 / 4436.61 | 4167.00 / 4500.62 | — |
| Running requests p50/p95 | 2.00 / 2.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 0.00 | (shared) | — |
| KV usage p95 (%) | 15.86 | (shared) | — |
| Spec acceptance | 52.03% | (shared) | — |
| Batch order | 4 direct-first | 6 gateway-first | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 3523.60 / 3606.81 | 3643.65 / 3898.92 | 133.935 ms |
| Batch tok/s p50/p95 | 85.14 / 86.30 | 82.34 / 84.26 | -3.1898 |
| Aggregate tok/s | 84.999279 | 81.459984 | — |
| Per-request TTFT p50/p95 (ms) | 138.14 / 164.77 | 253.01 / 306.03 | — |
| Per-request total p50/p95 (ms) | 3496.96 / 3567.72 | 3629.16 / 3812.45 | — |
| Running requests p50/p95 | 2.00 / 2.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 0.00 | (shared) | — |
| KV usage p95 (%) | 17.17 | (shared) | — |
| Spec acceptance | 65.61% | (shared) | — |
| Batch order | 6 direct-first | 4 gateway-first | — |

## Workload C: long

### Warm mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 2410.42 / 2487.83 | 2601.63 / 2668.52 | 174.7475 ms |
| Batch tok/s p50/p95 | 82.97 / 84.94 | 76.88 / 78.54 | -5.5255 |
| Aggregate tok/s | 82.789804 | 76.873647 | — |
| Per-request TTFT p50/p95 (ms) | 137.11 / 167.15 | 309.25 / 361.23 | — |
| Per-request total p50/p95 (ms) | 2410.12 / 2498.65 | 2601.22 / 2677.26 | — |
| Running requests p50/p95 | 2.00 / 2.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 0.00 | (shared) | — |
| KV usage p95 (%) | 43.45 | (shared) | — |
| Spec acceptance | 64.77% | (shared) | — |
| Batch order | 7 direct-first | 3 gateway-first | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 2966.25 / 3204.86 | 3131.71 / 3611.80 | 225.418 ms |
| Batch tok/s p50/p95 | 67.43 / 69.13 | 63.87 / 65.37 | -4.979 |
| Aggregate tok/s | 66.571483 | 61.327279 | — |
| Per-request TTFT p50/p95 (ms) | 458.75 / 581.75 | 629.76 / 779.97 | — |
| Per-request total p50/p95 (ms) | 2940.58 / 3228.98 | 3131.26 / 3593.12 | — |
| Running requests p50/p95 | 2.00 / 2.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 0.00 | (shared) | — |
| KV usage p95 (%) | 79.31 | (shared) | — |
| Spec acceptance | 58.08% | (shared) | — |
| Batch order | 4 direct-first | 6 gateway-first | — |

## GPU telemetry

- Samples: 671
- Temperature C: `{'count': 671, 'min': 54.0, 'mean': 83.4441, 'median': 85.0, 'p95': 87.0, 'p99': 88.0, 'max': 90.0, 'p99_reliable': True}`
- Utilization %: `{'count': 671, 'min': 0.0, 'mean': 76.2474, 'median': 79.0, 'p95': 91.0, 'p99': 100.0, 'max': 100.0, 'p99_reliable': True}`
- Power W: `{'count': 671, 'min': 2.76, 'mean': 51.6504, 'median': 52.56, 'p95': 57.445, 'p99': 62.612, 'max': 65.03, 'p99_reliable': True}`
- SM clock MHz: `{'count': 671, 'min': 210.0, 'mean': 2362.4218, 'median': 2445.0, 'p95': 2475.0, 'p99': 2475.0, 'max': 2475.0, 'p99_reliable': True}`
- Memory clock MHz: `{'count': 671, 'min': 374.0, 'mean': 7987.1446, 'median': 8000.0, 'p95': 8000.0, 'p99': 8000.0, 'max': 8000.0, 'p99_reliable': True}`
- Memory used MiB: `{'count': 671, 'min': 5340.0, 'mean': 5344.2936, 'median': 5344.0, 'p95': 5345.0, 'p99': 5345.0, 'max': 5345.0, 'p99_reliable': True}`

## Notes

- Each paired unit is a batch of concurrent requests to one path followed by an equally sized batch to the other path; batch order is randomized.
- With max-num-seqs=1, the second simultaneous request is expected to wait.
- Cold mode varies the prompt prefix so prefix caching cannot produce a hit.
- GPU telemetry covers warmups and timed batches.

