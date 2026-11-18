# Worker 1 Stage 3 concurrent paired benchmark

- Label: `kv512mib-seqs1-conc2`
- Timestamp (UTC): `20261117T185459Z`
- KV: 512 MiB; max-num-seqs: 1
- Concurrency per batch: 2
- Batches per workload/mode: 10
- Seed: `1270830883`

Each batch sends the configured number of simultaneous requests to one path;
the paired batch is sent to the other path immediately afterwards.

## Workload A: short

### Warm mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 1712.39 / 1995.33 | 1816.67 / 2032.93 | 78.439 ms |
| Batch tok/s p50/p95 | 37.37 / 37.99 | 35.23 / 36.24 | -1.6469 |
| Aggregate tok/s | 36.30806 | 34.63697 | — |
| Per-request TTFT p50/p95 (ms) | 531.43 / 1103.75 | 611.73 / 1128.38 | — |
| Per-request total p50/p95 (ms) | 1384.13 / 1883.85 | 1473.97 / 2004.14 | — |
| Running requests p50/p95 | 1.00 / 1.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 1.00 | (shared) | — |
| KV usage p95 (%) | 2.76 | (shared) | — |
| Spec acceptance | 41.18% | (shared) | — |
| Batch order | 7 direct-first | 3 gateway-first | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 1571.58 / 1686.52 | 1607.78 / 1795.25 | 84.8045 ms |
| Batch tok/s p50/p95 | 40.75 / 46.38 | 39.81 / 43.73 | -2.5091 |
| Aggregate tok/s | 41.339995 | 39.807491 | — |
| Per-request TTFT p50/p95 (ms) | 453.47 / 1003.44 | 569.40 / 1052.67 | — |
| Per-request total p50/p95 (ms) | 1147.37 / 1684.75 | 1219.97 / 1749.32 | — |
| Running requests p50/p95 | 1.00 / 1.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 1.00 | (shared) | — |
| KV usage p95 (%) | 3.45 | (shared) | — |
| Spec acceptance | 75.91% | (shared) | — |
| Batch order | 3 direct-first | 7 gateway-first | — |

## Workload B: standard

### Warm mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 7635.44 / 7766.25 | 7702.87 / 7910.38 | 68.0715 ms |
| Batch tok/s p50/p95 | 39.29 / 39.89 | 38.95 / 39.64 | -0.3352 |
| Aggregate tok/s | 39.279775 | 38.80867 | — |
| Per-request TTFT p50/p95 (ms) | 1982.71 / 3976.97 | 2055.06 / 4173.75 | — |
| Per-request total p50/p95 (ms) | 5692.73 / 7723.37 | 5833.45 / 7908.64 | — |
| Running requests p50/p95 | 1.00 / 1.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 1.00 | (shared) | — |
| KV usage p95 (%) | 8.28 | (shared) | — |
| Spec acceptance | 52.03% | (shared) | — |
| Batch order | 2 direct-first | 8 gateway-first | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 7288.66 / 8147.63 | 7462.16 / 7962.40 | 36.082 ms |
| Batch tok/s p50/p95 | 41.16 / 42.45 | 40.20 / 42.61 | -0.2038 |
| Aggregate tok/s | 39.999182 | 40.115733 | — |
| Per-request TTFT p50/p95 (ms) | 1845.40 / 4001.95 | 1927.56 / 4079.03 | — |
| Per-request total p50/p95 (ms) | 5565.55 / 8022.52 | 5438.30 / 7836.40 | — |
| Running requests p50/p95 | 1.00 / 1.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 1.00 | (shared) | — |
| KV usage p95 (%) | 8.97 | (shared) | — |
| Spec acceptance | 65.55% | (shared) | — |
| Batch order | 6 direct-first | 4 gateway-first | — |

## Workload C: long

### Warm mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 4623.04 / 4666.67 | 4789.36 / 4884.43 | 173.125 ms |
| Batch tok/s p50/p95 | 43.26 / 44.00 | 41.76 / 42.14 | -1.5844 |
| Aggregate tok/s | 43.320527 | 41.665703 | — |
| Per-request TTFT p50/p95 (ms) | 1237.51 / 2453.86 | 1424.47 / 2621.10 | — |
| Per-request total p50/p95 (ms) | 3442.72 / 4665.40 | 3656.70 / 4855.13 | — |
| Running requests p50/p95 | 1.00 / 1.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 1.00 | (shared) | — |
| KV usage p95 (%) | 38.62 | (shared) | — |
| Spec acceptance | 64.77% | (shared) | — |
| Batch order | 7 direct-first | 3 gateway-first | — |

### Cold mode

| Metric | Direct | Gateway | Paired delta |
|---|---:|---:|---:|
| Success | 20/20 | 20/20 | — |
| Batch wall p50/p95 (ms) | 5606.91 / 6124.08 | 6291.96 / 6338.55 | 208.1145 ms |
| Batch tok/s p50/p95 | 35.67 / 38.04 | 31.79 / 36.81 | -1.0885 |
| Aggregate tok/s | 35.021248 | 33.131249 | — |
| Per-request TTFT p50/p95 (ms) | 1587.25 / 3385.23 | 1791.50 / 3582.75 | — |
| Per-request total p50/p95 (ms) | 4096.01 / 6114.53 | 4263.18 / 6336.28 | — |
| Running requests p50/p95 | 1.00 / 1.00 | (shared) | — |
| Waiting requests p50/p95 | 0.00 / 1.00 | (shared) | — |
| KV usage p95 (%) | 40.00 | (shared) | — |
| Spec acceptance | 58.82% | (shared) | — |
| Batch order | 8 direct-first | 2 gateway-first | — |

## GPU telemetry

- Samples: 1284
- Temperature C: `{'count': 1284, 'min': 54.0, 'mean': 84.8061, 'median': 86.0, 'p95': 87.0, 'p99': 89.0, 'max': 90.0, 'p99_reliable': True}`
- Utilization %: `{'count': 1284, 'min': 0.0, 'mean': 73.9642, 'median': 77.0, 'p95': 84.0, 'p99': 97.17, 'max': 100.0, 'p99_reliable': True}`
- Power W: `{'count': 1284, 'min': 2.64, 'mean': 51.3458, 'median': 51.64, 'p95': 57.104, 'p99': 59.8512, 'max': 75.64, 'p99_reliable': True}`
- SM clock MHz: `{'count': 1284, 'min': 210.0, 'mean': 2361.6355, 'median': 2385.0, 'p95': 2460.0, 'p99': 2475.0, 'max': 2490.0, 'p99_reliable': True}`
- Memory clock MHz: `{'count': 1284, 'min': 374.0, 'mean': 7992.5031, 'median': 8000.0, 'p95': 8000.0, 'p99': 8000.0, 'max': 8000.0, 'p99_reliable': True}`
- Memory used MiB: `{'count': 1284, 'min': 5272.0, 'mean': 5272.6612, 'median': 5273.0, 'p95': 5273.0, 'p99': 5273.0, 'max': 5273.0, 'p99_reliable': True}`

## Notes

- Each paired unit is a batch of concurrent requests to one path followed by an equally sized batch to the other path; batch order is randomized.
- With max-num-seqs=1, the second simultaneous request is expected to wait.
- Cold mode varies the prompt prefix so prefix caching cannot produce a hit.
- GPU telemetry covers warmups and timed batches.

