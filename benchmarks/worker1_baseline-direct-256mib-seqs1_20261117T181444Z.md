# Worker 1 benchmark result

- Label: `baseline-direct-256mib-seqs1`
- Timestamp (UTC): `20261117T181444Z`
- Base URL: `http://localhost:8000`
- Workload: `single`
- Requests: 30 total, 30 successful, 0 failed (success rate 1.0)

## Latency (successful requests)

| Metric | p50 | p95 | p99 | min | max | n | unit |
|---|---:|---:|---:|---:|---:|---:|---|
| TTFT | 102.289 | 139.992 | 151.244 | 95.357 | 154.785 | 30 | ms |
| Total latency | 3711.893 | 4179.129 | 4240.689 | 3434.797 | 4248.238 | 30 | ms |
| Inter-chunk | 47.042 | 70.884 | 92.111 | 0.026 | 127.085 | 2190 | ms |
| Per-request throughput | 41.578 | 44.490 | 44.873 | 36.329 | 44.975 | 30 | tokens/s |

Aggregate timed throughput: `40.068654` tokens/s
(timed wall `112.322593` s, active batch wall `112.307242` s).

## Speculative decoding

- Counter deltas: `{'drafts': 2220.0, 'draft_tokens': 4440.0, 'accepted_tokens': 2310.0}`
- Acceptance rate: `0.52027`
- Mean acceptance length: `2.040541`
- Accepted tokens per position: `{'0': 1350.0, '1': 960.0}`

## GPU telemetry

- Samples: 224
- Memory used (MiB): `{'count': 224, 'min': 5009.0, 'mean': 5009.0, 'median': 5009.0, 'p95': 5009.0, 'p99': 5009.0, 'max': 5009.0, 'p99_reliable': True}`
- Memory delta (MiB): `0.0`
- GPU util (%): `{'count': 224, 'min': 55.0, 'mean': 77.5848, 'median': 81.0, 'p95': 86.0, 'p99': 87.0, 'max': 88.0, 'p99_reliable': True}`
- Temperature (C): `{'count': 224, 'min': 62.0, 'mean': 76.3839, 'median': 78.0, 'p95': 84.0, 'p99': 84.0, 'max': 85.0, 'p99_reliable': True}`
- Power (W): `{'count': 224, 'min': 37.16, 'mean': 53.4395, 'median': 53.87, 'p95': 57.9655, 'p99': 58.9577, 'max': 63.03, 'p99_reliable': True}`

## KV cache

- KV usage (%): `{'count': 222, 'min': 0.0, 'mean': 0.1027, 'median': 0.1042, 'p95': 0.1667, 'p99': 0.1667, 'max': 0.1667, 'p99_reliable': True}`
- Requests running: `{'count': 222, 'min': 0.0, 'mean': 0.9775, 'median': 1.0, 'p95': 1.0, 'p99': 1.0, 'max': 1.0, 'p99_reliable': True}`
- Requests waiting: `{'count': 222, 'min': 0.0, 'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0, 'p99_reliable': True}`

## Failures

- None.

## Measurement notes

- TTFT is the time from request start to the first content-bearing SSE chunk.
- Total latency ends when the stream closes after `[DONE]`.
- Inter-chunk latency is measured between content-bearing SSE events, not raw tokens.
- GPU samples are periodic `nvidia-smi` polls; short bursts may be missed.
- Speculative counters are global deltas for the timed phase and exclude warm-up.
- p99 is marked unreliable below 100 samples in the JSON payload.
