# Worker 1 benchmark result

- Label: `stage4-e2e-direct`
- Timestamp (UTC): `20261118T142528Z`
- Base URL: `http://localhost:8000`
- Workload: `single`
- Requests: 10 total, 10 successful, 0 failed (success rate 1.0)

## Latency (successful requests)

| Metric | p50 | p95 | p99 | min | max | n | unit |
|---|---:|---:|---:|---:|---:|---:|---|
| TTFT | 267.792 | 336.381 | 360.843 | 227.866 | 366.958 | 10 | ms |
| Total latency | 9319.864 | 9975.482 | 10147.917 | 8988.380 | 10191.026 | 10 | ms |
| Inter-chunk | 120.739 | 176.579 | 238.096 | 0.644 | 292.584 | 730 | ms |
| Per-request throughput | 16.605 | 17.100 | 17.118 | 15.086 | 17.122 | 10 | tokens/s |

Aggregate timed throughput: `15.937259` tokens/s
(timed wall `94.149335` s, active batch wall `94.119068` s).

## Speculative decoding

- Counter deltas: `{'drafts': 740.0, 'draft_tokens': 1480.0, 'accepted_tokens': 770.0}`
- Acceptance rate: `0.52027`
- Mean acceptance length: `2.040541`
- Accepted tokens per position: `{'0': 450.0, '1': 320.0}`

## GPU telemetry

- Samples: 188
- Memory used (MiB): `{'count': 188, 'min': 5326.0, 'mean': 5354.6117, 'median': 5353.0, 'p95': 5374.0, 'p99': 5374.0, 'max': 5374.0, 'p99_reliable': True}`
- Memory delta (MiB): `23.0`
- GPU util (%): `{'count': 188, 'min': 18.0, 'mean': 41.6915, 'median': 43.0, 'p95': 49.0, 'p99': 52.13, 'max': 53.0, 'p99_reliable': True}`
- Temperature (C): `{'count': 188, 'min': 54.0, 'mean': 60.2181, 'median': 61.0, 'p95': 63.0, 'p99': 63.0, 'max': 63.0, 'p99_reliable': True}`
- Power (W): `{'count': 188, 'min': 26.11, 'mean': 30.368, 'median': 30.5, 'p95': 31.506, 'p99': 31.7069, 'max': 31.91, 'p99_reliable': True}`

## KV cache

- KV usage (%): `{'count': 183, 'min': 0.0, 'mean': 0.05, 'median': 0.0483, 'p95': 0.0828, 'p99': 0.0828, 'max': 0.0828, 'p99_reliable': True}`
- Requests running: `{'count': 183, 'min': 0.0, 'mean': 0.9617, 'median': 1.0, 'p95': 1.0, 'p99': 1.0, 'max': 1.0, 'p99_reliable': True}`
- Requests waiting: `{'count': 183, 'min': 0.0, 'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0, 'p99_reliable': True}`

## Failures

- None.

## Measurement notes

- TTFT is the time from request start to the first content-bearing SSE chunk.
- Total latency ends when the stream closes after `[DONE]`.
- Inter-chunk latency is measured between content-bearing SSE events, not raw tokens.
- GPU samples are periodic `nvidia-smi` polls; short bursts may be missed.
- Speculative counters are global deltas for the timed phase and exclude warm-up.
- p99 is marked unreliable below 100 samples in the JSON payload.
