# Worker 1 benchmark result

- Label: `stage4-e2e-gateway`
- Timestamp (UTC): `20261118T142724Z`
- Base URL: `http://localhost:8080`
- Workload: `single`
- Requests: 10 total, 10 successful, 0 failed (success rate 1.0)

## Latency (successful requests)

| Metric | p50 | p95 | p99 | min | max | n | unit |
|---|---:|---:|---:|---:|---:|---:|---|
| TTFT | 391.764 | 461.544 | 468.629 | 333.515 | 470.400 | 10 | ms |
| Total latency | 9252.239 | 9657.806 | 9700.775 | 8798.195 | 9711.518 | 10 | ms |
| Inter-chunk | 116.796 | 165.062 | 228.900 | 0.789 | 314.934 | 730 | ms |
| Per-request throughput | 16.945 | 17.632 | 17.703 | 16.232 | 17.721 | 10 | tokens/s |

Aggregate timed throughput: `16.19069` tokens/s
(timed wall `92.727613` s, active batch wall `92.645836` s).

## Speculative decoding

- Counter deltas: `{'drafts': 740.0, 'draft_tokens': 1480.0, 'accepted_tokens': 770.0}`
- Acceptance rate: `0.52027`
- Mean acceptance length: `2.040541`
- Accepted tokens per position: `{'0': 450.0, '1': 320.0}`

## GPU telemetry

- Samples: 185
- Memory used (MiB): `{'count': 185, 'min': 5345.0, 'mean': 5365.8865, 'median': 5374.0, 'p95': 5376.0, 'p99': 5382.0, 'max': 5382.0, 'p99_reliable': True}`
- Memory delta (MiB): `34.0`
- GPU util (%): `{'count': 185, 'min': 0.0, 'mean': 42.4162, 'median': 45.0, 'p95': 52.8, 'p99': 56.0, 'max': 58.0, 'p99_reliable': True}`
- Temperature (C): `{'count': 185, 'min': 60.0, 'mean': 63.7459, 'median': 64.0, 'p95': 65.0, 'p99': 65.16, 'max': 66.0, 'p99_reliable': True}`
- Power (W): `{'count': 185, 'min': 26.39, 'mean': 30.9673, 'median': 31.28, 'p95': 32.446, 'p99': 33.0264, 'max': 33.11, 'p99_reliable': True}`

## KV cache

- KV usage (%): `{'count': 180, 'min': 0.0, 'mean': 0.0502, 'median': 0.0483, 'p95': 0.0828, 'p99': 0.0828, 'max': 0.0828, 'p99_reliable': True}`
- Requests running: `{'count': 180, 'min': 0.0, 'mean': 0.9667, 'median': 1.0, 'p95': 1.0, 'p99': 1.0, 'max': 1.0, 'p99_reliable': True}`
- Requests waiting: `{'count': 180, 'min': 0.0, 'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0, 'p99_reliable': True}`

## Failures

- None.

## Measurement notes

- TTFT is the time from request start to the first content-bearing SSE chunk.
- Total latency ends when the stream closes after `[DONE]`.
- Inter-chunk latency is measured between content-bearing SSE events, not raw tokens.
- GPU samples are periodic `nvidia-smi` polls; short bursts may be missed.
- Speculative counters are global deltas for the timed phase and exclude warm-up.
- p99 is marked unreliable below 100 samples in the JSON payload.
