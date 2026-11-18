# Worker 1 benchmark result

- Label: `baseline-gateway-sync`
- Timestamp (UTC): `20261117T181645Z`
- Base URL: `http://localhost:8080`
- Workload: `single`
- Requests: 30 total, 30 successful, 0 failed (success rate 1.0)

## Latency (successful requests)

| Metric | p50 | p95 | p99 | min | max | n | unit |
|---|---:|---:|---:|---:|---:|---:|---|
| TTFT | 149.955 | 165.802 | 184.185 | 130.521 | 191.107 | 30 | ms |
| Total latency | 3770.884 | 4143.907 | 4264.211 | 3699.520 | 4270.658 | 30 | ms |
| Inter-chunk | 49.340 | 58.713 | 98.028 | 0.028 | 149.397 | 2190 | ms |
| Per-request throughput | 41.362 | 42.211 | 42.288 | 36.491 | 42.291 | 30 | tokens/s |

Aggregate timed throughput: `39.334309` tokens/s
(timed wall `114.421735` s, active batch wall `114.403941` s).

## Speculative decoding

- Counter deltas: `{'drafts': 2220.0, 'draft_tokens': 4440.0, 'accepted_tokens': 2310.0}`
- Acceptance rate: `0.52027`
- Mean acceptance length: `2.040541`
- Accepted tokens per position: `{'0': 1350.0, '1': 960.0}`

## GPU telemetry

- Samples: 229
- Memory used (MiB): `{'count': 229, 'min': 5009.0, 'mean': 5009.0, 'median': 5009.0, 'p95': 5009.0, 'p99': 5009.0, 'max': 5009.0, 'p99_reliable': True}`
- Memory delta (MiB): `0.0`
- GPU util (%): `{'count': 229, 'min': 49.0, 'mean': 75.8515, 'median': 78.0, 'p95': 82.0, 'p99': 83.0, 'max': 83.0, 'p99_reliable': True}`
- Temperature (C): `{'count': 229, 'min': 81.0, 'mean': 85.1266, 'median': 85.0, 'p95': 86.0, 'p99': 86.0, 'max': 87.0, 'p99_reliable': True}`
- Power (W): `{'count': 229, 'min': 35.78, 'mean': 55.4692, 'median': 55.99, 'p95': 57.614, 'p99': 62.6548, 'max': 67.71, 'p99_reliable': True}`

## KV cache

- KV usage (%): `{'count': 226, 'min': 0.0, 'mean': 0.1013, 'median': 0.0972, 'p95': 0.1667, 'p99': 0.1667, 'max': 0.1667, 'p99_reliable': True}`
- Requests running: `{'count': 226, 'min': 0.0, 'mean': 0.9646, 'median': 1.0, 'p95': 1.0, 'p99': 1.0, 'max': 1.0, 'p99_reliable': True}`
- Requests waiting: `{'count': 226, 'min': 0.0, 'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0, 'p99_reliable': True}`

## Failures

- None.

## Measurement notes

- TTFT is the time from request start to the first content-bearing SSE chunk.
- Total latency ends when the stream closes after `[DONE]`.
- Inter-chunk latency is measured between content-bearing SSE events, not raw tokens.
- GPU samples are periodic `nvidia-smi` polls; short bursts may be missed.
- Speculative counters are global deltas for the timed phase and exclude warm-up.
- p99 is marked unreliable below 100 samples in the JSON payload.
