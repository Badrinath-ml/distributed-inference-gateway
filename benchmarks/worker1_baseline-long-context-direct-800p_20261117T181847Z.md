# Worker 1 benchmark result

- Label: `baseline-long-context-direct-800p`
- Timestamp (UTC): `20261117T181847Z`
- Base URL: `http://localhost:8000`
- Workload: `single`
- Requests: 3 total, 3 successful, 0 failed (success rate 1.0)

## Latency (successful requests)

| Metric | p50 | p95 | p99 | min | max | n | unit |
|---|---:|---:|---:|---:|---:|---:|---|
| TTFT | 107.796 | 121.332 | 122.535 | 105.117 | 122.836 | 3 | ms |
| Total latency | 2246.611 | 2289.562 | 2293.380 | 2208.062 | 2294.335 | 3 | ms |
| Inter-chunk | 48.975 | 55.009 | 96.711 | 0.357 | 132.130 | 129 | ms |
| Per-request throughput | 47.086 | 47.560 | 47.602 | 45.678 | 47.613 | 3 | tokens/s |

Aggregate timed throughput: `44.450971` tokens/s
(timed wall `6.765214` s, active batch wall `6.749009` s).

## Speculative decoding

- Counter deltas: `{'drafts': 132.0, 'draft_tokens': 264.0, 'accepted_tokens': 171.0}`
- Acceptance rate: `0.647727`
- Mean acceptance length: `2.295455`
- Accepted tokens per position: `{'0': 96.0, '1': 75.0}`

## GPU telemetry

- Samples: 14
- Memory used (MiB): `{'count': 14, 'min': 5009.0, 'mean': 5009.0, 'median': 5009.0, 'p95': 5009.0, 'p99': 5009.0, 'max': 5009.0, 'p99_reliable': False}`
- Memory delta (MiB): `0.0`
- GPU util (%): `{'count': 14, 'min': 66.0, 'mean': 81.1429, 'median': 83.0, 'p95': 86.0, 'p99': 86.0, 'max': 86.0, 'p99_reliable': False}`
- Temperature (C): `{'count': 14, 'min': 81.0, 'mean': 83.1429, 'median': 83.0, 'p95': 84.0, 'p99': 84.0, 'max': 84.0, 'p99_reliable': False}`
- Power (W): `{'count': 14, 'min': 26.91, 'mean': 54.9857, 'median': 57.26, 'p95': 59.163, 'p99': 59.6726, 'max': 59.8, 'p99_reliable': False}`

## KV cache

- KV usage (%): `{'count': 14, 'min': 0.0, 'mean': 0.6349, 'median': 0.7361, 'p95': 0.7778, 'p99': 0.7778, 'max': 0.7778, 'p99_reliable': False}`
- Requests running: `{'count': 14, 'min': 0.0, 'mean': 0.8571, 'median': 1.0, 'p95': 1.0, 'p99': 1.0, 'max': 1.0, 'p99_reliable': False}`
- Requests waiting: `{'count': 14, 'min': 0.0, 'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0, 'p99_reliable': False}`

## Failures

- None.

## Measurement notes

- TTFT is the time from request start to the first content-bearing SSE chunk.
- Total latency ends when the stream closes after `[DONE]`.
- Inter-chunk latency is measured between content-bearing SSE events, not raw tokens.
- GPU samples are periodic `nvidia-smi` polls; short bursts may be missed.
- Speculative counters are global deltas for the timed phase and exclude warm-up.
- p99 is marked unreliable below 100 samples in the JSON payload.
