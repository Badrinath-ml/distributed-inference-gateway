# Worker 1 benchmark result

- Label: `baseline-long-context-gateway-800p`
- Timestamp (UTC): `20261117T181903Z`
- Base URL: `http://localhost:8080`
- Workload: `single`
- Requests: 3 total, 3 successful, 0 failed (success rate 1.0)

## Latency (successful requests)

| Metric | p50 | p95 | p99 | min | max | n | unit |
|---|---:|---:|---:|---:|---:|---:|---|
| TTFT | 184.724 | 188.538 | 188.877 | 179.547 | 188.962 | 3 | ms |
| Total latency | 2287.377 | 2302.908 | 2304.288 | 2244.975 | 2304.633 | 3 | ms |
| Inter-chunk | 48.260 | 52.660 | 94.749 | 0.365 | 98.110 | 129 | ms |
| Per-request throughput | 47.655 | 48.450 | 48.520 | 47.057 | 48.538 | 3 | tokens/s |

Aggregate timed throughput: `43.878991` tokens/s
(timed wall `6.852276` s, active batch wall `6.836985` s).

## Speculative decoding

- Counter deltas: `{'drafts': 132.0, 'draft_tokens': 264.0, 'accepted_tokens': 171.0}`
- Acceptance rate: `0.647727`
- Mean acceptance length: `2.295455`
- Accepted tokens per position: `{'0': 96.0, '1': 75.0}`

## GPU telemetry

- Samples: 14
- Memory used (MiB): `{'count': 14, 'min': 5009.0, 'mean': 5009.0, 'median': 5009.0, 'p95': 5009.0, 'p99': 5009.0, 'max': 5009.0, 'p99_reliable': False}`
- Memory delta (MiB): `0.0`
- GPU util (%): `{'count': 14, 'min': 56.0, 'mean': 80.0714, 'median': 83.0, 'p95': 88.0, 'p99': 88.0, 'max': 88.0, 'p99_reliable': False}`
- Temperature (C): `{'count': 14, 'min': 78.0, 'mean': 81.0, 'median': 81.0, 'p95': 82.35, 'p99': 82.87, 'max': 83.0, 'p99_reliable': False}`
- Power (W): `{'count': 14, 'min': 36.04, 'mean': 55.6743, 'median': 57.28, 'p95': 59.8045, 'p99': 61.9209, 'max': 62.45, 'p99_reliable': False}`

## KV cache

- KV usage (%): `{'count': 14, 'min': 0.0, 'mean': 0.6319, 'median': 0.7222, 'p95': 0.7778, 'p99': 0.7778, 'max': 0.7778, 'p99_reliable': False}`
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
