# Worker 1 embedding and routing microbenchmark

- Timestamp (UTC): `20261117T181023Z`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Torch threads: 8
- Load time: 1.783 s

## Latency

| Measurement | p50 | p95 | p99 | n |
|---|---:|---:|---:|---:|
| Encode (1 prompt) | 3.4639 | 5.1656 | 5.9027 | 50 |
| Encode (4 prompts) | 7.4064 | 8.7149 | 9.2524 | 25 |
| Route | 3.7476 | 5.6185 | 7.0396 | 50 |

## Concurrent routing

| Workers | Calls | Wall s | Calls/s | p50 ms | p95 ms | Categories match | Scores match |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 48 | 0.202929 | 236.5364 | 3.9053 | 5.8133 | True | True |
| 2 | 48 | 0.423245 | 113.4094 | 17.2559 | 19.0607 | True | True |
| 4 | 48 | 0.334904 | 143.3245 | 27.0473 | 31.5348 | True | True |
| 8 | 48 | 0.48285 | 99.4098 | 78.8493 | 86.4281 | True | True |

## Consistency

- Batch vs single max abs diff: `9.97e-08`
- Batch vs single consistent: `True`
