# Multi-worker routing benchmark

- Label: `three-mock-workers`
- Timestamp (UTC): `20261118T142056Z`
- Workers: 3 mock workers
- Simulated worker latency: 25.0 ms
- Worker max concurrency: 4
- Semantic router: embedding-free fake (routing overhead only)

## Per-repetition results

| Rep | Conc | Requests | req/s | p50 ms | p95 ms | p99 ms | Distribution | Errors |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 1 | 100 | 33.1252 | 30.1762 | 31.037 | 31.9991 | `{'mock-1': 33, 'mock-2': 33, 'mock-3': 34}` | 0 |
| 1 | 2 | 100 | 62.9286 | 31.5958 | 33.2749 | 35.5283 | `{'mock-1': 34, 'mock-2': 33, 'mock-3': 33}` | 0 |
| 1 | 4 | 100 | 119.1974 | 33.3371 | 35.5149 | 36.009 | `{'mock-1': 33, 'mock-2': 34, 'mock-3': 33}` | 0 |
| 1 | 8 | 100 | 198.8698 | 38.9662 | 40.8469 | 41.3818 | `{'mock-1': 33, 'mock-2': 33, 'mock-3': 34}` | 0 |
| 2 | 1 | 100 | 33.3234 | 29.9681 | 30.9896 | 31.1912 | `{'mock-1': 34, 'mock-2': 33, 'mock-3': 33}` | 0 |
| 2 | 2 | 100 | 63.2321 | 31.6574 | 32.8409 | 33.0536 | `{'mock-1': 33, 'mock-2': 34, 'mock-3': 33}` | 0 |
| 2 | 4 | 100 | 119.0655 | 33.5308 | 34.6404 | 34.9351 | `{'mock-1': 33, 'mock-2': 33, 'mock-3': 34}` | 0 |
| 2 | 8 | 100 | 193.7021 | 38.7764 | 48.5905 | 49.8362 | `{'mock-1': 34, 'mock-2': 33, 'mock-3': 33}` | 0 |
| 3 | 1 | 100 | 33.2295 | 30.0626 | 30.9219 | 31.177 | `{'mock-1': 33, 'mock-2': 34, 'mock-3': 33}` | 0 |
| 3 | 2 | 100 | 62.756 | 31.7754 | 33.0363 | 33.3902 | `{'mock-1': 33, 'mock-2': 33, 'mock-3': 34}` | 0 |
| 3 | 4 | 100 | 118.2426 | 33.5577 | 35.2242 | 40.0193 | `{'mock-1': 34, 'mock-2': 33, 'mock-3': 33}` | 0 |
| 3 | 8 | 100 | 199.8159 | 38.304 | 40.5613 | 42.6636 | `{'mock-1': 33, 'mock-2': 34, 'mock-3': 33}` | 0 |

## Aggregated by concurrency

| Conc | Requests | req/s mean | p50 ms | p95 ms | p99 ms | Distribution | Errors |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 300 | 33.226 | 30.0626 | 31.0371 | 31.9157 | `{'mock-1': 100, 'mock-2': 100, 'mock-3': 100}` | 0 |
| 2 | 300 | 62.9722 | 31.6825 | 33.0286 | 34.2111 | `{'mock-1': 100, 'mock-2': 100, 'mock-3': 100}` | 0 |
| 4 | 300 | 118.8352 | 33.4929 | 35.1786 | 39.4711 | `{'mock-1': 100, 'mock-2': 100, 'mock-3': 100}` | 0 |
| 8 | 300 | 197.4626 | 38.5607 | 45.6604 | 49.4628 | `{'mock-1': 100, 'mock-2': 100, 'mock-3': 100}` | 0 |

## Health and availability scenarios

| Scenario | Requests | Success | Distribution | Status counts |
|---|---:|---:|---|---|
| one_worker_unhealthy | 40 | 40 | `{'mock-2': 20, 'mock-3': 20}` | `{200: 40}` |
| worker_recovered | 40 | 40 | `{'mock-1': 14, 'mock-2': 13, 'mock-3': 13}` | `{200: 40}` |
| all_workers_unavailable | 5 | 0 | `{}` | `{503: 5}` |

## Notes

- Mock workers require no GPU, API key, network or model download.
- The embedding-free router isolates scheduler/registry/dispatch cost.
- Worker failure scenarios are covered by the unit/integration suite; this benchmark reports health-state routing only.
