# Worker 1 Stage 3: max-num-seqs 1 versus 2

Generated from saved raw measurement files. No values are inferred.

## Inputs

- `concurrent_kv512mib-seqs1-conc2_20261117T185459Z.json`
- `concurrent_kv512mib-seqs2-conc2_20261117T190633Z.json`

## Protocol

- Two simultaneous requests per batch, followed by a paired batch to the other path.
- Batch order is randomized independently for every pair.
- Warm mode repeats one prompt; cold mode uses a unique prompt per request.
- Prefix caching remains enabled and is not reconfigured.
- Workloads: A short 32 tokens, B standard 150 tokens, C long 800 prompt + 100 completion.

### Warm mode

| KV/seqs | Workload | OK direct/gw | Batch wall p50 ms direct/gw | Batch tok/s p50 direct/gw | Aggregate tok/s direct/gw | Waiting p95 | KV p95 | Spec accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/1 | A | 20/20 / 20/20 | 1712.39 / 1816.67 | 37.37 / 35.23 | 36.31 / 34.64 | 1.00 | 2.76% | 41.18% |
| 512/1 | B | 20/20 / 20/20 | 7635.44 / 7702.87 | 39.29 / 38.95 | 39.28 / 38.81 | 1.00 | 8.28% | 52.03% |
| 512/1 | C | 20/20 / 20/20 | 4623.04 / 4789.36 | 43.26 / 41.76 | 43.32 / 41.67 | 1.00 | 38.62% | 64.77% |
| 512/2 | A | 20/20 / 20/20 | 935.59 / 1027.80 | 68.41 / 62.27 | 67.33 / 62.24 | 0.00 | 4.83% | 41.18% |
| 512/2 | B | 20/20 / 20/20 | 4070.05 / 4169.25 | 73.72 / 71.96 | 72.96 / 71.55 | 0.00 | 15.86% | 52.03% |
| 512/2 | C | 20/20 / 20/20 | 2410.42 / 2601.63 | 82.97 / 76.88 | 82.79 / 76.87 | 0.00 | 43.45% | 64.77% |

### Cold mode

| KV/seqs | Workload | OK direct/gw | Batch wall p50 ms direct/gw | Batch tok/s p50 direct/gw | Aggregate tok/s direct/gw | Waiting p95 | KV p95 | Spec accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512/1 | A | 20/20 / 20/20 | 1571.58 / 1607.78 | 40.75 / 39.81 | 41.34 / 39.81 | 1.00 | 3.45% | 75.91% |
| 512/1 | B | 20/20 / 20/20 | 7288.66 / 7462.16 | 41.16 / 40.20 | 40.00 / 40.12 | 1.00 | 8.97% | 65.55% |
| 512/1 | C | 20/20 / 20/20 | 5606.91 / 6291.96 | 35.67 / 31.79 | 35.02 / 33.13 | 1.00 | 40.00% | 58.82% |
| 512/2 | A | 20/20 / 20/20 | 703.03 / 835.11 | 91.03 / 76.65 | 89.54 / 76.93 | 0.00 | 6.55% | 76.44% |
| 512/2 | B | 20/20 / 20/20 | 3523.60 / 3643.65 | 85.14 / 82.34 | 85.00 / 81.46 | 0.00 | 17.17% | 65.61% |
| 512/2 | C | 20/20 / 20/20 | 2966.25 / 3131.71 | 67.43 / 63.87 | 66.57 / 61.33 | 0.00 | 79.31% | 58.08% |

## Startup facts

| KV MiB | seqs | KV tokens | Max concurrency | Model GiB | Load s | Engine init s |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 1168 | 1.14 | 4.33 | 13.836 | 2.110 |
| 384 | 1 | 1744 | 1.7 | 4.33 | 17.061 | 2.370 |
| 512 | 1 | 2336 | 2.28 | 4.33 | 14.547 | 2.120 |
| 512 | 2 | 2336 | 2.28 | 4.33 | 15.720 | 2.260 |

## GPU telemetry

| KV/seqs | Samples | Temp mean/max C | Power mean W | SM clock mean MHz |
|---|---:|---:|---:|---:|
| 512/1 | 1284 | 84.81/90.00 | 51.35 | 2361.64 |
| 512/2 | 671 | 83.44/90.00 | 51.65 | 2362.42 |

## Decision rule

Prefer a setting only if every request succeeds and KV pressure leaves no waiting for the supported concurrency. Then compare aggregate throughput at concurrency two, followed by per-request latency.

