# Worker 1 Stage 2 KV cache sweep

Generated from saved raw measurement files. No values are inferred.

## Inputs

- `paired_kv256mib-seqs1_20261117T183043Z.json`
- `paired_kv384mib-seqs1_20261117T184006Z.json`
- `paired_kv512mib-seqs1_20261117T184810Z.json`

## Protocol

- Direct and gateway requests are paired back-to-back; order is randomized per pair.
- Warm mode repeats one prompt; cold mode uses a unique prompt for every request and path.
- Prefix caching stays enabled; cold mode defeats it with unique prefixes instead of changing config.
- Workloads: A short 32 tokens, B standard 150 tokens, C long 800 prompt + 100 completion.
- Speculative decoding: `Qwen/Qwen3-0.6B` draft, two speculative tokens, throughout.

### Warm mode

| KV/seqs | Workload | OK direct/gw | TTFT direct/gw (ms) | Paired TTFT delta ms (p50/p95) | Paired total delta ms | Aggregate tok/s direct/gw | Spec accept | KV p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256/1 | A | 12/12 / 12/12 | 100.12 / 145.18 | 45.76 / 56.18 (45.35%) | 40.95 (4.84%) | 37.17 / 35.41 | 41.18% | 5.56% |
| 256/1 | B | 12/12 / 12/12 | 102.33 / 144.10 | 39.68 / 78.40 (40.05%) | 31.78 (0.87%) | 39.08 / 38.46 | 52.03% | 16.67% |
| 256/1 | C | 12/12 / 12/12 | 110.03 / 186.41 | 76.86 / 101.94 (70.47%) | 107.19 (4.67%) | 43.00 / 41.62 | 64.77% | 77.78% |
| 384/1 | A | 12/12 / 12/12 | 102.98 / 136.90 | 34.08 / 57.97 (34.32%) | 43.07 (5.08%) | 36.90 / 35.56 | 41.18% | 3.70% |
| 384/1 | B | 12/12 / 12/12 | 103.27 / 154.22 | 50.47 / 77.21 (47.82%) | 50.43 (1.38%) | 40.33 / 40.01 | 52.03% | 11.11% |
| 384/1 | C | 12/12 / 12/12 | 110.71 / 191.57 | 78.39 / 112.52 (70.38%) | 67.19 (2.75%) | 40.24 / 39.42 | 64.77% | 51.85% |
| 512/1 | A | 12/12 / 12/12 | 98.30 / 147.40 | 50.63 / 70.08 (51.76%) | 56.92 (6.81%) | 37.88 / 35.29 | 41.18% | 2.76% |
| 512/1 | B | 12/12 / 12/12 | 103.47 / 144.57 | 43.33 / 61.01 (43.89%) | 52.39 (1.43%) | 41.08 / 40.67 | 52.03% | 8.28% |
| 512/1 | C | 12/12 / 12/12 | 111.25 / 198.64 | 83.40 / 111.75 (73.97%) | 123.60 (4.81%) | 41.42 / 39.54 | 64.77% | 38.62% |

### Cold mode

| KV/seqs | Workload | OK direct/gw | TTFT direct/gw (ms) | Paired TTFT delta ms (p50/p95) | Paired total delta ms | Aggregate tok/s direct/gw | Spec accept | KV p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256/1 | A | 12/12 / 12/12 | 101.27 / 151.21 | 52.25 / 79.23 (51.98%) | 89.96 (11.75%) | 43.95 / 40.54 | 73.95% | 6.94% |
| 256/1 | B | 12/12 / 12/12 | 104.68 / 146.53 | 42.92 / 70.01 (40.13%) | -23.71 (-0.71%) | 43.48 / 43.95 | 63.39% | 18.06% |
| 256/1 | C | 12/12 / 12/12 | 271.71 / 348.69 | 72.21 / 96.06 (26.99%) | -60.94 (-2.04%) | 33.67 / 34.32 | 56.34% | 79.17% |
| 384/1 | A | 12/12 / 12/12 | 99.80 / 146.50 | 42.96 / 55.72 (43.24%) | 47.90 (7.24%) | 47.03 / 44.16 | 78.00% | 4.63% |
| 384/1 | B | 12/12 / 12/12 | 109.10 / 156.76 | 46.07 / 66.63 (43.21%) | -17.56 (-0.42%) | 41.11 / 41.54 | 63.19% | 11.53% |
| 384/1 | C | 12/12 / 12/12 | 280.15 / 350.89 | 67.48 / 100.31 (24.71%) | 22.18 (0.82%) | 37.76 / 37.32 | 57.13% | 52.78% |
| 512/1 | A | 12/12 / 12/12 | 99.01 / 148.16 | 49.06 / 85.73 (49.49%) | 53.32 (8.03%) | 46.65 / 43.49 | 78.00% | 3.45% |
| 512/1 | B | 12/12 / 12/12 | 107.00 / 154.75 | 46.73 / 80.82 (44.20%) | 93.02 (2.82%) | 42.94 / 41.82 | 64.20% | 8.72% |
| 512/1 | C | 12/12 / 12/12 | 285.81 / 360.44 | 76.00 / 114.44 (27.86%) | 118.11 (4.43%) | 37.12 / 35.14 | 57.05% | 39.31% |

## Startup facts

| KV MiB | seqs | OK | KV tokens | Max concurrency | Model GiB | Load s | Engine init s |
|---:|---:|---|---:|---:|---:|---:|---:|
| 256 | 1 | True | 1168 | 1.14 | 4.33 | 13.836 | 2.110 |
| 384 | 1 | True | 1744 | 1.7 | 4.33 | 17.061 | 2.370 |
| 512 | 1 | True | 2336 | 2.28 | 4.33 | 14.547 | 2.120 |

## GPU telemetry by configuration

| KV/seqs | Samples | Temp mean/max C | Power mean W | SM clock mean MHz | Memory clock mean MHz |
|---|---:|---:|---:|---:|---:|
| 256/1 | 744 | 82.69/90.00 | 53.22 | 2406.98 | 7988.41 |
| 384/1 | 738 | 82.91/89.00 | 52.93 | 2399.76 | 7988.31 |
| 512/1 | 735 | 83.81/90.00 | 52.69 | 2381.37 | 7988.26 |

## Selection criteria

The final recommendation must weigh, in this order: request success rate, memory headroom, context capacity, latency, then aggregate throughput. A configuration that fails requests or leaves unsafe memory headroom is not viable regardless of latency.

