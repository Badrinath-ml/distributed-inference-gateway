# Worker 1 Stage 2/3 final report: KV cache and max-num-seqs

- Date: 2026-11-18 (UTC)
- Scope: Worker 1 local vLLM (Qwen3-1.7B + Qwen3-0.6B draft, two speculative tokens)
- Status: experiments complete; selected configuration **promoted as the default on 2026-11-18**
- Raw data: `benchmarks/stage2/`, `benchmarks/stage3/`, startup records in `benchmarks/stage2/startup_*.json`

## Recommendation

**Use 512 MiB KV cache with `max-num-seqs=2`.** This is now the default in
`scripts/worker1.env`.

Launch with:

```bash
scripts/start_worker1.sh
```

Effective defaults: `VLLM_KV_CACHE_MEMORY_BYTES=536870912` (512 MiB) and
`VLLM_MAX_NUM_SEQS=2`.

Rollback to the previous baseline (256 MiB / seqs1) without editing the env file:

```bash
VLLM_KV_CACHE_MEMORY_BYTES=268435456 VLLM_MAX_NUM_SEQS=1 scripts/start_worker1.sh
```

or with the safety-checked restart helper used during the experiments:

```bash
.venv/bin/python scripts/stage2_restart_vllm.py --kv-mib 256 --max-num-seqs 1
```

The pre-promotion env file is preserved at
`backups/worker1.env.pre-promotion-20261117T192040Z`.

The gateway on port 8080 was not restarted or modified.

## Constraints preserved throughout

- Speculative decoding stayed enabled: `Qwen/Qwen3-0.6B`, 2 speculative tokens, FP16.
- `max-model-len` 1024, eager mode, `gpu-memory-utilization` 0.80 unchanged.
- Prefix caching was never reconfigured; cold measurements vary the prompt prefix instead.
- Gateway source byte-identical to `backups/pre-worker1-20261117T180528Z.tar.gz`
  (`APP_SOURCE_IDENTICAL_TO_BACKUP`); 15 tests passed in 16.343 s.
- Only the verified vLLM PID was signalled, after checking ports 8000/8080 and
  the PID file. No SIGKILL escalation, no unrelated process handling.
- 256 MiB baseline preserved as the documented rollback configuration.
- `scripts/worker1.env` was only changed during the approved promotion; the
  pre-promotion file is stored in `backups/`.

## Experimental controls

- Random GPU cooldowns before benchmarks (target 50–55 °C) and before restarts;
  cooldown telemetry saved per run.
- GPU temperature, utilization, power, SM clock and memory clock sampled at 0.5 s
  during every benchmark.
- Direct and gateway requests adjacent within a pair, order randomized per pair
  with recorded seeds; warm and cold prompt modes measured separately.
- Workloads identical across comparisons: A short (32 tokens), B standard (150),
  C long (800-token prompt, 100 completion). Temperature 0.0.
- Every saved result file was re-read by `scripts/verify_benchmark_results.py`:
  **948/948 measured requests succeeded, 0 failures**. No OOM, no startup failure,
  no failed restart in any configuration.

## Stage 2: KV cache sweep at max-num-seqs=1

Each configuration: 12 randomized pairs × warm/cold × workloads A/B/C = 144
measured requests, plus warmups. All succeeded.

| KV | KV tokens | Max concurrency | GPU mem peak | Free of 6,141 MiB | Max KV p95 | Warm B total delta | Warm C total delta | Cold C total delta | GPU max | SM clock mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 MiB | 1,168 | 1.14x | 5,050 MiB | ~1,091 MiB | 79.17% | +31.8 ms (0.87%) | +107.2 ms (4.67%) | −60.9 ms (−2.04%) | 90 °C | 2,407 MHz |
| 384 MiB | 1,744 | 1.70x | 5,175 MiB | ~966 MiB | 52.78% | +50.4 ms (1.38%) | +67.2 ms (2.75%) | +22.2 ms (0.82%) | 89 °C | 2,400 MHz |
| 512 MiB | 2,336 | 2.28x | 5,273 MiB | ~868 MiB | 39.31% | +52.4 ms (1.43%) | +123.6 ms (4.81%) | +118.1 ms (4.43%) | 90 °C | 2,381 MHz |

- Paired deltas are gateway minus direct medians; positive total deltas are small
  relative to request times and fluctuated in sign across modes/configs.
- No KV setting showed a consistent latency advantage at `max-num-seqs=1`.
- **Only 512 MiB can hold two full 1,024-token requests** (2,336 tokens vs 2,048
  needed). 256 MiB supports 1.14x and 384 MiB 1.70x of one full request.
- Selection: 512 MiB, based on success rate, KV headroom for concurrency and
  context capacity; ~868 MiB GPU memory remained free at seqs1.

Detailed tables: `benchmarks/stage2_kv_comparison_20261118T0000Z.md`.

## Stage 3: max-num-seqs 1 versus 2 at 512 MiB

Each configuration: 10 randomized batch pairs × warm/cold × A/B/C, concurrency 2,
20 requests per path per group = 240 measured requests. All succeeded in both.

### Warm mode (direct path; gateway in parentheses where useful)

| Workload | seqs | Batch wall p50 | Aggregate tok/s | Waiting p95 | KV p95 | Spec accept |
|---|---:|---:|---:|---:|---:|---:|
| A short | 1 | 1,712 ms | 36.3 | 1.00 | 2.76% | 41.18% |
| A short | 2 | 936 ms | 67.3 | 0.00 | 4.83% | 41.18% |
| B standard | 1 | 7,635 ms | 39.3 | 1.00 | 8.28% | 52.03% |
| B standard | 2 | 4,070 ms | 73.0 | 0.00 | 15.86% | 52.03% |
| C long | 1 | 4,623 ms | 43.3 | 1.00 | 38.62% | 64.77% |
| C long | 2 | 2,410 ms | 82.8 | 0.00 | 43.45% | 64.77% |

### Cold mode (unique prompt every request)

| Workload | seqs | Batch wall p50 | Aggregate tok/s | Waiting p95 | KV p95 | Spec accept |
|---|---:|---:|---:|---:|---:|---:|
| A short | 1 | 1,572 ms | 41.3 | 1.00 | 3.45% | 75.91% |
| A short | 2 | 703 ms | 89.5 | 0.00 | 6.55% | 76.44% |
| B standard | 1 | 7,289 ms | 40.0 | 1.00 | 8.97% | 65.55% |
| B standard | 2 | 3,524 ms | 85.0 | 0.00 | 17.17% | 65.61% |
| C long | 1 | 5,607 ms | 35.0 | 1.00 | 40.00% | 58.82% |
| C long | 2 | 2,966 ms | 66.6 | 0.00 | 79.31% | 58.08% |

Derived gains at seqs2 (direct path, computed from saved values):

- Batch wall: A warm 1.83x faster, B warm 1.88x, C warm 1.92x, C cold 1.89x.
- Aggregate throughput: B warm 39.3 → 73.0 tok/s (1.86x); C warm 43.3 → 82.8
  (1.91x); C cold 35.0 → 66.6 (1.90x).
- Waiting requests disappeared: p95 1.00 → 0.00; running p50 1.00 → 2.00.
- Speculative decoding behavior unchanged: warm acceptance identical (41.18 /
  52.03 / 64.77%), cold within 0.8 percentage points across settings.

### KV and memory headroom at the selected configuration

- KV max reached exactly 80.0% in cold C (p95 79.31%), leaving ~20% KV headroom.
  No request failed or waited, but the sampled metrics contain no eviction
  counter, so absence of eviction is not asserted.
- GPU memory peaked at 5,345 MiB, i.e. ~796 MiB free of 6,141 MiB. The peak is
  within 5 MiB of the post-startup value, so no meaningful growth under load.
- Context capacity: 2,336 KV tokens supports two 1,024-token requests with 288
  tokens to spare. This is the minimum viable capacity for concurrency 2 at
  maximum context length.

Detailed tables: `benchmarks/stage3_seqs_comparison_20261118T0000Z.md`.

## Single-request regression check at 512 MiB / seqs2

Short confirmation run after the concurrency sweep (6 randomized warm pairs ×
A/B/C, 36 measured requests, 36/36 succeeded):

| Workload | Direct total p50 | Gateway total p50 | Paired delta | seqs1 sweep direct total p50 |
|---|---:|---:|---:|---:|
| A short | 861.5 ms | 901.5 ms | +32.1 ms | 842.1 ms |
| B standard | 3,549.4 ms | 3,610.9 ms | +55.8 ms | 3,640.0 ms |
| C long | 2,213.5 ms | 2,305.2 ms | +94.7 ms | 2,386.5 ms |

Direct single-request times differ by +2.3%, −2.5% and −7.2% from the seqs1
sweep, i.e. within cross-run thermal variation. No single-request regression is
detectable, and the gateway delta band is unchanged. GPU peaked at 87 °C in this
shorter run.

## Assessment against the requested criteria (in order)

1. **Request success rate** — 100% in every configuration: 144/144 per KV
   setting, 240/240 per seqs setting, 36/36 regression. No OOM, no startup or
   request failures.
2. **Memory headroom** — 512 MiB/seqs2 leaves ~796 MiB GPU memory and ~20% KV.
   This is smaller than the 256 MiB baseline (~1,091 MiB free) but stable through
   all 480 concurrency requests and the longest 800-token prompts.
3. **Context capacity** — 2,336 KV tokens vs 1,168 at baseline; the only tested
   setting able to admit two maximum-length requests simultaneously (2.28x).
4. **Latency** — single-request latency unchanged within run noise; at
   concurrency 2, seqs2 cuts batch wall time by roughly half (1.83–1.92x).
5. **Aggregate throughput** — seqs2 nearly doubles throughput (B warm 1.86x,
   C warm 1.91x, C cold 1.90x) with zero waiting.

Not viable alternatives: 256 MiB cannot serve two long requests concurrently and
its KV p95 already reached 79.17% at concurrency 1; 384 MiB is insufficient for
two full requests and showed no latency benefit at seqs1.

## Gateway overhead (context for the separately gated Stage 4)

Measured, not changed:

- Single-request TTFT penalty at the selected setting: +35.8 ms (A), +46.9 ms
  (B), +75.8 ms (C) median paired delta.
- Two-request batch wall penalty: +94 ms (A warm) to +225 ms (C cold).
- Aggregate throughput at seqs2 via gateway vs direct: A 62.2 vs 67.3, B 71.6 vs
  73.0, C 76.9 vs 82.8 (warm); C cold 61.3 vs 66.6 tok/s. Relative gap ~2–8%.
- Gateway overhead is additive on top of the vLLM gains above; it was left
  untouched per instruction.

## Thermal caveats

- Every heavy run peaked at 87–90 °C on this RTX 4050 Laptop GPU. Peak
  temperature does not discriminate between configurations; it reflects total
  sustained load.
- SM clock means ranged 2,361–2,444 MHz across runs. The 512 MiB seqs1 sweep ran
  marginally hotter/slower than 256 and 384, which biases against 512 slightly;
  randomized paired ordering limits direct/gateway bias but not run-to-run drift.
- Sustained production use on this hardware should include cooling/power
  headroom; the performance conclusions above are relative and measured.

## Exact commands used for the selected configuration

Safe restart with verification and cooldowns:

```bash
.venv/bin/python scripts/stage2_restart_vllm.py \
  --kv-mib 512 --max-num-seqs 2 --label kv512mib-seqs2 \
  --prestart-max-c 60 --benchmark-max-c 55 --cooldown-timeout 900
```

Selected startup facts (from
`benchmarks/stage2/startup_kv512mib-seqs2_20261117T190504Z.json`):

- 2,336 KV tokens; 2.28x maximum concurrency for 1,024-token requests
- Model 4.33 GiB; load 15.72 s; engine init 2.26 s; healthy after 46.5 s
- GPU 5,340 MiB after startup, 55 °C

## Promotion record (2026-11-18)

Changed files:

- `scripts/worker1.env` — default KV cache changed from 268435456 to 536870912
  bytes; default `max-num-seqs` changed from 1 to 2; outdated "Final selected
  configuration after Stage 5" comment replaced with the Stage 2/3 selection,
  evidence pointer and rollback instructions. All other values unchanged,
  including target `Qwen/Qwen3-1.7B`, draft `Qwen/Qwen3-0.6B`, FP16, max model
  length 1024, two speculative tokens and eager mode.
- `scripts/verify_worker1_config.py` — new read-only verifier.
- `benchmarks/final_stage2_stage3_report_20261118T0000Z.md` — this promotion
  record and updated recommendation/rollback text.
- `backups/worker1.env.pre-promotion-20261117T192040Z` — byte-identical copy of
  the previous env file (SHA-256
  `07a96053d8a0211dbe9bef65235ed403a71ed808a39fe9602706a06ff4811bcc`).

Verification without restarting the healthy vLLM process:

- `scripts/start_worker1.sh --print`, run with all `VLLM_*` variables removed
  from the environment, expands to the promoted values from the env file.
- The live process (PID 38409) was already running with exactly those arguments,
  so the promotion required no restart. Health endpoint returned HTTP 200.
- `scripts/verify_worker1_config.py` reported PASS: 0 mismatches for env
  defaults, launch-script expansion and live process arguments.

## Evidence files

- `benchmarks/stage2_kv_comparison_20261118T0000Z.md` (KV sweep tables)
- `benchmarks/stage3_seqs_comparison_20261118T0000Z.md` (seqs comparison tables)
- `benchmarks/stage2/paired_kv256mib-seqs1_20261117T183043Z.json`
- `benchmarks/stage2/paired_kv384mib-seqs1_20261117T184006Z.json`
- `benchmarks/stage2/paired_kv512mib-seqs1_20261117T184810Z.json`
- `benchmarks/stage2/paired_kv512mib-seqs2-single_20261117T191310Z.json`
- `benchmarks/stage3/concurrent_kv512mib-seqs1-conc2_20261117T185459Z.json`
- `benchmarks/stage3/concurrent_kv512mib-seqs2-conc2_20261117T190633Z.json`
- `benchmarks/stage2/startup_kv*.json` (startup facts; the 256 record is
  reconstructed from the actual baseline log and startup telemetry)
- `benchmarks/logs/worker1_stage2_vllm_kv512mib-seqs2_20261117T190504Z.log`
- `backups/worker1.env.pre-promotion-20261117T192040Z` (previous 256 MiB/seqs1 defaults)
- `scripts/verify_worker1_config.py` → PASS (env, launch expansion, live process)
- Verification: `scripts/verify_benchmark_results.py` → 948/948 ok, 0 failed

## Not done / open items

- `scripts/worker1.env` now defaults to 512 MiB/seqs2; the live process already
  matched, so no restart was needed for the promotion.
- No `max-num-seqs` > 2 tested, and no load beyond 2 concurrent requests.
- Gateway optimization not started (separately gated); gateway source unchanged.
- No long-duration soak test; thermals were recorded, not controlled.
