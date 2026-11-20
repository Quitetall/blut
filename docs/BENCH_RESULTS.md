# BLUT Engine Benchmark Ledger

Criterion microbenchmarks in `benches/framework.rs` measure orchestration
overhead, hashing, cache operations, plan construction, broker decisions,
status emission, and metric-store queries.

## Release procedure

Run on a quiet, fixed Linux x86_64 benchmark host using Rust 1.99.0:

```bash
# Intentionally establish or replace release baseline.
bash scripts/run_benchmarks.sh --save

# Restore committed baseline into target/criterion, then compare current tree.
bash scripts/run_benchmarks.sh compare
```

`BLUT_BENCH_BASELINE` selects another baseline name. Script fails closed when
the requested committed baseline is absent. Criterion's generated HTML remains
untracked; only baseline JSON belongs under
`bench-baselines/framework/<benchmark>/<baseline>/`.

Both commands build first, then wait until the host's 1-minute load average is
at or below `BLUT_BENCH_MAX_LOAD` (default `1.0`) before measuring; they exit 3
if it has not fallen there within `BLUT_BENCH_QUIET_WAIT` seconds (default
`900`). `compare` fails only when a benchmark is over the limit in two passes:
a first pass that flags anything is followed, after another wait for quiet, by
a second, and a slowdown that does not repeat is reported and passed.

A baseline is only comparable on the machine that recorded it. Record it and
run `compare` in one session on the same host.

Any confirmed regression requires investigation and commit-message
justification. Never refresh a baseline merely to hide regression.

## Current release state

- Historical `opt0` baseline: 2026-06-18, pre-public internal development.
- `release-0.2` baseline: **not committed**. It is recorded at the release
  commit, on the host that then runs `compare`.
- 2026-11-13 trial, commit `44491b7`, Rust 1.99.0, a rented machine taken in
  full (AMD Ryzen 9 5900XT, 32 threads, 62 GB; load 0.01 to 0.05 before the
  runs). The save recorded all 27 current benchmarks. Two single-pass
  comparisons of the same code followed:

  | Pass | Median change, all benchmarks | Over the 10% limit | Gate |
  |---|---|---|---|
  | 1 | −5.1% to +17.3% | `status emit 1000 StageStep (lossy)` at +17.3% | failed |
  | 2 | −1.7% to +7.6% | none | passed |

  Unchanged code failed one pass in two, so a single pass cannot be the gate.
  That trial is why a slowdown now has to repeat. The two-pass gate itself has
  not yet run on a quiet host.

A 2026-07-12 attempt was rejected as invalid evidence: workstation carried a
high concurrent CPU load, medians varied sharply between consecutive runs, and
`opt0` lacked newer benchmark names. No noisy measurements were committed.

Cookbook/trainer throughput belongs in cookbook-specific ledgers, not this
engine benchmark file.
