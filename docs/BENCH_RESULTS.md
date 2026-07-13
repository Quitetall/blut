# BLUT Engine Benchmark Ledger

Criterion microbenchmarks in `benches/framework.rs` measure orchestration
overhead, hashing, cache operations, plan construction, broker decisions,
status emission, and metric-store queries.

## Release procedure

Run on a quiet, fixed Linux x86_64 benchmark host using Rust 1.88:

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

Any median regression above 5% requires investigation and commit-message
justification. Never refresh a baseline merely to hide regression.

## Current release state

- Historical `opt0` baseline: 2026-06-18, pre-public internal development.
- `release-0.2` baseline: **not established**.
- Release gate: **blocked until a quiet-host run records every current
  benchmark, including benchmarks added after `opt0`, and a second comparison
  run confirms reproducibility.**

A 2026-07-12 attempt was rejected as invalid evidence: workstation carried a
high concurrent CPU load, medians varied sharply between consecutive runs, and
`opt0` lacked newer benchmark names. No noisy measurements were committed.

Cookbook/trainer throughput belongs in cookbook-specific ledgers, not this
engine benchmark file.
