# BLUT engine benchmark results

Criterion microbenchmarks for the BLUT **framework engine** hot paths
(`benches/framework.rs`). The engine is orchestration overhead that sits in
FRONT of hours of actual training — this measurement plane exists to answer one
question with numbers, not guesses:

> **Is any engine hot path slow enough to be worth optimizing, or is the engine
> measurement-confirmed negligible vs the training it wraps?**

Run it with:

```
cargo bench -p blut --bench framework
```

The committed baseline lives in `bench-baselines/framework/<bench>/opt0/`
(criterion's saved-baseline JSON — `estimates.json` carries the median; the
bulky HTML `report/` plots are intentionally NOT committed). To diff a change
against it:

```
cargo bench -p blut --bench framework -- --baseline opt0
```

Refresh the baseline only when a change INTENTIONALLY moves engine perf, with
`-- --save-baseline opt0` (then re-copy `target/criterion/*/opt0/*.json` into
`bench-baselines/framework/`), and explain WHY in the commit message — mirroring
the codec `bench-baselines/codec/` convention (the codec lives in a sibling
repo; this is blut's own engine baseline).

---

## OPT-0 baseline — 2026-06-18

- Commit: `1a0127e` (branch `feat/channel-agnostic-codec`)
- Host: `onyx-maurader-BrianBigPC` (x86_64)
- Toolchain: `cargo bench` release profile, criterion 0.5

Medians (criterion's point estimate). The first block is the pre-existing
hash/cache coverage; the second is the OPT-0 engine-orchestration plane added
here.

### Hash / cache (pre-existing)

| Bench | Median | Class |
|---|---:|---|
| `hash_file 10 MiB` | 4.761 ms | I/O-bound (per-artifact) |
| `hash_file 100 MiB (mmap)` | 43.607 ms | I/O-bound (per-artifact) |
| `hash_dir parallel (50 × 1 MiB)` | 2.566 ms | I/O-bound (per-ckpt) |
| `hash_dir_serial (50 × 1 MiB)` | 24.771 ms | I/O-bound (serial ref) |
| `ContentHash::to_hex` | 15.99 ns | negligible |
| `cache key_for (Value, canonicalizes)` | 524.21 ns | per-stage |
| `cache key_for_canon_bytes (precomputed)` | 148.18 ns | per-stage |
| `ErasedArtifact round trip` | 87.06 ns | per-artifact |
| `cache insert + lookup round trip` | 13.270 µs | per-stage (incl. fs) |

### Engine orchestration (OPT-0, new)

| Bench | Median | Frequency | Class |
|---|---:|---|---|
| `broker Drivers::estimate` | 2.03 ns | per launch | **negligible** |
| `broker decide (pure box-fit)` | 2.24 ns | per launch | **negligible** |
| `broker FootprintStore::resolve (measured hit)` | 76.30 ns | per launch | **negligible** |
| `broker ResourceSnapshot::probe (syscall-bound)` | 38.527 ms | per launch | syscall-bound¹ |
| `plan topo_order (linear-8)` | 90.88 ns | per run | **negligible** |
| `plan topo_order (fan-out/in)` | 34.27 ns | per run | **negligible** |
| `plan graph_structure (linear-8)` | 198.84 ns | per launch | **negligible** |
| `plan from_erased_chain (8 erased)` | 493.34 ns | per `.toml` load | **negligible** |
| `plan from_components (16×linear-8)` | 10.488 µs | per HPO fan-out | negligible |
| `registry new + register + find` | 9.90 ns | per cold-start | **negligible**² |
| `status emit 1000 StageStep (lossy)` | 60.831 µs | per 1000 steps | ~61 ns/step |
| `status emit 1000 StageBegin (lifecycle)` | 138.110 µs | per 1000 events | ~138 ns/event |
| `metric fold_metrics (1000-step status.jsonl)` | 1.316 ms | per `blut compare` | sub-ms-to-ms³ |
| `metric fold_gauges (200-sample status.jsonl)` | 988.564 µs | per `blut compare` | sub-ms³ |
| `metric final_metrics (1000-row DB)` | 6.041 µs | per query | negligible |
| `metric top_runs_by_metric val_r (1000-row DB)` | 16.677 µs | per query | negligible |
| `metric gpu_saturation (200-sample DB)` | 15.763 µs | per query | negligible |

¹ `ResourceSnapshot::probe` reads `/proc/meminfo` and shells out to `nvidia-smi`
— it is SYSCALL/process-spawn-bound, not a CPU micro-benchmark. The 38.5 ms is
dominated by the `nvidia-smi` fork+exec. It is paid **once per launch**, never
per step, so it is negligible against a multi-hour run; do NOT "optimize" it by
caching unless the launch-rate ever spikes.

² Registry build is benched against a SYNTHETIC single-recipe cookbook because
blut is lib-only — the real recipes live in `blut-lamquant`. This measures the
registry's own machinery (a `Vec::push` + `find` iteration) as a lower bound;
the real cold-start cost is the cookbook crate's static recipe slice
(constructed once at process start, also trivial — it is `&'static` data).

³ `fold_metrics`/`fold_gauges` parse a job's `status.jsonl` into queryable rows.
These are the only paths near the millisecond range. They run **once per `blut
compare` / `hpo show` invocation** (a human-driven read), NOT inside the train
loop, so ~1 ms over a 1000-step log is invisible.

---

## Verdict — OPT-1 is a measurement-confirmed no-op

Every **per-step** and **per-run** engine path is sub-µs to low-µs:

- Per training **step** the engine costs ~61 ns (`StageStep` emit) — at, say,
  10 steps/s that is 0.6 µs/s of orchestration over a GPU step measured in tens
  of milliseconds. Six orders of magnitude below the work it wraps.
- Per **run/launch** the engine costs ~10 µs of plan compile + ~78 ns of
  footprint resolve + ~2 ns of admission math. The single non-trivial launch
  cost is `probe` at 38 ms (syscall/`nvidia-smi`), still negligible vs an
  hours-long run and not CPU-improvable.
- The metric-store **reads** (`fold_*` at ~1 ms, SQL at 6–17 µs) are
  human-triggered (`blut compare`), not hot-loop, and already sub-ms.

There is no engine hot path slow enough to justify OPT-1 optimization work.
The engine is **confirmed negligible vs training**. This baseline is the guard:
if a future engine change regresses any of these by >5% the `--baseline opt0`
diff will show it, and a regression must then be justified in the commit
message (a per-step path crossing into the µs range would be the first thing to
re-examine).

---

## OPT-2 #2 — cache-conditioned LMA dataloader overlap (typed path) — 2026-06-18

A different measurement plane from the criterion engine benches above: this is a
**training-throughput** A/B, not an engine microbench. It answers whether raising
decode-overlap depth on the LMA-direct *typed* dataloader (`lma_typed_l3`
ingredient + `LmaTypedL3Dataset`) feeds the **decode-bound** GPU. The baseline
problem: one LMA window decode ≈ 301 ms vs an ≈ 8.3 ms model step, so at low
worker counts the GPU starves waiting on serial CPU decode (measured
`gpu_saturation` ≈ 30 %, `gpu_wasted` ≈ 0.92).

The change (this commit): the typed path's `LMA_NUM_WORKERS` default and the
adapter's `prefetch_factor` default become **cache-conditioned** — 4 when the
on-disk L3/FB decode cache holds content (warm), 2 when cold — mirroring the
SNN path's long-standing `L3_CACHE_DIR`-conditioned default. An explicit
`LMA_NUM_WORKERS` / `LMA_PREFETCH_FACTOR` env still overrides. The assembled
batch is byte-identical (overlap depth only; zero val_R risk).

A/B harness: `--config fast --tier 1 --detail-bands none --batch-size 4
--windows-per-epoch 400 --epochs-warmup 4 --epochs-quant 0` (encoder 435,846 /
decoder 24,213 params) on the `tuh` LMA root + `split_manifest_v11` (67,735 train
stems), caches warm. Every run **contained** under `systemd-run --user --scope
MemoryMax=16G MemorySwapMax=0` (never OOMed the box). GPU util sampled via
`nvidia-smi` every 0.5 s, started after the `Phase 1: WARM` marker (+6 s settle)
to skip the ≈ 3 min cold-start dataset construction. `LMA_NUM_WORKERS` set
explicitly per run (the cache-conditioned default is the production behaviour,
bypassed here for control).

|                            | workers=1 (RUN A) | workers=4 (RUN B) |
|----------------------------|-------------------|-------------------|
| gpu_saturation (mean %)    | 25.8              | 16.0  (*)         |
| gpu_wasted (frac util<50)  | 0.945             | 0.964 (*)         |
| warm-epoch it/s (ep2–4)    | ~7.7 (6–10)       | ~30.9 (27–31)     |
| dashboard bps (mid-epoch)  | 7 – 24            | 87 – 107          |
| warm-epoch walltime (s)    | 10.3 / 16.6 / 13.0| 3.7 / 3.3 / 3.2   |
| peak RSS (GB)              | 10.0              | 16.0              |
| val_r ep1→ep4              | 0.449 → 0.575     | 0.449 → 0.570     |

(*) The external `nvidia-smi` `gpu_saturation` / `gpu_wasted` are **misleading
for this mandated tiny tier-1 probe config**: the model step is < 10 ms, so each
GPU burst is brief, and at workers=4 the warm epochs finish in ≈ 3.3 s — the 70 s
sampler then mostly catches the inter-epoch *validation* idle gaps, not the
per-step loop. The reliable decode-overlap signal is the **per-step rate**
(warm it/s + the trainer's own `bps`), which rose **~4–13×** (7–24 bps → 87–107
bps; warm it/s 7.7 → 30.9; warm-epoch walltime 13 s → 3.3 s).

**Verdict.** Raising decode overlap (workers 1→4, prefetch 2→4) did **not** raise
the nvidia-smi util% on this config — it *can't*, the GPU step is trivially short
here — but it **cut the per-step decode wait ~4×**, confirming the dataloader was
the limiter and more overlap fixes it. So `gpu_wasted` did **not** drop from the
0.92 baseline on this probe; the GPU is *fed* in the sense that decode no longer
gates each step, but the tier-1 model is too small to convert that into sustained
util — that only shows on a real tier-2/3 decoder where the step is long enough
to overlap a full decode queue (→ the next levers are #3 single-decode and a real
warm-precompute stage). RAM cost of workers=4: +6 GB peak (10 → 16 GB),
contained well under the 16 G cap. `val_r` climbs identically (byte-identical
batch — pure overlap change, no quality effect).
