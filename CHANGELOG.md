# Changelog

All notable changes to the **blut** engine crate are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`blut` is the domain-agnostic engine (Stage → Plan → Recipe → Registry +
CLI/TUI + resource broker). The user-facing binaries live in separate cookbook
crates (`blut-lamquant` → binary `blut`, `blut-lamu` → binary `blut-lamu`) that
depend on this crate.

## [Unreleased]

### Added

- **Durable resume (BLUT-API Phase D).** A killed training run no longer
  restarts from scratch. The orchestrator owns a crash-gated *policy*
  (`framework::resume`): a stable per-config resume directory keyed on the stage
  cache key, a `state.json` run-state marker with a heartbeat, and a 5-row
  decision (resume an in-process retry or a crashed prior run; refuse a live
  concurrent run; start fresh otherwise). The trainer owns the *mechanics*
  (`durable_resume.py`): atomic, prev-rotated recovery checkpoints at each
  validation embedding the optimizer + RNG state, and a `--resume` that restores
  model + **optimizer** + RNG (a continuous loss curve, no cold-optimizer dip).
  Covers in-process OOM retry, cross-invocation re-run after a crash, and clean
  optimizer resume. A `no_resume` recipe arg forces a fresh start. (Epoch-boundary
  granularity; mid-epoch dataloader-position resume remains out of scope.)

## [0.10.0] — 2026-06-12

First tagged release. The engine has graduated from the early `0.1` prototype
to a never-OOM-the-box, parallel DAG orchestrator with a distributed launch
seam, run lineage, and declarative scheduling.

### Highlights

- **Never-OOM-the-box resource broker (ADR 0046).** Every memory-spending
  stage runs inside a cgroup-capped systemd unit sized from a *scaling*
  footprint model, gated by a live admission probe. A wrong (too-small)
  estimate is caught by the cgroup as a clean unit kill — never the OS OOM
  killer taking the host.
- **Parallel DAG execution.** The `ParallelExecutor` runs independent stages
  concurrently (the long-missing piece), with capacity-aware memory admission
  so concurrency never overcommits the box.

### Added

- **Resource broker / footprint model** — system probe, conservative-high
  scaling RAM estimate (`workers × prefetch + tier + latent + batch + in_ch`),
  and a fail-closed admission gate that refuses or right-sizes a run against
  live free RAM (slice-1).
- **Calibration store** — measured cgroup peaks are recorded and MAX-merged per
  config key so the broker stops over-refusing the conservative estimate
  (slice-2); an OOM records an `OomCorrected` lower bound that the next
  admission **escalates** (self-heal), box-fit clamped so an escalation can
  never push the cgroup past the host.
- **Self-healing OOM retry** — `RetryOn::OutOfMemoryOnly`: an OOM-killed stage
  re-admits at the escalated cap on retry instead of re-sitting on the same
  failing cap.
- **Capacity-aware parallel admission** — the parallel executor reserves each
  stage's footprint before scheduling it concurrently.
- **`in_ch` (input-channel) footprint term** — a 168-channel fullband run is no
  longer billed like a 21-channel L3 run (the under-bill that caused
  admit-then-cgroup-kill on every fullband launch).
- **Distributed launch seam** — `blut … --launcher local|slurm|ray`. A
  backend-agnostic `WrappedCommand` + `LaunchTarget` routes a train stage to a
  cluster (which owns its own memory); `Local` keeps the cgroup-contained
  this-box path byte-identical.
- **LineageDB** — an embedded, rebuildable SQLite index of every run with git +
  hardware provenance captured on completion; `blut lineage trace <hash>` /
  `reindex`.
- **Declarative recipe schedules** — `blut`-managed systemd timers install
  recurring recipe runs (charset-validated unit names).
- **`register_recipe!` macro** + `schema_of` / `compile_erased` helpers for
  registering cookbook recipes with typed args.
- **Runtime DAG mutation** — `KILL-on-NaN` branch pruning: a stage emitting NaN
  cancels its branch + descendants and frees the GPU while sibling branches
  continue (a no-op policy path is byte-identical to static execution).
- **Subprocess liveness watchdog + heartbeat**, per-stage **retry policy** and
  **soft/hard timeouts**.
- **CLI** — `--json` output flags, `blut runs diff`, cache-stats + artifact
  inspection, and a **build-commit stamp** (`blut --version` →
  `0.10.0+<commit>[-dirty]`) with a **stale-binary warning** at startup when the
  source tree has moved past the running binary (the "git pull, forgot
  `cargo install --force`" trap).

### Fixed

- Footprint drivers are now the single cross-crate source of truth, fixing a
  calibration key-parity bug where the CLI (RESOLVE) and the stage (RECORD)
  keyed differently and the calibration never hit.
- The warm fullband-cache precompute stage (`lamquant_warm_fb_cache`, cookbook)
  was the last uncontained stage — it forked a worker pool under a flat
  reservation and could OOM the **box**. It now runs contained, sized from a
  dedicated fork-pool model, box-fit clamped, with its worker count pinned to
  the billed cap; a warm OOM is non-fatal (the contained trainer re-decodes).
- Footprint right-sized to the measured envelope (tier-3 ~35G → ~23G) so the
  broker stops over-refusing; honest per-worker term + conservative worker cap.
- Numerous review-driven hardening fixes across the broker, executor, launcher,
  and lineage paths (see commit history).

### Known limitations

- **No mid-epoch durable resume yet** (BLUT-API Phase D). Stages retry on OOM,
  but a killed run restarts from scratch rather than resuming a checkpoint. This
  does not block normal runs.
- **Warm fork-pool sizing is clamped against total box RAM, not live-free
  RAM.** On a heavily loaded box the warm default may not complete (the run
  falls back to a slower decode-bound path and warns you to lower the worker
  count). Tune it down per run where needed.
- The LamQuant python payload still lives under this crate's `python/`
  transitionally; it moves to `blut-lamquant` in a later migration.

[0.10.0]: https://github.com/Quitetall/blut/releases/tag/v0.10.0
