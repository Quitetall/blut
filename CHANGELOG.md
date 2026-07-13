# Changelog

All notable changes to the **blut** engine crate are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`blut` is the domain-agnostic engine (Stage → Plan → Recipe → Registry + CLI +
resource broker). The user-facing binaries live in separate cookbook crates that
depend on this crate.

## [Unreleased]

No changes yet.

## 0.2.0-alpha.1

### Changed
- **Version truth repair.** The package family is reset to
  `0.2.0-alpha.1`. The local 1.x tags are retained as internal milestone
  history; they were not crates.io releases. A real 1.0 remains gated on the
  M6 release train.
- **Public preview boundary.** `blut-types`, `blut`, and `blut-dsl` form the
  publishable preview. Deprecated worker and experimental operator crates are
  explicitly unpublished. Network object-store support is deferred; the cloud
  queue keeps its local-filesystem proof path.
- **MSRV.** Public preview crates now require and test Rust 1.88. The
  unpublished operator requires Rust 1.89.
- **Package hygiene.** Crate contents use a fail-closed allowlist; internal
  agent memory, deployment fixtures, and private host paths are excluded.

### Added
- **Typed dynamic DAGs (ADR 0078).** `blut recipe declare` now accepts a
  `.json` [`PlanSpec`](API.md) (arbitrary kind-checked DAG) and a `.star`
  Starlark script alongside the existing linear `.toml`. `PlanSpec` is the
  engine-native, versioned plan IR — also the door for a future Python SDK.
  - `CompiledPlan::from_erased_graph` extends the before-execution kind check
    from linear chains to arbitrary fork/merge DAGs.
  - **`map_output`** typed runtime fan-out: a stage whose output is a
    `ListOf<E>` fans out one sub-plan (template) per element, executed on the
    parallel executor's completion seam with cache-stable shard keys.
  - The Starlark front-end ships as a SEPARATE binary, `crates/blut-dsl`
    (`blut-dsl <script.star> --args <json>` → `PlanSpec` JSON), which the
    engine shells out to. `starlark` is kept out of the engine binary because
    it forces `serde_json/arbitrary_precision` — incompatible with the
    engine's internally-tagged enums under Cargo feature unification.
- **Runtime governance.** Tenant-namespaced cache and admission, capability
  RBAC, audited `SecretRef` handling, deployment/model registries, built-in
  fail-closed checks, typed connectors, model cards, provenance graphs, and
  run-diff reporting.
- **Scheduling and recovery.** GPU-aware per-device admission, per-node retry
  and timeout overrides, step-granular resume state, partition status cells,
  user-priority DAG scheduling, and opt-in auto-tune proposals.
- **Engine console.** Live run, broker, cache, lineage, mesh, privacy, recipe,
  and plan-builder views plus cookbook TUI extension hooks.
- **Public release gates.** Rustdoc with warnings denied, declared-MSRV builds,
  package-content checks, dependency/advisory/license policy, secret scanning,
  and in-tree public-example coverage.

### Fixed
- P2P certificate pinning, signed-task argument coverage, authenticated receive
  ordering, encrypted blob transfer, dispatch races, and peer-registry
  persistence now fail closed.
- Cache atomic writes preserve `fsync` failures; containment cleans failed
  cgroup setup; lineage, tenant, registry, RBAC, connector, and auto-tune edge
  cases now preserve their safety invariants.
- Public documentation, packaged links, version claims, and rustdoc links match
  the library-only preview surface.

### Security
- Deprecated worker job IDs are validated as flat ASCII identifiers at API and
  queue boundaries; duplicate submissions cannot overwrite files. Its
  plaintext prototype now refuses every non-loopback bind.
- Absolute source paths are no longer embedded by default. Local stale-source
  detection requires explicit `BLUT_EMBED_SRC_DIR=1` opt-in.
- Locked dependencies were updated for active RustSec findings. `bincode 1.3`
  remains temporarily under a documented unmaintained-only exception because
  replacement changes cache and P2P wire formats; migration is required before
  1.0.

## [1.0.0] — 2026-06-19 (historical internal milestone; never published)

This local milestone recorded the engine carve. It was not a public crates.io
release; the package version was later corrected to the `0.2.0-alpha.1`
development line. The architectural changes below remain historical facts.

### Changed
- **License: AGPL-3.0-or-later** (with a commercial license offered separately).
  Per-file SPDX headers on every source file; the canonical AGPL text in `LICENSE`.
- **Interactive TUI cockpit ships (default-on `tui` feature).** Bare `blut`
  opens a ratatui cockpit whose panels read the engine's OWN state — jobs, log,
  run DAG (`graph_snapshot`), lineage/provenance, artifacts, run history,
  metric leaderboard, compare, run metrics, and a registry-driven recipe
  catalog — instead of any domain file convention. `--no-default-features`
  drops `ratatui`/`crossterm`/`fuzzy-matcher` for a lean CLI-only binary (bare
  `blut` then prints help). The cockpit is fully domain-clean and covered by a
  headless render/interaction test harness (`tui --check` + 57 unit tests).
- **Proposed stable surface contract.** This milestone drafted the future 1.0
  additive-only contract; the current preview terms live in `API.md`.

### Removed
- The bundled generic-LLM cookbook (concrete stages + lamu/hf backends + the
  Python payload) — extracted to a separate `blut-backends` crate. The engine
  ships zero concrete stages/recipes/backends and zero Python.
- The `lerna` git-dependency (the sole `cargo publish` blocker) — replaced by a
  native, Hydra-compatible `config::hydra` compose module (serde_yaml only).
- The `meta_repo_root()` meta-repo assumption — the engine no longer assumes it
  lives inside a parent repo; meta-repo detection moved to the cookbook.

### Added
- `examples/first_cookbook.rs` — a runnable end-to-end demo (typed stages → plan
  → executor → content-addressed cache hit), doubling as an integration test.
- Off-systemd graceful degrade: containment falls back to a bare spawn (with a
  warning) when `systemd-run` is absent or `BLUT_NO_CONTAIN` is set.
- `run_contained.sh` embedded via `include_str!` so containment works from a
  clean install with no repo-layout dependency.
- A `macos-latest` CI compile-check; a `tui`-feature CI lint/test pass.
- crates.io publish hygiene: a trimmed package (`exclude` of dev artifacts) plus
  `homepage`/`documentation` metadata.

## [0.11.0] — 2026-06-17

**Feature-complete milestone (pre-1.0, ADR 0044).** This release lands the last
permitted breaking core edits before the optimization gate and 1.0.0 — all of
the BLUT-API phases A–G plus the cookbook taxonomy rebuild.

*Version note.* A forward minor bump for breaking pre-1.0 changes (semver 0.x).
The "feature-complete, pre-optimization" milestone semantics live in the 1.0
charter (ADR 0044), not in the version integer — `v0.10.0` was already
published, so the milestone moves forward rather than down to a `0.9.0` marker.

### Added

#### Cookbook taxonomy + ingredient registry (ADR 0050 / 0051)

- **Course** — `RecipeCategory` renamed to `Course`, the culinary layer in
  BLUT ▸ Cookbook ▸ Course ▸ Recipe ▸ Ingredient, with new `Pretraining` and
  `Gate` variants and a compiler-forced exhaustive `order()` driving the TUI menu
  sort. `pub type RecipeCategory = Course` is kept as a transitional alias.
- **Ingredient registry** — `IngredientSpec` + a fail-closed
  `build_ingredient(kind, name, cfg, **extra)` over 13 kinds (data / sampler /
  preprocess / model / forward / loss / optimizer / scheduler / step / ema /
  eval / checkpoint / logging), generalizing ADR 0050's optimizer registry. Each
  spec coerces a frozen dataclass config and declares whether its selection
  changes the cached artifact (`cache_relevant`).
- **Trainer decomposition** — the canonical trainers' sub-stages are extracted
  into byte-identical ingredients (optimizer construction, the WSD scheduler,
  EMA, the QAT generator step, the shared SNN SSM step, the joint / 4-state / MAE
  / teacher losses, the joint + 4-state evals, atomic-save / manager /
  durable-resume checkpointing, and the LMA data adapters), and all five canon
  trainers (`train_l3_teacher`, `pretrain_mae`, `pretrain_ssl_tueg`,
  `train_4state_controller`, `train_joint`) call `build_ingredient` instead of
  inlining the primitive.

#### Durable + auto resume (Phase D)

- **Durable resume.** A killed training run no longer restarts from scratch. The
  orchestrator owns a crash-gated *policy* (`framework::resume`): a stable
  per-config resume directory keyed on the stage cache key, a `state.json`
  run-state marker with a heartbeat, and a 5-row decision (resume an in-process
  retry or a crashed prior run; refuse a live concurrent run; start fresh
  otherwise). The trainer owns the *mechanics* (`durable_resume.py`): atomic,
  prev-rotated recovery checkpoints at each validation embedding the optimizer +
  RNG state, and a `--resume` that restores model + **optimizer** + RNG (a
  continuous loss curve, no cold-optimizer dip). Covers in-process OOM retry,
  cross-invocation re-run after a crash, and clean optimizer resume. A
  `no_resume` recipe arg forces a fresh start. (Epoch-boundary granularity;
  mid-epoch dataloader-position resume remains out of scope.)
- **Auto-resume on retry** — a retried stage reuses `decide_resume`; on
  attempt ≥ 2 with the same run id the decision is deterministically `Resume` and
  the trainer is relaunched with `--resume`.
- **`StageError::Diverged`** — a new transient, retryable error; a `KillOnNaN`
  divergence is reclassified as `Diverged`, so a NaN kill becomes a bounded
  retry → auto-resume instead of a hard failure.
- **Stage hooks** — defaulted `preflight` / `resume_handle` / `divergence_check`
  methods on the `Stage` trait, mirrored on the dyn shim (zero ripple to existing
  `impl Stage`).

#### Hyperparameter optimization

- Search space + samplers, `blut hpo run` fan-out, a median/percentile early-stop
  scheduler, **ASHA** async successive halving, **PBT**, a **TPE** Parzen-window
  sampler, trial tracking with `blut hpo show` / `best`, `Control::Spawn` runtime
  sub-plan injection, and a queryable graph backend behind `blut dag`.

#### Metric store + GPU saturation (Phase E)

- A queryable metric store — additive `metrics` / `gauges` tables folded from
  `status.jsonl` at run-end — with `blut compare A B`; a live GPU-saturation
  sampler recording `gpu_util / mem / temp / power` per run, deriving the
  `gpu_saturation` / `gpu_wasted` headline numbers with a starvation sentinel; and
  a per-epoch `BLUT_METRIC` val_r line surfaced as a `StageStep`.

#### Partitions, declarative recipes, sensors, scheduler (Phase G)

- A Dagster-class partition primitive with `blut partition {define,list,status,
  backfill}`; a `.toml` declarative-recipe engine over an erased stage registry;
  lineage **freshness** (flag outputs stale vs code/data drift); **named sensors**
  (the resume policy reconciled as a `Sensor`); and a per-device parallel backfill
  scheduler with multi-GPU locks (one cell per GPU).

#### Schema, TUI, launcher

- Schema-preflight args validation (`recipe show` lists typed fields; bad-type
  args rejected pre-dispatch); a real `blut tui --check` flag; in-TUI arg
  validation + auto-focus on the launched job's log; a `Launcher::capacity()`
  device probe.

### Changed

- **`code_sha` in the cache key (one-time global cache bust).** Stage cache keys
  and the nondeterministic-output fingerprint now fold a `code_sha` (cookbook git
  SHA ‖ resolved script content hash), so editing a trainer re-runs the stage
  instead of serving a stale checkpoint. Key tag bumped `v1 → v2` — a one-time
  global invalidation; pre-1.0, no migration is owed.
- Build hash re-stamped on commits, not only branch switches.
- Sequestered 15 dead scripts + 5 tests to `deprecated/`, relocated the optimizer
  modules under `lamquant/ingredients/optimizers/`, and retired the archived
  mamba-SNN + per-arch student trainers.

### Fixed

- `blut footprint list|forget` (audited calibration heal); the warm fork pool
  sized to the measured ~8 G/worker peak; per-worker peak-RSS logging
  (`LAMQUANT_RSS_DEBUG`); recovery checkpoints load with `weights_only=False`; and
  assorted HPO / executor / partition / scheduler review hardening.

## [0.10.0] — 2026-06-12

First tagged release. The engine has graduated from the early `0.1` prototype
to a memory-admitting, parallel DAG orchestrator with a distributed launch
seam, run lineage, and declarative scheduling.

### Highlights

- **Memory-admission resource broker (ADR 0046).** Every memory-spending
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

[Unreleased]: https://github.com/Quitetall/blut/commits/main
