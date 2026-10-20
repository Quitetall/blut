<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0044 — BLUT API standardization: one launch contract, placement-as-runtime, durability + retry as a Stage contract

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress` (Accepted as a design contract, implementation the owner's parallel lane · In progress 2026-08-27)

- **Date:** 2026-06-08
- **Supersedes/extends:** [ADR 0034](0034-blut-scope-charter.md) (BLUT scope), [ADR 0037](0037-blut-cookbook-architecture.md) (cookbook arch), the cookbook-split work (C2c done) `[redacted: private repository]`
- **Grounded in:** an evidence audit 2026-06-08 (4 dims, file:line-cited) + two code-verified adversarial critiques run in-session (an ML-infra architect review and a Rust-correctness review) `[redacted: private repository]`. This ADR turns each finding into a principle. Where the audit was wrong, the critique corrected it — noted inline.

## Context

BLUT is a clean 3-repo split (C2c): the `blut` engine + two domain cookbooks `[redacted: private repository]`. The owner is solidifying the API so **all usage is standardized and every future capability is an *addition* to the core, never an edit of it** — so BLUT evolves in parallel with training. Owner's fixed rules: (1) **no args → TUI, args → CLI**; (2) **retry + protections** — a world-class, ML-focused Dagster, not a script-runner.

(*Lerna* = a vendored, pinned Rust crate giving Hydra-compatible layered config + override-grammar + sweep expansion, adopted as BLUT's config/sweep resolver per the steal-map of [ADR 0037](0037-blut-cookbook-architecture.md); it stays an internal dependency and is referenced only in the concept map below.)

### Honest current state (audit + critique, code-verified)

**Works:** engine + an AI-via-CLI launch+monitor path for ONE recipe.
- Entry duality exists: `blut`→TUI (10 views); `blut recipe run <name> --args <json>`→CLI.
- Core abstractions present: `Recipe`/`RecipeDef`, `Stage<I,O>`, `Artifact` (typed KIND), `Cookbook`+`Registry` (catalog inverted — C1), `TrainingBackend`, `SequentialExecutor`.
- **Event streaming already exists in core**: `ExecCtx.status_tx: broadcast::Sender<StageEvent>` + `spawn_status_writer` write `<job_dir>/status.jsonl` for **every** recipe incl. lamquant (executor.rs:46/139/354, status.rs:111). *(Corrects audit dim-2, which grepped only the domain cookbook and wrongly concluded "no status.jsonl".)*
- Containment machinery real + smoke-verified (`run_contained.sh`→`systemd-run --user MemoryMax`). Kill-chain robust (setsid pgroup, killpg SIGTERM→SIGKILL, pid-reuse identity guard, unit teardown). PYTHONPATH crosses the unit boundary. Job tracking + dead-pid→Failed reconciliation works. The per-epoch validation metric is persisted to a TUI-readable `epochs_*.csv`. Live monitoring works **via the terminal CLI** (tracing re-emits python stdout, runner.rs:205).

**Broken / short of "perfect" — the gap ledger:**

| # | Gap (evidence) | Impact |
|---|---|---|
| **G1** | 6/7 recipes not launchable: a data-conversion stage resolves a missing script *before* its skip, and a trainer script was archived out from under two recipes `[redacted: codec research]`. | only one codec recipe near-launchable |
| **G1b** | That recipe's stage stats a checkpoint path that differs (dir + suffix) from where the trainer writes it, so it fails *after* a full run; the recipe also forwards bare `--epochs/--lr` flags the trainer does not accept → argparse error on override `[redacted: codec research]`. | the experiment trains then false-fails; probe flags die at argparse |
| **G1c** | **The current experiment's config is not expressible by the recipe**: the stage builds only config/tier/seed/epochs/batch/lr/...; the experiment needs a set of model-shape and training flags the recipe cannot forward `[redacted: codec research]`. | **the experiment runs as the raw script today, NOT the recipe** |
| **G2** | `args_schema_fn` emits bare `{"$ref":"#/definitions/Args"}` (schemars `subschema_for` drops definitions). | `recipe show` uninformative for ALL recipes; TUI editor empty → un-fillable by a human |
| **G3** | TUI `spawn_recipe` pipes child to `Stdio::null()` (mod.rs:467); the validation metric arrives via `tracing` not `StageStep` so it's absent from status.jsonl. | launch-from-TUI is blind; Log pane lacks the live metric |
| **G4** | trainer writes logs `__file__`-relative to its own tree; TUI tails `$LAMQUANT_HOME`-or-cwd. MetricLog parquet invisible (`find_logs` `.csv`-only); wandb default-off+offline. | TUI shows stale logs sans env var |
| **G5** | `BLUT_CONTAINED` never auto-set → bare `blut recipe run` uncapped; an experiment probe script (since retired; training now goes through the cookbook recipe) ran raw `$PY`, no wrapper. | the 2026-06-04 global-OOM can recur |
| **G6** | `blut tui --check` (#174 "done") absent in binary (unit-test only). | false-done |
| **G7** | No `RetryPolicy`, no error classifier, no auto-resume in the Executor (grep: comment only). `--resume` exists per-stage but defaults off + Executor never sets it; a failed stage is never cached so a relaunch re-runs from scratch (executor.rs FW-2). | "retry + protections" is ~100% aspirational |
| **G8** | `Plan<Out,B>` is compile-time parameterized over backend `B`; there is **no `Launcher`**. Placement is fused into the plan's type. | Slurm/Ray = core surgery, violates "additions-only" |
| **G9** | cache key = `sha256(name‖schema‖input‖args)` (cache.rs:85), **no `code_sha`**; big artifacts fingerprint on path+size+**mtime** (lamquant.rs:14). | edit a kernel, same args → silently reuses buggy ckpt; regen dataset → false cache-bust |
| U3–U7 | args-form/picker, swap, kind-validation, user-recipe dir, HF-in-lamquant-binary — all `pending`. | short of "perfect via TUI" |

**Verdict to "perfect tool yet?": No.** Solid engine + a working *AI-via-CLI* path for one codec recipe (enough to run the current experiment today **as the raw script, wrapped**, with the checkpoint-path bug fixed for the recipe path). NOT "all training+monitoring, perfect, human-drivable": 6/7 recipes dead, args undiscoverable, TUI launch blind, containment opt-in, retry/resume unimplemented, placement welded into the type system. Gaps are well-scoped; several are bigger than they look.

---

## Decision — the standardized API (12 principles)

Each names the gap it closes + the correction if the draft/audit was wrong. The `blut` crate is the stable surface; domain growth = cookbook additions.

### P1 — Entry duality is a contract (exists)
`blut` (no positional) → TUI. `blut <recipe> [--args <json>]` → CLI. Both are thin front-ends over ONE **launch+event contract** (P4). Neither holds privileged logic.

### P2 — Core surface + semver reality
The stable surface is a small set of **abstractions, not "six traits"**: the traits `Stage<I,O>`, `Cookbook`, `Recipe`, `TrainingBackend`, `Launcher` (P3) — **plus** the `RecipeDef` struct (its public `fn`-pointer fields `args_schema_fn`/`compile_fn` ARE the ABI every cookbook constructs), the `Artifact` struct + KIND, and the public error enums (`StageError`/`RecipeError`/`PlanError`). Document this whole set as the contract.
- **Semver is impossible at `0.1.0`** (0.x: any bump is breaking; a cookbook pinning `"0.1"` won't receive `0.2`). So "cookbooks pin a major, a minor never breaks them" requires **cutting `blut` 1.0**. Precondition: **the semver clock starts AFTER the core-edit phases below (A–F) land** — P3/P4/P7/etc. are themselves core edits, so they are the *last* permitted ones; then tag 1.0 and hold the line. Until 1.0: additive-by-convention, no guarantee.

### P3 — Placement is runtime, not a type parameter: `Launcher` ≠ `Backend` (closes G8; highest risk)
Split two conflated axes:
- **`Backend` = engine identity** (HF Trainer | LAMU trainer | LamQuant kernels). Compile-time tagging is fine.
- **`Launcher` = runtime placement** (`Local` | `Systemd` | `Slurm` | `RayJobSubmit`). A **`dyn Launcher` selected per-run**, NOT a `Plan` type parameter. The Executor takes `&dyn Launcher`; it exposes **capacity** (n_devices) + owns `CUDA_VISIBLE_DEVICES` assignment so a sweep can pack/space runs.
- **Do this before the semver freeze** — otherwise the cluster move forces a breaking 1.x release on the exact "distributed-later" axis the owner named, silently breaking "additions never edit core."

### P4 — One launch+event contract, detached + attach (closes G3; corrects draft P3)
Do **not** add an in-process `execute()`/`EventSink` — the streaming machinery already exists (`status_tx` → `status.jsonl`) and an in-process call would fight process-global state (GPU lock keyed on `process::id()`, the process-wide SIGTERM trap, single `ACTIVE_CHILD`). Instead:
- TUI and CLI share one **launch contract**: spawn the run **detached** (the `systemd-run --user` transient unit from P8 already outlives the parent → survives TUI/SSH close with **no daemon**) and **attach** to its `status.jsonl` event stream.
- Fix the real gaps: TUI `spawn_recipe` must stop nulling stdout and instead **tail `status.jsonl`** into the Log pane; **promote the per-epoch validation metric to a `StageStep` event** (today it's only a `tracing` line, so it never reaches status.jsonl). "Same code path" = same launch+event contract, not same OS process.

### P5 — Single schema source (closes G2)
`args_schema_fn` MUST emit a complete schema (`schema_for!`/`RootSchema`, not `subschema_for`). **Also fix the consumer**: for a flat `Args` struct schemars 0.8 inlines `properties` at the top level and leaves `definitions` empty, so the TUI `template_for` (reads `definitions.Args.properties`) and `recipe show` must be updated to read **top-level `properties`** — else they still render `{}`. Arg **validation** (`required`/types) is net-new: `compile_fn` today only `serde_json::from_value` (ignores unknowns); add a JSON-Schema validator dep.

### P6 — Submit-time preflight (closes G1, G1b, G1c)
Fail fast, with a precise message, before any work runs. **Preflight lives on the cookbook/`Stage` side, not in the domain-agnostic Executor** (a `Stage::preflight()` hook is itself a core *addition* — acceptable, but name it as such; don't inject Python-script knowledge into core):
- **script-exists** (names the recipe + missing path; catches the missing/archived scripts at submit). Restore/repoint those scripts; move the convert skip *ahead* of resolution.
- **recipe↔script arg contract**: forward only flags the script accepts (kill bare `--epochs/--lr`) **and add the experiment's flags** `[redacted: codec research]` so the recipe can actually express it — otherwise "the codec recipe is byte-runnable for the experiment" is hollow and the experiment stays out-of-band.
- **output contract** (#257): stage passes an explicit run-id-stamped `--ckpt-dir/--out` and stats exactly that → success ≠ false-failure (also unblocks same-path A/B).
- **kind-check** where applicable (U5).

### P7 — Durability + retry are a `Stage` contract the Executor drives (closes G7; **the single most important change**)
Today the Executor owns retry but has zero hook to the checkpoint; the cookbook owns the checkpoint but no hook to retry. Close it:
- Add to the `Stage` trait: `resume_handle(stage_dir) -> Option<ResumeToken>` (where the checkpoint is, what it must contain, how to validate) and `divergence_check()`.
- **`RetryPolicy { max_attempts, backoff, classify }`** is net-new (no retry loop exists). Classify: transient (CUDA OOM, ECC/Xid, SIGTERM-preempt, transient I/O, **NaN/divergence**) → bounded backoff; deterministic (bad config/data/script, validation) → **0 retries**. On retry of a non-deterministic stage the Executor **auto-injects `--resume`** from the discovered checkpoint.
- **Checkpoint payload contract**: weights + optimizer + LR-schedule pos + AMP GradScaler + RNG/dataloader cursor + EMA + **quantization-calibration state** + step counter. Resume preflight **validates the token contains them** or fails-fast (prior bites: #49 optimizer-state, a quantization-scale explosion).
- **Corruption-safe checkpoints**: `tmp + fsync + atomic rename`, keep-last-N, **validate-on-load** (sidecar SHA / dry `torch.load`), + a **free-space preflight** (disk-full mid-`torch.save` → silent garbage today).
- **Per-step divergence sentinel** in the event stream (loss NaN/Inf, grad-norm spike, loss > k·EMA) → typed `Diverged` error early (don't burn the whole GPU budget; the gate only fires post-run).
- **Env-var ownership boundary** (avoids a future "whose job is it"): the **Launcher** (P3) sets placement env (`CUDA_VISIBLE_DEVICES`, rank/world_size); the **Backend/stage** sets determinism env (`CUBLAS_WORKSPACE_CONFIG`, the seed) and reads the seed from validated args; the **Executor** sets only retry/resume env (`--resume`, attempt counter). One writer per axis.

### P8 — Protections + containment by default (closes G5)
- Containment is **opt-OUT**: the Executor wraps every launch in `run_contained` (`systemd-run --user MemoryMax`) by default; a **real `--no-contain` flag** (net-new; today only `BLUT_CONTAINED=1` env exists) reachable from both front-ends; a **capability probe** for `systemctl --user`. **Fail-CLOSED** when user-systemd is absent: refuse to launch and instruct the operator to pass `--no-contain` explicitly — do NOT silently run uncapped (the documented failure mode is a global-OOM that kills the whole user session; a silent fail-open re-arms exactly that). Update the bare-spawn unit tests.
- **Caveat: `MemoryMax` caps host RAM, not VRAM.** The OOM that "swept user@1000" is often VRAM — bounded only by the P3 Launcher device reservation (per-process VRAM fraction / MPS / `CUDA_VISIBLE_DEVICES`), not cgroups. State this; don't over-claim.
- Fail-closed gates (Dagster asset-check ≡ PCCP). `RunManifest` (git_sha+config+data/ckpt SHAs) written even on crash.

### P9 — Cache honesty (closes G9)
- Rename the concept: **"stage identity memoization,"** not "content-hash" (a stochastic re-run is not byte-identical; the key is deliberately identity, not output content).
- **Add `code_sha`** (cookbook git SHA + resolved script hash) to the key, both `key_for` and the nondet synthesized hash — else editing a kernel with identical args silently reuses the old checkpoint.
- **Datasets keyed on stored content-version** (the manifest hash that already exists), not path+size+**mtime**.
- **Non-deterministic stages RUN on explicit relaunch** by default (`--use-cache` to opt into memoization, or a loud `SKIPPED (cache hit, unverified) — pass --force`); silent skip-on-relaunch is wrong for training.
- The cache resumes only **between** stages; intra-stage durability is P7's checkpoint contract (Temporal's durability is intra-activity — borrow it at the right granularity).

### P10 — Observability is a core contract, anchored to the run dir (closes G4)
- Anchor metric/log writes to **`BLUT_JOB_DIR`** (already passed + crosses the systemd `--setenv` boundary; trainer just doesn't read it yet) — **not** a `__file__`-relative root nor `blut_python_root`. This sidesteps both the `LAMQUANT_HOME` divergence *and* the migration-#2 breakage (when `python/` moves, a file-relative path follows the file while Rust stays anchored to the engine root).
- Enrich beyond per-epoch: **per-step** loss/grad-norm/lr/throughput, **system gauges** (GPU util/mem/temp/power, host RAM/disk), and an **artifact lineage edge** (ckpt ← data+code+parent-ckpt). TUI `find_logs` reads parquet **and** csv. Add `blut compare <runA> <runB>`. Same stream for TUI/CLI/wandb; don't lean on offline-wandb as the comparison plane — keep a queryable local run index (Parquet/SQLite) for HPO ranking.

### P11 — Partitions, data-version/freshness, sensors (concept-map completeness)
- **Partitions** as a core concept: a recipe materializes a partitioned asset; the Executor caches/retries/**backfills per partition key**. Sweep cells (seed×tier×corpus) and stratified CV folds ARE partitions — without this, a 30-cell sweep losing cell 17 has no first-class "re-run cell 17," plus a multi-cell scheduler (bounded concurrency over the Launcher's capacity) is required.
- **Data-version + freshness**: surface dataset content-version as the input identity (repeated dataset-manifest version churn deserves a named concept) + a freshness check ("ckpt stale vs current manifest").
- **Sensors**: reconcile the existing `policy.rs` (a cron-driven auto-train policy) as a Dagster *sensor* (fires runs on a condition) into the contract.

### P12 — No false-done (closes G6)
A task is done only when the affordance exists in the shipped binary, verified by running it (`blut tui --check` must be a real flag). Reconcile #174.

### Concept map (steal concepts, reject runtimes — ADR 0034 / 0037)
`asset→Artifact` · `op→Stage` · `asset-check→fail-closed gate` · `code-location→Cookbook` · `job→Recipe` · `partition→partition key (P11)` · `run/backfill→execute + Lerna sweep + per-partition backfill` · `data-version/freshness→P9/P11` · `sensor→policy.rs (P11)` · `durable-execution→intra-stage checkpoint (P7) + between-stage memoization (P9)` · `retry/heartbeat→RetryPolicy (P7) + dead-pid reconcile`. **One Rust binary per cookbook; no daemon/DB/web-UI/server** (Lerna stays an internal pinned dep, never in the public API).

---

## Implementation path (phased; owner's parallel lane; each independently shippable)

Ordered: correctness + the seams that must precede the semver freeze, then ergonomics. **All of A–F are core edits; the semver clock + `blut` 1.0 come after F (P2).**

- **Phase A — Launch correctness (makes "all 7 launch" true).** Fix #257 (run-id-stamped `--ckpt-dir/--out`, stat-exactly-that). Restore/repoint the missing scripts (+ move skip ahead of resolution). Recipe↔script arg contract: drop dead `--epochs/--lr`, **add the experiment's flag set** so the recipe expresses it. **Gate:** all 7 pass `recipe_paths_contract`; 1-epoch smoke each; an experiment-config dry-run compiles.
- **Phase B — Schema source (P5).** `RootSchema` fix **+ update `template_for`/`recipe show` to top-level `properties`**; codec recipe `default_args`; add a validator dep. **Gate:** `recipe show <each>` lists fields; TUI editor pre-filled.
- **Phase C — Launcher seam (P3) [do before freeze].** Split `Backend`(engine) / `dyn Launcher`(placement); Executor takes `&dyn Launcher`; `Local`+`Systemd` impls; capacity + `CUDA_VISIBLE_DEVICES`. **Gate:** a 2-cell sweep packs onto available devices; adding a stub `Slurm` launcher needs no `Plan`/`Stage` signature change.
- **Phase D — Durability + retry (P7) [the headline].** `Stage::resume_handle/divergence_check`; `RetryPolicy` + classifier; auto-`--resume` on retry; checkpoint payload contract + atomic-write + validate-on-load + free-space preflight; per-step NaN sentinel. **Gate:** kill a training stage → relaunch resumes at last ckpt (optimizer+LR+RNG intact); inject NaN → early `Diverged` + classified retry/fail; corrupt ckpt → fail-fast not silent-garbage.
- **Phase E — Protections + cache honesty + observability (P8/P9/P10).** Containment opt-OUT + `--no-contain` + capability probe; `code_sha` in key + dataset content-version + force-fresh; `BLUT_JOB_DIR`-anchored per-step+system+lineage stream, parquet-visible, `blut compare`. **Gate:** forced host-OOM kills only the unit; kernel edit (same args) re-runs not skips; `blut tui` shows the live metric with zero env vars.
- **Phase F — Unification + bus-factor (P4/P12 + U-items).** One launch contract, detached+attach, TUI tails status.jsonl + the validation metric as StageStep; U3 args-form/picker, U5 kind-validation, U6 user-recipe dir+loader, U7 HF in lamquant binary, real `tui --check`. **Gate:** a non-AI human launches+watches+resumes any run entirely in the TUI.
- **Phase G — Sweeps + partitions + sensors (P11).** Partitions + per-partition cache/retry/**backfill**; the bounded-concurrency multi-cell scheduler over the Launcher's capacity; data-version/freshness; reconcile `policy.rs` as a sensor. **Gate:** a killed N-cell sweep backfills only the failed cells; a manifest bump flags stale downstream ckpts.
- **Then: tag `blut` 1.0, freeze the surface (P2).**

## Consequences
- **+** A genuinely durable, safe, human-drivable ML platform (retry/resume/divergence/containment), Rust-native, one binary — the world-class-Dagster the owner wants.
- **+** Placement-as-runtime (`dyn Launcher`) means the cluster move is an *addition*, not a breaking core release — the standardization holds on the axis that matters.
- **+** Unblocks BLUT evolving in parallel with training (this ADR is the handoff).
- **−** A–F are real core work and are the *last* core edits before 1.0; the "additions-only" promise is honest only from 1.0 on (stated, not hidden).
- **−** **The current experiment cannot wait for this.** Until Phase A's arg contract lands, it runs as the raw trainer script wrapped in `run_contained`, monitored via terminal stdout + `epochs_*.csv` — not via the BLUT recipe `[redacted: private path]`. **(Historical: that raw-launcher path was retired 2026-06-28; training is now exclusively via the domain cookbook recipe.)**

## Anti-reinvention
Reject (ADR 0034 / 0037 / here): Dagster/Temporal/Ray/Hydra **runtimes**, dynamic dylib plugins, a second config language. Adopt their **concepts** + Arrow/Lerna **libraries**, Rust-native. Recorded so they are not re-litigated.

## Progress Log *(append-only; newest last)*

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. P3 landed, the semver precondition did not. Launcher is a runtime choice as decided -- `src/config/launcher.rs:967` returns `Box<dyn Launcher>` from `launcher_for(target)`, not a Plan type parameter. But P2 states plainly that the contract this ADR defines cannot hold at 0.x and that 'cutting blut 1.0' is the precondition, with the core-edit phases A-F as the last permitted breaking changes. `Cargo.toml` still reads `version = 0.2.0-alpha.1`, which is what was published, so the semver clock has not started and the stable surface is still additive-by-convention.

- _2026-08-27_ — gave this record a runnable citation gate `[redacted: private path]` (it proves the record still points at files that exist; it does not claim the decision is implemented) and removed its closure-debt entry `[redacted: private path]`.
