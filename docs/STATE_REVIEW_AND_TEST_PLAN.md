# BLUT — Full State Review, Issue Catalog & All-States Test Plan

> Synthesis of 6 subsystem diagnostics (framework-core, recipes, tui-cockpit,
> orchestration-reliability, build/test/clippy, python sidecar).
> Date: 2026-05-28. Branch: `decomp/lossless-extract`.
> Crate: `/mnt/4tb/LamQuant/blut` (single crate `blut` v0.1.0, edition 2024, rustc 1.85).
>
> The bar: BLUT must (1) drive the full LamQuant pipeline end-to-end,
> (2) have a complete/usable TUI cockpit (queue + monitor + custom metrics +
> custom training data), (3) be standalone-correct (build/test/clippy clean),
> and (4) be cross-machine reproducible. **LamQuant-first.**

---

## 1. Executive Summary

**Does BLUT work today? NO — not against the bar.**

BLUT builds (`cargo build --workspace --all-targets` exits 0) and its 341
library unit tests + 70 framework tests + 49 recipe tests are green. The
*engine* (Stage→Plan→Recipe DAG, typed compile-time wiring, content cache,
sequential executor) is architecturally sound and correct **for the happy
sequential path**. But every load-bearing capability the bar asks for is
either broken, unwired, or absent:

- **End-to-end LamQuant pipeline: BROKEN.** The monorepo→submodule split
  orphaned every wrapped script. With the default `lamquant_home`
  (`~/Desktop/LamQuant`, which no longer has `ai_models/`), all five
  LamQuant recipes die at their first stage's `script.exists()` preflight.
  No single `lamquant_home` value can satisfy both `<home>/scripts/` (meta-repo)
  and `<home>/ai_models/` (Neural submodule) at once. There is **no recipe**
  for the actual EDF→.lma encode (`lml encode`), no recipe/stage for
  split-manifest generation, no full pipeline recipe chaining
  encode→labels→split→train→gate→promote, and **no TNN training recipe at all.**

- **TUI cockpit: INCOMPLETE.** Jobs list, per-job log tail, and the full
  system panel are written-but-dead-code (`draw_jobs`/`draw_log`/`draw_system`
  never called by `draw()`). The cockpit can launch a single recipe via a
  raw-JSON editor but cannot queue, cannot show terminal-state jobs, cancel
  targets an invisible selection, and there is no dataset picker or
  custom-metrics UI (U3/U4/U5/U6 pending).

- **Queue: ABSENT.** No queue, no `Queued` JobState, no scheduler. A second
  run hard-errors `"GPU held by ..."` (non-blocking exclusive lock) and writes
  `JobState::Failed` silently.

- **Reliable cancel/cleanup: BROKEN.** This is the session's actual failure
  (40 leaked semaphores + runaway trainer). `blut cancel` is a **no-op for the
  primary recipe/TUI path** (no pid file written), kills only the direct child
  pid (orphaning DataLoader/distributed grandchildren — no process groups),
  and the in-process `CancellationToken` is never observed during a running
  training stage.

- **Cross-machine reproducible: FAILS HARD.** The executor hashes the bincode
  artifact *handle* (which embeds absolute `PathBuf` + size/loss metadata)
  instead of the artifact's own `content_hash()`. Identical content at a
  different path → different downstream cache key → `--shared-cache` never hits
  across machines or relocated job dirs.

### Top 5 blocking issues

| # | ID | Severity | One-liner |
|---|----|----------|-----------|
| 1 | KILL-1 / KILL-2 / KILL-3 | critical | Cancel is a no-op on the main path; when it does fire it kills only the direct child, orphaning the GPU process tree (the leaked-semaphore failure). |
| 2 | RCP-1 / RCP-7 | critical | Submodule split orphaned every wrapped script; no single `lamquant_home` resolves both `scripts/` and `ai_models/`. Every LamQuant recipe fails at stage 1. |
| 3 | FW-1 | critical | Downstream cache key depends on upstream artifact's absolute path + metadata, not content. Breaks cross-machine repro and same-machine relocation; `content_hash()` is never invoked. |
| 4 | FW-2 | critical | No partial-output cleanup on crash/cancel/failure; some stages write outside `stage_dir` entirely (`weights/snn/*.pt`). Half-written checkpoints survive resume. |
| 5 | RCP-3 + (no TNN recipe) + QUEUE-1 | critical/high | No split-manifest stage (every train recipe hard-requires it), no TNN training recipe, no job queue. Pipeline cannot be self-contained or serialized. |

---

## 2. Per-Subsystem State

### 2.1 Framework core engine (`src/framework/`)
**State: SOUND for happy sequential path. 70/70 framework lib tests pass.**

3-layer Stage→Plan→Recipe orchestrator on tokio.
- **Artifacts** (`artifact.rs`): `Artifact` trait = KIND + SCHEMA + HASH_CONTENTS
  + `content_hash()` + `primary_path()`. `ContentHash` = SHA-256 newtype
  (mmap ≥16MiB / streamed; rayon merkle for dirs). `ArtifactMetadata` sidecar.
  Tuple impls with arity domain-separation. Concrete artifacts store
  `content_hash` **as a struct field** alongside `path: PathBuf`.
- **Stage** (`stage.rs`): typed trait (Input/Output/Args + NAME/SCHEMA/
  RESOURCES/DETERMINISTIC). Object-safe `StageDyn` blanket impl.
  `ErasedArtifact = {kind, schema, payload: bincode}`. `into_typed` checks
  kind then schema then bincode-decodes.
- **Plan** (`plan.rs`): typed `Plan<Out,B>` DAG, compile-error on wrong wiring,
  Kahn topo-sort with cycle/empty detection.
- **Cache** (`cache.rs`): key = SHA-256(`blut.cache.v1` ‖ stage_name ‖ schema ‖
  input_hash ‖ canonical_json(args)). Per-job `_cache/<key>/output.bin`,
  optional `--shared-cache`. Atomic insert (tmp+rename+sync_all). atime LRU.
- **Executor** (`executor.rs`): `SequentialExecutor` only (parallel declared as
  future commit). **Key fact: the executor NEVER calls `Artifact::content_hash()`.**
  `content_hash_from_erased` (executor.rs:498) hashes `art.kind + art.schema +
  art.payload` — the *whole bincode handle* including absolute paths.
- **Status** (`status.rs`): broadcast → `status.jsonl` writer, 4096-bounded
  channel, immediate flush on lifecycle, 500ms batch for StageStep.

### 2.2 Recipes (`src/recipes/`)
**State: COMPILE-clean (49/49 tests), but NONE of the LamQuant recipes can RUN.**

11 recipes. Family (A) — generic lamu/HF (`finetune_*`, `dpo_*`, `distill_*`,
`eval_suite`, `hf_finetune_from_dataset`) — compile and are out of scope.
Family (B) — LamQuant (`lamquant_data_prep`, `lamquant_snn`, `lamquant_encoder`,
`lamquant_oracle`, `lamquant_combined_decoder`) — all compile but all abort at
the first stage's `script.exists()` preflight because of repo-layout drift.
`default_lamquant_home()` (runner.rs:400) returns `~/Desktop/LamQuant`, whose
`ai_models/` is **gone** (now in `LamQuant-Neural/ai_models/`; `scripts/` at
meta-repo level). `lamquant_data_prep` is a misleading 1-node stub (convert_lma
only). No TNN training recipe. No full-pipeline recipe.

> **CORRECTION vs the recipes diagnostic:** the diagnostic claimed
> `generate_activity_labels.py` (BLUT-4) and `precompute_fullband.py` (BLUT-8)
> are MISSING. **Verified present** at
> `LamQuant-Neural/ai_models/snn/generate_activity_labels.py` and
> `LamQuant-Neural/ai_models/dataset_sim/precompute_fullband.py`. The script
> *path-resolution* break (wrong `lamquant_home` root) is real; the
> scripts-deleted claim is not. `pccp_gate.py` and the `lml` binary and the
> `/mnt/4tb/data/Training/lma` corpus all verified present.

### 2.3 TUI cockpit (`src/tui/mod.rs` 967 lines + `system.rs`)
**State: launches single recipes; monitoring & queue & custom-data UI INCOMPLETE.**

Single-screen ratatui app. Renders header → "Pipeline status" (running jobs
only) → "Resources" → recipe menu → keys → footer. Refreshes every 1.5s.
Can launch via fuzzy Picker → raw-JSON Editor → detached `blut recipe run`.
`draw_jobs`/`draw_log`/`draw_system` are **dead code** (compiler-confirmed).
Cancel targets an unrendered `ListState` (blind). No dataset picker (U3),
no swap-candidates (U4, helper exists unwired), no kind validation (U5),
no user-recipe loading (U6). `blut tui </dev/null` exits 1 (no headless mode).
**ZERO tests in `src/tui/`.**

### 2.4 Orchestration + reliability (`scheduler_lock`, `jobs`, `python_kill`, `backends`, ...)
**State: unit-tests green; integration-seam reliability is broken.**

GPU arbitration = cross-process advisory lockfile (O_EXCL, RAII Drop). Explicitly
NOT a fairness queue. KILL path = single-pid SIGTERM→SIGKILL, no process groups.
`RecipeCommand::Run` never writes a pid file → `blut cancel` no-ops on the main
path. The one pid that *is* written (`run_train_via_recipe`) is blut's own pid.
`--background` is documented-but-unimplemented (returns Ok, leaves phantom
Running job). Custom-metrics plumbing (EvalReport → merge_reports) is clean and
working. datasets_db (sha256 dedup, WAL) + atomic YAML registry are solid.

### 2.5 Build / Test / Clippy (standalone correctness)
**State: build PASS; test 1 FAIL (cascading); clippy FAIL HARD; no CI.**

- **Build:** PASS, 16 warnings (4 unused imports, 1 unused var, ~10 dead_code
  concentrated in `src/tui/`).
- **Test:** 341 lib tests pass; `cli_smoke::help_prints_top_level` FAILS on a
  **stale assertion** (`tests/cli_smoke.rs:30` expects `"Local fine-tuning"`;
  actual about line is `"BLUT — interactive training cockpit"`). This failure
  **aborts the rest of the integration suite** — `convert_smoke.rs` (5 tests)
  and `python_backend_smoke.rs` (4 tests) never run. Line 29 also asserts
  `"lamu-train"` (fragile).
- **Clippy:** `cargo clippy --all-targets -- -D warnings` FAILS (lib 38 errors,
  lib-test 72 errors): ~16 promoted build warnings + ~22 genuine lints
  (`assertions_on_constants` ×14 and `await_holding_lock` ×7 are test-only/intentional).
- **CI:** **NO `.github/workflows/` exists** — verified. nextest NOT installed
  despite task #133 marked complete. Neither gate is enforced anywhere.

### 2.6 Python sidecar (`blut/python/`)
**State: self-check PASS (4 tests), but only `build_optimizer` covered.**

`trainer.py` (SFT), `trainer_dpo.py`, `trainer_distill.py`, `datasets_loader.py`.
Only `test_build_optimizer.py` (dependency-free, 4 tests) passes. DPO/distill/
loader have ZERO python-side tests; exercised only via Rust `*_smoke` tests —
which themselves didn't run (cli_smoke abort).

---

## 3. Issue Catalog (ranked critical → low)

| id | severity | subsystem | title | file:line | fix-sketch |
|----|----------|-----------|-------|-----------|------------|
| FW-1 | **critical** | framework | Downstream cache key = upstream handle bincode (abs path + metadata), not content; `content_hash()` never called | `executor.rs:498-506`, `:154,:400,:483` | Have executor seed `logical_outputs` from `art.content_hash()` (the typed handle's stable address), not `content_hash_from_erased`. Strip `path` from hashed payload or hash content not handle. |
| FW-2 | **critical** | framework | No partial-output cleanup on crash/cancel/fail; some stages write outside `stage_dir` (`weights/snn/*.pt`) | `executor.rs:350-415`; `sft_train.rs:75,147`; `lamquant_train_mamba_snn.rs:166-179` | Write stage outputs to `stage_dir/.tmp` then atomic-rename on success; on Err/cancel rm the tmp. Force all stage writes inside `stage_dir`. |
| KILL-1 | **critical** | orchestration | cancel kills only direct child pid, not process group → orphaned GPU grandchildren (the leaked-semaphore failure) | `python_kill.rs:30-75`; backends spawn sites (no `process_group`/`setsid`) | `setsid`/`pre_exec(setpgid)` on spawn; `killpg(-pgid, SIG…)` in graceful_kill; waitpid-confirm reap. |
| KILL-2 | **critical** | orchestration | `RecipeCommand::Run` never writes a pid file → `blut cancel` no-ops for primary + all TUI launches | `main.rs:699-767`; `jobs.rs:325-333`; `tui/mod.rs:199-213,388` | Write the python child pid (from backend handle) to the job's pid file at spawn; have cancel read it. |
| KILL-3 | **critical** | orchestration | The one pid written is blut's own pid; cancel SIGTERMs blut, which has no signal handler | `main.rs:1338`; no `signal::`/`ctrl_c` in main | Record child pid not `process::id()`; install a SIGTERM/ctrl_c handler that cancels the token + kills the process group. |
| RCP-1 | **critical** | recipes | Submodule split orphaned every wrapped script; all LamQuant recipes fail at stage-1 preflight | `runner.rs:400-408`; all `lamquant_*` stage `script_path()` | Make `lamquant_home` resolution multi-root (separate `scripts_root` + `ai_models_root`, or `BLUT_SCRIPTS`/`BLUT_AI_MODELS` env). |
| RCP-3 | **critical** | recipes | No split-manifest stage, yet every train recipe HARD-REQUIRES `split_manifest` | recipes reject empty `split_manifest` (e.g. `lamquant_snn.rs:166`); no wrapping stage | Add `lamquant_build_split_manifest` stage wrapping `build_seizure_split_manifest.py` / `build_snn_train_val_split.py`; chain into a pipeline recipe. |
| QUEUE-1 | high | orchestration | No job queue — second run hard-errors `"GPU held"` or (with `--background`) is silently dropped | `scheduler_lock.rs:31-33,199-214`; `jobs.rs:48-55` | Add `JobState::Queued` + a persistent FIFO queue drained by one scheduler behind the lock. |
| MONITOR-1 | high | orchestration | `CancellationToken` checked only between stages; never during a running training stage | `executor.rs:163,351-354`; `lamquant_train_mamba_snn.rs:279-283` | `tokio::select!` `run_erased` vs `ctx.cancel.cancelled()`; pass cancel into backend, call `backend.cancel()` on trip. |
| RCP-7 | high | recipes | No single `lamquant_home` resolves both `<home>/scripts/` and `<home>/ai_models/` | `lamquant_convert_lma.rs:80-82` vs train/gate `home.join("ai_models")` | Same multi-root fix as RCP-1; split path roots structurally. |
| RCP-2 | high | recipes | `convert_lma` wraps `bulk_lml_to_lma.py` against a DELETED LML tree; the real EDF→.lma encode (`lml encode`) has no recipe | `bulk_lml_to_lma.py:65-66` hardcoded; `tui/mod.rs:269` only `lml` mention | Add `lamquant_encode_lma` stage wrapping the `lml encode` binary on EDF dirs; deprecate bulk-LML path. |
| FW-3 | high | framework | Tuple/merge input_hash omits child KINDs (single-input path keeps kind) → possible false cache hit | `cache.rs:99-117`; `executor.rs:254-272` | Include child `art.kind` in the tuple input_hash; add input KIND to the cache key. |
| FW-4 | high | framework | Stale cache on Args default-value change (no schema-of-args) and HASH_CONTENTS=false stat-fingerprint staleness | `cache.rs:299-356`; `artifact.rs:300-319`; `lamquant.rs:68,92,115` | Hash an args *schema fingerprint* into the key; consult `content_hash()` live for HASH_CONTENTS=false artifacts. |
| FW-5 | high | framework | Resource semaphores acquired in declared order with no global order → deadlock under planned ParallelExecutor; ResourceTimeout never used | `executor.rs:327-348`; `error.rs:45` | Sort `RESOURCES` by a canonical `Resource` ordinal before acquire; add an acquisition timeout → `ResourceTimeout`. |
| BLD-T1 | high | build/test | `cli_smoke::help_prints_top_level` FAILS on stale about-line; aborts 9 downstream integration tests | `tests/cli_smoke.rs:29-30` | Update assertion to `"BLUT — interactive training cockpit"` / tool name `blut`. |
| BLD-C1 | high | build/test | `cargo clippy --all-targets -- -D warnings` FAILS (lib 38, lib-test 72); no CI to enforce either way | clippy log | Fix unused imports/dead-code; `#[allow]` the intentional test-only lints (`assertions_on_constants`, `await_holding_lock`); add a CI lane. |
| TUI-01 | high | tui | Cannot queue — 2nd launch fails instantly & silently | `main.rs:731`; `scheduler_lock.rs:154-161`; `tui/mod.rs:199-214` | Depends on QUEUE-1; surface `Queued`/`Failed` reason in cockpit. |
| TUI-02 | high | tui | Jobs list / log tail / system panel are dead code — monitoring degraded | `tui/mod.rs:568-774` draw(); `:857,:907,:932` uncalled | Wire `draw_jobs`/`draw_log`/`draw_system` into `draw()` layout; add a TestBackend render test. |
| TUI-03 | high | tui | No custom-training-data UI (no dataset picker — U3/U5) | `tui/mod.rs:553-560`; `datasets_db.rs:291` unused | Implement U3 dataset picker fed by `datasets_db::list()`; U5 validate kind vs `input_kinds`. |
| RCP-4 | high | recipes | SNN-label generation un-recipe'd (script exists but no stage chained) | `lamquant_generate_snn_labels.rs:73`; not in `RECIPES` | Chain `lamquant_generate_snn_labels` into a data-prep pipeline recipe; fix path root (RCP-1). |
| RCP-5 | high | recipes | `lamquant_data_prep` is a 1-node stub (convert only; no labels/split) | `lamquant_data_prep.rs:7-12` | Add build_manifest + generate_labels + split bridges (needs fork/tuple wiring); rename so it isn't advertised as "full setup". |
| RCP-6 | high | recipes | `convert_lma` labels-dir default points at dead monorepo path (3 inconsistent label paths) | `lamquant_convert_lma.rs:142-143`; `bulk_lml_to_lma.py:66` | Default to `/mnt/4tb/data/Training/labels`; unify across stage + script. |
| MONITOR-2 | medium | orchestration | Liveness inferred from on-disk state only; crashed job shows Running forever | `jobs.rs:77-88,199-201` | Reconcile JobState vs `pid_alive` on `jobs`/refresh; mark dead-pid Running jobs as Failed. |
| QUEUE-2 | medium | orchestration | `await_unlock`→`acquire_exclusive` TOCTOU race; thundering-herd `--allow-evict` waiters | `scheduler_lock.rs:199-214,235-250`; `main.rs:1145-1155` | Single atomic acquire-or-wait; add jitter/backoff or replace with the FIFO queue. |
| FW-6 | medium | framework | GPU-holding stages do multi-GB `hash_dir` while holding the GPU permit | `executor.rs:357,400-415`; `sft_train.rs:147` | Drop permits before hashing/insert; or use stat-fingerprint for large GPU outputs. |
| FW-7 | medium | framework | Artifact SCHEMA bump → hard `BadInput` on cached downstream entry instead of self-heal | `stage.rs:96-103,298-301`; `cache.rs:99-117` | On schema-mismatch decode, treat as cache MISS (re-run) not error; or fold upstream artifact schema into the key. |
| FW-8 | medium | framework | `cache.insert` failure (ENOSPC) is warn-and-continue → expensive nondeterministic run not memoized | `executor.rs:410-415`; `status.rs:150-152` | Treat cache.insert failure on a nondeterministic stage as a hard error (so resume re-runs deterministically) or retry. |
| FW-9 | medium | framework | Mid-stage cancel not wired into running stage; cancelled training output may cache as success | `executor.rs:163`; `sft_train.rs:142-145`; `lamquant_train_mamba_snn.rs:280-283` | Same as MONITOR-1; on cancel, do NOT insert cache, return `StageError::Cancelled`. |
| FW-10 | medium | framework | Tuple/merge reconstruction relies on edge insertion order == fork order (unverified; hardcoded schema:1) | `executor.rs:178-231,254`; no E2E fork test | Carry explicit child indices on edges; assert order; derive tuple schema from children. |
| KILL-4 | medium | orchestration | pid-reuse window in `graceful_kill_pid`; no waitpid/identity recheck | `jobs.rs:325-333`; `python_kill.rs:35-53` | Validate start-time/ppid before signalling; waitpid-confirm reap. |
| REL-1 | medium | orchestration | `status.jsonl` append not atomic; torn final Done/Failed line silently dropped on read | `jobs.rs:107-124,135-139`; `status.rs:111-161` | Single writer per job; line ≤ PIPE_BUF or O_APPEND with fsync of terminal events. |
| REL-2 | medium | orchestration | `--background` documented but unimplemented → phantom Running job, training never starts | `main.rs:1120-1132,1377-1384` | Implement real detach (daemonize / nohup child) or remove the flag + its job-dir write. |
| TUI-04 | medium | tui | No custom-metrics UI; StatusUpdate is a closed enum | `protocol.rs` StatusUpdate; `hf_finetune_from_dataset.rs:60-64` | Add an open metrics map to StatusUpdate; metrics-selection pane. |
| TUI-05 | medium | tui | Swap-candidates (U4) unwired; recipe sub-screen (U3) absent | `recipe.rs:102`; `tui/mod.rs:46-50` | Add recipe sub-screen calling `swap_candidates()`. |
| TUI-06 | medium | tui | Cancel target invisible (selection bound to unrendered list) | `tui/mod.rs:374-394`; `draw_jobs` dead | Depends on TUI-02; render the jobs list with highlight. |
| RCP-8 | medium | recipes | Orphaned legacy stages reference q31-era paths (precompute_l3 q31_events, build_manifest --q31-dir) | `lamquant_precompute_l3.rs:67`; `build_manifest.rs` header | Either remove from catalog or repoint to LMA-era inputs; mark deprecated. |
| RCP-9 | medium | recipes | PCCP gate stages run from wrong home; `pccp/` at meta-repo, gate script in submodule | `lamquant_pccp_gate_snn.rs:107`; `pccp_gate.py` in Neural | Pass explicit `--pccp-root` (meta-repo) + resolve script via ai_models root. |
| REL-3 | medium | orchestration | Non-transactional dataset lineage; WAL pragma may silently not apply under contention | `main.rs:1085-1091`; `datasets_db.rs:157-177` | Make register failure hard before training; raise pragma retry budget. |
| BLD-C2 | medium | build/test | `await_holding_lock` ×7 on test `TEST_LOCK` across `.await` | `executor.rs:629,650` | `#[allow(clippy::await_holding_lock)]` on the test mod (intentional serialization). |
| FW-11 | low | framework | `args_schema()`/args serialization use `.expect()` → panic on NaN lr etc. on plan-build hot path | `plan.rs:133-134,…`; `stage.rs:280`; `sft_train.rs:38` | Return `RecipeError` from `to_value` errors instead of `.expect()`. |
| FW-12 | low | framework | kind/topo-coverage checks are `debug_assert!` only — compiled out of release | `executor.rs:113-117,122-126,371-375` | Promote data-path kind checks to real returned errors. |
| FW-13 | low | framework | `from_hex` byte-slices after len==64 only (multibyte → panic); atime LRU degrades on relatime mounts | `artifact.rs:235-244`; `cache.rs:239-242` | Validate ASCII-hex before slicing; LRU by an explicit access-log not atime. |
| TUI-07 | low | tui | Picker cursor unbounded; Down past end → Enter no-ops | `tui/mod.rs:501,510-519,822` | Clamp cursor to `filtered.len()-1`; add PageUp/Down + scroll-into-view. |
| TUI-08 | low | tui | Dead scaffolding (MenuItem/MenuAction/lamquant_default_args) | `tui/mod.rs:56,63,265` | Wire `lamquant_default_args` into launch path or delete. |
| TUI-09 | low | tui | No headless/`--check` mode; TUI hard-fails without a TTY | `tui/mod.rs:399`; `main.rs:108` | Add `blut tui --check` that builds App + renders to TestBackend and exits 0. |
| TUI-10 | low | tui | Log only re-tailed for selected job; stale pid-`-` jobs shown live forever | `tui/mod.rs:332-356`; `jobs.rs:200` | Depends on MONITOR-2 reconciliation + TUI-02. |
| CONVERT-1 | low | orchestration | convert step not cancellable / not lock-protected; orphaned f16 on interrupt | `convert.rs:60-100`; `main.rs:1195` | `kill_on_drop(true)` + pid capture + cancel hook; rm intermediate f16 on Err. |
| RCP-10 | low | recipes | Recipe docstrings advertise "convert LML to LMA" but live corpus is EDF→LMA via `lml` binary | `lamquant_snn.rs:128`; `convert_lma.rs:7` | Update narrative once RCP-2 encode recipe lands. |
| CONST-C3 | low | build/test | `assertions_on_constants` ×14 in lamquant.rs test mod | `artifacts/lamquant.rs:412+` | Wrap in `const { assert!(...) }` or `#[allow]`. Intentional. |
| BLD-W1 | low | build/test | 16 build warnings (unused imports, dead code) | build log; `tui/mod.rs`, `jobs.rs:586` | Remove unused imports; resolve dead code via TUI-02/TUI-08. |
| BLD-D1 | low | build/test | Doctest coverage near-zero (1 compile-fail doctest) | `framework/plan.rs` | Add runnable doctests on Stage/Plan/Recipe public API. |

**Counts:** critical 5, high 14, medium 16, low 12 — **47 total.**

---

## 4. Gaps vs the Bar

### 4.1 Full-pipeline orchestration — FAILS
The end-to-end chain the bar wants is **encode (EDF→.lma) → generate labels →
build split-manifest → train (SNN/TNN/encoder/decoder/oracle) → PCCP gate →
promote.** What's missing:
- **EDF→.lma encode:** no recipe/stage at all (real step is the `lml encode`
  binary via `encode_corpora.sh`; BLUT only knows the dead `bulk_lml_to_lma.py`
  LML→LMA path). [RCP-2]
- **Split-manifest:** no stage; every train recipe hard-requires it. [RCP-3]
- **Labels:** script exists but un-recipe'd. [RCP-4]
- **TNN training:** **no recipe exists.** The bar's "train SNN/TNN/decoder" is
  unmet for TNN. `lamquant_export_firmware` exists as a stage but is chained
  into nothing.
- **No full-pipeline recipe** chaining the above. Closest (`lamquant_snn`)
  starts at `convert_lma` and assumes labels+split already exist.
- **Everything is unrunnable anyway** under the default home (RCP-1/RCP-7).
- **PCCP gate→promote tail** is unreachable E2E and split across two repo roots
  (RCP-9). Safe dry-run default means promotion is gated off — fine, but
  unexercisable today.

### 4.2 TUI completeness — INCOMPLETE
Cockpit can launch one known recipe via a raw-JSON editor. Missing: rendered
jobs list, live log tail, full system panel (all dead code, TUI-02); visible
cancel target (TUI-06); recipe sub-screen (TUI-05); headless mode (TUI-09).

### 4.3 Queue / Monitor / Custom-metrics / Custom-data
- **Queue: NOT MET.** No queue, no `Queued` state, no scheduler (QUEUE-1).
  2nd run hard-errors silently (TUI-01).
- **Monitor: PARTIALLY MET.** Running-job step/loss + system stats refresh
  live; but no rendered log tail, no terminal-state visibility (TUI-02), and
  liveness isn't reconciled against pid (MONITOR-2) so crashed jobs show Running
  forever.
- **Custom metrics: plumbing MET (EvalReport→merge_reports), UI NOT MET**
  (closed StatusUpdate enum, raw-JSON only) (TUI-04).
- **Custom training data: backend MOSTLY MET (datasets_db solid), UI NOT MET**
  (no picker, no kind validation — U3/U5) (TUI-03).

### 4.4 Reproducibility — FAILS
Cache key = bincode handle (absolute path + size/loss metadata), not content;
`content_hash()` never invoked (FW-1). HASH_CONTENTS=false stat fingerprints
(path+size+mtime) are baked into the handle and are inherently machine-specific
(FW-4b). `--shared-cache` never hits across machines or relocated job dirs.

### 4.5 Standalone correctness — FAILS
Build PASS, but `cargo test` has 1 failure that cascades to abort 9 integration
tests (BLD-T1), `clippy -D warnings` fails (BLD-C1), and there is **no CI**
(no `.github/workflows/`) to enforce any of it. Release-mode kind/topo checks
are `debug_assert!`-only (FW-12).

---

## 5. ALL-STATES Test Plan

Enumerated per subsystem. **[ZERO]** = no current coverage. **[unit-only]** =
covered in isolation but not at the integration seam that actually breaks.

### 5.1 Cache states
| State | Test | Coverage |
|-------|------|----------|
| Cache HIT (same args/input) | `cache_hit_skips_run_on_repeat_execution` | covered |
| Cache MISS (changed args) | exists at unit level | covered |
| Cache CORRUPT `.bin` (truncated/garbage) | feed a truncated `output.bin`, assert downgrade-to-miss + warn | **[ZERO]** |
| Cache key stable across job_dir / abs path (cross-machine) | produce same content at 2 paths, assert equal downstream key | **[ZERO]** (and currently FALSE — FW-1) |
| `cache.insert` failure (ENOSPC / read-only dir) | RO cache dir, assert nondeterministic stage handles non-memoization | **[ZERO]** (FW-8) |
| HASH_CONTENTS=false stat-fingerprint staleness (file mutated in place) | mutate file post-handle, assert NO stale downstream hit | **[ZERO]** (FW-4b) |
| `--shared-cache` global hit then per-job write | exercise shared path | **[unit-only]** |
| atime LRU prune | `lru_prune` unit | covered (but degrades on relatime — FW-13) |
| Artifact SCHEMA bump on cached downstream | bump upstream SCHEMA, assert re-run not hard `BadInput` | **[ZERO]** (FW-7, currently bricks) |

### 5.2 Resume / crash / partial-write states
| State | Test | Coverage |
|-------|------|----------|
| Resume after fully-completed stage | covered | covered |
| Resume after stage killed mid-write (partial `stage_dir`) | kill mid-run, assert cleanup or correct re-exec | **[ZERO]** (FW-2) |
| Resume after write OUTSIDE stage_dir (`weights/snn/*.pt`) | kill mid-train, assert orphan ckpt detected/cleaned | **[ZERO]** (FW-2) |
| Cancelled-then-resumed training stage | assert partial output NOT cached as success | **[ZERO]** (FW-9) |
| status.jsonl vs cache resume-oracle disagree | drop a StageEnd, assert resume still correct | **[ZERO]** (FW-8) |

### 5.3 Lock / queue states
| State | Test | Coverage |
|-------|------|----------|
| Lock free → acquire | `scheduler_lock` unit | covered |
| Lock HELD by live pid → 2nd acquire errors | unit | covered |
| Lock STALE (dead holder) → recovery | unit (kill(pid,0)) | covered |
| `await_unlock` → `acquire` TOCTOU race | 2 racers, assert no double-hold / clean loser | **[ZERO]** (QUEUE-2) |
| N waiters fairness / starvation | enqueue N, assert FIFO order | **[ZERO]** (no queue — QUEUE-1) |
| RAII Drop unlinks only own lock | unit | covered |
| Drop NOT run on abrupt SIGTERM → stale lock | SIGTERM blut mid-run, assert next acquire recovers | **[ZERO]** (KILL-3) |

### 5.4 Cancel / kill states
| State | Test | Coverage |
|-------|------|----------|
| Cancel BEFORE first stage | `cancel_before_first_stage_returns_cancelled` | covered |
| Cancel DURING long-running stage | cancel mid-`run_erased`, assert subprocess aborts | **[ZERO]** (MONITOR-1) |
| Cancel kills full process tree (grandchildren) | spawn child→grandchild, assert tree dies | **[ZERO]** (KILL-1, the actual failure) |
| `blut cancel` on recipe job (no pid file) | assert it actually kills (it doesn't today) | **[ZERO]** (KILL-2) |
| Cancel targets child pid not blut's own | assert written pid == child | **[ZERO]** (KILL-3) |
| pid-reuse window in graceful_kill | reuse pid, assert identity recheck | **[ZERO]** (KILL-4) |
| pid file round-trip | `write_pid`/`read_pid` unit | covered |

### 5.5 Recipe success/failure/partial states
| State | Test | Coverage |
|-------|------|----------|
| Recipe compiles (n_nodes/n_edges, arg-reject) | per-recipe unit | covered |
| Recipe RUNS E2E vs real/fixture lamquant_home | run a family-(B) recipe with real scripts | **[ZERO]** (RCP-1 — would catch the drift) |
| Wrapped scripts EXIST at resolved home | contract test asserting `script.exists()` for current layout | **[ZERO]** (stubs mask it — RCP-1) |
| Multi-root resolution (scripts/ + ai_models/ under one home) | assert both resolve | **[ZERO]** (RCP-7) |
| Split-manifest generation | no stage exists | **[ZERO]** (RCP-3) |
| Labels generation wrapping | no recipe; script exists | **[ZERO]** (RCP-4) |
| `convert_lma` idempotent-skip on live corpus | point at `/mnt/4tb/data/Training/lma`, assert LmaCorpus emitted | **[ZERO]** (RCP-2) |
| PCCP gate vs real registry location | run gate w/ real `pccp/registry.yaml` vs script in submodule | **[ZERO]** (RCP-9) |
| Fork→merge E2E through executor | run a forked plan, assert tuple payload order + input_hash | **[ZERO]** (FW-10; plan test only checks counts) |
| Recipe failure (non-zero subprocess exit) | assert StageError + JobState::Failed | **[unit-only]** |
| Recipe partial (some stages cached, some run) | mixed-cache run | **[unit-only]** |

### 5.6 Resource contention states
| State | Test | Coverage |
|-------|------|----------|
| Single-resource acquire/release | implicit | **[unit-only]** |
| Multi-resource acquire order (deadlock risk under parallel) | 2 stages [Gpu,Net] vs [Net,Gpu], assert no deadlock | **[ZERO]** (FW-5) |
| StageBlocked event emitted under contention | assert event | **[ZERO]** |
| ResourceTimeout | never constructed | **[ZERO]** (FW-5) |
| ParallelExecutor (entire path) | unwritten | **[ZERO]** (no parallel executor exists) |

### 5.7 Missing-env / config states
| State | Test | Coverage |
|-------|------|----------|
| `LAMQUANT_HOME` unset → default | implicit | **[unit-only]** |
| `LAMQUANT_HOME` set but wrong root | assert clean error not panic | **[ZERO]** |
| `current_exe()` fails → fallback to `blut` on PATH | assert behavior | **[ZERO]** (REL-3 / spawn_recipe) |
| Missing python / venv | assert clean error | **[unit-only]** |
| `blut tui` without TTY | verified exits 1 (no headless) | **[ZERO]** (TUI-09) |
| WAL pragma fails under contention | assert fallback | **[ZERO]** (REL-3) |

### 5.8 Monitor / status states
| State | Test | Coverage |
|-------|------|----------|
| status.jsonl normal write/read | unit | covered |
| Lagged broadcast burst (event drop) | flood channel, assert no lifecycle-event loss | **[ZERO]** (REL-1, status.rs:150) |
| Torn/partial final status line | write torn line, assert terminal event not silently lost | **[ZERO]** (REL-1) |
| JobState reconciliation vs dead pid | crash mid-run, assert Running→Failed sweep | **[ZERO]** (MONITOR-2) |
| `--background` phantom Running | assert (currently leaves phantom) | **[ZERO]** (REL-2) |

### 5.9 TUI states
| State | Test | Coverage |
|-------|------|----------|
| `handle_key` Overlay None→Picker→Editor + Esc + Ctrl-C | pure state-transition unit | **[ZERO]** (highly testable) |
| `filter_recipes` fuzzy ordering | pure unit | **[ZERO]** |
| `template_for` schema→JSON per recipe + `{}` fallback | pure unit | **[ZERO]** |
| `recipe_menu` hotkey assignment / reserved-key exclusion | pure unit | **[ZERO]** |
| `draw()` actually renders jobs/log/system | TestBackend buffer assertion (would catch dead-code) | **[ZERO]** (TUI-02) |
| Picker Down-past-end no-op | unit | **[ZERO]** (TUI-07) |
| `spawn_recipe`/`cancel_selected` (shell out) | needs injectable command runner | **[legitimately hard]** |
| `SystemSnapshot::probe` parse helpers | refactor to take strings, unit-test | **[ZERO]** |

### 5.10 Build/test/clippy/python states
| State | Test | Coverage |
|-------|------|----------|
| `cargo build --all-targets` | manual | PASS |
| `cargo test --workspace` | manual | **1 FAIL cascades (BLD-T1)** |
| `clippy -D warnings` | manual | **FAIL (BLD-C1)** |
| CI gate enforcement | no `.github/workflows/` | **[ZERO]** |
| nextest runner | not installed | **[ZERO]** (task #133 mismarked) |
| python trainer_dpo/distill/loader | no tests | **[ZERO]** |
| `--no-default-features` firmware-style build | n/a (single crate) | n/a |

---

## 6. Recommended Fix Order (critical-first, dependency-aware)

**Phase 0 — unblock the test/CI gate (so every later fix is verifiable):**
1. **BLD-T1** — fix the stale `cli_smoke` assertion (1-line); unblocks 9
   integration tests. *No dependencies; do first.*
2. **BLD-C1 + BLD-W1 + BLD-C2 + CONST-C3** — clean dead code / unused imports,
   `#[allow]` the intentional test-only lints, get `clippy -D warnings` green.
3. **Add a CI lane** (`.github/workflows/ci.yml`: build + `cargo test` +
   `clippy -D warnings`; install nextest). Locks in everything below.

**Phase 1 — the session's actual failure (cancel/orphan):**
4. **KILL-1** — process groups on spawn (`setsid`/`setpgid`) + `killpg` in
   graceful_kill + waitpid-confirm. *Prereq for KILL-2/3 to be meaningful.*
5. **KILL-2** — write the python child pid to the job pid file on the recipe
   path. *Depends on KILL-1.*
6. **KILL-3** — record child pid (not `process::id()`) + install a SIGTERM/
   ctrl_c handler in blut that cancels the token + kills the group.
7. **MONITOR-1 / FW-9** — `tokio::select!` cancel vs `run_erased`; on cancel
   don't cache, return `Cancelled`. *Depends on the cancel token actually
   reaching the backend (KILL-3 handler).*
   *Test gates: 5.4 process-tree-dies + cancel-during-stage.*

**Phase 2 — make the LamQuant pipeline runnable:**
8. **RCP-1 + RCP-7** — multi-root path resolution (`scripts_root` +
   `ai_models_root`, env-overridable). *Prereq for every other recipe fix.*
9. **RCP-6** — unify the labels-dir default to `/mnt/4tb/data/Training/labels`.
10. **RCP-2** — add `lamquant_encode_lma` stage (`lml encode` EDF→.lma).
11. **RCP-4** — chain `generate_snn_labels` stage.
12. **RCP-3** — add `build_split_manifest` stage.
13. **New: `lamquant_tnn` training recipe** + chain `lamquant_export_firmware`
    (closes the TNN gap).
14. **New: `lamquant_full_pipeline`** recipe chaining encode→labels→split→
    train→gate→promote (needs fork/tuple bridges; RCP-5 stub becomes real).
15. **RCP-9** — pass explicit `--pccp-root` to gate stages.
    *Test gates: 5.5 E2E recipe run + script-exists contract.*

**Phase 3 — reproducibility (cache correctness):**
16. **FW-1** — executor uses `Artifact::content_hash()` for `logical_outputs`,
    not the handle bincode. *The cross-machine fix; highest-leverage.*
17. **FW-4** — args-schema fingerprint in key + live `content_hash()` for
    HASH_CONTENTS=false. *Depends on FW-1 (same hashing path).*
18. **FW-2** — atomic tmp-dir + rename for stage outputs; force writes inside
    `stage_dir`; cleanup on Err/cancel. *Pairs with FW-1 for correct resume.*
19. **FW-3 / FW-10** — fold input KIND into tuple key; verify fork/merge order.
20. **FW-7 / FW-8** — schema-mismatch → cache-miss (self-heal); cache.insert
    failure handling.
    *Test gates: 5.1 cross-path key stability + 5.2 partial-write cleanup.*

**Phase 4 — queue + monitor reliability:**
21. **QUEUE-1** — `JobState::Queued` + persistent FIFO queue + single scheduler
    behind the lock. *Prereq for TUI-01.*
22. **QUEUE-2** — atomic acquire-or-wait (eliminates TOCTOU once queue exists).
23. **MONITOR-2** — reconcile JobState vs `pid_alive`.
24. **REL-2** — implement (or remove) `--background`.
25. **REL-1 / KILL-4 / CONVERT-1 / REL-3** — status atomicity, pid identity
    recheck, convert cleanup, lineage transactionality.

**Phase 5 — TUI completeness:**
26. **TUI-02** — wire `draw_jobs`/`draw_log`/`draw_system` into `draw()`
    (also add a TestBackend render test — prevents the dead-code regression).
27. **TUI-06** — visible cancel selection (depends on TUI-02).
28. **TUI-01** — surface Queued/Failed-reason in cockpit (depends on QUEUE-1).
29. **TUI-03 / U3 + TUI-05 / U4 + U5** — dataset picker + kind validation +
    swap-candidates sub-screen.
30. **TUI-04** — open metrics map + metrics-selection pane.
31. **TUI-09 / TUI-07 / TUI-08 / TUI-10 / U6** — headless `--check`, picker
    clamp, dead-scaffolding cleanup, user-recipe loading.

**Phase 6 — hardening (low):**
32. **FW-5** — canonical resource-acquire order + timeout (prereq for the
    eventual ParallelExecutor; do before parallel lands).
33. **FW-11 / FW-12 / FW-13** — remove `.expect()` on plan-build, promote
    debug-asserts to real errors, fix `from_hex` slicing.
34. **FW-6** — drop GPU permit before heavy hashing.
35. **BLD-D1** + python sidecar tests (dpo/distill/loader).

---

### Dependency notes
- Phase 0 must precede everything (you can't verify a fix without a green gate).
- KILL-1 is the root of the cancel chain (KILL-2/3, MONITOR-1, FW-9 all build on it).
- RCP-1/RCP-7 is the root of the recipe chain (every other recipe fix is dead
  without multi-root resolution).
- FW-1 is the root of the reproducibility chain (FW-2/3/4/7/8).
- QUEUE-1 gates TUI-01; TUI-02 gates TUI-06/TUI-10.
