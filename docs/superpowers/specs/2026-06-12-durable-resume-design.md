# Durable Resume (BLUT-API Phase D) — Design

**Status:** approved 2026-06-12. **Scope:** in-process stage retry + cross-invocation
re-run + clean optimizer resume. **Approach:** Hybrid (Rust orchestrator owns the
crash-gated *policy*; Python trainer owns the checkpoint *mechanics*).

## Problem

A killed training run restarts from scratch. Stages retry on OOM (the cgroup
self-heals the cap), but each retry re-invokes `train_joint.py` from epoch 0. The
trainer already has resume *bones* (`--resume`, an every-50-epoch recovery
snapshot, `warm_latest.ckpt`) but: the orchestration never auto-enables resume,
the cadence is too coarse, `--resume` restores model weights + epoch + phase but
**not the optimizer** (SOAP preconditioners/momentum lost), and a cross-run
re-run cannot find the prior run's checkpoint (job_dir is fresh per run_id).

## The interface (the only Rust↔Python contract)

A stable directory **outside** the per-run job_dir, under the data root:

```
<LAMQUANT_DATA_ROOT>/Training/resume/<recipe>-<resume_key>/
  state.json            # run-state marker — Rust reads, Python writes
  recovery_latest.ckpt  # model + optimizer + RNG + epoch + phase + scaler
  recovery_prev.ckpt    # one-deep rotation (a mid-write kill can't corrupt both)
```

- **`resume_key` = the stage's existing cache key** (threaded into `StageContext`
  like `fb_warm`/`launch_target`). It is already the engine's canonical
  "same training" fingerprint: same args + same input corpus → same key → same
  dir → finds the checkpoint. Any training-relevant arg change → different key →
  fresh dir → fresh run (conservative: a config change never resumes onto a stale
  checkpoint). A cosmetic arg change (e.g. `--logger`) also forks the dir — an
  accepted trade (never *wrongly* resume).
- The flags: the stage always passes `--resume-dir <dir>` (where the trainer
  writes recovery + state); it passes `--resume <dir>` only when the policy says
  resume.

## Rust — crash-gated policy (the train stage + an engine `resume` module)

`state.json` carries `{status, run_id, pid, heartbeat_unix}`. A pure decision
function (engine, unit-tested) over `(state, this_run_id, now, stale_threshold)`:

| `state.json` | Decision |
|---|---|
| absent | **Fresh** (never ran) |
| `status:"finished"` | **Fresh** (prior training completed) |
| `run_id == this run` | **Resume** (in-process retry: a prior *attempt* of THIS run died) |
| `run_id != this`, heartbeat **stale** (≥ `stale_threshold`) | **Resume** (prior run crashed) |
| `run_id != this`, heartbeat **fresh** | **RefuseConcurrent** (a live run owns this dir — hard error, not a silent fresh start) |

`run_id` (from the job_dir basename) unifies in-process OOM retry and
cross-invocation resume through one mechanism; the heartbeat distinguishes
"crashed" from "concurrent". `stale_threshold` = 3× the trainer's heartbeat
interval. A `no_resume: true` recipe arg forces Fresh (the override).

- New engine module `framework/resume.rs`: `ResumeState` (serde for `state.json`),
  `ResumeDecision` enum, `decide_resume(...)` pure fn, `resume_dir(data_root,
  recipe, resume_key)` path helper. Domain-agnostic — a generic durable-resume
  primitive any cookbook stage can use.
- `StageContext` gains `cache_key` (set from `task.key` in `run_node`; `for_test`
  default = a zero key).
- `lamquant_train_joint` stage: compute `resume_dir`, read `state.json`, call
  `decide_resume`; on `RefuseConcurrent` → `StageError`; on `Resume` → add
  `--resume <dir>`; always add `--resume-dir <dir>`. Honor `no_resume`.
- `lamquant_joint_codec` recipe: add `no_resume: bool` (default false).

## Python — checkpoint mechanics (`train_joint.py`)

- **Recovery checkpoint at every validation boundary** (replaces the every-50-epoch
  coarse recovery → loss bounded to ~`val_interval` epochs). Atomic write + the
  one-deep `recovery_prev` rotation. Contents extend today's: `encoder`,
  `decoder`, **`optimizer.state_dict()`**, **RNG states** (torch/cuda/numpy/python),
  `epoch`, `phase`, `scaler` (if AMP), and the embedded `resume_key` (config-hash
  guard).
- **`--resume <dir>`** loads `<dir>/recovery_latest.ckpt` (falling back to
  `recovery_prev` on a corrupt latest) and restores model + **optimizer** + RNG +
  scaler + epoch + phase. Rejects a checkpoint whose embedded `resume_key` ≠ the
  current one (never load a foreign checkpoint). The added `optimizer` + RNG
  restore is the "clean optimizer resume" — a continuous loss curve, no SOAP
  cold-start dip.
- **`state.json`**: write `{status:"running", run_id, pid, heartbeat_unix}` at
  start; rewrite `heartbeat_unix` each validation; `status:"finished"` at clean
  exit. `--run-id` is passed by the stage.
- **Mid-epoch is out of scope** — no dataloader/sampler-position restore; resume
  restarts at the next epoch boundary.

## Edge cases

- Corrupt `recovery_latest` (killed mid-write) → fall back to `recovery_prev`; both
  unreadable → Fresh (warn).
- A checkpoint whose embedded `resume_key` ≠ current → rejected (Fresh).
- In-process retry × broker: unchanged — the OOM still escalates the cgroup cap;
  resume just means the retry continues from the last epoch instead of epoch 0.

## Testing

- **Rust** (`resume.rs`): the decision table — all 5 rows — as a pure function over
  a constructed `ResumeState` + run_id + clock; the concurrency-refusal row; the
  `resume_dir` path derivation.
- **Python**: a save→kill→resume round-trip asserting optimizer state + epoch +
  phase + RNG restore (loss continuity), and the corrupt-`latest`→`prev` fallback,
  and the foreign-`resume_key` rejection.
