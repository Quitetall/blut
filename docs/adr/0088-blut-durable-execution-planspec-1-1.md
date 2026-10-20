<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0088 — Extend durable resume to mid-epoch via PlanSpec v1.1 additive per-node retry/timeout and a checkpoint-cadence contract

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text; correct an earlier statement with a *new* dated
> entry beneath it. When the deliverable passes its gate, fill **Completion** and
> set Status to `Complete — frozen YYYY-MM-DD`.

## Status

`Proposed 2026-07-10` → `In progress` (part 1 of 3 landed) → `Complete — frozen 2026-07-10` (all 3 gate parts pass)

## Context

BLUT already resumes a killed run: the landed durable-resume mechanism is
crash-gated Rust state (`state.json` + heartbeat) plus python atomic checkpoint
recovery, so a run restarts from the **last completed epoch**. That is
node-level durability — a dead stage re-runs, and the content-addressed cache
(ADR 0078) makes already-completed upstream nodes free.

Two gaps remain against the durable-execution bar set by Temporal (which
guarantees an interrupted workflow resumes at the exact step, with per-activity
retry policies and timeouts as first-class config):

1. **Resume is epoch-granular, not step-granular.** A run killed 90% through a
   long epoch replays that entire epoch. For a multi-hour codec epoch that is
   hours of wasted compute even though a mid-epoch checkpoint existed.
2. **Retry and timeout are not in the plan.** PlanSpec v1 (ADR 0078) has no
   per-node `retry` or `timeout` field. Today a flaky stage either fails the run
   or relies on an executor-global default; there is no typed, per-node,
   content-addressed way to say "retry this ingest 3× with backoff, kill it at
   30 min." The recovery policy lives in code, not in the checked graph.

The charter keeps content-addressed cache keys unchanged (0078): whatever we add
must **not** perturb the hash of an existing node, or every cached artifact
silently invalidates.

## Decision

Ship **PlanSpec v1.1** as a strictly *additive* serde evolution plus a
**checkpoint-cadence contract** for mid-epoch resume:

1. **Additive PlanSpec fields.** Add two optional, `#[serde(default)]` fields to
   the node struct: `retry: RetryPolicy` (`max_attempts`, `backoff`,
   `retry_on: {AnyError | Transient}`) and `timeout: Option<Duration>`. Both
   default to the current executor behaviour (retry policy = the existing
   crash-gated single re-run; timeout = none). **Crucially, neither field enters
   the cache-key hash** — the cache key is over the node's *inputs and
   computation*, and a retry/timeout is an execution-control knob, not a
   computation change. v1 plans deserialize unchanged; v1.1 plans with the
   fields absent hash identically to v1.
2. **Checkpoint-cadence contract.** A resumable training stage declares a
   `checkpoint_cadence` (steps or wall-time) and writes crash-consistent
   step-checkpoints via the existing python atomic-recovery path (write-temp +
   fsync + atomic rename), each recording `{step, rng_state, optimizer_state,
   data_cursor}`. On resume, the Rust `state.json` records the *last durable
   step* (not just epoch); recovery seeks the data cursor and RNG to that step.
   Cadence is a contract, not best-effort: a stage that claims mid-epoch
   resumability but writes no step-checkpoint fails its own gate.
3. **Per-node retry honoured by the executor**, bounded by the never-OOM broker:
   a retry re-enters admission (0046/0047) like any launch — a retry storm can
   never bypass the broker or OOM the box.

## Rationale

- **Additive-serde-with-defaults is the only cache-safe evolution.** Because the
  new fields carry `#[serde(default)]` and are excluded from the hash input, an
  existing plan's every node keeps its exact cache key — no artifact
  invalidation, satisfying the 0078 charter constraint literally.
- **Step-granular resume is where the compute is.** Codec epochs are hours;
  epoch-granular replay on a late kill wastes the most expensive resource we
  have. The recovery machinery (atomic rename, heartbeat) already exists —
  v1.1 only extends what `state.json` records from epoch to step.
- **Retry/timeout in the plan, not in code, matches the typed-DAG thesis.** ADR
  0078 made the graph the checked source of truth; recovery policy belongs there
  too, so a reviewer reads a node's resilience from its spec.
- **Retry through the broker keeps the never-OOM invariant total** — retries are
  not a side channel around admission.

## Alternatives Considered

- **A PlanSpec v2 with a version bump and migration.** Rejected: a major version
  forces re-hashing or a compatibility shim across every cached artifact — the
  exact cache-invalidation the charter forbids. Additive v1.1 needs no migration
  because absent fields are byte-identical to v1.
- **Executor-global retry/timeout only (no per-node).** Rejected: a flaky
  network-ingest stage and a deterministic compute stage need opposite policies;
  a single global knob either over-retries pure compute (masking real bugs) or
  under-retries transient I/O. Per-node is the granularity that matters.
- **Full Temporal-style event-sourced replay (deterministic re-execution from an
  event log).** Rejected as over-scope: it demands every stage be a
  deterministic, side-effect-free replay function — incompatible with GPU
  training's nondeterminism. Content-addressed checkpoints + step cursors give
  the same *resume-from-here* guarantee without the determinism straitjacket.

## Consequences

- The node struct in `blut-types` grows two optional fields; `blut-types` stays
  wasm32-clean (both are plain serde types). Every PlanSpec producer (Starlark
  front-end, `from_erased_graph`) gains optional setters.
- `state.json` schema extends from `{epoch}` to `{epoch, step, data_cursor}`;
  the Rust recovery reader must tolerate old `state.json` (epoch-only) files —
  itself an additive, defaulted read.
- A stage advertising `checkpoint_cadence` now *owns* a correctness obligation:
  its checkpoint must be crash-consistent (atomic) or resume corrupts silently.
  The gate below exercises the kill-mid-epoch path to enforce this.
- Retry policy becomes load-bearing config: a `max_attempts` too high can mask a
  deterministic failure as flakiness. The `retry_on: Transient` classifier keeps
  deterministic crashes from being retried blindly.

## Implementation Plan  *(the contract — amend only by a dated Progress Log entry)*

- **Deliverable:** PlanSpec v1.1 (`RetryPolicy` + `timeout`, additive serde,
  hash-excluded); step-level `state.json`; a checkpoint-cadence contract honoured
  by one training stage end-to-end.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut planspec_v1_1_additive && cargo test -p blut
    mid_epoch_resume && pytest -k atomic_step_checkpoint`
  - `pass when:` (a) a v1 plan and the same plan re-serialized as v1.1 produce
    **identical node cache keys**; (b) a run killed at step K resumes at step K
    (not epoch start), reloaded state value-equal to a clean run; (c) a retry
    re-enters broker admission (asserted via the admission log).
- **Rollback:** the fields are `#[serde(default)]` and hash-excluded — dropping
  them leaves v1 plans and all cache keys unchanged; revert the `state.json`
  reader to epoch-only (it already tolerates the older schema).

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-10 — PART 1 of 3 LANDED (engine main `ed26b46`): PlanSpec v1.1 additive
  retry/timeout._ `SpecNode` gains optional `retry: RetryPolicy` + `timeout:
  StageTimeout`, both `#[serde(default, skip_serializing_if=Option::is_none)]`;
  the four retry types (`RetryPolicy`/`Backoff`/`RetryOn`/`StageTimeout`) now derive
  serde (Duration is serde-native). `CompiledPlan::apply_execution_overrides` threads
  them onto the executor's `PlanNode` (which already honours retry/timeout);
  `PlanSpec::compile` applies them after `from_erased_graph`. A `None` leaves the
  stage's own `RETRY`/`TIMEOUT` default — v1 plans are byte-identical.
  **Gate part 1 met:** `cargo test -p blut planspec_v1_1_additive` (exit 0) —
  v1 == v1.1-unset canonical bytes; unset fields omitted from the wire; setting
  retry/timeout leaves the node cache-key inputs (stage+args) byte-identical while
  the plan FINGERPRINT moves; compile threads them onto `PlanNode`. `PLAN_SPEC_VERSION`
  stays 1 (additive). 693 lib tests green; clippy `-D warnings` + fmt clean.
  **REMAINING (parts 2–3, own effort):** mid-epoch step-granular resume (extend
  `state.json` from epoch to `{epoch, step, data_cursor}` + the `mid_epoch_resume`
  test) and the python atomic step-checkpoint (`pytest -k atomic_step_checkpoint` in
  the domain cookbook's training loop). ADR stays In progress until all three gate parts pass._
- _2026-07-10 — PARTS 2–3 LANDED; ADR FROZEN. **Part 2** (engine main `417a125`):
  `ResumeState` gains optional `epoch`/`step`/`data_cursor` (`Option<u64>`,
  `#[serde(default)]`) after `heartbeat_unix`; derives `Default` so the
  `..Default::default()` shorthand carries the additive fields (three existing
  `executor_tests` state.json literals adopt it). A pre-0088 epoch-only marker still
  parses (absent step ⇒ epoch-granular fallback); `decide_resume` is unchanged.
  `cargo test -p blut mid_epoch_resume` = 1 passed; MiMo PASS (one non-blocking nit:
  `Default` widens the constructor surface — accepted, empty status⇒not-finished /
  empty run_id⇒Fresh is safe). **Part 3** (lamquant cookbook `[redacted: private repository]`):
  `CheckpointManager.save_step_recovery` / `load_step_recovery` — atomic step-checkpoint
  (write-temp → `f.flush()` → `os.fsync(fileno())` → `os.replace`), payload
  `{step, epoch, data_cursor, rng_state, optimizer, state_dict, provenance}`.
  `pytest -k atomic_step_checkpoint` = 2 passed (atomicity + value-equal kill-resume).
  MiMo follow-up `[redacted: private repository]` hardened two verified findings: a fail-loud provenance
  key-collision assert (kept flat-payload house style) and a `weights_only=True`-first
  load (fall back to False only for the optimizer/rng payload). Deferred-but-noted
  hardening: stale-`.tmp` cleanup, payload `schema_version`, a `data_cursor != step`
  test variant — none block the gate (dir is process-owned via broker/containment).
  All three gate parts pass ⇒ Status → Complete — frozen 2026-07-10._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut planspec_v1_1_additive && cargo test -p blut mid_epoch_resume && pytest -k atomic_step_checkpoint`
- **result:** All three gate parts green. (a) `planspec_v1_1_additive` — v1 == v1.1-unset canonical bytes, node cache-key inputs unchanged when retry/timeout set, `PLAN_SPEC_VERSION` stays 1 (part 1, engine `ed26b46`). (b) `mid_epoch_resume` (1 passed) — `ResumeState` round-trips `{epoch, step, data_cursor}`, a pre-0088 epoch-only marker parses with `step=None` (additive/back-compat), `decide_resume` byte-for-byte unchanged (part 2, engine `417a125`). (c) `pytest -k atomic_step_checkpoint` (2 passed) — `test_atomic_step_checkpoint` proves write-temp→fsync→os.replace atomicity (a `torch.save` failure at step 20 leaves the step-10 checkpoint intact, no orphaned `.tmp`), and `test_atomic_step_checkpoint_resume_is_value_equal` runs a tiny loop, 'crashes' at step 3, resumes from the step-checkpoint (rng+optimizer+data_cursor), and asserts value-equality with an uninterrupted run — the pass-when (b) mid-epoch-not-epoch-restart guarantee (part 3, cookbook `[redacted: private repository]`).
- **git_sha:** engine `417a125` (parts 1–2) · lamquant cookbook `[redacted: private repository]` (part 3)  ·  **artifact sha256:** n/a (source-only change)

## Related Decisions

ADR 0078 (PlanSpec IR + content-addressed cache keys — extended additively, not
superseded), ADR 0067 (T1.2 async-checkpoint primitive this cadence contract
builds on), ADR 0046/0047 (broker admission a retry must re-enter), the landed
durable-resume (crash-gated `state.json` + python atomic recovery — extended from
epoch to step).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: adding a `retry`/`timeout` field ever changes an existing node's cache
key; a mid-epoch resume replays from epoch start instead of the last durable
step; a step-checkpoint is observed torn (non-atomic) after a kill; or a retry
launches without passing broker admission. Any of these triggers a corrective
entry or a superseding ADR.
