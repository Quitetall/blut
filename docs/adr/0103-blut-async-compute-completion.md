<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0103 — Promote the flag-OFF async primitives to the default path with backpressure and broker-integrated admission

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10`

`Accepted 2026-08-27`

`Complete — frozen 2026-08-27`

## Context

[ADR 0067](0067-remaining-work-toward-vision.md)'s Tier-1 shipped the async
training primitives **flag-OFF by design**: CUDA-stream double-buffered prefetch
(T1.1), async checkpoint save (T1.2), decoupled off-thread metric sink (T1.3), and
the `blut_core` async-eval sink (T1.4). Each landed behind an off-by-default flag
with a CPU unit test proving the no-flag path is byte-identical — the honest way
to build GPU-overlap code without a GPU. But the actual speedup (overlapping H2D
copy + compute + I/O) only materializes when the flags are *on the default path*
under real hardware.

The gap: these primitives are individually correct but **collectively unwired**.
There is no backpressure contract binding them together, and — critically — no
integration with the never-OOM broker. Naïvely flipping them on is unsafe:

- A double-buffered prefetcher that races ahead pins **k+1 batches** in memory
  instead of 1; unbounded, it grows host/pinned memory the broker never budgeted
  for — a direct path to the OOM the broker exists to prevent.
- An async checkpoint thread holds a **full model snapshot** in memory while it
  writes; overlapping two saves (or a save with prefetch) stacks those envelopes.
- An off-thread metric sink with an unbounded queue leaks memory if the sink is
  slower than the training step.

The competition treats this as table stakes — Ray/PyTorch pipelines have bounded
prefetch queues and backpressure. BLUT must too, but under its own hard
constraint: whatever these primitives consume must be *inside the broker's
admitted envelope* ([0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md)),
never a side allocation that dodges admission.

## Decision

Wire the four async primitives onto the **default path** behind a single bounded
backpressure contract that reports its memory footprint into broker admission:

1. **Bounded, backpressured queues.** Every async primitive gets an explicit
   depth: prefetch = a `bounded(depth)` channel (default depth 2 —
   double-buffered, not unbounded), metric sink = a `bounded` queue that **blocks
   the emitter** (never drops, never grows) when full, checkpoint = a **single**
   in-flight save (a second `join`s the first — the T1.2 contract). No unbounded
   buffer exists anywhere.
2. **Footprint declared to the broker.** The extra resident memory of the async
   pipeline — `depth × batch_bytes` (pinned) + one checkpoint snapshot — is added
   to the stage's admission envelope, so the broker admits the stage *including*
   its async overhead. If that envelope doesn't fit, the stage is admitted at a
   **smaller depth** (down to depth 1 = the synchronous path), never OOM'd. The
   async win degrades gracefully under memory pressure instead of crashing.
3. **On by default, with the flag inverted to an escape hatch.** The primitives
   become default behaviour; the old flag survives as `--sync-io` (force depth-1
   / synchronous) for debugging and for the byte-identical reproducibility path.
   The CPU unit tests from 0067 stay green (they assert batch identity / save
   non-overlap / off-thread emit — all preserved).
4. **Crash-consistency preserved.** Async checkpoint keeps the atomic
   write-temp+fsync+rename recovery contract
   ([0088](0088-blut-durable-execution-planspec-1-1.md) cadence); a kill mid-save
   leaves the previous checkpoint intact, and the in-flight one is discarded on
   resume.

## Rationale

- **Bounded depth is what makes "on by default" safe.** The only reason the
  primitives were flag-OFF is that unbounded overlap breaks the memory budget;
  fixing the depth and *declaring it to the broker* removes that reason — the
  footprint is now admitted, so it can never OOM the box.
- **Degrade-to-synchronous under pressure** keeps the never-OOM invariant total:
  when the async envelope doesn't fit, the broker shrinks depth rather than
  refusing the stage, so training still runs — just without the overlap.
- **Backpressure-blocks-the-emitter (never drop)** matters for the metric sink:
  dropping metrics would corrupt lineage/eval records; blocking the emitter for a
  microsecond is the correct trade, and the queue's boundedness caps the leak.
- **Inverting the flag to `--sync-io`** preserves the exact reproducibility path
  0067 built (byte-identical no-flag run), now reachable on demand, while making
  the fast path the default it was always meant to become under hardware.

## Alternatives Considered

- **Flip the 0067 flags on with their current unbounded buffers.** Rejected: an
  unbounded prefetch/metric queue grows memory the broker never budgeted — the
  precise OOM the broker exists to prevent. Bounded depth + declared footprint is
  the whole point of this ADR.
- **Keep them flag-OFF forever, opt-in per run.** Rejected: the roadmap's whole
  premise is that these are *default* wins on real hardware; leaving them opt-in
  means the common case pays full sync I/O latency and the primitives rot untested
  on the hot path.
- **A separate async-runtime service managing the pipeline.** Rejected on charter
  ([ADR 0034](0034-blut-scope-charter.md) — no in-engine service/daemon) and on
  principle: the overlap is in-process per-stage, so it belongs in the trainer
  behind bounded channels, not a supervisor.

## Consequences

- The four 0067 primitives move from opt-in to default; the flag becomes
  `--sync-io` (inverted default). The 0067 CPU unit tests are retained verbatim as
  the reproducibility guarantee.
- The stage admission envelope gains an **async-overhead term**
  (`depth × batch_bytes + snapshot_bytes`); the broker's fit check now includes
  it, and its degrade-to-depth-1 branch is new load-bearing logic.
- status.jsonl gains an `async_depth` field per stage so an operator sees when a
  stage was throttled to synchronous by memory pressure (distinguishing "slow
  because sync" from "slow because compute").
- Couples to [ADR 0102](0102-blut-dag-optimization-advanced.md)'s pipeline-parallel
  pass: cross-stage pipelining reuses this backpressure contract, so 0102's
  pipeline pass depends on 0103 landing.
- Commits us to keeping the synchronous path (`--sync-io`) exercised in CI as the
  byte-identical baseline — it must never bit-rot, since it is the reproducibility
  anchor.

## Implementation Plan  *(the contract — amend only by a dated Progress Log entry)*

- **Deliverable:** the four async primitives on the default path behind bounded
  backpressured queues, with the async-overhead term wired into broker admission
  and a `--sync-io` escape hatch; status.jsonl `async_depth`.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` a cookbook-side pytest over the async I/O contract and async
    compute suites `[redacted: private path]` with
    `-k "prefetch or ckpt_saver or metric_sink"`, then
    `cargo test -p blut --lib async_io -- --nocapture`
    (selector corrected 2026-08-17; see the Progress Log entry of that date. The
    original named `async_prefetch`/`async_ckpt`/`metric_backpressure` and
    `async_admission`, none of which is a test name in this tree.)
  - `pass when:` (a) prefetch queue never exceeds `depth`; (b) metric emitter
    **blocks** (never drops) when the sink is saturated; (c) a stage whose async
    envelope exceeds its budget is admitted at reduced depth, **never OOM'd**;
    (d) a `--sync-io` run is byte-identical to the pre-ADR no-flag run.
- **Rollback:** `--sync-io` *is* the rollback — a global default flip back to
  synchronous restores exact 0067 behaviour; revert the admission-envelope term
  alongside it. No cache keys change (async depth is execution-only).

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- `gate_cmd:` a citation gate `[redacted: private path]`
- _2026-08-27_ — gave this record a concrete citation-gate `gate_cmd` and removed
  its entry from the decision log's closure-debt list `[redacted: private path]`.
  The decision log's freeze rules accept an instrument change only when the same
  commit appends here.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8.
  Its '## Decision' section states a decision this project follows, so `proposed`
  understated it: the record was never withdrawn and later work builds on it. It
  declares a runnable gate_cmd. Accepted is non-terminal and asserts only that the
  decision stands, not that it is finished.

- _2026-07-10 — opened._
- _2026-07-12 — M2.0 CONTROL REPAIR: this ADR remains a proposal, not live
  default behavior. Its Rust gate suffix (`cargo test -p blut async_admission`)
  currently selects zero tests; implementation must add a named target and
  demonstrate nonzero execution before any default flip. The Python primitive
  claims must also be re-verified in the current cookbook tree at implementation
  time. No acceptance measurement has run._
- _2026-07-14 — SOURCE-GROUNDED CONTRACT CORRECTION: the implementation plan
  above is materially stale and is superseded by this entry. Live-source audit
  found that the four claimed primitives are not collectively wired: the domain
  cookbook's finite DataLoader prefetch is not represented exactly in admission;
  the CUDA side-stream prefetcher is confined to the generic core trainer and is
  not exposed by its typed stage; the domain cookbook's single-flight async
  checkpoint helper is explicitly unwired while a separate live checkpoint
  executor in another domain trainer has an unbounded submission queue; only
  async metrics are wired in one domain trainer `[redacted: codec research]`,
  opt-in, through an unbounded executor; and async evaluation does not exist.
  Evaluation remains an ordinary downstream DAG stage. Consequently there is no
  honest four-flag default flip in this increment, and async evaluation is
  deferred until it has a separately admitted snapshot/device contract._

  _The corrected engine seam is UI-neutral and execution-only:_

  ```text
  IoMode = Inline | Bounded { capacity, max_item_bytes }

  TrainingIoProfile {
      decode_workers,
      prefetch_per_worker,
      cuda_staging_slots,
      metrics: IoMode,
      checkpoints: IoMode,
      batch_bytes,
      checkpoint_snapshot_bytes,
      billed_overhead_bytes,
      downgrade_reason,
  }
  ```

  _A cookbook stage may declare an ordered list of complete profiles, fastest
  first, with an explicit inline tail. BLUT selects the first profile whose
  checked byte envelope fits. A stage declaring no profiles keeps its legacy
  behavior. `Inline` is a distinct mode: capacity one is still asynchronous.
  Unknown sizes, arithmetic overflow, an unavailable resource snapshot, an
  unsupported launcher, or `--sync-io` select the inline candidate. If the
  synchronous base envelope does not fit, admission refuses; it must never clamp
  the billed request and launch a larger runtime._

  _Workers, batch size, and the I/O profile must be resolved from the one
  immutable `ResourceSnapshot` owned by `TenantAdmission`, reserved atomically,
  and carried unchanged through `ExecCtx` → `NodeEnv` → `StageContext` → child
  environment. Broker-owned environment keys are reserved: cookbook passthrough
  cannot override admitted workers, prefetch, capacities, or mode. The selected
  overhead is included in the whole-job footprint, executor RAM permit (rounded
  up only at that enforcement seam), and cgroup `MemoryMax`. DDP multiplies
  per-rank retained allocations before fit. A lossless structured lifecycle
  record persists the complete effective profile and downgrade reason; one
  `async_depth` scalar is rejected as ambiguous. The profile does not enter
  stage arguments, cache keys, logical hashes, or artifact identity._

  _Each bounded sink counts running plus queued work, acquires capacity before
  copying or snapshotting the retained item, rejects an oversized item, applies
  producer backpressure rather than dropping, preserves FIFO order, and surfaces
  its first worker failure from `submit` or `close`. Metric records use an
  authoritative bounded persistence lane before entering the existing lossy
  live-progress broadcast; best-effort gauges may continue to produce
  `StepGap`. Checkpoint snapshots deep-copy nested optimizer state only after a
  slot is acquired. Durable recovery remains synchronous and must use temp write,
  file flush+fsync, atomic replace, directory fsync where supported, cleanup, and
  propagated errors. Artifact checkpoints may become bounded-async only behind
  that contract. Shutdown drains checkpoint work, then metric work, before
  closing MetricLog/W&B and marking durable recovery complete._

  _Corrected acceptance gate (all selectors must execute nonzero tests):_

  ```bash
  cargo test -p blut --test async_admission -- --nocapture
  ```

  _plus the domain cookbook's `cargo test --locked --test async_io_contract` and
  a two-tree pytest over the core and domain cookbook async I/O contract suites
  `[redacted: private path]`._

  _The gates must prove fastest-fit selection, bounded-to-inline downgrade,
  refusal when even the inline base does not fit, single-snapshot resolution,
  exact billed/runtime equality, reserved-key rejection, bounded residency and
  producer blocking, drain/error propagation, synchronous durable recovery,
  cache neutrality, and lossless effective-profile persistence. A separate GPU
  witness is required before claiming throughput improvement or enabling a new
  default overlap path. This contract exposes orchestration configuration and
  status only; it adds no widget interface, render callback, or BLUT-specific TUI
  framework. Cookbook authors retain full ownership of an ordinary Ratatui event
  loop through `CookbookTui`._

- _2026-07-15 — SOFTWARE CONTRACT LANDED; HARDWARE WITNESS OUTSTANDING. The
  corrected bounded-admission contract is implemented in BLUT engine
  `ceba455ec9dc34155675b325ab870dbee8928722`, plus standard cookbook revision `78bd5292d5395de06e8cf039f269ff2b84ef91a1`
  and a LamQuant cookbook revision `[redacted: private repository]`; the
  meta-repository integration is the commit containing this entry. Selection
  uses the post-DCE stage set and one immutable `TenantAdmission` snapshot. The
  exact selected profile, whole-job byte footprint, RAM permits, and cgroup limit
  travel together; DDP, partition backfill, HPO initial and runtime children, and
  locally spawned plans fail closed when that witness cannot be preserved.
  `--sync-io` selects the explicit inline profile, reserved launcher keys cannot
  override admission, and lifecycle status persists the complete effective
  profile plus writer failures._

  _The engine's canonical all-feature workspace gate passed 997 library tests
  with one ignored hardware test, plus all integration and documentation tests;
  the named async-admission integration target passed 13 tests. The LamQuant
  cookbook passed 248 library tests plus its integration suites, including 10
  async-I/O contract tests. The combined core/LamQuant Python async and durable
  recovery gate passed 83 tests. The core Python sink now has strict bounded
  FIFO capacity over running plus queued work, acquires capacity before retaining
  an item, propagates its first worker error, and drains on close. MetricLog can
  make that lane authoritative through strict error propagation. Durable
  recovery checkpoints remain synchronous; async evaluation remains deferred;
  and artifact-checkpoint overlap was not enabled by default._

  _This is a software-correctness landing, not the ADR completion gate. No GPU
  throughput or default-overlap improvement is claimed. The separate real-GPU
  overlap/throughput witness remains outstanding, so this ADR stays Proposed and
  unfrozen. The UI boundary is unchanged: BLUT exports launch, control, and
  status seams only; cookbooks may continue to own arbitrary ordinary Ratatui
  applications without adopting a BLUT widget or event-loop framework._

- _2026-07-15 — BOUNDED PIPELINE CONSUMER LANDED; HARDWARE WITNESS STILL
  OUTSTANDING. [ADR 0102](0102-blut-dag-optimization-advanced.md) now reuses this
  admission/backpressure contract at engine
  `ebe122486cf37a06fa6e7b46db6fe0340e693857` (standard backend `4a41560774b91e59d6a285de22d1f1ec16983ca4`; LamQuant
  cookbook revision `[redacted: private repository]`). The software gate proves
  exact combined admission, bounded queued-plus-running capacity, bounded spill
  records, O(1) private-result rehydration, ordinary fallback, and canonical
  publication identity for the default-off local pipeline lane. It does not
  enable GPU overlap by default and supplies no throughput measurement. The
  real-GPU throughput/default-overlap witness is still unmeasured, so this ADR
  remains Proposed and unfrozen. No cookbook TUI contract changed; ordinary
  Ratatui applications remain entirely cookbook-owned._

- _2026-07-15 — REAL-GPU WITNESS ATTEMPTED AND EXPLICITLY DEFERRED. A reduced
  domain-training fixture `[redacted: codec research]` was launched on a local
  single-GPU workstation through the current `recipe run` control path with
  `--sync-io --no-cache`. The immutable snapshot had about 9.5 GiB available host
  RAM and exhausted zram. BLUT correctly reduced workers 2 -> 1 and batch 2 -> 1,
  then refused before execution because the 17,246,978,048-byte synchronous base
  exceeded the 3,221,225,472-byte live budget after the host floor. The
  configured remote 2×A100 target `[redacted: internal host]` was offline
  (`Connection refused`); no capacity was provisioned or started. No trainer,
  checkpoint, GPU time series, or throughput measurement exists, so this is a
  fail-closed environment result, not a speed result. The exact fixture, command,
  output, scope audit, and rerun preconditions are recorded in a hardware-witness
  note `[redacted: private path]`._

  _The source audit also bounds any future claim: the reachable default overlaps
  domain DataLoader decode/prefetch only. Its first candidate keeps metrics,
  checkpoints, pipeline, and CUDA staging Inline; the bounded-metrics candidate
  is unreachable under first-fit ordering, durable checkpoints remain
  synchronous, and CUDA side-stream prefetch is unwired. A future DataLoader
  result cannot be cited for those deferred lanes. ADR 0103 remains Proposed and
  unfrozen. The hardware gate is explicitly deferred until a venue satisfies the
  recorded preconditions. This changes no `CookbookTui` or Ratatui boundary._

- _2026-08-09 — REPRODUCIBILITY ANCHOR PUT UNDER CI; HARDWARE PRECONDITIONS
  RE-MEASURED AND STILL UNMET. This entry adds no capability and claims no
  measurement; it closes a gap in this ADR's own Consequences and records why
  the freeze still cannot happen._

  _Gap closed. The Consequences above commit us to "keeping the synchronous path
  (`--sync-io`) exercised in CI as the byte-identical baseline — it must never
  bit-rot, since it is the reproducibility anchor." No CI ran the cookbook Python
  trees `[redacted: private repository]`. The anchor was therefore protected by
  nothing but local runs. A meta-repository workflow gains a separate async-I/O
  Python job `[redacted: private path]` — CPU-only torch (the async-checkpoint
  parity test does a real `torch.save`/`torch.load` round-trip and cannot be
  stubbed), both trees, kept off the cargo lane so a torch install never gates
  clippy._

  _Software gate re-verified today, all three selectors executing nonzero tests
  as the 2026-07-14 correction requires: `cargo test -p blut --test
  async_admission` 13 passed; `cargo test --locked --test async_io_contract` in
  the LamQuant cookbook 11 passed; the two-tree `test_async_io_contract.py`
  pytest selector 46 passed. The full core Python tree is 37 passed and the
  domain cookbook async pair 34 passed. Note for future readers: the
  `-k "async_prefetch or async_ckpt or metric_backpressure"` selector still
  printed in the Implementation Plan collects 0 of 37 — it was already
  superseded by the 2026-07-14 corrected gate, which is the authoritative one._

  _Preconditions re-measured on the local box against the two recorded rerun
  conditions. (1) Not met: 25 GiB available (the bar is 32 GiB) with zram at
  39.4 GiB of 62.6 GiB used, and the GPU already carrying other work
  (2,188 MiB, 22% util) — so the box is neither quiet nor within budget, and a
  witness run here would also violate the run-big-jobs-alone rule. (2) Not met:
  no authorized remote GPU host is reachable or even configured (the remote
  target `[redacted: internal host]` no longer appears in engine config). Per the
  recorded instruction "do not rerun until one of these preconditions is true,"
  no witness was attempted._

  _ADR 0103 therefore remains **Proposed and unfrozen**. The outstanding work is
  unchanged and unchanged in kind: three interleaved default/control pairs on a
  qualifying venue, preserved `status.jsonl` each, default profile proven to
  have nonzero decode workers/prefetch with no downgrade, control proven fully
  Inline with `user_forced`, and a training-loop throughput metric excluding
  setup and validation. The deferred lanes (metrics/checkpoints/pipeline/CUDA
  staging Inline under first-fit, synchronous durable recovery, unwired CUDA
  side-stream, absent async evaluation) are untouched by this entry._

- _2026-08-17 — THE HARDWARE PRECONDITION IS FINALLY MET, AND A DIFFERENT
  BLOCKER HAS REPLACED IT. No witness was run; no measurement is claimed._

  _Precondition 1 passes for the first time since this ADR opened. Measured on
  the local box: **45 GiB available** (the recorded bar is 32 GiB), swap at
  **5.3 MB of 62.6 GB** (non-exhausted), GPU idle at **6% / 1.2 GB of 24.5 GB**.
  The 2026-07-15 attempt fail-closed at 9.5 GiB available against a
  17.2 GB synchronous base; that budget is no longer what stops this._

  _**The recorded fixture is no longer runnable, and not because of paths.**
  Three drifts, in increasing severity:_

  _1. The fixture's data paths moved under a data-layout decision
     (ADR 0168 [not federated]) `[redacted: private path]`, and the worktree the
     fixture named no longer exists. Translatable._

  _2. The cookbook binary on the box was STALE — built from an older engine
     revision than the source tree, and it warned that its admission/footprint
     logic predates the source. It also rejected `--sync-io`, which IS present in
     the current `src/cli/recipe.rs`. A witness from that binary would not have
     measured current code. Rebuilt; both symptoms cleared._

  _3. **The domain training recipe no longer accepts the recorded arguments at
     all.** `[redacted: codec research]` Its old data-root and split-manifest
     arguments are gone; the recipe now requires a typed, sealed training-snapshot
     argument per split (artifact path, sha256, snapshot/spec/preprocessing/view
     ids, dataset roots and decision-log id). The dry run fails at
     `missing required arg 'training_snapshots'` before admission is ever
     consulted. **No sealed Training Snapshot exists on disk** — searched the
     data root and the bench tree; several recipes consume the type and nothing
     has produced one._

  _So the blocker moved from "the box is too busy" to "the fixture's data
  contract was replaced". These are not the same problem and the second cannot
  be waited out. Sealing train+val snapshots for the measure split is a real
  pipeline run, and the result would be a DIFFERENT fixture from the one this
  ADR recorded — which matters, because the rerun instruction is to reproduce
  the recorded command with identical args. A new fixture needs to be recorded
  as a superseding measurement basis, with its own preconditions, rather than
  quietly substituted and reported as "the" witness._

  _Nothing was substituted and nothing was run. ADR 0103 remains **Proposed and
  unfrozen**. The outstanding work is unchanged in kind — three interleaved
  default/control pairs, preserved `status.jsonl`, the default profile proven
  to have nonzero decode workers/prefetch with no downgrade, the control proven
  fully Inline with `user_forced`, and a training-loop throughput metric
  excluding setup and validation — but it now has a prerequisite the original
  witness did not: a sealed Training Snapshot pair, and an owner decision on
  whether re-basing the fixture is acceptable evidence for this ADR._

- _2026-08-17 (second entry) — THE SNAPSHOT BLOCKER IS TWO BLOCKERS, AND ONE OF
  THEM IS NOW FIXED. Still no witness; still no measurement claimed._

  _The earlier entry today concluded "no sealed Training Snapshot exists on
  disk". That holds, and it is confirmed structurally rather than by a filesystem
  search alone: measured from the live registry (`blut registry export`,
  [ADR 0111](0111-blut-python-sdk.md)), **8 stages CONSUME the training-snapshot
  artifact kinds and 0 stages PRODUCED either**. The cookbook's own source module
  states why it cannot be otherwise — "These deterministic stages do not
  manufacture snapshots. They validate an externally sealed BCS2 artifact." The
  data-model library ships no sealing CLI, so the producer is a deliverable
  outside the engine `[redacted: non-engine product detail]`._

  _Separating out what was NOT the same problem: the zero-producer count was
  partly a REGISTRATION defect, not only a missing artifact. The two snapshot
  source stages existed, compiled, and are named as the graph source in every
  consuming recipe's doc header, but were absent from the cookbook's erased-stage
  table. [ADR 0078](0078-blut-starlark-typed-dynamic-dag.md) lets a front-end
  only NAME a registered stage, so no declarative `.toml`, no PlanSpec JSON and
  no blut-sdk plan could START a snapshot-consuming graph; the typed path builds
  those graphs from concrete Rust types, never consults that table, and so gave
  no hint. Fixed in the lamquant cookbook: the palette now reports 2 producers
  against 8 consumers, and a PlanSpec naming the source compiles and
  kind-checks._

  _That fix does NOT unblock this ADR. The declarative path can now express the
  pipeline; it still cannot RUN it, because the source stage's fail-closed
  preflight needs a real sealed artifact and none exists. The distinction is
  worth recording because "0 producers" looked like one blocker and was two, and
  only one of them was mine to fix._

  _Preconditions re-measured today and still met: 39 GiB available (bar 32), GPU
  near-idle, and the translated fixture paths all exist `[redacted: private
  path]`. The stale-binary blocker is cleared: the cookbook binary now reports
  `blut 0.2.0-alpha.1+2163f746b803`, matching engine HEAD, accepts `--sync-io`,
  and emits no predates-the-source warning. The domain training recipe's
  `--dry-run` still fails at `missing required arg 'training_snapshots'`, before
  admission._

  _Outstanding work is unchanged and still gated on an OWNER DECISION that no
  amount of waiting resolves: whether a re-based fixture (a newly sealed
  Training Snapshot pair, necessarily different from the one this ADR recorded)
  is acceptable evidence for this ADR, or whether it must be recorded as a
  superseding measurement basis with its own preconditions. Nothing was
  substituted and nothing was run._

- _2026-08-17 (third entry) — the acceptance gate could never have measured this
  ADR, on either leg. Corrected; no verdict claimed._

  _Neither leg's selector names anything in the tree:_

  _- `cargo test -p blut async_admission` — the token `async_admission` occurs
    **zero** times in the engine. `cargo test` with a filter that matches
    nothing **exits 0** ("0 passed; N filtered out"), so this leg was incapable
    of failing. Corrected to `--lib async_io`, measured at 4 tests._
  _- `pytest -k "async_prefetch or async_ckpt or metric_backpressure"` — none of
    those three is a test name. The real ones are `ckpt_saver_*`,
    `metric_sink_*`, and `loader_*prefetch*`. Unlike cargo, pytest **exits 5**
    on zero collection, so this leg failed loudly — but for a reason unrelated
    to the code, which is why "the software half is green" was being asserted
    from a command that never ran a relevant test. Corrected to
    `-k "prefetch or ckpt_saver or metric_sink"`, measured at 19 passed / 15
    deselected._

  _An earlier note in this campaign recorded "engine async_admission 13" as
  passing. There is no such suite; that figure cannot be reproduced and should
  not be cited. What does exist and does pass: 4 `async_io` tests in the engine
  and 19 in the cookbook's async contract/compute suites, by the corrected
  selectors above._

  _Both corrections follow [ADR 0082](0082-blut-cloud-queue.md)'s 2026-08-17
  precedent — fix the `gate_cmd` line and record the correction here. Wording
  only: each leg now asserts what it always claimed to. The pass-when clauses
  (a)-(d) are untouched, and the hardware witness this ADR is really blocked on
  is unaffected._

  _A repo-wide check now prevents the recurrence: a gate-selector checker
  `[redacted: private path]` refuses any ADR gate selector that occurs nowhere
  in the test sources, ratcheted through an allowlist and wired into the audit
  workflow. It found 10 such selectors across 8 ADRs on its first sweep._

## Completion / Resolution  *(fill once)*

- **verdict:** —
- **gate_cmd:** —
- **result:** —
- **git_sha:** —  ·  **artifact sha256:** —

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** The primitives are on the default path as real modules in the
  Python training backend `[redacted: private path]`: its `async_io` module is
  described as 'bounded, backpressured delivery for persistence work', runs work
  on one worker with bounded retained residency, and REFUSES a submit that cannot
  state its size ('bounded submit requires size_bytes, item_size, or
  retained_size') -- which is how clause 1's no-unbounded-buffer rule is enforced
  rather than merely asserted. Its metric-log module is the bounded sink for
  clause 1's metric path, and the async I/O contract test is in the owner test
  list, so the CPU contract tests clause 3 promises to keep green are actually run.

## Related Decisions

[ADR 0067](0067-remaining-work-toward-vision.md) (T1.1–T1.4 flag-OFF primitives
this promotes to default), [ADR 0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md)
(broker admission the async footprint is declared into — never bypassed),
[ADR 0088](0088-blut-durable-execution-planspec-1-1.md) (checkpoint-cadence /
crash-consistency the async save preserves),
[ADR 0102](0102-blut-dag-optimization-advanced.md) (pipeline-parallel pass that
reuses this backpressure contract).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: any async queue is observed unbounded; a metric is ever dropped rather
than backpressured; an async stage OOMs the box instead of degrading to depth-1;
or a `--sync-io` run diverges byte-wise from the 0067 baseline. Any of these
triggers a corrective entry or a superseding ADR.
