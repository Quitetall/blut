<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0102 — Extend dag_opt beyond schedule-hints/DCE with stage fusion, speculative execution, pipeline parallelism, and cache/priority scheduling

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-15`
(engine `ebe122486cf37a06fa6e7b46db6fe0340e693857`)

## Context

`dag_opt.rs` today does two proven passes: **dead-code elimination** (drop nodes
whose outputs are unused, renumber to a dense index) and **schedule hints**
(annotate nodes for the executor's ordering), both landed under ADR 0067 and
gated by the DCE↔hint invariant (T1.5). That is the *first* optimization tier —
correctness-preserving pruning and ordering.

Higher-value passes are unbuilt. The competition treats the DAG as a compilation
target: Ray fuses chained tasks and pipelines stages across the object store;
Dagster/Flyte cache-aware-schedule to prefer nodes whose upstreams are already
materialized. BLUT leaves this on the table — a two-node chain that could run in
one process still pays a full stage boundary; a node whose inputs are cache-hot
is scheduled no earlier than a cold one; there is no priority notion, so a
latency-critical eval waits behind bulk ingest.

This is Vision Phase 3. It rides three non-negotiable rails:
- **content-addressed cache keys stay unchanged (0078)** — an optimization may
  reorder or merge *execution*, but the artifact a node produces must hash
  identically to the un-optimized run, or the cache and reproducibility break;
- **every launch still passes broker admission (0046/0047)** — fusion and
  speculation change *what* runs, never *whether the box can afford it*;
- **speculation must be side-effect-safe** — a speculatively-run node that is
  later discarded must leave no observable trace beyond its (cache-safe) artifact.

## Decision

Add an **advanced-optimization tier** to `dag_opt.rs`, each pass off by default
behind a flag, each preserving per-node output hashes:

1. **Stage fusion.** Merge a linear chain of nodes with no external consumers of
   the intermediate into a single executor stage (one process, one admission),
   eliminating the serialize/spawn boundary between them. The **fused stage's
   cache key is derived from the fused chain's inputs and is required to equal
   the key of the chain's terminal node** — fusion is an execution optimization,
   never a new artifact identity.
2. **Speculative execution.** For nodes gated on a not-yet-resolved branch
   condition, optionally pre-run the *cache-safe, side-effect-free* candidate on
   spare (broker-admitted) capacity; keep its content-addressed artifact if the
   branch is taken, discard the reservation otherwise. Only nodes flagged
   `pure`/idempotent are eligible — a stage with side effects is never
   speculated.
3. **Pipeline parallelism.** For a producer→consumer pair over a chunked/
   partitioned stream, overlap the consumer's early chunks with the producer's
   later ones (bounded by the async-primitive backpressure of ADR 0103), rather
   than materializing the whole intermediate first.
4. **Cache-aware + priority scheduling.** The scheduler's ready-queue orders by
   (a) **cache-hot upstreams first** (prefer nodes whose inputs are already
   materialized — cheapest path to progress) then (b) a per-node `priority`
   (additive PlanSpec field) so a latency-critical stage jumps bulk work — all
   still subject to broker admission, which is the hard gate.

Each pass runs *after* DCE (on the dense-renumbered plan, preserving the 0067
invariant) and emits a verifiable equivalence witness (below).

## Rationale

- **Hash-preserving fusion is the safe form of the big win.** The spawn/serialize
  boundary between chained stages is pure overhead when nothing else reads the
  intermediate; fusing removes it while the required key-equality check keeps the
  produced artifact byte-identical, so the cache and reproducibility are intact.
- **Speculation only on pure nodes** is the only charter-safe form: a
  discarded speculative run of a side-effecting stage would leave observable
  state; restricting to idempotent, content-addressed nodes means a discard is a
  no-op beyond a wasted (admitted) reservation.
- **Cache-aware ordering is free ROI** — preferring materialized-upstream nodes
  shortens critical path with zero correctness cost, since order among
  ready nodes is already a scheduler degree of freedom (the DCE↔hint invariant
  proves reordering is sound).
- **Priority under the broker, not around it.** Priority reorders the ready
  queue; it never grants a stage resources the broker would refuse — the never-OOM
  invariant is untouched.

## Alternatives Considered

- **Aggressive fusion that rewrites intermediates (fuse and drop the terminal
  node's separate artifact).** Rejected: changing what artifact a node produces
  changes its hash, invalidating the cache and breaking reproducibility against
  the un-fused run — a direct 0078 violation. Fusion must be hash-preserving.
- **Speculate all branch candidates (including impure stages), roll back on
  discard.** Rejected: rollback of side effects is exactly the fragile,
  hard-to-verify machinery the content-addressed model exists to avoid; restrict
  to pure nodes and discard is trivially safe.
- **Defer the whole tier and keep only DCE/hints (do nothing).** Rejected as the
  status quo that leaves the Phase-3 speedups unrealized; but each pass ships
  independently behind a flag, so the tier is incremental, not big-bang.

## Consequences

- `dag_opt.rs` grows a pass registry; each pass is a flag-gated
  `Plan → Plan` transform emitting an equivalence witness. Default-off keeps the
  optimized plan identical to today's DCE+hints output.
- A `priority: Option<i32>` and a `pure: bool` node attribute become additive
  PlanSpec fields (defaulted; excluded from the cache key — they are scheduling
  metadata, not computation).
- Commits us to a **per-pass equivalence check**: every advanced pass must prove
  its output plan produces the same per-node artifact hashes as the un-optimized
  plan on the gate fixtures, or it does not ship.
- Pipeline parallelism couples to ADR 0103's backpressure; without it, overlap
  could outrun the broker's memory envelope — so 0102's pipeline pass is gated on
  0103 landing.

## Implementation Plan  *(the contract — amend only by a dated Progress Log entry)*

- **Deliverable:** the `dag_opt` advanced pass registry (fusion, speculation,
  pipeline, cache-aware+priority scheduling), each flag-gated, each with an
  equivalence witness; additive `priority`/`pure` PlanSpec fields.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut dag_opt_advanced -- --nocapture` (for a
    fixture DAG: fused chain's terminal artifact hash == un-fused hash;
    speculated-then-discarded pure node leaves no state; priority reorders the
    ready queue but never bypasses admission; cache-hot node scheduled before a
    cold sibling; the DCE↔hint invariant from 0067 still holds post-pass)
  - `pass when:` exit 0 **and** every pass's optimized plan yields per-node
    artifact hashes identical to the un-optimized plan.
- **Rollback:** each pass is a self-contained flag defaulting off; revert the
  flag and the plan reverts to DCE+hints (0067). The PlanSpec fields default and
  are hash-excluded, so removing them is cache-neutral.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-10 — INCREMENT 1 LANDED (engine main `7b51950`): pass #4's
  **user-priority scheduling** (the tier is flag-gated & incremental per the
  Decision, so it ships pass-by-pass, not big-bang)._ Additive PlanSpec field
  `SpecNode.priority: Option<i32>` (`#[serde(default, skip_serializing_if)]`),
  threaded onto `PlanNode.priority` by `apply_execution_overrides` (now a named
  `ExecutionOverrides` struct) — the same cache-safety contract as 0088's
  retry/timeout: scheduling metadata, NOT a node cache-key input; an unset plan's
  canonical bytes are byte-identical to pre-0102. New `ScheduleHint.user_priority`
  populated by a flag-gated `DagOptimizer.priority_aware` pass (default OFF); the
  executor's `next_ready` key becomes `(user_priority, critical_path_len,
  Reverse(id))`, which reduces to the historical `(critical_path_len, Reverse(id))`
  when the pass is off — byte-identical default schedule. Sound by the 0067
  DCE↔hint invariant (reordering ready nodes never changes which run or their
  outputs) and 0078 (priority is metadata; per-node artifact hashes preserved).
  **Sub-gate met:** `cargo test -p blut dag_opt_priority` = 7 passed (pass
  sets/neutralises priority; off is byte-identical; node-set/args/canon-bytes/edges
  preserved = equivalence witness; priority reorders the ready queue; zero-priority
  falls back byte-identically; a non-ready high-priority node is never selected
  ⇒ admission not bypassed; PlanSpec field additive+cache-safe). 857 lib green;
  clippy+fmt+scope_guard clean. MiMo PASS WITH NITS → applied the
  tuple→`ExecutionOverrides` struct nit; verified the plan-fingerprint move feeds
  only provenance (lineage), never an execution cache key.
  **REMAINING (own increments, each its own flag + equivalence witness):** stage
  fusion (fused-stage executor path + key-equality check), speculative execution
  (a `pure` flag + spare-capacity run/discard), pipeline parallelism (gated on
  ADR 0103 backpressure), and wiring the already-computed `cache_warm` hint into
  ready-queue ordering. ADR stays In progress until the tier's full gate
  (`cargo test -p blut dag_opt_advanced`) passes._
- _2026-07-12 — M2.0 CONTROL/GATE REPAIR (engine main `058c232`)._ The original
  `cargo test -p blut dag_opt_advanced -- --nocapture` command was a substring
  filter that executed **zero tests** and exited 0. The runnable progress gate
  is now `cargo test -p blut --test dag_opt_advanced -- --nocapture` (2 passed):
  it proves priority-hint propagation/default-off neutrality and executes two
  simultaneously-ready nodes with `max_in_flight=1`, proving higher user
  priority controls actual executor order when the pass is enabled. This does
  not complete the ADR. `cache_warm` is currently a determinism proxy and is not
  in `next_ready`; real cache-warm ordering, fusion, speculation, and pipeline
  parallelism remain unbuilt. ADR stays In progress._
- _2026-07-13 — INCREMENT 2 LANDED (engine main `bc30296`): **live
  cache-aware ready ordering**._ This corrects the 2026-07-12 implementation
  state above: the static `deterministic()` warmth proxy is removed.
  `DagOptimizer.cache_aware` now defaults OFF; when enabled, the parallel
  executor waits until a node is genuinely ready, derives the exact
  ADR-0078/0101 input-dependent key through the same `node_cache_key` function
  used by `build_task`, and probes the run's live tenant-scoped cache. Ready
  order is `(cache_warm, user_priority, critical_path_len, Reverse(id))`, so a
  decodable hit outranks cold work while the disabled path retains historical
  ordering. `--no-cache`/force-recompute performs no scheduling probe. Probes
  are exact-key memoized, bounded to eight concurrent lookups, and moved to
  `spawn_blocking`; prepared hits flow directly into `run_node` and never enter
  P2P dispatch. Exact-key completion invalidates a prior miss, closing the
  scheduler-miss/run-hit race without re-probing unrelated keys. Deadline and
  cancellation checks bracket the probe batch, post-processing, and both the
  P2P-submit and local-spawn boundaries, so slow stores cannot launch work
  after the plan stops. **Sub-gate met:**
  `cargo test -p blut --test dag_opt_advanced -- --nocapture` = 11 passed,
  covering default parity, real root/downstream hits, precedence, every-node
  hash equivalence, bypass, bounded cold probes, miss→hit invalidation,
  runtime responsiveness, and no `StageBegin` after a probe crosses the plan
  deadline. The full all-feature workspace suite, strict all-feature clippy,
  fmt, both WASM targets, scope guard, diff check, and cross-repo partition
  acceptance test are green; independent Standards and Spec reviews returned
  PASS; mandatory `review_commit` returned PASS WITH NITS after verification.
  A non-blocking performance follow-up is to retain prepared decoded
  descriptors only for the launch window if custom cookbooks begin embedding
  unusually large payloads in `ErasedArtifact`; production artifacts are
  descriptor-shaped, the feature is opt-in, and lookup concurrency is bounded.
  **REMAINING:** stage fusion, speculative execution, and pipeline parallelism
  (still gated on ADR 0103). ADR stays In progress._
- _2026-07-13 — INCREMENT 3a LANDED (engine main `249eced`): **conservative
  whole-plan deterministic linear coalescing.** `DagOptimizer.stage_fusion`
  defaults OFF. When enabled, an exact whole-plan chain of at least two
  ordinary deterministic stages is executed by one Tokio/executor task with
  one lazily acquired shared admission envelope. Admission is acquired only at
  the first real cache miss, so an all-cache-hit chain acquires no broker
  permit. Every eligible stage must declare the same CPU/GPU/memory/network/
  disk envelope; branches, expansions, non-deterministic or advisory stages,
  hand-written `StageDyn` implementations without typed-handoff support,
  runtime control, P2P execution, cache-aware scheduling, and heterogeneous
  admission envelopes retain the ordinary executor path. The hidden
  `InProcessArtifact` seam lets normal typed `Stage` implementations pass an
  intermediate directly on the successful consecutive-cache-miss fast path,
  eliminating that inter-stage bincode decode while preserving the public
  cookbook surface. A cache hit followed by a miss, or a retry after the typed
  value is consumed, deliberately decodes the authoritative erased artifact.
  Each original node still performs its normal task construction, stage run,
  FW-2 promotion, metadata write, content-hash/cache-key derivation, and cache
  insert; its serialized artifact and cache identity remain authoritative.
  Failure, panic, and cancellation outcomes retain ordinary-path parity. The
  normal retry policy still applies, but the identical shared admission is
  intentionally held across fused stage boundaries and retry/backoff instead
  of following the ordinary per-node lease lifetime. **Sub-gate met:**
  `cargo test -p blut --test dag_opt_advanced -- --nocapture` = 18
  passed, covering default-off parity; byte-equal terminal output and equal
  per-node hashes/keys; one task versus three; zero consecutive-miss fused
  handoff decodes; continuous and all-hit-zero admission; conservative
  fallbacks; and panic, failure, and cancellation parity. The full all-target/
  all-feature workspace suite, strict clippy, fmt, both WASM targets, scope
  guard, diff check, and cross-repo partition acceptance test are green.
  Independent Standards and Spec reviews returned PASS; mandatory post-commit
  review returned PASS WITH NITS, with no verified defect. This is deliberately
  increment **3a**, not
  completion of the Decision's general stage-fusion pass. **REMAINING:** an
  internal-subchain `Plan → Plan` fused-node representation, literal
  subprocess fusion, heterogeneous-envelope policy, speculative execution,
  and pipeline parallelism (still gated on ADR 0103). ADR stays In progress._
- _2026-07-14 — INCREMENT 3b LANDED (engine commits `4f2b566`, `8cbc21e`,
  `514b82e`; engine main `514b82e`): **optimizer-owned internal linear-subchain
  fusion.** `stage_fusion` remains default OFF. After DCE, the optimizer records
  maximal eligible linear candidates in `CompiledPlan::fused_subchains`; stale
  witnesses are cleared before optimization, and the executor derives every
  whole-plan or internal runtime fused group solely from that witness. This
  generalizes 3a to eligible internal subchains in branched plans without
  rewriting semantic nodes, edges, artifacts, hashes, or cache keys. Eligibility
  remains conservative: ordinary deterministic, non-advisory typed `Stage`
  implementations with in-process handoff. Cache-aware scheduling,
  user-priority scheduling, runtime control, P2P, and expansions retain ordinary
  execution boundaries. Candidates split into contiguous equal-
  `AdmissionRequest` groups, with singleton fragments falling back to ordinary
  execution. Each group uses one lazy shared admission and direct typed handoffs
  while still emitting ordinary per-node outcomes, artifacts, metadata, hashes,
  keys, cache entries, and events. Deadline/cancellation is rechecked at every
  fused boundary; panic/failure classification and duplicate-key single-flight
  semantics are preserved. The increment also repairs post-DCE dense node IDs
  when a middle node is eliminated. **Sub-gate met:**
  `cargo test -p blut --test dag_opt_advanced -- --nocapture` = 24 passed,
  including internal branched fusion, optimizer witness/default-off parity,
  middle-hole DCE, deadline boundary, cache single-flight, heterogeneous-envelope
  splitting, and the prior 3a/cache/priority witnesses. Full workspace
  all-target/all-feature tests, strict clippy, fmt, both WASM targets, scope
  guard, diff check, and cross-repo partition pytest are green. Independent
  Standards, Spec, and runtime reviews returned PASS. Mandatory reviews for the
  three split commits returned PASS WITH NITS, PASS after a verified
  false-positive re-review, and PASS WITH NITS; the remaining nits are defensive
  test/assertion hygiene, not live defects. **REMAINING:** an explicit
  conservative policy/capability for literal stage-owned subprocess boundaries;
  additive purity and conditional-branch IR followed by speculative execution;
  ADR 0103 bounded async/backpressure; then pipeline parallelism. ADR remains In
  progress._
- _2026-07-14 — SPECULATION CONTRACT CORRECTED BEFORE IMPLEMENTATION._ The
  original Decision did not define who may certify purity, how a branch
  condition differs from a typed data edge, what an untaken discard may expose,
  or when capacity is genuinely spare. The fail-closed implementation contract
  is now: (1) additive PlanSpec `pure: bool` remains default false and
  cache-key-neutral, but it is only an author request; eligibility additionally
  requires a stage-owned speculation-safe capability whose default is false, so
  raw JSON cannot declare an opaque side-effecting stage safe; (2) PlanSpec gains
  a separate boolean condition-gate relation whose selector emits the closed
  engine `BranchDecision` artifact. It participates in cycle checking, DCE
  liveness, dense-ID remapping, and plan identity, but never typed input gathering
  or a node cache key; (3) the first slice rejects reconvergent/phi-shaped
  conditional graphs rather than guessing how a missing arm enters a merge. A
  false gate prunes only its target's exclusive data descendants; (4) a
  speculative attempt runs in a private scratch job/cache/status context. Until
  selection it creates no normal lifecycle event, final stage directory, cache
  entry/proof, or lineage row. Selection publishes the prepared result once;
  rejection/cancellation deletes scratch; and (5) spare means ordinary ready work
  was offered first, a `max_in_flight` slot remains, and the candidate's complete
  CPU/GPU/memory/network/disk envelope can be acquired immediately with a
  non-blocking try-only lease. Failure to acquire never waits and never fails the
  plan; when selected, the node falls back to ordinary admission/execution. The
  optimizer remains the sole owner of post-DCE speculation witnesses, and the
  executor may only refine or decline them. This correction narrows the original
  promise; it does not mark the pass landed or freeze the ADR._
- _2026-07-14 — INCREMENT 4a LANDED (engine code `95b923b`, API follow-up
  `5c71983`): **stage-certified purity and executable conditional control IR.**
  `SpecNode.pure` is additive, default false, omitted from legacy canonical
  bytes, and cache-key-neutral. Compilation accepts it only when the registered
  stage independently declares deterministic output and
  `Stage::SPECULATION_SAFE = true`; map-template purity is rejected in v1 rather
  than silently discarded. `BranchDecision` is a closed typed artifact and
  `ConditionGateSpec` is a separate control relation: it participates in plan
  identity, cycle checks, topology, graph persistence, DCE liveness, component
  remapping, and dense-ID repair, but never typed input gathering or a node cache
  key. V1 accepts one gated, non-reconvergent terminal branch and rejects maps,
  duplicate data/control wiring, invalid kinds, and advisory selectors. The
  parallel executor withholds the target until the decision resolves; a losing
  target and its data descendants emit `StagePruned`, create no normal stage
  lifecycle/cache materialization, and appear as neutral `not_selected` states
  in graph, console, and HPO accounting. Root-input seeds are removed on every
  prune path. Conditional final output remains pinned to the validated data
  terminal even when runtime control appends work, and advisory fallback may
  select only completed data ancestors, never the selector artifact. Fusion
  fails closed for conditional plans. The public API documents the additive IR;
  no `Cookbook` or `CookbookTui` surface changed. **Sub-gates met:** the public
  conditional integration target ran 5 passed + 1 intentional subprocess-entry
  ignore; the advanced DAG target remained 24 passed. The workspace
  all-target/all-feature run, strict clippy, fmt, both WASM targets, scope guard,
  diff check, and cross-repo partition pytest were green. Independent source
  review found no verified defect. Mandatory code and API reviews returned PASS
  WITH NITS and recommended merge; the notes concern v2/malformed-manual-stage
  defense-in-depth and prose clarity, not a live typed-stage blocker.
  **REMAINING:** literal subprocess-boundary capability/policy; the separate
  default-off speculation pass and discard-equivalence witness; ADR 0103 bounded
  async/backpressure; then pipeline parallelism. ADR remains In progress._
- _2026-07-14 — INCREMENT 4b LANDED (engine `3c16c33`): **fail-closed execution
  boundaries for fusion.** Public, non-exhaustive `StageExecutionBoundary`
  classifies stage work as `Opaque`, `InProcess`, or `Subprocess`. Both typed
  `Stage` and manual `StageDyn` implementations default to `Opaque`; therefore
  existing and third-party cookbook stages continue to execute normally but
  cannot be silently folded across an admission/task/lifecycle boundary. The
  fusion pass now requires an explicit `InProcess` declaration in addition to
  determinism, non-advisory behavior, topology/resource equality, and typed
  handoff support. Shipped checks and the P2P echo opt into `InProcess`; the two
  external connector transfer stages declare `Subprocess`, while their manifest
  source is `InProcess`. The engine never guesses from names, arguments, or UI
  implementation. Two five-node witnesses prove eligible groups still fuse on
  each side of both an unclassified stage and a declared subprocess without
  crossing either boundary. The advanced DAG target now runs 26 passed tests;
  the all-feature workspace run reports 967 passed library tests plus one
  intentional ignore, with integrations and benches green. Strict clippy, fmt,
  both WASM checks, scope guard, diff check, and cross-repo partition acceptance
  are green. Independent source review found no blocker; the mandatory commit
  review returned PASS and recommended merge. The public API explains that the
  declaration affects optimizer policy only; no `Cookbook` or `CookbookTui`
  surface changed. **REMAINING:** the separate default-off private-scratch
  speculation pass and discard witness; corrected ADR 0103 bounded
  async/backpressure; then pipeline parallelism and the freeze gate. ADR remains
  In progress._
- _2026-07-14 — INCREMENT 4c LANDED (engine `8b67cec`): **default-off,
  private-attempt speculative execution.** `DagOptimizer.speculative_execution`
  defaults false and only consumes optimizer-owned post-DCE witnesses. A node
  is eligible only when the author marks it pure, its registered stage is
  deterministic, and that stage independently declares
  `SPECULATION_SAFE = true`. The executor may start one same-key attempt only
  in a private job/cache/status/scratch context, and only after a try-only
  all-envelope admission proves the capacity is genuinely spare after every
  ordinary in-flight demand is shadow-reserved. Ordinary work always wins;
  shared-provider presence suppresses speculative I/O; rejection, cancellation,
  deadline, panic, and supersession never fail or delay the canonical path.
  Selection publishes the exact canonical artifact, hash, cache key, lifecycle,
  and status identities; discard leaves those identities byte-for-byte aligned
  with speculation disabled. Scratch cleanup is checked on setup failure,
  rejection, panic, deadline, unrelated terminal error, retained states, and
  late completion. Publication is panic-contained, and rollback failure is an
  inspectable `SpeculationRollback` node failure rather than an invisible
  partial promotion. Exact step replay preserves resource gap accounting, while
  same-key single-flight prevents duplicate speculative ownership. The public
  API documents the additive `DagOptimizer` field and struct-update migration;
  no `Cookbook` or `CookbookTui` surface changed. **Sub-gates met:** the advanced
  DAG target ran 27 passed tests; the conditional/speculation integration target
  ran 27 passed plus one intentional subprocess-entry ignore. The workspace
  all-target/all-feature run reported 978 passed library tests plus one
  intentional ignore, with every integration and bench target green. Strict
  clippy, fmt, both WASM checks, scope guard, diff check, first-cookbook cold/warm
  cache acceptance, and cross-repo partition pytest were green. Independent
  standards and specification reviews found no release blocker. The mandatory
  commit review returned PASS after two scheduler-ledger findings were verified
  against source as false positives. **REMAINING:** correct and land ADR 0103's
  bounded async-I/O admission/backpressure, then bounded pipeline parallelism
  and the full ADR 0102 freeze gate. ADR remains In progress._
- _2026-07-15 — INCREMENT 5 LANDED AND ADR FROZEN (engine
  `ebe122486cf37a06fa6e7b46db6fe0340e693857`; standard backend `4a41560774b91e59d6a285de22d1f1ec16983ca4`; LamQuant
  cookbook revision `[redacted: private repository]`): **default-off bounded
  `map_output` pipeline parallelism.** The optimizer may certify one
  deterministic `InProcess` list producer and one sequential private consumer.
  The producer supplies a manifest of element content hashes and emits through
  an executor-owned lane whose capacity counts queued plus running items. Exact
  parent-plus-child admission includes `(capacity + 2) × max_item_bytes`; both
  input and private-result spills are bounded and decoded with fixed-int,
  trailing-byte rejection, and a hard limit. Private results are validated and
  published one at a time, so extra resident payload remains O(1), while the
  combined admission lease is held through canonical publication._

  _The final authoritative `ListOf` must match manifest cardinality, order,
  kind, and hashes before any private child becomes visible. Every optional
  decline, invalid manifest, warm/shared cache, late admission refusal, spill
  corruption, oversize record, parent failure, cancellation, or publication
  failure either follows ordinary execution or terminates with checked cleanup;
  predicted single-flight keys are explicitly abandoned and waiters awakened.
  Successful publication preserves ordinary node IDs, logical hashes, cache
  keys, artifacts, lifecycle events, and status. V1 is deliberately local-only
  and excludes GPU, retry, control, P2P, advisory, and multi-consumer cases. It
  adds no PlanSpec field and changes no `Cookbook` or `CookbookTui` surface.
  Cookbook authors continue to own arbitrary ordinary Ratatui applications;
  BLUT supplies only discovery, launch, status, and control seams._

  _Terminal gates were nonzero and green: `dag_opt_advanced` passed 42 tests;
  conditional execution passed 27 with one intentional subprocess-entry
  ignore; default-feature `async_admission` passed 12 and the all-feature run
  passed its P2P-gated case as well. The locked all-feature engine workspace ran
  1,018 passed library tests with one intentional ignore plus every integration
  and documentation target. Strict all-target/all-feature Clippy, all-target
  compile, no-default library check, both WASM targets, fmt, scope guard, and
  diff checks passed. The first-cookbook cold/warm witness produced 3 misses
  then 3 hits, and the cross-repository partition acceptance test passed. The
  standard backend passed 93 tests and strict Clippy. The LamQuant cookbook
  passed 279 tests with one intentional ignore, including 10 async-I/O contract
  tests, strict Clippy, locked metadata, and single-revision dependency checks.
  Mandatory commit reviews returned PASS after every reported concern was
  checked against source and either fixed or recorded as a false positive. The
  full advanced tier now satisfies the corrected acceptance gate and is frozen.
  ADR 0103's separate real-GPU throughput/default-overlap witness remains
  outstanding and is not implied by this software result._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test --locked --test dag_opt_advanced -- --nocapture`;
  `cargo test --locked --test conditional_execution -- --nocapture`;
  `cargo test --locked --workspace --all-features`; plus the full engine,
  backend, cookbook, first-cookbook, and cross-repository gates listed in the
  final Progress Log entry
- **result:** all five default-off advanced passes retained ordinary artifact,
  hash, cache, admission, cleanup, and scheduler identity; the bounded pipeline
  lane passed capacity, fallback, cancellation, spill, publication, and
  default-parity witnesses
- **git_sha:** engine `ebe122486cf37a06fa6e7b46db6fe0340e693857` · standard backend
  `4a41560774b91e59d6a285de22d1f1ec16983ca4` · LamQuant cookbook `[redacted: private repository]` · **artifact sha256:** n/a
  (source-only change)

## Related Decisions

ADR 0067 (the landed DCE + schedule-hints tier this extends, and its DCE↔hint
invariant), ADR 0078 (content-addressed cache keys every pass must preserve),
ADR 0087 (GPU-aware scheduling — priority/placement co-consumer), ADR 0103
(async backpressure the pipeline-parallel pass depends on), ADR 0046/0047
(admission every optimized launch still passes).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: any advanced pass changes a node's output artifact hash; a speculated
impure node leaves observable state; priority scheduling admits a stage the
broker would refuse; or a pass breaks the DCE↔hint id-space invariant. Any of
these triggers a corrective entry or a superseding ADR.
