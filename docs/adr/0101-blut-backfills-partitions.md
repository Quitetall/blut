<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0101 — Add a partition abstraction with partition-aware execution, backfill windows, and a status matrix

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` — `Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-12` (engine `6744182`)

## Context

BLUT executes a DAG **once** over its inputs. There is no first-class notion of
"run this same graph over each slice of a partitioned domain, track which slices
are done, and fill the gaps." Every partitioned workload today is faked: a shell
loop over corpora, a hand-maintained list of which datasets have been ingested,
a re-run of the whole graph when one corpus arrives late.

This is genuine white-space versus the competition. Dagster's central data-model
is the **partitioned asset** — a definition parameterized by a partition key
(daily dates, categories), with a UI matrix of which partitions are
materialized/stale/failed, and a **backfill** that launches the graph across a
selected key range. Airflow has scheduled backfills over `execution_date`
windows. BLUT has neither the key nor the matrix — so it cannot answer "which of
the ~40 corpora have a current training-ready manifest?" without a side script,
and cannot re-run only the stale slice.

The workloads that want this are concrete:
- one archive manifest **per corpus** (~40 corpora), each a partition;
- domain evaluation **per dataset × preset**, a 2-D partition space;
- time-windowed ingest as new recordings land.

Constraints: partitions must ride on the existing content-addressed cache
([0078](0078-blut-starlark-typed-dynamic-dag.md)) so a materialized partition is
*free* to skip; a backfill launches many stages and therefore must flow through
the never-OOM broker ([0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md))
— a backfill can never storm the box; and the clinical/PHI hard-block
([0061](0061-blut-p2p-compute.md)) applies per partition (a `Restricted`
partition is refused, never silently skipped-as-done).

## Decision

Add a **partition abstraction** as a thin, typed layer over the existing DAG —
the graph is unchanged; a partition parameterizes *which inputs* it runs over:

1. **`PartitionKey` + `PartitionSpec`.** A stage/asset may declare a
   `PartitionSpec`: `Time { granularity, tz }`, `Categorical { values }`, or a
   `Multi` product of the two (dataset × preset). A concrete run carries a
   `PartitionKey` (e.g. `2026-07-05` or `tuh_eeg×fast`) that flows into the
   node's cache key — so each partition is a distinct content-addressed
   artifact, and a materialized one is a cache hit.
2. **Partition-aware execution.** `blut backfill <asset> --partitions <range>`
   expands the range into one PlanSpec instance per key (reusing 0078's dynamic
   fan-out, `map_output`), submits them, and lets the cache skip already-done
   keys. Each expanded stage is an ordinary broker-admitted launch — the backfill
   is bounded by admission, never a parallel flood.
3. **Partition status matrix.** A `PartitionStatus` store (a table in the
   existing lineage SQLite, no new service) records per key:
   `{materialized | stale | failed | restricted | missing}`, derived from cache
   presence + last-run lineage rows. `blut partitions status <asset>` renders the
   matrix as text (and the TUI/web sidecars read the same table — charter-safe,
   no in-engine server). "Stale" = an upstream partition's artifact hash changed.

`--partitions` accepts a range (`2026-07-01:2026-07-08`), an explicit set, or
`--missing`/`--stale` selectors so a backfill targets exactly the gaps.

## Rationale

- **Partition key in the cache key is the whole trick.** It makes "materialized"
  and "skip if done" fall out of the existing content-addressed cache for free —
  no new done-tracking state to keep consistent, and it honours the 0078 charter
  (the key is *extended* with the partition, existing un-partitioned keys are
  unchanged since their `PartitionKey` is `None`).
- **Backfill = fan-out we already have.** ADR 0078's `map_output` dynamic DAGs
  already express "one instance per element"; a backfill is that pattern with the
  element being a partition key, so this rides proven machinery.
- **The matrix answers the operator's real question** — "what's done, what's
  stale, what failed" — from the lineage DB the sidecars already read, needing no
  charter-violating dashboard service.
- **Per-partition PHI hard-block** keeps a `Restricted` corpus from ever being
  marked done-by-skip: it shows `restricted` in the matrix and is refused at
  admission, fail-closed (0061).

## Alternatives Considered

- **Keep the shell-loop status quo (external per-corpus scripts).** Rejected:
  the done-list is hand-maintained, drifts from reality, has no staleness notion,
  and re-runs the whole graph when one corpus lands late — the exact toil this
  abstraction removes. It also can't express dataset×preset 2-D spaces cleanly.
- **A separate partition-tracking service/daemon (Dagster-daemon style).**
  Rejected on charter: [ADR 0034](0034-blut-scope-charter.md) forbids an
  in-engine service/supervisor. The status lives in the existing lineage SQLite
  and is read by CLI + sidecars — no new process.
- **Encode partitions as distinct hand-written DAGs (one graph per corpus).**
  Rejected: N copies of the same graph is unmaintainable and defeats the cache's
  cross-partition reuse; a change to the shared stage would need N edits.

## Consequences

- `PartitionSpec`/`PartitionKey` become additive `blut-types` serde types; the
  node cache-key function gains a `partition: Option<PartitionKey>` input that is
  `None` for existing un-partitioned nodes (their keys unchanged).
- A new `partition_status` table in the lineage SQLite (additive migration); the
  TUI/web sidecars gain a matrix view reading it.
- `blut backfill` and `blut partitions status` are new CLI subcommands;
  `backfill` is a thin driver over 0078 fan-out + broker admission.
- Commits us to defining "stale" precisely (upstream artifact-hash change); a
  wrong staleness rule either re-runs too much or serves outdated partitions.
- A large backfill (40 corpora × N presets) is throttled by the broker, so it may
  take wall-clock time — correct by design (never OOM), but the matrix must show
  `missing` vs `in-flight` so an operator doesn't mistake throttling for failure.

## Implementation Plan  *(the contract — amend only by a dated Progress Log entry)*

- **Deliverable:** `PartitionSpec`/`PartitionKey` types + partitioned cache key +
  `blut backfill` + `blut partitions status` + the lineage-DB status matrix.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut partitions -- --nocapture` (a 3-key
    categorical asset: backfill all 3 → matrix all `materialized`; invalidate 1
    upstream → that key alone reads `stale`; `--missing` re-runs only it; a
    `Restricted` key is refused and shows `restricted`, never `materialized`) +
    `pytest -k partition_backfill`
  - `pass when:` exit 0 **and** an un-partitioned asset's cache keys are
    byte-identical to the pre-ADR baseline (partition = `None`).
- **Rollback:** the partition field defaults to `None` and is the last cache-key
  input; drop it and every existing key reverts. `blut backfill`/`partitions` are
  additive subcommands — removing them touches no existing path.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-11 — INCREMENT 1 LANDED (engine main `bc63bc6`): the richer per-cell
  status MATRIX._ `src/config/partition.rs`: `CellStatus`
  {Materialized | Stale | Failed | Restricted | Missing} + `CellStatus::derive`
  (clinical DOMINATES fail-closed — a restricted cell is Restricted regardless of
  run state; then None⇒Missing, non-`done`⇒Failed, done+upstream-drift⇒Stale,
  done+fresh⇒Materialized). `status_matrix()` maps a set's cells → status, with
  the `restricted`/`stale` predicates INJECTED (the ADR-0100/lineage hash
  comparison) so the derivation is a pure, testable function; the `stale_of`
  check is short-circuited to materialized cells only (its lineage cost is not
  paid on Missing/Failed/Restricted cells). This is the read-side the
  `--missing`/`--stale` backfill selectors target. Gate part: `cargo test -p blut
  partition` = 15 passed (incl `cell_status_matrix_derives_all_states` — all five
  states + clinical-dominates). 857 lib green; clippy+fmt clean. MiMo PASS WITH
  NITS → the stale short-circuit. **REMAINING (own increments):** the partitioned
  cache key (partition as the LAST, default-`None` cache-key input ⇒ byte-identical
  to the pre-ADR baseline when unpartitioned), `blut backfill <asset>
  --partitions <range>` execution wired to this matrix, the `blut partitions
  status` CLI render, the lineage-hash stale-detection feeding `stale_of`, and the
  cross-repo `pytest -k partition_backfill`. ADR stays In progress._
- _2026-07-12 — M2.0 CONTROL/GATE REPAIR (engine main `058c232`)._ The original
  `cargo test -p blut partitions -- --nocapture` command was a substring filter
  that executed **zero tests** and exited 0, so it was not a fail-closed gate.
  The runnable progress gate is now `cargo test -p blut --test
  partition_backfill -- --nocapture` (1 passed). It proves the already-landed
  persisted missing-cell work list plus all five read-side matrix states. Source
  audit also corrects the prior remaining-work wording: `blut partition
  backfill` already runs cells through `run_one_recipe`, per-cell admission, and
  a shared cross-cell RAM semaphore. Still open: partition identity in cache
  keys, stale/missing selectors wired to the matrix, lineage-derived staleness,
  matrix-aware CLI rendering, and per-cell Restricted refusal. Therefore this
  named target is a progress gate only; the full acceptance gate remains unmet
  and the ADR stays In progress._
- _2026-07-12 — M2.3 PARTITION COMPLETION LANDED (engine main `6744182`; feature
  `81e617a`, reviewed hardening `6744182`)._ `PartitionSpec`, `PartitionKey`,
  `PartitionValue`, and time granularity are WASM-safe `blut-types` contracts.
  Every static node, map template, and runtime-spawned subplan carries the
  partition into the cache key; `partition=None` retains the exact pre-ADR
  golden key. The lineage v5 projection and tenant-scoped canonical JSONL
  derive Materialized only from a matching done run, current source/compiled
  fingerprint, valid terminal sidecar hash, and a live decodable cache entry;
  cache eviction therefore reads Stale. Canonical JSONL force-reconciles a
  disposable SQLite row even when that row is clock-skewed into the future.
  Cache-hit jobs emit current-job metadata and cache proof, so a warm retry is
  Materialized without rerunning its stage. Concurrent device chains serialize
  complete JSONL record appends, and Restricted classification walks nested
  args iteratively.

  _CLI acceptance now covers categorical/multi/time definitions, inclusive
  explicit/range selection, default/`--missing`/`--stale`/`--force` selectors,
  tenant isolation, exact one-cell stale recomputation, and pre-launch
  Restricted refusal. This corrects the original gate wording: an invalidated
  materialized cell is selected by `--stale`, not `--missing`. The fail-closed
  named gates are `cargo test -p blut --test partition_backfill -- --nocapture`
  (3 passed), the in-crate real CLI test
  `partition_backfill_executes_three_cells_indexes_matrix_and_refuses_restricted`
  (1 passed), and a cross-repository pytest `[redacted: private path]` `-k
  partition_backfill` (1 passed). Final repository evidence: 776 default tests;
  949 all-feature tests passed with one expected S3 ignore; all integration and
  bench targets passed; strict all-feature clippy, fmt, `blut-types`
  `wasm32-wasip1`, scope guard, and diff checks passed. Independent Standards
  and Spec reviews returned PASS. Mandatory commit reviews returned PASS WITH
  NITS; the verified concurrent-append concern was fixed in `6744182`, while
  delimiter, legacy-precedence, and canonical-JSON claims were rejected against
  source and recorded in the follow-up commit message._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut --test partition_backfill -- --nocapture`;
  `cargo test -p blut
  partition_backfill_executes_three_cells_indexes_matrix_and_refuses_restricted
  --lib -- --nocapture`; the cross-repository pytest `[redacted: private path]`
  `-k partition_backfill`; plus the full repository gates listed in the final
  Progress Log entry
- **result:** the public contract target, real three-cell CLI path, exact
  stale-only rerun, cache-hit materialization, cache eviction, tenant isolation,
  dynamic-spawn cache separation, and Restricted refusal all passed
- **git_sha:** engine `674418241baee65be2476eb2c84062a7cf4a69c7`  ·  **artifact sha256:** n/a (source-only change)

## Related Decisions

[ADR 0078](0078-blut-starlark-typed-dynamic-dag.md) (PlanSpec fan-out +
content-addressed keys this partitions — extended, not superseded),
[ADR 0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md)
(broker bounds every backfill launch), [ADR 0061](0061-blut-p2p-compute.md)
(per-partition PHI hard-block), [ADR 0094](0094-blut-eventing-triggers-sla.md)
(allocated: lineage/asset catalog the status matrix reads from).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: a partitioned run collides cache artifacts across keys; a `Restricted`
partition is ever reported `materialized`; a backfill launches stages outside
broker admission; or adding partitions changes an un-partitioned node's cache
key. Any of these triggers a corrective entry or a superseding ADR.
