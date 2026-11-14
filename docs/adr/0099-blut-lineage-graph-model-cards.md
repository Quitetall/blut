<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0099 — Expose a full artifact-provenance graph, per-model cards, and a run-diff tool over the lineage_db substrate

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` — `Proposed 2026-07-10` → `In progress` (increment 1 — provenance graph + lineage tenant column) → `Complete — frozen 2026-07-11` (graph + card + diff; gate passes)

## Context

Asset-level lineage is a headline feature of the modern orchestrator: Dagster's
asset graph and Flyte's lineage view let a user click any output and walk
backward to every input, code version, and parameter that produced it — the
audit story regulated and research users both demand. BLUT has the raw material
but not the product. `src/lineage_db.rs` (the substrate landed under the
[ADR 0090](0090-blut-experiment-model-dataset-registry.md) vision-Phase-11 track)
records rows: which stage produced which content-addressed artifact, with which
args, under which run. What is missing is the **graph** over those rows
(artifact→artifact provenance you can traverse), a **model card** that collates
everything about one trained model (its training data, args, metrics, and
acceptance-gate outcomes) into one signed view, and a **run-diff** tool that
answers "what changed between run A and run B?" — the first question anyone
asks when two runs of the same recipe disagree. This ADR extends the lineage_db
substrate; it does not re-decide the schema.

## Decision

Build three read-side capabilities over `src/lineage_db.rs`, keeping all
graph/card/diff computation as pure lineage reads in the engine (charter-fine —
no server) with rendering delegated to sidecars. **(1) Provenance graph:** a
`blut lineage graph <artifact>` query that traverses the producer/consumer edges
transitively to emit the full artifact→artifact DAG (as typed
`blut-types` rows and a DOT/JSON export), rooted at any content-addressed
artifact and walking backward to source inputs or forward to descendants. **(2)
Model cards:** `blut lineage card <model-artifact>` collates one model's
training-data artifacts, resolved args, recorded metrics, and its
acceptance-gate outcomes into a single deterministic, content-addressed card
(re-running on the same lineage rows yields a byte-identical card). **(3)
Run-diff:** `blut lineage diff <runA> <runB>` reports the symmetric difference of
two runs — differing args, differing input artifact hashes, differing metrics,
and gate-outcome deltas. Cache keys are unchanged: the graph reads the existing
content-addressed identities, it does not mint new ones. Clinical/PHI-tenant
lineage rows are never exported to a remote sink
([ADR 0061](0061-blut-p2p-compute.md) fail-closed); the web rendering
([ADR 0083](0083-blut-interface-stack.md) `blut-web`) reads the same typed rows
the CLI emits.

## Rationale

- **The substrate is already there; only the read side is missing.** lineage_db
  records the edges — turning rows into a traversable graph is a query layer, not
  a new write path, so provenance parity with Dagster/Flyte is cheap and honest.
- **Model cards are the audit deliverable.** Collating data+args+metrics+gate
  outcomes into one signed, content-addressed view is exactly what a change-control
  verification record needs `[redacted: non-engine product detail]` — the card
  becomes a reusable audit artifact, not just a dashboard tile.
- **Run-diff is the debugging workhorse.** "Why do these two runs differ?" is the
  most common question; answering it by diffing lineage rows (args + input hashes
  + metrics + gates) is deterministic and beats eyeballing two log files.
- **Determinism is a correctness property.** Because cards and diffs are pure
  functions of content-addressed lineage rows, they are reproducible and
  themselves content-addressable — a card can be cited by hash in an ADR or a
  verification record.
- **No charter drift.** All traversal is a read in the engine; only rendering
  leaves the engine, into the existing sidecars.

## Alternatives Considered

- **Rebuild lineage on a graph database (Neo4j) or an external catalog
  (OpenLineage/Marquez).** Rejected: it forks the source of truth away from the
  content-addressed lineage_db, adds a server dependency the charter forbids in
  the engine, and creates a sync problem between the cache's artifact identities
  and the catalog's. The graph must be a view over lineage_db, not a copy.
- **Model cards as hand-written Markdown per run.** Rejected: non-deterministic,
  un-diffable, drifts from reality the moment args change, and cannot be
  content-addressed or gate-checked. Cards must be derived, not authored.
- **Line-by-line log diff instead of structured run-diff.** Rejected: logs are
  noisy (timestamps, ordering) and don't isolate the semantically meaningful
  deltas (args/input-hash/metric/gate); a structured diff over lineage rows does.

## Consequences

- Upside: BLUT reaches artifact-lineage parity with Dagster/Flyte and adds
  deterministic, content-addressed model cards + run-diff that the incumbents
  don't offer as first-class, hash-citable artifacts.
- Upside: model cards double as change-control verification-record inputs
  `[redacted: non-engine product detail]`; run-diff becomes the standard triage
  tool when two runs of a recipe disagree.
- Upside: the web dashboard ([ADR 0083](0083-blut-interface-stack.md)) gets a
  lineage view for free by reading the same typed rows the CLI emits.
- Downside: transitive graph traversal cost grows with lineage depth; deep DAGs
  will need bounded/paged queries (the CLI exposes a depth limit) to stay
  interactive.
- Constrains: the graph/card/diff row types in `blut-types` become a shared read
  schema (engine CLI + web); changes need a Progress-Log entry. lineage_db's
  producer/consumer edge recording is now load-bearing — a stage that fails to
  record an edge produces a card/graph with a silent hole.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** `blut lineage graph|card|diff` CLI subcommands (pure
  lineage_db reads) emitting typed `blut-types` rows + DOT/JSON export; a
  deterministic content-addressed model-card format; `blut-web` lineage/card/diff
  views over the same rows.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `scripts/scope_guard.sh && cargo test -p blut --test lineage_graph_card_diff`
  - `pass when:` `exit==0` — against a fixture lineage_db: `graph` returns the
    full transitive producer set for a leaf artifact (no missing/extra edges);
    `card` is byte-identical across two runs on the same rows (content-addressed
    determinism) and includes data+args+metrics+gate-outcome sections; `diff`
    reports exactly the seeded arg/input-hash/metric/gate deltas between two runs
    and nothing else; and a clinical-tenant row is never present in an exported
    graph/card.
- **Rollback:** remove the `blut lineage` subcommands + `blut-web` views;
  lineage_db keeps recording rows as before (read-side only, additive), so revert
  is clean with no data-path change.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-11 — INCREMENT 1 LANDED (engine main `8bbae1e`): the provenance GRAPH
  (capability 1 of 3) + the [ADR-0096](0096-blut-multi-tenancy.md) lineage tenant
  column that powers the fail-closed clinical export._ `lineage_db` v2→v3 adds
  `runs.tenant TEXT NOT NULL DEFAULT 'default'` (additive; a `PRAGMA
  table_info`-guarded `ALTER TABLE` migrates an extant v2 db); `RunRow.tenant`
  threaded through record_run/get_run/row_to_run, with the upsert using
  `COALESCE(NULLIF(excluded.tenant,'default'), runs.tenant)` so a re-ingest can
  never DOWNGRADE a clinical run out of its ADR-0061 boundary. `LineageDb::
  inputs_for_output` (all inputs of an output — the multi-branch generalisation
  of the single-input `trace`) + `graph_upstream(hash, exclude_restricted)`:
  cycle-guarded BFS backward over `lineage_edges`, following every fan-in;
  `exclude_restricted` drops any clinical/restricted-tenant node AND (post-filter)
  every edge touching it, so no `clinical → …` edge leaks. `src/lineage_report.rs`:
  `ProvenanceGraph`/`GraphNode` + a deterministic DOT export (root highlighted,
  labels DOT-escaped against injection) + `sources()`. CLI `blut lineage graph
  <hash> [--json]` — DOT by default, an EXPORT so clinical nodes are excluded;
  an unknown hash errors clearly. **Gate part met:** `cargo test -p blut --test
  lineage_graph_card_diff` = 3 passed (full transitive producer set with exact
  edges+sources; clinical node AND its edge excluded from the export; DOT
  byte-stable; tenant never downgraded on re-ingest) + 3 `lineage_report` unit
  tests. 857 lib green; clippy+fmt+scope_guard clean. MiMo → DOT-label escaping,
  `table_info` migration probe (vs error-string match), the no-downgrade upsert,
  the unknown-hash guard. **REMAINING (increment 2, same test binary):** `blut
  lineage card` (deterministic content-addressed model card:
  data+args+metrics+gate-outcome) and `blut lineage diff` (symmetric run diff).
  Deferred perf: graph_upstream issues one query per node (O(N) round-trips) —
  batch it if a real 10k-node DAG appears. ADR stays In progress._
- _2026-07-11 — INCREMENT 2 LANDED + ADR FROZEN (engine main `ba76f9c`): the
  model CARD + run DIFF, completing all three capabilities._ `LineageDb::
  model_card(hash, exclude_restricted)` collates data sources (from
  graph_upstream) + recipe/config_fingerprint + headline metrics (final_metrics)
  + gate outcome into a DETERMINISTIC content-addressed card: `CardContent::
  content_hash` = sha256 over canonical JSON (stable serde field order + all
  vectors sorted at the builder), so a rebuild on the same rows is byte-identical;
  `ModelCard::verify()` re-checks the address; the export drops clinical data
  sources (a restricted model → `None`). `LineageDb::run_diff(a, b)` = the
  symmetric difference (recipe / config_fingerprint / gate outcome / headline
  metrics, union of names, bit-compared). CLI `blut lineage {card,diff}`. **Full
  gate met:** `cargo test -p blut --test lineage_graph_card_diff` = 5 passed
  (graph + card determinism/sections/clinical-exclusion + diff exact-deltas +
  tenant no-downgrade). MiMo → explicit data_sources sort (belt-and-suspenders
  determinism), `verify()` + doc that `card_hash` is over `content` only, human
  diff render polish. NOTE `blut-web` lineage/card/diff views ride the ADR-0083
  crate split when `blut-web` lands (out of this gate). Status → Complete —
  frozen 2026-07-11._

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** `passed`
- **gate_cmd:** `scripts/scope_guard.sh && cargo test -p blut --test lineage_graph_card_diff`
- **result:** exit 0 (5 passed). All three read-side capabilities against a
  fixture db: (graph) returns the full transitive producer set of a leaf with
  exactly the right edges + sources, DOT byte-stable; (card) byte-identical
  `card_hash` across rebuilds with the data + args + metrics + gate-outcome
  sections populated; (diff) reports exactly the seeded config/gate/metric deltas
  (identical recipe + identical `loss` metric correctly omitted) and nothing
  else; and — the clinical boundary — a clinical/restricted node is never present
  in an exported graph OR card (its node AND every edge touching it dropped), and
  a set tenant is never downgraded on re-ingest. scope_guard PASS; 857 lib green;
  clippy + fmt clean.
- **git_sha:** engine `ba76f9c` (incr 1 `8bbae1e` + incr 2 `ba76f9c`)  ·  **artifact sha256:** n/a (source-only)

## Related Decisions

[ADR 0090](0090-blut-experiment-model-dataset-registry.md) (lineage_db.rs
substrate — this extends it), [ADR 0072](0072-blut-error-domains-and-fuzzing.md)
(error-domain / fault model), [ADR 0083](0083-blut-interface-stack.md) (blut-web
rendering sidecar + blut-types schema), [ADR 0061](0061-blut-p2p-compute.md)
(clinical/PHI export hard-block), plus the change-control track the model card
feeds `[redacted: non-engine product detail]`.

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: a model card or provenance graph is non-deterministic across identical
lineage rows (loses content-addressability); a graph shows a hole because a stage
skipped edge recording; a clinical-tenant lineage row leaks into an export; the
graph query forks off lineage_db into a separate catalog that can drift; or
run-diff surfaces noise (timestamps/ordering) instead of the semantic deltas.
