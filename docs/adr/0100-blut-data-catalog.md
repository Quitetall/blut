<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0100 — Build a searchable, typed, lineage-linked dataset catalog over BLUT artifacts and deliberately stop short of a DVC-style dedup hub

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` — `Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-11`

## Context

BLUT knows a great deal about every artifact it produces — the ABIR type
(ADR 0069 [not federated] / ADR 0074 [not federated]), the content hash
([0078](0078-blut-starlark-typed-dynamic-dag.md)), the producing stage and its
inputs (`src/lineage_db.rs`) — but a human cannot *ask* about it. "Which datasets
are EEG at 256 Hz?", "what produced `dataset://tuh@v3` and what did it feed?",
"show me every ABIR manifest tagged `sleep`" are all unanswerable without grepping
raw lineage. Dagster ships an asset catalog; DVC ships a dataset registry with a
content-addressed dedup hub and remote push/pull. The catalog UI is genuine
white-space for BLUT — nobody has built the *browse/search* surface — but the
dedup-hub half is **not** white-space we want: the content-addressed cache (0078)
already deduplicates, and a push/pull hub is a network data plane the mesh
([0079](0079-blut-symmetric-mesh.md)) and cloud queue
([0082](0082-blut-cloud-queue.md)) already own. The gap is discovery, not storage.

## Decision

Build a **read-only dataset catalog** as an engine-side view over the lineage DB
and the ABIR type registry — a `blut catalog` surface (`search`, `show`, `tag`)
plus a queryable index — with **no new storage and no hub**. Each catalog entry
binds: the dataset's `dataset://name@v` handle (resolver shared with
[ADR 0090](0090-blut-experiment-model-dataset-registry.md)), its **schema**
(channel count, sample rate, dtype, window shape — read from the ABIR manifest,
not re-derived), its **artifact kind** (the ABIR/0069 type), its **lineage links**
(the producing stage + downstream consumers, straight from `lineage_db`),
free-form **tags**, and **provenance** (source corpus, hash, ingest time).
`blut catalog search "modality:eeg fs:256 tag:sleep"` filters the index;
`blut catalog show <name>` renders the entry + its lineage neighborhood;
`blut catalog tag <name> <tag>` appends a tag row. The catalog is a *projection*
— every field is derived from existing sources (ABIR types, lineage rows, the
0090 registry), so it can be rebuilt from scratch and holds no authoritative
state except tags. It **deliberately does not** implement content dedup (0078
does), a remote push/pull hub (0079/0082 do), or blob storage of any kind.
Clinical/PHI datasets carry the [0061](0061-blut-p2p-compute.md) flag in their
catalog entry and are filtered out of any cloud-surfaced view; browsing is a local
read (no HTTP — a future `blut-web` sidecar, [0083](0083-blut-interface-stack.md),
can serve it).

## Rationale

- **The data to catalog already exists, uncatalogued.** ABIR types + lineage +
  hashes are all recorded; the catalog is an *index and query* layer, so the
  build cost is a searchable projection, not new bookkeeping.
- **Schema comes from ABIR, not re-derivation.** 0069/0074 made the manifest
  carry typed modality/rate/shape; the catalog reads those fields, so it can
  never disagree with the artifact about its own schema.
- **Lineage-linked entries turn "what is this?" into "what made it and what
  used it?"** — the graph the operator actually needs, drawn from the edges
  `lineage_db` already stores.
- **Stopping short of a hub is a feature, not a shortfall.** Dedup (0078) and
  the network data plane (0079/0082) are already solved elsewhere; re-solving
  them here would duplicate a cache and a mesh. The honest scope is discovery.
- **A rebuildable projection has no drift class of bug** — except tags, there is
  no authoritative catalog state to corrupt; `blut catalog rebuild` regenerates
  it from lineage + ABIR.

## Alternatives Considered

- **Adopt DVC as the dataset layer.** Rejected: DVC's core value is its
  content-addressed dedup + remote hub, both of which BLUT already has (0078 for
  dedup, 0079/0082 for transport) — we'd import a second, overlapping data plane
  and a `.dvc` metadata format that duplicates ABIR. We want DVC's *catalog*
  ergonomics, not its storage engine.
- **Build the catalog as a writable source of truth (its own DB).** Rejected: a
  second authoritative store for schema/lineage that ABIR and `lineage_db`
  already own — guaranteed to drift. A rebuildable projection cannot drift by
  construction (tags excepted, which are additive).
- **Ship only the ADR 0090 `dataset://` resolver, no browse surface.** Rejected:
  the resolver answers "give me *this* dataset by name"; it cannot answer "find
  the datasets matching X" — discovery is the whole point of a catalog and the
  actual white-space here.

## Consequences

- A new catalog index (materialized view over `lineage_db` + ABIR types) and the
  `blut catalog search|show|tag|rebuild` subcommands.
- Tags are the one piece of authoritative catalog state — they need an append
  path and must survive a `rebuild` (rebuild reprojects derived fields, preserves
  tags).
- The catalog is **read-mostly and non-authoritative** — it must never be the
  place a schema or lineage fact is *edited*; those belong to ABIR and
  `lineage_db`. A field that disagrees with its source is a bug in the
  projection.
- Explicitly foregoing a hub means cross-machine dataset sharing stays the job
  of the mesh/cloud queue — the catalog shows what's *here*, plus registry
  handles for what could be fetched.

## Implementation Plan  *(the contract — amend only via a dated Progress Log entry)*

- **Deliverable:** a catalog index (view over `lineage_db` + ABIR type registry),
  a tags table, and `blut catalog search|show|tag|rebuild` subcommands with a
  typed query grammar (`modality:` `fs:` `kind:` `tag:` `hash:`).
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut catalog:: && blut catalog rebuild && blut catalog search "modality:eeg fs:256" && blut catalog show tuh@v3 | grep -q 'produced_by'`
  - `pass when:` `search returns the seeded EEG-256 entries and no others; show renders the entry's schema + producing stage + downstream consumers from lineage; a tag added via 'catalog tag' survives 'catalog rebuild'; a clinical-flagged (0061) entry is absent from any cloud-surfaced view`
  - `gate_cmd:` `blut catalog show <schema-mismatch-fixture>; test $? -ne 0`
  - `pass when:` `a catalog entry whose projected schema disagrees with its ABIR manifest fails the consistency check rather than serving a stale field`
- **Rollback:** drop the catalog index + tags table + subcommands; discovery
  reverts to raw lineage grep; `lineage_db`, ABIR, and the 0090 registry are
  untouched.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-11 — LANDED + FROZEN (engine main `d3d6d73`). `src/catalog.rs`: a
  read-only PROJECTION over `datasets_db` + `lineage_db` (no new storage/hub/blob).
  `Schema` (modality/fs/channels/dtype) read from the dataset's ABIR manifest =
  its `metadata` JSON, never re-derived; `CatalogEntry` projected from a
  `DatasetRecord` + lineage neighborhood (`LineageDb::producing_stage` +
  `consumers_of`, both added); a clinical flag (restricted-tenant tag or
  `clinical:true`). `CatalogQuery::parse` (`modality:/fs:/kind:/tag:/hash:`,
  fail-loud on unknown key / bad term / non-numeric fs); `search()` with a
  `cloud_surfaced` flag that fail-closed excludes clinical entries (ADR 0061).
  Tags are the ONLY persisted state (`catalog_tags`, WAL) — they survive a
  rebuild; `verify_consistency()` fails on schema drift from the source manifest.
  CLI `blut catalog {rebuild,search,show,tag}` (`--cloud`/`--json`). **Gate met:**
  `cargo test -p blut catalog::` = 6 passed — fail-loud grammar; search returns
  only the `modality:eeg fs:256` entries; a clinical entry is local-visible but
  ABSENT from a cloud view; a tag persists (idempotent) + filters; a schema drift
  (fs 256→512) fails the consistency check; the lineage neighborhood projects.
  857 lib green; clippy+fmt+scope_guard clean. MiMo PASS WITH NITS → WAL on the
  tags store; verified build_index is used, hashes are lowercase at write, and
  schema-only consistency is the gate's intent. **GATE-FORM NOTE:** the ADR's
  `blut catalog rebuild/search/show …` shell invocations are adapted to the
  in-crate `catalog::` tests (the engine is lib-only — no engine binary; the CLI
  is wired for cookbook binaries); `blut-web` catalog browsing rides the ADR-0083
  crate split. Status → Complete — frozen 2026-07-11._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut catalog::`
- **result:** exit 0 (6 passed). Search returns the seeded EEG-256 entries and no
  others; show projects the schema + producing stage + downstream consumers from
  lineage; a tag survives a rebuild (tags are the only persisted state); a
  clinical-flagged entry is absent from a cloud-surfaced view; and a schema-drift
  fixture fails the consistency check rather than serving a stale field.
  scope_guard PASS; 857 lib green; clippy + fmt clean.
- **git_sha:** engine `d3d6d73`  ·  **artifact sha256:** n/a (source-only)

## Related Decisions

ADR 0069 [not federated] (the typed IR — the schema/kind the catalog indexes),
ADR 0074 [not federated] (manifests carry the typed fields),
[0078](0078-blut-starlark-typed-dynamic-dag.md) (content cache — the dedup the
catalog does NOT re-implement), [0079](0079-blut-symmetric-mesh.md)/[0082](0082-blut-cloud-queue.md)
(mesh / cloud queue — the transport the catalog does NOT re-implement),
[0090](0090-blut-experiment-model-dataset-registry.md) (registry — shares the
`dataset://` resolver).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: a catalog field is edited as a source of truth (schema/lineage must
live in ABIR/`lineage_db`); a `rebuild` drops user tags; a projected schema
diverges from its ABIR manifest without failing; a clinical-flagged dataset
surfaces in a cloud view; or scope creeps into blob storage / a push-pull hub
(that belongs to 0078/0079/0082).
