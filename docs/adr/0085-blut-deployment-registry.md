<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0085 — Give BLUT a versioned plan registry so a fingerprinted DAG can be published, promoted to a named pointer, and rolled back atomically

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text; correct an earlier statement with a *new* dated
> entry beneath it, so the original stays as historical record. When the
> deliverable passes its gate, fill **Completion** and set Status to
> `Complete — frozen YYYY-MM-DD`. A frozen ADR is permanent. To reverse a frozen
> decision later, write a *new* ADR that supersedes it (updating this one's
> Status pointer is the only edit allowed after freeze).

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-10`

## Context

Flyte registers every workflow as an immutable, versioned entity and lets an
operator activate/deactivate a launch plan by version; Dagster code-locations
serve versioned pipeline definitions the daemon reloads. Both give a team the
operation "ship *this* pipeline to *that* environment, and undo it" without
re-running or editing source.

BLUT today has the raw material but no registry. ADR 0078 landed the PlanSpec IR
and its content-addressed fingerprint (`plan_fingerprint`, blake3 over the
canonicalised typed graph), and the Starlark front-end compiles a DAG to that IR
out-of-process. But there is nothing that *names* a fingerprint, records who
published it, or lets `registry://plan@prod` resolve to a specific frozen graph.
A run today is launched from a source path; "roll back prod to yesterday's graph"
means "find the old source, hope it still compiles the same, re-run" — which is
neither atomic nor verifiable, and silently drifts if a transitive stage
registration changed. There is also no place that a publish is type-checked
*before* it can be promoted, so a broken graph is only discovered at dispatch.

## Decision

BLUT will add a local **plan registry** backed by two SQLite tables in
`~/.blut/registry.db`: `deployments` (immutable rows — `plan_fingerprint` PK,
canonical PlanSpec bytes, publisher, timestamp, source-provenance, tenant) and
`deployment_pointers` (mutable named pointers — `(tenant, name) → plan_fingerprint`,
with a `history` of prior fingerprints per pointer). Four engine subcommands
operate it, all pure-local file/DB work (charter-safe, no server): `blut plan
publish <spec>` canonicalises + typechecks the PlanSpec and inserts an immutable
`deployments` row keyed by its ADR-0078 fingerprint (idempotent — republishing
identical bytes is a no-op returning the same id); `blut plan promote
<fingerprint> registry://plan@<name>` moves a pointer and appends the previous
target to that pointer's history; `blut plan rollback registry://plan@<name>`
pops the pointer back to its previous history entry atomically (single DB
transaction); `blut plan history registry://plan@<name>` prints the pointer's
audit trail. The fingerprint IS the deploy identity — `blut run
registry://plan@prod` resolves the pointer to a frozen `deployments` row and
dispatches that exact graph. Typecheck-on-publish is fail-closed: a PlanSpec that
does not type-resolve against the compiled stage registry cannot enter
`deployments`. Every publish/promote/rollback carries the acting tenant (ADR
0096) and refuses to cross the clinical boundary (ADR 0061) — a Restricted-tenant
fingerprint can never be promoted onto a shared-tenant pointer.

## Rationale

- The fingerprint already exists and is content-addressed (ADR 0078); reusing it
  as deploy identity means "prod == fingerprint X" is verifiable by recompiling
  the source and comparing hashes — no separate version counter to drift.
- Immutable `deployments` + mutable pointers is the minimal shape that gives
  atomic promote/rollback: rollback is a single-row pointer update inside one
  transaction, never a re-run.
- Typecheck-on-publish moves the failure from dispatch time (after broker
  admission, after side effects) to publish time, which is free and reversible.
- Pointer history is the audit substrate RBAC (ADR 0095) and multi-tenancy (ADR
  0096) hang off — "who put what on prod when" is one table.
- SQLite + files keeps the registry inside the engine process (no HTTP server,
  ADR 0034); the web dashboard (ADR 0093) reads it, never owns it.

## Alternatives Considered

- **Git tags / branches as the registry.** Rejected: a git ref names a source
  commit, not a *typed compiled graph* — the same commit can compile to a
  different PlanSpec if a transitively-registered stage changed between
  compiles, so a git ref is not a stable deploy identity the way the ADR-0078
  fingerprint is. It also cannot express per-tenant pointers or a fail-closed
  typecheck gate.
- **A monotonic integer version per plan name (Flyte-style `v1, v2`).** Rejected:
  a counter is an extra source of truth that must be allocated, synchronised
  across the mesh (ADR 0079), and can collide; the content hash is already
  globally unique and mesh-safe with zero coordination. We keep a human name only
  at the *pointer* layer, never as the artifact identity.
- **Serve the registry from the `blut-worker`/axum sidecar as the source of
  truth.** Rejected: that would make the control plane depend on a running server
  for a `blut run` to resolve a pointer, violating the "engine ships as the CLI
  tool, offline-capable" premise (ADR 0083). The sidecar may *read* the DB; it is
  not the registry.

## Consequences

- Upside: `blut run registry://plan@prod` and one-command rollback become real;
  the plan graph shown in the console/web (ADR 0093) can be pinned to a
  fingerprint, so the UI stops guessing which source is live.
- Upside: the registry is the natural home for the RBAC audit log and tenant
  quota accounting — later ADRs extend it rather than inventing new storage.
- Downside: a second stateful store (`registry.db`) to back up and migrate
  alongside the lineage DB; a schema-migration discipline is now load-bearing.
- Downside: typecheck-on-publish couples publish to the *current* compiled stage
  registry — publishing a plan that references a stage removed from the binary
  fails, by design, which some workflows will experience as friction.
- Constraint inherited: pointers are per-tenant; a shared control plane cannot
  have one global `prod` — every pointer name is tenant-qualified (ADR 0096).

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** `~/.blut/registry.db` with `deployments` (immutable,
  fingerprint-keyed) + `deployment_pointers` (named, with history) tables, and
  `blut plan {publish,promote,rollback,history}` subcommands; a `registry://` URI
  resolver in the run dispatcher.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut --test registry_roundtrip` — publishes a
    known PlanSpec, promotes it to `registry://plan@test`, publishes a second,
    promotes, rolls back, and asserts the resolved fingerprint + that a
    type-broken PlanSpec is rejected at publish and a Restricted-tenant
    fingerprint is refused at promote.
  - `pass when:` `exit == 0` (rollback resolves to the exact prior fingerprint;
    broken-publish and clinical-cross-promote both return non-zero and write no
    row).
- **Rollback:** feature-gate the subcommands behind `--features registry`; if
  abandoned, drop the two tables and the URI resolver — `blut run <path>` (source
  dispatch) is unaffected.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-10 — LANDED + FROZEN (engine main `f8082c7`). `src/registry_db.rs`:
  `~/.blut/registry.db` (SQLite, `$BLUT_REGISTRY_DB` override) with `deployments`
  (immutable, ADR-0078 fingerprint PK, canonical spec bytes + publisher + tenant
  + source), `deployment_pointers` (`(tenant,name)→fingerprint`) and an
  append-only `pointer_history` trail. API: `publish` (fail-closed typecheck via
  `spec.compile(reg)` then insert; idempotent same-tenant, refuses cross-tenant
  re-publish of identical bytes), `promote` (cross-tenant refusal + upsert +
  history in one txn), `rollback` (BEGIN IMMEDIATE, read+write one snapshot, pops
  to the previous target with a tenant re-check), `resolve_pointer`/`resolve_spec`,
  `history`, `parse_pointer_uri` (`[A-Za-z0-9_.-]` allowlist). CLI: `blut plan
  {publish,promote,rollback,history}` (`--tenant`); the run dispatcher resolves
  `blut recipe declare registry://plan@<name> [--run] [--tenant]` to the frozen
  spec. **Gate met:** `cargo test -p blut --test registry_roundtrip` = 1 passed —
  publish A → promote → publish B → promote → rollback resolves to A's exact
  fingerprint + `resolve_spec` loads that graph; idempotent republish; distinct
  specs get distinct fingerprints; history trail `[A,B,A]`; a broken-stage spec
  refused at publish; a Restricted-tenant fingerprint refused at promote onto a
  shared pointer (pointer stays unset); cross-tenant re-publish refused; URI-name
  allowlist enforced. 857 lib green; clippy+fmt+scope_guard clean. MiMo (3
  rounds) → fixed a rollback TOCTOU (immediate txn), publish INSERT-OR-IGNORE
  tenant-masking + its narrow race-arm re-verify, added the tenant boundary to
  rollback + `--tenant` threading + URI validation. NOTE `RESTRICTED_TENANT` is a
  minimal clinical-boundary stand-in until ADR 0096 multi-tenancy lands. Status →
  Complete — frozen 2026-07-10._

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut --test registry_roundtrip`
- **result:** exit 0 (1 passed). Rollback resolves to the exact prior
  fingerprint; the broken-publish and clinical-cross-promote both return errors
  and write no pointer row; plus idempotent republish, distinct-fingerprint,
  full history trail, cross-tenant re-publish refusal, and URI-name validation
  all asserted.
- **git_sha:** engine `f8082c7`  ·  **artifact sha256:** n/a (source-only)

## Related Decisions

ADR 0078 (PlanSpec IR + content-addressed fingerprint — this extends it, does not
supersede), ADR 0083 (interface stack — the web sidecar reads this DB), ADR 0090
(roadmap — deployment/rollout), ADR 0095 (RBAC gates publish/promote), ADR 0096
(per-tenant pointer namespaces), ADR 0061 (clinical hard-block at promote).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: a promoted pointer ever resolves to a fingerprint whose stored PlanSpec
no longer typechecks (publish gate leaked); if two operators promoting
concurrently on the mesh (ADR 0079) can leave a pointer with a lost update (the
transaction isn't atomic across nodes); or if teams route around the registry
because pointer resolution is slower than re-running from source. Watch the
publish→promote→run latency and the rate of `registry://` vs raw-path runs.
