<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0096 — Give BLUT projects/domains with per-tenant namespaces for cache, registry, and plans — and make the clinical zone a hard tenant boundary a Restricted tenant's data can never cross

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-13`
(engine `550f910`)

## Context

Flyte's projects + domains are the axis every other capability hangs off:
resource quotas, access control, and workflow isolation are all scoped to a
`(project, domain)`. It is what lets one Flyte cluster serve many teams and a
dev/staging/prod split without cross-contamination. No premier orchestrator
ships without a tenancy model.

BLUT has none — this is genuine white-space (nowhere in the roadmap is it
planned). Today the content-addressed cache ([ADR 0078](0078-blut-starlark-typed-dynamic-dag.md)), the plan registry (ADR
0085), lineage, and the ε-ledger ([ADR 0080](0080-blut-async-dp-training.md)) are one flat global namespace on the
box; every run shares one cache, one set of plan pointers, one quota (the broker
envelope, [ADR 0046](0046-blut-resource-broker.md)). For a research box that is fine. For a system that also
processes clinical/PHI data under a hard-block ([ADR 0061](0061-blut-p2p-compute.md)), a *flat* namespace is
actively dangerous: a clinical run's cached intermediate artifact sits in the same
content-addressed store a shared run reads, and a shared `prod` pointer could
point at a clinical fingerprint. The clinical block is currently enforced per
dispatch, but there is no *structural* boundary keeping Restricted artifacts out
of shared namespaces.

## Decision

BLUT will introduce **tenants** (a Flyte-style `project`, optionally with a
`domain` suffix, e.g. `research/dev`, `clinical/prod`) as the top-level isolation
axis, and namespace the three stateful stores by tenant: the content-addressed
cache (`~/.blut/cache/<tenant>/…`, keyed by `(tenant, plan_fingerprint)` — the
[ADR-0078](0078-blut-starlark-typed-dynamic-dag.md) key algorithm is **unchanged**, tenant is a namespace *prefix* on the
store path, not a change to the fingerprint), the plan registry ([ADR 0085](0085-blut-deployment-registry.md)
pointers are `(tenant, name)` — there is no global `prod`, only
`clinical/prod`, `research/prod`, …), and plans/lineage/ε-ledger rows carry a
`tenant` column. Each tenant gets a **quota** — a broker sub-envelope (ADR
0046/0047) capping its concurrent resource claim, so one tenant cannot starve
another; admission is never bypassed, it is now per-tenant. **The clinical zone
is a HARD boundary, not a soft label**: a tenant marked `restricted` ([ADR 0061](0061-blut-p2p-compute.md))
is a sealed namespace — its cache is never read by a non-restricted tenant, its
plan fingerprints can never be promoted onto a shared pointer ([ADR 0085](0085-blut-deployment-registry.md)), its
lineage/ε rows are never served to a shared-tenant session ([ADR 0093](0093-blut-web-control-plane.md)/0095), and
mesh replication ([ADR 0079](0079-blut-symmetric-mesh.md)) and every sidecar refuse to move a Restricted
artifact off the owning box. The boundary is enforced fail-closed at **every**
dispatch/gate/registry/cache/mesh boundary: a cross-tenant read of a `restricted`
namespace, or any operation that would place a Restricted artifact in a shared
namespace, is a hard deny, logged ([ADR 0095](0095-blut-rbac-auth.md)), with no fallthrough. Cross-tenant
cache sharing between *non*-restricted tenants is disallowed by default (a run in
tenant A cannot reuse tenant B's cached node) to prevent accidental provenance
bleed; an explicit, audited `shared/` tenant is the only place deliberately
common artifacts live.

## Rationale

- Namespacing the store *path* by tenant while leaving the [ADR-0078](0078-blut-starlark-typed-dynamic-dag.md) fingerprint
  algorithm untouched keeps cache correctness (the key still identifies the graph)
  while giving structural isolation — one prefix, zero change to the hash we've
  already validated.
- Making the clinical zone a *namespace* (not just a per-dispatch check) turns the
  [ADR-0061](0061-blut-p2p-compute.md) block from "enforced everywhere we remembered" into "a Restricted
  artifact structurally cannot exist in a shared store" — the boundary is a
  property of where bytes live, which is far harder to leak past than a scattered
  runtime check.
- Per-tenant broker sub-envelopes give real quota (Flyte parity) by reusing the
  admission mechanism already trusted to prevent OOM ([ADR 0046](0046-blut-resource-broker.md)) — no new resource
  governor.
- Tenant is the column RBAC ([ADR 0095](0095-blut-rbac-auth.md)) already binds tokens to, so authz and
  isolation share one axis instead of two overlapping ones.
- Default-no-cross-tenant-sharing fails safe: the accident direction is "we
  recomputed a node we could have reused," never "tenant A read tenant B's data."

## Alternatives Considered

- **Separate BLUT installs / separate boxes per tenant (physical isolation
  only).** Rejected: it gives isolation but throws away the shared broker,
  shared mesh, and single control plane that make BLUT one system — and it still
  doesn't answer the in-box clinical/shared split on a research box that
  occasionally touches PHI. Physical separation is a *deployment* an operator can
  still choose; it is not a substitute for an in-engine tenancy model.
- **A soft `tenant` label on rows with no namespaced stores (filter, don't
  isolate).** Rejected: a filter-based boundary means a Restricted artifact
  physically sits in the shared content-addressed cache and is one missing
  `WHERE tenant=` clause away from being served — exactly the leak [ADR 0061](0061-blut-p2p-compute.md)
  forbids. The clinical boundary must be structural (separate store paths, refused
  at replication), not a query predicate.
- **Reuse Linux namespaces / cgroups as the tenant boundary.** Rejected: those
  isolate *processes and resources* but not the *logical* stores (cache, registry,
  lineage) that must be tenant-scoped, and the broker already owns cgroup
  containment ([ADR 0046](0046-blut-resource-broker.md)); tenancy is a data-namespacing concern layered above
  containment, not the same thing.

## Consequences

- Upside: BLUT can serve multiple teams and a dev/prod split on one box with real
  quota, and the clinical block becomes a structural namespace boundary instead of
  a runtime check repeated at each site — much stronger custody.
- Upside: RBAC ([ADR 0095](0095-blut-rbac-auth.md)), the registry ([ADR 0085](0085-blut-deployment-registry.md)), and the ε-ledger ([ADR 0080](0080-blut-async-dp-training.md))
  all gain their scoping axis "for free" — they already needed `tenant`.
- Downside: every store path, every registry pointer, and many DB queries grow a
  tenant dimension — a wide, migration-heavy change; a query that forgets the
  tenant scope is an isolation bug, so the gate must probe cross-tenant reads
  directly.
- Downside: default-no-cross-tenant cache sharing costs recompute when two
  non-restricted tenants genuinely run the same graph; the explicit `shared/`
  tenant is the escape hatch, at the price of an audited opt-in.
- Constraint locked: no future feature (mesh, web, notify, sensors) may move a
  `restricted` artifact off its owning box or into a shared namespace — this ADR
  is the structural statement of [ADR 0061](0061-blut-p2p-compute.md) that all of them must honour.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** a `Tenant` (`project[/domain]`) axis; tenant-prefixed cache /
  registry / plan / lineage / ε namespaces; per-tenant broker sub-envelopes; and
  a fail-closed clinical-zone boundary refused at dispatch, cache, registry,
  mesh, and sidecar surfaces.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut --test tenant_isolation` — runs the same
    graph in tenant `research/dev` and `clinical/prod`, asserts their cache
    entries live in disjoint store paths and neither reads the other's; asserts
    promoting the `clinical/prod` fingerprint onto a `research/prod` pointer is
    denied; asserts the mesh-replication and `blut-notify` paths refuse the
    `restricted` artifact; and asserts the [ADR-0078](0078-blut-starlark-typed-dynamic-dag.md) fingerprint of the graph is
    byte-identical across the two tenants (namespacing changed the path, not the
    key).
  - `pass when:` `exit == 0` — disjoint cache paths, cross-tenant read denied,
    clinical→shared promote/replicate/notify all return non-zero, fingerprint
    unchanged across tenants.
- **Rollback:** default tenant is a single `default/` namespace; with one tenant
  configured the system behaves exactly as the current flat store (the prefix is a
  constant), so tenancy can be disabled by pinning every run to `default/`.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-10 — INCREMENT 1 LANDED (engine main `db9c846`): the `Tenant` axis +
  tenant-namespaced content-addressed cache + the registry clinical boundary._
  `src/tenant.rs`: `Tenant` (`project[/domain]`) — `parse` (rejects empty
  segments, `.`/`..` traversal, non-`[A-Za-z0-9_.-]`), `as_path`, `is_default`
  (`default` = the flat store), `is_restricted` (`clinical`/`restricted`,
  case-insensitive = sealed clinical zone per [ADR 0061](0061-blut-p2p-compute.md)). `CacheHandle::with_tenant`
  prefixes the GLOBAL tier by the tenant path (additive: `default` is a no-op, so
  every existing cache path + the single-tenant engine are byte-identical);
  `job_local` untouched (already per-job). Wired `--tenant` into `blut recipe
  run`/`declare` (threaded run_one_recipe → run_recipe_sweep →
  launch_compiled_plan). The [ADR-0078](0078-blut-starlark-typed-dynamic-dag.md) key is UNCHANGED (tenant is a path prefix,
  not a key input). **Gate part met:** `cargo test -p blut --test tenant_isolation`
  = 2 passed — (a) two tenants get DISJOINT cache roots (neither an ancestor of
  the other), the default tenant is the flat store, the key is byte-identical
  across tenants (same hex, different parent), and a research-written entry is
  invisible under clinical's root; (b) a `clinical/prod` fingerprint (keyed by
  `Tenant::to_string()`) can NOT be promoted onto a `research/prod` pointer
  ([ADR-0061](0061-blut-p2p-compute.md) deny), the research pointer stays unset, but promotes onto its own
  clinical pointer — reusing the already-frozen [ADR-0085](0085-blut-deployment-registry.md) registry, whose
  `(tenant, name)` pointers + cross-tenant refusal ARE the 0096 registry
  requirement. Plus 4 `Tenant` unit tests. 857 lib green; clippy+fmt+scope_guard
  clean. Model review → removed the unvalidated `Tenant::new` (path-traversal footgun) +
  made `is_restricted` case-insensitive. **REMAINING (own increments):**
  per-tenant broker sub-envelopes (quota, [ADR 0046](0046-blut-resource-broker.md)/0047), tenant columns on
  plans/lineage/ε rows, and the mesh-replication ([ADR 0079](0079-blut-symmetric-mesh.md)) + `blut-notify`
  Restricted refusals (those components are not landed) + partition-cell tenant
  threading. ADR stays In progress until the full `tenant_isolation` gate
  (incl. mesh/notify refusal) passes._
- _2026-07-11 — the "tenant columns on lineage" remaining item PARTLY landed via
  [ADR 0099](0099-blut-lineage-graph-model-cards.md) increment 1 (engine main `8bbae1e`): `lineage_db.runs` gains a
  `tenant` column (v2→v3 migration) with a no-downgrade upsert, and the 0099
  provenance-graph export is fail-closed against clinical/restricted-tenant rows
  ([ADR 0061](0061-blut-p2p-compute.md)) — reusing this ADR's `Tenant::is_restricted`. Still remaining here:
  tenant columns on plans/ε rows, per-tenant broker sub-envelopes, threading the
  run's ACTUAL tenant into lineage ingestion (`ingest_job` currently defaults),
  and the mesh/notify refusals._
- _2026-07-12 — INCREMENT (engine main `e4fb77d`): the per-tenant broker
  SUB-ENVELOPE (quota) primitive._ `src/broker/admission.rs` (additive, PURE —
  extends the frozen [ADR-0046](0046-blut-resource-broker.md) gate): `TenantBudget{tenant, in_flight_gb,
  ceiling_gb}` = a tenant's slice of the admission envelope, supplied by the
  caller (same DI shape as the probed `ResourceSnapshot`); `from_fraction` derives
  `ceiling = fraction·(total−floor)`, clamped to `(0,1]` and fail-OPEN on a
  nonsense fraction (≤0/NaN ⇒ whole usable box, so a misconfigured quota never
  wedges a tenant). `decide_with_tenant_quota(snap, fp, floor, budget)` runs the
  box gate `decide` FIRST, then the tenant ceiling — the quota NEVER bypasses
  `decide` (a job that can't fit the box is refused regardless of quota headroom;
  admission is never weakened, [ADR 0046](0046-blut-resource-broker.md)/0047), and `budget=None` is byte-identical
  to `decide` (un-tenanted parity). Gate part: `cargo test -p blut admission::` =
  12 pass (+5: None==decide parity, over-ceiling refuses even with box room,
  under-ceiling admits, box-refusal-dominates-quota, from_fraction ceiling +
  fail-open/clamp). clippy 0; fmt+scope clean. Model review PASS WITH NITS → all 3 verified
  as matching the module's existing conventions (`floor_gib` mirrors `decide`'s
  param + `DEFAULT_FLOOR_GIB`; `pub` fields match the `ResourceSnapshot`/`Footprint`
  DI pattern; `GIB as f64` is a required u64 cast, clippy-0-confirmed). This is the
  pure decision primitive + type. **REMAINING (own increment):** the STATEFUL
  executor wiring — track per-tenant in-flight RAM, release on stage completion,
  thread the run's tenant into the admission call — plus the still-open items
  above (plans/ε tenant columns, ingest_job tenant, mesh/notify refusals). ADR
  stays In progress._
- _2026-07-12 — M2.0 CONTROL REPAIR (meta status only): frontmatter corrected
  from `proposed` to `in-progress` so generated views match the Status section
  and landed tenant/cache/quota primitives. The stateful executor counter,
  release-on-completion, tenant propagation, remaining DB columns, and
  Restricted mesh/notify refusals are still open._
- _2026-07-12 — M2.1 TENANT ENFORCEMENT LANDED (engine main `0ce79d6`)._
  Tenant policy now loads from `$BLUT_TENANTS_CONFIG` or
  `~/.config/blut/tenants.toml`, accepts exactly one of static equal shares or
  explicit fractions, and refuses malformed, duplicate, unknown, non-positive,
  non-finite, or overcommitted configuration. `TenantAdmission` takes one
  coherent resource snapshot and `TenantQuotaTracker` holds the check and
  whole-job reservation under one mutex; its RAII permit releases accounting on
  success, error, cancellation, and unwind. Fractional policy refuses when
  `MemTotal` is unmeasured, while an unconfigured `default` tenant preserves the
  prior 100 percent probe-miss behavior. Executor concurrency is bounded by the
  tenant ceiling and refuses sub-GiB ceilings that its integer semaphore cannot
  represent safely. Recipe, HPO, and plan-resume paths all use this seam; HPO
  and resume caches are tenant-prefixed. Every launched job persists a canonical
  tenant marker, and lineage ingestion plus reindex read that marker without
  downgrading a restricted row to `default`. Execution and stage contexts carry
  `Tenant`; privacy-ledger schema v2 signs the tenant into each round and hash
  chain while retaining default-tenant v1 read compatibility. Restricted work
  is refused before P2P classification, again by the coordinator, again by the
  dispatch matrix, and again at peer receipt. The engine API also records that
  cookbook authors retain arbitrary Ratatui/sidecar TUI freedom and only use
  BLUT launch/lifecycle glue. Named gates: default
  `cargo test -p blut --test tenant_isolation` = 6 passed; P2P
  `cargo test -p blut --features p2p --test tenant_isolation` = 7 passed; the
  all-target/all-feature suite, strict clippy, fmt, wasm `blut-types` check,
  `scripts/scope_guard.sh`, and cold/warm `first_cookbook` example are green.
  Independent spec and standards reviews found no blocking defect. The required
  commit reviewer returned PASS WITH NITS; its alleged reservation race was
  verified false because the mutex spans check plus increment and the admission
  function adds the new footprint, and its drop-order claim was verified false
  because exact reservation subtraction is commutative. **REMAINING:** the ADR's
  literal `blut-notify` refusal cannot be measured because that sidecar is not in
  the engine repository. The default-feature gate also cannot exercise its P2P
  assertion, so the explicit P2P variant above remains required. This ADR stays
  In progress and Completion remains blank until a real notify path and the full
  fail-closed acceptance surface exist._
- _2026-07-12 — M2.1 ACCEPTANCE TAIL COMPLETE (engine main `550f910`)._
  `Tenant`, `DataClass`, `TrustLevel`, `DispatchMatrix`, and the shared
  `custody_allows_off_box` rule now live in the WASM-safe `blut-types` keystone;
  the old `blut::tenant`, `blut::trust`, and `blut::p2p::trust` paths remain
  additive re-exports. Tenant wire identity is validated and redacted on parse
  failure; leading/trailing-dot path segments are rejected so filesystem
  normalization cannot alias a sealed tenant with a non-restricted identity.
  `crates/blut-notify` is a real standalone sidecar with no normal dependency
  edge back into the engine or Ratatui. It accepts envelopes on stdin rather
  than argv, applies custody before any sink sees the envelope, refuses both a
  Restricted tenant and a Restricted data class at an off-box boundary, permits
  node-local handling, redacts `Debug`, and maps malformed wire input to one
  fixed error that does not echo malformed wire values. The named default gate
  now compiles and executes the same
  two-stage graph under `research/dev` and `clinical/prod` against one shared
  cache base: both runs report zero hits/two misses, while the materialized key
  sets are identical below disjoint tenant roots. The P2P variant starts a real
  QUIC `Coordinator` and invokes its concrete `DispatchSubmitter` seam for both
  clinical/Public and research/Restricted work; both refuse before manifest,
  pending-task, or transport output. The notification gate invokes the real
  executable and proves restricted summary bodies never reach stdout/stderr;
  refusal stderr contains only tenant/class policy metadata, while malformed
  wire values are not echoed. Default tenant gate 7/7, P2P tenant gate 9/9, and
  `blut-notify` 2 library + 3 CLI tests passed. Workspace all-target and engine
  all-feature suites, strict Clippy, fmt, both WASM targets, scope guard, diff
  checks, and the cross-repo partition gate passed. Independent Standards and
  Spec reviews returned PASS. Mandatory commit review returned PASS WITH NITS;
  its verified tenant path-identity concern was fixed and the follow-up commit
  re-reviewed. Cookbook authors still retain arbitrary Ratatui/sidecar TUI
  freedom: this increment adds transport/custody seams, not a BLUT widget
  framework._
- _2026-07-13 — FREEZE._ Meta-repo documentation gates passed; generated ADR
  views move 0096 from `in-progress` to `accepted`, and Completion records exact
  reviewed engine SHA and all three nonzero acceptance commands.

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut --test tenant_isolation -- --nocapture`;
  `cargo test -p blut --features p2p --test tenant_isolation -- --nocapture`;
  `cargo test -p blut-notify`; plus the full repository gates listed in the
  final Progress Log entry
- **result:** the same graph executed independently in two tenant namespaces;
  the real coordinator refused both Restricted custody axes before dispatch;
  the real notification process refused restricted and malformed envelopes
  without disclosing summary bodies or malformed wire values; quota, registry,
  lineage, privacy, and default
  tenant parity assertions remained green
- **git_sha:** engine `550f910531834c3a97e20497d5c69cb85b21dbf0`  ·  **artifact sha256:** n/a (source-only change)

## Related Decisions

[ADR 0061](0061-blut-p2p-compute.md) (clinical hard-block — this makes it a structural namespace boundary),
[ADR 0095](0095-blut-rbac-auth.md) (RBAC binds tokens to the tenant axis this defines), ADR 0107 [not federated] (roadmap —
tenancy/governance), [ADR 0078](0078-blut-starlark-typed-dynamic-dag.md) (cache key algorithm unchanged; tenant is a store
prefix — extends, does not supersede), [ADR 0085](0085-blut-deployment-registry.md) (per-tenant registry pointers),
[ADR 0046](0046-blut-resource-broker.md)/0047 (per-tenant broker sub-envelopes), [ADR 0079](0079-blut-symmetric-mesh.md)/0080 (mesh + ε-ledger
gain tenant scope).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: a `restricted` artifact is ever found in a shared cache path, on a peer
node, or in a shared-tenant web/notify surface; if a cross-tenant cache read
succeeds without the explicit `shared/` opt-in; if namespacing changed the
[ADR-0078](0078-blut-starlark-typed-dynamic-dag.md) fingerprint (cache correctness regression); or if a tenant can exceed its
broker sub-envelope and starve another. Add a periodic scan asserting no
`restricted` bytes exist under any non-restricted namespace path.
