# BLUT — Software Architecture Specification

Version 0.1.0, proposed draft. This is BLUT's source architecture contract,
not an assertion that any requirement has been satisfied. `war sas propose
0.1.0` records its bytes; only a human may accept that revision. Warrants
trace to the stable requirement IDs in §106 and phases in §98.

## Scope and boundaries

BLUT is a domain-neutral semantic compiler for ML pipelines and a host
execution engine. The graph core declares determinism, effect, and partiality;
lowering targets MCU ahead-of-time, host stream, and durable host realms.
`src/` implements the engine, `crates/blut-graph-core/` the portable graph
contract, and `crates/blut-*` sidecars and adapters. Domain-specific recipes
and data remain in external cookbooks. Cookbook-owned TUIs may use Ratatui or
another toolkit; BLUT's generic cockpit is optional.

Source presence is not qualification. The current source version is
`0.2.0-alpha.1`; the repo README's scaling ladder and
`docs/DISTRIBUTED_VALIDATION.md` bound distributed claims. A phase exits only
on the stated evidence and governed records, never on a filename, tag, or
green structural check alone.

## 6.10. Levels

| Level | Unit | Closure authority |
|---|---|---|
| Release | accepted revision of this SAS | human SAS acceptance |
| Objective | one §98 phase | resolved Warrants and phase exit evidence |
| Requirement | one §106 row | resolved implementing Warrants |
| Warrant | one bounded intervention | human authorization and resolution |
| Stage | one dispatchable checkpoint | recorded evidence and independent verification |

## 98. Implementation phases

### Phase 1 — Establish governance

Deliver:

- OpenWarrant repository configuration, authored SAS, local gate definition,
  and first adoption Warrant;
- human-owned signer and role registers;
- a proposed SAS revision, followed by human acceptance, Warrant
  authorization, independent verification, and human resolution.

Exit:

- `BLUT-WAR-0001` has a human-signed resolution with independent verification
  and recorded gate evidence. A passing `war check` alone is not exit.

### Phase 2 — Prove single-box contract

Deliver:

- bounded Warrants for graph semantics, effect-directed execution, durable
  recovery, and an external cookbook with a toolkit-independent TUI;
- positive and refusal tests tied to exact source versions and fixtures.

Exit:

- each accepted requirement has a resolved Warrant and reproducible local
  runtime evidence; evidence scope distinguishes engine from cookbook.

### Phase 3 — Prove sidecar and operations boundaries

Deliver:

- bounded Warrants for web read-only authorization, event ingress,
  notification custody, observability, and operator-facing surfaces;
- threat/refusal fixtures and restart tests for each external boundary.

Exit:

- independent verification records cover declared trust boundaries and the
  relevant CI/runtime gates, with unresolved findings explicit.

### Phase 4 — Prove distributed rungs

Deliver:

- separate Warrants for multi-GPU, multi-node, cluster, and P2P placement;
- real-hardware or independent-host evidence where each rung claims it.

Exit:

- each promoted rung has its own reproducible proof. Component or loopback
  tests alone cannot promote a multi-host claim.

### Phase 5 — Release train

Deliver:

- package, documentation, kind smoke, publication, provenance, and rollback
  records for a named candidate, with clean required CI gates.

Exit:

- human release decision cites exact candidate and evidence; no alpha source
  or historical 1.x tag is silently treated as a public 1.0 release.

## 106. Architecture requirements index

Stable IDs; changing an accepted row requires a controlled SAS revision.
`war check` validates references, not whether the claim is true.

| ID | Requirement |
|---|---|
| BLUT-SAS-RQ-001 | This SAS is accepted by a human and each authorized Warrant pins the accepted revision. |
| BLUT-SAS-RQ-002 | A human authorizes and resolves each delivery Warrant; an agent cannot perform either act. |
| BLUT-SAS-RQ-003 | Resolution requires bounded gate evidence and verification independent of the performer. |
| BLUT-SAS-RQ-004 | Graph kind checks, rewrites, and lowering preserve the declared semantic identity over stated fixtures. |
| BLUT-SAS-RQ-005 | Effect and partiality declarations select tested runtime protocols, including explicit refusal and gap paths. |
| BLUT-SAS-RQ-006 | Durable resume proves recovery and cache identity under named crash and replay fixtures. |
| BLUT-SAS-RQ-007 | External cookbooks can run typed plans and their own TUI without adopting BLUT's widget toolkit. |
| BLUT-SAS-RQ-008 | Web, ingress, notification, and telemetry sidecars enforce their declared authorization and custody boundaries. |
| BLUT-SAS-RQ-009 | Distributed capabilities are claimed rung by rung only at the scale directly exercised. |
| BLUT-SAS-RQ-010 | A release candidate has reproducible package, documentation, smoke, CI, provenance, and rollback evidence before publication. |
