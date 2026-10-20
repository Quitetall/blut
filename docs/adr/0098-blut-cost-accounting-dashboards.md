<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0098 — Build cost/chargeback dashboards and fail-closed budget guards over the cloud-queue cost ledger

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-08-03`

## Context

Cloud spend is the line item an operator running remote compute watches most
closely, and it is a gap across the orchestrator field: Dagster, Prefect,
Metaflow, and Ray have no native chargeback — teams bolt on Kubecost or a cloud
provider's billing console after the fact. [ADR 0082](0082-blut-cloud-queue.md) landed the cloud compute
queue plus a **cost ledger** that records, per submitted job, the provider,
resource class, wall-time, and billed cost. That ledger exists and is written,
but nothing reads it: there is no per-run / per-tenant / per-provider spend
rollup, no budget cap that can actually stop a run before it overspends, and no
forecast of where an in-flight campaign's burn is heading. This ADR extends ADR
0082 (it does not supersede it) by turning that raw ledger into governance:
dashboards plus a fail-closed budget guard.

## Decision

Add cost accounting on top of the [ADR 0082](0082-blut-cloud-queue.md) cost ledger in two layers. **(1)
Engine layer — the budget guard:** a pure lineage/ledger read in `src`
(charter-fine, no server) that, at job-submission time, sums prior spend for the
run/tenant against a declared budget cap and **refuses admission when the cap
would be exceeded** — a fail-closed STOP routed through the existing broker
admission path ([ADR 0046](0046-blut-resource-broker.md)/0047), never bypassing it; a missing or unreadable cap
is treated as cap=0 ⇒ refuse, not as "unlimited." **(2) Sidecar layer — the
dashboards:** `crates/blut-web` ([ADR 0083](0083-blut-interface-stack.md)) renders spend rolled up per run,
per tenant, and per provider, plus a linear burn-rate forecast (spend-to-date ÷
progress) with a projected time-to-cap. Every cost type consumed by both layers
lives in `blut-types` so the guard, the CLI (`blut cost report`), and the web
view read one schema. No PHI or clinical-tenant cost data crosses into a remote
provider ([ADR 0061](0061-blut-p2p-compute.md) hard-block remains fail-closed); cost rows for clinical
tenants stay local-only.

## Rationale

- **Native chargeback is a real differentiator.** Making per-tenant/provider
  spend a first-class, queryable rollup is something the incumbents punt to
  external tooling; BLUT already owns the ledger, so it can own the answer.
- **A budget cap is only real if it can STOP.** A dashboard that reports overspend
  after the money is gone is theater. Putting the cap check on the admission path
  means the cap is enforced before the provider is billed — the same fail-closed
  posture as every other governance gate in the system.
- **Reuse the enforcement path we already trust.** The cap check refuses via the
  broker admission gate rather than inventing a second kill path, so there is one
  place a run gets stopped and one audit trail.
- **Forecast turns accounting into planning.** Spend-to-date over run progress
  gives a time-to-cap projection an operator can act on mid-campaign instead of
  post-mortem.
- **Clinical isolation is non-negotiable.** Cost rows for PHI-bearing tenants
  never leave the local host, consistent with [ADR 0061](0061-blut-p2p-compute.md).

## Alternatives Considered

- **Report-only dashboards, no enforcement (the Kubecost shape).** Rejected: it
  leaves the one thing operators actually want — a hard spending ceiling —
  unbuilt, and overspend is only discovered after billing. Visibility without a
  STOP is not governance.
- **Enforce the cap inside the cloud-queue submitter as a bespoke check, outside
  the broker.** Rejected: a second admission/kill path that can drift from the
  broker's, duplicating the fail-closed logic [ADR 0046](0046-blut-resource-broker.md)/0047 centralizes. The cap
  must ride the existing admission gate.
- **Defer to the cloud provider's own billing alerts.** Rejected: provider
  alerts are per-account, lagging (hours), and blind to BLUT's per-run/per-tenant
  attribution — they cannot stop a specific run, only notify after the fact.

## Consequences

- Upside: BLUT ships native chargeback + a hard, pre-billing budget ceiling —
  ahead of the incumbents on cost governance.
- Upside: the [ADR 0082](0082-blut-cloud-queue.md) ledger stops being write-only; `blut cost report`, the web
  dashboard, and `blut-notify` budget-threshold SLA rules all consume it.
- Downside: budget caps become load-bearing config — a mis-set or absent cap
  fails closed and blocks submission (by design), which will surface as "why did
  my run get refused?" support load until operators internalize cap=0⇒refuse.
- Downside: forecasting is linear and therefore wrong for bursty/non-uniform
  workloads; it is advisory, and the hard STOP is the cap, not the forecast.
- Constrains: cost record schema in `blut-types` is now shared by engine guard +
  web + CLI; changes need a Progress-Log entry.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** engine-side `budget_guard` (lineage/ledger read hooked into the
  broker admission path) + `blut cost report` CLI; `blut-web` cost views
  (per run/tenant/provider + burn forecast); cost record types in `blut-types`.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `scripts/scope_guard.sh && cargo test -p blut --test budget_guard && cargo test -p blut-web --test cost_rollup`
  - `pass when:` `exit==0` — the budget_guard test asserts a job whose projected
    spend crosses the cap is REFUSED via the broker admission path (not a bespoke
    exit), that a missing/unreadable cap refuses (cap=0, not unlimited), and that
    a clinical-tenant cost row is never emitted to a remote provider; the
    cost_rollup test asserts per-run/tenant/provider sums and the forecast match
    a fixture ledger to the cent.
- **Rollback:** disable the budget_guard admission hook (jobs admit on broker
  policy alone, as before this ADR) and drop the `blut-web` cost views; the ADR
  0082 ledger continues to be written, just unread. Additive-only, clean revert.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-08-03 — BUILT + FROZEN (engine `e042529`)._ Both layers landed over one
  schema in `blut_types::cost`.
  * **Money is integer micro-dollars, never a float.** The predecessor ledger
    ([ADR 0067](0067-remaining-work-toward-vision.md) T3.1e) recorded `f64` abstract units and its own comment warned a
    currency ledger must be fixed-point; this ADR's gate requires sums matching
    "to the cent", which floating point cannot promise under repeated addition.
  * **One implementation of the rollup arithmetic**, in the keystone, called by
    the guard, the ledger reader, and the dashboard — duplicating it is how a
    system refuses a job over a figure its own dashboard disagrees with.
  * **Fail-closed:** a missing, unreadable, malformed, zero, or tenant-less cap
    all resolve to `BudgetCap::REFUSE_ALL` ($0.00, which refuses) — never
    "unlimited". `budget_guard` returns the broker's own `AdmitDecision`, so a
    budget refusal travels the same path as a memory refusal rather than a
    bespoke exit that would skip admission's auditing and containment.
    Restricted-tenant rows are filtered once, in shared code, before anything is
    summed for export — while staying fully visible to LOCAL accounting, since
    local-only is not the same as invisible. `forecast()` returns `None` for
    progress of 0/negative/>1/NaN rather than fabricating a projection, because
    a projected total of `0` reads as "free".
  * Found and fixed by its own test: `to_dollars_string` split on the CENT
    divisor, rendering `$0.99` as `$99.00`.
  * Noted for ADR 0092 [not federated]: `dot_blut()` now exists in three modules (sensord, cli,
    cost); consolidating it is precisely the ownership question 0092 owns, so it
    was left alone rather than refactored mid-ADR.

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** passed
- **gate_cmd:** `bash scripts/scope_guard.sh && cargo test --test budget_guard`
  and `cargo test --test cost_rollup` in `crates/blut-web` (the sidecar is a
  standalone workspace, so its half runs from the crate directory)
- **result:** scope guard PASS. budget_guard: a job crossing the cap is REFUSED
  through the broker `AdmitDecision` with a legible "$10.01 > $10.00" reason; a
  missing AND a malformed cap both refuse; another tenant's spend cannot consume
  this tenant's budget; the clinical row is withheld from export yet still
  counted locally. cost_rollup: per-run / per-tenant / per-provider sums and the
  burn forecast match the fixture to the cent ($15.00 at 25% progress projects
  $60.00, flagged over a $20.00 cap; zero progress yields no projection).
  Engine 19/19 suites (947 tests), blut-types 30, blut-web 5 suites; clippy 0
  everywhere, fmt clean, wasm32 keystone green.
- **git_sha:** —  ·  **artifact sha256:** —

## Related Decisions

[ADR 0082](0082-blut-cloud-queue.md) (cloud queue + cost ledger — this extends it), [ADR 0096](0096-blut-multi-tenancy.md) (GPU/resource
scheduling that feeds resource-class cost), ADR 0108 [not federated] (tenancy/quota governance),
[ADR 0046](0046-blut-resource-broker.md)/0047 (broker admission path the cap rides), [ADR 0061](0061-blut-p2p-compute.md) (clinical/PHI
hard-block), [ADR 0083](0083-blut-interface-stack.md) (blut-web sidecar host).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: a run overspends past its declared cap without being refused (the STOP
leaked); a missing cap ever defaults to unlimited instead of refusing; clinical
cost data appears in a remote-provider ledger row; the cap check grows a second
kill path outside broker admission; or the reported rollups diverge from the raw
[ADR 0082](0082-blut-cloud-queue.md) ledger totals.
