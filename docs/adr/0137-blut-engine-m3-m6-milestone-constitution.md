<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0137 — Commit the BLUT-engine M3–M6 milestone constitution — bind each remaining premier milestone to its constituent ADRs and one fail-closed gate, and fix "engine 1.0" as the crates.io release distinct from the ADR-0118 [not federated] family gate

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress` (accepted 2026-07-19; in progress 2026-08-27)

## Status

`Accepted 2026-07-19`

`In progress — 2026-08-27`

## Context

The premier-heavyweight campaign has a milestone structure — Month 1 … Month 6
(M1…M6) — but it lives **only** in an uncommitted planning file outside the
repository [redacted: private path]. The repo has never
carried the M-number → work-bundle → gate mapping. What the repo *does* carry is
orthogonal to milestones:

- **A premier gap-map proposal** [redacted: private path] — the 26 gap ADRs (0085–0112),
  organized by *capability group* and *ROI tier*, not by month.
- **The individual ADRs in the decision log** — each capability's decision.
  Most are `accepted` (decided, not built); a few (0083, 0095, 0096, 0099, 0100,
  0101, 0102, 0110, 0112, 0133) are built and/or frozen.

The consequence: "finish M3, then M4/M5, then M6" has no reviewable, CI-visible
definition of *done*. ADR 0083 (the interface stack) was driven to a hard,
frozen endpoint precisely because it named its own increments and gates in the
repo. M3–M6 have had no equivalent. This ADR closes that gap by **importing the
roadmap's milestone structure into the repo as a governing decision** — so the
plan is reviewable, and each milestone freezes against a concrete gate rather
than a private checklist.

There is also a load-bearing ambiguity to settle. Two documents say "1.0":

- **This campaign's M6** = the **BLUT engine** as a publishable OSS orchestrator:
  `cargo install blut`, dist automation, a docs site, a green kind smoke, and a
  truthful README. Self-contained; reachable inside this campaign.
- **ADR 0118 [not federated]** = the **platform family** 1.0: a set of independently
  versioned platform artifacts [redacted: non-engine product detail] under one atomic `GO`/`NO-GO` gate,
  currently `NO-GO`. That is the whole *platform*, not the
  orchestrator, and it needs normative specs, conformance packs, and signed
  evidence far beyond engine code.

Conflating them would let a `cargo publish` launder the family's missing
platform evidence [redacted: non-engine product detail], or — worse — hold the engine's
release hostage to the entire platform. They must be named as disjoint gates.

## Decision

**1. Import the M3–M6 milestone structure into the repo, here.** This ADR is the
committed, reviewable definition of the remaining BLUT-engine milestones. The
uncommitted roadmap plan remains a working note; where it and this ADR disagree,
**this ADR governs** (amended only by a dated Progress Log entry). Each milestone
is a bundle of already-accepted capability ADRs plus one fail-closed acceptance
gate. A milestone is *done* when every constituent ADR is frozen and the
milestone gate passes on the exact source; a milestone card here is filled and
its Progress Log entry appended at that point.

**2. The milestone → ADR → gate binding** (the contract; see the table in the
Implementation Plan). Ordering follows the gap-map's ROI tiers, not calendar
months — the "M" labels are retained for continuity with the roadmap and the
owner's directives, not as a schedule.

**3. "Engine 1.0" (M6) is the crates.io/OSS release gate, and is explicitly
disjoint from ADR 0118's [not federated] family gate.** M6 freezing publishes the `blut` crate
family and flips the OSS switch; it makes **no** family-1.0, platform-1.0, or
clinical claim, and it does not require any ADR-0118 artifact to pass. Reaching
`GO` on ADR 0118 [not federated] is a separate, later track this campaign does not touch.
Neither gate may be cited as evidence for the other.

**4. Charter and governance invariants carry forward unchanged.** Every M3–M6
increment obeys ADR 0034 (engine stays HTTP-free; servers live in sidecars),
routes mutations through the exec bridge (ADR 0083 §4 / 0093), never bypasses
broker admission (ADR 0046/0047) on a triggered run, and holds the clinical
hard-block (ADR 0061/0096) fail-closed at every new ingress. A milestone gate
that would require violating one of these is itself wrong and blocks the
milestone.

## Rationale

- Committing the milestone map makes "done" falsifiable and reviewable — the same
  discipline that let ADR 0083 reach a clean freeze, applied to the rest of the
  campaign. A private checklist cannot gate CI or survive a context reset.
- Binding milestones to *existing* accepted ADRs (rather than re-deciding the
  work) means this ADR adds governance, not scope — each capability keeps its own
  decision record, and this one only sequences and gates them.
- Naming the two "1.0"s as disjoint gates prevents both failure modes: an engine
  release over-claiming platform maturity, and a platform gate freezing the
  engine's OSS availability. Each can succeed on its own evidence.
- Retaining the "M" labels while sequencing by ROI tier keeps the owner's
  directives ("finish M3, then M4/M5, then M6") legible without pretending the
  work is calendar-bound.

## Alternatives Considered

- **Drive M3–M6 straight off the uncommitted plan file (no ADR).** Rejected: the
  milestone spec stays invisible to review and CI, and a context reset loses it —
  exactly the fragility this ADR removes. (This was offered and declined at
  authoring time.)
- **Fold M3–M6 into ADR 0118 [not federated].** Rejected: 0118 is the platform-family
  constitution with an atomic platform-wide gate [redacted: non-engine product detail]; the engine's OSS
  release is a strict, smaller subset that must be able to ship without the
  family passing. Folding them recreates the exact conflation this ADR forbids.
- **Write one ADR per milestone (four ADRs).** Rejected as redundant: the
  capability ADRs (0093, 0094, 0085, 0086, 0090, 0091, …) already own the
  per-feature decisions. A milestone needs a *binding + gate*, not a fourth copy
  of a decision already recorded — one constitution referencing them is leaner
  and cannot drift from them.
- **No M6 at all; publish crates ad hoc.** Rejected: the README already
  over-claimed a crates.io release once (the campaign's origin defect); an
  explicit, gated release milestone with a truth pass is the correction.

## Consequences

- Upside: M3–M6 gain a committed definition of done; each milestone freezes like
  0083 did, with a Progress Log audit trail.
- Upside: the engine can reach a truthful OSS 1.0 independent of the platform
  family's readiness; "1.0" stops being ambiguous in this repo.
- Downside: two "1.0" gates now coexist and must be kept from cross-citing — this
  ADR's disjointness clause is the guard, and any doc that conflates them is a
  defect to fix.
- Constraint: this ADR does not itself build anything. It is `accepted` on
  authoring; it only reaches `Complete — frozen` when **all four** milestone
  cards below are filled (i.e. M6 has shipped). Until then it governs in-flight.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

Milestone → constituent ADRs → fail-closed gate. Ordering by gap-map ROI tier.

| Milestone | Scope (constituent ADRs) | Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL) | Status |
|---|---|---|---|
| **M3 — web read-only + eventing** | 0093 (web dashboard v1, read-only over the M2 sidecar), 0094 (file/spool/cron sensors + `blut sensord` + webhook ingress in blut-web + `blut-notify` rules + SLA) | `cargo test -p blut-web --test dashboard_readonly` (SSE reflects a fixture state transition; every route side-effect-free; a `restricted` fixture is not served) **and** `cargo test -p blut --test sensor_trigger && cargo test -p blut-notify --test sla_notify` (one broker-admitted dispatch per unique event, dedupe holds, an SLA breach writes `sla.jsonl` + one PHI-redacted notification). Both 0093 and 0094 frozen. | **DONE 2026-07-19** — 0093 frozen (engine `a3ac797`, SSE gate); 0094 frozen (engine `c30d584`, cookbook [redacted: private repository], both gate halves + webhook). |
| **M4 — deployment registry + live control + experiments** | 0085 (deployment registry: `blut plan publish/rollback/history`, `--plan name@vN`), exec bridge (built in 0083 §4; M4 adds the `~/.blut/audit.jsonl` end-to-end integration test), 0090-start (experiments tables + `blut exp`) | publish→rollback round-trip recorded in the registry; a cancel-from-web lands an `audit.jsonl` row (bridge integration test under a scratch `XDG_RUNTIME_DIR`). 0085 frozen; 0090 experiment slice frozen. | **DONE 2026-07-19 (verified)** — 0085 frozen (`registry_roundtrip` green: publish→rollback), 0090 frozen (`registry::` 16 green); exec-bridge audit met by M2 blut-web tests (`anonymous_mutation_is_denied_and_audited`, `viewer_token_cannot_mutate_operator_can` — cancel/run → `audit.jsonl` row in a scratch tempdir). |
| **M5 — secrets + registries + checks** | 0086 (secrets: `~/.blut/secrets.age`, `SecretRef`, redaction), 0090 (model + dataset registries: `registry://name@prod`, `dataset://name@v`), 0091 (check stages: `check_jsonl` + `check_tensor_stats`), 0095 (RBAC-lite — already accepted/built) | secret-echo redaction test (a plan echoing a secret produces a redacted stream; a secret never enters the PlanSpec IR or events); `registry://…@prod` pin recorded in lineage. 0086/0090/0091 frozen. | **DONE 2026-07-19 (verified)** — 0086 frozen (`secrets_leak` 3 green: redaction), 0090 frozen (`registry::` 16 green: model/dataset pins), 0091 frozen, 0095 frozen. All constituent gates re-run green on `c30d584`. |
| **M6 — engine crates.io 1.0** | dist automation + docs site + k8s-operator kind smoke + `cargo publish` chain (`blut-types` → `blut` → `blut-dsl` → `blut-notify`; the sidecars `blut-tui`/`blut-web`/`blut-operator` are `publish=false` and ship as release binaries) + README/RELEASING truth pass + 0112 (ecosystem integrations, as applicable) | `cargo publish --dry-run` green for the whole chain in dependency order; kind smoke (`scripts/k8s_kind_smoke.sh`) green; README carries no false release claim. **STOP before the real `cargo publish`** — the actual crates.io push is an owner-gated, irreversible external action. | **AUTOMATION DONE 2026-07-19 (engine `9720704`); publish PENDING owner.** Built: `release.yml` (tag→gate+sidecar binaries; manual owner-gated publish job behind the protected `crates-io` env + token), `docs.yml`+`book.toml`+`book_src` (mdBook→Pages, `mdbook build` verified, includes-not-copies), `k8s_kind_smoke.sh` (CRD install + BlutPlan admission), RELEASING.md refreshed. Chain proven: `blut-types` full dry-run clean; `blut`/`blut-dsl`/`blut-notify` carry `version=` pins and package once the upstream is indexed (the standard publish-order dependency the retry loop absorbs). README already truthful; `blut` name available. **The only remaining step is the irreversible `cargo publish`, which this card forbids me to run.** |

- **Deliverable:** four filled milestone cards above, each backed by its frozen
  constituent ADRs and a passing gate command, plus this ADR frozen once M6 ships.
- **Acceptance gate (this ADR's own freeze):** all four milestone rows show
  `frozen` constituent ADRs and a recorded passing gate; the engine crate chain
  passes `cargo publish --dry-run` in order; and no doc cross-cites this gate and
  ADR 0118's [not federated] family gate as evidence for one another.
- **Rollback:** this ADR is governance, not code — superseding it (a new ADR)
  reverts the milestone bundling with zero engine impact. Individual milestone
  work rolls back via its own constituent ADR's rollback clause.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. The milestone constitution cannot be satisfied while its constituents are open. It defines a milestone as done only when every constituent ADR is FROZEN and the milestone gate passes on the exact source; M3-M6 bundle the 0104-0109 capability ADRs (only 0106 and 0109 are federated), every one of which this same pass has just measured as unbuilt. Its own gate is red on scripts/k8s_kind_smoke.sh, which exists nowhere in the fleet.

- `gate_cmd:` a decision-log citation gate [redacted: private path] (the floor: it proves the record still points at files that exist; it does not claim the decision is implemented).
- _2026-08-27_ — gave this record a concrete citation-gate `gate_cmd` and removed its closure-debt entry [redacted: private path].

- _2026-07-19 — opened._ Imports the roadmap's M3–M6 structure into the repo as
  a governing decision (owner chose "commit a milestone ADR first"). Fixes M6 as
  the BLUT-engine crates.io/OSS 1.0, explicitly disjoint from ADR 0118's [not federated]
  platform family gate (owner chose "BLUT engine crates.io 1.0").
  Predecessor context: ADR 0083 (interface stack) frozen 2026-07-19 — its M1
  (types+dispatch), M2 (blut-web read-only + exec bridge + blut-tui extraction +
  embedded Leptos dashboard), and M3 (blut-worker deletion) are all complete, so
  the campaign's *engine* M1–M2 substrate for the surfaces M3 builds on is in
  place. Next execution block = M3 (0093 read-only dashboard SSE gate, then 0094
  eventing/SLA), per the gate table above.
- _2026-07-19 — M3 CARD DONE._ Both constituent ADRs frozen: **0093**
  (engine `a3ac797`) — the SSE live-tail `GET /api/jobs/{id}/events` completed
  the read-only v1 (the JSON surface + Leptos dashboard were M2; v2 control is
  the M2 exec bridge), gated by `dashboard_readonly`. **0094** (engine
  `c30d584`, cookbook [redacted: private repository]) — the trigger substrate + admission-gated
  dispatch, SLA rules + PHI-redacted breach notifications, and the runnable
  `blut sensord` / `blut sla check` / webhook-ingress wiring, gated by
  `sensor_trigger` + `sla_notify`. Next execution block = M4 (0085 deployment
  registry + the exec-bridge audit integration test + 0090 experiments).
- _2026-07-19 — M4 + M5 CARDS DONE (by verification)._ Audit found every
  constituent ADR already frozen with a passing gate, built in the
  2026-07-10→12 capability sprint that preceded this campaign's interface work:
  0085 (deployment registry), 0086 (secrets), 0090 (experiment/model/dataset
  registries), 0091 (check stages), 0095 (RBAC). Their named gates re-run green
  on engine `c30d584` (`registry_roundtrip`, `secrets_leak`, `registry::` ×16),
  and M4's one genuinely-new requirement — a cancel/run through the exec bridge
  landing an `audit.jsonl` row — is met by the M2 blut-web tests
  (`anonymous_mutation_is_denied_and_audited`,
  `viewer_token_cannot_mutate_operator_can`). So M4/M5 required no new build,
  only verification (honest per the Validation clause: each card cites a
  reproducible gate command, not a bare "frozen"). The campaign's remaining
  capability gap was always the interface stack (0083 = M2) + eventing/web
  (0093/0094 = M3), now closed. **Only M6 (engine crates.io 1.0) remains.**
- _2026-07-19 — M6 AUTOMATION DONE; the publish is the sole remaining step, and
  it is the owner's (engine `9720704`)._ Built and verified everything short of
  the irreversible crates.io push: `release.yml` (tag → full gate + sidecar
  binaries attached to the Release; a SEPARATE manual `workflow_dispatch`
  publishes the `blut-types → blut → blut-dsl → blut-notify` chain behind the
  protected `crates-io` environment + `CARGO_REGISTRY_TOKEN`, fail-closed
  without it), `docs.yml` + `book.toml` + `book_src/` (the mdBook site → GitHub
  Pages; `{{#include}}`s the top-level docs so there is no drift; `mdbook build`
  green, all pages render real content; ADRs linked to the meta-repo, not
  vendored), `scripts/k8s_kind_smoke.sh` (installs the operator CRDs on a kind
  cluster and asserts a `BlutPlan` CR is accepted server-side), and a refreshed
  `RELEASING.md`. Publish-chain readiness proven as far as is local:
  `blut-types` full `--dry-run` packages + verifies clean; the downstream crates
  each carry a `version=` path-dep and package once the upstream is on crates.io
  (a pure publish-ORDER dependency — `blut` fails locally only with "no matching
  package `blut-types` found", exactly what the ordered publish + the retry loop
  resolve). The `blut` crate name is confirmed available (owner) and the README
  carries no false release claim. **The remaining action — the actual
  `cargo publish` — is owner-gated and irreversible; per this ADR's M6 card and
  the standing operating rule on irreversible outward-facing actions, I do not
  run it.** This ADR (0137) therefore stays `accepted`/in-progress: its own
  Completion clause freezes only once M6 has *shipped*.
- _2026-07-20 — M3 EVIDENCE REPAIRED; M4/M5 REVERIFIED; M6 REMAINS
  FAIL-CLOSED (engine `d249762` + `df6d312` + `49bce35`)._ A contract audit found the M3
  card's 2026-07-19 ADR-0094 evidence premature. `df6d312` supplies the missing
  file/spool/cron daemon, signed webhook, exact child-admission handshake,
  shared SLA freshness evaluator, and declarative runnable notifier tails; it
  also keeps the ADR-0093 dashboard export read-only and Restricted-filtered.
  The three named M3 gates and the full locked workspace/sidecar suites pass.
  M4/M5 accepted-ADR states and their registry, audit, secrets, redaction,
  check-stage, and RBAC gates were re-run and require no further code tail.

  M6 automation is materially stronger: `d249762` made release packaging and
  publish refusal fail-closed; `df6d312` adds the first-cookbook extension gate,
  compiles the benchmark harness in hosted CI, requires designated quiet-host
  evidence for the exact SHA only on the manual publish path, packages all four
  sidecar binaries, and adds the single-sourced eventing operations page;
  `49bce35` keeps the notifier's packaged AGPL text single-sourced without
  weakening the repository leanness ratchet. The
  mdBook, pinned workflow lint, real kind admission smoke, four binary builds,
  rustdoc, wasm, deny, leak, scope, TUI render, and package-list gates pass.
  Nevertheless the M6 row's own release gate is **not complete**: no committed
  `release-0.2` quiet-host benchmark baseline exists; mandatory V4 Pro commit
  review is unavailable (`Transport closed`) for these commits; manifests are
  still `0.2.0-alpha.1` while this constitution requires engine `1.0`; and
  downstream `cargo publish --dry-run` verification cannot resolve the exact
  unpublished upstream versions until the owner-gated chain begins indexing.
  The real publish remains irreversible and explicitly unauthorized here.
  Therefore prior language that publication was the *only* remaining step is
  superseded by this correction, ADR 0137 stays accepted/in-progress, and ADR
  0118 [not federated] does not begin until these M6 decisions and gates are closed.

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** —  *(freeze only when all four milestone cards are filled and M6 has shipped)*
- **gate_cmd:** —
- **result:** —
- **git_sha:** —

## Related Decisions

ADR 0083 (interface stack — M1/M2/M3 substrate this builds on; the exec-bridge
M4 extends), ADR 0093 (web control plane — M3), ADR 0094 (eventing/triggers/SLA
— M3), ADR 0085 (deployment registry — M4), ADR 0090 (experiment/model/dataset
registries — M4/M5), ADR 0086 (secrets — M5), ADR 0091 (check stages — M5), ADR
0095 (RBAC — M5), ADR 0034 (charter — engine stays HTTP-free), ADR 0046/0047
(broker admission — never bypassed by a trigger), ADR 0061/0096 (clinical
hard-block at every ingress), ADR 0112 (ecosystem — M6), ADR 0118 [not federated] (platform
family 1.0 — the DISJOINT platform gate M6 does not touch or claim),
the premier gap-map proposal [redacted: private path] (the capability/tier source this
sequences).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: a milestone is declared done while a constituent ADR is unfrozen or its
gate unmeasured; if any doc cites the engine crates.io 1.0 and the ADR-0118 [not federated]
family `GO` as evidence for each other; if an M3–M6 increment adds an HTTP
listener to the engine binary, bypasses broker admission on a triggered run, or
serves a Restricted artifact off-box; or if the private roadmap plan is treated
as authority over this committed ADR. Watch that each milestone freeze cites a
reproducible gate command bound to exact source — not an audit, a clean worktree,
or a plan.
