<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0072 — BLUT error-domain + fault-attribution model, adversarial hardening, fuzz testing

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (Proposed 2026-06-30 · Accepted 2026-08-27 · Complete — frozen 2026-08-27)

> **Numbering note.** This record was written when two of the records it cites
> carried earlier numbers: "ADR 0051" below is now [ADR 0055](0055-ingredient-taxonomy-and-registry.md)
> and "ADR 0071" is now [ADR 0156](0156-blut-experiment-execution-ux.md).

- **Extends:** ADR 0051 (= [ADR 0055](0055-ingredient-taxonomy-and-registry.md), Cookbook → Course → Recipe → Ingredient taxonomy), ADR 0071 (= [ADR 0156](0156-blut-experiment-execution-ux.md))
  (experiment-execution UX — advisory stages, `blut results`; this ADR builds on the
  advisory mechanism and the `StageFailure`/`ErrorDomain` scaffolding ADR 0071 left
  unused)
- **Why this exists:** running the codec ablation grid surfaced a UX problem (ADR
  0071); a follow-on ask surfaced a *diagnosability* problem — when a stage fails,
  nothing in the system says WHOSE code is at fault (blut engine internals vs. this
  cookbook's Rust glue vs. the underlying Python process/environment), and nothing
  attributes a failure to the cookbook's own Course/Recipe/Ingredient taxonomy. The
  engine already ships a structured-error mechanism (`StageFailure`/`ErrorDomain` in
  `error_domain.rs`) built for exactly this — it has zero call sites. This ADR wires
  it up, extends it with a fault-origin axis, and pairs it with fuzz/property testing
  so the new plumbing (and the pre-existing hot paths it touches) don't regress
  silently.

## 1. `FaultOrigin` — a new axis, not a new type

```rust
pub enum FaultOrigin { Engine, Cookbook, External }
```

- `Engine` — a bug or contract violation in `blut` core itself (containment/launcher
  plumbing, admission math, DAG/executor internals).
- `Cookbook` — a bug in this cookbook's Rust glue, a malformed recipe arg, or a
  contract the cookbook owns on both sides (e.g. a kernel exiting 0 without writing
  its promised checkpoint).
- `External` — the underlying process or environment: a Python script crash, OOM,
  a missing binary/venv, a generic OS anomaly.

This extends the EXISTING `StageFailure{code, domain, stage, severity, context,
message}` (`error_domain.rs`) rather than introducing a parallel error type —
`StageFailure` already solves anyhow-downcast survival and has lineage/
`status.jsonl` wiring via `extract_failure_summary`; duplicating that plumbing for
a fresh type would be pure cost. `StageFailure` gains `origin: FaultOrigin` (builder
method, defaults `Cookbook` in `::new()` — most failures genuinely are cookbook
glue; only the `Engine`/`External` minority needs an explicit override), plus
`course: Option<String>`, `recipe: Option<String>`, `ingredient: Option<String>`
for the Course/Recipe/Ingredient axis (§2). `course` is a plain `String`, not
`Option<Course>` — `error_domain.rs` sits in `framework`, lower-level than
`recipes`; taking a dependency on `recipes::recipe::Course` would be a layering
inversion for a cosmetic label.

`FailureSummary` (the flattened struct actually persisted to `status.jsonl`) is
extended to mirror all of these — including `stage`, which it was **already
silently dropping** before this ADR (an unrelated pre-existing bug, fixed in
passing since leaving it would mean `blut errors show` sees enriched data only on
a live in-process error, never from a replayed `status.jsonl`).

**Scope: observational, not behavioral.** `origin` is a label that gets printed
(CLI, `blut errors show`) and stored (lineage). It does NOT change retry policy,
admission behavior, or control flow this round. Behavioral consumption (e.g. "only
auto-retry on `External`") is a natural extension, deliberately deferred — keeping
this pass additive and low-risk.

## 2. Course/Recipe/Ingredient stays cookbook-side

ADR 0051's taxonomy (`Course` enum: `DataPrep, Pretrain, Train, Eval, Gate, Export,
Pipeline, User`) is real and typed but lives purely in `Recipe::CATEGORY` — never
threaded into the engine's `Plan`/`StageContext`. This ADR does **not** thread it
through the engine. Two reasons: (1) it's a cookbook-owned taxonomy, growing an
engine-core type for it is disproportionate; (2) `recipe_name: String` is already
threaded automatically through `StageContext` (`executor.rs:1593` ← `plan.name()`)
— a precedent that looked reusable but isn't a perfect fit, since threading `course`
the same way would require every one of ~22 stage call sites to manually pass the
right `Course` value, copy-pasted from that recipe's `CATEGORY` const, with zero
compiler-enforced link between the two. Instead: a single, small
`recipe_name → Course` lookup table lives in the cookbook (built once from the 7
recipes' `CATEGORY` consts, which the cookbook already owns) and is resolved
internally by a shared `stage_failure()` builder helper. Zero engine changes, zero
per-call-site drift risk.

## 3. `BackendError → StageFailure`: the real fix

The cookbook's `BackendError` (`Spawn`, `Wait`, `Failed{outcome,...}`,
`MissingOutput`, `ContainmentUnavailable`, `Launch`) already has decent variant
granularity describing *how* a contained run failed. Today every variant except
`Failed{outcome:OomKilled}`/`MissingOutput` collapses into an identical
`StageError::Backend(anyhow::anyhow!(e))` — a Python crash, a missing binary, and a
broken cgroup wrapper are indistinguishable in the logs. This ADR classifies each
variant by tracing its actual call graph (not guessing):

| Variant | Origin | Why |
|---|---|---|
| `Spawn` (`ErrorKind::NotFound`) | `External` | env missing the binary |
| `Spawn` (other kind, `inv.contained`) | `Engine` | engine's containment/systemd-run wrapper construction |
| `Spawn` (other kind, bare) | `External` | generic OS spawn anomaly |
| `Wait` (`inv.contained`) | `Engine` | engine-owned process-supervision bookkeeping |
| `Wait` (bare) | `External` | generic OS/tokio anomaly |
| `Failed{outcome: OtherFailure}` | `External` | genuine Python script crash |
| `MissingOutput` | `Cookbook` | the kernel-output contract is cookbook-owned on both sides |
| `ContainmentUnavailable` | `Cookbook` | fires only when a cookbook call site skips `blut_env` — the production path always sets `BLUT_JOB_DIR`/`BLUT_STAGE_NAME` |
| `Launch` | `Engine` | cluster launcher spec construction lives in engine's `config::launcher` |

Each variant's existing rich fields (status, peak_ram_bytes, etc. — already
captured, currently `Display`-stringified into an opaque message) thread into
`StageFailure.context` instead of being lost.

## 4. Ingredient attribution: two choke points, not hot-loop instrumentation

`IngredientSpec` (Python, the generic cookbook core's `spec.py`) has a single construction choke
point, `build_ingredient()` (its `registry.py`) — traced by reading it. Once
built, an ingredient is invoked per-step inside each trainer's own hot loop with no
shared wrapper. Instrumenting every hot-loop call site (loss, optimizer.step(),
scheduler.step(), ...) across every trainer would be disproportionate and risky on
a hot path. Instead: a `BLUT_ERROR {json}` stdout sentinel (mirroring the existing
`BLUT_METRIC` convention) is emitted at exactly two points — `build_ingredient()`
(precise `ingredient_kind`/`ingredient_name` attribution for config/package/
license-gate failures) and each trainer's top-level entrypoint (best-effort
attribution for anything mid-run: exception type/message, `ingredient=None` unless
cheap context like phase is already at hand). The full Python traceback still
prints normally either way — this is additive telemetry, not exception-swallowing.

## 5. Fuzzing policy

Two tools, deliberately not one: **cargo-fuzz** (confirmed available — nightly
toolchain + `cargo-fuzz` already installed) targets genuine untrusted-bytes
deserialization boundaries (recipe TOML, config YAML, job spec JSON, bincode
artifacts, subprocess stdout sentinels, and — highest priority — the `p2p` bincode
`BundleManifest`/JSON transport, the only network-attacker-reachable boundary in
the system). Structure-aware (`Arbitrary`-derived) where feasible. CI runs a short
smoke budget per target (`-max_total_time=30`); full corpus-building campaigns are
a documented local workflow, not a CI gate — fuzzing is unbounded by nature and a
30s CI smoke check catches regressions without blowing CI time budgets.
**proptest** (stable Rust, no new toolchain) covers pure decision-logic invariants
that fuzzing doesn't reach efficiently: `workers_to_fit_and_saturate`'s never-OOM
property, arg-range guards rejecting their full invalid domain (not just the
hand-picked unit-test values), parser functions never panicking on adversarial
strings.

## 6. Adversarial audit, split in two

A single audit pass after all this code lands would conflate "pre-existing bug" with
"bug this hardening work introduced." Split: **C0** runs BEFORE any Part A/B code
lands, sweeping the current baseline (panics on the executor hot path,
error-swallowing, race conditions, resource leaks on early-return, never-OOM bypass
paths, the p2p trust boundary) — confirmed findings are fixed and committed first,
so Part A's diff lands against an already-hardened baseline. **C1** runs AFTER,
reviewing ONLY the new code, with a specific dimension the baseline pass can't
cover: is `FaultOrigin` ever mis-attributed in a way that would mislead an
operator (constructed failure scenarios per `BackendError` variant, checked against
the §3 table).

## Rollback

Every piece is additive/opt-in: `FaultOrigin` is a new enum with a safe default,
new CLI subcommands (`blut errors list/show`) don't change existing command
behavior, new `fuzz/` directories don't touch the build unless invoked, the
`BLUT_ERROR` sentinel is new stdout output that existing parsers ignore (unknown
prefix). Rollback is "drop the new code" — no data migration, no wire-format
change.

## Out of scope

PCCP gate semantics/wiring/a domain gate config `[redacted: private path]` stay the user's, per the
standing carve-out from ADR 0071. The one PCCP-adjacent touch in this program is a
read-only fuzz target on the two existing PCCP-verdict JSON parse calls — zero
semantic or wiring change, flagged for an easy veto. Behavioral consumption of
`FaultOrigin` (§1) and per-step ingredient instrumentation (§4) are explicitly
deferred, not rejected.

## Progress Log

- _2026-08-27_ — gave this record a concrete citation gate `[redacted: private path]` (it proves the record still points at files that exist; it does not claim the decision is implemented) and removed its closure-debt entry `[redacted: private path]`.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Its decision (stated without a Decision heading) is one this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

## Completion / Resolution

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** Both halves of the Decision are in the tree. The error-domain model is a real surface, not a type nobody calls: `src/cli/errors.rs` implements 'blut errors list [--json]' over the registered ErrorDomain catalogs and a per-job failure breakdown, citing this ADR's own A4 at :3 and :14. The adversarial half is there too -- `fuzz/fuzz_targets/` holds six targets covering the declarative recipe, erased artifact, failure summary, plan spec and both P2P wire surfaces.
