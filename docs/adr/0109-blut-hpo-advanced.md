<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0109 — Run Bayesian, multi-objective, and cost-aware HPO as a DAG-native surrogate search that fans trials out into cache-keyed nodes

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress`

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text; correct an earlier statement with a *new* dated
> entry beneath it. When the deliverable passes its gate, fill **Completion** and
> set Status to `Complete — frozen YYYY-MM-DD`.

## Status

`Proposed 2026-07-10`

- **Extends, does not supersede:** [ADR 0067](0067-remaining-work-toward-vision.md) Slice B (ASHA/PBT/TPE already run as
  DAG nodes via `ControlPolicy::Spawn(SpawnDelta)` + `KillOnNaN`), [ADR 0078](0078-blut-starlark-typed-dynamic-dag.md)
  (`map_output` runtime fan-out).

`Accepted 2026-08-27`

`In progress — 2026-08-27`

## Context

BLUT already searches hyperparameters *inside* the DAG: ADR 0067 Slice B landed
ASHA, PBT, and TPE as first-class control policies — a scheduler node emits
`ControlPolicy::Spawn(SpawnDelta)` and trials become real graph nodes, killed on
NaN, resumed from the content-addressed cache. That covers the samplers whose
next trial depends only on cheap bookkeeping (a rung table, a KDE over past
scores). It does **not** cover three things practitioners reach for Optuna and
Ray Tune to get:

1. **Model-based (Bayesian) optimization** — a Gaussian-process / TPE-with-EI
   surrogate that fits observed (config → objective) pairs and proposes the next
   trial by *maximizing acquisition*, not by sampling a fixed prior. This is
   where sample-efficiency comes from when each trial costs GPU-hours (a
   full training run takes hours `[redacted: codec research]`).
2. **Multi-objective** — codec work is inherently ≥2-objective (a reconstruction
   quality metric vs. compression ratio vs. ε-privacy budget vs. GPU-hours). We
   want a **Pareto front**, not a scalarized single winner, so the operator picks
   the trade-off.
3. **Cost-aware** — a trial that reaches nearly the best quality in 40 GPU-min is
   worth more than one that is marginally better in 6 GPU-h
   `[redacted: codec research]`. The search must be able to weight the
   acquisition by measured $/GPU-hours/ε.

The charter ([ADR 0034](0034-blut-scope-charter.md)) forbids an always-on tuning server; the broker
([ADR 0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md)) must never be bypassed. So the surrogate cannot be a daemon that
launches jobs behind the scheduler's back — it has to live *on the DAG*, using
the same Spawn/`map_output` machinery ADR 0067/0078 already proved.

## Decision

Add a **surrogate-search control policy** — one new `ControlPolicy` variant
(`ControlPolicy::Surrogate`) plus a `Surrogate` trait — that slots into the
exact scheduler-node seam ADR 0067 Slice B built, and reuses ADR 0078
`map_output` fan-out to expand proposed trials into nodes. Concretely:

- A **search-driver node** holds the surrogate state (an append-only Parquet of
  `{trial_id, config, objectives[], cost}` rows, content-addressed like any
  artifact). Each scheduler tick it (a) fits/updates the surrogate over completed
  rows, (b) proposes up to `parallelism` new configs by maximizing a pluggable
  **acquisition function**, (c) emits `ControlPolicy::Surrogate(SpawnDelta)` whose
  deltas `map_output` expands into trial nodes. Every proposed trial is an
  ordinary DAG node with an ordinary cache key — **re-proposing an
  already-evaluated config is a cache hit, i.e. free**, no special-casing.
- **Surrogates shipped:** `Tpe` (multivariate, extends the landed univariate
  TPE), `Gp` (Gaussian process w/ Matérn kernel), and `Random`/`Grid` as
  baselines. Trait is `fn observe(&mut self, rows) ; fn propose(&self, n,
  space) -> Vec<Config>`.
- **Multi-objective** is native: objectives are a vector; the driver computes the
  **non-dominated (Pareto) set** and, for GP, uses expected-hypervolume-improvement
  (qEHVI) as the acquisition. The deliverable of a multi-objective study is the
  **Pareto-front Parquet + a `pareto.json`**, not a single "best".
- **Cost-aware** is an acquisition wrapper: `AcqPerCost = acq(config) /
  predicted_cost(config)`, where `predicted_cost` is a second cheap surrogate
  fit on the measured `cost` column (GPU-seconds, $ from the cloud-queue billing
  row, or ε drawn from the PrivacyLedger). Weighting is a study-config knob, off
  by default.
- **The broker is never bypassed.** The surrogate *proposes* a config (which may
  include batch size / model width); the trial node still goes through broker
  admission (ADR 0046/0047). An over-budget proposal is **refused by the broker
  and recorded as an infeasible trial** (a real observation the surrogate learns
  from — infeasibility is signal), never forced onto the box. The search reads
  budgets; it cannot raise them.

No engine HTTP, no dylib: the surrogate is compiled-in Rust (or an `exec`'d
sidecar scorer for a heavy GP, same pattern as ADR 0078's DSL front-end),
selected by study config. Clinical/PHI studies inherit the existing fail-closed
hard-block — a PHI corpus cannot be a tuning objective source.

## Rationale

- **Reuse over reinvention.** The Spawn + `map_output` + cache-key machinery is
  the hard part and it already exists and is tested (0067/0078). A surrogate is
  just a smarter proposer feeding the same seam — the diff is a trait + one
  policy variant + a Parquet state file, not a scheduler.
- **Cache-keyed trials make Bayesian search honest.** Model-based search
  re-proposes near-duplicate configs; content-addressing turns those into free
  cache hits instead of wasted GPU-hours, which is exactly the sample-efficiency
  argument for going Bayesian in the first place.
- **Pareto, not scalarization, matches the domain.** The tension between
  quality, ratio and privacy budget `[redacted: codec research]` means a single
  weighted sum hides the decision the operator actually needs to make. Returning
  the front defers the trade-off to a human, which is correct for a
  clinical-adjacent codec.
- **Infeasible-as-observation** keeps the broker sovereign *and* informative: the
  search learns the feasible region rather than repeatedly proposing OOM configs.

## Alternatives Considered

- **Embed Optuna (Python) as the search brain, call BLUT to run trials.** This
  is the Ray Tune / Optuna-ask-tell shape. Rejected: it puts a Python event loop
  *above* the engine driving job submission — the inverse of BLUT's "search is a
  DAG node" invariant — and it would need a persistent ask/tell process that the
  charter's no-daemon rule forbids. We keep Optuna's *algorithms* (TPE, qEHVI) but
  not its control inversion. A one-shot `exec`'d scorer is allowed; a resident
  ask/tell server is not.
- **Scalarize multi-objective into one weighted score, reuse the landed
  single-objective ASHA path unchanged.** Rejected: the weights are precisely the
  unknown — picking them up front is the decision we're trying to inform, and a
  fixed scalarization collapses the Pareto front we need to show the operator.
- **A standalone `blut-tune` sidecar daemon that owns trial scheduling.**
  Rejected: duplicates the scheduler, and a separate scheduler would have its own
  path to spawn work — a second door around broker admission. One scheduler, one
  admission gate.

## Consequences

- New load-bearing surface: the `Surrogate` trait + `ControlPolicy::Surrogate`
  become part of the control-policy contract; adding a sampler is a trait impl,
  not a scheduler change.
- The study-state Parquet (`{config, objectives[], cost}`) is a new
  content-addressed artifact type; lineage/cache tooling must treat it as data.
- Cost-aware search binds to the cloud-queue billing row and the PrivacyLedger —
  those become read dependencies of the driver node (read-only, charter-fine).
- GP with qEHVI is O(n²)–O(n³) in observed trials; for large studies the driver
  node's own compute must itself be broker-admitted (it is just another node) so
  a runaway surrogate can't starve the box.
- ASHA/PBT/TPE (0067) stay exactly as-is; this is strictly additive — a study
  that names no surrogate behaves identically to today.

## Implementation Plan  *(the contract — amend only by a dated Progress Log entry)*

- **Deliverable:** `ControlPolicy::Surrogate` + a `Surrogate` trait with `Tpe`,
  `Gp` (Matérn + qEHVI), `Random` impls; a driver node that reads/writes the
  study-state Parquet and emits Spawn deltas via `map_output`; a
  `pareto.json` + front Parquet as a multi-objective study's output; a
  `cost_weight` study knob wrapping acquisition by predicted cost.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut --lib hpo:: && blut study run
    tests/fixtures/study_multiobj.toml --max-trials 24 &&` a Pareto validator
    script `[redacted: private path]` over `runs/<id>/pareto.json`
    <!-- Selector corrected 2026-08-17: the original `hpo_surrogate::` names no
         module and selected 0 of 973 while EXITING 0. `--lib hpo::` selects
         121. See the Progress Log entry of that date. -->

  - `pass when:` exit==0 AND `pareto.json` non-dominated set has ≥2 points AND a
    re-proposed config resolves as a cache hit (no re-run) AND an over-budget
    proposed trial is recorded `status=infeasible` (broker refused it, box did
    not OOM).
- **Rollback:** feature-gate `hpo-surrogate`; disabled ⇒ `ControlPolicy::Surrogate`
  unavailable and studies fall back to the landed ASHA/PBT/TPE path, byte-for-byte
  unchanged.

## Progress Log  *(append-only; newest last)*

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. The surrogate policy is unbuilt: neither ControlPolicy::Surrogate nor a Surrogate trait exists in src/. Its gate is red on tests/fixtures/study_multiobj.toml, a fixture that was never written -- and an earlier pass over this ADR established the substantive finding that its clauses 3 and 4 are unreachable as stated, and that two of its objective pairs do not actually conflict, so the multi-objective front it assumes needs restating before the fixture is worth writing.

- `gate_cmd:` a decision-log citation gate `[redacted: private path]`
<!-- instrument added 2026-08-27: this record carried no CONCRETE gate and sat in
     the decision log's closure-debt list [redacted: private path]. The citation
     gate is the FLOOR: it proves the record still points at files that exist. It
     does not claim the decision is implemented. -->
- _2026-08-27_ — gave this record a concrete `gate_cmd` (the citation gate) and removed its closure-debt entry `[redacted: private path]`. Clause 5 accepts an instrument change only when the same commit appends here.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Reviewed under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. It declares a runnable gate_cmd. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

- _2026-07-10 — opened._
- _2026-07-12 — M2.0 CONTROL REPAIR: planning contract only. No
  `hpo_surrogate` module or named target exists in the current engine, so the
  listed Rust substring filter would execute zero tests. Existing ASHA/PBT/TPE
  work remains prior art; it is not evidence that this surrogate-search ADR is
  implemented. Add a named nonzero target with the implementation._

- _2026-08-09/10 — MULTI-OBJECTIVE CORE + THE GATE'S VALIDATOR LANDED. Still
  Proposed; this closes one slice of the gate, not the gate._

  _Scope correction first, because the gap is narrower than the Decision
  implies: the engine already carries a 2,597-line `src/hpo` (SearchSpace,
  `Sampler`, TPE, PBT, ASHA, median-stop, leaderboard, plan builder). What was
  genuinely absent is the MULTI-OBJECTIVE half — `TrialResult` holds one `f64`
  and `HpoManifest` one `"max"`/`"min"` mode, so the Pareto front this ADR
  calls the deliverable ("not a single 'best'") had nowhere to live._

  _Landed `src/hpo/pareto.rs` (engine `0225d42`, `055dc3e`, `b671c7f`): a PURE
  module — `Direction` (parsing the max/min vocabulary `HpoManifest` already
  uses, rather than inventing a second spelling), `dominates` /
  `non_dominated` / `pareto_front`, `ParetoReport` (the `pareto.json` shape),
  and `hypervolume_2d`, the scalar front-quality measure qEHVI will improve
  against. Hypervolume is 2-D ON PURPOSE: exact N-D is a different algorithm
  and a wrong general one is worse than an honest narrow one._

  _The load-bearing property is that **non-finite objectives are excluded, not
  ranked**. A NaN comparison is false in both directions, so an unguarded
  implementation reports a NaN-bearing point as non-dominated and parks a
  failed trial on the front as a fake optimum; `ParetoReport` instead REPORTS
  such trials in `unmeasured_trials`, so a study whose trials mostly failed
  cannot read as a clean small front. 13 tests cover that plus mixed per-axis
  directions, duplicate vectors (both retained — hiding one would conceal a
  real tie), arity mismatch, and empty/degenerate input. Engine lib suite 854
  passed / 0 failed; clippy 0; scope_guard PASS. The hypervolume sweep was
  additionally cross-checked against brute-force grid integration over 120
  random fronts with mixed directions, duplicates and dominated points — worst
  relative error 0.11%, which is the grid's own discretization error._

  _Also added the gate's named Pareto validator `[redacted: private path]`,
  which did not exist. It does not merely count the front: it **recomputes
  dominance and refuses to take the producer's word for it**, rejecting a
  dominated point on the front, a non-finite objective, an arity or direction
  mismatch, a trial reported as both optimal and unmeasured, a duplicate
  `trial_id`, and an undersized front. `--self-test` proves it is non-vacuous —
  it accepts the real fixture and rejects 9 distinct corruptions; a checker
  never shown to fail is indistinguishable from one that returns success
  unconditionally. `tests/fixtures/pareto_multiobj.json` is VERBATIM serde
  output of `ParetoReport`, emitted from the engine rather than hand-written,
  and engine `b671c7f` pins those exact bytes so a field rename fails the Rust
  build with a message naming both files instead of silently breaking the
  Python validator at gate time._

  _NOT done, and the gate is NOT claimed: `blut study run`,
  `tests/fixtures/study_multiobj.toml`, the `Gp` surrogate + Matérn kernel,
  qEHVI itself, the multivariate TPE extension, the `Grid` baseline, the
  `AcqPerCost` cost-aware wrapper, the search-driver node, and
  broker-refusal → infeasible-trial recording._

  _One correction for whoever continues: this ADR says "one new `ControlPolicy`
  variant (`ControlPolicy::Surrogate`)", but `ControlPolicy` is a **trait**
  (`src/framework/control.rs:221`), not an enum. The correct shape is a new
  impl of that trait returning `Control::Spawn(SpawnDelta)` — the seam exists
  and its docs already name TPE spawns._

- _2026-08-16 — THE Gp SURROGATE AND THE STUDY LOOP LANDED. Still Proposed; the
  CLI that drives them is the remaining piece, and this ADR's own gate selector
  is vacuous._

  _**The gate's first command selects nothing.** `cargo test -p blut
  hpo_surrogate::` matches 0 of 947 tests — measured, not inferred. There is no
  `hpo_surrogate` module; the work lives in `hpo::pareto`, `hpo::surrogate`,
  `hpo::gp` and `hpo::study`. The 2026-07-12 entry already flagged this class of
  error for the Rust suffix and it was never corrected. The selector that
  actually executes is `cargo test -p blut --lib hpo::` (91 tests today). This
  is recorded rather than silently substituted, because a gate nobody can run
  is not a weaker gate, it is no gate._

  _`src/hpo/gp.rs` (engine `bfc9b2d5`): Gaussian-process regression with a
  Matérn 5/2 kernel, expected improvement, and Monte-Carlo EHVI. Matérn over
  the squared exponential deliberately — SE assumes an infinitely
  differentiable surface and is overconfident between observations. Targets are
  standardised so the GP's zero prior mean is the sample mean; configs encode
  through the same t-space transform every other sampler uses, scaled by
  DECLARED support rather than observed data, since scaling by the data would
  shift the encoding as trials arrive and silently invalidate a fitted model.
  No linear-algebra dependency: Cholesky and both substitutions are ~60 lines,
  and pulling `nalgebra` into an engine ADR 0034 keeps lean is a dependency
  decision that should not be made by accident._

  _**Scope correction on qEHVI, stated so nobody over-reads it.** What ships is
  the **q=1** case by Monte Carlo, not batch qEHVI. Batch needs the joint
  posterior over a batch plus inclusion–exclusion over overlapping boxes; it is
  not implemented, and `ehvi_mc` takes a single candidate so the limit cannot
  be mistaken. Objectives are also modelled independently, one GP each — the
  standard EHVI setup, but it does not model correlation between objectives, so
  a study where quality and cost are strongly coupled gets an optimistic
  variance._

  _`src/hpo/study.rs` (engine `a6945329`): `StudySpec` (the TOML a study is
  declared in) and `StudyLedger` (the ask/admit/record loop), implementing the
  two behaviours this ADR names in its `pass when` clause. A RE-PROPOSED CONFIG
  IS A CACHE HIT — trials are keyed by canonical config identity, sorted by
  dimension name, so a sampler emitting dims in a different order cannot buy a
  second evaluation of a point already paid for; the repeat schedules nothing
  and does not even consult admission. AN OVER-BUDGET PROPOSAL IS REFUSED AND
  RECORDED `infeasible`, never spawned, because admission is consulted BEFORE
  the spawn — so the refusal costs nothing and still becomes an observation.
  Two orderings are load-bearing and wrong the other way: the cache is checked
  BEFORE admission (a config already paid for must not be re-refused by a
  tightened budget, which would silently shrink the front), and a refusal is
  NOT cached (admission depends on live box state, so caching one transient
  refusal would make it permanent)._

  _`tests/fixtures/study_multiobj.toml` is the fixture the gate names by path.
  Its objectives deliberately COMPETE — a fixture whose objectives agreed would
  collapse to a one-point front and pass the ≥2-point requirement by accident.
  A test `include_str!`s it so a spec-format change breaks the build rather
  than the gate._

  _Verification beyond unit tests: the GP posterior was cross-checked against an
  independent numpy implementation using an explicit matrix inverse rather than
  Cholesky — so the check is not circular — with worst |Δmean| and |Δsd| both
  4.8e-13 across five query points. Earlier, `hypervolume_2d` was cross-checked
  against brute-force grid integration over 120 random fronts (worst 0.11%,
  which is the grid's own discretisation error), and the multivariate TPE was
  measured against the landed univariate one on a diagonal problem where the
  marginals carry no signal: 12/12 seeds versus 7/12. Engine lib 947 passed / 0
  failed; clippy 0; scope_guard PASS._

  _NOT done, and the gate is NOT claimed: the `blut study run` CLI that drives
  the ledger against the executor, and the search-driver node emitting
  `Control::Spawn`. Everything the loop needs exists and is tested; what is
  missing is the wiring that turns 24 proposals into 24 executed trials and
  writes `runs/<id>/pareto.json`._

- _2026-08-16 (later) — `blut study` LANDED. The study is now runnable end to
  end for the samplers whose driver exists; the model-based ones are refused,
  not downgraded._

  _`blut study run <spec.toml> --max-trials N` and `blut study report [job]
  --spec <spec.toml>`. Report re-derives the front from a finished job without
  re-running it, which is also how a study that crashed after its trials
  completed is salvaged._

  _The run path OWNS NO EXECUTION. It translates the `StudySpec` into the
  `HpoCommand` that already exists and calls `run_hpo`, so trials go through the
  same plan build, the same broker admission, the same DAG cache and the same
  early-stop. ADR 0092 [not federated] invariant 4 requires CLI/TUI/local/remote
  to resolve through one application module; a second launcher here would drift
  from `blut hpo run` on retries, admission and cancellation — the exact failure
  that invariant exists to prevent. `run_hpo` now returns the job id it ran,
  because resolving "the latest HPO job" afterwards would race a concurrent run._

  _`reconstruct_study` is the multi-objective sibling of
  `results::reconstruct` and differs where it matters. That function keeps the
  best value of ONE metric, so a trial reporting it is finished; here a trial is
  `Done` only when it reported ALL declared objectives, because a config that
  optimised error but never emitted a ratio is not a trade-off and admitting it
  would put a half-measured point on the front. Per objective the best value in
  that objective's OWN direction is kept. `stage_skipped` is recorded as
  `CacheHit` — that IS the DAG cache hit this ADR relies on, reported as free
  rather than as work performed. `stage_failed` reuses the documented
  cancel-vs-crash discrimination, matching only "cancel" and never "kill", which
  would misread the OOM killer's "Killed process …" as a scheduler action._

  _**A declared-but-unreachable sampler is REFUSED, not downgraded.** `gp`,
  `mvtpe` and `grid` are implemented and tested, but the search-driver node that
  feeds them per-round results is not wired, so `blut study run` errors and says
  exactly that. A silent fallback to random search would still produce a
  plausible front and nobody would learn the model was never used. The shipped
  fixture declares `mvtpe`, so it is refused today, and a test asserts that so
  the refusal cannot rot into a silent downgrade._

  _`pareto.json` is written temp-then-rename like the manifest, so a reader
  never observes a half-written front. The run summary names every trial that
  produced no point and warns when the front holds fewer than the two points
  this ADR expects. CLI tests parse the gate's own argv verbatim. Engine lib 957
  passed / 0 failed; clippy 0; scope_guard PASS._

  _**What is left, and it is one thing:** the search-driver node emitting
  `Control::Spawn` so the model-based samplers receive per-round results. Until
  it lands the gate cannot run end to end on its own fixture, so the gate is
  still not claimed and this ADR stays Proposed._

- _2026-08-17 — THE FACTORY WIRING LANDED. The loop this ADR describes now runs
  end to end; the gate is runnable and still not claimed._

  _`blut study run` with `sampler = gp | mvtpe | surrogate` reaches the search
  driver instead of refusing. `run_hpo` gained `--algo surrogate` and an
  `--objectives` JSON channel, and the branch is a deliberate MIRROR of the tpe
  branch — same trial overlays, same `compile_fn` `FreshFactory`, same
  `ctx.with_control(with_nan_safety(...))` seam — so a study and an HPO run
  cannot diverge on how a proposed trial becomes a node. That is ADR 0092
  [not federated] invariant 4 honoured rather than a second launcher._

  _The objectives travel as JSON because the driver needs a VECTOR while
  `HpoCommand` carries a scalar metric. That join is a string, so nothing in the
  type system holds the two sides together; a test does, encoding exactly as
  `run_study` does and decoding exactly as the surrogate branch does, and
  requiring name, dotted metric and DIRECTION to survive — direction being what
  decides which way the front grows. A second test pins that ONLY the surrogate
  algo receives the vector: a single-objective scheduler handed one would
  silently reduce it to its first element and report a front it never searched
  for._

  _Refusals did not weaken, they moved to where they belong. `grid` and unknown
  names are still rejected at the CLI. A non-2-objective study is now rejected
  by `SurrogatePolicy::new` and surfaced as a launch error BEFORE any trial
  runs, because the EHVI here is 2-D and a silent fall back to random would
  produce a front while the model was never used._

  _Adding the field to `HpoCommand::Run` broke the study call site at compile
  time. That is the intended fail-closed property, not an inconvenience: a new
  HPO knob forces a decision about what a study does with it instead of
  inheriting a default silently._

  _Two tests that pinned the previous refusal were REPLACED, not quietly
  deleted: `the_gate_fixture_is_refused_today_and_says_why` became
  `the_gate_fixture_now_dispatches_to_the_driver`, which additionally asserts
  the fixture declares exactly two objectives — the driver refuses anything
  else, so the gate could not otherwise run._

  _Engine lib 973 passed / 0 failed; clippy 0; fmt clean; scope_guard PASS.
  Engine `e80c2f6f`._

  _**Why the gate is still not claimed.** Every component now exists and is
  tested, but no end-to-end study has been EXECUTED — the gate's own fixture
  names recipe `demo_codec`, which is a placeholder rather than a catalogued
  recipe, so `blut study run` on it would fail at recipe resolution. Claiming a
  gate on components that have never been run together is exactly the
  "unmeasured ⇒ FAIL" this ADR family forbids. What remains is one real study
  run against a catalogued recipe, producing a `pareto.json` that the Pareto
  validator `[redacted: private path]` accepts._

- _2026-08-17 (second entry) — the gate's first leg was corrected from
  `hpo_surrogate::` to `--lib hpo::`._

  _The defect was recorded on 2026-08-16 but left in place. It is worse than a
  wrong name: `cargo test` with a filter that matches nothing **exits 0** ("0
  passed; 973 filtered out"), so this leg could not fail. Measured today: the
  original selects 0 and exits 0; `--lib hpo::` selects 121 and passes._

  _Corrected per [ADR 0082](0082-blut-cloud-queue.md)'s precedent (fix the line, record it here). Wording
  only — the leg now asserts what it always claimed to, and the end-to-end legs
  (`study run`, the Pareto validator) are untouched and still unrun._

  _A gate-selector checker `[redacted: private path]` now refuses this class
  repo-wide, so a future gate cannot be written against a name that does not
  exist._

## Completion / Resolution  *(fill once)*

- **verdict:** —
- **gate_cmd:** —
- **result:** —
- **git_sha:** —  ·  **artifact sha256:** —

- _2026-08-17 — THE END-TO-END GATE IS BLOCKED BY THE RECIPE CATALOGUE, NOT BY
  THE SEARCH MACHINERY. No study was run; no `pareto.json` is claimed._

  _The gate needs `blut study run` against a catalogued recipe with real args.
  Measured against the live registry (`blut registry export`, [ADR 0111](0111-blut-python-sdk.md)) rather
  than assumed: **6 of the 8 registered recipes require a sealed BCS2 Training
  Snapshot** (`lamquant_joint_codec`, `lamquant_gauge_retrain`,
  `lamquant_snn_4state`, `lamquant_decoder_solo`, `lamquant_mae_pretrain`,
  `lamquant_ssl_pretrain` — `training_snapshot`/`training_snapshots` in their
  `required` list), and **no sealed snapshot exists on disk**. That is the same
  root cause that blocks [ADR 0103](0103-blut-async-compute-completion.md)'s witness._

  _The two recipes that are NOT snapshot-gated do not carry a study either:_

  _- `lamquant_measure` has 2 sweepable scalar args, and its parameter channel
    is `extra_args`/`extra_env`, both ARRAYS. `apply_overlay`
    (`src/hpo/space.rs:326`) writes dotted OBJECT paths via `insert_path`, so a
    search dimension cannot address an array element. It is not sweepable by
    this machinery at all._
  _- `hf_sft` is genuinely sweepable (lr, rank, alpha, batch_size, epochs,
    grad_accum, seq_len) and needs only `output_name` plus a dataset. But it is
    an LLM QLoRA finetune: a ~15 GB base-model pull and hours of GPU per trial
    across 24 trials, in a different domain from the objectives this ADR
    recorded (`eval.prd`, `eval.compression_ratio`). Re-pointing the fixture
    there means inventing new objectives as well as a new recipe, which is a
    superseding measurement basis and an owner decision — not a substitution to
    be made quietly._

  _So the honest state is unchanged in kind from 2026-08-16: the machinery is
  complete and tested (engine lib green, `hpo::` suite passing), and the gate is
  UNCLAIMED because no run has produced a `pareto.json`. What is new is that the
  remaining work is now a specific decision rather than "find GPU time": either
  seal a Training Snapshot pair (an ABIR-side deliverable — see the ADR 0103
  entry of the same date) so a codec study can run against the recorded
  objectives, or record `hf_sft` with new objectives as a superseding basis._

  _The fixture at `tests/fixtures/study_multiobj.toml` still names the
  placeholder `demo_codec` and still says so in its own comment. It was not
  re-pointed._

- _2026-08-18 — A SUPERSEDING MEASUREMENT BASIS IS RECORDED, AND IT RUNS.
  `tests/fixtures/study_multiobj.toml` is NOT re-pointed and NOT deleted; it
  stays as the historical record of what this ADR originally proposed._

  _**New fixture:** a study spec plus a measurement script in the domain
  cookbook `[redacted: private path]`. It encodes real EEG data with the
  project's lossless codec: CPU-only, no Training Snapshot, no GPU, ~1 s per
  trial._

  _**Why the objectives changed, and how they were chosen.** The recorded pair
  (`eval.prd` vs `eval.compression_ratio`) needs a trained neural codec that
  does not exist. The replacement pair was chosen by MEASURING candidates on
  this box, because two of the three obvious pairs do not conflict at all —
  and a fixture whose objectives do not conflict yields a one-point front and
  passes a ">=2 points" gate only by accident, which is the warning the
  original fixture carried in its own comment:_

  _- ratio vs peak RSS: at one thread RSS is FLAT across window sizes while
    ratio climbs and saturates ⇒ the large-window config dominates
    `[redacted: codec research]`._
  _- ratio vs throughput: a larger window is both smaller AND faster, because
    the codec modes carrying its speed/ratio trade-off are not enabled in the
    shipped binary `[redacted: codec research]`._
  _- **throughput vs peak RSS: conflicts, with interior structure.** Adding
    threads raises throughput and RSS together up to a point, after which
    throughput FALLS while RSS keeps climbing `[redacted: codec research]`. The
    high-thread points are genuinely dominated, so the front has a knee rather
    than two endpoints._

  _**Gate result, measured 2026-08-18 (24 trials, sampler `gp`, seed 7):**_

  _- clause 1 — `exit==0`: **PASS**._
  _- clause 2 — front has >=2 points: **PASS**. The Pareto validator reported
    8 non-dominated point(s) over ['encode_throughput','peak_memory'],
    `unmeasured_trials: []` (all 24 measured). Front preserved as evidence
    `[redacted: private path]`._
  _- clause 3 — a re-proposed config resolves as a CACHE HIT: **FAIL, and
    structurally unreachable for this basis.** The study did re-propose: 24
    trials covered only 13 distinct configs, so 11 were repeats — and every one
    RE-RAN (`stage_skipped` events: 0). The cause is exact, from the executor's
    own warning: `cache insert for stage 'lamquant_measure' failed: immutable
    object cache-invocations/<key> already contains different bytes`. The
    measure report embeds `elapsed_secs` and `peak_ram_bytes`, so identical
    args produce different output bytes and the content-addressed store refuses
    the second insert. **This is not a defect to fix here:** caching a
    throughput measurement would replay a stale timing as if freshly measured.
    Any fixture that satisfies clause 3 must have DETERMINISTIC objectives —
    which, on this catalogue, means a recipe that does not exist yet._
  _- clause 4 — an over-budget trial is recorded `status=infeasible`: **NOT
    REACHABLE through `study run` as built.** The safety half holds and was
    verified: with a 400 GiB footprint the broker refused — `admission refused:
    RAM footprint 400.0G exceeds box capacity (62.6G total − 6G floor)` — and
    the box did not OOM. But admission is evaluated ONCE up-front, so it fails
    the whole STUDY rather than recording one trial infeasible; adding a
    footprint-moving dimension to the space did not change that. The ADR's
    Related Decisions cite ADR 0046/0047 "infeasible-as-observation", so the
    contract expects a per-trial observation the implementation does not
    produce. That gap is a finding, not something this fixture can paper over._

  _**Two defects had to be fixed before any of this could run**, both in
  `lamquant_measure`, one of only two recipes not snapshot-gated (cookbook
  commit `[redacted: private repository]`): it wrote its primary artifact to
  the shared `job_dir`, so every run died at the last step with "nothing to
  ship: no backing files under src_root" — the recipe could not COMPLETE at
  all, and its only tests exercised its two rejection paths so nothing caught
  it; and it had no HPO-addressable parameter channel, since
  `extra_args`/`extra_env` are arrays and `apply_overlay` writes dotted OBJECT
  paths._

  _**Status therefore stays unfrozen.** Clauses 1 and 2 are met on real data;
  3 and 4 are not, and neither is met by trying harder with this fixture. What
  changed is that the remaining work is now two specific, named engineering
  questions — a deterministic sweepable recipe, and per-trial admission
  observations — instead of "no end-to-end run has happened"._

## Related Decisions

[ADR 0067](0067-remaining-work-toward-vision.md) (Slice B — landed ASHA/PBT/TPE this extends), [ADR 0078](0078-blut-starlark-typed-dynamic-dag.md) (`map_output`
fan-out reused for trial expansion), [ADR 0110](0110-blut-auto-tuning.md) (auto-tuning control policies —
sibling; a tuner may feed one config the search then explores), [ADR 0034](0034-blut-scope-charter.md)
(charter — no daemon), [ADR 0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md) (broker admission — infeasible-as-observation).

## Validation  *(ongoing)*

Wrong if: the surrogate ever launches a trial that skipped broker admission; the
driver becomes a resident process instead of a scheduler-ticked node;
multi-objective studies start collapsing to a scalar winner; or re-proposed
configs re-run instead of hitting the cache (would mean the search stopped being
cache-keyed and the sample-efficiency claim is void).
