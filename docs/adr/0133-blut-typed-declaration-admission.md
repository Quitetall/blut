<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0133 — Typed-declaration admission — the engine derives launch footprints from Stage declarations, never from interpreting cookbook recipe JSON

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (frozen 2026-07-18)

## Status

`Proposed 2026-07-17` → `Accepted 2026-07-18` → `Complete — frozen 2026-07-18`
(engine `3873c94`; cookbook and backends commits [redacted: private repository]). This ADR formalizes the direction attempted
by engine commit `ba40d6b` ("close final public-boundary gaps"), which was
deliberately NOT carried through the 2026-07-17 reconciliation rebase because it
half-replaced admission machinery that `origin/main`'s async-io line had just
extended. The direction deserves a real decision, not a conflict-resolution
default.

## Context

The engine's pre-launch RAM/VRAM admission today has TWO footprint sources that
coexist after the 2026-07-17 reconciliation:

1. **Recipe-JSON drivers (the incumbent launch path).**
   `broker::Drivers::from_args_json` parses cookbook recipe JSON *by field name*
   — `workers`, `batch`, `tier`, `latent`, `warm_fb_cache`, `in_ch` — and feeds
   `recipe_footprint`/`recipe_footprint_tuned`/`recipe_footprint_sync_base`
   (`src/cli/gate.rs`). This path powers the capabilities we actually rely on:
   the measured-peak `FootprintStore` calibration with OOM-aware self-heal, the
   ADR 0071 (renumbered; now [ADR 0156](0156-blut-experiment-execution-ux.md)) fit-and-saturate auto-tune (`admitted_workers`/`admitted_batch_size`),
   warm-cache billing (`fb_warm`), and the ADR 0103 async-I/O profile billing.
   The cost: the ENGINE knows what "tier" and "latent" mean. That is
   domain-specific cookbook knowledge inside `blut/src` — exactly the class of
   coupling the scope charter (ADR 0034) and the module-ownership contracts
   (ADR 0092 [not federated]) exist to prevent, and `API.md` already flags the `StageContext`
   carriers (`fb_warm`, `admitted_workers`, `admitted_batch_size`) as
   "transitional compatibility hooks … may move out of the engine before 1.0".

2. **Typed stage declarations (the landed beachhead).**
   `CompiledPlan::declared_footprint()` derives a generic envelope from the
   typed seam that already exists: `Stage::MEMORY_GIB` / `memory_gib_for(&Args)`
   / `gpu_request(&Args).min_vram_mib`, with a small 2 GiB compatibility
   estimate for undeclared plans. Two call sites shipped in the reconciliation:
   the `--dry-run` footprint report, and the runtime-spawn envelope guard (an
   HPO/PBT sub-plan whose largest declared stage exceeds the whole-job
   reservation is refused — warn-and-drop for a best-effort suggestion,
   fail-closed for a required map shard).

`ba40d6b` attempted to complete the flip in one commit: launch admission from
declarations only, executor memory budget = the whole-job reservation (optionally
tightened by the ADR 0096 tenant ceiling), and the engine never reading a recipe
key. It collided with the async-io admission work and was split during the
rebase: the beachhead (2) was kept, the replacement of (1) was reverted.

## Decision  *(proposed)*

Adopt **typed-declaration admission as the engine's contract**, migrating in
increments that never regress the incumbent capabilities:

1. **The seam is `Stage`, not JSON.** The engine derives every pre-launch
   footprint from typed stage declarations (`memory_gib_for(&Args)`,
   `gpu_request(&Args)`, and — new — an optional `io_profile(&Args)` for the
   ADR 0103 billing terms). A stage's `Args` are its own typed struct, so
   args-dependent footprints remain fully expressible — the *cookbook*
   implements the math the engine's `Drivers` does today, against its own types.
2. **Calibration and auto-tune become engine MECHANISM + cookbook POLICY.** The
   `FootprintStore` (measured peaks, OOM self-heal) and the fit-and-saturate
   search stay in the engine — they are domain-free mechanisms — but they key on
   a stage-provided `footprint_key(&Args)` instead of engine-parsed recipe
   fields, and deliver their verdicts through the existing typed carriers.
3. **`broker::Drivers::from_args_json` and the recipe-JSON parsing in
   `cli/gate.rs` are deleted at the end state**, together with the
   "transitional" `StageContext` language — the hooks stay, their population
   moves behind the typed seam. Cache keys (ADR 0078) are untouched throughout
   (footprints were never cache-key inputs).
4. **Parity is the migration invariant**: at every increment, the canonical
   LamQuant recipes must produce byte-identical admission verdicts (same
   refuse/admit, same memmax, same tuned workers/batch) from the typed path as
   from the JSON path, proven by an A/B gate before the JSON path is removed.

### Open Questions  *(to resolve in discussion before acceptance)*

- **Where does the calibration store's key live?** A stage-provided
  `footprint_key` keeps the engine generic but lets a cookbook accidentally
  collide keys across stages; an engine-composed key (stage name + canon-args
  digest) is collision-safe but loses the deliberate `|w` warm-variant sharing.
- **Auto-tune ownership.** Fit-and-saturate searches *candidate batch/worker
  values* — candidates are domain policy. Does the cookbook enumerate candidates
  (engine searches), or does the cookbook own the whole search behind a trait?
- **ADR 0103 profiles.** Async-I/O billing terms are currently computed from
  recipe drivers; is `io_profile(&Args)` on `Stage` the right shape, or should
  profiles be a separate declared artifact?
- **Sequencing vs the 0.2.0-alpha publish.** The public API surface ships soon;
  do we hold the `Drivers` deletion until after 0.2 (additive-only now), or land
  the flip first so the published surface is already clean?
- **The 2 GiB undeclared-plan floor** — keep, raise, or make failing-loud for
  cookbooks that never declare?

## Rationale

- **Charter alignment (ADR 0034; ADR 0092 [not federated]):** the engine interpreting `tier`/`latent`
  by string name is domain knowledge in the wrong module; every other
  domain-specific concern (governance gates, connectors, PCCP) is already
  delegated behind a typed or exec seam. Admission is the last holdout.
- **The typed seam already exists and is already load-bearing** —
  `memory_gib_for` gates executor concurrency today; `declared_footprint`
  guards runtime spawns. This ADR extends a proven contract rather than
  inventing one.
- **Portability of the OSS engine:** a third-party cookbook today gets
  second-class admission (the JSON parser knows only LamQuant's field names).
  Typed declarations make admission first-class for every cookbook.
- **Honesty:** the incumbent path's "never-OOM" framing was already corrected to
  "memory-admission accounting" in the release line; a declaration-based
  contract states plainly that admission is only as good as the declarations —
  which the calibration mechanism then improves with measurements.

## Alternatives Considered

- **Keep the dual system permanently (status quo after the reconciliation).**
  Rejected as an end state: two footprint sources with different answers is a
  standing source of drift (which one refused this launch?), and the charter
  violation remains. Acceptable only as the migration's intermediate state.
- **Complete the flip in one step (what `ba40d6b` attempted).** Rejected: it
  silently dropped calibration, auto-tune, and async-io billing richness on the
  floor — the rebase review showed those capabilities have no typed-side
  equivalent yet. Capability-regression-free migration requires the parity gate.
- **Push footprint math into recipe *metadata* (a declared JSON schema the
  engine evaluates).** Rejected: it re-creates the same coupling one level up —
  the engine would interpret a footprint DSL instead of field names, gaining
  complexity without gaining the type-checked seam `Stage` already provides.

## Consequences

- The `Stage` trait grows (at most) `footprint_key(&Args)` and
  `io_profile(&Args)` defaults — additive, default-preserving, so every
  existing stage compiles unchanged.
- The LamQuant cookbook takes ownership of its footprint model — the `Drivers`
  estimate formula moves there verbatim first (no behavior change), then can
  evolve independently of engine releases.
- The engine loses ~all recipe-JSON knowledge: `Drivers::from_args_json`, the
  `cli/gate.rs` recipe parsers, and the JSON-keyed calibration keying are
  deleted at end state. `scope_guard` gains a fingerprint for recipe-key
  parsing in `blut/src` so it cannot return.
- Until the final increment, BOTH paths run with the parity A/B gate comparing
  them — a real but temporary cost in test time.

## Implementation Plan  *(the contract — amend only via a dated Progress Log entry)*

- **Deliverable:** typed-declaration admission end-to-end (launch gate, executor
  budget, calibration, auto-tune, async-io billing) with the recipe-JSON path
  deleted and scope-guarded against reintroduction.
- **Increments:** (1) cookbook implements `memory_gib_for` reproducing the
  `Drivers` formula + parity A/B gate; (2) calibration/auto-tune re-keyed via
  the typed seam; (3) async-io profile declaration; (4) delete the JSON path +
  scope-guard fingerprint.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut admission_parity && bash scripts/scope_guard.sh`
  - `pass when:` the parity suite shows byte-identical admission verdicts
    (admit/refuse, memmax, tuned workers/batch) between the typed and JSON paths
    on the canonical LamQuant recipes at increment 1–3 boundaries; at increment
    4 the JSON path no longer exists, `scope_guard` rejects recipe-key parsing
    in `blut/src`, and the full engine + cookbook suites are green.
- **Rollback:** each increment is flag-free and additive until (4); reverting
  (4) restores the JSON path from history with zero data loss (calibration
  stores are keyed compatibly by the parity requirement).

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-17 — opened._ Beachhead already in tree (engine `89c76f5`):
  `CompiledPlan::declared_footprint()` + dry-run report + runtime-spawn envelope
  guard (best-effort spawn = warn-drop, required map shard = fail-closed,
  covered by `runtime_hpo_spawn_larger_than_job_reservation_is_rejected` and the
  map-shard rejection test). The `ba40d6b` full-flip attempt is recorded in that
  commit's message and the reconciliation notes; its executor-budget-from-
  reservation piece (`executor_memory_budget_gib`, tenant-ceiling-tightened) is
  a candidate for increment 2. **Discussion pending — do not implement past the
  beachhead until the Open Questions are resolved with the owner.**
- _2026-07-18 — OWNER DISCUSSION, positions resolved (status stays `proposed`
  pending the floor policy + formal acceptance)._
  **Resolved:**
  1. **One method, one struct:** `Stage` gains a single
     `resource_envelope(&Args) -> ResourceEnvelope` whose default composes from
     the existing `memory_gib_for`/`gpu_request` (every stage compiles
     unchanged). All new capability — calibration dimensions, tunables,
     io-profile terms — are FIELDS on the struct, versioned with defaults like a
     wire type. Trait growth stops at one method.
  2. **Calibration keys are ENGINE-composed from cookbook-DECLARED dimensions**
     (stage identity + the envelope's `calibration_dimensions`), collision-safe
     by construction; warm/cold sharing is a dimension, not a key suffix. An
     explicit `shared_calibration_group("…")` is the audited opt-in for stages
     that genuinely share physics (train vs resume). Accidental collision is
     impossible; deliberate sharing is visible.
  3. **Cookbook enumerates, engine searches, search is FIRST-CLASS:**
     fit-and-saturate leaves `cli/gate.rs` and becomes an engine subsystem over
     declared ORDERED tunables (workers-then-batch order is behavior and is
     preserved by declaration order), each carrying an explicit `monotonic`
     contract — non-monotonic dimensions get exhaustive evaluation or refusal,
     never a silently wrong bisection. Verdicts generalize the existing typed
     carriers.
  4. **De-publicize `Drivers`/`from_args_json` BEFORE the 0.2 publish**
     (`pub(crate)`/doc-hidden — a small standalone PR). This decouples the clean
     public surface from the migration's completion; internals then migrate on
     normal cadence.
  **No-regression contract (clarified against the "does the engine not already
  have calibration/auto-tune/billing?" question):** the engine's MECHANISMS
  (FootprintStore + OOM self-heal, fit-and-saturate, ADR 0103 billing) are NOT
  moved — only their INPUT source changes (`ba40d6b` regressed by starving the
  mechanisms, not by deleting them). Migration is a strangler pattern:
  Phase A = formula moves to the cookbook verbatim behind the envelope, no
  consumer; Phase B = SHADOW-MODE parity, CI-gated byte-equality on FIVE
  outputs (verdict, memmax bytes, calibration KEY STRING, tuned values, billing
  term decomposition) over the canonical corpus + property-tested args;
  Phase C = consumers flip one at a time under the green shadow; Phase D =
  delete JSON path + scope-guard fingerprint after soak. Calibration-key
  CONTINUITY is a named regression vector: early increments must reproduce the
  incumbent key string exactly; the dimension-based key format is its own later
  step with dual-write/read-prefer-new migration. The 0103 hardware witness
  bench re-runs after the billing consumer flips.
  **Still open:** the undeclared-plan floor policy (proposed: keep 2 GiB,
  fail-LOUD warning event per launch + operator strict mode that refuses
  undeclared plans) and formal acceptance.
- _2026-07-18 — ACCEPTED + PHASE A LANDED (engine `ebcd46c`, cookbook
  [redacted: private repository])._ Floor policy resolved as proposed (2 GiB kept, fail-loud;
  strict refuse-undeclared mode arrives with the Phase-C launch flip). Landed:
  **Engine** — `blut_types::envelope::ResourceEnvelope` (ram_bytes + gpu +
  ordered calibration_dimensions + audited shared_calibration_group; serde
  forward-compat), `Stage::resource_envelope` default/erased/blanket (legacy
  composition = zero behavior change), `broker::envelope_calibration_key`
  (engine-composed, byte-compat with `FootprintKey::flat` — pinned by the
  `envelope_key_continuity` test: `train_joint|3|32|2|w` exactly),
  `#[doc(hidden)]` on `Drivers`/`from_args_json` + API.md rewritten around the
  declarative surface (the de-publicize), `has_resource_declarations` + the
  dry-run loud-floor warning. **Cookbook** — `footprint_model.rs` (the formula
  verbatim beside the trainer; conservative args-only driver resolution),
  `lamquant_train_joint::resource_envelope`, and the `adr0133_parity` gate:
  byte-equality on estimate/memmax/KEY-STRING/verdict over the canonical
  tier×batch grid + a 500-point deterministic sweep — all green (cookbook 282/0
  vs engine `ebcd46c`; engine 860/0). **Phase-A findings for increment 2:**
  (a) `warm` and admitted `workers` are CONTEXT dimensions, not Args — the
  envelope bills cold/cap (conservative-HIGH) and the engine appends context
  dimensions at key-composition time next; (b) the incumbent reads
  latent/in_ch from `extra_args` tokens and IGNORES the typed
  `encoder_width`/`detail_bands` fields — the typed path is more truthful, but
  parity pins incumbent behavior; the truth-improvement lands as a deliberate,
  calibrated change, never an accident; (c) `blut-backends` rev-pins the
  engine → the cookbook carries a `[patch]` onto the meta's engine submodule
  until blut-backends' pin is bumped (follow-up). Next: increment 2 —
  calibration/auto-tune re-keyed via the seam + context-dimension append.
- _2026-07-18 — INCREMENT 2 LANDED (engine `0252027`, cookbook [redacted: private repository]): the
  context-dimension append._ `envelope_calibration_key_with_context`: runtime
  facts (auto-tuned workers, warm) REPLACE declared dimension values by NAME —
  layout + store continuity unchanged; unknown names ignored. Cookbook
  `TrainDrivers::with_context` mirrors the tuned estimate. Parity now pins the
  TUNED path too (estimate/memmax/key, `train_joint|3|32|6|w` exact) — 4/4;
  engine 861/0. Remaining: wire the store/auto-tune CONSUMERS through the seam
  (increment 2b), io-profile terms (3), delete + scope-guard (4)._
- _2026-07-18 — INCREMENT 2b LANDED (engine `a82d57f`, cookbook [redacted: private repository]):
  the first BEHAVIOR-CARRYING consumer flip._ The MAIN launch path's untuned
  arm (cli/recipe.rs) now resolves through the seam: `plan_footprint_declared`
  = max declared envelope → engine-composed key with the RECIPE identity (the
  SAME store rows the JSON path calibrates — `resolve_flat` shares the store
  internals, and the row-sharing test pins a Measured peak served to both
  paths) → `FootprintStore::resolve_flat`. Undeclared plans fall back to JSON
  unchanged; the strangler SHADOW computes both and warns on divergence.
  Parity 4/4 against the flipped engine; lib 862/0; scope PASS. **2b
  remainder:** the hpo per-trial + resume callers, and the TUNED arm (its
  worker/batch overrides change the ESTIMATE itself — needs the structured
  cost terms arriving with increment 3's io-profile envelope fields)._
- _2026-07-18 — INCREMENTS 3a+3b LANDED (engine `352bd9d`, cookbook
  [redacted: private repository])._ **3a:** `CostTerm` (affine per-unit cold/warm coefficients,
  cookbook-declared ceiling, `sync_base_excluded` for 0103-billed terms) +
  `ResourceEnvelope::cost_terms`; engine `envelope_footprint_at` /
  `envelope_sync_base` / `fit_and_saturate_env` / `shrink_to_fit_env` (ADR 0071 — now ADR 0156 —
  semantics preserved); the TUNED launch path, the resume path, both auto-tune
  searches, and the declared sync base all flip through the seam
  (shadow-refereed, JSON fallback for undeclared); `warm_context()` is the one
  blessed context read that survives Phase D. Parity: 4 engine equivalence
  tests + cookbook 5/5 pin tuned estimates, sync base, and both searches
  byte-equal to the incumbent across the grid. Full suite 979/0; cookbook
  284/0. **3b (the endpoint artifact):** `examples/first_cookbook.rs` declares
  an envelope (CI-run), README "Declare your resources" paragraph + snippet,
  API.md cost-term sentence. **3a remainder:** HPO per-trial plan threading
  (`hpo_trial_admission` passes `None` today — its callers hold the plan).
  Then Phase D per the approved plan (delete + scope-guard + freeze) after the
  soak._
- _2026-07-18 — 3a COMPLETE (engine `48dc330`, cookbook [redacted: private repository]): per-trial
  HPO envelopes + width truth-billing._ Gate fns now take the RESOLVED declared
  envelope; launch/resume scope it once, HPO bills each trial its OWN
  node-range envelope (`max_declared_envelope_in`) — a small trial is never
  billed the biggest trial's footprint (an IMPROVEMENT over the incumbent's
  per-trial JSON, which shared one recipe-level estimate shape). Runtime-spawn
  trials stay JSON (documented; the spawn-envelope guard bounds them).
  **Capability-regression audit closed:** the one at-risk item — the envelope's
  Phase-A conservative width defaults under-billing wide-encoder runs the
  incumbent ALSO couldn't see (it read extra_args tokens only) — is fixed by
  `with_model_shape` (latent from `encoder_width`, in_ch from `detail_bands`):
  exact at the true shape, never below the incumbent, shadow-logged where
  higher. Parity 6/6; engine 979/0; cookbook 285/0. Remaining to the endpoint:
  SOAK (shadow must stay silent), then Phase D._
- _2026-07-18 — PHASE D EXECUTED, ADR FROZEN (engine `3873c94`; cookbook and
  backends commits [redacted: private repository])._ (1) Cookbook took
  FULL formula ownership first (~25 call sites: trainer estimate, memmax, all
  constants, the WARM-stage model, the `--detail-bands` parser — the cookbook
  parsing the COOKBOOK's args) + cost-term envelopes on all 7 remaining
  trainers ⇒ every recipe's calibration history resolves through the seam.
  (2) Engine delete (−1,431 lines): `Drivers`/`from_args_json`/
  `in_ch_from_args*`/`estimate*`/warm model/legacy searches/`recipe_footprint*`
  /the shadow referee. The seam is THE contract: declared+dims →
  store-resolved; no dims → estimate-only; undeclared → `footprint_or_floor`
  (loud 2G floor; `BLUT_STRICT_DECLARATIONS=1` refuses — the strict mode is
  real). `sync_base_footprint` env-only; `warm_context()` = the one blessed
  read. (3) scope_guard ADR-0133 fingerprint, PLANT-VERIFIED (a planted
  `from_args_json` fails the guard with the correct label; clean tree passes).
  (4) The parity referee retired with the migration; PERMANENT golden pins
  remain (grid-point literals 30G/28G/38G/23G-narrow/33G-wide, the
  `recipe|tier|batch|workers|w-or-c` key wire format, affine-seam equality,
  warm-model goldens) + the engine's golden-model affine/key/search tests.
  (5) blut-backends pin bumped (71/0) → cookbook `[patch]` DROPPED — one
  engine instance from pure git pins; standalone clones build. Gates: engine
  952/0 full targets + clippy 0 + scope PASS; cookbook 283/0; real dry-run on
  the box resolves the IDENTICAL 40.0G through the seam (live store
  continuity), no floor warning. The endpoint's paragraph shipped in 3b
  (README + CI-run example). DONE — every freeze-gate item met._


## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut && bash scripts/scope_guard.sh` (engine) +
  `cargo test` (cookbook, incl. the golden `adr0133_parity` pins) + planted
  scope-guard violation demonstrably FAILS + `lqt recipe run … --dry-run` on
  the box.
- **result:** engine 952/0 (JSON path gone, `grep from_args_json src/` empty
  outside the guard), clippy 0, scope PASS + plant-FAIL; cookbook 283/0;
  blut-backends 71/0; dry-run byte-continuous (40.0G via the seam).
- **git_sha:** engine `3873c94` · cookbook and backends [redacted: private repository]

## Related Decisions

ADR 0034 (scope charter — the coupling this removes), ADR 0092 [not federated] (module
ownership/contracts), ADR 0046/0047 (broker admission — mechanism unchanged,
sourcing changes), ADR 0071, now [ADR 0156](0156-blut-experiment-execution-ux.md) (auto-tune — re-keyed, not removed), ADR 0096
(tenant ceiling composes with the executor budget), ADR 0103 (async-I/O billing
— must gain a typed source), ADR 0078 (cache keys untouched).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: any increment changes an admission verdict on the canonical recipes
without a Progress-Log entry justifying it; the engine re-grows recipe-key
parsing after increment 4 (scope_guard must catch it); or calibration quality
measurably regresses (self-heal stops converging) after the re-keying.
