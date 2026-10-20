<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0055 — Ingredient taxonomy + IngredientSpec registry (full trainer decomposition) + the Course grouping layer

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (Proposed 2026-06-16 · Accepted 2026-08-27 · Complete — frozen 2026-08-27)

> **Numbering note.** This record was first written as ADR 0051 and renumbered
> to 0055 because of a number collision in the source log. Engine citations
> written before the renumbering said "ADR 0051"; they now read ADR 0055.

(Design handoff, like ADR 0050 [not federated]. Implementation lands across `blut` +
the domain cookbook and its Python payload `[redacted: private repository]` as part of the
canon-only cookbook rebuild; this ADR is the contract, not yet-merged
code. Supersedes ADR 0050's deferral clause — see Related — and absorbs
ADR 0050's `OptimizerSpec` as the `kind="optimizer"` special case.)

## Context

The lamquant cookbook is being rebuilt to canon-only (lean but complete).
Two structural problems motivate adding taxonomy layers in the same pass:

1. **Sub-stage primitives are duplicated and monolithic.** Optimizer
   selection is three drifting `if/elif` chains (ADR 0050's context).
   Worse, a learning-rate scheduler is defined inside one trainer and
   **cross-imported by other trainers** — one trainer reaching into another.
   Optimizer param-group routing is copy-pasted byte-for-byte between two
   trainers. And one trainer's `run()` is a ~2500-line monolith mixing data,
   forward, a multi-term loss, several optimizer arms, an adversarial sub-step,
   EMA, durable resume, and other model-specific logic in one function
   `[redacted: codec research]`. ADR 0050
   introduced the optimizer registry but **explicitly deferred** the full
   trainer-into-primitives decomposition ("data → forward → loss → backward
   → step → log → checkpoint") to "a later ADR." This is that ADR.

2. **Recipes have no grouping layer.** The cookbook exposes recipes under a
   flat `RecipeCategory` tag (DataPrep/Train/Eval/Export/Pipeline/User); the
   owner wants a named, culinary grouping by lifecycle phase so the menu
   reads as a pipeline.

The owner's locked taxonomy (2026-06-16):
**BLUT → Cookbook → Course → Recipe → Ingredient.** The owner **overrode
ADR 0050's deferral** and chose to do the full trainer decomposition now —
on the explicit condition that each decomposition is gated by a
behavioral-equivalence test (below), which is what makes a high-blast-radius
refactor safe.

## Decision

Introduce two layers, neither of which changes the Rust framework
(`framework/stage.rs`, `cache.rs`, `executor.rs`, `plan.rs` are untouched):

- **Ingredient** — a Python sub-stage primitive that lives *inside* one
  trainer process, **below** the Rust `Stage`. A typed `IngredientSpec`
  registry in the cookbook's `ingredients/` package `[redacted: private path]` generalizes ADR 0050's
  `OptimizerSpec` across **13 kinds** (`data, sampler, preprocess, model,
  forward, loss, optimizer, scheduler, step, ema, eval, checkpoint,
  logging`). A single `build_ingredient(kind, name, cfg, **extra)` entry
  point — fail-closed on unknown name, missing hyperparameter, or unmet
  `requires` gate — replaces every inline `if/elif`. Each of the **5
  canonical trainers** `[redacted: codec research]`
  reassembles its `run()` as a fixed sequence of `build_ingredient(...)`
  calls. Shared ingredients (the optimizer registry, the shared scheduler,
  the dataset+cache, the EMA helper, the atomic/durable save) become
  first-class and are imported from `ingredients/`, not cross-imported
  between trainers.

- **Course** — a culinary rename of the existing `RecipeCategory` enum (a
  static tag on each recipe, **not** a new registered object), plus two new
  variants `Pretrain` and `Gate`. It is the within-cookbook grouping by
  lifecycle phase that already drives `recipe list` + the TUI cockpit
  sections.

**The cache-relevant invariant (hard rule).** Every ingredient whose
selection changes the trained artifact (`cache_relevant=True`: data,
sampler, preprocess, model, forward, loss, optimizer, scheduler, step; the
remaining four — `ema, eval, checkpoint, logging` — are observational /
persistence and `cache_relevant=False`)
**MUST** be selected through a typed recipe/stage `Args` field or the
`extra_args` passthrough — both of which the framework already hashes into
the stage cache key ([ADR 0037](0037-blut-cookbook-architecture.md) content-addressing + ADR 0050 `code_sha`).
It must **never** be selected via an environment variable or a Python-side
default, which would let two materially different trainings collide on one
cache key and serve a stale artifact. A CI lint enforces this.

## Rationale

- **Stage stays the cache + containment granule; Ingredient is strictly
  finer and Python-only.** The framework already owns the word for "atomic
  unit in a recipe DAG" (`Stage`). An Ingredient is one process down, intra-
  trainer. This keeps blast radius inside the Python tree and honours
  [ADR 0034](0034-blut-scope-charter.md): forward/loss/optimizer/step is research math → it lives in Python,
  not the Rust control plane.
- **`OptimizerSpec` is the proven template.** ADR 0050 already validated the
  pattern (typed spec, owns its routing, fail-closed resolve, sweep arm).
  `IngredientSpec` is the same object one kind wider; `build_optimizer`
  becomes `build_ingredient("optimizer", …)` with zero contract change.
- **The win is reuse + correctness, not a new cache layer.** Sub-stage
  ingredients get no independent content-addressed cache (the Stage is the
  granule). The payoff is: one `build_param_groups` shared across three
  trainers (kills the copy-paste), the scheduler no longer imported
  from a sibling trainer, and `optimizers=[…]` / `losses=[…]` A/Bs become a
  config edit via the existing `sweep.rs` fan-out.
- **No Rust framework change is needed.** Selection rides existing hashed
  `Args`/`extra_args`; one existing cookbook recipe already proves the
  end-to-end path (typed `optimizer:` arg → `compile()` fail-closed validate
  → `--optimizer` argv → Python registry). Every other kind copies that shape.
- **Course already exists as `RecipeCategory`.** Making the grouping a new
  typed object would re-introduce the dynamic-registry shape ADR 0037
  rejected; the static enum tag + `register_recipe!` macro is the whole
  object. The rename keeps the serde wire format (only `label()` is
  serialized) and an alias avoids churning ~15 call sites.
- **Equivalence tests are the safety net that makes the override safe.**
  ADR 0050 deferred the decomposition for blast-radius reasons; the
  byte-identical loss-trajectory + param-group test per trainer is precisely
  what bounds that risk to "the test fails loudly" rather than "the model
  silently changes."

## Alternatives Considered

- **Ingredient == Stage (rename Stage → Ingredient).** Rejected: pure
  identifier churn across `stage.rs`/`plan.rs`/`executor.rs`/`cache.rs` for
  zero capability, and it would still leave the monolithic trainer
  undecomposed — the actual goal.
- **The grouping as a new typed object the cookbook registers** (the literal
  "Class" reading of the original proposition). Rejected: a
  `Vec<Box<dyn Course>>` re-introduces the dynamic registry ADR 0037 pushed
  back on; a Course has no per-instance state. It is an enum tag.
- **Split the cookbook into more cookbooks instead of a Course layer.**
  Rejected: each would re-declare the same backend, payload, artifact kinds,
  and path roots, and a Train recipe's output feeding an Eval recipe would
  cross a cookbook boundary the within-cookbook artifact handoff doesn't
  span. Course is the cheap intra-pack grouping; Cookbook is the expensive
  inter-domain boundary.
- **Keep deferring the full decomposition (ADR 0050's stance).** Overridden
  by the owner — but only because the equivalence-test gate now exists.
  Without that gate the deferral would still stand.
- **Model ingredient selection in the Rust control plane.** Rejected per ADR
  0050/0034: construction needs live `nn.Parameter` objects and param-name
  routing. Rust selects the recipe; Python selects the ingredient inside it.

## Consequences

- The three optimizer `if/elif` chains collapse to one `build_ingredient`
  call each; new optimizers/losses/schedulers are a single spec file.
- The shared scheduler moves into `ingredients/schedules/`; the
  trainer-to-trainer cross-import is deleted. The monolithic `run()` shrinks
  to an assembly skeleton + its default `IngredientSelection`.
- **The optimizer move is path-coupled** (the canon trainers import the
  optimizer modules with bare names via `sys.path`): the move
  must rewrite those import sites or the trainers break. This is the
  highest-risk edge of the rebuild and is sequenced first.
- The `cache_relevant` invariant becomes load-bearing and a CI lint; a
  regression there is a silent correctness bug, not a crash.
- **A model-specific, feasibility-first selection rule must stay owned
  by its eval ingredient** — it must not be genericized into a
  "max metric" checkpoint ingredient `[redacted: codec research]`.
- Recipe `OPTIMIZERS`/`losses`/`schedulers` sweep fields multiply the schema
  surface; coordinate with the single-schema-source work so they show up in
  `recipe show` + the TUI form once.
- Orphan-stage → recipe promotion (filling the empty DataPrep/Eval/Gate
  Courses end-to-end) is **deferred to a later pass**; this pass only renames
  + re-tags Courses and decomposes the existing trainers. The LLM cookbook
  (a different optimizer pool + `SFTTrainer` loop) is unaffected.

## Related Decisions

- ADR 0050 [not federated] — the prototype ingredient (optimizer registry); its
  "decomposing the trainers into primitives gets its own ADR" deferral is
  superseded here.
- [ADR 0044](0044-blut-api-standardization.md) — BLUT-API / typed recipes (`register_recipe!`, single schema).
- [ADR 0037](0037-blut-cookbook-architecture.md) — cookbook architecture (BLUT▷Cookbook▷Recipe; static registry).
- [ADR 0034](0034-blut-scope-charter.md) — BLUT scope charter (Rust selects/launches, Python is research).
- ADR 0029 [not federated] — owns the model-specific selection rule above.
- ADR 0030 [not federated] — an optimizer flag the optimizer configs carry.

## Validation

- **Behavioral equivalence per trainer (the headline gate).** Fixed seed,
  1 epoch, the `build_ingredient`-assembled `run()` must produce a
  byte-identical loss trajectory (per-step `loss.item()`, ≤1e-6) **and**
  identical param-group membership vs the pre-decomposition path, for each
  of the 5 trainers. Also pin: RNG batch-stream hash, the model-specific
  selection key, provenance hash, saved state_dict keys.
- **Cross-trainer dedup proof.** The shared `build_param_groups` yields identical
  groups whether called from either trainer's assembly.
- **Sweep parity (from ADR 0050).** An N=1 sweep (a single-element `losses=[…]`)
  equals the unswept run — the registry/sweep adds no behavioral delta.
- **Cache-key correctness.** A CI lint asserts no `cache_relevant=True`
  ingredient is selected via env var or Python default; a kernel/byte edit
  to a selected ingredient re-runs the stage (ADR 0050 `code_sha`).
- **Course exhaustiveness.** A test asserts every `Course` variant appears in
  the TUI `category_order` array (catches a silently-dropped variant).
- **Revisit trigger.** If the full decomposition's `step`/`forward` split
  introduces a measurable `torch.compile` recompile or a trajectory drift
  the equivalence test can't reconcile, revisit the `step`-ingredient
  granularity (keep the quantization-aware step verbatim as one ingredient).

## Progress Log

- _2026-08-27_ — gave this record a concrete citation gate `[redacted: private path]` (it proves the record still points at files that exist; it does not claim the decision is implemented) and removed its closure-debt entry `[redacted: private path]`.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

## Completion / Resolution

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** The ingredient layer exists at the shape the Decision drew. The cookbook's `ingredients/` package `[redacted: private path]` holds exactly thirteen kind packages -- checkpoint, data, ema, entropy, evals, forward, logging, loss, model, optimizers, sampler, schedules, steps -- beside the `registry.py` and `spec.py` that make an entry an `IngredientSpec` rather than a bare class, and the trainers select through it (the trainer tests monkeypatch `build_ingredient(kind, name, config, **kwargs)`, which is only possible because that is the real seam). The Rust framework files the Decision promised not to touch are untouched.
