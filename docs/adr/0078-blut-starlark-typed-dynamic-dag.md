<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0078 — Typed dynamic DAGs: scripts compose registered stages into a checked graph

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (frozen 2026-07-07). Partially supersedes [ADR 0037](0037-blut-cookbook-architecture.md)'s "the declarative layer is linear TOML" boundary.

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text. When the deliverable passes its gate, fill
> **Completion** and set Status to `Complete — frozen YYYY-MM-DD`.

## Status

Complete — frozen 2026-07-07. Part 1 of the "BLUT real-tool" roadmap
(a planning note, `[redacted: private path]`). Partially supersedes
[ADR 0037](0037-blut-cookbook-architecture.md)'s
"the declarative layer is linear TOML" boundary (see Decision). Precedes
[ADR 0079](0079-blut-symmetric-mesh.md) (symmetric mesh) and
[ADR 0080](0080-blut-async-dp-training.md) (async DP training), which build on the plan
IR introduced here.

## Context

BLUT's power is a **compile-time typed Plan DAG**: `Plan<Out, B>::then::<S>`
refuses to wire two stages whose artifact types don't line up, and a wrong wire
is a `cargo build` error. ADR 0037 kept a *hybrid*: typed Rust recipes plus a
**declarative TOML** path (`DeclarativeRecipe` → `CompiledPlan::from_erased_chain`)
that resolves stage names against the registry and runtime-kind-checks the chain.
ADR 0037 explicitly **rejected** an *untyped runtime cookbook registry* because it
"loses the compile-time typed Plan DAG"; ADR 0056 [not federated] rejected a
Cranelift JIT for BLUT as "the wrong tool"; [ADR 0034](0034-blut-scope-charter.md)
(the scope charter) bans dynamic code loading and in-engine HTTP servers.

Two limits now bite as BLUT grows toward a general tool:

1. **The declarative path is linear-only.** `from_erased_chain` builds the chain
   `0→1→…→n-1`; there is no way to express fork/merge/map topology from data
   without writing Rust. Every dynamic pipeline (HPO fan-out via
   `CompiledPlan::from_components`, the PBT/TPE `TrialFactory`/`FreshFactory`
   closures) is bespoke Rust.
2. **There is no scripting surface.** Flyte/Dagster won adoption with an
   *embedded* DSL (write Python, the SDK emits a graph IR the engine executes).
   BLUT has no equivalent — a non-Rust user cannot author a pipeline.

The runtime machinery to *grow* a DAG already exists and is proven:
`ControlPolicy::on_step` may return `Control::Spawn(SpawnDelta{ subplan, .. })`,
and the executor injects the sub-plan via `inject_spawn` on its single-threaded
coordinator seam (used today by PBT and TPE). What is missing is a *typed,
data-authored* front door to that machinery.

## Decision

Introduce a **typed dynamic DAG facility** with three parts, all resting on one
plain-data contract:

1. **A serde IR — `PlanSpec`** (`src/framework/plan_spec.rs`): a
   JSON-serializable graph of `{stage-name, args}` nodes + `(from,to)` edges +
   typed map `expansions`. `PlanSpec` is the single contract; three front-ends
   emit it — Starlark scripts (`.star`), raw `.json` (the future Python SDK's
   entry point), and (unchanged) TOML via the existing linear path.

2. **A full-topology kind-checker — `CompiledPlan::from_erased_graph`**: extends
   the guarantee `from_erased_chain` gives a linear chain to an arbitrary DAG.
   Every node is checked before execution: a root (0 predecessors) must take the
   unit graph input (`input_kind() == "()"`); a 1-predecessor node's input kind
   must equal its predecessor's output kind; an N-predecessor node must take
   `tuple<N>` (edge insertion order = tuple element order, matching the
   executor's `gather_input` contract — element kinds are re-verified at
   `decode_erased`, exactly as the typed `merge` path already relies on). A
   kind break is a **plan compile error naming both stages and both kinds**,
   never an executor panic.

3. **A hermetic Starlark front-end** (`src/dsl/`, behind a non-default `dsl`
   feature): scripts define `build(b, args)` and call `b.add(stage, args=,
   after=)` to compose **registered** stages. Fork = shared `after`; merge =
   list `after`; compile-time map = a plain `for` loop; runtime map over a
   stage's list output = `b.map_output(parent, body)` (a typed `MapSpec`
   expansion executed on the executor's completion seam). Starlark is evaluated
   **once, before launch**; the executor never re-enters the interpreter.

**"Typed" is the load-bearing word.** Scripted and JSON plans are kind-checked at
plan-compile time (the strength Flyte gets at registration time), while Rust
recipes keep their `cargo build`-time guarantee. Both check before anything
executes — strictly stronger than an untyped runtime registry.

### What this supersedes in ADR 0037

Only the boundary "the declarative layer is linear TOML." The rejection of an
*untyped* runtime registry **stands** — this facility is typed. The hybrid model
(compiled Rust stages as the vocabulary) is preserved, not overturned.

### What stays forbidden (non-goals, enforced)

- **No dynamic stage/plugin loading.** Scripts and JSON may name only stages
  resolvable via `Registry::find_erased_stage`; an unknown stage is a compile
  error. New stage *types* still require Rust + cookbook registration. (ADR 0034
  charter, unchanged.)
- **No script I/O or nondeterminism.** Starlark runs with no `load()`, no file
  loader, and only the default globals (no time/env/random — starlark-rust is
  built for Buck2 hermetic evaluation). A step/heap limit bounds pathological
  scripts. This keeps a script's emitted `PlanSpec` a pure function of
  `(source, args)`, so it is content-hashable.
- **No in-engine HTTP server** (ADR 0034). The Python SDK, when it lands, emits
  `PlanSpec` JSON over an out-of-engine channel; the engine only *reads* JSON.
- **No runtime interpreter re-entry.** The executor evaluates zero Starlark; all
  dynamism is either compiled into the `PlanSpec` at eval time or driven by the
  existing typed `SpawnDelta`/`inject_spawn` seam.

### Hash discipline

Per-stage cache keys are **unchanged**: `hash(code_sha ⊕ canon_args ⊕
input_hash)`. Topology was never part of a stage's cache key, so a scripted plan
caches correctly for free — an unchanged sub-DAG hits the cache regardless of how
the plan was authored. A **plan provenance fingerprint** =
`sha256("blut-star-v1" ‖ script_bytes ‖ canon_args ‖ PlanSpec::canonical_bytes())`
is recorded in `recipe_args` for lineage. Args-dependent topology is captured by
construction: different args ⇒ different `PlanSpec` ⇒ different fingerprint, while
unchanged sub-DAGs still hit the per-stage cache.

## Alternatives rejected

- **A custom DSL syntax** (à la early Airflow XML / CWL): consistently lost in the
  field to embedded DSLs. Rejected — Starlark is Python-shaped and already known.
- **Rhai instead of Starlark**: lighter, but not hermetic by default; determinism
  (and thus the content-hash guarantee) would need bespoke sandboxing. Rejected.
- **Untyped runtime stage registry** (re-litigating ADR 0037): rejected again —
  this facility is typed end to end.
- **Cranelift/JIT** (ADR 0056 [not federated]): not relevant; there is no
  machine-code generation here, only graph construction. The "JIT dynamic DAG"
  ask is served by typed map expansion on the existing spawn seam.

## Consequences

- A new `dsl` Cargo feature pulls `starlark`; kept **out of default features** so
  the lean CLI build is unaffected (cookbook binaries opt in).
- `PlanSpec` becomes a **stable versioned wire contract** (the Python SDK's
  interface); it carries a `version` field and evolves additive-only.
- The executor's completion seam gains a typed map-expansion trigger (ADR-scoped
  to reuse `pending_spawns` + `initial`-style output seeding, so the coordinator's
  between-joins mutation discipline is untouched — see the plan's Part 1 Phase 4).

## Anti-re-litigation clause

This ADR does **not** reopen ADR 0037's rejection of an untyped runtime registry,
nor ADR 0034's ban on dynamic code loading and HTTP servers. A future proposal to
let scripts load stage *code* (not just name registered stages), to relax
hermeticity, or to embed a general-purpose runtime in the executor must supersede
this ADR explicitly with new evidence — it is not an incremental tweak.

## Progress Log

- 2026-07-07 — ADR written. Implementation sequenced as Part 1 Phases 0–5 of
  the roadmap plan (`[redacted: private path]`): PlanSpec IR + `from_erased_graph`
  (Phase 1), Starlark front-end (Phase 2), CLI surface (Phase 3), typed runtime
  fan-out (Phase 4), hardening/fuzz/bench/docs (Phase 5). Each phase lands with
  its own tests and appends a progress entry here.
- 2026-07-07 — **CORRECTION (load-bearing): the Starlark front-end moved OUT
  of the engine into a separate binary `crates/blut-dsl`** (engine commits
  `0cc92f5` + `49f4203`). Phase 2 originally added it as an in-engine `dsl`
  Cargo feature; that is **unsound**. `starlark` hard-depends on
  `serde_json`'s `arbitrary_precision` feature, and Cargo feature unification
  enables it for the WHOLE engine binary whenever `dsl` is on. That feature is
  incompatible with serde's internally-tagged enums (`#[serde(tag="kind")]`),
  so it silently breaks deserialization of the engine's core wire formats —
  `StatusUpdate`, `StageEvent` (status.jsonl), `TrainSpec`. Verified:
  `from_str::<StatusUpdate>` on valid JSON returns `Err` under the old `dsl`
  feature. FIX: the engine never links Starlark; `blut-dsl` is a subprocess
  adapter (the same pattern as kubectl/slurm/blut-worker) that emits `PlanSpec`
  JSON, which the engine consumes through the plain `.json` path. This updates
  the Decision's part 3: Starlark runs out-of-process, not behind an in-engine
  feature. Everything else (the `PlanSpec` IR, `from_erased_graph`, the
  hermetic contract, the ban list) is unchanged.
- 2026-07-07 — **Phase 4 landed** (typed runtime `map_output` fan-out; engine
  commits `ab5c8b3` + `bff1372` + `8335a2a` + `920ffa8`, blut-dsl `4bf74e0`;
  MiMo-reviewed). Four slices: (4a) `ListOf<E>` list artifact envelope +
  `Artifact::ELEMENT_KIND`/`StageDyn::output_element_kind()`; (4b) `MapSpec` in
  the `PlanSpec` IR + `MapExpansion`/`CompiledTemplate` in `CompiledPlan`,
  `from_erased_template` (single-root, element-kind-checked), compile-time
  validation (parent-is-a-list, no nested maps), and DCE skips map plans; (4c)
  the executor runtime fan-out — on the lossless completion seam, a
  list-producing parent spawns one template instance per element via the
  existing `SpawnDelta`/`inject_spawn` seam (`SpawnDelta` gained `root_seeds` +
  `provenance_parent`, `Control::Spawn` boxed), element logical hash =
  `sha256("map-elem" ‖ parent_logical ‖ i)` for cache-stable shards of a
  nondeterministic parent, cap-overflow FAILS loudly (map shards drain before
  best-effort HPO spawns); (4d) the `map_output(parent, body, label=)` Starlark
  builtin in blut-dsl (a draft-stack: push template scope, run body, pop,
  attach). Verified end-to-end: `.star` → PlanSpec JSON with expansions →
  engine compile → execute → N children spawned → second run all cache-hit.
- 2026-07-07 — **Phase 3 landed** (same commits): `blut recipe declare`
  dispatches by extension — `.toml` (existing chain), `.json` (PlanSpec,
  feature-independent, the Python-SDK door), `.star` (shells out to `blut-dsl`
  via `$BLUT_DSL_BIN`/sibling-exe/PATH, actionable error if absent). New
  `--args` flag (semantic empty check; rejected for `.toml`/`.json`).
  `PlanSpec::provenance_fingerprint` (engine-side) + `override_recipe_args`
  stamp script path + fingerprint + args into lineage. `scan_user_recipes_at`
  lists `.star`/`.json`. blut-dsl carries the script→PlanSpec tests (incl. a
  fan-out proptest); the engine carries the dispatch/routing tests.
- 2026-07-07 — **Phase 1 landed** (engine commits `7e72347` + `28fae66`,
  MiMo-reviewed). `src/framework/plan_spec.rs` (the `PlanSpec` serde IR:
  `deny_unknown_fields`, `canonical_bytes()`, `compile(&Registry)`, version 1)
  and `CompiledPlan::from_erased_graph` (full-topology kind-check: `()`-roots,
  1-pred out==in, N-pred `tuple<N>`; rejects cycles/empty/dangling/**duplicate**
  edges as typed errors, never panics). New `PlanError` variants
  `RootNotGraphInput`/`KindBreak`/`BadMergeArity`(carries the real
  `expected_kind`)/`EdgeOutOfRange`/`DuplicateEdge`. Stage cache keys unchanged;
  a graph-built plan caches exactly like a chain-built one (verified by an
  execute + second-run-all-cache-hits integration test). 17 new tests incl. 2
  proptests; `from_erased_chain` untouched. Next: Phase 2 (Starlark front-end
  behind the non-default `dsl` feature).

- 2026-07-07 — **Phase 5 landed** (engine commit — fuzz + bench + docs): a
  `fuzz_plan_spec` target (arbitrary bytes → `PlanSpec` parse → compile, never
  panics; guards the nested-template serde-recursion path), a 1k-node
  `from_erased_graph` compile Criterion bench (~109 µs), and `API.md` +
  `CHANGELOG` documenting the `.toml`/`.json`/`.star` declared-recipe surface,
  the `PlanSpec` v1 JSON shape, and `map_output` semantics.

## Completion

**Complete — 2026-07-07.** All six phases (0–5) of the typed dynamic-DAG
facility landed and are MiMo-reviewed: ADR (0), `PlanSpec` IR +
`from_erased_graph` (1), the out-of-process Starlark front-end `crates/blut-dsl`
(2), the `.toml`/`.json`/`.star` CLI dispatch (3), typed runtime `map_output`
fan-out (4), and fuzz/bench/docs hardening (5). End-to-end verified: a `.star`
script → `blut-dsl` → `PlanSpec` JSON → engine compile → execute → runtime
fan-out → cache-stable re-run. Charter honored throughout (no dynamic stage
loading; no in-engine HTTP; Starlark isolated out-of-process). The one
correction to the original Decision — Starlark runs out-of-process rather than
behind an in-engine `dsl` feature — is recorded in the 2026-07-07 progress
entry (the `serde_json/arbitrary_precision` feature-unification hazard).

This ADR is now frozen. A later change to the typed dynamic-DAG facility (e.g.
nested maps, args-aware `CPU_CORES`, or a Python SDK emitting the IR) is a new
ADR that references this one; the roadmap's Part 2 (symmetric mesh, k8s
operator, DP async training) lands under ADRs [0079](0079-blut-symmetric-mesh.md)/[0080](0080-blut-async-dp-training.md).
