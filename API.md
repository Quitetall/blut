# BLUT — API reference

The authoritative, always-current reference is rustdoc:

```bash
cargo doc --no-deps --open
```

This file is a curated map of the load-bearing public surface — what to reach
for and where it lives. Crate root: `blut` (`src/lib.rs`). `blut` is a
**library**; it ships no binary and no domain code.

## What BLUT owns

A domain-agnostic ML-training orchestrator: a compile-time typed DAG (stages →
recipes → plans), a content-addressed cache, never-OOM systemd-cgroup
containment + admission, the `status.jsonl` observability stream, an HPO loop, a
lineage index, and the CLI that drives all of it. It ships **zero** concrete
stages, recipes, backends, or Python — those live in downstream *cookbook*
crates that depend on `blut`.

## The abstraction model

`BLUT ▸ Cookbook ▸ Course ▸ Recipe ▸ Ingredient`, mapped to the code:

| Layer | What it is | Code |
|---|---|---|
| **Cookbook** | a domain pack (recipes + backend), loaded into BLUT | `framework::Cookbook` + `framework::Registry` |
| **Course** | the phase, selected + orchestrated in order (DataPrep → … → Export) | `recipes::Course` (`RecipeCategory` alias) |
| **Recipe** | middle-level orchestration: composes ingredients into a typed `Plan` | `recipes::Recipe` + `register_recipe!` → `RecipeDef` |
| **Ingredient** | atomic primitive: a typed `Stage` (`Input → Output`) | `framework::Stage` |
| **BLUT** | the engine: loads cookbooks, runs a recipe's plan over the cache | `cli::run(Registry)` |

See `examples/first_cookbook.rs` for one of each, runnable.

## Framework (`framework/`) — the engine

| Item | Role |
|---|---|
| `framework::stage::{Stage, StageContext, StageError, StageEvent}` | a typed `Input → Output` unit of work + its run context |
| `framework::compat::Compatible<B>` | compile-time gate: a stage may only join a `Plan<_, B>` it is `Compatible` with |
| `framework::artifact::{Artifact, ArtifactMetadata, ContentHash}` | typed handle to on-disk bytes, content-hashed |
| `framework::plan::{Plan, CompiledPlan, NodeId, PlanError}` | the typed DAG builder (`.start().then().finish().into_compiled()`) |
| `framework::executor::{ExecCtx, SequentialExecutor, ParallelExecutor, PlanResult, execute_plan}` | run a `CompiledPlan` with per-`Resource` + memory-admission semaphores |
| `framework::cache::{CacheHandle, CacheHit, lru_prune}` | `(stage, schema, input_hash, args_hash)` → output |
| `framework::resource::Resource` | `Gpu \| Cpu \| Network \| Disk` capacity declarations |
| `framework::retry::{RetryPolicy, Backoff, RetryOn, StageTimeout, RetryHook}` | per-stage retry / timeout / backoff |
| `framework::status::{StatusHub, StageEvent, make_broadcast, spawn_status_writer}` | the `status.jsonl` event stream |
| `framework::cookbook::{Cookbook, Registry, StageDescriptor, ArtifactDescriptor}` | the domain-pack seam |
| `framework::graph::{PlanGraph, GraphSnapshot, NodeStatus, graph_snapshot}` | DAG inspection |

## Recipes (`recipes/`) — the named catalog

| Item | Role |
|---|---|
| `recipes::recipe::{Recipe, RecipeDef, Course}` | the per-recipe contract: typed `Args`/`Backend`/`Input`/`Output` + a compile fn |
| `register_recipe!` | macro — emits a `pub static DEF: RecipeDef` from a `Recipe` impl |
| `recipes::{find, by_category, args_template}` | lookup, menu grouping, schema-driven args prefill |

A `RecipeDef` carries `name`, `description`, `backend_id`, `category`,
`input_kinds`, `output_kind`, an args JSON-schema fn, and a compile fn that
parses args → a backend-erased `CompiledPlan`.

## Config (`config/`) — Hydra compose, sweeps, launchers

Native, in-tree **Hydra-style config compose** (`config::hydra` —
`ConfigLoader` / `ConfigValue` / `expand_simple_sweeps`): YAML config dirs, a
`defaults:` list with `@package` + group selection, the `key=val` / `+add` /
`~del` override grammar, and choice/range sweep expansion. `config::compose`
freezes a composed tree to JSON + a content fingerprint; `config::sweep` expands
+ fingerprints each combo. The `Launcher` trait (`LocalSystemd` now; `Slurm` /
`RayJobSubmit` are present but deferred) builds the OS command, optionally inside
a resource-capped systemd unit.

## Broker (`broker/`) — never-OOM admission

`broker::{decide, Drivers, Footprint, FootprintStore}` size a per-stage RAM
footprint, gate admission against the box-fit budget (`MemTotal − floor`), and
calibrate the estimate from measured peaks (with an OOM-aware self-heal). This is
the engine behind the containment guarantee: a job that wouldn't fit is refused,
not admitted-then-killed. A recipe invoked with **no args** is billed a light
base footprint (a heavy data-trainer always declares required args); everything
else uses the conservative `Drivers` estimate. A first-class **per-recipe
declared footprint** is a planned post-1.0 addition (see "Not yet stable").

## Backends (`backends/`)

`backends::TrainingBackend` — a marker trait (`ID` + `DESCRIPTION`) that gives a
backend its compile-time identity. The engine ships **no concrete backend**; a
cookbook tags its own. `Plan<Out, B>` is parameterized over `B`, so the compiler
refuses to wire a stage tagged for a different backend.

## Other surface

| Module | Role |
|---|---|
| `hpo` | hyperparameter search (TPE / median / PBT samplers + schedulers) over a recipe |
| `lineage_db` | content-addressed artifact lineage index |
| `datasets_db` | dataset registry |
| `sensor`, `schedule`, `policy` | named freshness sensors, systemd-timer schedules, the auto-retrain policy |
| `config::launcher::Launcher` | placement abstraction (local / cluster) |
| `error::TrainError` | top-level error type |

## CLI (`cli`)

`blut::cli::run(registry: Registry).await` — the single public entry point. A
cookbook binary supplies its `Registry` and delegates the whole CLI to it
(`recipe`, `jobs`, `log`, `cancel`, `plan`, `cache`, `footprint`, `partition`,
`schedule`, `sensor`, `policy`, …). The interactive `tui` subcommand + the
bare-command cockpit ship in the default build (the default-on `tui` feature);
`--no-default-features` yields a lean CLI-only binary (bare `blut` prints help).

## Platform

Containment (per-stage systemd-cgroup `MemoryMax`) requires **Linux + systemd**.
Off-systemd, `LocalSystemd::wrap` degrades to a bare spawn with a warning;
`BLUT_NO_CONTAIN=1` forces it. The crate compiles on macOS (containment off);
Windows is unsupported.

---

## Stable surface (the 1.0 contract)

From **1.0.0**, the following are the public API blut promises to keep
semver-stable. Minor releases are **additive-only**; a breaking change to any of
these requires a major bump. Cookbooks should pin `blut = "1"`.

- **Traits:** `framework::Stage`, `framework::Artifact`, `framework::Compatible`,
  `recipes::recipe::Recipe`, `framework::Cookbook`, `backends::TrainingBackend`,
  `config::launcher::Launcher`, `framework::control::ControlPolicy`.
- **Types:** `framework::{Plan, CompiledPlan, NodeId, ExecCtx, PlanResult,
  Resource, ContentHash, ArtifactMetadata, StageContext}`,
  `framework::{SequentialExecutor, ParallelExecutor}`,
  `recipes::recipe::{RecipeDef, Course}`, `error::TrainError`.
- **Error enums (all variants):** `framework::{StageError, PlanError}`,
  `recipes::recipe::RecipeError`.
- **Macro:** `register_recipe!` and the `RecipeDef` field shape it emits
  (`args_schema_fn` / `compile_fn` function pointers included).
- **CLI entry:** `cli::run(Registry)`.

**Not yet stable** (may change before they're promoted): the cockpit's internal
`tui` module surface (the cockpit ships in 1.0, but it is driven entirely
through the stable `cli::run` entry — the `View`/drawer internals are not a
public contract); the `Slurm` / `Ray` launchers (deferred); the
`broker::Drivers` footprint-driver shape (a planned post-1.0 refactor moves its
domain-specific arg parsing into cookbooks — additive, but the `Drivers` fields
may change). Treat anything not listed under "Stable surface" as subject to
change.
