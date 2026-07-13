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

A cookbook's optional `CookbookTui` is discovery/lifecycle glue, not a widget
framework. It may launch any normal Ratatui application or sidecar binary with
its own event loop, layout, component model, and dependency choices. The generic
BLUT cockpit is only a fallback; cookbook authors do not rewrite bespoke TUIs in
BLUT-owned widgets.

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

## Declared recipes: `.toml` / `.json` / `.star` (ADR 0078)

Beyond compiled `RecipeDef`s, `blut recipe declare <file>` compiles a recipe
authored as data, dispatching by extension. All three resolve stage **names**
against the registered cookbooks (no dynamic code loading) and are fully
kind-checked before anything runs.

| Extension | Shape | Notes |
|---|---|---|
| `.toml` | a linear chain of `{stage, args}` | the original declarative path (`DeclarativeRecipe`) |
| `.json` | a `PlanSpec` (arbitrary DAG + map fan-outs) | the engine-native IR — also the Python-SDK door |
| `.star` | a Starlark script | evaluated OUT OF PROCESS by the `blut-dsl` binary |

**`PlanSpec` (v1)** — `framework::plan_spec::{PlanSpec, SpecNode, MapSpec}`, the
stable, versioned wire IR. Evolve additive-only (`#[serde(default)]`); bump
`version` only on a breaking change.

```json
{
  "name": "demo",
  "nodes": [ {"stage": "prepare_data", "args": {"corpus": "tuh"}},
             {"stage": "train_model",  "args": {"tier": 5}} ],
  "edges": [ [0, 1] ],
  "expansions": [
    { "parent": 0,
      "template": {"name": "t", "nodes": [{"stage": "eval_shard"}], "edges": [], "expansions": []},
      "label": "shard" }
  ],
  "version": 1
}
```

- `edges` are `[producer, consumer]` index pairs; **edge order into a node is
  the tuple-element order** a merge consumes (`tuple<N>` input).
- `expansions` are typed runtime fan-outs (`map_output`): when `nodes[parent]`
  completes with a `ListOf<E>` output, the engine runs `template` once per
  element, seeding its single root with the element. The template's root must
  take `E` (kind-checked at compile); nested maps are rejected in v1.
- `PlanSpec::compile(&Registry) -> CompiledPlan`; `provenance_fingerprint`
  hashes `(source, args, spec)` for lineage.

**`.star` contract** (evaluated by `blut-dsl`, hermetic — no `load()`/IO/clock/
randomness, so a script's plan is a pure function of `(source, args)`):

```python
def build(args):                       # required entry point
    root = add("prepare_data", {"corpus": args["corpus"]})   # -> handle (int)
    heads = []
    for t in args["tiers"]:            # compile-time fan-out: a plain loop
        heads.append(add("train", {"tier": t}, after=root))
    merged = add("compare", after=heads)                     # list after = merge
    def per_shard():                   # a map template (its first add = the root)
        add("eval_item")               # consumes the list element at runtime
    map_output(root_producing_a_list, per_shard, label="shard")
```

`add(stage, args=None, *, after=None) -> int` and
`map_output(parent, body, *, label=None)` are the whole surface; handles are
plain ints. The `blut-dsl` binary emits the resulting `PlanSpec` JSON on
stdout, which the engine consumes via the `.json` path. (Starlark is kept
out-of-process because it forces `serde_json/arbitrary_precision`, which would
break the engine's internally-tagged enums — see ADR 0078.)

## Config (`config/`) — Hydra compose, sweeps, launchers

Native, in-tree **Hydra-style config compose** (`config::hydra` —
`ConfigLoader` / `ConfigValue` / `expand_simple_sweeps`): YAML config dirs, a
`defaults:` list with `@package` + group selection, the `key=val` / `+add` /
`~del` override grammar, and choice/range sweep expansion. `config::compose`
freezes a composed tree to JSON + a content fingerprint; `config::sweep` expands
+ fingerprints each combo. The `Launcher` trait (`LocalSystemd` now; `Slurm` /
`RayJobSubmit` are present but deferred) builds the OS command, optionally inside
a resource-capped systemd unit.

`config::tenants::TenantQuotaPolicy` loads `$BLUT_TENANTS_CONFIG` or
`~/.config/blut/tenants.toml`. One file chooses either static equal shares:

```toml
tenants = ["research/dev", "clinical/prod"]
```

or explicit fractions:

```toml
[fractions]
"research/dev" = 0.75
"clinical/prod" = 0.25
```

Without a file, only `default` exists and owns 100% of usable RAM. Explicit
files fail closed on malformed/duplicate tenants, invalid or overcommitted
fractions, mixed modes, and unknown launch tenants.

## Broker (`broker/`) — never-OOM admission

`broker::{decide, Drivers, Footprint, FootprintStore}` size a per-stage RAM
footprint, gate admission against the box-fit budget (`MemTotal − floor`), and
calibrate the estimate from measured peaks (with an OOM-aware self-heal). This is
the engine behind the containment guarantee: a job that wouldn't fit is refused,
not admitted-then-killed. A recipe invoked with **no args** is billed a light
base footprint (a heavy data-trainer always declares required args); everything
else uses the conservative `Drivers` estimate. A first-class **per-recipe
declared footprint** is a planned post-1.0 addition (see "Not yet stable").

`broker::tenant_quota::TenantQuotaTracker` atomically reserves each admitted
job's resolved footprint against its tenant sub-envelope. Its RAII reservation
releases on success, error, cancellation, or unwind. The executor memory
semaphore is also sized to that tenant ceiling, so one HPO/wide-DAG job cannot
exceed its share through concurrent stages. `ExecCtx::with_tenant` threads the
same tenant into every `StageContext`; cookbook persistence and sidecars should
scope themselves with `ctx.tenant`.

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
| `tenant` | validated `project[/domain]` isolation axis; `clinical`/`restricted` are sealed |
| `datasets_db` | raw local dataset source records |
| `dataset_registry` | immutable tenant-scoped `dataset://name@version` bindings with live hash verification |
| `model_registry` | governed `model://name@alias` checkpoint pointers with append-only history |
| `experiment_registry` | tenant-scoped `experiment://recipe/run` lineage views and latest-run comparison |
| `registry_args` | recursive registry-URI resolution before recipe args are typed/deserialized |
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

`blut dataset pin` binds an existing raw source to an immutable
`dataset://name@version`; `blut exp compare` compares the two newest lineage
runs for an explicit `--experiment` campaign and tenant (falling back to the
recipe name for legacy/default runs); and governed `blut model promote`
processes run asynchronously under `--gate-timeout` (default five minutes).
`recipe run`, resumable recipe markers, HPO, declarative TOML, JSON PlanSpecs,
frozen registry PlanSpecs, and Starlark build/stage args resolve these URIs
recursively before cookbook-defined typed args are deserialized. Dataset
handles become a live hash-verified local path, model handles become the
immutable checkpoint hash, and experiment handles become the tenant-scoped run
id. Restricted dataset/model handles refuse every non-local launcher. Model
commands default to the same `default` tenant as recipe launches; records made
under the older `shared` default remain reachable with explicit `--tenant
shared`.

`blut partition define` persists a finite set from repeatable `--dim
axis=v1,v2` flags or a WASM-safe `PartitionSpec` JSON (`time`, `categorical`,
or `multi`; time windows are UTC and use `--partitions START:END`). `partition
backfill` accepts explicit keys/first-axis values, an inclusive range,
`--missing`, or `--stale`. Every selected cell is an ordinary broker-admitted
recipe run with its `PartitionKey` appended to the cache key; unpartitioned
keys remain byte-identical. `partition status` derives its five-state matrix
from the canonical status log plus the rebuildable lineage SQLite index.
Restricted policy is evaluated per cell: selecting one refuses before launch
without blocking unrelated safe selector targets.

## Platform

Containment (per-stage systemd-cgroup `MemoryMax`) requires **Linux + systemd**.
Off-systemd, `LocalSystemd::wrap` degrades to a bare spawn with a warning;
`BLUT_NO_CONTAIN=1` forces it. The crate compiles on macOS (containment off);
Windows is unsupported.

---

## Preview surface (targeted for the 1.0 contract)

The current `0.2.0-alpha.1` line is a development preview, not a semver-stable
1.0 release. The following surface is the intended 1.0 contract, but it may
still change before the M6 release gate. Cookbooks should pin the exact preview
version while the campaign is in progress.

- **Traits:** `framework::Stage`, `framework::Artifact`, `framework::Compatible`,
  `recipes::recipe::Recipe`, `framework::Cookbook`, `backends::TrainingBackend`,
  `config::launcher::Launcher`, `framework::control::ControlPolicy`.
- **Types:** `framework::{Plan, CompiledPlan, NodeId, ExecCtx, PlanResult,
  Resource, ContentHash, ArtifactMetadata, StageContext}`,
  `framework::{SequentialExecutor, ParallelExecutor}`,
  `recipes::recipe::{RecipeDef, Course}`,
  `config::partition::{PartitionSpec, PartitionKey, PartitionValue}`,
  `error::TrainError`.
- **Error enums (all variants):** `framework::{StageError, PlanError}`,
  `recipes::recipe::RecipeError`.
- **Macro:** `register_recipe!` and the `RecipeDef` field shape it emits
  (`args_schema_fn` / `compile_fn` function pointers included).
- **CLI entry:** `cli::run(Registry)`.

**Not yet stable** (may change before they're promoted): the cockpit's internal
`tui` module surface (the cockpit is included in the preview, but it is driven
entirely through the intended `cli::run` entry — the `View`/drawer internals are
not a public contract); the `Slurm` / `Ray` launchers (deferred); the
`broker::Drivers` footprint-driver shape (a planned post-1.0 refactor moves its
domain-specific arg parsing into cookbooks — additive, but the `Drivers` fields
may change). Treat anything not listed under "Preview surface" as subject to
change.
