# BLUT — API reference

The authoritative, always-current reference is rustdoc:

```bash
cargo doc --no-deps --open
```

This file is a curated map of the load-bearing public surface — what to
reach for and where it lives. Crate root: `blut` (`src/lib.rs`).

## What BLUT owns

The universal trainer: a compile-time typed DAG (stages → recipes →
plans), a content-addressed cache, systemd-cgroup containment, the
observability stream, the TUI cockpit, and the LamQuant Python pipeline
payload (`python/lamquant/`: encode → label → split → load). It
*consumes* `LamQuant-Lossless` (the `.lml`/`.lma` codec) and
`LamQuant-Neural` (model defs); their formats are documented in those
repos' `API.md`.

## Framework (`framework/`) — the engine

| Item | Role |
|---|---|
| `framework::stage::{Stage, StageContext, StageError, StageEvent}` | a typed `Input → Output` unit of work + its run context |
| `framework::artifact::{Artifact, ContentHash, Kind}` | typed handle to on-disk bytes, content-hashed |
| `framework::plan::{Plan, PlanError}` | the typed DAG builder (`.start().then().finish()`) |
| `framework::executor` | runs a `CompiledPlan` with per-`Resource` semaphores |
| `framework::cache` | `(stage, schema, input_hash, args_hash)` → output; `lru_prune` |
| `framework::resource::Resource` | `Gpu \| Cpu \| Network \| Disk` capacity declarations |
| `framework::status::{StatusUpdate, make_broadcast}` | the `status.jsonl` event stream |
| `framework::cookbook::{Cookbook, Registry, StageDescriptor, ArtifactDescriptor}` | the domain-pack seam (ADR 0037) |

## Recipes (`recipes/`) — the catalog

| Item | Role |
|---|---|
| `recipes::{RECIPES, RecipeDef, Recipe, RecipeCategory}` | the static recipe catalog + per-recipe contract |
| `recipes::{find, by_category, swap_candidates}` | lookup, menu grouping, I/O-compatible swap discovery |

A `RecipeDef` carries `name`, `description`, `backend_id`, `category`,
`input_kinds`, `output_kind`, an args JSON-schema fn, and a compile fn
that parses args → backend-erased `CompiledPlan`. See the recipe table
in [`README.md`](README.md) or run `blut recipe list`.

## Config (`config/`) — resolution + sweeps

Layered config resolve + fingerprint, Cartesian sweep expansion, and the
`Launcher` trait (`LocalSystemd` now; `Slurm` / `RayJobSubmit` deferred).
Hydra-style overrides resolve through the vendored
[Lerna](https://github.com/nbprint/lerna) Rust core. Non-finite floats
serialize to their string form (never `Null`) so distinct configs never
collide on a fingerprint.

## Backends (`backends/`) + top-level adapters

| Item | Role |
|---|---|
| `backends::LamquantBackend` | drives the LamQuant Python payload via subprocess + frozen-JSON args |
| `PythonTrainBackend`, `backend::{TrainBackend, TrainArtifact}` | the generic local-LLM (lamu) backend |
| `spec::{TrainSpec, Method, Optim, DatasetSource}` | generic-LLM training spec |
| `error::TrainError` | top-level error type |

## CLI (`main.rs`)

`blut` (bare → TUI) · `train` · `jobs` · `log` · `cancel` · `recipe`
(`list`/`run`/`show`) · `plan` (`resume`/`inspect`) · `cache` (`prune`)
· `stage` · `data` · `auto` · `policy` · `tui`. The TUI lives in
`tui::` (ratatui cockpit).

## Containment + observability

- `BLUT_CONTAINED=1` runs each stage under a `systemd-run --user`
  transient unit with `MemoryMax`; see `backends::lamquant::runner`.
  When set but `BLUT_JOB_DIR`/`BLUT_STAGE_NAME` are absent, it warns
  loudly and falls back to a bare spawn (no containment).
- Per-epoch metrics stream to a Parquet/CSV log
  (`python/lamquant/common/metric_log.py`) and optionally wandb
  (`--logger {none,wandb}`, offline by default).

See also: [`README.md`](README.md) · meta index [`../API.md`](../API.md).
