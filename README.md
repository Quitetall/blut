# BLUT — Brian Lam's Universal Trainer

A **Rust-native, compile-time-typed orchestration framework for local ML
training.** You wire stages into a typed DAG; blut runs it against a
content-addressed cache, under per-stage memory containment, with structured
observability — and refuses to wire two stages whose types don't line up.

`blut` is a **library crate** (no binary of its own). You build a *cookbook* on
top of it — your stages, your recipes, your CLI binary — in a few hundred lines.
It ships **zero** domain code: no bundled recipes, no Python, no opinion about
what you train.

```toml
[dependencies]
blut = "1"
```

**API reference:** [`API.md`](API.md) · `cargo doc --no-deps --open` · runnable
demo: [`examples/first_cookbook.rs`](examples/first_cookbook.rs).

## The abstraction model

BLUT organizes work as a five-layer hierarchy:

```text
  BLUT   ▸   Cookbook   ▸   Course   ▸   Recipe    ▸   Ingredient
 engine     domain pack      phase      workflow       primitive
```

- **Cookbook** — a domain pack: a set of recipes plus the backend they target.
  Cookbooks are **loaded into** BLUT (via a `Registry`). *(the `Cookbook` trait)*
- **Course** — the phase a recipe belongs to, **selected and orchestrated in
  order**: `DataPrep → Pretrain → Train → Eval → Gate → Export` (a `Pipeline`
  course chains several end-to-end). Recipes are grouped by course. *(the
  `Course` enum)*
- **Recipe** — a **middle-level orchestration function**: it composes ingredients
  into a typed `Plan` and exposes typed args. *(the `Recipe` trait +
  `register_recipe!`)*
- **Ingredient** — an **atomic primitive**: a typed `Stage` (`Input → Output`),
  the smallest reusable unit of work. A training cookbook also has finer
  primitives — optimizers, schedulers, losses — that a stage composes. *(the
  `Stage` trait)*
- **BLUT** — the engine: it loads cookbooks, lists/selects recipes by course, and
  runs a recipe's plan against the content-addressed cache under per-stage
  resource + memory admission. *(`blut::cli::run(registry)`)*

[`examples/first_cookbook.rs`](examples/first_cookbook.rs) builds one of each
layer and runs it end-to-end:

```text
$ cargo run --example first_cookbook
BLUT loaded cookbook 'demo'. Recipes by course:
  • User  count_to_three  — MakeOne → Increment → Increment (demo)
Run 1 (cold cache):  → 3 stages, 0 hits, 3 misses
Run 2 (warm cache):  → 3 stages, 3 hits, 0 misses
```

## Why

Local ML pipelines accrete ad-hoc shell glue: dump data, kick off a trainer,
wait, convert a checkpoint, copy it somewhere. Each step grows its own retry
logic, logging, and cache; a crash mid-run replays everything; a DataLoader
that over-allocates takes down your whole login session. blut replaces the glue
with a typed pipeline:

```rust
let plan = Plan::<(), MyBackend>::new("train", json!({}))
    .start(PrepareData, prep_args)
    .then(Train, train_args)
    .then(Evaluate, eval_args)
    .finish()
    .into_compiled();

let result = SequentialExecutor::execute(plan, ExecCtx::new(job_dir)).await?;
```

- **Stages** declare a typed `Input → Output` and the resources they hold
  (`Gpu`, `Cpu`, `Network`, `Disk`). Wrong wiring is a `cargo build` error, not a
  runtime panic — `Train`'s `Input` must equal `PrepareData`'s `Output`.
- **Plans** are typed DAGs; **recipes** compile typed args into plans (a named,
  args-driven catalog); **cookbooks** group recipes + their backend.
- **Cache** content-addresses every stage output by
  `(stage, schema, input_hash, args_hash)` — crash mid-run, re-run, and finished
  stages are served from cache instead of recomputed.
- **Containment** (Linux + systemd) runs each stage under a `systemd-run --user`
  transient unit with a `MemoryMax` cap sized from a per-stage footprint, plus a
  box-fit **admission gate** that refuses a job that wouldn't fit — so an OOM is
  a contained unit-kill, never a session-wide crash.
- **Observability** streams a `status.jsonl` event log per job and exposes a
  metric store, a lineage index, and an HPO loop.

## Build your own cookbook

The four moving parts (see the runnable
[`examples/first_cookbook.rs`](examples/first_cookbook.rs)):

1. A **backend identity** — a unit struct implementing `TrainingBackend`. A
   `Plan<Out, B>` is parameterized over its backend `B`; a stage only joins a
   plan whose backend it is `Compatible` with.
2. An **artifact** — a `Serialize`/`Deserialize` struct implementing `Artifact`
   (a content-hashed handle to a stage's output).
3. **Stages** — typed `Input → Output` units implementing `Stage`.
4. A **plan / recipe** — wire stages with `Plan::start().then().finish()`, or
   register a named recipe with the `register_recipe!` macro and group recipes
   into a `Cookbook`.

Your binary is then a thin stub — supply your `Registry` and hand off to the
engine's CLI:

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    blut::cli::run(my_cookbook::registry()).await
}
```

That one call gives your binary the whole CLI: `recipe list` / `recipe run`,
`jobs`, `log`, `cancel`, `plan resume/inspect`, `cache prune`, `footprint`,
`partition`, `schedule`, `sensor`, and more (`--help` for the full set).

## Platform support

- **Linux + systemd** — full support, including never-OOM cgroup containment.
- **macOS / other Unix** — compiles and runs; containment **degrades to a bare
  spawn** (no memory cap — the admission gate still refuses oversized jobs, and
  the kernel OOM killer is the only hard backstop). Set `BLUT_NO_CONTAIN=1` to
  force the bare path. A `macos-latest` CI job keeps the build green.
- **Windows** — not supported (Unix process/signal primitives).

## Status — 1.0

The framework, typed DAG, content-addressed cache, containment + admission,
observability, HPO, lineage, and the recipe/cookbook system are stable and
end-to-end runnable. From 1.0, minor releases are **additive-only** (see the
"Stable surface" contract in [`API.md`](API.md)); cookbooks should pin `blut = "1"`.

**1.0 is CLI-only.** The interactive TUI cockpit is behind an off-by-default
`tui` feature (`--features tui`) and is **unstable until 1.1**, when it returns
as a first-class feature.

## License

[GNU AGPL-3.0-or-later](LICENSE). BLUT is free software: you may use, study,
modify, and redistribute it under the GNU Affero General Public License, version
3 or (at your option) any later version.

Note the AGPL's **network clause** (§13): if you run a modified version of BLUT
as a network-accessible service, you must offer that service's users the
corresponding source of your modified version.

A **commercial license** — for use without the AGPL's copyleft / source-
availability obligations — is available from the maintainer on request.

Contributions are welcome under the same terms — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
