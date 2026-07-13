# BLUT — Brian Lam's Universal Trainer

A **Rust-native, compile-time-typed orchestration framework for ML training.**
You wire stages into a typed DAG; BLUT runs it against a content-addressed
cache, with memory admission, available process containment, and structured
observability. BLUT refuses to wire two stages whose types do not line up.
Local orchestration is exercised; multi-GPU, cluster, P2P, and cloud paths have
only the bounded evidence recorded in the [scaling ladder](#scaling-ladder).

```toml
[dependencies]
blut = "=0.2.0-alpha.1"
```

## What BLUT is

BLUT is a DAG orchestrator. Everything else, resource brokerage, containment, P2P, HPO, lineage, cloud compute, is just a layer on top.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI / TUI cockpit                            │
├─────────────────────────────────────────────────────────────────┤
│  Recipes (typed Plan builders)                                  │
├─────────────────────────────────────────────────────────────────┤
│  DAG Orchestrator (Stage → Plan → Executor → Cache)             │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ Resource │  Async   │  DAG     │ Process  │ Algorithmic         │
│ Broker   │  Runtime │ Opt      │ Opt      │ Opt                 │
├──────────┴──────────┴──────────┴──────────┴─────────────────────┤
│  LaunchTarget (Local / Slurm / Ray / P2P / Cloud)               │
├─────────────────────────────────────────────────────────────────┤
│  Containment (systemd / cgroup2 / rlimit / bare fallback)       │
├─────────────────────────────────────────────────────────────────┤
│  P2P Transport (QUIC + Ed25519 + AES-256-GCM)                  │
├─────────────────────────────────────────────────────────────────┤
│  Platform (Linux / macOS / Windows / AMD / NVIDIA / Apple)      │
└─────────────────────────────────────────────────────────────────┘
```

## The abstraction model

```text
  BLUT   ▸   Cookbook   ▸   Course   ▸   Recipe    ▸   Ingredient
 engine     domain pack      phase      workflow       primitive
```

- **Ingredient** — an atomic primitive: a typed `Stage` (`Input → Output`).
- **Recipe** — composes ingredients into a typed `Plan` with typed args.
- **Cookbook** — a domain pack: recipes + backend + ingredients.
- **BLUT** — the engine: loads cookbooks, runs plans, manages cache/containment.

## Why

Local ML pipelines accrete ad-hoc shell glue. BLUT replaces it with a typed
pipeline with admission control, available process containment, durable state,
and placement seams that can grow beyond one GPU without rewriting the DAG:

```rust
let plan = Plan::<(), MyBackend>::new("train", json!({}))
    .start(PrepareData, prep_args)
    .then(Train, train_args)
    .then(Evaluate, eval_args)
    .finish();

let result = ParallelExecutor::execute(plan.into_compiled(), ctx).await?;
```

## Key features

### v0.2.0-alpha.1 — current development preview ([CHANGELOG](CHANGELOG.md))

| Feature | What |
|---------|------|
| **DAG orchestrator** | Stage → Plan → Recipe, typed wiring, content-addressed cache |
| **Parallel executor** | Resource semaphores (GPU/CPU/Disk/Network), box-fit budget |
| **Broker** | RAM admission gate, footprint estimation, OOM-aware calibration |
| **Containment** | systemd → cgroup2 → rlimit → bare fallback chain |
| **Durable resume** | Crash-gated recovery, epoch-level resume |
| **Observability** | status.jsonl, metric store, lineage index |
| **TUI cockpit** | default-on `tui` feature; `--no-default-features` for a lean CLI |
| **Multi-tenancy** | tenant-scoped cache/registry/lineage/privacy, RAM sub-envelopes, Restricted node-local custody |
| **Registries** | governed `model://`, immutable `dataset://`, and lineage-backed `experiment://` handles resolved in recipe args |

Registry handles are resolved before typed cookbook/declarative args compile,
and resumable jobs retain the original handles so datasets are hash-revalidated
on resume. Model and recipe commands both default to tenant `default`; use
`--tenant shared` to address model records created under the former default.

### Experimental development surface

Landed in the development tree, but not yet cut as a release. Validation status
per capability is tracked in the
[distributed-validation ledger](https://github.com/Quitetall/blut/blob/v0.2.0-alpha.1/docs/DISTRIBUTED_VALIDATION.md):

| Feature | What |
|---------|------|
| **Ingredient system** | 12 kinds, registry pattern, frozen config validation |
| **P2P module** | Trust model, Ed25519, AES-256-GCM, QUIC transport, dispatch policy |
| **RemoteJob trait** | Slurm + Ray launchers |
| **DDP single-node** | `torchrun --nproc_per_node=N`, auto-detect from `WORLD_SIZE` |
| **DDP multi-node** | `MASTER_ADDR`/`MASTER_PORT`/`NODE_RANK` → torchrun rendezvous (multi-node NCCL run still pending hardware) |
| **Multi-GPU discovery** | Per-GPU VRAM via nvidia-smi + rocm-smi |
| **DAG optimizer** | Dead code elimination, critical path scheduling, cache/memory-aware ordering |
| **Notify sidecar** | `crates/blut-notify`: stdin envelope → custody gate → sink; Restricted is node-local |

## Scaling ladder

| Rung | Scale | What you can do | Status |
|------|-------|-----------------|--------|
| **1 GPU** | 1 machine | Typed orchestration and resource admission | Locally exercised; cookbook runtime evidence is separate |
| **Multi-GPU** | 1 machine, N GPUs | `torchrun` argument construction | Component-tested; real training parity is cookbook-owned |
| **Multi-Node** | 1 cluster, N machines | Slurm/torchrun rendezvous wiring | Component-tested; NCCL run pending |
| **Cluster** | 1 datacenter | Ray `RemoteJob` command construction | Component-tested; at-scale run pending |
| **P2P Mesh** | N machines, async | Trust-based dispatch and transport | Loopback/component-tested; independent-host run pending |

## Platform support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux x86_64 (NVIDIA)** | Primary | Local systemd/cgroup2/rlimit tests and `nvidia-smi` discovery |
| **Linux x86_64 (AMD)** | Component-tested | `rocm-smi` discovery path; no release-scale training claim |
| **macOS ARM64** | Compile-gated | CI compile-check is specified; containment degrades off-systemd |
| **Windows** | Deferred | Bare fallback only; Job Object containment is absent |

## Build your own cookbook

```rust
// 1. Define your backend
struct MyBackend;
impl TrainingBackend for MyBackend { const ID: &str = "my"; }

// 2. Define your stages
struct Train;
#[async_trait]
impl Stage for Train {
    const NAME: &str = "train";
    type Input = Dataset;
    type Output = Checkpoint;
    type Args = TrainArgs;
    // ...
}

// 3. Wire a recipe
fn compile(args: TrainArgs) -> Plan<(), MyBackend> {
    Plan::new("my_recipe", json!(args))
        .start(LoadData, load_args)
        .then(Train, train_args)
        .finish()
}

// 4. Run it
blut::cli::run(registry()).await
```

Cookbooks may also expose an arbitrary custom TUI. `CookbookTui` only tells the
engine how to discover and launch it; it does not replace Ratatui or prescribe a
widget toolkit, event loop, layout, or component model. A cookbook can ship a
normal Ratatui application or sidecar binary and use BLUT's generic cockpit only
as an optional fallback.

## Status

**Source version: 0.2.0-alpha.1.** Check crates.io before assuming registry
availability. Local 1.x tags are preserved as internal milestone history, not
public-release evidence.

**Experimental:** P2P module, DDP launch wiring, and DAG optimizer remain under
active validation; per-rung status lives in the
[distributed-validation ledger](https://github.com/Quitetall/blut/blob/v0.2.0-alpha.1/docs/DISTRIBUTED_VALIDATION.md).

Engine integration and property tests run in CI; exact counts live in CI, not
this README. The standalone fuzz workspace is not yet CI-gated. Deprecated
worker and experimental operator prototypes are not part of the public preview
or published packages.

## License

[GNU AGPL-3.0-or-later](LICENSE). BLUT is free software.

Note the AGPL's **network clause** (§13): if you run a modified version of BLUT
as a network-accessible service, you must offer that service's users the
corresponding source of your modified version.

A **commercial license** is available from the maintainer on request.

## Links

- [API reference](API.md)
- [Validation status](https://github.com/Quitetall/blut/blob/v0.2.0-alpha.1/docs/DISTRIBUTED_VALIDATION.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Release procedure](RELEASING.md)
