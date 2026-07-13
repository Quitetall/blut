# BLUT — Brian Lam's Universal Trainer

A **Rust-native, compile-time-typed orchestration framework for ML training.**
You wire stages into a typed DAG; BLUT runs it against a content-addressed
cache, under per-stage memory containment, with structured observability, and
BLUT refuses to wire two stages whose types don't line up. Validated from a
single GPU box to Slurm/Ray clusters; a P2P mesh and cloud job queue are in
active development (see the [scaling ladder](#scaling-ladder) for what is
validated vs. in flight).

```toml
[dependencies]
blut = "1"
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
│  Containment (systemd / cgroup2 / rlimit / bare / JobObject)    │
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
pipeline built to never OOM the box, never lose data, and grow from one GPU
to clusters without rewriting the pipeline:

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

### On `main`, unreleased

Landed and CI-gated, but not yet cut as a release — validation status per
capability is tracked in [DISTRIBUTED_VALIDATION.md](docs/DISTRIBUTED_VALIDATION.md):

| Feature | What |
|---------|------|
| **Ingredient system** | 12 kinds, registry pattern, frozen config validation |
| **P2P module** | Trust model, Ed25519, AES-256-GCM, QUIC transport, dispatch policy |
| **RemoteJob trait** | Slurm + Ray launchers |
| **DDP single-node** | `torchrun --nproc_per_node=N`, auto-detect from `WORLD_SIZE` |
| **DDP multi-node** | `MASTER_ADDR`/`MASTER_PORT`/`NODE_RANK` → torchrun rendezvous (multi-node NCCL run still pending hardware) |
| **Multi-GPU discovery** | Per-GPU VRAM via nvidia-smi + rocm-smi |
| **DAG optimizer** | Dead code elimination, critical path scheduling, cache/memory-aware ordering |
| **Cloud worker** | `crates/blut-worker`: file-based queue + token-guarded REST API (experimental) |

## Scaling ladder

| Rung | Scale | What you can do | Status |
|------|-------|-----------------|--------|
| **1 GPU** | 1 machine | Train any model, any recipe, full ingredient system | ✅ Validated |
| **Multi-GPU** | 1 machine, N GPUs | torchrun DDP wrap, N× throughput | ✅ Validated (single-node) |
| **Multi-Node** | 1 cluster, N machines | Slurm/torchrun rendezvous wiring | 🟡 Architected; NCCL run pending hardware |
| **Cluster** | 1 datacenter | Ray `RemoteJob`, distributed cache | 🟡 Launchers validated; at-scale runs pending |
| **P2P Mesh** | N machines, async | Trust-based dispatch, heterogeneous compute | 🟡 In development |
| **Cloud Queue** | Any | Submit jobs, get results, zero environment setup | 🧪 Experimental (`blut-worker`) |

## Cloud compute queue (experimental)

Instead of renting a cloud GPU and setting up an environment, submit a BLUT
job to a worker's queue. `POST /jobs` executes recipes — arbitrary code — so
the API is fail-closed: it binds loopback by default, and serving a
non-loopback address requires a bearer token (the worker refuses to start
otherwise).

```bash
# On the worker box: token in the env (never argv), explicit non-loopback bind
BLUT_WORKER_TOKEN=$(openssl rand -hex 32) \
  blut-worker --queue-dir /var/blut/queue --work-dir /var/blut/work \
              --results-dir /var/blut/results --api-port 8080 --api-bind 0.0.0.0

# Submit a job
curl -X POST http://worker:8080/jobs \
  -H "Authorization: Bearer $BLUT_WORKER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"recipe": "train_from_dataset", "args": {"dataset": "imdb", "model": "distilbert-base-uncased", "epochs": 3}}'

# Check status
curl -H "Authorization: Bearer $BLUT_WORKER_TOKEN" http://worker:8080/jobs/job-abc
```

The worker runs the full BLUT DAG (data prep → training → eval → checkpoint)
and returns the result. The queue is currently file-based; the productionized
data plane (P2P artifact bundles + object store) is specified in the cloud
compute queue proposal linked below.

## Platform support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux x86_64 (NVIDIA)** | ✅ Full | systemd/cgroup2 containment, nvidia-smi GPU discovery |
| **Linux x86_64 (AMD)** | ✅ Validated | rocm-smi GPU discovery (validated on MI300X) |
| **macOS ARM64** | ✅ Validated | rlimit containment, CI compile-check (containment degrades off-systemd) |
| **Windows** | ⚠️ Partial | Compiles, bare containment only (JobObject containment is a stub) |

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

**Development tree: 0.2.0-alpha.1.** The latest crates.io `blut` release is
0.1.0; this preview has not been published. Local 1.x tags are preserved as
internal milestone history, not public-release evidence.

**On main, unreleased:** P2P module, DDP launch wiring, DAG optimizer, and
the experimental cloud worker are end-to-end runnable and CI-gated; per-rung
validation status lives in
[DISTRIBUTED_VALIDATION.md](docs/DISTRIBUTED_VALIDATION.md).

The engine suite (620+ lib tests plus integration, property, and fuzz
targets) and the worker suite run green in CI; exact counts live in CI, not
this README.

## License

[GNU AGPL-3.0-or-later](LICENSE). BLUT is free software.

Note the AGPL's **network clause** (§13): if you run a modified version of BLUT
as a network-accessible service, you must offer that service's users the
corresponding source of your modified version.

A **commercial license** is available from the maintainer on request.

## Links

- [API reference](API.md)
- [Validation status](docs/DISTRIBUTED_VALIDATION.md)
- [Contributing](CONTRIBUTING.md)

Design history (these live in the parent meta-repo, not this crate — the
links resolve only from a meta-repo checkout):

- [Full vision plan](../../docs/proposals/blut-full-vision-2026-06.md)
- [Cloud compute queue](../../docs/proposals/blut-cloud-compute-queue.md)
- [P2P distributed compute](../../docs/decisions/0061-blut-p2p-compute.md)
