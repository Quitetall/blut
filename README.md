# BLUT — Brian Lam's Universal Trainer

A **Rust-native, compile-time-typed orchestration framework for ML training
that scales from 1 GPU to 100 datacenters.** You wire stages into a typed DAG;
BLUT runs it against a content-addressed cache, under per-stage memory
containment, with structured observability — and refuses to wire two stages
whose types don't line up.

```toml
[dependencies]
blut = "1.2"
```

## What BLUT is

BLUT is the **git of ML training**. The core is a DAG orchestrator. Everything
else — resource brokerage, containment, P2P, HPO, lineage, cloud compute — is
a layer on top.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI / TUI (v1.3.0)                           │
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
pipeline that never OOMs, never loses data, and scales from 1 GPU to 100
datacenters:

```rust
let plan = Plan::<(), MyBackend>::new("train", json!({}))
    .start(PrepareData, prep_args)
    .then(Train, train_args)
    .then(Evaluate, eval_args)
    .finish();

let result = ParallelExecutor::execute(plan.into_compiled(), ctx).await?;
```

## Key features

### v1.2.0 — Current release

| Feature | What |
|---------|------|
| **Core cookbook** | 34 generic ingredient specs (data/model/optimizer/scheduler/loss/step/ema/checkpoint/eval/sampler/logging/forward) |
| **DDP single-node** | `torchrun --nproc_per_node=N`, auto-detect from `WORLD_SIZE` |
| **DDP multi-node** | `MASTER_ADDR`/`MASTER_PORT`/`NODE_RANK` → torchrun rendezvous |
| **Multi-GPU discovery** | Per-GPU VRAM via nvidia-smi + rocm-smi |
| **DAG optimizer** | Dead code elimination, critical path scheduling, cache/memory-aware ordering |
| **Cloud worker** | File-based queue + REST API (`POST /jobs`, `GET /jobs/:id`) |
| **Platform validation** | Linux x86_64, Apple M1, AMD MI300X |

### v1.1.0

| Feature | What |
|---------|------|
| **Ingredient system** | 12 kinds, registry pattern, frozen config validation |
| **P2P module** | Trust model, Ed25519, AES-256-GCM, QUIC transport, dispatch policy |
| **RemoteJob trait** | Slurm + Ray launchers |
| **Containment** | systemd → cgroup2 → rlimit → bare fallback chain |

### v1.0.0

| Feature | What |
|---------|------|
| **DAG orchestrator** | Stage → Plan → Recipe, typed wiring, content-addressed cache |
| **Parallel executor** | Resource semaphores (GPU/CPU/Disk/Network), box-fit budget |
| **Broker** | RAM admission gate, footprint estimation, OOM-aware calibration |
| **Durable resume** | Crash-gated recovery, epoch-level resume |
| **Observability** | status.jsonl, metric store, lineage index |

## Scaling ladder

| Rung | Scale | What you can do |
|------|-------|-----------------|
| **1 GPU** | 1 machine | Train any model, any recipe, full ingredient system |
| **Multi-GPU** | 1 machine, N GPUs | DDP/FSDP, N× throughput, larger models |
| **Multi-Node** | 1 cluster, N machines | Slurm/torchrun, NCCL interconnect, 100+ GPUs |
| **Cluster** | 1 datacenter | K8s/Ray, auto-scaling, distributed cache |
| **P2P Mesh** | N machines, async | Trust-based dispatch, heterogeneous compute |
| **Cloud Queue** | Any | Submit jobs, get results, zero environment setup |

## Cloud compute queue

Instead of renting a cloud GPU and setting up an environment, submit a BLUT
job to a cloud queue:

```bash
# Submit a job
curl -X POST http://worker:8080/jobs \
  -H 'Content-Type: application/json' \
  -d '{"recipe": "train_from_dataset", "args": {"dataset": "imdb", "model": "distilbert-base-uncased", "epochs": 3}}'

# Check status
curl http://worker:8080/jobs/job-abc
```

The worker runs the full BLUT DAG (data prep → training → eval → checkpoint)
and returns the result. No environment setup. No idle GPU time.

## Platform support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux x86_64** | ✅ Full | systemd/cgroup2 containment, nvidia-smi GPU discovery |
| **Linux AMD64** | ✅ Full | rocm-smi GPU discovery, HIP via CUDA shim |
| **macOS ARM64** | ✅ Validated | MPS, rlimit containment, 34 ingredients build |
| **Windows** | ⚠️ Partial | Compiles, bare containment only |

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

## Status — 1.2

The framework, typed DAG, content-addressed cache, containment + admission,
DDP, DAG optimizer, and cloud worker are stable and end-to-end runnable.
The TUI cockpit ships in v1.3.0.

**644 tests pass** (562 engine + 17 core cookbook + 59 backends + 6 worker).

## License

[GNU AGPL-3.0-or-later](LICENSE). BLUT is free software.

Note the AGPL's **network clause** (§13): if you run a modified version of BLUT
as a network-accessible service, you must offer that service's users the
corresponding source of your modified version.

A **commercial license** is available from the maintainer on request.

## Links

- [Full vision plan](../../docs/proposals/blut-full-vision-2026-06.md)
- [Cloud compute queue](../../docs/proposals/blut-cloud-compute-queue.md)
- [P2P distributed compute](../../docs/decisions/0061-blut-p2p-compute.md)
- [API reference](API.md)
- [Contributing](CONTRIBUTING.md)
