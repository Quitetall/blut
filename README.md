# BLUT — Basically Less Unsound Training

<sub>(affectionately, *Brian Lam's Universal Trainer*.)</sub>

A **semantic compiler for ML pipelines.** You declare what each node *means* —
how deterministic it is, what effects it has, whether it may produce gaps — and
BLUT kind-checks the graph, fuses what is safe to fuse, lowers it to an
execution realm, and then **runs it under the semantics you declared.**

The name is the design goal, hedges included. Not *sound* — soundness is a
strong word and this is alpha software. **Basically less unsound**: every
release should make it harder to express a pipeline whose behaviour does not
match what it claims, and honest about how far that has got.

```toml
[dependencies]
blut = "=0.2.0-alpha.1"
```

## What BLUT is

Most pipeline tools schedule tasks: they decide *when* things run. BLUT
compiles a graph: it decides whether the graph is *meaningful*, rewrites it, and
emits a plan whose execution protocol follows from the declaration.

Every node carries three declared properties, and they are lattices, not tags:

```
Determinism : BitExact → NumericallyEquivalent → Seeded → Nondeterministic
Effect      : Pure → Idempotent → Transactional → AtMostOnce → AtLeastOnce
Partiality  : Atomic | ExplicitGaps
```

**Those declarations select the runtime protocol.** Declare `Transactional` and
the executor drives prepare / commit / abort with derived idempotency keys;
declare `ExplicitGaps` and a partial result must produce a structured gap
receipt rather than quietly succeeding. The compiler's front end and the
runtime are not two systems agreeing by convention — the declaration *is* the
interface between them, and the executor branches on it.

The same compiled graph targets more than one **execution realm**:

```
McuAot        ahead-of-time plan for a microcontroller
HostStream    streaming host execution
BlutDurable   durable, cache-backed host execution
```

The compiler core lives in [`crates/blut-graph-core`](crates/blut-graph-core)
— roughly 9k lines, three dependencies (blake3, serde, postcard), `no_std`-capable
— and is published separately so it can be used without the rest of the engine.
Fusion is guarded by an identity property (`fusion_preserves_semantic_identity`):
a fused graph must denote what the unfused graph denoted.

Everything else in this repository — the resource broker, containment,
content-addressed cache, lineage, HPO, P2P, cloud queue — is the **runtime and
standard library** that makes those declarations enforceable on a real machine.

### What is exercised, and what is not

Local, single-box execution is exercised: the engine's own suite runs in CI, and
the codec cookbook that drives this project has hundreds of completed runs in
its ledger. Multi-GPU, cluster, P2P, and cloud paths carry only the bounded
evidence recorded in the [scaling ladder](#scaling-ladder) — **BLUT is a
single-box system in practice today.** Distributed scheduling is designed and
partially built, not proven. Treat the scaling ladder, not this paragraph, as
the authority.

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
| **Semantic compiler** | per-node determinism / effect / partiality; kind-checked graph; fusion guarded by an identity property |
| **Realm lowering** | one graph → `McuAot`, `HostStream`, or `BlutDurable` |
| **Effect-directed runtime** | `Transactional` drives prepare/commit/abort; `ExplicitGaps` requires a structured gap receipt |
| **DAG orchestrator** | Stage → Plan → Recipe, typed wiring, content-addressed cache |
| **Parallel executor** | Resource semaphores (GPU/CPU/Disk/Network), box-fit budget |
| **Broker** | RAM admission gate, footprint estimation, OOM-aware calibration |
| **Containment** | systemd → cgroup2 → rlimit → bare fallback chain |
| **Durable resume** | Crash-gated recovery, epoch-level resume |
| **Observability** | status.jsonl, metric store, lineage index |
| **TUI cockpit** | the `crates/blut-tui` sidecar (ADR 0083): in-process via `blut_tui::hook()` in a cookbook binary, or `blut tui` execs the `blut-tui` binary |
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
| **Event daemon** | `blut sensord`: file-drop, spool-threshold, and cron triggers → durable dedupe → exact recipe admission acknowledgement |
| **Webhook ingress** | `blut-web`: HMAC-SHA256 signed, timestamp-bounded, content-addressed replay-safe events; Restricted stays local |
| **Notify sidecar** | `crates/blut-notify`: durable status/SLA tail → declarative rules → Slack, Discord, ntfy, SMTP, or exec; Restricted is node-local |

See [Eventing, SLA, and notifications](EVENTING.md) for the shared trigger file,
webhook signing contract, SLA rules, notification sinks, and restart semantics.

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

### Declare your resources — one method, the whole stack

Implement `Stage::resource_envelope` and your stage gets BLUT's entire
resource machinery: the launch gate admits it against real memory, the
measured-peak store calibrates the estimate run over run (with an OOM-aware
self-heal), and the engine auto-tunes any dimension you declare a cost term
for — you enumerate the coefficients and ceilings, the engine owns the
search. Skip it and your stage still runs, billed a loud 2 GiB
compatibility floor.

```rust
fn resource_envelope(&self, args: &Self::Args) -> ResourceEnvelope {
    ResourceEnvelope {
        ram_bytes: 30 << 30,                                    // 30 GiB peak
        calibration_dimensions: vec![("batch".into(), args.batch.to_string())],
        cost_terms: vec![CostTerm {                             // auto-tunable
            dimension: "batch".into(),
            declared_units: args.batch,
            ram_bytes_per_unit: 64 << 20,                       // 64 MiB/unit
            max_units: args.batch,
            ..Default::default()
        }],
        ..Default::default()
    }
}
```

(The full working version lives in `examples/first_cookbook.rs` — CI runs it.)


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
