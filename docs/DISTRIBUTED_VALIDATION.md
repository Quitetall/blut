# BLUT Distributed / Cross-Platform — Validation Status

Tracks the 1→100 GPU + cross-platform/cloud scaling work against what is
**validated**, **architected (code landed, validation pending hardware)**, and
**deferred**.

Last updated: 2026-06-27 (v1.2.0).

## Summary by layer

| Layer | Capability | Status | Validated on |
|-------|-----------|--------|--------------|
| **Containment** | Pluggable trait (systemd/cgroup2/rlimit/bare) | ✅ Validated | local + rlimit end-to-end |
| **Containment** | `RLIMIT_AS` portable cap (cloud fallback) | ✅ Validated | real spawn: 100 GiB → MemoryError under 512 MiB cap |
| **Containment** | `CgroupV2Direct` (delegated cgroup) | 🟡 Architected | needs writable cgroup subtree |
| **Containment** | `WindowsJobObject` | ⬜ Stub | future |
| **L1 DDP** | GPU-permit resource model (1 job spans N GPUs) | ✅ Validated | executor tests (2-GPU pool coexistence) |
| **L1 DDP** | torchrun argv wrap (single-node) | ✅ Validated | core trainer + LAMU + HF backends |
| **L1 DDP** | torchrun argv wrap (multi-node) | ✅ Validated | MASTER_ADDR/PORT/RANK wiring |
| **L1 DDP** | Python DDP (train_joint.py) | ✅ Written | syntax clean; NCCL validation pending |
| **L1 DDP** | 2×A100 end-to-end DDP run | ⬜ Pending hardware | needs GPU instance |
| **L1 DDP** | Core cookbook DDP (blut_core/trainer.py) | ✅ Validated | auto-detect DDP, DDP-wrap, DistributedSampler, rank-0 saves |
| **L2 Cluster** | Slurm/Ray `RemoteJob` | ✅ Validated | 17 unit tests (sacct/ray parsing, sbatch gen) |
| **L2 Cluster** | Multi-node Slurm + MASTER_ADDR preamble | ✅ Validated | 2 unit tests |
| **L2 DAG Opt** | Dead code elimination | ✅ Validated | 3 tests (disconnected, diamond, linear) |
| **L2 DAG Opt** | Critical path scheduling | ✅ Validated | 2 tests (linear, diamond) |
| **L2 DAG Opt** | Memory-aware scheduling | ✅ Validated | 1 test (concurrent memory per level) |
| **L3 P2P** | Trust model + crypto + QUIC + dispatch | ✅ Validated | 48 unit tests |
| **L3 P2P** | Live peer validation | ⬜ Deferred | needs two machines |
| **L4 Cloud** | Worker agent (file queue + REST API) | ✅ Validated | 6 integration tests |
| **L4 Cloud** | Real queue backend (Redis/SQS) | ⬜ Deferred | needs queue infrastructure |
| **L4 Cloud** | Artifact storage (S3/R2) | ⬜ Deferred | needs storage infrastructure |

## What is VALIDATED (tests green)

### Engine (562 lib tests)

- **Containment trait** + four backends + factory. `Availability` is three-state
  (`Present | BusOffline | Unavailable`). Factory probes the bus and falls
  through: systemd → cgroup2 → rlimit → bare.
- **`RLIMIT_AS` cap** proven end-to-end (`tests/rlimit_containment.rs`).
- **GPU resource model**: DDP stage holding N GPU permits serializes correctly.
  Two single-GPU cells overlap on a 2-GPU pool.
- **`RemoteJob` trait**: `SlurmJob` + `RayJob` implementations. 17 tests.
- **Multi-node Slurm**: `SlurmLauncher` with `nodes`/`ntasks_per_node`,
  sbatch preamble exports `MASTER_ADDR`/`MASTER_PORT`/`NODE_RANK`.
- **DAG optimizer**: dead code elimination, critical path scheduling,
  cache-aware hints, memory-aware scheduling. 6 tests.
- **GPU discovery**: per-GPU VRAM via nvidia-smi + rocm-smi. GpuInfo struct
  with index, model, vram_total/free. Conservative fail-safe for AMD.

### Core cookbook (17 lib tests)

- 34 generic ingredient specs across 12 kinds.
- 3 stages: LoadDataset, TrainModel, EvaluateModel.
- 3 recipes: train_from_dataset, finetune_pretrained, eval_only.
- TrainModel: `nproc_per_node` + `nnodes` args, gpu_permits returns N,
  launches via torchrun when N>1, multi-node rendezvous from env.

### Backends (59 lib tests)

- LAMU backend: DDP via torchrun when `nproc_per_node > 1`.
- HF Trainer backend: DDP via torchrun when `nproc_per_node > 1`.
- `TrainSpec` + `HfTrainerJob` have `nproc_per_node`/`nnodes` fields.

### Cloud worker (6 integration tests)

- File-based queue (JSON files in a directory).
- REST API: `POST /jobs`, `GET /jobs`, `GET /jobs/:id`, `GET /health`.
- Atomic job processing (rename to .processing).
- Malformed job handling, concurrent queue access.

### Platform validation

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | ✅ Full | 562 engine tests, systemd/cgroup2, nvidia-smi |
| Apple M1 (macOS ARM64) | ✅ Validated | 34 ingredients, MPS, rlimit containment |
| AMD MI300X (RunPod) | ✅ Validated | 556 engine tests, rocm-smi |

## What needs hardware validation (code landed, not yet run on GPU)

1. **2×A100 DDP end-to-end**: run `train_joint.py` under `blut recipe run` with
   `nproc_per_node=2`. Verify val_r parity 1-GPU vs 2-GPU at same global batch.
2. **Core cookbook DDP on real GPU**: run `train_from_dataset` with
   `nproc_per_node=2` on a multi-GPU box.
3. **cgroup2 cap on delegated box**: confirm OOM-kill at cap + `memory.peak` read.

## What is DEFERRED (needs infrastructure)

- **Cloud queue backend**: Replace file queue with Redis/SQS for multi-worker.
- **Artifact storage**: S3/R2 for checkpoint/metric storage.
- **P2P live validation**: Two machines, real QUIC handshake, task dispatch.
- **Multi-node NCCL across nodes**: torchrun c10d rendezvous + sbatch preamble
  are landed. Cannot be exercised on a single node.

## Full vision

See `docs/proposals/blut-full-vision-2026-06.md` for the 12-phase plan from
1 GPU to 100 datacenters. See `docs/proposals/blut-cloud-compute-queue.md`
for the cloud compute queue vision.
