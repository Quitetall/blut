# BLUT Distributed / Cross-Platform — Validation Status

Tracks the 1→100 GPU + cross-platform/cloud scaling work against what is
**validated**, **architected (code landed, validation pending hardware)**, and
**deferred**. Companion to the plan `~/.claude/plans/rosy-beaming-hollerith.md`.

Branch: `feat/distributed-scaling` (engine + lamquant cookbook).

## Summary by layer

| Layer | Capability | Status | Validated on |
|-------|-----------|--------|--------------|
| Containment | Pluggable trait (systemd/cgroup2/rlimit/bare) + cloud-bug fix | ✅ **Validated** | local + (rlimit) end-to-end |
| Containment | `RLIMIT_AS` portable cap (cloud fallback) | ✅ **Validated** | real spawn: 100 GiB → MemoryError under 512 MiB cap |
| Containment | `CgroupV2Direct` (delegated cgroup, no systemd) | 🟡 Architected | needs a box that delegates a cgroup subtree (Thunder k8s = read-only → falls back, correct) |
| Containment | `WindowsJobObject` | ⬜ Stub | future |
| L1 DDP | GPU-permit resource model (1 job spans N GPUs) | ✅ **Validated** | engine executor tests (2-GPU pool coexistence) |
| L1 DDP | torchrun argv wrap (recipe→stage→runner) | ✅ **Validated** | `kernel_argv` unit tests (single/single-node/multi-node) |
| L1 DDP | Python `train_joint.py` DDP | ✅ **Written** | syntax clean; gloo/NCCL validation pending |
| L1 DDP | 2×A100 end-to-end DDP run | ⬜ Pending hardware | needs GPU instance |
| L2 Cluster | Slurm/Ray `RemoteJob` (submit+poll+stream+cancel) | ✅ **Written** | 17 unit tests (sacct/ray parsing, sbatch gen); cluster validation pending |
| L3 Multi-node | SlurmLauncher `nodes`/`ntasks_per_node` + sbatch MASTER_ADDR preamble | ✅ **Written** | 2 unit tests (preamble present/absent); cluster validation pending |

## What is VALIDATED (tests green, on this machine)

**Engine** (556 lib tests, clippy 0 warnings):
- Containment trait + four backends + factory. `Availability` is three-state
  (`Present | BusOffline | Unavailable`) — the **cloud bug fix**: the old
  `containment_available()` checked the `systemd-run` *binary* (present on k8s)
  instead of the *bus* (offline there), reporting available and then failing at
  runtime. Now the factory probes the bus and falls through `BusOffline`
  backends: systemd → cgroup2 → rlimit → bare.
- **`RLIMIT_AS` cap proven end-to-end** (`tests/rlimit_containment.rs`): driving
  the real `RlimitAddressSpace::wrap_command` + spawn, a 100 GiB allocation
  under a 512 MiB cap aborts with `MemoryError`; a 64 MiB allocation under 4 GiB
  succeeds. This is the portable cloud fallback (no root, no cgroup, no
  systemd). **Caveat (documented in the backend):** `RLIMIT_AS` caps virtual
  address space, which a CUDA process reserves hugely — so it is applied only on
  explicit opt-in and sized with CUDA's VA in mind; cgroup2/systemd are the
  RSS-precise preferred caps.
- **GPU resource model** (`framework/executor.rs` tests): a DDP stage holding
  all GPU permits (2 on a 2-GPU pool) serializes a single-GPU cell; two
  single-GPU cells overlap on a 2-GPU pool. `Stage::gpu_permits(&Args)` +
  `memory_gib_for(&Args)` scale a DDP job ×nproc; the CLI sizes the pool to the
  launcher's `capacity()`.
- **`RemoteJob` trait** (`config/launcher.rs`): `JobState` enum, `RemoteJob`
  trait (id/poll/stream/cancel), `SlurmJob` (sbatch --parsable submit, sacct
  poll, log-tail stream, scancel), `RayJob` (ray job submit --no-wait, status
  poll, logs --follow, stop). 17 new tests: sacct state parsing ×9, ray status
  parsing ×7, sbatch script gen ×2, submit_async reject ×2.
- **Multi-node Slurm** (`config/launcher.rs`): `SlurmLauncher` fields
  `nodes`/`ntasks_per_node`, `--nodes`/`--ntasks-per-node` flags in `wrap()`,
  sbatch preamble exports `MASTER_ADDR`/`MASTER_PORT`/`NODE_RANK` via
  `scontrol show hostnames`. Warns when `nodes>1` without `ntasks_per_node`.

**Cookbook** (159 lib tests, clippy clean, `blut` binary builds):
- Runner wired to `containment_for()` with backend-agnostic fail-closed refusal.
- `kernel_argv` torchrun wrap (4 tests): single-process = plain python;
  single-node DDP = `python -m torch.distributed.run --standalone
  --nproc_per_node=n`; multi-node = c10d rendezvous flags from launcher env
  (with a warn on missing `MASTER_ADDR`/`NODE_RANK` to catch silent
  self-rendezvous).
- `nproc_per_node` / `nnodes` threaded recipe → stage → invocation.

**Python DDP** (`train_joint.py` + `lma_typed_adapter.py`):
- DDP init: reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE` from torchrun env,
  `init_process_group("nccl")`, per-rank seed, non-rank-0 print suppression
  (restored in finally block).
- Submodule DDP wrap: encoder (`find_unused_parameters=True`), decoder, disc,
  sz_head wrapped separately (composite codec has direct `encoder.encode` calls).
- Data shard: `_DDPClinicalSampler` + `_DDPRankSampler` yield rank-disjoint
  subsets preserving stem-grouped cache locality.
- Rank-0-only: `_emit`/BLUT_METRIC, all `torch.save` + `codec.save_*` (12
  sites), `dist.barrier()` after each save.
- torch.compile `mode='default'` under DDP.
- `_unwrap_ddp` at all critical access points (resume, CDF recal, EMA, entropy,
  seizure-head encode — documented rank-local approximation).
- `dist.destroy_process_group()` at end of `run()`.
- `lma_index_path` warning when provided but file not found.

**Earlier hardware validation** (prior session, 2×A100 Thunder box, see
`[[project_blut_multigpu_validation]]`): the per-device scheduling + CUDA_VISIBLE
pinning + a synthetic torchrun DDP prototype (world_size=2, NCCL all-reduce)
were proven on real 2×A100. The work above promotes that prototype to typed,
tested production code.

## What needs hardware validation (code landed, not yet run on GPU)

1. **2×A100 DDP end-to-end**: run `train_joint.py` under `blut recipe run` with
   `nproc_per_node=2`. Verify val_r parity 1-GPU vs 2-GPU at same global batch,
   BLUT_METRIC streams live, mid-run cancel works, OOM cap holds for ×nproc RAM.
2. **cgroup2 cap on delegated box**: confirm OOM-kill at cap + `memory.peak` read
   + `cgroup.kill` teardown on a box with writable cgroup delegation.

## What is DEFERRED (needs a real cluster — not validatable on a single box)

- **L2 cluster end-to-end**: Slurm `sbatch` → `sacct` poll → log stream →
  `scancel` mid-run; Ray `--no-wait` → status → logs → stop. Unit tests green;
  needs a real Slurm/Ray cluster to validate the full lifecycle.
- **L3 multi-node NCCL across nodes, IB-vs-TCP, data stage-in, 100-GPU scale:**
  torchrun c10d rendezvous + sbatch MASTER_ADDR preamble are landed. Cannot be
  exercised on a single node.

## How to validate the pending pieces when a box is up

1. `tnr create --gpu a100 --num-gpus 2`; re-establish `ssh thunder-blut` from
   `tnr status --json` (instances are ephemeral — uuid/ip/port/key all change).
2. Sync the `feat/distributed-scaling` branch of engine + cookbook; `cargo build
   --bin blut`.
3. cgroup2 cap proof: on a box that delegates a cgroup subtree (or via the
   passwordless-sudo root cgroup), confirm a 100 GiB allocation is OOM-killed at
   the cap and `memory.peak` is read; `cgroup.kill` reaps the tree.
4. DDP: run the joint recipe with `nproc_per_node: 2`; confirm `world_size=2`,
   both A100s used, val_r parity vs a 1-GPU run at the same global batch, and a
   mid-run kill reaps the whole rank tree.
