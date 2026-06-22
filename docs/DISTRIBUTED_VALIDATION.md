# BLUT Distributed / Cross-Platform — Validation Status

Tracks the 1→100 GPU + cross-platform/cloud scaling work against what is
**validated**, **architected (code landed, validation pending hardware)**, and
**deferred**. Companion to the plan `~/.claude/plans/rosy-beaming-hollerith.md`.

Branch: `feat/distributed-scaling` (engine + lamquant cookbook). All work is on a
feature branch off `main`; not yet merged/pushed (owner action).

## Summary by layer

| Layer | Capability | Status | Validated on |
|-------|-----------|--------|--------------|
| Containment | Pluggable trait (systemd/cgroup2/rlimit/bare) + cloud-bug fix | ✅ **Validated** | local + (rlimit) end-to-end |
| Containment | `RLIMIT_AS` portable cap (cloud fallback) | ✅ **Validated** | real spawn: 100 GiB → MemoryError under 512 MiB cap |
| Containment | `CgroupV2Direct` (delegated cgroup, no systemd) | 🟡 Architected | needs a box that delegates a cgroup subtree (Thunder k8s = read-only → falls back, correct) |
| Containment | `WindowsJobObject` | ⬜ Stub | future |
| L1 DDP | GPU-permit resource model (1 job spans N GPUs) | ✅ **Validated** | engine executor tests (2-GPU pool coexistence) |
| L1 DDP | torchrun argv wrap (recipe→stage→runner) | ✅ **Validated** | `kernel_argv` unit tests (single/single-node/multi-node) |
| L1 DDP | Python `train_joint.py` DDP | ⬜ **Architected (not yet written)** | needs private wheel + EEG data |
| L1 DDP | 2×A100 end-to-end DDP run | ⬜ Pending hardware | Thunder instance was torn down |
| L2 Cluster | Slurm/Ray submit+poll+stream+cancel (`RemoteJob`) | ⬜ Not started | — |
| L3 Multi-node | torchrun c10d rendezvous + Slurm node flags | 🟡 Argv landed (in runner `kernel_argv`) | needs a real cluster |

## What is VALIDATED (tests green, on this machine)

**Engine** (`cargo test`: 534 lib + 2 integration, clippy 0 warnings):
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

**Cookbook** (`cargo test`: 159 lib, clippy clean on touched files, `blut`
binary builds):
- Runner wired to `containment_for()` with backend-agnostic fail-closed refusal.
- `kernel_argv` torchrun wrap (4 tests): single-process = plain python;
  single-node DDP = `python -m torch.distributed.run --standalone
  --nproc_per_node=n`; multi-node = c10d rendezvous flags from launcher env
  (with a warn on missing `MASTER_ADDR`/`NODE_RANK` to catch silent
  self-rendezvous).
- `nproc_per_node` / `nnodes` threaded recipe → stage → invocation.

**Earlier hardware validation** (prior session, 2×A100 Thunder box, see
`[[project_blut_multigpu_validation]]`): the per-device scheduling + CUDA_VISIBLE
pinning + a synthetic torchrun DDP prototype (world_size=2, NCCL all-reduce)
were proven on real 2×A100. The work above promotes that prototype to typed,
tested production code.

## What is ARCHITECTED but NOT yet written / validated

- **Python `train_joint.py` DDP (B4):** the exact changes are specified in the
  plan (init_process_group/nccl, per-rank seed + param broadcast, submodule-level
  DDP wrap of encoder/decoder/GAN-disc/seizure-head with
  `find_unused_parameters=True`, data shard in `lma_typed_adapter.py`'s
  `_sample_epoch_indices` by global rank, rank-0-only saves/metrics with a
  barrier, `mode='default'` torch.compile under DDP, `.module`/`_orig_mod`
  state-dict unwrap). Not yet written — gloo-testable without GPUs; the real run
  needs the private `lamquant-neural` wheel + EEG data.
- **2×A100 end-to-end DDP run:** the Thunder instance was torn down mid-session;
  re-validate when a box is up.

## What is DEFERRED (needs a real cluster — not validatable on a single box)

- **L2 `RemoteJob` (Slurm/Ray submit+poll+stream+cancel):** the current cluster
  arm spawns the submit *client* and waits on it, so remote progress/OOM/cancel
  are broken. The fix (a `RemoteJob` handle with `sbatch --parsable` + `sacct`
  poll + log tail + `scancel`; `ray job submit --no-wait` + status/logs/stop) is
  designed in the plan, unit-testable now (argv/handle shape), cluster-validated
  later. **Not started.**
- **L3 multi-node NCCL across nodes, IB-vs-TCP, data stage-in, 100-GPU scale:**
  the torchrun multi-node argv is in `kernel_argv`; the Slurm node flags +
  sbatch MASTER_ADDR preamble are designed. Cannot be exercised on a single node.

## How to validate the pending pieces when a box is up

1. `tnr create --gpu a100 --num-gpus 2`; re-establish `ssh thunder-blut` from
   `tnr status --json` (instances are ephemeral — uuid/ip/port/key all change).
2. Sync the `feat/distributed-scaling` branch of engine + cookbook; `cargo build
   --bin blut`.
3. cgroup2 cap proof: on a box that delegates a cgroup subtree (or via the
   passwordless-sudo root cgroup), confirm a 100 GiB allocation is OOM-killed at
   the cap and `memory.peak` is read; `cgroup.kill` reaps the tree.
4. DDP: once B4 lands, run the joint recipe with `nproc_per_node: 2`; confirm
   `world_size=2`, both A100s used, val_r parity vs a 1-GPU run at the same
   global batch, and a mid-run kill reaps the whole rank tree.
