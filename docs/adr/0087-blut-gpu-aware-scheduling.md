<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0087 — Schedule GPU stages by topology and VRAM under a raised semaphore, with the never-OOM broker still gating RAM

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text; correct an earlier statement with a *new* dated
> entry beneath it. When the deliverable passes its gate, fill **Completion** and
> set Status to `Complete — frozen YYYY-MM-DD`.

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-10`

## Context

BLUT's executor serializes every GPU-bearing stage behind a single global
semaphore (`gpu_permits = 1`). That was correct for the 1-GPU dev box: it made
"never OOM the GPU" trivially true by never letting two CUDA jobs coexist. But
the roadmap targets 2+ GPU boxes (FSDP2, ADR 0065 [not federated]) and multi-GPU cloud workers
(ADR 0082), where a permits-of-1 gate leaves N-1 GPUs idle and forces
sequential execution of independent stages that would fit side by side.

The competition schedules on the accelerator as a first-class dimension — Ray
tracks fractional `num_gpus` per task with a placement-group topology model;
Flyte requests `gpu` + `gpu-type` and the k8s scheduler bin-packs. BLUT has
nothing between "one GPU semaphore" and "the RAM broker" (ADR 0046/0047), which
knows only host memory. VRAM is unmodeled: a stage that requests a GPU gets the
whole machine's GPU serialization whether it needs 2 GB or 40 GB.

The charter constraint is load-bearing here. ADR 0034 forbids **GPU monitoring**
inside the engine (`src/`) — no polling daemon, no telemetry loop. So the
GPU inventory must be a **one-shot probe at admission time**, not a background
watcher. (This is the seam ADR 0083's Progress Log flagged: the two `nvidia-smi`
one-shot probes get consolidated here.)

## Decision

Add a **GPU-aware admission layer** that co-schedules with — never replaces —
the never-OOM RAM broker. Three parts:

1. **GPU inventory (one-shot, charter-safe).** A `GpuInventory` built once per
   executor run by a single `nvidia-smi --query-gpu=index,memory.total,uuid`
   probe (already exemption-listed), or `ROCm`/`LaunchTarget` equivalent, or an
   explicit `BLUT_GPU_INVENTORY` env override for CI/no-CUDA hosts. It is a
   *snapshot*, not a monitored resource — no polling thread, no charter breach.
2. **Raised, per-device semaphores.** Replace the single `gpu_permits=1`
   semaphore with one semaphore **per physical GPU** (permits sized to the
   inventory). A stage declares `GpuRequest { count, min_vram_mib, exclusive }`
   in its already-typed resource envelope; the scheduler picks the device set
   whose free-VRAM estimate satisfies `min_vram_mib` and holds those permits for
   the stage's lifetime. Default request stays `count=1, exclusive=true` — so an
   un-annotated stage behaves exactly as today.
3. **Two-key admission (VRAM ∧ RAM).** GPU admission is a *second* gate in front
   of the broker, never a bypass of it: a stage is admitted only when the GPU
   scheduler grants its device set **and** the RAM broker (0046/0047) admits its
   host-memory envelope. Over-budget on **either** axis ⇒ the stage waits, never
   launches — the box is never OOM'd on host RAM or on VRAM.

`CUDA_VISIBLE_DEVICES` is set from the granted device set at `pre_exec`, so the
child process sees exactly its slice; placement is enforced at the process
boundary, not trusted to the framework.

## Rationale

- **Permits-per-device is the minimal generalization** of the existing
  semaphore: at inventory size 1 it is byte-identical to today's behaviour, so
  the dev box is unaffected and the change is provably safe there.
- **VRAM as a soft admission estimate, not a hard monitor.** We cannot poll
  (charter), so we admit on a *declared* `min_vram_mib` + a conservative
  per-device free estimate seeded from the one-shot probe and decremented by
  granted requests. This mirrors 0047's measured-envelope philosophy: admit on a
  budget, let the process own the enforcement.
- **The broker is never bypassed.** Making GPU admission a gate *in series with*
  the RAM broker (both must say yes) keeps the never-OOM invariant total — a GPU
  grant can't sneak a RAM-heavy stage past 0046.
- **Topology-aware placement earns real speedup only on multi-GPU.** Co-locating
  an FSDP2 stage's `count=k` devices on the same NVLink island (from UUID/index
  adjacency) is where the win is; on a 1-GPU box it is a no-op.

## Alternatives Considered

- **Fractional GPU sharing (Ray-style `num_gpus=0.5`).** Rejected for v1: MPS /
  time-slicing gives no VRAM isolation, so two half-GPU stages can still jointly
  OOM the device — a direct violation of the never-OOM invariant we are here to
  preserve. Exclusive-by-default whole-device grants keep the guarantee; revisit
  fractional only behind hard MIG partitioning.
- **Delegate GPU placement entirely to the framework (torchrun / FSDP picks
  devices).** Rejected: the framework has no view of *other* BLUT stages
  contending for the same box, so two concurrently-admitted stages would both
  grab GPU 0. The scheduler must own device assignment to prevent collisions.
- **A continuous GPU telemetry service (like Ray's dashboard).** Rejected on
  charter: ADR 0034 bans in-engine GPU monitoring. One-shot inventory only.

## Consequences

- The single `gpu_permits` field becomes a `GpuScheduler` holding per-device
  semaphores; every current call site that acquired the global permit migrates
  to `acquire(GpuRequest)`. Default request preserves current serialization.
- `GpuRequest` becomes part of the typed resource envelope (serde, in
  `blut-types`) — a new additive field, defaulting to the whole-device request.
- Two-key admission adds a wait path that can block on VRAM even when RAM is
  free; the status.jsonl reason string must distinguish `waiting: vram` from
  `waiting: ram` so operators can tell which key is short.
- Commits us to keeping VRAM an *estimate*: if a stage under-declares
  `min_vram_mib` and OOMs its own GPU, that is a stage-local crash (durable
  resume catches it), not a box-wide OOM — the isolation boundary holds.

## Implementation Plan  *(the contract — amend only by a dated Progress Log entry)*

- **Deliverable:** `GpuScheduler` + `GpuInventory` + `GpuRequest` (typed,
  serde), wired into the executor admission path in series with the RAM broker;
  `CUDA_VISIBLE_DEVICES` set from the grant at `pre_exec`.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut gpu_sched -- --nocapture` (inventory
    override to a synthetic 4-GPU set; asserts: `count=2` stage co-locates on
    adjacent indices, VRAM-over-budget stage waits, a granted set masks
    `CUDA_VISIBLE_DEVICES`, and a 1-GPU inventory is byte-identical to the
    legacy single-semaphore path)
  - `pass when:` exit 0 **and** a no-annotation run on inventory size 1
    reproduces the current serialization order exactly.
- **Rollback:** the scheduler is one module behind `gpu_permits` construction;
  revert to a fixed `Semaphore::new(1)` and drop the `GpuRequest` field default
  — un-annotated stages are unaffected, so rollback is behaviour-neutral.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-10 — LANDED (engine main `642cbd3`)._ `src/broker/gpu.rs`:
  `GpuInventory` (the single consolidated one-shot probe — the scattered
  `nvidia-smi` shell-outs in `probe.rs` + `launcher.rs` now source from it; the
  launcher's `nvidia-smi -L` is gone; `BLUT_GPU_INVENTORY` override for CI/tests),
  `GpuRequest{count,min_vram_mib,exclusive}` (serde, default = whole-device
  exclusive == legacy), `GpuScheduler` (a counting `avail` admission gate —
  deadlock-free, no fixed device ordering — then non-blocking
  availability-preferring device grabs), `GpuGrant` (exclusive per-device permits
  + `cuda_visible_devices()`). Executor: `ExecCtx`/`NodeEnv` carry
  `gpu: Arc<GpuScheduler>` (default 1 device ⇒ byte-identical to the old
  `Semaphore::new(1)`); `run_node` acquires the grant IN SERIES with the RAM broker
  and stamps the granted devices onto `StageContext` (`gpu_devices` + `device_index`)
  so a backend masks `CUDA_VISIBLE_DEVICES` to exactly its slice — the exclusive
  per-device semaphore is the hard no-collision guarantee regardless of the
  backend. `Stage`/`StageDyn::gpu_request` derives from `gpu_permits`.
  `with_resource_limit(Gpu,n)` sizes the scheduler homogeneous (tests unchanged);
  `with_gpu_scheduler` is the prod path (live probe, VRAM-aware).
  Scope-guard updated (gpu.rs is the by-design probe; launcher entry retired) —
  `scope_guard.sh` PASS. Review (MiMo) fixes folded in: env-test lock, restored the
  DDP-width degradation warning, `CUDA_VISIBLE_DEVICES=""` ⇒ no GPUs.
  Out of scope (own follow-ups): Ray/K8s GPU-flag threading; CUDA-OOM→re-grant;
  `blut-types` migration of `GpuRequest` (rides the ADR-0083 crate split)._
- _2026-07-12 — POST-FREEZE FOLLOW-UPS CLOSED (engine main `8d4e27f`), two of the
  three deferred items, now that ADR 0083 M1 landed the `blut-types` keystone._
  (1) **`GpuRequest` → `blut-types`.** It's a pure-serde resource-envelope WIRE
  type (rides the PlanSpec, crosses to the distributed launchers + the blut-web
  sidecar), so it moved to `crates/blut-types/src/gpu.rs`; `src/broker/gpu.rs`
  re-exports it at the historical `crate::broker::gpu::GpuRequest` path (`pub use
  blut_types::gpu::GpuRequest`) — zero churn, `GpuInventory`/`GpuScheduler` stay
  engine-side (tokio/process, not wasm-safe). (2) **Ray/K8s GPU threading.**
  `RayLauncher.num_gpus: Option<u32>` → `--entrypoint-num-gpus=<n>` before the
  `--` (sourced from `BLUT_RAY_NUM_GPUS`, mirroring `BLUT_SLURM_GPUS`);
  `PlanSubmission.gpus: u32` → a `gpus:` BlutPlan manifest field the operator maps
  to a `nvidia.com/gpu` pod limit (the CLI operator-submit path that sets it rides
  ADR 0105 [not federated]). PARITY invariant held: `None`/`0` emit nothing ⇒ byte-identical to the
  pre-ADR command/manifest. Gate: `cargo test -p blut-types` = 6 (incl 2
  GpuRequest), wasm32 check green (GpuRequest is wasm-safe), `gpu_sched` = 6
  (re-export intact), launcher `ray` = 10, `k8s` = 5; full suite green; MiMo
  PASS WITH NITS (the `PlanSubmission::gpus` serde-default flag was a verified FP —
  the struct has no `Deserialize` derive). REMAINING deferred: CUDA-OOM→re-grant
  (`RetryOn::OutOfMemoryOnly` + raise `min_vram_mib`) — a later slice._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut gpu_sched`
- **result:** `exit 0 — 6 tests` (count=2 adjacency · VRAM-floor unsatisfiable-not-a-hang
  · busy-device-waits · size-1 legacy parity · `CUDA_VISIBLE_DEVICES` masking · env
  override). Also: 848 `tui,p2p` lib tests green incl. the DDP-holds-pool + two-cells-
  overlap barrier tests (now scheduler-backed); clippy `-D warnings` (default + tui,p2p)
  + fmt clean; `scope_guard.sh` PASS. Hardware-gated tail (real 2-GPU concurrent
  execution) remains for the tier-4 validation ledger.
- **git_sha:** `642cbd3`  ·  **artifact sha256:** — (source change; no artifact)

## Related Decisions

ADR 0046 (never-OOM RAM broker — this ADR's co-scheduler, never bypassed), ADR
0047 (measured-envelope admission — the VRAM-estimate precedent), ADR 0065 [not federated]
(FSDP2 — the primary multi-GPU consumer), ADR 0034 (charter — one-shot probe, no
monitoring), ADR 0104 [not federated] (allocated: multi-GPU cost/utilization accounting).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: any concurrently-admitted pair of stages is ever assigned an
overlapping device set; a GPU grant lets a stage bypass the RAM broker; the
inventory acquires a polling thread (charter breach); or a 1-GPU box's execution
order diverges from the pre-ADR baseline. Any of these triggers a corrective
Progress-Log entry or a superseding ADR.
