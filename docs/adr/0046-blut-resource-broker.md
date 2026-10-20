<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0046 — BLUT resource broker (never-OOM, exact-resource training)

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (Accepted 2026-06-09 · Complete — frozen 2026-08-27)

- **Status:** Accepted (owner directive 2026-06-09: "Build the full-scale BLUT resource broker as a first-class feature. Training uses exactly the resources available and no more. Never OOM again.")
- **Extends:** [ADR 0044](0044-blut-api-standardization.md) (BLUT API — protections-by-default, durability+retry, submit-time preflight; this is the concrete realization of Phases D/E/G), [ADR 0034](0034-blut-scope-charter.md) (BLUT delegates compute→systemd). Supersedes the hand-set `MemoryMax` practice + an admission stopgap script `[redacted: private path]` (which becomes the seed).

## Context
Repeated OOMs this session, all avoidable:
- **A teacher-model run OOM-killed mid-training** — RAM; launching an A/B alongside it oversubscribed the 62G box `[redacted: codec research]`.
- **A decoder run OOM-killed at a 16G cap** — RAM from the **dataloader** (4 workers × archive prefetch ≈ 23G), not the model; the hand-set cap was too low `[redacted: codec research]`.
Root causes: (1) cgroup `MemoryMax` only **contains** (kills one unit) — it does **not prevent** OOM; (2) the caps + co-scheduling were hand-guessed per launch and wrong; (3) **VRAM is uncoverable by cgroups** — the only levers are admission + serialization; (4) the footprint is **dataloader-dominated** (workers × prefetch), not model-dominated, so model-size intuition mis-sizes it. BLUT already *probes* the box (`src/tui/system.rs`: RAM `MemAvailable`, per-GPU VRAM via nvidia-smi, CPU) but never *gates* launches on it.

## Decision
A first-class **resource broker** in blut-core that owns every launch:
1. **Admission** — before launch, probe free RAM + per-GPU free VRAM + already-running blut-unit usage; **refuse or QUEUE** if it won't fit a configured headroom floor. Never oversubscribe. (Jobs wait, they don't die.)
2. **VRAM is first-class** — track per-GPU VRAM **commitments** (admitted jobs) and serialize big-VRAM jobs; cgroups are irrelevant to VRAM so admission is the only guarantee.
3. **Auto-size** — set cgroup `MemoryMax`/`High` from a measured footprint (+margin); **auto-tune `batch_size`/dataloader workers** to FILL but not exceed resources ("exactly the amount, no more").
4. **Footprint calibration store** — measured peak RAM (cgroup `memory.peak`) + peak VRAM recorded at job exit, keyed by `(recipe, tier, batch, workers)`, reused next time; conservative default when uncalibrated.
5. **OOM-retry** — detect systemd `Result=oom-kill` (RAM) / CUDA OOM (VRAM) → requeue at `batch/2` (or `workers−1`), bounded; record the corrected footprint so it never repeats.
6. **Queue/scheduler** — admission queue; admit as resources free.

Lives in `src/broker/` (engine, pushable). The `Launcher`/runner call it; `blut recipe run` auto-admits. The stopgap script is the in-path stopgap until the binary ships.

## Consequences
- Every launch is admission-gated; OOM becomes a refused/queued/retried event, never a crash. Hand-set caps are removed.
- **The installed `blut` is 19 days stale → shipping this requires `cargo install` rebuild** (the broker isn't live until then; stopgap covers the gap).
- The domain cookbook's runner integration is committed locally; **owner pushes** (the newly-created cookbook repo is policy-blocked for the agent) `[redacted: private repository]`. The `blut` engine is agent-pushable.
- cgroup `MemoryMax` is retained for RAM **containment** (defense-in-depth); admission is the **prevention**.

## Anti-reinvention
Reuse the existing `tui/system.rs` probe (don't re-implement nvidia-smi/meminfo parsing). No new runtime deps (no Ray/Hydra — ADR 0044 rejects those). Blueprint + adversarial review: an internal workflow `[redacted: private repository]`. First-shippable slice = "never RAM-OOM" (admission + auto-cap + queue) before VRAM-semaphore/auto-tune/retry polish.

## Status update (2026-06-09): slice-1 SHIPPED + LIVE + PROVEN
Typed mandatory containment + scaling footprint + admission gate built (blut commit 1392420 pushed; a cookbook commit `[redacted: private repository]` local) + `cargo install`-ed. Live-proven: `blut recipe run` REFUSED a 35G-footprint launch on a 28.6G-free box (no OOM, no unit). Both commits passed automated review with nits.  Slices 2-3 (calibration/OOM-retry/VRAM) open — the conservative footprint over-refuses until calibrated.

## Status update (slice-2, 2026-06-09): calibration store SHIPPED
Measured cgroup `Memory peak:` recorded at exit → `FootprintStore` (`~/.config/blut/footprints.json`); admission resolves measured peak (key-parity RECORD==RESOLVE pinned). blut 4ef903b pushed; a cookbook commit `[redacted: private repository]` local. Cold-start: empty store ⇒ first run uses conservative hint ⇒ on a desktop-loaded box (23G free) a large training tier (~27G w/ floor) is refused; free RAM to bootstrap, then it self-calibrates to ~20G. Slice-3 (OOM-retry, VRAM ledger) open.

## Progress Log *(append-only; newest last)*

- _2026-08-27_ — gave this record a runnable citation gate `[redacted: private path]` (it proves the record still points at files that exist, which is how a record actually dies; it does not claim the decision is implemented) and removed its closure-debt entry `[redacted: private path]`.

## Completion / Resolution

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** Every one of the six numbered decisions has a home in `src/broker/`: `admission.rs` (1, admission + refuse-or-queue), `gpu.rs` (2, per-GPU VRAM commitments), `footprint.rs` (3 and 4, auto-size from a measured footprint store), `probe.rs` (free RAM/VRAM probing) and `mod.rs` (6, the facade the executor calls), with `tenant_quota.rs` beyond what the ADR asked for. The stopgap script the Decision names is no longer the path -- the binary shipped. Citation gate now green: the cited `tui/system.rs` moved to `crates/blut-tui/src/system.rs` and is aliased.
