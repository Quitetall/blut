<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0047 — BLUT broker measured-envelope hardening + codec dataloader efficiency + production handoff

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (Accepted · Complete — frozen 2026-08-27)

- **Date:** 2026-06-10
- **Supersedes/amends:** tightens [ADR 0046](0046-blut-resource-broker.md) (broker) slice-2; companions the
  efficiency work to [ADR 0034](0034-blut-scope-charter.md) (BLUT scope). Path-to-production captured in
  a roadmap document `[redacted: private path]`.

## Context

Continuation session. A decoder A/B and every other codec warm-start training
run was OOM-killed at the cgroup cap (8/8 overnight launches dead), which the
prior session misattributed in turn to: a feature cache, a dataloader/worker leak,
`torch.compile`, the parent Python heap, and a "per-cumulative-window leak." All
five were **disproven by measurement** (tracemalloc flat across epochs; cuda flat;
dataloader-only RSS probe flat 4.6 G over 3 epochs; the exact same config later
COMPLETED successfully at 13 G on a settled box). The codec also had a genuine
dataloader-efficiency problem (decode re-run every epoch; an O(600 GB)
whole-recording cache) `[redacted: codec research]`.

## Decision

### 1. The OOM was a resource-MODEL error, not a code leak
The codec warm run's true working set is ≈ **32 GiB** (≈23 G RSS + ≈9 G swap at
4 dataloader workers). The cap was 25 G because a prior "right-size" (blut
`51bcc43`) lowered `PREFETCH_PER_WORKER` 3.5→2.0 G — the cap then sat AT the
swapping peak with zero headroom and OOM-killed under any added box pressure
(page-cache of a very large memory-mapped training corpus + an 8-launch
swap-thrash). **Fix (blut `db39698` + a cookbook commit `[redacted: private repository]`):**
- `UNCALIBRATED_WORKER_CAP` 4→**2** (+ cli + the trainer's worker-count
  defaults, key-parity preserved). Each DataLoader fork-worker is a CoW copy of
  the ~6 G parent + decode/LRU, so RAM scales ~linearly; halving workers drops
  demand ~23 G → ~16–20 G — real headroom under the 25 G cap, no swap. The codec
  loop is decode-bound but the GPU is idle, so 2 workers still converge:
  **robustness > marginal throughput.** (Verified: the run now peaks ~17 G, no
  OOM, and the quality metric keeps climbing `[redacted: codec research]`.)
- `PREFETCH_PER_WORKER` 2.0→**4.0 GiB** — the honest measured per-worker envelope.
- `record_train_footprint` **GUARD:** discard any "measured" peak within 10 % of
  the cgroup cap. The cgroup `Memory peak:` is RAM-only and hard-capped, so a
  swapping run reports a peak ≈ cap that UNDER-states true demand; recording it
  taught a too-low envelope and **self-reinforced the tightness** (perpetual
  under-size / OOM-on-pressure). The bogus 24.8 G `footprints.json` sample was
  purged so the store re-learns from clean workers=2 runs.

**Invariant:** a cgroup OOM under memory pressure is NOT evidence of a code leak.
Diagnose with parent-RSS + cuda-mem + a dataloader-only probe BEFORE assuming a
leak; a "survives once on a clean box" run does not prove robustness under load —
re-verify under the real config and real box pressure.

### 2. Codec dataloader is decode-bound; cache the WINDOWS, not the signal
The codec's direct-from-archive dataset re-decodes the lossless recording
every epoch (GPU ~0 %). **Fix (blut `ebe852f`/`a50e0fd`):** the disk cache
stores only the **used training windows** keyed `(stem, win_idx)`, fp16 — tens of
GB for a full manifest / a few GB per A/B, vs the old whole-recording cache
(~600 GB, reclaimed). Epochs 2+ mmap the window, never re-decode. The cache stores
decoded **values** (the decode is the cost) — it is not a LUT.
`[redacted: codec research]` (window shape and corpus sizes).

### 3. Cache + memmap locations are standardized and un-misconfigurable
A cookbook cache-paths module `[redacted: private path]`: ONE knob (a data-root
environment variable, else a canonical default) → fixed, auto-created,
mutually-consistent per-layer subpaths; `apply_env()` force-overwrites the
per-layer env so they can never disagree or be unset; fail-closed if the default
root is missing; mandatory in the trainer (no skip flag). 4 contract tests.

### 4. Production handoff
A production roadmap `[redacted: private path]` enumerates every remaining
deliverable to a shippable product `[redacted: non-engine product detail]`
`[redacted: hardware]` `[redacted: codec research]`.

## Consequences
- Codec training is OOM-robust at workers=2 with cap headroom; the never-
  OOM-the-box guarantee (cgroup containment) held throughout this session.
- Default decode throughput halves (4→2 workers); acceptable because the loop is
  decode-bound-with-idle-GPU and a deeper worker pool just re-introduced the OOM.
  A larger box can opt back to `workers=4` per-recipe (footprint scales honestly).
- **Open (owner / next agent):** broker **slice-3** (OomCorrected retry-growth +
  VRAM ledger) is in-flight UNCOMMITTED in `src/broker/footprint.rs`; a
  warm-only vs joint recipe **output-contract** mismatch (false
  "failed"); a cookbook commit `[redacted: private repository]` is LOCAL
  (agent-push-blocked) and the owner must push it before the meta-repository
  pin is consistent for a fresh clone.

## Progress Log *(append-only; newest last)*

- _2026-08-27_ — gave this record a runnable citation gate `[redacted: private path]` (it proves the record still points at files that exist; it does not claim the decision is implemented) and removed its closure-debt entry `[redacted: private path]`.

## Completion / Resolution

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** The specific corrections this ADR makes are in the code at the values it prescribes. `UNCALIBRATED_WORKER_CAP` is 2 at `src/broker/footprint.rs:41` -- the 4 to 2 change the Decision argues for -- and it is read at the launch seam (`cli/gate.rs:225,316,319`), so the cap is enforced rather than merely declared. The record's own fix commits (blut `db39698` and a cookbook commit) are cited in the Decision itself; the tree still carries their result.
