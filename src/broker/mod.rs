// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT resource broker — SLICE-1 ("box never RAM-OOMs").
//!
//! Per ADR 0046 (the adversarial review's *corrected* smallest-first
//! shippable slice — containment-first), this module ships ONLY the
//! pieces that give the hard floor:
//!
//!   1. [`footprint`] — the calibrated footprint store plus a legacy
//!      LamQuant-specific scaling model. Generic launch admission uses typed
//!      stage resource declarations; it never interprets recipe JSON fields.
//!      The compatibility model remains available to `blut-lamquant` for its
//!      train-stage cgroup `MemoryMax` cap and measured calibration keys.
//!   2. [`probe`] — a cheap, best-effort `ResourceSnapshot` of free
//!      RAM (`/proc/meminfo` MemAvailable) + free VRAM (`nvidia-smi`).
//!      Independent of the TUI's `SystemSnapshot` so the broker has no
//!      coupling to the presentation layer.
//!   3. [`admission`] — a PURE `decide()` fn: fits → Ok, footprint >
//!      free RAM → Refuse, footprint > box capacity → Refuse. NO
//!      poll-queue: the review verified `scheduler_lock` already
//!      serializes blut-vs-blut GPU jobs (fail-fast), so admission is
//!      a single-job over-subscription guard placed BEFORE the lock.
//!
//! SLICE-2 adds the [`footprint::FootprintStore`] calibration store:
//! measured cgroup-attributed peak RAM (recorded at job exit by the
//! cookbook runner) MAX-merged per `(recipe,tier,batch,workers)` key. This is a
//! transitional cookbook compatibility surface, not a generic engine contract.
//!
//! DEFERRED to later slices (NOT built here): VRAM byte-ledger /
//! per-GPU iteration, OOM-detect+retry, auto-tune-up. The store carries
//! a `vram_mib` field but `resolve()` keeps the conservative VRAM
//! estimate (RAM is the over-refuse constraint).
//!
//! The HARD floor — "the box never goes down" — is the cgroup
//! `MemoryMax` containment applied on the train path (see
//! `blut-lamquant`'s `LamquantInvocation.contained` /
//! `cgroup_memmax`). Admission only reduces *job-level* OOM-kills and
//! only against blut-launched load. Containment is enforced in the cookbook's
//! runner.

pub mod admission;
pub mod footprint;
pub mod gpu;
pub mod probe;
pub mod tenant_quota;

pub use admission::{AdmitDecision, decide, gate};
pub use footprint::{
    Footprint, FootprintEntry, FootprintKey, FootprintSource, FootprintStore, GIB,
    UNCALIBRATED_WORKER_CAP, footprint_key,
};
pub use probe::ResourceSnapshot;
