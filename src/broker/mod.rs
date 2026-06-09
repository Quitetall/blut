//! BLUT resource broker — SLICE-1 ("box never RAM-OOMs").
//!
//! Per ADR 0046 (the adversarial review's *corrected* smallest-first
//! shippable slice — containment-first), this module ships ONLY the
//! pieces that give the hard floor:
//!
//!   1. [`footprint`] — a SCALING RAM-footprint estimator (not a
//!      constant): conservative-high RAM bytes as a function of the
//!      cost drivers (dataloader `workers × prefetch` dominant, then
//!      batch, tier, latent_dim). Used BOTH by the launch-path
//!      admission gate (blut) and by the train stage (blut-lamquant,
//!      which also feeds it into the cgroup `MemoryMax` cap).
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
//! DEFERRED to later slices (NOT built here): calibration store, VRAM
//! byte-ledger / per-GPU iteration, OOM-detect+retry, auto-tune-up.
//!
//! The HARD floor — "the box never goes down" — is the cgroup
//! `MemoryMax` containment applied on the train path (see
//! `blut-lamquant`'s `LamquantInvocation.contained` /
//! `cgroup_memmax`). Admission only reduces *job-level* OOM-kills and
//! only against blut-launched load. This module is the admission half;
//! containment is enforced in the cookbook's runner.

pub mod admission;
pub mod footprint;
pub mod probe;

pub use admission::{AdmitDecision, decide};
pub use footprint::{Footprint, GIB, estimate_ram_bytes};
pub use probe::ResourceSnapshot;
