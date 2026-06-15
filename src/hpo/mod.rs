//! BLUT v0.20 — native hyperparameter optimization (HPO).
//!
//! Trials are parallel nodes in ONE plan, reusing the existing machinery: the
//! never-OOM broker + GPU semaphore gate concurrency, a trial's overlaid args
//! give it a distinct cache key + durable-resume dir for free, and the
//! `ControlPolicy`/`KillBranch` runtime-control path drives early-stop. See the
//! v0.20 plan for the full architecture + phasing.
//!
//! Slices land additively:
//! - **Phase 1 (this slice):** the search space ([`space`]) + samplers
//!   ([`sampler`]).
//! - Later: the fan-out plan builder, the scheduler `ControlPolicy`
//!   (median/ASHA via the existing `KillBranch`), `Control::Spawn` (PBT/TPE),
//!   and trial tracking.

pub mod plan_build;
pub mod sampler;
pub mod space;

pub use plan_build::{TrialPlan, build_hpo_plan};
pub use sampler::{RandomSampler, Sampler};
pub use space::{Dist, Overlay, SearchSpace, TrialResult, apply_overlay};
