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

pub mod asha;
pub mod median;
pub mod pbt;
pub mod plan_build;
pub mod results;
pub mod sampler;
pub mod scheduler;
pub mod space;
pub mod tpe;

pub use asha::AshaStop;
pub use median::MedianStop;
pub use pbt::{PbtConfig, PbtPolicy, PbtResume, PbtTrial, TrialFactory};
pub use plan_build::{TrialPlan, build_hpo_plan};
pub use results::{HpoManifest, TrialOutcome, TrialRec, leaderboard};
pub use sampler::{RandomSampler, Sampler};
pub use scheduler::{EarlyStop, HpoScheduler, build_trial_of_topo};
pub use space::{Dist, Overlay, SearchSpace, TrialResult, apply_overlay};
pub use tpe::{FreshFactory, TpeConfig, TpePolicy, TpePolicyConfig, TpeSampler};
