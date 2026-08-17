// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT v0.20 — native hyperparameter optimization (HPO).
//!
//! Trials are parallel nodes in ONE plan, reusing the existing machinery: the
//! memory-admission broker + GPU semaphore gate concurrency, a trial's overlaid args
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
pub mod gp;
pub mod median;
pub mod pareto;
pub mod pbt;
pub mod plan_build;
pub mod results;
pub mod sampler;
pub mod scheduler;
pub mod space;
pub mod study;
pub mod surrogate;
pub mod tpe;

pub use asha::AshaStop;
pub use gp::{GpConfig, GpModel, GpSampler, ehvi_mc, expected_improvement};
pub use median::MedianStop;
pub use pareto::{
    Direction, ParetoPoint, ParetoReport, dominates, hypervolume_2d, non_dominated, pareto_front,
};
pub use pbt::{PbtConfig, PbtPolicy, PbtResume, PbtTrial, TrialFactory};
pub use plan_build::{TrialPlan, build_hpo_plan};
pub use results::{HpoManifest, TrialOutcome, TrialRec, leaderboard};
pub use sampler::{RandomSampler, Sampler};
pub use scheduler::{EarlyStop, HpoScheduler, build_trial_of_topo};
pub use space::{Dist, Overlay, SearchSpace, TrialResult, apply_overlay};
pub use study::{Objective, Proposal, StudyLedger, StudySpec, StudyTrial, TrialStatus, config_key};
pub use surrogate::{
    CostModel, CostObservation, GridSampler, MvTpeConfig, MvTpeSampler, acq_per_cost,
};
pub use tpe::{FreshFactory, TpeConfig, TpePolicy, TpePolicyConfig, TpeSampler};
