// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT Kubernetes operator (ADR 0067 T4.5).
//!
//! K8s is an ADAPTER, not the control plane (ADR 0037): the operator maintains
//! `BlutWorkerPool` StatefulSets and runs `BlutPlan`s as Jobs, but the mesh
//! (ADR 0079) is the runtime and the engine's broker/cache/dispatch stay
//! authoritative. All kube-rs lives in THIS crate; the engine (`blut`) has zero
//! kube dependency — it shells out via `kubectl` (`LaunchTarget::K8s`) or is
//! called in-process here.

pub mod crds;
pub mod reconcile;

pub use crds::{
    BlutPlan, BlutPlanSpec, BlutPlanStatus, BlutWorkerPool, BlutWorkerPoolSpec,
    BlutWorkerPoolStatus, CacheConfig, DataClassCeiling, PlanPhase, PlanSource, PoolResources,
};
