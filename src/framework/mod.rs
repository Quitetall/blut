// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT framework core — typed Stages, Plans, Recipes.
//!
//! BLUT (Brian Lam's Universal Trainer) is built around three layers:
//!
//!   - **Artifacts** — typed in-memory handles to on-disk bytes,
//!     content-addressed by a deterministic hash. The boundary
//!     between stages.
//!   - **Stages** — typed atoms with `Input → Output → Error`, each
//!     declaring the resources it holds while running. The unit of
//!     work the executor schedules.
//!   - **Plans** — typed DAGs of stages built via a `Plan<Out>`
//!     PhantomData witness so wrong wiring fails at `cargo build`.
//!     Recipes compile typed args into Plans.
//!
//! Commit 1 (this one): just `Artifact` + `ContentHash` + sidecar
//! metadata. Stages, Plans, executor, recipes land in subsequent
//! commits per the approved plan in
//! `~/.claude/plans/unified-launching-quill.md`.
//!
//! Why this is here: the existing crate ships a working SFT runner
//! against the legacy `TrainSpec` linear flow. The framework module
//! is pure-additive — nothing in the legacy flow uses it yet — so
//! commits 1-3 carry no behavioural-change risk. Commit 4 ports the
//! pipeline to the framework with `LAMU_TRAIN_USE_LEGACY=1` as the
//! kill-switch; commit 8 deletes the legacy path.

pub mod artifact;
pub mod async_io;
pub mod cache;
pub mod compat;
pub mod control;
pub mod cookbook;
pub mod dag_opt;
pub mod error;
pub mod error_domain;
pub mod executor;
pub mod gpu_sampler;
pub mod graph;
pub mod lineage;
pub mod object_store;
pub mod plan;
pub mod plan_spec;
pub mod resource;
pub mod resume;
pub mod retry;
/// ADR 0139 semantic-graph adapter. Behind the OFF-BY-DEFAULT `semantic-graph`
/// feature: `blut-graph-core`'s node/port model carries ABIR's semantic type
/// taxonomy (`AbirRootType`/`AbirViewType`), which is DOMAIN vocabulary. The
/// engine stays domain-agnostic (ADR 0034), so a cookbook that wants ABIR
/// graphs opts in; the default engine never links it.
#[cfg(feature = "semantic-graph")]
pub mod semantic_plan;
pub mod stage;
pub mod status;

pub use artifact::{Artifact, ArtifactMetadata, BranchDecision, ContentHash, ListOf};
pub use async_io::{
    IoMode, TrainingIoAdmissionError, TrainingIoCandidate, TrainingIoDowngradeReason,
    TrainingIoHints, TrainingIoProfile, profile_is_declared, select_training_io_profile,
};
#[cfg(feature = "semantic-graph")]
pub use blut_graph_core as semantic_graph;
pub use cache::{CacheHandle, CacheHit, lru_prune};
pub use compat::Compatible;
pub use cookbook::{
    ArtifactDescriptor, Cookbook, CookbookTui, Ingredient, Registry, StageDescriptor,
};
pub use dag_opt::{DagOptimizer, ScheduleHint};
pub use error::{PlanError, RecipeError, StageError};
pub use error_domain::{
    ErrorDomain, ErrorDomainDef, FailureSummary, FaultOrigin, Severity, StageFailure,
    extract_failure_summary,
};
pub use executor::{ExecCtx, ParallelExecutor, PlanResult, SequentialExecutor, execute_plan};
pub use graph::{GraphSnapshot, NodeStatus, PlanGraph, graph_snapshot};
pub use object_store::{BlobStore, FsBlobStore};
pub use plan::{CompiledPlan, NodeId, Plan};
pub use plan_spec::{ConditionGateSpec, PLAN_SPEC_VERSION, PlanSpec, PlanSpecError, SpecNode};
pub use resource::Resource;
pub use retry::{Backoff, RetryEvent, RetryHook, RetryOn, RetryPolicy, StageTimeout};
#[cfg(feature = "semantic-graph")]
pub use semantic_plan::{
    DurableAdaptedPlan, DurablePlanError, DurableStepContract, DurableStepResolver,
    ResolvedDurableStep, adapt_durable_plan,
};
pub use stage::{
    ErasedArtifact, ErasedDecodeError, PipelineManifest, Stage, StageContext, StageDyn,
    StageExecutionBoundary,
};
pub use status::{HostedEvent, StageEvent, StatusHub, make_broadcast, spawn_status_writer};
