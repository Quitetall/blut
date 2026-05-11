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
pub mod cache;
pub mod error;
pub mod executor;
pub mod plan;
pub mod resource;
pub mod stage;
pub mod status;

pub use artifact::{Artifact, ArtifactMetadata, ContentHash};
pub use cache::{lru_prune, CacheHandle, CacheHit};
pub use error::{PlanError, RecipeError, StageError};
pub use executor::{ExecCtx, PlanResult, SequentialExecutor};
pub use plan::{NodeId, Plan};
pub use resource::Resource;
pub use stage::{ErasedArtifact, ErasedDecodeError, Stage, StageContext, StageDyn};
pub use status::{make_broadcast, spawn_status_writer, StageEvent};
