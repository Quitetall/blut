// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Canonical execution lifecycle for local, P2P, and cloud stage attempts.
//!
//! Adapters own transport only. This module owns the request and outcome wire
//! shapes, absolute deadlines, assignment fencing, legal phase transitions,
//! first-terminal-wins semantics, cancellation, and final artifact restoration.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::framework::artifact::{ArtifactContentId, ContentHash, InvocationKey};
use crate::framework::artifact_store::{ArtifactRole, StoredArtifact, restore};
use crate::framework::error::StageError;
use crate::framework::error_domain::StageFailure;
use crate::framework::stage::{ErasedArtifact, StageDyn};

/// Version of the transport-neutral execution request/outcome contract.
pub const EXECUTION_PROTOCOL_VERSION: u16 = 2;

/// Default hard ceiling for a remote attempt whose stage and plan set no bound.
pub const DEFAULT_REMOTE_TIMEOUT: Duration = Duration::from_secs(60 * 60);

const CANCEL_ACK_TIMEOUT: Duration = Duration::from_millis(250);

/// Execution placement. The same lifecycle laws apply to every variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Local,
    P2p,
    Cloud,
}

/// Non-terminal lifecycle phase. Terminal disposition lives separately so it
/// cannot be confused with progress or overwritten by a late adapter update.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Preparing,
    UploadingInput,
    Queued,
    Assigned,
    Running,
    UploadingOutput,
    DownloadingOutput,
}

/// Ownership token for one assignment attempt. `generation` fences late results
/// after a lease is reclaimed and re-assigned, even to the same worker name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    pub owner: String,
    pub generation: u64,
}

impl Assignment {
    pub fn new(owner: impl Into<String>, generation: u64) -> Self {
        Self {
            owner: owner.into(),
            generation,
        }
    }
}

/// Absolute wall-clock deadlines carried across process and host boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionDeadline {
    pub soft_unix_ms: Option<u64>,
    pub hard_unix_ms: u64,
}

impl ExecutionDeadline {
    /// Build a deadline from relative budgets. The optional soft bound is
    /// clamped to the hard bound so no adapter observes an impossible ordering.
    /// A zero hard budget becomes one millisecond: callers get an immediately
    /// expiring, but still well-formed, absolute deadline.
    pub fn from_now(soft: Option<Duration>, hard: Duration) -> Self {
        let now = unix_ms();
        let hard_ms = duration_ms(hard.max(Duration::from_millis(1)));
        let hard_unix_ms = now.saturating_add(hard_ms);
        let soft_unix_ms =
            soft.map(|value| now.saturating_add(duration_ms(value)).min(hard_unix_ms));
        Self {
            soft_unix_ms,
            hard_unix_ms,
        }
    }

    /// Construct an already-absolute deadline, validating ordering.
    pub fn absolute(soft_unix_ms: Option<u64>, hard_unix_ms: u64) -> Result<Self, LifecycleError> {
        if soft_unix_ms.is_some_and(|soft| soft > hard_unix_ms) {
            return Err(LifecycleError::InvalidDeadline);
        }
        Ok(Self {
            soft_unix_ms,
            hard_unix_ms,
        })
    }

    pub fn hard_remaining(self) -> Duration {
        remaining(self.hard_unix_ms)
    }

    pub fn soft_remaining(self) -> Option<Duration> {
        self.soft_unix_ms.map(remaining)
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn remaining(deadline_unix_ms: u64) -> Duration {
    Duration::from_millis(deadline_unix_ms.saturating_sub(unix_ms()))
}

/// Resource requirements remain transport-neutral; P2P/cloud adapters translate
/// this value into their provider-specific record without widening the seam.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionResources {
    pub cpu_cores: u32,
    pub memory_gib: u32,
    pub gpu: bool,
    pub gpu_vram_gib: Option<u32>,
}

/// Transport-neutral data sensitivity. Adapters translate this closed domain to
/// their trust-policy type; invalid magic numbers cannot enter the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataClassification {
    Public,
    Internal,
    Restricted,
}

impl TryFrom<u8> for DataClassification {
    type Error = ExecutionFailure;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Public),
            1 => Ok(Self::Internal),
            2 => Ok(Self::Restricted),
            other => Err(ExecutionFailure::protocol(format!(
                "unknown data classification {other}"
            ))),
        }
    }
}

impl From<DataClassification> for u8 {
    fn from(value: DataClassification) -> Self {
        match value {
            DataClassification::Public => 0,
            DataClassification::Internal => 1,
            DataClassification::Restricted => 2,
        }
    }
}

/// Owned, serializable stage attempt. The input is A09's canonical portable
/// artifact, so an adapter never needs a producer-local source path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub protocol_version: u16,
    pub execution_id: String,
    /// Custody namespace. Adapters must enforce it together with data class.
    pub tenant: crate::tenant::Tenant,
    pub stage_name: String,
    pub stage_schema: u32,
    pub invocation_key: InvocationKey,
    pub args_hash: ContentHash,
    pub args: serde_json::Value,
    /// Portable input for adapters that cross a process or host boundary.
    /// Local placement keeps the already-admitted in-memory input in the
    /// executor and therefore submits `None`; P2P and cloud MUST reject a
    /// request without this value.
    pub input: Option<StoredArtifact>,
    pub expected_content_id: Option<ArtifactContentId>,
    pub resources: ExecutionResources,
    pub data_class: DataClassification,
    pub deadline: ExecutionDeadline,
}

/// Stable failure category used by retry, telemetry, and cross-host decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionFailureKind {
    Unavailable,
    Protocol,
    Transport,
    Disconnected,
    Artifact,
    Stage,
    Storage,
    Unknown,
}

/// Serializable failure identity. A cookbook [`StageFailure`] survives transport
/// intact instead of being flattened into a display string.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionFailure {
    pub kind: ExecutionFailureKind,
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub stage_failure: Option<StageFailure>,
}

impl ExecutionFailure {
    pub fn new(
        kind: ExecutionFailureKind,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            code: code.into(),
            message: message.into(),
            retryable: matches!(
                kind,
                ExecutionFailureKind::Unavailable
                    | ExecutionFailureKind::Transport
                    | ExecutionFailureKind::Disconnected
                    | ExecutionFailureKind::Storage
            ),
            stage_failure: None,
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(
            ExecutionFailureKind::Unavailable,
            "EXECUTION_UNAVAILABLE",
            message,
        )
    }

    pub fn transport(message: impl Into<String>) -> Self {
        Self::new(
            ExecutionFailureKind::Transport,
            "EXECUTION_TRANSPORT",
            message,
        )
    }

    pub fn disconnected(message: impl Into<String>) -> Self {
        Self::new(
            ExecutionFailureKind::Disconnected,
            "EXECUTION_DISCONNECTED",
            message,
        )
    }

    pub fn artifact(message: impl Into<String>) -> Self {
        Self::new(
            ExecutionFailureKind::Artifact,
            "EXECUTION_ARTIFACT",
            message,
        )
    }

    pub fn protocol(message: impl Into<String>) -> Self {
        Self::new(
            ExecutionFailureKind::Protocol,
            "EXECUTION_PROTOCOL",
            message,
        )
    }

    pub fn from_stage_error(stage_name: &str, error: &StageError) -> Self {
        if let StageError::Execution(failure) = error {
            return failure.as_ref().clone();
        }
        let stage_failure = match error {
            StageError::Backend(source) => StageFailure::try_extract(source).cloned(),
            _ => None,
        };
        Self {
            kind: ExecutionFailureKind::Stage,
            code: stage_failure
                .as_ref()
                .map(|failure| failure.code.clone())
                .unwrap_or_else(|| "EXECUTION_STAGE".into()),
            message: format!("stage '{stage_name}' failed: {error}"),
            retryable: crate::framework::retry::is_retryable(
                error,
                crate::framework::retry::RetryOn::Transient,
            ),
            stage_failure,
        }
    }

    /// Re-enter the local executor's existing typed retry/error path.
    pub fn into_stage_error(self) -> StageError {
        StageError::Execution(Box::new(self))
    }
}

impl std::fmt::Display for ExecutionFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for ExecutionFailure {}

/// Success payload shared by every mode. Remote adapters MUST populate `stored`;
/// local execution may use `None` only for an explicitly non-portable artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionArtifact {
    pub content_id: ArtifactContentId,
    pub stored: Option<StoredArtifact>,
}

/// Exactly one terminal disposition is latched for an execution attempt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExecutionTerminal {
    Succeeded {
        artifact: ExecutionArtifact,
        wall_time_ms: u64,
    },
    Failed {
        failure: ExecutionFailure,
    },
    Cancelled {
        reason: String,
    },
    TimedOut {
        phase: ExecutionPhase,
        deadline_unix_ms: u64,
    },
}

/// Serializable read model returned by all adapter handles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionSnapshot {
    pub mode: ExecutionMode,
    pub phase: ExecutionPhase,
    pub assignment: Option<Assignment>,
    pub terminal: Option<ExecutionTerminal>,
}

/// Shared state-machine implementation used by local and remote adapters.
#[derive(Clone)]
pub struct ExecutionLifecycle {
    inner: Arc<Mutex<LifecycleState>>,
}

struct LifecycleState {
    snapshot: ExecutionSnapshot,
    max_generation: u64,
}

impl ExecutionLifecycle {
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LifecycleState {
                snapshot: ExecutionSnapshot {
                    mode,
                    phase: ExecutionPhase::Preparing,
                    assignment: None,
                    terminal: None,
                },
                max_generation: 0,
            })),
        }
    }

    pub fn snapshot(&self) -> ExecutionSnapshot {
        self.inner.lock().snapshot.clone()
    }

    /// Move to a non-terminal phase. Re-assignment requires a strictly newer
    /// generation; queueing clears ownership so an expired lease cannot finish.
    pub fn transition(
        &self,
        next: ExecutionPhase,
        assignment: Option<Assignment>,
    ) -> Result<ExecutionSnapshot, LifecycleError> {
        let mut state = self.inner.lock();
        if state.snapshot.terminal.is_some() {
            return Err(LifecycleError::AlreadyTerminal);
        }
        if !legal_transition(state.snapshot.phase, next) {
            return Err(LifecycleError::IllegalTransition {
                from: state.snapshot.phase,
                to: next,
            });
        }

        let next_assignment = if next == ExecutionPhase::Queued {
            if assignment.is_some() {
                return Err(LifecycleError::UnexpectedAssignment(next));
            }
            None
        } else {
            assignment.or_else(|| state.snapshot.assignment.clone())
        };
        if phase_requires_assignment(next) && next_assignment.is_none() {
            return Err(LifecycleError::AssignmentRequired(next));
        }
        if let (Some(current), Some(candidate)) = (&state.snapshot.assignment, &next_assignment)
            && (candidate.generation < current.generation
                || (candidate.generation == current.generation && candidate != current))
        {
            return Err(LifecycleError::StaleAssignment);
        }
        if let Some(candidate) = &next_assignment {
            if candidate.generation < state.max_generation
                || (candidate.generation == state.max_generation
                    && state
                        .snapshot
                        .assignment
                        .as_ref()
                        .is_none_or(|current| current != candidate))
            {
                return Err(LifecycleError::StaleAssignment);
            }
            state.max_generation = state.max_generation.max(candidate.generation);
        }
        state.snapshot.phase = next;
        state.snapshot.assignment = next_assignment;
        Ok(state.snapshot.clone())
    }

    /// Latch the first terminal result. When `assignment` is supplied, it must
    /// exactly match the current owner+generation, fencing stale completions.
    pub fn finish(
        &self,
        assignment: Option<&Assignment>,
        terminal: ExecutionTerminal,
    ) -> Result<ExecutionSnapshot, LifecycleError> {
        let mut state = self.inner.lock();
        if state.snapshot.terminal.is_some() {
            return Err(LifecycleError::AlreadyTerminal);
        }
        if let Some(expected) = assignment
            && state.snapshot.assignment.as_ref() != Some(expected)
        {
            return Err(LifecycleError::StaleAssignment);
        }
        if let ExecutionTerminal::Succeeded { artifact, .. } = &terminal {
            // Defense in depth: lifecycle producers are checked here, and the
            // driver repeats the check after deserializing an adapter snapshot.
            if let Some(stored) = &artifact.stored
                && stored.manifest.content_id != artifact.content_id
            {
                return Err(LifecycleError::IdentityMismatch);
            }
        }
        state.snapshot.terminal = Some(terminal);
        Ok(state.snapshot.clone())
    }
}

fn phase_requires_assignment(phase: ExecutionPhase) -> bool {
    matches!(
        phase,
        ExecutionPhase::Assigned
            | ExecutionPhase::Running
            | ExecutionPhase::UploadingOutput
            | ExecutionPhase::DownloadingOutput
    )
}

fn legal_transition(from: ExecutionPhase, to: ExecutionPhase) -> bool {
    from == to
        || matches!(
            (from, to),
            (ExecutionPhase::Preparing, ExecutionPhase::UploadingInput)
                | (ExecutionPhase::Preparing, ExecutionPhase::Queued)
                | (ExecutionPhase::Preparing, ExecutionPhase::Assigned)
                | (ExecutionPhase::Preparing, ExecutionPhase::Running)
                | (ExecutionPhase::UploadingInput, ExecutionPhase::Queued)
                | (ExecutionPhase::UploadingInput, ExecutionPhase::Assigned)
                | (ExecutionPhase::UploadingInput, ExecutionPhase::Running)
                | (ExecutionPhase::Queued, ExecutionPhase::Assigned)
                | (ExecutionPhase::Assigned, ExecutionPhase::UploadingInput)
                | (ExecutionPhase::Assigned, ExecutionPhase::Running)
                | (ExecutionPhase::Running, ExecutionPhase::Queued)
                | (ExecutionPhase::Running, ExecutionPhase::UploadingOutput)
                | (ExecutionPhase::Running, ExecutionPhase::DownloadingOutput)
                | (
                    ExecutionPhase::UploadingOutput,
                    ExecutionPhase::DownloadingOutput
                )
        )
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LifecycleError {
    #[error("execution already has a terminal outcome")]
    AlreadyTerminal,
    #[error("illegal execution transition {from:?} -> {to:?}")]
    IllegalTransition {
        from: ExecutionPhase,
        to: ExecutionPhase,
    },
    #[error("phase {0:?} requires assignment ownership")]
    AssignmentRequired(ExecutionPhase),
    #[error("phase {0:?} must not carry assignment ownership")]
    UnexpectedAssignment(ExecutionPhase),
    #[error("completion or transition used a stale assignment")]
    StaleAssignment,
    #[error("success content identity does not match the stored artifact")]
    IdentityMismatch,
    #[error("soft execution deadline is later than hard deadline")]
    InvalidDeadline,
}

/// Transport adapter handle. Snapshot and cancellation are async so polling and
/// provider calls can be interrupted by the canonical driver's deadline.
#[async_trait]
pub trait ExecutionHandle: Send + Sync {
    async fn snapshot(&self) -> Result<ExecutionSnapshot, ExecutionFailure>;
    async fn cancel(&self) -> Result<(), ExecutionFailure>;
}

/// The only executor-facing seam for local/mesh/cloud placement variation.
///
/// `submit` is cancellation-safe: dropping its future before a handle is
/// returned MUST leave no queued, assigned, or durable work behind. Adapters
/// that perform I/O return a handle first and run submission behind that handle.
#[async_trait]
pub trait ExecutionAdapter: Send + Sync {
    fn mode(&self) -> ExecutionMode;
    async fn submit(
        &self,
        request: ExecutionRequest,
    ) -> Result<Box<dyn ExecutionHandle>, ExecutionFailure>;
}

/// Local placement adapter over the canonical execution lifecycle.
///
/// Stage work stays in the executor so fused typed handoff never serializes.
/// This adapter owns placement-specific submission, cancellation, assignment,
/// and terminal publication; the executor owns admitted stage invocation and
/// calls the terminal methods only after output identity is established.
pub struct LocalExecutionAdapter {
    lifecycle: ExecutionLifecycle,
    assignment: Assignment,
    cancellation: CancellationToken,
}

impl LocalExecutionAdapter {
    pub fn queued(
        generation: u64,
        cancellation: CancellationToken,
    ) -> Result<Self, LifecycleError> {
        let lifecycle = ExecutionLifecycle::new(ExecutionMode::Local);
        lifecycle.transition(ExecutionPhase::Queued, None)?;
        Ok(Self {
            lifecycle,
            assignment: Assignment::new("local", generation),
            cancellation,
        })
    }

    pub fn snapshot(&self) -> ExecutionSnapshot {
        self.lifecycle.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn lifecycle(&self) -> ExecutionLifecycle {
        self.lifecycle.clone()
    }

    pub fn succeed(
        &self,
        content_id: ArtifactContentId,
        stored: Option<StoredArtifact>,
        wall_time: Duration,
    ) -> Result<ExecutionSnapshot, LifecycleError> {
        self.lifecycle.finish(
            Some(&self.assignment),
            ExecutionTerminal::Succeeded {
                artifact: ExecutionArtifact { content_id, stored },
                wall_time_ms: duration_ms(wall_time),
            },
        )
    }

    pub fn fail(
        &self,
        stage_name: &str,
        error: &StageError,
    ) -> Result<ExecutionSnapshot, LifecycleError> {
        let terminal = if matches!(error, StageError::Cancelled) {
            ExecutionTerminal::Cancelled {
                reason: "local stage cancelled".into(),
            }
        } else {
            ExecutionTerminal::Failed {
                failure: ExecutionFailure::from_stage_error(stage_name, error),
            }
        };
        self.lifecycle.finish(Some(&self.assignment), terminal)
    }

    pub fn timeout(
        &self,
        phase: ExecutionPhase,
        deadline_unix_ms: u64,
    ) -> Result<ExecutionSnapshot, LifecycleError> {
        self.lifecycle.finish(
            Some(&self.assignment),
            ExecutionTerminal::TimedOut {
                phase,
                deadline_unix_ms,
            },
        )
    }
}

impl Drop for LocalExecutionAdapter {
    fn drop(&mut self) {
        if self.lifecycle.snapshot().terminal.is_none() {
            self.cancellation.cancel();
            let _ = self.lifecycle.finish(
                None,
                ExecutionTerminal::Failed {
                    failure: ExecutionFailure::new(
                        ExecutionFailureKind::Unknown,
                        "EXECUTION_ABANDONED",
                        "local attempt exited before recording a terminal outcome",
                    ),
                },
            );
        }
    }
}

#[async_trait]
impl ExecutionAdapter for LocalExecutionAdapter {
    fn mode(&self) -> ExecutionMode {
        ExecutionMode::Local
    }

    async fn submit(
        &self,
        request: ExecutionRequest,
    ) -> Result<Box<dyn ExecutionHandle>, ExecutionFailure> {
        if request.protocol_version != EXECUTION_PROTOCOL_VERSION {
            return Err(ExecutionFailure::protocol(format!(
                "execution protocol {} unsupported (want {})",
                request.protocol_version, EXECUTION_PROTOCOL_VERSION
            )));
        }
        if request.input.is_some() {
            return Err(ExecutionFailure::protocol(
                "local execution request must retain input in the executor",
            ));
        }
        self.lifecycle
            .transition(ExecutionPhase::Assigned, Some(self.assignment.clone()))
            .and_then(|_| self.lifecycle.transition(ExecutionPhase::Running, None))
            .map_err(|error| ExecutionFailure::protocol(error.to_string()))?;
        Ok(Box::new(LocalExecutionHandle {
            lifecycle: self.lifecycle.clone(),
            cancellation: self.cancellation.clone(),
        }))
    }
}

struct LocalExecutionHandle {
    lifecycle: ExecutionLifecycle,
    cancellation: CancellationToken,
}

#[async_trait]
impl ExecutionHandle for LocalExecutionHandle {
    async fn snapshot(&self) -> Result<ExecutionSnapshot, ExecutionFailure> {
        Ok(self.lifecycle.snapshot())
    }

    async fn cancel(&self) -> Result<(), ExecutionFailure> {
        let _ = self.lifecycle.finish(
            None,
            ExecutionTerminal::Cancelled {
                reason: "local execution cancelled".into(),
            },
        );
        self.cancellation.cancel();
        Ok(())
    }
}

/// Validated completion returned to the executor after success has been restored
/// beneath a consumer-owned directory.
pub enum ExecutionResult {
    Succeeded {
        artifact: ErasedArtifact,
        content_id: ArtifactContentId,
        wall_time_ms: u64,
    },
    Failed(ExecutionFailure),
    Cancelled,
    TimedOut {
        phase: ExecutionPhase,
        deadline_unix_ms: u64,
    },
}

/// Submit, poll, cancel, enforce deadlines, and validate/rehydrate success. A
/// hanging submit, poll, upload, or download cannot outlive the hard deadline.
pub async fn drive_execution(
    adapter: &dyn ExecutionAdapter,
    request: ExecutionRequest,
    cancel: &CancellationToken,
    stage: Arc<dyn StageDyn>,
    output_dir: &Path,
    poll_interval: Duration,
) -> ExecutionResult {
    let deadline = request.deadline;
    let expected_content_id = request.expected_content_id;
    let mut last_phase = ExecutionPhase::Preparing;

    let submit = adapter.submit(request);
    tokio::pin!(submit);
    let handle = tokio::select! {
        result = &mut submit => match result {
            Ok(handle) => handle,
            Err(failure) => return ExecutionResult::Failed(failure),
        },
        _ = cancel.cancelled() => return ExecutionResult::Cancelled,
        _ = sleep_optional(deadline.soft_remaining()) => {
            return ExecutionResult::TimedOut {
                phase: last_phase,
                deadline_unix_ms: deadline.soft_unix_ms.unwrap_or(deadline.hard_unix_ms),
            };
        }
        _ = tokio::time::sleep(deadline.hard_remaining()) => {
            return ExecutionResult::TimedOut {
                phase: last_phase,
                deadline_unix_ms: deadline.hard_unix_ms,
            };
        }
    };

    loop {
        let snapshot = handle.snapshot();
        tokio::pin!(snapshot);
        let state = tokio::select! {
            result = &mut snapshot => match result {
                Ok(state) => state,
                Err(failure) => return ExecutionResult::Failed(failure),
            },
            _ = cancel.cancelled() => {
                request_cancel(handle.as_ref()).await;
                return ExecutionResult::Cancelled;
            }
            _ = sleep_optional(deadline.soft_remaining()) => {
                request_cancel(handle.as_ref()).await;
                return ExecutionResult::TimedOut {
                    phase: last_phase,
                    deadline_unix_ms: deadline.soft_unix_ms.unwrap_or(deadline.hard_unix_ms),
                };
            }
            _ = tokio::time::sleep(deadline.hard_remaining()) => {
                request_cancel(handle.as_ref()).await;
                return ExecutionResult::TimedOut {
                    phase: last_phase,
                    deadline_unix_ms: deadline.hard_unix_ms,
                };
            }
        };
        last_phase = state.phase;
        if let Some(terminal) = state.terminal {
            return validate_terminal(
                terminal,
                expected_content_id,
                stage,
                output_dir.to_path_buf(),
                deadline,
            )
            .await;
        }

        tokio::select! {
            _ = tokio::time::sleep(poll_interval) => {}
            _ = cancel.cancelled() => {
                request_cancel(handle.as_ref()).await;
                return ExecutionResult::Cancelled;
            }
            _ = sleep_optional(deadline.soft_remaining()) => {
                request_cancel(handle.as_ref()).await;
                return ExecutionResult::TimedOut {
                    phase: last_phase,
                    deadline_unix_ms: deadline.soft_unix_ms.unwrap_or(deadline.hard_unix_ms),
                };
            }
            _ = tokio::time::sleep(deadline.hard_remaining()) => {
                request_cancel(handle.as_ref()).await;
                return ExecutionResult::TimedOut {
                    phase: last_phase,
                    deadline_unix_ms: deadline.hard_unix_ms,
                };
            }
        }
    }
}

async fn request_cancel(handle: &dyn ExecutionHandle) {
    let _ = tokio::time::timeout(CANCEL_ACK_TIMEOUT, handle.cancel()).await;
}

async fn sleep_optional(duration: Option<Duration>) {
    match duration {
        Some(duration) => tokio::time::sleep(duration).await,
        None => std::future::pending::<()>().await,
    }
}

async fn validate_terminal(
    terminal: ExecutionTerminal,
    expected_content_id: Option<ArtifactContentId>,
    stage: Arc<dyn StageDyn>,
    output_dir: std::path::PathBuf,
    deadline: ExecutionDeadline,
) -> ExecutionResult {
    match terminal {
        ExecutionTerminal::Succeeded {
            artifact,
            wall_time_ms,
        } => {
            let Some(stored) = artifact.stored else {
                return ExecutionResult::Failed(ExecutionFailure::artifact(
                    "remote adapter reported success without a portable artifact",
                ));
            };
            if stored.manifest.content_id != artifact.content_id {
                return ExecutionResult::Failed(ExecutionFailure::artifact(format!(
                    "terminal identity {} != stored artifact {}",
                    artifact.content_id, stored.manifest.content_id
                )));
            }
            if let Some(expected) = expected_content_id
                && expected != artifact.content_id
            {
                return ExecutionResult::Failed(ExecutionFailure::artifact(format!(
                    "output identity {} != analytically expected {}",
                    artifact.content_id, expected
                )));
            }
            let content_id = artifact.content_id;
            let quarantine_root = output_dir
                .parent()
                .unwrap_or(&output_dir)
                .join(format!(".execution-restore-{}", uuid::Uuid::new_v4()));
            let quarantine_import = quarantine_root
                .join(".artifact-import")
                .join(content_id.to_hex());
            let restore_root = quarantine_root.clone();
            let mut restore_task = tokio::task::spawn_blocking(move || {
                let quarantine = RestoreQuarantine::new(restore_root, quarantine_import);
                let output = restore(
                    stage.as_ref(),
                    &stored,
                    &quarantine.root,
                    ArtifactRole::Output,
                    Some(content_id),
                )?;
                Ok::<_, crate::framework::artifact_store::ArtifactStoreError>((
                    stage, quarantine, output,
                ))
            });
            let (restore_budget, restore_deadline) = match deadline.soft_remaining() {
                Some(soft) => (soft, deadline.soft_unix_ms.expect("matched Some")),
                None => (deadline.hard_remaining(), deadline.hard_unix_ms),
            };
            let (stage, quarantine, restored) =
                match tokio::time::timeout(restore_budget, &mut restore_task).await {
                    Ok(Ok(Ok(restored))) => restored,
                    Ok(Ok(Err(error))) => {
                        return ExecutionResult::Failed(ExecutionFailure::artifact(format!(
                            "restore remote output: {error}"
                        )));
                    }
                    Ok(Err(error)) => {
                        return ExecutionResult::Failed(ExecutionFailure::artifact(format!(
                            "restore task failed: {error}"
                        )));
                    }
                    Err(_) => {
                        // `spawn_blocking` cannot interrupt a syscall already in
                        // progress, but the task writes only to its quarantine. If it
                        // eventually returns, its detached result is dropped and the
                        // quarantine guard cleans up without racing attempt cleanup.
                        restore_task.abort();
                        return ExecutionResult::TimedOut {
                            phase: ExecutionPhase::DownloadingOutput,
                            deadline_unix_ms: restore_deadline,
                        };
                    }
                };
            let target_import = output_dir
                .join(".artifact-import")
                .join(content_id.to_hex());
            if let Some(parent) = target_import.parent()
                && let Err(error) = std::fs::create_dir_all(parent)
            {
                return ExecutionResult::Failed(ExecutionFailure::artifact(format!(
                    "create restored output parent: {error}"
                )));
            }
            let _ = std::fs::remove_dir_all(&target_import);
            if let Err(error) = std::fs::rename(&quarantine.import_root, &target_import) {
                return ExecutionResult::Failed(ExecutionFailure::artifact(format!(
                    "publish restored output: {error}"
                )));
            }
            let Some(restored) = stage.rebase_output_paths_checked(
                restored,
                &quarantine.import_root,
                &target_import,
            ) else {
                let _ = std::fs::remove_dir_all(&target_import);
                return ExecutionResult::Failed(ExecutionFailure::artifact(
                    "rebase published remote output",
                ));
            };
            ExecutionResult::Succeeded {
                artifact: restored,
                content_id,
                wall_time_ms,
            }
        }
        ExecutionTerminal::Failed { failure } => ExecutionResult::Failed(failure),
        ExecutionTerminal::Cancelled { .. } => ExecutionResult::Cancelled,
        ExecutionTerminal::TimedOut {
            phase,
            deadline_unix_ms,
        } => ExecutionResult::TimedOut {
            phase,
            deadline_unix_ms,
        },
    }
}

/// Restore output is quarantined beside (not inside) the caller's attempt
/// directory. If a timed-out blocking task eventually returns, dropping its
/// detached result removes this directory without racing attempt cleanup.
struct RestoreQuarantine {
    root: std::path::PathBuf,
    import_root: std::path::PathBuf,
}

impl RestoreQuarantine {
    fn new(root: std::path::PathBuf, import_root: std::path::PathBuf) -> Self {
        Self { root, import_root }
    }
}

impl Drop for RestoreQuarantine {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
