//! Cloud job + result records (ADR 0067 · T3.1b).
//!
//! A `CloudJob` is the cloud analog of a `p2p::TaskManifest`: everything a worker
//! needs to execute one stage on shipped data. The bundle PACK (the bytes) lives in
//! the object store keyed by `input_blob_key`; only the small `input_manifest`
//! (per-file table + erased handle) rides the job record. `CloudResult` mirrors it
//! for the output, plus the `wall_time_ms` the billing ledger (T3.1e) consumes.

use serde::{Deserialize, Serialize};

use crate::framework::artifact::{ArtifactContentId, ContentHash, InvocationKey};
use crate::framework::execution::{
    Assignment, ExecutionDeadline, ExecutionFailure, ExecutionPhase,
};
use crate::p2p::bundle::BundleManifest;
use crate::p2p::task::ResourceRequest;
use crate::p2p::trust::DataClass;

pub const CLOUD_JOB_PROTOCOL_VERSION: u16 = 3;

fn legacy_protocol_version() -> u16 {
    1
}

/// One unit of dispatchable work: a stage + its bundled input + the expected output
/// address (content-addressed, verified fail-closed on the worker).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudJob {
    #[serde(default = "legacy_protocol_version")]
    pub protocol_version: u16,
    /// Unique, traversal-safe id (the queue key).
    pub id: String,
    /// Custody namespace; checked with `data_class` before off-box execution.
    #[serde(default)]
    pub tenant: crate::tenant::Tenant,
    pub stage_name: String,
    pub stage_schema: u32,
    pub invocation_key: InvocationKey,
    /// Canonical hash of the serialized stage arguments.
    pub args_hash: ContentHash,
    pub args: serde_json::Value,
    /// Object-store key of the input bundle pack (the pack bytes' ContentHash).
    pub input_blob_key: ContentHash,
    /// The small bundle manifest (file table + erased handle) — rides the record.
    pub input_manifest: BundleManifest,
    /// Analytically known output identity, when the stage can provide one before
    /// execution. Otherwise the worker derives and returns the actual identity.
    #[serde(rename = "expected_output_hash")]
    pub expected_content_id: Option<ArtifactContentId>,
    pub resources: ResourceRequest,
    /// Data sensitivity — the reused `DispatchMatrix` refuses `Restricted` to a
    /// cloud worker below `Trusted` (clinical EEG hard-block in v1).
    pub data_class: DataClass,
    /// Higher runs sooner; jobs are claimed in priority-descending order.
    pub priority: i32,
    /// Hard wall-clock budget for the worker's stage run.
    pub timeout_secs: u64,
    /// Absolute end-to-end deadline. Queueing and transfer consume this budget;
    /// workers never rebuild a fresh relative timeout after claiming.
    pub deadline: ExecutionDeadline,
}

/// Terminal disposition of a job (mirrors p2p's verdict + the retired blut-worker prototype's JobStatus).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobOutcome {
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
}

/// The worker's reply: where the output bundle landed + how long it took.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudResult {
    #[serde(default = "legacy_protocol_version")]
    pub protocol_version: u16,
    pub job_id: String,
    /// Exact lease that produced this result. Pending cancellation is terminal
    /// without an assignment; every worker-produced result carries one.
    pub assignment: Option<Assignment>,
    pub outcome: JobOutcome,
    /// Object-store key of the output bundle pack (None on failure).
    pub output_blob_key: Option<ContentHash>,
    /// The output bundle manifest (None on failure).
    pub output_manifest: Option<BundleManifest>,
    /// The output artifact's content identity (None on failure).
    #[serde(rename = "output_hash")]
    pub content_id: Option<ArtifactContentId>,
    /// Wall-clock the stage ran on the worker — the billing signal (T3.1e).
    pub wall_time_ms: u64,
    /// Typed failure identity when `outcome == Failed`.
    pub failure: Option<ExecutionFailure>,
    /// Timeout attribution when `outcome == TimedOut`.
    pub timeout_phase: Option<ExecutionPhase>,
    pub deadline_unix_ms: Option<u64>,
}

impl CloudResult {
    /// Construct a failure result for `job_id` with `wall_time_ms` already spent.
    pub fn failed(job_id: impl Into<String>, wall_time_ms: u64, failure: ExecutionFailure) -> Self {
        Self {
            protocol_version: CLOUD_JOB_PROTOCOL_VERSION,
            job_id: job_id.into(),
            assignment: None,
            outcome: JobOutcome::Failed,
            output_blob_key: None,
            output_manifest: None,
            content_id: None,
            wall_time_ms,
            failure: Some(failure),
            timeout_phase: None,
            deadline_unix_ms: None,
        }
    }

    pub fn cancelled(job_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            protocol_version: CLOUD_JOB_PROTOCOL_VERSION,
            job_id: job_id.into(),
            assignment: None,
            outcome: JobOutcome::Cancelled,
            output_blob_key: None,
            output_manifest: None,
            content_id: None,
            wall_time_ms: 0,
            failure: Some(ExecutionFailure::new(
                crate::framework::execution::ExecutionFailureKind::Stage,
                "EXECUTION_CANCELLED",
                reason,
            )),
            timeout_phase: None,
            deadline_unix_ms: None,
        }
    }

    pub fn timed_out(
        job_id: impl Into<String>,
        wall_time_ms: u64,
        timeout_phase: ExecutionPhase,
        deadline_unix_ms: u64,
    ) -> Self {
        Self {
            protocol_version: CLOUD_JOB_PROTOCOL_VERSION,
            job_id: job_id.into(),
            assignment: None,
            outcome: JobOutcome::TimedOut,
            output_blob_key: None,
            output_manifest: None,
            content_id: None,
            wall_time_ms,
            failure: None,
            timeout_phase: Some(timeout_phase),
            deadline_unix_ms: Some(deadline_unix_ms),
        }
    }
}
