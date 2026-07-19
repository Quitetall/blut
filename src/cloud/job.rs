//! Cloud job + result records (ADR 0067 · T3.1b).
//!
//! A `CloudJob` is the cloud analog of a `p2p::TaskManifest`: everything a worker
//! needs to execute one stage on shipped data. The bundle PACK (the bytes) lives in
//! the object store keyed by `input_blob_key`; only the small `input_manifest`
//! (per-file table + erased handle) rides the job record. `CloudResult` mirrors it
//! for the output, plus the `wall_time_ms` the billing ledger (T3.1e) consumes.

use serde::{Deserialize, Serialize};

use crate::framework::artifact::ContentHash;
use crate::p2p::bundle::BundleManifest;
use crate::p2p::task::ResourceRequest;
use crate::p2p::trust::DataClass;

/// One unit of dispatchable work: a stage + its bundled input + the expected output
/// address (content-addressed, verified fail-closed on the worker).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudJob {
    /// Unique, traversal-safe id (the queue key).
    pub id: String,
    pub stage_name: String,
    pub stage_schema: u32,
    pub args: serde_json::Value,
    /// Object-store key of the input bundle pack (the pack bytes' ContentHash).
    pub input_blob_key: ContentHash,
    /// The small bundle manifest (file table + erased handle) — rides the record.
    pub input_manifest: BundleManifest,
    /// The address the worker's output MUST reproduce (the 4 bundle gates enforce).
    pub expected_output_hash: ContentHash,
    pub resources: ResourceRequest,
    /// Data sensitivity — the reused `DispatchMatrix` refuses `Restricted` to a
    /// cloud worker below `Trusted` (clinical EEG hard-block in v1).
    pub data_class: DataClass,
    /// Higher runs sooner; jobs are claimed in priority-descending order.
    pub priority: i32,
    /// Hard wall-clock budget for the worker's stage run.
    pub timeout_secs: u64,
}

/// Terminal disposition of a job (mirrors p2p's verdict + the retired blut-worker prototype's JobStatus).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobOutcome {
    Succeeded,
    Failed,
    Cancelled,
}

/// The worker's reply: where the output bundle landed + how long it took.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloudResult {
    pub job_id: String,
    pub outcome: JobOutcome,
    /// Object-store key of the output bundle pack (None on failure).
    pub output_blob_key: Option<ContentHash>,
    /// The output bundle manifest (None on failure).
    pub output_manifest: Option<BundleManifest>,
    /// The output artifact's content address (None on failure).
    pub output_hash: Option<ContentHash>,
    /// Wall-clock the stage ran on the worker — the billing signal (T3.1e).
    pub wall_time_ms: u64,
    /// Failure detail, if `outcome != Succeeded`.
    pub error: Option<String>,
}

impl CloudResult {
    /// Construct a failure result for `job_id` with `wall_time_ms` already spent.
    pub fn failed(job_id: impl Into<String>, wall_time_ms: u64, error: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            outcome: JobOutcome::Failed,
            output_blob_key: None,
            output_manifest: None,
            output_hash: None,
            wall_time_ms,
            error: Some(error.into()),
        }
    }
}
