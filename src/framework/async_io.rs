// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! UI-neutral, execution-only async-I/O admission profiles (ADR 0103).
//!
//! Cookbooks declare complete candidates in fastest-first order. The engine
//! turns exactly one candidate into a concrete [`TrainingIoProfile`] using
//! checked byte arithmetic before a stage runs. Profiles are runtime policy:
//! they never enter stage arguments, schemas, cache keys, logical hashes, or
//! artifact identity.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Producer behavior for one retained-I/O lane.
///
/// `Bounded { capacity: 1, .. }` is still asynchronous. `Inline` is the only
/// synchronous mode and serializes with capacity/max-item values of zero in
/// [`TrainingIoProfile::env_pairs`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
#[non_exhaustive]
pub enum IoMode {
    #[default]
    Inline,
    Bounded {
        capacity: u32,
        max_item_bytes: u64,
    },
}

impl IoMode {
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline)
    }

    fn capacity_and_max_item(&self) -> (u32, u64) {
        match self {
            Self::Inline => (0, 0),
            Self::Bounded {
                capacity,
                max_item_bytes,
            } => (*capacity, *max_item_bytes),
        }
    }

    fn retained_bytes(&self) -> Result<u64, CandidateFailure> {
        match self {
            Self::Inline => Ok(0),
            Self::Bounded { capacity: 0, .. }
            | Self::Bounded {
                max_item_bytes: 0, ..
            } => Err(CandidateFailure::UnknownSize),
            Self::Bounded {
                capacity,
                max_item_bytes,
            } => u64::from(*capacity)
                .checked_mul(*max_item_bytes)
                .ok_or(CandidateFailure::ArithmeticOverflow),
        }
    }

    /// Retained envelope bound for the cross-stage pipeline lane.
    ///
    /// Capacity covers each queued/running [`PipelineEmission`](crate::framework::stage::PipelineEmission).
    /// The one running item is also retained by `NodeTask` for retry-safe
    /// dispatch and by the erased stage call while it decodes. After the call,
    /// the latter allowance becomes the cap for that child's engine-private
    /// output/status spill. Those two single-consumer records are independent
    /// of queue width, so the engine-side envelope bound is
    /// `(capacity + 2) * max_item_bytes`.
    fn pipeline_retained_bytes(&self) -> Result<u64, CandidateFailure> {
        match self {
            Self::Inline => Ok(0),
            Self::Bounded { capacity: 0, .. }
            | Self::Bounded {
                max_item_bytes: 0, ..
            } => Err(CandidateFailure::UnknownSize),
            Self::Bounded {
                capacity,
                max_item_bytes,
            } => u64::from(*capacity)
                .checked_add(2)
                .and_then(|copies| copies.checked_mul(*max_item_bytes))
                .ok_or(CandidateFailure::ArithmeticOverflow),
        }
    }
}

/// One stage-owned candidate before optional size measurements are resolved.
///
/// `None` is an honest unknown. It never becomes zero silently: a candidate
/// that needs that measurement is skipped and the explicit inline tail is
/// selected instead.
///
/// Alpha migration note: ADR 0102 added the required `pipeline` lane to this
/// public literal-built struct. Existing external literals must add
/// `pipeline: IoMode::Inline` or migrate to `..TrainingIoCandidate::default()`.
/// The latter is the forward-compatible authoring form for optional lanes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingIoCandidate {
    /// Number of independent data-pipeline replicas retaining prefetched and
    /// CUDA-staged batches (for example DDP ranks). Worker counts remain
    /// per-replica so subprocess environment values stay truthful.
    pub data_replicas: u32,
    pub decode_workers: u32,
    pub prefetch_per_worker: u32,
    pub cuda_staging_slots: u32,
    /// Bounded retained-item lane used by cross-stage pipeline overlap.
    pub pipeline: IoMode,
    pub metrics: IoMode,
    pub checkpoints: IoMode,
    pub batch_bytes: Option<u64>,
    pub checkpoint_snapshot_bytes: Option<u64>,
    /// Additional measured resident bytes PER data replica not represented by
    /// queue payloads (decoder process RSS, allocator/runtime state, and
    /// similar). This value must EXCLUDE the explicit
    /// prefetch/CUDA/metric/checkpoint queue terms; the engine multiplies it by
    /// `data_replicas` and adds those queue terms with checked arithmetic.
    pub fixed_overhead_bytes: Option<u64>,
}

impl Default for TrainingIoCandidate {
    /// Fully-inline, measurement-free candidate suitable as the explicit tail
    /// and as the update base for cookbook literals. Cookbook authors should
    /// prefer `..TrainingIoCandidate::default()` so future optional lanes do
    /// not create another source migration.
    fn default() -> Self {
        Self {
            data_replicas: 1,
            decode_workers: 0,
            prefetch_per_worker: 0,
            cuda_staging_slots: 0,
            pipeline: IoMode::Inline,
            metrics: IoMode::Inline,
            checkpoints: IoMode::Inline,
            batch_bytes: None,
            checkpoint_snapshot_bytes: None,
            fixed_overhead_bytes: None,
        }
    }
}

impl TrainingIoCandidate {
    fn is_inline_tail(&self) -> bool {
        self.decode_workers == 0
            && self.prefetch_per_worker == 0
            && self.cuda_staging_slots == 0
            && self.pipeline.is_inline()
            && self.metrics.is_inline()
            && self.checkpoints.is_inline()
    }

    fn resolve(&self) -> Result<TrainingIoProfile, CandidateFailure> {
        let needs_batch =
            self.decode_workers > 0 && self.prefetch_per_worker > 0 || self.cuda_staging_slots > 0;
        let batch_bytes = match (needs_batch, self.batch_bytes) {
            (true, None | Some(0)) => return Err(CandidateFailure::UnknownSize),
            (_, measured) => measured.unwrap_or(0),
        };
        let checkpoint_snapshot_bytes = match (&self.checkpoints, self.checkpoint_snapshot_bytes) {
            (IoMode::Bounded { .. }, None) => return Err(CandidateFailure::UnknownSize),
            (IoMode::Bounded { max_item_bytes, .. }, Some(snapshot_bytes))
                if *max_item_bytes != snapshot_bytes =>
            {
                // There is one checkpoint snapshot size. Letting the lifecycle
                // record and queue bill disagree would make runtime equality
                // unprovable, so this candidate cannot be admitted.
                return Err(CandidateFailure::UnknownSize);
            }
            (_, measured) => measured.unwrap_or(0),
        };
        let has_async_retention = !self.is_inline_tail();
        let fixed_overhead_bytes_per_replica =
            match (has_async_retention, self.fixed_overhead_bytes) {
                (true, None) => return Err(CandidateFailure::UnknownSize),
                (_, measured) => measured.unwrap_or(0),
            };
        if has_async_retention && self.data_replicas == 0 {
            return Err(CandidateFailure::UnknownSize);
        }

        let prefetched_batches = u64::from(self.decode_workers)
            .checked_mul(u64::from(self.prefetch_per_worker))
            .ok_or(CandidateFailure::ArithmeticOverflow)?;
        let prefetch_bytes = prefetched_batches
            .checked_mul(batch_bytes)
            .and_then(|bytes| bytes.checked_mul(u64::from(self.data_replicas)))
            .ok_or(CandidateFailure::ArithmeticOverflow)?;
        let cuda_bytes = u64::from(self.cuda_staging_slots)
            .checked_mul(batch_bytes)
            .and_then(|bytes| bytes.checked_mul(u64::from(self.data_replicas)))
            .ok_or(CandidateFailure::ArithmeticOverflow)?;
        let pipeline_bytes = self.pipeline.pipeline_retained_bytes()?;
        let metrics_bytes = self.metrics.retained_bytes()?;
        let checkpoint_bytes = self.checkpoints.retained_bytes()?;
        let fixed_overhead_bytes = fixed_overhead_bytes_per_replica
            .checked_mul(u64::from(self.data_replicas))
            .ok_or(CandidateFailure::ArithmeticOverflow)?;
        let billed_overhead_bytes = [
            prefetch_bytes,
            cuda_bytes,
            pipeline_bytes,
            metrics_bytes,
            checkpoint_bytes,
            fixed_overhead_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, retained| {
            total
                .checked_add(retained)
                .ok_or(CandidateFailure::ArithmeticOverflow)
        })?;

        // Built with a placeholder depth, then set from the ONE definition —
        // so the serialized value cannot encode a different rule than
        // `derived_async_depth`.
        let mut profile = TrainingIoProfile {
            async_depth: 0,
            sync_base_bytes: 0,
            data_replicas: self.data_replicas,
            decode_workers: self.decode_workers,
            prefetch_per_worker: self.prefetch_per_worker,
            cuda_staging_slots: self.cuda_staging_slots,
            pipeline: self.pipeline.clone(),
            metrics: self.metrics.clone(),
            checkpoints: self.checkpoints.clone(),
            batch_bytes,
            checkpoint_snapshot_bytes,
            fixed_overhead_bytes: fixed_overhead_bytes_per_replica,
            billed_overhead_bytes,
            downgrade_reason: None,
        };
        profile.async_depth = profile.derived_async_depth();
        Ok(profile)
    }
}

/// True when `profile` is the concrete checked form of one declared candidate.
/// The resolver-owned downgrade reason is intentionally ignored.
pub fn profile_is_declared(
    profile: &TrainingIoProfile,
    candidates: &[TrainingIoCandidate],
) -> bool {
    candidates.iter().any(|candidate| {
        candidate.resolve().is_ok_and(|mut declared| {
            declared.sync_base_bytes = profile.sync_base_bytes;
            declared.downgrade_reason = profile.downgrade_reason.clone();
            declared == *profile
        })
    })
}

/// Immutable hints captured once by the launch path and supplied to each
/// stage's candidate declaration. This type contains no resource probe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TrainingIoHints {
    pub admitted_decode_workers: Option<u32>,
    pub admitted_batch_size: Option<u32>,
    pub cache_warm: bool,
}

/// Why the resolver did not select the first (fastest) candidate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TrainingIoDowngradeReason {
    /// Compatibility spelling retained for persisted pre-ADR-0103 previews.
    /// New selections use one of the explicit causes below.
    ForcedInline,
    UserForced,
    SnapshotUnavailable,
    UnsupportedLauncher,
    /// The compiled optimizer witness does not authorize the cross-stage
    /// pipeline lane for this node, so bounded pipeline candidates were
    /// excluded before selection.
    PipelineUnavailable,
    UnknownSize,
    ArithmeticOverflow,
    BudgetPressure,
}

/// The one concrete runtime profile admitted for a node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub struct TrainingIoProfile {
    /// Exact synchronous/base envelope selected from the immutable launch
    /// snapshot. The executor adds `billed_overhead_bytes` and rounds only
    /// when acquiring whole-GiB permits; cookbook containment can use the same
    /// value without probing or recalibrating at run time.
    pub sync_base_bytes: u64,
    pub data_replicas: u32,
    pub decode_workers: u32,
    pub prefetch_per_worker: u32,
    pub cuda_staging_slots: u32,
    /// Effective cross-stage pipeline lane selected by admission.
    #[serde(default)]
    pub pipeline: IoMode,
    pub metrics: IoMode,
    pub checkpoints: IoMode,
    pub batch_bytes: u64,
    pub checkpoint_snapshot_bytes: u64,
    /// Measured fixed resident bytes PER data replica. The aggregate
    /// `data_replicas * fixed_overhead_bytes` term is already included in
    /// `billed_overhead_bytes`.
    pub fixed_overhead_bytes: u64,
    pub billed_overhead_bytes: u64,
    /// In-flight batch depth (ADR 0103). Surfaced on `status.jsonl` so an
    /// operator can see that a memory-pressured stage was admitted at reduced
    /// depth rather than refused — `1` means it ran the synchronous path.
    /// Set ONLY by the resolver, and pinned to [`Self::derived_async_depth`]
    /// by test so the reported number cannot drift from the profile.
    #[serde(default)]
    pub async_depth: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downgrade_reason: Option<TrainingIoDowngradeReason>,
}

impl TrainingIoProfile {
    /// Effective async DEPTH (ADR 0103): how many batches may be in flight.
    /// `1` IS the synchronous path, which is what makes "degrade to depth 1"
    /// and "fall back to inline" the same statement — the ADR's graceful
    /// degradation has no separate mechanism to keep in sync.
    ///
    /// Derived, never stored twice: [`Self::async_depth`] is the one definition
    /// and the serialized `async_depth` field is pinned to it by test.
    pub fn derived_async_depth(&self) -> u32 {
        if self.is_inline() {
            1
        } else {
            self.prefetch_per_worker.max(1)
        }
    }

    pub fn is_inline(&self) -> bool {
        self.decode_workers == 0
            && self.prefetch_per_worker == 0
            && self.cuda_staging_slots == 0
            && self.pipeline.is_inline()
            && self.metrics.is_inline()
            && self.checkpoints.is_inline()
            && self.fixed_overhead_bytes == 0
            && self.billed_overhead_bytes == 0
    }

    /// Canonical BLUT-owned subprocess environment. Cookbook-specific aliases
    /// (for example a trainer's worker/prefetch names) are intentionally left
    /// to that cookbook adapter.
    pub fn env_pairs(&self) -> BTreeMap<&'static str, String> {
        let (pipeline_capacity, pipeline_max_item) = self.pipeline.capacity_and_max_item();
        let (metrics_capacity, metrics_max_item) = self.metrics.capacity_and_max_item();
        let (checkpoint_capacity, checkpoint_max_item) = self.checkpoints.capacity_and_max_item();
        BTreeMap::from([
            (
                "BLUT_IO_ASYNC_DEPTH",
                self.derived_async_depth().to_string(),
            ),
            (
                "BLUT_IO_PIPELINE_MODE",
                if self.pipeline.is_inline() {
                    "inline"
                } else {
                    "bounded"
                }
                .to_string(),
            ),
            ("BLUT_IO_PIPELINE_CAPACITY", pipeline_capacity.to_string()),
            (
                "BLUT_IO_PIPELINE_MAX_ITEM_BYTES",
                pipeline_max_item.to_string(),
            ),
            (
                "BLUT_IO_METRICS_MODE",
                if self.metrics.is_inline() {
                    "inline"
                } else {
                    "bounded"
                }
                .to_string(),
            ),
            ("BLUT_IO_METRICS_CAPACITY", metrics_capacity.to_string()),
            (
                "BLUT_IO_METRICS_MAX_ITEM_BYTES",
                metrics_max_item.to_string(),
            ),
            (
                "BLUT_IO_CHECKPOINTS_MODE",
                if self.checkpoints.is_inline() {
                    "inline"
                } else {
                    "bounded"
                }
                .to_string(),
            ),
            (
                "BLUT_IO_CHECKPOINTS_CAPACITY",
                checkpoint_capacity.to_string(),
            ),
            (
                "BLUT_IO_CHECKPOINTS_MAX_ITEM_BYTES",
                checkpoint_max_item.to_string(),
            ),
            (
                "BLUT_IO_CUDA_STAGING_SLOTS",
                self.cuda_staging_slots.to_string(),
            ),
            ("BLUT_IO_BATCH_BYTES", self.batch_bytes.to_string()),
            (
                "BLUT_IO_CHECKPOINT_SNAPSHOT_BYTES",
                self.checkpoint_snapshot_bytes.to_string(),
            ),
            (
                "BLUT_IO_BILLED_OVERHEAD_BYTES",
                self.billed_overhead_bytes.to_string(),
            ),
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateFailure {
    UnknownSize,
    ArithmeticOverflow,
}

impl From<CandidateFailure> for TrainingIoDowngradeReason {
    fn from(value: CandidateFailure) -> Self {
        match value {
            CandidateFailure::UnknownSize => Self::UnknownSize,
            CandidateFailure::ArithmeticOverflow => Self::ArithmeticOverflow,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainingIoAdmissionError {
    #[error("training I/O candidates require an explicit fully-inline tail")]
    MissingInlineFallback,
    #[error(
        "synchronous base envelope {base_bytes} bytes exceeds memory budget {budget_bytes} bytes"
    )]
    BaseDoesNotFit { base_bytes: u64, budget_bytes: u64 },
    #[error("the explicit inline fallback is incomplete or has nonzero retained overhead")]
    InvalidInlineFallback,
}

/// Select the first complete candidate whose checked byte envelope fits.
///
/// The last candidate must be a fully-inline tail. Unknown or overflowing fast
/// candidates never receive a guessed size; the selector falls back to that
/// tail. The synchronous base is checked first and is never clamped.
pub fn select_training_io_profile(
    base_bytes: u64,
    budget_bytes: u64,
    candidates: &[TrainingIoCandidate],
    force_inline: bool,
) -> Result<TrainingIoProfile, TrainingIoAdmissionError> {
    select_training_io_profile_with_reason(
        base_bytes,
        budget_bytes,
        candidates,
        force_inline.then_some(TrainingIoDowngradeReason::UserForced),
    )
}

pub(crate) fn select_training_io_profile_with_reason(
    base_bytes: u64,
    budget_bytes: u64,
    candidates: &[TrainingIoCandidate],
    force_inline_reason: Option<TrainingIoDowngradeReason>,
) -> Result<TrainingIoProfile, TrainingIoAdmissionError> {
    if base_bytes > budget_bytes {
        return Err(TrainingIoAdmissionError::BaseDoesNotFit {
            base_bytes,
            budget_bytes,
        });
    }
    let inline = candidates
        .last()
        .filter(|candidate| candidate.is_inline_tail())
        .ok_or(TrainingIoAdmissionError::MissingInlineFallback)?;
    let inline_profile = inline
        .resolve()
        .map_err(|_| TrainingIoAdmissionError::InvalidInlineFallback)?;
    if !inline_profile.is_inline() {
        return Err(TrainingIoAdmissionError::InvalidInlineFallback);
    }

    if let Some(reason) = force_inline_reason {
        let mut profile = inline_profile;
        profile.sync_base_bytes = base_bytes;
        profile.downgrade_reason = Some(reason);
        return Ok(profile);
    }

    let mut downgrade_reason = None;
    for candidate in candidates {
        let mut profile = match candidate.resolve() {
            Ok(profile) => profile,
            Err(failure) => {
                downgrade_reason.get_or_insert_with(|| failure.into());
                continue;
            }
        };
        profile.sync_base_bytes = base_bytes;
        let Some(total_bytes) = base_bytes.checked_add(profile.billed_overhead_bytes) else {
            downgrade_reason.get_or_insert(TrainingIoDowngradeReason::ArithmeticOverflow);
            continue;
        };
        if total_bytes <= budget_bytes {
            profile.downgrade_reason = downgrade_reason;
            return Ok(profile);
        }
        downgrade_reason.get_or_insert(TrainingIoDowngradeReason::BudgetPressure);
    }

    // The synchronous base was proven to fit above, so reaching this point
    // means the advertised inline tail was not actually complete/zero-cost.
    Err(TrainingIoAdmissionError::InvalidInlineFallback)
}

#[cfg(test)]
mod tests {
    use super::{
        IoMode, TrainingIoCandidate, TrainingIoDowngradeReason, TrainingIoProfile,
        profile_is_declared, select_training_io_profile,
    };

    fn inline_candidate() -> TrainingIoCandidate {
        TrainingIoCandidate {
            data_replicas: 1,
            decode_workers: 0,
            prefetch_per_worker: 0,
            cuda_staging_slots: 0,
            pipeline: IoMode::Inline,
            metrics: IoMode::Inline,
            checkpoints: IoMode::Inline,
            batch_bytes: Some(0),
            checkpoint_snapshot_bytes: Some(0),
            fixed_overhead_bytes: Some(0),
        }
    }

    fn bounded_pipeline_candidate() -> TrainingIoCandidate {
        TrainingIoCandidate {
            data_replicas: 1,
            decode_workers: 0,
            prefetch_per_worker: 0,
            cuda_staging_slots: 0,
            pipeline: IoMode::Bounded {
                capacity: 3,
                max_item_bytes: 5,
            },
            metrics: IoMode::Inline,
            checkpoints: IoMode::Inline,
            batch_bytes: Some(0),
            checkpoint_snapshot_bytes: Some(0),
            fixed_overhead_bytes: Some(0),
        }
    }

    #[test]
    fn pipeline_lane_bills_queued_and_running_erased_residency() {
        let candidates = [bounded_pipeline_candidate(), inline_candidate()];
        let profile = select_training_io_profile(100, 125, &candidates, false)
            .expect("bounded pipeline fits exactly");

        assert!(!profile.is_inline());
        assert_eq!(profile.billed_overhead_bytes, 25);
        assert!(profile_is_declared(&profile, &candidates));
        let env = profile.env_pairs();
        assert_eq!(
            env.get("BLUT_IO_PIPELINE_MODE"),
            Some(&"bounded".to_string())
        );
        assert_eq!(env.get("BLUT_IO_PIPELINE_CAPACITY"), Some(&"3".to_string()));
        assert_eq!(
            env.get("BLUT_IO_PIPELINE_MAX_ITEM_BYTES"),
            Some(&"5".to_string())
        );

        let mut undeclared = profile.clone();
        undeclared.pipeline = IoMode::Inline;
        assert!(!profile_is_declared(&undeclared, &candidates));
    }

    #[test]
    fn pipeline_lane_downgrades_on_budget_or_overflow() {
        let candidates = [bounded_pipeline_candidate(), inline_candidate()];
        let budget_limited =
            select_training_io_profile(100, 124, &candidates, false).expect("inline tail fits");
        assert!(budget_limited.is_inline());
        assert_eq!(
            budget_limited.downgrade_reason,
            Some(TrainingIoDowngradeReason::BudgetPressure)
        );

        let mut overflowing = bounded_pipeline_candidate();
        overflowing.pipeline = IoMode::Bounded {
            capacity: u32::MAX,
            max_item_bytes: u64::MAX,
        };
        let overflowed =
            select_training_io_profile(0, u64::MAX, &[overflowing, inline_candidate()], false)
                .expect("overflow selects inline tail");
        assert!(overflowed.is_inline());
        assert_eq!(
            overflowed.downgrade_reason,
            Some(TrainingIoDowngradeReason::ArithmeticOverflow)
        );

        let mut zero_capacity = bounded_pipeline_candidate();
        zero_capacity.pipeline = IoMode::Bounded {
            capacity: 0,
            max_item_bytes: 5,
        };
        let unknown =
            select_training_io_profile(0, 15, &[zero_capacity, inline_candidate()], false)
                .expect("invalid bounded lane selects inline tail");
        assert!(unknown.is_inline());
        assert_eq!(
            unknown.downgrade_reason,
            Some(TrainingIoDowngradeReason::UnknownSize)
        );
    }

    #[test]
    fn force_inline_disables_pipeline_lane_and_exports_zero_capacity() {
        let profile = select_training_io_profile(
            100,
            115,
            &[bounded_pipeline_candidate(), inline_candidate()],
            true,
        )
        .expect("forced inline tail fits");

        assert!(profile.is_inline());
        assert_eq!(profile.pipeline, IoMode::Inline);
        assert_eq!(
            profile.downgrade_reason,
            Some(TrainingIoDowngradeReason::UserForced)
        );
        let env = profile.env_pairs();
        assert_eq!(
            env.get("BLUT_IO_PIPELINE_MODE"),
            Some(&"inline".to_string())
        );
        assert_eq!(env.get("BLUT_IO_PIPELINE_CAPACITY"), Some(&"0".to_string()));
        assert_eq!(
            env.get("BLUT_IO_PIPELINE_MAX_ITEM_BYTES"),
            Some(&"0".to_string())
        );
    }

    #[test]
    fn legacy_profile_without_pipeline_lane_decodes_as_inline() {
        let profile: TrainingIoProfile = serde_json::from_str(
            r#"{
                "sync_base_bytes": 100,
                "data_replicas": 1,
                "decode_workers": 0,
                "prefetch_per_worker": 0,
                "cuda_staging_slots": 0,
                "metrics": {"mode": "inline"},
                "checkpoints": {"mode": "inline"},
                "batch_bytes": 0,
                "checkpoint_snapshot_bytes": 0,
                "fixed_overhead_bytes": 0,
                "billed_overhead_bytes": 0
            }"#,
        )
        .expect("pre-pipeline status profile remains readable");

        assert_eq!(profile.pipeline, IoMode::Inline);
        assert!(profile.is_inline());
    }
}
