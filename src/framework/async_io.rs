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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
#[non_exhaustive]
pub enum IoMode {
    Inline,
    Bounded { capacity: u32, max_item_bytes: u64 },
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
}

/// One stage-owned candidate before optional size measurements are resolved.
///
/// `None` is an honest unknown. It never becomes zero silently: a candidate
/// that needs that measurement is skipped and the explicit inline tail is
/// selected instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingIoCandidate {
    /// Number of independent data-pipeline replicas retaining prefetched and
    /// CUDA-staged batches (for example DDP ranks). Worker counts remain
    /// per-replica so subprocess environment values stay truthful.
    pub data_replicas: u32,
    pub decode_workers: u32,
    pub prefetch_per_worker: u32,
    pub cuda_staging_slots: u32,
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

impl TrainingIoCandidate {
    fn is_inline_tail(&self) -> bool {
        self.decode_workers == 0
            && self.prefetch_per_worker == 0
            && self.cuda_staging_slots == 0
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
        let metrics_bytes = self.metrics.retained_bytes()?;
        let checkpoint_bytes = self.checkpoints.retained_bytes()?;
        let fixed_overhead_bytes = fixed_overhead_bytes_per_replica
            .checked_mul(u64::from(self.data_replicas))
            .ok_or(CandidateFailure::ArithmeticOverflow)?;
        let billed_overhead_bytes = [
            prefetch_bytes,
            cuda_bytes,
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

        Ok(TrainingIoProfile {
            sync_base_bytes: 0,
            data_replicas: self.data_replicas,
            decode_workers: self.decode_workers,
            prefetch_per_worker: self.prefetch_per_worker,
            cuda_staging_slots: self.cuda_staging_slots,
            metrics: self.metrics.clone(),
            checkpoints: self.checkpoints.clone(),
            batch_bytes,
            checkpoint_snapshot_bytes,
            fixed_overhead_bytes: fixed_overhead_bytes_per_replica,
            billed_overhead_bytes,
            downgrade_reason: None,
        })
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
    pub metrics: IoMode,
    pub checkpoints: IoMode,
    pub batch_bytes: u64,
    pub checkpoint_snapshot_bytes: u64,
    /// Measured fixed resident bytes PER data replica. The aggregate
    /// `data_replicas * fixed_overhead_bytes` term is already included in
    /// `billed_overhead_bytes`.
    pub fixed_overhead_bytes: u64,
    pub billed_overhead_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downgrade_reason: Option<TrainingIoDowngradeReason>,
}

impl TrainingIoProfile {
    pub fn is_inline(&self) -> bool {
        self.decode_workers == 0
            && self.prefetch_per_worker == 0
            && self.cuda_staging_slots == 0
            && self.metrics.is_inline()
            && self.checkpoints.is_inline()
            && self.fixed_overhead_bytes == 0
            && self.billed_overhead_bytes == 0
    }

    /// Canonical BLUT-owned subprocess environment. Cookbook-specific aliases
    /// (for example a trainer's worker/prefetch names) are intentionally left
    /// to that cookbook adapter.
    pub fn env_pairs(&self) -> BTreeMap<&'static str, String> {
        let (metrics_capacity, metrics_max_item) = self.metrics.capacity_and_max_item();
        let (checkpoint_capacity, checkpoint_max_item) = self.checkpoints.capacity_and_max_item();
        BTreeMap::from([
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
