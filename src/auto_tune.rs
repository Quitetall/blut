// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Auto-tuner control policies (ADR 0110) — five OPT-IN tuners that PROPOSE a
//! config change; a tuner never edits a running trainer, and every proposal is
//! re-admitted by the broker before it takes effect. A tuner emits a typed
//! [`TuneProposal`]; the scheduler applies it by spawning a NEW stage-run with
//! the amended (still content-addressed) config, and appends the proposal + the
//! broker's accept/refuse verdict to the run's `tuning.jsonl`.
//!
//! The five: `AutoBatch` (the broker's largest-admissible batch — never a
//! launch-and-catch), `GradAccum` (arithmetic on already-admitted numbers),
//! `LrFinder` (min-gradient LR from a bounded probe curve), `AutoAmp` (a dtype
//! from a capability probe, refused for fp32-only stages), `LossSpikeTuner`
//! (reactive: a finite spike → LR back-off; a NaN still hits `KillOnNaN`).
//! All off by default — a run with no tuner flag is byte-identical to today.

use serde::{Deserialize, Serialize};

/// A typed proposal: change `field` from `old` to `new` because `reason`. Values
/// are JSON so one record covers batch/accum/lr/amp uniformly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneProposal {
    pub policy: String,
    pub field: String,
    pub old: serde_json::Value,
    pub new: serde_json::Value,
    pub reason: String,
}

impl TuneProposal {
    fn new(
        policy: &str,
        field: &str,
        old: serde_json::Value,
        new: serde_json::Value,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            policy: policy.into(),
            field: field.into(),
            old,
            new,
            reason: reason.into(),
        }
    }
}

/// The broker's verdict on re-admitting a proposal (the proposal is NEVER
/// applied without passing admission).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerVerdict {
    Accepted,
    Refused { reason: String },
}

// ── AutoBatch ──────────────────────────────────────────────────────

/// Propose the largest batch that ADMITS under the current budget — the caller
/// passes `admissible_ceiling` from the broker's analytic
/// `broker::footprint::batch_size_to_fit` (never a launch-and-catch). Returns
/// `None` when the batch is already the ceiling (nothing to do) OR when the
/// ceiling is below the operator `floor_batch` (fail-closed: the tuner proposes
/// nothing and the stage fails admission as it would today — the tuner never
/// lowers a stage below its floor).
pub fn auto_batch(
    current_batch: u32,
    admissible_ceiling: u32,
    floor_batch: u32,
) -> Option<TuneProposal> {
    if admissible_ceiling == 0
        || admissible_ceiling < floor_batch
        || admissible_ceiling == current_batch
    {
        return None;
    }
    Some(TuneProposal::new(
        "auto_batch",
        "batch",
        serde_json::json!(current_batch),
        serde_json::json!(admissible_ceiling),
        format!("largest batch admitting under budget: {admissible_ceiling}"),
    ))
}

// ── GradAccum ──────────────────────────────────────────────────────

/// Given a target EFFECTIVE batch and the `micro`-batch the broker admitted,
/// propose `accum_steps = ceil(target / micro)`. Pure arithmetic on
/// already-admitted numbers — never raises the micro-batch. `None` if the micro
/// is 0 or the target already fits in one micro-step.
pub fn grad_accum(target_effective: u32, micro: u32) -> Option<TuneProposal> {
    if micro == 0 {
        return None;
    }
    let steps = target_effective.div_ceil(micro).max(1);
    if steps <= 1 {
        return None;
    }
    Some(TuneProposal::new(
        "grad_accum",
        "accum_steps",
        serde_json::Value::Null,
        serde_json::json!(steps),
        format!("ceil(target {target_effective} / micro {micro}) = {steps}"),
    ))
}

// ── LrFinder ───────────────────────────────────────────────────────

/// Propose the min-gradient learning rate from a bounded LR-range probe curve
/// (`(lr, loss)` points, ascending lr). "Min gradient" = the steepest DOWNWARD
/// slope of loss vs `log(lr)` (the classic LR-finder pick). `None` for a curve
/// with < 3 points or no descending segment (nothing to learn).
pub fn lr_finder(curve: &[(f64, f64)]) -> Option<TuneProposal> {
    if curve.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, f64)> = None; // (lr, slope)
    for w in curve.windows(2) {
        let (lr0, l0) = w[0];
        let (lr1, l1) = w[1];
        if lr0 <= 0.0 || lr1 <= 0.0 {
            continue;
        }
        let dlog = lr1.ln() - lr0.ln();
        if dlog.abs() < f64::EPSILON {
            continue;
        }
        let slope = (l1 - l0) / dlog; // dloss/dlog(lr)
        // The pick is the lr at the START of the steepest DOWNWARD segment.
        if slope < 0.0 && best.map(|(_, s)| slope < s).unwrap_or(true) {
            best = Some((lr0, slope));
        }
    }
    best.map(|(lr, slope)| {
        TuneProposal::new(
            "lr_finder",
            "lr",
            serde_json::Value::Null,
            serde_json::json!(lr),
            format!("min-gradient lr (steepest descent, slope {slope:.4})"),
        )
    })
}

// ── AutoAmp ────────────────────────────────────────────────────────

/// AMP dtype selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmpDtype {
    Bf16,
    Fp16,
    Off,
}

/// Propose an AMP dtype: `bf16` where supported, else `fp16` (with a loss
/// scaler), else off. REFUSES (returns `None`) for a stage the numerics-guard
/// marks fp32-only (e.g. ternary-QAT calibration) — the tuner never proposes AMP
/// where it would break numerics. `None` too when the proposed dtype already
/// equals `current` (no no-op audit rows); `current` is recorded as `old`.
pub fn auto_amp(bf16_supported: bool, fp32_only: bool, current: AmpDtype) -> Option<TuneProposal> {
    if fp32_only {
        return None;
    }
    let dtype = if bf16_supported {
        AmpDtype::Bf16
    } else {
        AmpDtype::Fp16
    };
    if dtype == current {
        return None;
    }
    Some(TuneProposal::new(
        "auto_amp",
        "amp",
        serde_json::to_value(current).unwrap(),
        serde_json::to_value(dtype).unwrap(),
        format!("capability probe → {dtype:?}"),
    ))
}

// ── LossSpikeTuner (reactive) ──────────────────────────────────────

/// The reactive response to the latest loss sample.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpikeAction {
    /// No anomaly — continue.
    None,
    /// A finite spike: back the LR off by the factor.
    BackOff(f64),
    /// Composes with the landed `KillOnNaN` — a NaN STILL kills; the tuner does
    /// not override it, it defers.
    Kill,
}

/// React to a loss window: a NaN latest sample defers to `KillOnNaN` (`Kill`); a
/// FINITE latest sample whose z-score over the prior window exceeds `z_threshold`
/// gets a graduated `BackOff(factor)`; otherwise `None`. Never overrides
/// `KillOnNaN` (NaN → Kill), never reacts to normal noise.
pub fn loss_spike(window: &[f64], z_threshold: f64, backoff: f64) -> SpikeAction {
    let Some(&latest) = window.last() else {
        return SpikeAction::None;
    };
    if latest.is_nan() {
        return SpikeAction::Kill; // NaN still hits KillOnNaN
    }
    let prior = &window[..window.len().saturating_sub(1)];
    if prior.len() < 2 {
        return SpikeAction::None;
    }
    let n = prior.len() as f64;
    let mean = prior.iter().sum::<f64>() / n;
    // Population variance (÷n, not the Bessel ÷(n-1)) — intentional: for a
    // conservative z-score spike heuristic the tighter estimate is fine.
    let var = prior.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd < f64::EPSILON {
        return SpikeAction::None;
    }
    let z = (latest - mean) / sd;
    if z > z_threshold {
        SpikeAction::BackOff(backoff)
    } else {
        SpikeAction::None
    }
}

// ── tuning.jsonl audit ─────────────────────────────────────────────

/// Append a proposal + the broker's verdict to `path` as one JSON line — the
/// run's tuning audit trail (ADR 0110). One `write_all` under `O_APPEND` (POSIX
/// guarantees atomic append for records ≤ PIPE_BUF; a single tuning line is well
/// under that).
pub fn append_tuning(
    path: &std::path::Path,
    proposal: &TuneProposal,
    verdict: &BrokerVerdict,
    now_unix: i64,
) -> std::io::Result<()> {
    use std::io::Write as _;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut line = serde_json::json!({
        "ts": now_unix,
        "proposal": proposal,
        "verdict": verdict,
    })
    .to_string();
    line.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    f.write_all(line.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_batch_equals_the_broker_admission_ceiling() {
        // A big requested batch on a tight budget: the broker's ceiling (as the
        // caller resolves it via `shrink_to_fit_env` over the DECLARED batch
        // cost term, ADR 0133) is below the request. AutoBatch proposes EXACTLY
        // that ceiling (never over-budget). Ceiling pinned as a literal — the
        // formula lives in the cookbook now.
        let requested = 64u32;
        let ceiling = 12u32;
        let p = auto_batch(requested, ceiling, 1);
        let p = p.expect("a below-request ceiling ⇒ a proposal");
        assert_eq!(
            p.new,
            serde_json::json!(ceiling),
            "proposes the broker ceiling exactly"
        );
        assert_eq!(p.field, "batch");
        assert!(
            auto_batch(64, 64, 1).is_none(),
            "already at ceiling ⇒ no proposal"
        );
        // Fail-closed: a ceiling below the operator floor ⇒ propose NOTHING.
        assert!(
            auto_batch(64, 4, 8).is_none(),
            "ceiling < floor ⇒ no proposal (fail-closed)"
        );
        // A zero ceiling (nothing fits) never proposes a zero batch.
        assert!(auto_batch(5, 0, 1).is_none(), "ceiling 0 ⇒ no proposal");
    }

    #[test]
    fn grad_accum_is_ceil_division() {
        let p = grad_accum(256, 24).unwrap();
        assert_eq!(p.new, serde_json::json!(11)); // ceil(256/24)=11
        assert!(
            grad_accum(16, 16).is_none(),
            "target fits one micro ⇒ no accum"
        );
        assert!(grad_accum(100, 0).is_none());
    }

    #[test]
    fn lr_finder_picks_steepest_descent() {
        // Loss falls fastest around lr=1e-2, then diverges.
        let curve = [
            (1e-5, 2.30),
            (1e-4, 2.28),
            (1e-3, 2.10),
            (1e-2, 1.20),
            (1e-1, 3.50),
        ];
        let p = lr_finder(&curve).unwrap();
        // Steepest downward slope is the 1e-3 → 1e-2 segment; the pick is its start.
        assert_eq!(p.new, serde_json::json!(1e-3));
        assert!(
            lr_finder(&[(1e-3, 1.0), (1e-2, 2.0)]).is_none(),
            "no descent, <3 pts"
        );
    }

    #[test]
    fn auto_amp_prefers_bf16_refuses_fp32_only() {
        let p = auto_amp(true, false, AmpDtype::Off).unwrap();
        assert_eq!(p.new, serde_json::to_value(AmpDtype::Bf16).unwrap());
        assert_eq!(p.old, serde_json::to_value(AmpDtype::Off).unwrap());
        assert_eq!(
            auto_amp(false, false, AmpDtype::Off).unwrap().new,
            serde_json::to_value(AmpDtype::Fp16).unwrap()
        );
        assert!(
            auto_amp(true, true, AmpDtype::Off).is_none(),
            "fp32-only stage refuses AMP"
        );
        assert!(
            auto_amp(true, false, AmpDtype::Bf16).is_none(),
            "already bf16 ⇒ no no-op proposal"
        );
    }

    #[test]
    fn loss_spike_backs_off_but_nan_defers_to_kill() {
        let calm = [1.0, 1.01, 0.99, 1.0, 1.02];
        assert_eq!(loss_spike(&calm, 3.0, 0.5), SpikeAction::None);
        let spike = [1.0, 1.01, 0.99, 1.0, 9.5];
        assert_eq!(loss_spike(&spike, 3.0, 0.5), SpikeAction::BackOff(0.5));
        let nan = [1.0, 1.0, 1.0, f64::NAN];
        assert_eq!(
            loss_spike(&nan, 3.0, 0.5),
            SpikeAction::Kill,
            "NaN still hits KillOnNaN"
        );
    }

    #[test]
    fn tuning_audit_records_proposal_and_verdict() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("tuning.jsonl");
        let p = auto_batch(64, 32, 1).unwrap();
        append_tuning(&path, &p, &BrokerVerdict::Accepted, 1000).unwrap();
        append_tuning(
            &path,
            &grad_accum(256, 24).unwrap(),
            &BrokerVerdict::Refused {
                reason: "over budget".into(),
            },
            1001,
        )
        .unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body.lines().count(), 2, "one audit row per proposal");
        assert!(body.contains("auto_batch") && body.contains("accepted"));
        assert!(body.contains("grad_accum") && body.contains("refused"));
    }
}
