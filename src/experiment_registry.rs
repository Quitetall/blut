// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `experiment://<recipe>/<run>` views over lineage (ADR 0090 M2.2).
//!
//! An experiment is deliberately not a second mutable store. BLUT already
//! records every run with its recipe, tenant, input/argument fingerprint, gate
//! outcome, and headline metrics. This module gives that data a stable URI and
//! comparison surface while keeping lineage as the sole source of truth.

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};
use crate::lineage_db::{LineageDb, RunRow};
use crate::lineage_report::RunDiff;
use crate::tenant::Tenant;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentComparison {
    pub experiment: String,
    pub tenant: String,
    /// Older run in the comparison.
    pub run_a: String,
    /// Newer run in the comparison.
    pub run_b: String,
    pub diff: RunDiff,
}

/// Parse `experiment://<recipe>/<run>`. Both identifiers are kept path-free so
/// the URI cannot be reinterpreted as a filesystem location.
pub fn parse_experiment_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri.strip_prefix("experiment://")?;
    let (experiment, run) = rest.split_once('/')?;
    if !is_safe_experiment_name(experiment) || !is_safe_experiment_name(run) || run.contains('/') {
        return None;
    }
    Some((experiment, run))
}

/// Resolve one experiment URI against the exact owning tenant.
pub fn resolve_uri(db: &LineageDb, uri: &str, tenant: &Tenant) -> Result<RunRow> {
    let (experiment, run) = parse_experiment_uri(uri).ok_or_else(|| {
        TrainError::other(format!("not an `experiment://<name>/<run>` URI: {uri}"))
    })?;
    let row = db.get_run(run)?.ok_or_else(|| {
        TrainError::other(format!(
            "unresolved experiment run {uri} for tenant '{tenant}'"
        ))
    })?;
    let actual_experiment = row.experiment.as_deref().unwrap_or(&row.recipe);
    if actual_experiment != experiment || row.tenant != tenant.to_string() {
        return Err(TrainError::other(format!(
            "unresolved experiment run {uri} for tenant '{tenant}'"
        )));
    }
    Ok(row)
}

/// Compare the two newest runs for `(tenant, experiment)`. `run_a` is older and
/// `run_b` newer, matching the usual baseline→candidate reading direction.
pub fn compare_latest(
    db: &LineageDb,
    experiment: &str,
    tenant: &Tenant,
) -> Result<ExperimentComparison> {
    if !is_safe_experiment_name(experiment) {
        return Err(TrainError::other(format!(
            "invalid experiment name '{experiment}'"
        )));
    }
    let tenant_key = tenant.to_string();
    let newest = db.runs_for_experiment_tenant(experiment, &tenant_key, 2)?;
    if newest.len() < 2 {
        return Err(TrainError::other(format!(
            "experiment '{experiment}' needs at least two runs in tenant '{tenant}'"
        )));
    }
    let run_b = &newest[0];
    let run_a = &newest[1];
    let diff = db.run_diff_for_tenant(&run_a.job_id, &run_b.job_id, tenant)?;
    Ok(ExperimentComparison {
        experiment: experiment.to_string(),
        tenant: tenant.to_string(),
        run_a: run_a.job_id.clone(),
        run_b: run_b.job_id.clone(),
        diff,
    })
}

pub(crate) fn is_safe_experiment_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        && !value.starts_with('.')
        && !value.starts_with('-')
        && !value.contains("..")
}
