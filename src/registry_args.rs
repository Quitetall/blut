// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Resolve registry URIs in raw recipe arguments before typed deserialization.
//!
//! Resolution is recursive across JSON objects/arrays and exact-scheme only:
//! ordinary strings are byte-for-byte unchanged. Dataset handles become a
//! hash-verified local path, model handles become their immutable checkpoint
//! hash, and experiment handles become the tenant-scoped run id.

use rusqlite::Connection;

use crate::config::launcher::LaunchTarget;
use crate::error::{Result, TrainError};
use crate::lineage_db::LineageDb;
use crate::tenant::Tenant;

/// Production resolver. It opens only stores required by schemes actually
/// present in `raw`; URI-free args remain a no-I/O identity operation.
pub fn resolve_recipe_args(
    raw: serde_json::Value,
    tenant: &Tenant,
    launch_target: LaunchTarget,
) -> Result<serde_json::Value> {
    let needs = Needed::scan(&raw);
    if !needs.any() {
        return Ok(raw);
    }
    let datasets = needs.dataset.then(crate::datasets_db::open).transpose()?;
    let models = needs.model.then(crate::model_registry::open).transpose()?;
    let lineage = needs.experiment.then(LineageDb::open).transpose()?;
    resolve_inner(
        raw,
        tenant,
        launch_target,
        datasets.as_ref(),
        models.as_ref(),
        lineage.as_ref(),
    )
}

/// Injectable resolver for tests/embedders that already own their DB handles.
pub fn resolve_with(
    raw: serde_json::Value,
    tenant: &Tenant,
    launch_target: LaunchTarget,
    datasets: &Connection,
    models: &Connection,
    lineage: &LineageDb,
) -> Result<serde_json::Value> {
    resolve_inner(
        raw,
        tenant,
        launch_target,
        Some(datasets),
        Some(models),
        Some(lineage),
    )
}

fn resolve_inner(
    raw: serde_json::Value,
    tenant: &Tenant,
    launch_target: LaunchTarget,
    datasets: Option<&Connection>,
    models: Option<&Connection>,
    lineage: Option<&LineageDb>,
) -> Result<serde_json::Value> {
    match raw {
        serde_json::Value::String(value) => {
            resolve_string(value, tenant, launch_target, datasets, models, lineage)
        }
        serde_json::Value::Array(values) => values
            .into_iter()
            .map(|value| resolve_inner(value, tenant, launch_target, datasets, models, lineage))
            .collect::<Result<Vec<_>>>()
            .map(serde_json::Value::Array),
        serde_json::Value::Object(values) => values
            .into_iter()
            .map(|(key, value)| {
                resolve_inner(value, tenant, launch_target, datasets, models, lineage)
                    .map(|resolved| (key, resolved))
            })
            .collect::<Result<serde_json::Map<_, _>>>()
            .map(serde_json::Value::Object),
        scalar => Ok(scalar),
    }
}

fn resolve_string(
    value: String,
    tenant: &Tenant,
    launch_target: LaunchTarget,
    datasets: Option<&Connection>,
    models: Option<&Connection>,
    lineage: Option<&LineageDb>,
) -> Result<serde_json::Value> {
    if value.starts_with("dataset://") {
        let conn = datasets.ok_or_else(|| TrainError::other("dataset registry unavailable"))?;
        let resolved = crate::dataset_registry::resolve_uri(conn, &value, tenant, launch_target)?;
        let path = resolved.source_path.to_str().ok_or_else(|| {
            TrainError::other(format!(
                "dataset path for {value} is not valid UTF-8 and cannot enter JSON recipe args"
            ))
        })?;
        return Ok(serde_json::Value::String(path.to_string()));
    }
    if value.starts_with("model://") {
        let conn = models.ok_or_else(|| TrainError::other("model registry unavailable"))?;
        if tenant.is_restricted() && launch_target != LaunchTarget::Local {
            return Err(TrainError::other(format!(
                "model resolve refused: Restricted model handle {value} must stay node-local"
            )));
        }
        let (name, alias) = crate::model_registry::parse_model_uri(&value).ok_or_else(|| {
            TrainError::other(format!("not a `model://<name>@<alias>` URI: {value}"))
        })?;
        let tenant_key = tenant.to_string();
        let hash = crate::model_registry::resolve_pointer(conn, &tenant_key, name, alias)?
            .ok_or_else(|| {
                TrainError::other(format!(
                    "unresolved model handle {value} for tenant '{tenant}'"
                ))
            })?;
        return Ok(serde_json::Value::String(hash));
    }
    if value.starts_with("experiment://") {
        let db = lineage.ok_or_else(|| TrainError::other("lineage registry unavailable"))?;
        let run = crate::experiment_registry::resolve_uri(db, &value, tenant)?;
        return Ok(serde_json::Value::String(run.job_id));
    }
    Ok(serde_json::Value::String(value))
}

#[derive(Clone, Copy, Debug, Default)]
struct Needed {
    dataset: bool,
    model: bool,
    experiment: bool,
}

impl Needed {
    fn scan(value: &serde_json::Value) -> Self {
        let mut out = Self::default();
        out.visit(value);
        out
    }

    fn visit(&mut self, value: &serde_json::Value) {
        match value {
            serde_json::Value::String(value) => {
                self.dataset |= value.starts_with("dataset://");
                self.model |= value.starts_with("model://");
                self.experiment |= value.starts_with("experiment://");
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    self.visit(value);
                }
            }
            serde_json::Value::Object(values) => {
                for value in values.values() {
                    self.visit(value);
                }
            }
            _ => {}
        }
    }

    fn any(self) -> bool {
        self.dataset || self.model || self.experiment
    }
}
