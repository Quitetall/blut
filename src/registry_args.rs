// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Resolve registry URIs in raw recipe arguments before typed deserialization.
//!
//! Resolution is recursive across JSON objects/arrays and exact-scheme only:
//! ordinary strings are byte-for-byte unchanged. Dataset handles become a
//! hash-verified local path, model handles become their immutable checkpoint
//! hash, and experiment handles become the tenant-scoped run id.
//!
//! # Why resolution is also REPORTED, not just applied
//!
//! Substituting `dataset://tusz@v2.0.6` with its path is what the typed arg
//! needs, but the path is the least durable part of what was proved. To return
//! it, [`crate::dataset_registry::resolve_uri`] first re-hashed the live bytes
//! and refused on drift, checked tenancy, and enforced the clinical node-local
//! rule. All of that evidence used to be dropped on the floor at the moment of
//! substitution, so a persisted run recorded a bare path and nothing could
//! later answer "which pinned dataset was that, and at what digest?".
//!
//! [`ResolvedHandles`] carries that evidence out alongside the rewritten args.
//! The substitution itself is unchanged — a dataset handle still becomes a
//! plain path string, so every existing typed arg keeps deserializing — which
//! is why this is additive and [`resolve_recipe_args`] still exists with its
//! original signature.

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::config::launcher::LaunchTarget;
use crate::dataset_registry::DatasetResolution;
use crate::error::{Result, TrainError};
use crate::lineage_db::LineageDb;
use crate::tenant::Tenant;

/// Every registry handle the args resolved to, captured at resolution time.
///
/// Order is first-appearance in the recursive walk and duplicates are collapsed,
/// so the same handle used twice in one arg set is recorded once and the record
/// is deterministic for a given `raw` — it can be hashed or diffed across runs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedHandles {
    /// Dataset bindings, each carrying the digest that was re-verified.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<DatasetResolution>,
}

impl ResolvedHandles {
    /// True when no registry handle appeared in the args at all.
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// Fold another resolution's handles in, collapsing repeats.
    ///
    /// Used where one launch record must cover several independent resolution
    /// passes — an HPO sweep resolves its base args and then each `Choice`
    /// value separately, and every dataset any trial can reach belongs on the
    /// sweep's record.
    pub fn absorb(&mut self, other: Self) {
        for resolved in other.datasets {
            self.push_dataset(resolved);
        }
    }

    /// Record a dataset binding, ignoring a repeat of one already held.
    ///
    /// The key is `(tenant, name, version)` — a `dataset://` URI resolves
    /// identically every time within one walk, so a differing digest under the
    /// same key cannot arise here; the narrower key just keeps the record one
    /// row per distinct handle.
    fn push_dataset(&mut self, resolved: DatasetResolution) {
        if !self.datasets.iter().any(|held| {
            held.tenant == resolved.tenant
                && held.name == resolved.name
                && held.version == resolved.version
        }) {
            self.datasets.push(resolved);
        }
    }
}

/// Production resolver. It opens only stores required by schemes actually
/// present in `raw`; URI-free args remain a no-I/O identity operation.
pub fn resolve_recipe_args(
    raw: serde_json::Value,
    tenant: &Tenant,
    launch_target: LaunchTarget,
) -> Result<serde_json::Value> {
    resolve_recipe_args_reported(raw, tenant, launch_target).map(|(args, _)| args)
}

/// As [`resolve_recipe_args`], and also return what the handles resolved to.
///
/// Prefer this at any call site that persists a run: the args alone cannot say
/// which pinned dataset produced a path, and the digest proved during
/// resolution is exactly what makes the record auditable later.
pub fn resolve_recipe_args_reported(
    raw: serde_json::Value,
    tenant: &Tenant,
    launch_target: LaunchTarget,
) -> Result<(serde_json::Value, ResolvedHandles)> {
    let needs = Needed::scan(&raw);
    if !needs.any() {
        return Ok((raw, ResolvedHandles::default()));
    }
    let datasets = needs.dataset.then(crate::datasets_db::open).transpose()?;
    let models = needs.model.then(crate::model_registry::open).transpose()?;
    let lineage = needs.experiment.then(LineageDb::open).transpose()?;
    let mut handles = ResolvedHandles::default();
    let args = resolve_inner(
        raw,
        tenant,
        launch_target,
        datasets.as_ref(),
        models.as_ref(),
        lineage.as_ref(),
        &mut handles,
    )?;
    Ok((args, handles))
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
    resolve_with_reported(raw, tenant, launch_target, datasets, models, lineage)
        .map(|(args, _)| args)
}

/// As [`resolve_with`], and also return what the handles resolved to.
pub fn resolve_with_reported(
    raw: serde_json::Value,
    tenant: &Tenant,
    launch_target: LaunchTarget,
    datasets: &Connection,
    models: &Connection,
    lineage: &LineageDb,
) -> Result<(serde_json::Value, ResolvedHandles)> {
    let mut handles = ResolvedHandles::default();
    let args = resolve_inner(
        raw,
        tenant,
        launch_target,
        Some(datasets),
        Some(models),
        Some(lineage),
        &mut handles,
    )?;
    Ok((args, handles))
}

fn resolve_inner(
    raw: serde_json::Value,
    tenant: &Tenant,
    launch_target: LaunchTarget,
    datasets: Option<&Connection>,
    models: Option<&Connection>,
    lineage: Option<&LineageDb>,
    handles: &mut ResolvedHandles,
) -> Result<serde_json::Value> {
    match raw {
        serde_json::Value::String(value) => resolve_string(
            value,
            tenant,
            launch_target,
            datasets,
            models,
            lineage,
            handles,
        ),
        serde_json::Value::Array(values) => values
            .into_iter()
            .map(|value| {
                resolve_inner(
                    value,
                    tenant,
                    launch_target,
                    datasets,
                    models,
                    lineage,
                    handles,
                )
            })
            .collect::<Result<Vec<_>>>()
            .map(serde_json::Value::Array),
        serde_json::Value::Object(values) => values
            .into_iter()
            .map(|(key, value)| {
                resolve_inner(
                    value,
                    tenant,
                    launch_target,
                    datasets,
                    models,
                    lineage,
                    handles,
                )
                .map(|resolved| (key, resolved))
            })
            .collect::<Result<serde_json::Map<_, _>>>()
            .map(serde_json::Value::Object),
        scalar => Ok(scalar),
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_string(
    value: String,
    tenant: &Tenant,
    launch_target: LaunchTarget,
    datasets: Option<&Connection>,
    models: Option<&Connection>,
    lineage: Option<&LineageDb>,
    handles: &mut ResolvedHandles,
) -> Result<serde_json::Value> {
    if value.starts_with("dataset://") {
        let conn = datasets.ok_or_else(|| TrainError::other("dataset registry unavailable"))?;
        let resolved = crate::dataset_registry::resolve_uri(conn, &value, tenant, launch_target)?;
        let path = resolved.source_path.to_str().ok_or_else(|| {
            TrainError::other(format!(
                "dataset path for {value} is not valid UTF-8 and cannot enter JSON recipe args"
            ))
        })?;
        let path = path.to_string();
        handles.push_dataset(resolved);
        return Ok(serde_json::Value::String(path));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn resolution(name: &str, version: &str) -> DatasetResolution {
        DatasetResolution {
            tenant: "research/dev".into(),
            name: name.into(),
            version: version.into(),
            dataset_id: format!("id-{name}-{version}"),
            source_name: format!("{name}-source"),
            source_path: std::path::PathBuf::from(format!("/data/{name}.lma")),
            manifest_sha256: "a".repeat(64),
            kind: "dataset.lma".into(),
            clinical: false,
            pinned_at: 1,
        }
    }

    /// An HPO sweep resolves base args and each `Choice` in separate passes;
    /// the marker must end up with every distinct handle exactly once.
    #[test]
    fn absorb_unions_across_passes_and_collapses_repeats() {
        let mut base = ResolvedHandles::default();
        base.push_dataset(resolution("tusz", "v2.0.6"));

        let mut choice_a = ResolvedHandles::default();
        choice_a.push_dataset(resolution("tuev", "v2.0.1"));
        // A second dimension reusing the base handle must not duplicate it.
        choice_a.push_dataset(resolution("tusz", "v2.0.6"));

        base.absorb(choice_a);

        let named: Vec<_> = base.datasets.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(named, vec!["tusz", "tuev"], "union, first-appearance order");

        // Absorbing nothing changes nothing; absorbing a repeat is idempotent.
        let before = base.clone();
        base.absorb(ResolvedHandles::default());
        let mut repeat = ResolvedHandles::default();
        repeat.push_dataset(resolution("tuev", "v2.0.1"));
        base.absorb(repeat);
        assert_eq!(base, before);
    }

    #[test]
    fn default_handles_are_empty() {
        assert!(ResolvedHandles::default().is_empty());
    }
}
