// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Immutable `dataset://<name>@<version>` bindings (ADR 0090 M2.2).
//!
//! The existing [`crate::datasets_db`] remains the source record and owns the
//! local path, manifest/content hash, kind, and ABIR metadata. This module adds
//! only the missing naming layer in that same SQLite file. A version binding is
//! append-only: pinning the identical tuple is idempotent; rebinding a version
//! to different bytes is refused and requires a new version.

use std::path::PathBuf;

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::config::launcher::LaunchTarget;
use crate::datasets_db::{self, DatasetRecord};
use crate::error::{Result, TrainError};
use crate::tenant::Tenant;

const CREATE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS dataset_refs (
    tenant          TEXT NOT NULL,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    dataset_id      TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    clinical        INTEGER NOT NULL,
    pinned_at       INTEGER NOT NULL,
    PRIMARY KEY (tenant, name, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_refs_hash_owner
    ON dataset_refs(manifest_sha256);
";

/// A resolved immutable dataset binding. The URI resolves to a local source
/// path only after its live bytes still match `manifest_sha256`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetResolution {
    pub tenant: String,
    pub name: String,
    pub version: String,
    pub dataset_id: String,
    pub source_name: String,
    pub source_path: PathBuf,
    pub manifest_sha256: String,
    pub kind: String,
    pub clinical: bool,
    pub pinned_at: i64,
}

/// Parse `dataset://<name>@<version>`. Both components use the registry's safe
/// identifier grammar, so a URI cannot smuggle paths, whitespace, or another
/// delimiter into a log/DB key.
pub fn parse_dataset_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri.strip_prefix("dataset://")?;
    let (name, version) = rest.split_once('@')?;
    if !safe_ident(name) || !safe_ident(version) || version.contains('@') {
        return None;
    }
    Some((name, version))
}

/// Bind an existing raw dataset record to an immutable tenant-scoped version.
/// Same binding is idempotent; changing a pinned version is refused.
pub fn pin(
    conn: &Connection,
    source_name: &str,
    uri: &str,
    tenant: &Tenant,
    now_unix: i64,
) -> Result<DatasetResolution> {
    ensure_schema(conn)?;
    let (name, version) = parse_dataset_uri(uri).ok_or_else(|| {
        TrainError::other(format!("not a `dataset://<name>@<version>` URI: {uri}"))
    })?;
    let source = datasets_db::get_by_name(conn, source_name)?.ok_or_else(|| {
        TrainError::other(format!(
            "dataset pin refused: no registered source dataset '{source_name}'"
        ))
    })?;
    verify_source(&source)?;
    let clinical = classify_source(&source, tenant)?;
    let tenant_key = tenant.to_string();

    // A manifest hash has one tenant owner, mirroring the model registry's
    // content-hash identity rule. Deliberate common data belongs in an explicit
    // `shared` tenant; silently pinning the same bytes into two namespaces would
    // bypass the default-no-cross-tenant provenance boundary.
    let existing_identity: Option<(String, String, String)> = conn
        .query_row(
            "SELECT tenant, name, version FROM dataset_refs
             WHERE manifest_sha256 = ?1 LIMIT 1",
            params![source.sha256],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()
        .map_err(|e| TrainError::other(format!("query dataset hash owner: {e}")))?;
    if let Some((owner, pinned_name, pinned_version)) = existing_identity {
        if owner != tenant_key {
            return Err(TrainError::other(format!(
                "dataset pin refused: manifest {} already belongs to another tenant",
                source.sha256
            )));
        }
        if pinned_name != name || pinned_version != version {
            return Err(TrainError::other(format!(
                "dataset pin refused: manifest {} is already dataset://{}@{} in tenant '{}'",
                source.sha256, pinned_name, pinned_version, tenant
            )));
        }
    }

    if let Some(existing) = query_resolution(conn, &tenant_key, name, version)? {
        if existing.dataset_id == source.id
            && existing.manifest_sha256 == source.sha256
            && existing.clinical == clinical
        {
            return Ok(existing);
        }
        return Err(TrainError::other(format!(
            "dataset pin refused: dataset://{name}@{version} is immutable and already points at {}",
            existing.manifest_sha256
        )));
    }

    match conn.execute(
        "INSERT INTO dataset_refs
            (tenant, name, version, dataset_id, manifest_sha256, clinical, pinned_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            tenant_key,
            name,
            version,
            source.id,
            source.sha256,
            i64::from(clinical),
            now_unix,
        ],
    ) {
        Ok(_) => query_resolution(conn, &tenant_key, name, version)?.ok_or_else(|| {
            TrainError::other("dataset pin succeeded but the binding could not be read back")
        }),
        Err(rusqlite::Error::SqliteFailure(e, _))
            if e.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            let winner = query_resolution(conn, &tenant_key, name, version)?;
            match winner {
                Some(existing)
                    if existing.dataset_id == source.id
                        && existing.manifest_sha256 == source.sha256
                        && existing.clinical == clinical =>
                {
                    Ok(existing)
                }
                _ => Err(TrainError::other(format!(
                    "dataset pin refused: concurrent rebind of immutable dataset://{name}@{version}"
                ))),
            }
        }
        Err(e) => Err(TrainError::other(format!("pin dataset: {e}"))),
    }
}

/// Resolve a URI in exactly one tenant. Restricted/clinical bindings are
/// node-local: every non-local launcher is refused before a path is returned.
/// The source file is re-hashed so a pinned name never silently follows changed
/// bytes.
pub fn resolve_uri(
    conn: &Connection,
    uri: &str,
    tenant: &Tenant,
    launch_target: LaunchTarget,
) -> Result<DatasetResolution> {
    ensure_schema(conn)?;
    let (name, version) = parse_dataset_uri(uri).ok_or_else(|| {
        TrainError::other(format!("not a `dataset://<name>@<version>` URI: {uri}"))
    })?;
    let tenant_key = tenant.to_string();
    let resolved = query_resolution(conn, &tenant_key, name, version)?.ok_or_else(|| {
        // Do not reveal whether another tenant owns the same logical handle.
        TrainError::other(format!(
            "unresolved dataset handle dataset://{name}@{version} for tenant '{tenant}'"
        ))
    })?;
    if resolved.clinical && launch_target != LaunchTarget::Local {
        return Err(TrainError::other(format!(
            "dataset resolve refused: Restricted dataset://{name}@{version} must stay node-local"
        )));
    }
    let live = datasets_db::compute_file_sha256(&resolved.source_path)?;
    if live != resolved.manifest_sha256 {
        return Err(TrainError::other(format!(
            "dataset resolve refused: source bytes for dataset://{name}@{version} no longer match pinned sha256 {}",
            resolved.manifest_sha256
        )));
    }
    Ok(resolved)
}

fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(CREATE_SCHEMA)
        .map_err(|e| TrainError::other(format!("create dataset registry schema: {e}")))
}

fn query_resolution(
    conn: &Connection,
    tenant: &str,
    name: &str,
    version: &str,
) -> Result<Option<DatasetResolution>> {
    conn.query_row(
        "SELECT r.tenant, r.name, r.version, r.dataset_id, d.name, d.source_path,
                r.manifest_sha256, d.kind, r.clinical, r.pinned_at
         FROM dataset_refs r
         JOIN datasets d ON d.id = r.dataset_id
         WHERE r.tenant = ?1 AND r.name = ?2 AND r.version = ?3",
        params![tenant, name, version],
        |row| {
            Ok(DatasetResolution {
                tenant: row.get(0)?,
                name: row.get(1)?,
                version: row.get(2)?,
                dataset_id: row.get(3)?,
                source_name: row.get(4)?,
                source_path: PathBuf::from(row.get::<_, String>(5)?),
                manifest_sha256: row.get(6)?,
                kind: row.get(7)?,
                clinical: row.get::<_, i64>(8)? != 0,
                pinned_at: row.get(9)?,
            })
        },
    )
    .optional()
    .map_err(|e| TrainError::other(format!("resolve dataset: {e}")))
}

fn classify_source(source: &DatasetRecord, tenant: &Tenant) -> Result<bool> {
    let meta: serde_json::Value = source
        .metadata
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|e| TrainError::other(format!("dataset metadata is not valid JSON: {e}")))?
        .unwrap_or(serde_json::Value::Null);
    let tenant_tag = match meta.get("tenant") {
        None => None,
        Some(serde_json::Value::String(tag)) => Some(tag.as_str()),
        Some(other) => {
            return Err(TrainError::other(format!(
                "dataset metadata field 'tenant' must be a string, got {other}"
            )));
        }
    };
    if let Some(tag) = tenant_tag {
        let tagged = Tenant::parse(tag).ok_or_else(|| {
            TrainError::other(format!("dataset metadata carries invalid tenant '{tag}'"))
        })?;
        if &tagged != tenant {
            return Err(TrainError::other(format!(
                "dataset pin refused: source tenant '{tagged}' does not match binding tenant '{tenant}'"
            )));
        }
    }
    let explicit_clinical = match meta.get("clinical") {
        None => false,
        Some(serde_json::Value::Bool(clinical)) => *clinical,
        Some(other) => {
            return Err(TrainError::other(format!(
                "dataset metadata field 'clinical' must be a boolean, got {other}"
            )));
        }
    };
    if explicit_clinical && !tenant.is_restricted() {
        return Err(TrainError::other(format!(
            "dataset pin refused: clinical source cannot enter non-Restricted tenant '{tenant}'"
        )));
    }
    Ok(explicit_clinical || tenant.is_restricted())
}

fn verify_source(source: &DatasetRecord) -> Result<()> {
    let live = datasets_db::compute_file_sha256(&source.source_path)?;
    if live != source.sha256 {
        return Err(TrainError::other(format!(
            "dataset pin refused: registered source '{}' drifted from sha256 {}",
            source.name, source.sha256
        )));
    }
    Ok(())
}

fn safe_ident(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        && !value.starts_with('.')
        && !value.starts_with('-')
        && !value.contains("..")
}
