// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Read-only dataset catalog (ADR 0100) — a PROJECTION over the datasets
//! registry (`datasets_db`) + the lineage DB, with a small persisted tags table
//! as the only authoritative state. No new storage, no hub, no blob copies.
//!
//! Each entry binds a dataset's `name[@version]` handle to its schema (modality,
//! sample rate, channels, dtype — read from the ABIR manifest via the dataset's
//! `metadata`, never re-derived), its artifact kind + content hash, its lineage
//! neighborhood (producing stage + downstream consumers, straight from
//! `lineage_db`), free-form tags, and a clinical flag. Because every field is
//! derived, the index rebuilds from scratch; tags persist across a rebuild.
//! Clinical/PHI entries (ADR 0061) are excluded from any cloud-surfaced view.

use rusqlite::{Connection, OpenFlags, params};
use serde::{Deserialize, Serialize};

use crate::datasets_db::DatasetRecord;
use crate::error::{Result, TrainError};
use crate::lineage_db::LineageDb;

/// A dataset's schema, read from its ABIR manifest (the dataset `metadata` JSON).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    pub modality: Option<String>,
    /// Sample rate (Hz). Read from `fs` or `sample_rate`.
    pub fs: Option<f64>,
    pub channels: Option<i64>,
    pub dtype: Option<String>,
}

impl Schema {
    /// Project a schema from a dataset's `metadata` JSON (the ABIR manifest
    /// projection). Missing/invalid JSON ⇒ an all-`None` schema (not an error;
    /// a dataset may predate schema capture).
    pub fn from_metadata(meta: Option<&str>) -> Schema {
        let v: serde_json::Value = meta
            .and_then(|m| serde_json::from_str(m).ok())
            .unwrap_or(serde_json::Value::Null);
        Schema {
            modality: v.get("modality").and_then(|x| x.as_str()).map(String::from),
            fs: v
                .get("fs")
                .and_then(|x| x.as_f64())
                .or_else(|| v.get("sample_rate").and_then(|x| x.as_f64())),
            channels: v.get("channels").and_then(|x| x.as_i64()),
            dtype: v.get("dtype").and_then(|x| x.as_str()).map(String::from),
        }
    }
}

/// One catalog entry — a projection (holds no authoritative state but tags).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub version: Option<String>,
    pub kind: String,
    pub hash: String,
    pub schema: Schema,
    /// Clinical/PHI (ADR 0061) — excluded from any cloud-surfaced view.
    pub clinical: bool,
    /// The stage that produced this artifact (lineage neighborhood).
    pub produced_by: Option<String>,
    /// Downstream consumer artifact hashes (lineage neighborhood).
    pub consumers: Vec<String>,
    pub ingest_unix: i64,
}

/// Project a catalog entry from a dataset record + the lineage DB. Schema comes
/// from the ABIR manifest (`metadata`); the lineage neighborhood from the
/// dataset's content hash.
pub fn project(record: &DatasetRecord, lineage: &LineageDb) -> Result<CatalogEntry> {
    let meta: serde_json::Value = record
        .metadata
        .as_deref()
        .and_then(|m| serde_json::from_str(m).ok())
        .unwrap_or(serde_json::Value::Null);
    // Clinical: a restricted tenant tag, or an explicit `clinical: true`.
    let clinical = meta
        .get("tenant")
        .and_then(|t| t.as_str())
        .and_then(crate::tenant::Tenant::parse)
        .map(|t| t.is_restricted())
        .unwrap_or(false)
        || meta
            .get("clinical")
            .and_then(|c| c.as_bool())
            .unwrap_or(false);
    let (name, version) = match record.name.split_once('@') {
        Some((n, v)) => (n.to_string(), Some(v.to_string())),
        None => (
            record.name.clone(),
            meta.get("version")
                .and_then(|x| x.as_str())
                .map(String::from),
        ),
    };
    Ok(CatalogEntry {
        name,
        version,
        kind: record.kind.clone(),
        hash: record.sha256.clone(),
        schema: Schema::from_metadata(record.metadata.as_deref()),
        clinical,
        produced_by: lineage.producing_stage(&record.sha256)?,
        consumers: lineage.consumers_of(&record.sha256)?,
        ingest_unix: record.created_at,
    })
}

/// A materialized entry must still AGREE with its live source (the dataset's
/// ABIR manifest). A drift — the source schema changed under a stale catalog row
/// — is a hard error (ADR 0100), never a served stale field.
pub fn verify_consistency(entry: &CatalogEntry, record: &DatasetRecord) -> Result<()> {
    let fresh = Schema::from_metadata(record.metadata.as_deref());
    if fresh != entry.schema {
        return Err(TrainError::other(format!(
            "catalog schema for '{}' disagrees with its source manifest (rebuild the catalog)",
            entry.name
        )));
    }
    Ok(())
}

// ── query grammar (`modality: fs: kind: tag: hash:`) ───────────────

/// A parsed catalog query — a conjunction of `key:value` terms.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CatalogQuery {
    pub modality: Option<String>,
    pub fs: Option<f64>,
    pub kind: Option<String>,
    pub tag: Option<String>,
    pub hash: Option<String>,
}

impl CatalogQuery {
    /// Parse `"modality:eeg fs:256 tag:sleep kind:x hash:ab"`. Fail-loud on an
    /// unknown key or a malformed term (fail-closed: a bad query never silently
    /// matches everything).
    pub fn parse(q: &str) -> std::result::Result<CatalogQuery, String> {
        let mut out = CatalogQuery::default();
        for term in q.split_whitespace() {
            let (k, v) = term
                .split_once(':')
                .ok_or_else(|| format!("bad term '{term}' (expected key:value)"))?;
            match k {
                "modality" => out.modality = Some(v.to_lowercase()),
                "fs" => {
                    out.fs = Some(
                        v.parse()
                            .map_err(|_| format!("fs must be numeric: '{v}'"))?,
                    )
                }
                "kind" => out.kind = Some(v.to_string()),
                "tag" => out.tag = Some(v.to_string()),
                "hash" => out.hash = Some(v.to_lowercase()),
                _ => {
                    return Err(format!(
                        "unknown query key '{k}' (modality/fs/kind/tag/hash)"
                    ));
                }
            }
        }
        Ok(out)
    }

    /// Whether an entry (with its resolved `tags`) satisfies EVERY term.
    pub fn matches(&self, e: &CatalogEntry, tags: &[String]) -> bool {
        if let Some(m) = &self.modality {
            if e.schema
                .modality
                .as_deref()
                .map(str::to_lowercase)
                .as_deref()
                != Some(m.as_str())
            {
                return false;
            }
        }
        if let Some(f) = self.fs {
            // bit-compare (no float-cmp lint); fs:256 matches a stored 256.0.
            if e.schema.fs.map(f64::to_bits) != Some(f.to_bits()) {
                return false;
            }
        }
        if let Some(k) = &self.kind {
            if &e.kind != k {
                return false;
            }
        }
        if let Some(h) = &self.hash {
            if !e.hash.to_lowercase().starts_with(h) {
                return false;
            }
        }
        if let Some(t) = &self.tag {
            if !tags.iter().any(|x| x == t) {
                return false;
            }
        }
        true
    }
}

/// Filter `entries` by `query`. `cloud_surfaced` = a view that may leave the box
/// (ADR 0061): clinical entries are excluded fail-closed. `tags_of` resolves an
/// entry name to its persisted tags.
pub fn search<'a>(
    entries: &'a [CatalogEntry],
    tags_of: impl Fn(&str) -> Vec<String>,
    query: &CatalogQuery,
    cloud_surfaced: bool,
) -> Vec<&'a CatalogEntry> {
    entries
        .iter()
        .filter(|e| {
            if cloud_surfaced && e.clinical {
                return false;
            }
            query.matches(e, &tags_of(&e.name))
        })
        .collect()
}

// ── tags (the only persisted, authoritative state) ─────────────────

const CREATE_TAGS: &str = "
CREATE TABLE IF NOT EXISTS catalog_tags (
    name TEXT NOT NULL,
    tag  TEXT NOT NULL,
    PRIMARY KEY (name, tag)
);";

/// Path to the catalog tags store (`~/.blut/catalog.db`; `$BLUT_CATALOG_DB`
/// override).
pub fn catalog_db_path() -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BLUT_CATALOG_DB") {
        return Ok(std::path::PathBuf::from(p));
    }
    let dir = dirs::home_dir()
        .ok_or_else(|| TrainError::other("home_dir() unavailable; set $BLUT_CATALOG_DB"))?
        .join(".blut");
    std::fs::create_dir_all(&dir).map_err(|e| TrainError::Io {
        path: dir.clone(),
        source: e,
    })?;
    Ok(dir.join("catalog.db"))
}

/// Project the whole catalog index from the datasets registry + lineage DB (the
/// `rebuild` projection). Entries are derived; only tags persist.
pub fn build_index(datasets: &Connection, lineage: &LineageDb) -> Result<Vec<CatalogEntry>> {
    crate::datasets_db::list(datasets)?
        .iter()
        .map(|r| project(r, lineage))
        .collect()
}

/// Open (create-if-absent) the catalog tags store.
pub fn open_tags(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )
    .map_err(|e| TrainError::other(format!("open {}: {e}", path.display())))?;
    conn.execute_batch(CREATE_TAGS)
        .map_err(|e| TrainError::other(format!("create catalog_tags: {e}")))?;
    Ok(conn)
}

/// Append a tag to a dataset (idempotent). Tags survive a catalog rebuild — they
/// are the catalog's only authoritative state.
pub fn add_tag(conn: &Connection, name: &str, tag: &str) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO catalog_tags (name, tag) VALUES (?1, ?2)",
        params![name, tag],
    )
    .map_err(|e| TrainError::other(format!("add tag: {e}")))?;
    Ok(())
}

/// The tags for a dataset name (sorted).
pub fn tags_for(conn: &Connection, name: &str) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare("SELECT tag FROM catalog_tags WHERE name = ?1 ORDER BY tag")
        .map_err(|e| TrainError::other(format!("tags_for prepare: {e}")))?;
    let rows = stmt
        .query_map(params![name], |r| r.get::<_, String>(0))
        .map_err(|e| TrainError::other(format!("tags_for query: {e}")))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| TrainError::other(format!("tags_for collect: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(name: &str, kind: &str, hash: &str, meta: serde_json::Value) -> DatasetRecord {
        DatasetRecord {
            id: name.to_string(),
            name: name.to_string(),
            kind: kind.to_string(),
            source_path: std::path::PathBuf::from("/x"),
            sha256: hash.to_string(),
            n_examples: 1,
            n_tokens: None,
            created_at: 1000,
            metadata: Some(meta.to_string()),
        }
    }

    fn eeg_meta(fs: f64) -> serde_json::Value {
        serde_json::json!({ "modality": "eeg", "fs": fs, "channels": 21, "dtype": "i16" })
    }

    fn lineage() -> (LineageDb, tempfile::TempDir) {
        let td = tempfile::tempdir().unwrap();
        let db = LineageDb::open_at(td.path().join("l.db")).unwrap();
        (db, td)
    }

    #[test]
    fn query_parse_is_fail_loud() {
        let q = CatalogQuery::parse("modality:eeg fs:256 tag:sleep").unwrap();
        assert_eq!(q.modality.as_deref(), Some("eeg"));
        assert_eq!(q.fs, Some(256.0));
        assert_eq!(q.tag.as_deref(), Some("sleep"));
        assert!(CatalogQuery::parse("bogus:x").is_err()); // unknown key
        assert!(CatalogQuery::parse("noKeyValue").is_err()); // malformed term
        assert!(CatalogQuery::parse("fs:notanum").is_err()); // bad number
    }

    #[test]
    fn search_returns_only_matching_entries() {
        let (db, _td) = lineage();
        let entries = vec![
            project(&rec("tuh@v3", "abir", "aa", eeg_meta(256.0)), &db).unwrap(),
            project(
                &rec(
                    "emg@v1",
                    "abir",
                    "bb",
                    serde_json::json!({"modality":"emg","fs":256.0}),
                ),
                &db,
            )
            .unwrap(),
            project(&rec("sleep@v1", "abir", "cc", eeg_meta(128.0)), &db).unwrap(),
        ];
        let q = CatalogQuery::parse("modality:eeg fs:256").unwrap();
        let hits = search(&entries, |_| vec![], &q, false);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "tuh");
        assert_eq!(hits[0].version.as_deref(), Some("v3"));
    }

    #[test]
    fn clinical_entry_absent_from_cloud_view() {
        let (db, _td) = lineage();
        let mut clin_meta = eeg_meta(256.0);
        clin_meta["tenant"] = serde_json::json!("clinical/prod");
        let entries = vec![
            project(&rec("tuh@v3", "abir", "aa", eeg_meta(256.0)), &db).unwrap(),
            project(&rec("phi@v1", "abir", "dd", clin_meta), &db).unwrap(),
        ];
        assert!(entries[1].clinical);
        let q = CatalogQuery::parse("modality:eeg").unwrap();
        // Local view sees both; a cloud-surfaced view excludes the clinical one.
        assert_eq!(search(&entries, |_| vec![], &q, false).len(), 2);
        let cloud = search(&entries, |_| vec![], &q, true);
        assert_eq!(cloud.len(), 1);
        assert!(cloud.iter().all(|e| !e.clinical));
    }

    #[test]
    fn tags_persist_and_filter() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_tags(&td.path().join("c.db")).unwrap();
        add_tag(&conn, "tuh", "sleep").unwrap();
        add_tag(&conn, "tuh", "sleep").unwrap(); // idempotent
        assert_eq!(tags_for(&conn, "tuh").unwrap(), vec!["sleep"]);

        let (db, _l) = lineage();
        let entries = vec![project(&rec("tuh@v3", "abir", "aa", eeg_meta(256.0)), &db).unwrap()];
        let q = CatalogQuery::parse("tag:sleep").unwrap();
        let tags_of = |n: &str| tags_for(&conn, n).unwrap();
        assert_eq!(search(&entries, tags_of, &q, false).len(), 1);
        let q2 = CatalogQuery::parse("tag:awake").unwrap();
        assert_eq!(
            search(&entries, |n: &str| tags_for(&conn, n).unwrap(), &q2, false).len(),
            0
        );
    }

    #[test]
    fn schema_drift_fails_consistency() {
        let (db, _td) = lineage();
        let entry = project(&rec("tuh@v3", "abir", "aa", eeg_meta(256.0)), &db).unwrap();
        // Same record ⇒ consistent.
        assert!(verify_consistency(&entry, &rec("tuh@v3", "abir", "aa", eeg_meta(256.0))).is_ok());
        // The source manifest changed (fs 256 → 512) under the stale entry ⇒ fail.
        assert!(verify_consistency(&entry, &rec("tuh@v3", "abir", "aa", eeg_meta(512.0))).is_err());
    }

    #[test]
    fn lineage_neighborhood_is_projected() {
        use crate::lineage_db::{ArtifactRow, EdgeRow};
        let (db, _td) = lineage();
        // tuh(aa) → produced by "ingest"; consumed by an "encode" output "bb".
        db.record_artifact(&ArtifactRow {
            job_id: "j".into(),
            stage_idx: 0,
            stage_name: "ingest".into(),
            content_hash: "aa".into(),
            kind: "abir".into(),
            schema_ver: 1,
            sidecar_path: None,
            produced_unix: Some(1),
        })
        .unwrap();
        db.record_edge(&EdgeRow {
            job_id: "j".into(),
            to_idx: 1,
            input_hash: "aa".into(),
            output_hash: "bb".into(),
        })
        .unwrap();
        let e = project(&rec("tuh@v3", "abir", "aa", eeg_meta(256.0)), &db).unwrap();
        assert_eq!(e.produced_by.as_deref(), Some("ingest"));
        assert_eq!(e.consumers, vec!["bb".to_string()]);
    }
}
