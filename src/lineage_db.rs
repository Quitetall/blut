//! LineageDB — an embedded, queryable index over BLUT's content-addressed runs.
//!
//! ARCHITECTURE (the load-bearing principle): this database is a **rebuildable
//! INDEX**, never the source of truth. The canonical record stays the
//! content-addressed filesystem — per-stage `output.metadata.json` sidecars,
//! the `status.jsonl` lifecycle stream, and (when wired) the trainer
//! `RunManifest`. A corrupt or deleted `lineage.db` is fully recoverable by
//! re-scanning those and re-ingesting (`reindex`). That keeps content-addressing
//! authoritative and the DB derived — unlike a server-DB-as-source-of-truth
//! orchestrator (Dagster/Postgres), a clinical-grade lineage record must be
//! reproducible offline from the bytes themselves.
//!
//! Engine choice: SQLite (already in-binary via `datasets_db`/`conversations`).
//! The query patterns are OLTP point-lookups (`trace <hash>`) + moderate
//! aggregates — SQLite's wheelhouse. WAL mode tolerates CLI readers while a run
//! writes; a single writer (the run loop) appends on Done. DuckDB-class columnar
//! OLAP would be overkill here and a heavy dep; if sweep-analytics over the
//! metric Parquet logs ever needs it, that is a SEPARATE read-only layer.
//!
//! Reliability: every ingest is an idempotent UPSERT, so re-ingesting a run (a
//! `reindex`) is a no-op — safe + repeatable. Schema versioned via
//! `PRAGMA user_version`. Callers treat write failures as fail-soft (warn, never
//! fail the run — the sidecar is canonical).

use std::collections::HashSet;
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

/// Bump when the schema changes in a non-additive way (forces a `reindex`).
const SCHEMA_VERSION: i64 = 1;

const CREATE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS runs (
    job_id             TEXT PRIMARY KEY,
    recipe             TEXT NOT NULL,
    config_fingerprint TEXT,
    git_sha            TEXT,
    started_unix       INTEGER,
    ended_unix         INTEGER,
    outcome            TEXT,
    host               TEXT,
    gpu_name           TEXT,
    ram_gib            INTEGER,
    vram_mib           INTEGER
);
CREATE TABLE IF NOT EXISTS artifacts (
    job_id        TEXT NOT NULL,
    stage_idx     INTEGER NOT NULL,
    stage_name    TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    kind          TEXT NOT NULL,
    schema_ver    INTEGER NOT NULL,
    sidecar_path  TEXT,
    produced_unix INTEGER,
    PRIMARY KEY (job_id, stage_idx)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(content_hash);
CREATE TABLE IF NOT EXISTS lineage_edges (
    job_id      TEXT NOT NULL,
    to_idx      INTEGER NOT NULL,
    input_hash  TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    PRIMARY KEY (job_id, to_idx, input_hash)
);
CREATE INDEX IF NOT EXISTS idx_edges_output ON lineage_edges(output_hash);
";

/// One run's provenance row.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRow {
    pub job_id: String,
    pub recipe: String,
    pub config_fingerprint: Option<String>,
    pub git_sha: Option<String>,
    pub started_unix: Option<i64>,
    pub ended_unix: Option<i64>,
    pub outcome: Option<String>,
    pub host: Option<String>,
    pub gpu_name: Option<String>,
    pub ram_gib: Option<i64>,
    pub vram_mib: Option<i64>,
}

/// One materialized artifact (a stage output), located by content hash.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRow {
    pub job_id: String,
    pub stage_idx: i64,
    pub stage_name: String,
    pub content_hash: String,
    pub kind: String,
    pub schema_ver: i64,
    pub sidecar_path: Option<String>,
    pub produced_unix: Option<i64>,
}

/// One stage's input→output content-hash edge (the lineage graph).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeRow {
    pub job_id: String,
    pub to_idx: i64,
    pub input_hash: String,
    pub output_hash: String,
}

/// One hop of an upstream trace: the artifact + the run that produced it.
#[derive(Clone, Debug, Serialize)]
pub struct TraceStep {
    pub artifact: ArtifactRow,
    pub run: Option<RunRow>,
}

/// The embedded lineage index.
pub struct LineageDb {
    conn: Connection,
}

impl LineageDb {
    /// Default location: `<data_dir>/lineage.db` (honors `$LAMU_TRAIN_DATA_DIR`).
    pub fn open() -> Result<Self> {
        let dir = crate::paths::data_dir()?;
        std::fs::create_dir_all(&dir)
            .map_err(|e| TrainError::other(format!("mkdir lineage data dir: {e}")))?;
        Self::open_at(dir.join("lineage.db"))
    }

    /// Open (creating + migrating) at an explicit path. Tests pin a tempdir.
    pub fn open_at(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .map_err(|e| TrainError::other(format!("open lineage.db: {e}")))?;
        // WAL: concurrent CLI readers while the run loop writes. busy_timeout:
        // briefly block rather than error on a transient writer lock.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| TrainError::other(format!("set WAL: {e}")))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| TrainError::other(format!("busy_timeout: {e}")))?;
        // Migration guard: read the stored version BEFORE stamping. A new file
        // reports 0; a NEWER db (written by a future binary) must error loudly,
        // not get silently re-stamped to this (older) version. Older-than-current
        // is where future migrations would run before the version bump.
        let existing: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .map_err(|e| TrainError::other(format!("read user_version: {e}")))?;
        if existing > SCHEMA_VERSION {
            return Err(TrainError::other(format!(
                "lineage.db schema v{existing} is newer than this binary (v{SCHEMA_VERSION}); \
                 upgrade blut or `blut lineage reindex` a fresh db"
            )));
        }
        conn.execute_batch(CREATE_SCHEMA)
            .map_err(|e| TrainError::other(format!("create lineage schema: {e}")))?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .map_err(|e| TrainError::other(format!("set user_version: {e}")))?;
        Ok(Self { conn })
    }

    /// Idempotent upsert of a run row (re-ingest = no-op-equivalent overwrite).
    pub fn record_run(&self, r: &RunRow) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO runs (job_id, recipe, config_fingerprint, git_sha,
                    started_unix, ended_unix, outcome, host, gpu_name, ram_gib, vram_mib)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
                 ON CONFLICT(job_id) DO UPDATE SET
                    recipe=COALESCE(excluded.recipe, runs.recipe),
                    config_fingerprint=COALESCE(excluded.config_fingerprint, runs.config_fingerprint),
                    git_sha=COALESCE(excluded.git_sha, runs.git_sha),
                    started_unix=COALESCE(excluded.started_unix, runs.started_unix),
                    ended_unix=COALESCE(excluded.ended_unix, runs.ended_unix),
                    outcome=COALESCE(excluded.outcome, runs.outcome),
                    host=COALESCE(excluded.host, runs.host),
                    gpu_name=COALESCE(excluded.gpu_name, runs.gpu_name),
                    ram_gib=COALESCE(excluded.ram_gib, runs.ram_gib),
                    vram_mib=COALESCE(excluded.vram_mib, runs.vram_mib)",
                params![
                    r.job_id, r.recipe, r.config_fingerprint, r.git_sha, r.started_unix,
                    r.ended_unix, r.outcome, r.host, r.gpu_name, r.ram_gib, r.vram_mib
                ],
            )
            .map_err(|e| TrainError::other(format!("record run {}: {e}", r.job_id)))?;
        Ok(())
    }

    /// Idempotent upsert of an artifact row (keyed by job_id+stage_idx).
    /// Always a FULL row (built from the sidecar), so `INSERT OR REPLACE` is a
    /// safe overwrite. `content_hash` is lowercased so `=`/`trace` lookups stay
    /// consistent with the case-insensitive `find_artifacts` LIKE.
    pub fn record_artifact(&self, a: &ArtifactRow) -> Result<()> {
        let content_hash = a.content_hash.to_lowercase();
        self.conn
            .execute(
                "INSERT OR REPLACE INTO artifacts
                    (job_id, stage_idx, stage_name, content_hash, kind, schema_ver,
                     sidecar_path, produced_unix)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![
                    a.job_id, a.stage_idx, a.stage_name, content_hash, a.kind,
                    a.schema_ver, a.sidecar_path, a.produced_unix
                ],
            )
            .map_err(|e| TrainError::other(format!("record artifact {}: {e}", a.content_hash)))?;
        Ok(())
    }

    /// Idempotent upsert of a lineage edge. Hashes lowercased so the upstream
    /// `trace` walk (which lowercases) matches `artifact_by_hash`'s `=` lookup.
    pub fn record_edge(&self, edge: &EdgeRow) -> Result<()> {
        let (input_hash, output_hash) =
            (edge.input_hash.to_lowercase(), edge.output_hash.to_lowercase());
        self.conn
            .execute(
                "INSERT OR REPLACE INTO lineage_edges (job_id, to_idx, input_hash, output_hash)
                 VALUES (?1,?2,?3,?4)",
                params![edge.job_id, edge.to_idx, input_hash, output_hash],
            )
            .map_err(|err| TrainError::other(format!("record edge {}: {err}", edge.job_id)))?;
        Ok(())
    }

    /// Artifacts whose content hash starts with `prefix` (the `trace` key).
    pub fn find_artifacts(&self, prefix: &str) -> Result<Vec<ArtifactRow>> {
        let like = format!("{}%", prefix.to_lowercase());
        let mut stmt = self
            .conn
            .prepare(
                "SELECT job_id, stage_idx, stage_name, content_hash, kind, schema_ver,
                        sidecar_path, produced_unix
                 FROM artifacts WHERE content_hash LIKE ?1 ORDER BY produced_unix DESC",
            )
            .map_err(|e| TrainError::other(format!("prepare find_artifacts: {e}")))?;
        let rows = stmt
            .query_map(params![like], row_to_artifact)
            .map_err(|e| TrainError::other(format!("query find_artifacts: {e}")))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("collect find_artifacts: {e}")))?;
        Ok(rows)
    }

    /// A run's provenance row by job id.
    pub fn get_run(&self, job_id: &str) -> Result<Option<RunRow>> {
        self.conn
            .query_row(
                "SELECT job_id, recipe, config_fingerprint, git_sha, started_unix, ended_unix,
                        outcome, host, gpu_name, ram_gib, vram_mib
                 FROM runs WHERE job_id = ?1",
                params![job_id],
                row_to_run,
            )
            .optional()
            .map_err(|e| TrainError::other(format!("get_run {job_id}: {e}")))
    }

    /// Walk the lineage UPSTREAM from a full content hash: this artifact, then
    /// the input that produced it, recursively, each annotated with its run's
    /// provenance. Cycle-guarded. The first step is the queried artifact.
    ///
    /// SINGLE-INPUT walk: at a fan-in (merge) stage with multiple input edges
    /// this follows ONE (the primary). BLUT plans are predominantly linear
    /// (corpus→train→gate); a full multi-branch `trace_all` is a future add.
    pub fn trace(&self, content_hash: &str) -> Result<Vec<TraceStep>> {
        let mut chain = Vec::new();
        let mut seen = HashSet::new();
        let mut current = content_hash.to_lowercase();
        while seen.insert(current.clone()) {
            // The (most recent) artifact carrying this content hash.
            let Some(artifact) = self.artifact_by_hash(&current)? else {
                break;
            };
            let run = self.get_run(&artifact.job_id)?;
            chain.push(TraceStep { artifact, run });
            // The input hash of the edge that produced `current`.
            match self.input_hash_for_output(&current)? {
                Some(input) => current = input,
                None => break,
            }
        }
        Ok(chain)
    }

    fn artifact_by_hash(&self, content_hash: &str) -> Result<Option<ArtifactRow>> {
        self.conn
            .query_row(
                "SELECT job_id, stage_idx, stage_name, content_hash, kind, schema_ver,
                        sidecar_path, produced_unix
                 FROM artifacts WHERE content_hash = ?1
                 ORDER BY produced_unix DESC LIMIT 1",
                params![content_hash],
                row_to_artifact,
            )
            .optional()
            .map_err(|e| TrainError::other(format!("artifact_by_hash: {e}")))
    }

    fn input_hash_for_output(&self, output_hash: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT input_hash FROM lineage_edges WHERE output_hash = ?1 LIMIT 1",
                params![output_hash],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| TrainError::other(format!("input_hash_for_output: {e}")))
    }

    /// Number of indexed runs (for `reindex` reporting + tests).
    pub fn run_count(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM runs", [], |r| r.get(0))
            .map_err(|e| TrainError::other(format!("run_count: {e}")))
    }
}

/// Ingest a COMPLETED job into the index: run provenance (git SHA, hardware,
/// outcome, timestamp) + its artifacts (from the sidecars) + its lineage edges
/// (from `status.jsonl`). Idempotent — re-ingesting (a `reindex`) is safe.
/// Reads the content-addressed filesystem; the DB is the derived index.
pub fn ingest_job(job_id: &str, recipe: &str, outcome: &str) -> Result<()> {
    let db = LineageDb::open()?;
    let snap = crate::broker::ResourceSnapshot::probe();
    db.record_run(&RunRow {
        job_id: job_id.to_string(),
        recipe: recipe.to_string(),
        config_fingerprint: None, // (sweep/config path supplies this when present)
        git_sha: git_head_sha(),
        started_unix: None,
        ended_unix: Some(chrono::Utc::now().timestamp()),
        outcome: Some(outcome.to_string()),
        host: read_hostname(),
        gpu_name: None,
        ram_gib: Some(snap.mem_total_gb as i64),
        vram_mib: snap.vram_total_mib.map(|v| v as i64),
    })?;
    for rec in crate::framework::lineage::scan_artifacts(job_id)?.into_iter() {
        db.record_artifact(&ArtifactRow {
            job_id: job_id.to_string(),
            stage_idx: stage_idx_of(&rec.sidecar_path) as i64,
            stage_name: rec.meta.produced_by_stage.clone().unwrap_or_default(),
            content_hash: rec.meta.content_hash.to_hex(),
            kind: rec.meta.kind.clone(),
            schema_ver: rec.meta.schema as i64,
            sidecar_path: Some(rec.sidecar_path.display().to_string()),
            produced_unix: Some(rec.meta.produced_at_unix_secs as i64),
        })?;
    }
    for node in crate::framework::lineage::job_lineage(job_id)?.into_iter() {
        if let (Some(input), Some(output)) = (node.input_hash, node.output_hash) {
            db.record_edge(&EdgeRow {
                job_id: job_id.to_string(),
                to_idx: node.node_idx as i64,
                input_hash: input,
                output_hash: output,
            })?;
        }
    }
    Ok(())
}

/// Best-effort `git rev-parse HEAD` of the working tree (None if not a repo).
fn git_head_sha() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!sha.is_empty()).then_some(sha)
}

/// Best-effort hostname for hardware provenance (None if unreadable).
fn read_hostname() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
}

/// Numeric stage index from a sidecar path whose parent dir is `<idx>-<name>`.
fn stage_idx_of(sidecar: &Path) -> u32 {
    sidecar
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .and_then(|n| n.split('-').next())
        .and_then(|d| d.parse::<u32>().ok())
        .unwrap_or(0)
}

fn row_to_run(row: &rusqlite::Row) -> rusqlite::Result<RunRow> {
    Ok(RunRow {
        job_id: row.get(0)?,
        recipe: row.get(1)?,
        config_fingerprint: row.get(2)?,
        git_sha: row.get(3)?,
        started_unix: row.get(4)?,
        ended_unix: row.get(5)?,
        outcome: row.get(6)?,
        host: row.get(7)?,
        gpu_name: row.get(8)?,
        ram_gib: row.get(9)?,
        vram_mib: row.get(10)?,
    })
}

fn row_to_artifact(row: &rusqlite::Row) -> rusqlite::Result<ArtifactRow> {
    Ok(ArtifactRow {
        job_id: row.get(0)?,
        stage_idx: row.get(1)?,
        stage_name: row.get(2)?,
        content_hash: row.get(3)?,
        kind: row.get(4)?,
        schema_ver: row.get(5)?,
        sidecar_path: row.get(6)?,
        produced_unix: row.get(7)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> LineageDb {
        let td = tempfile::tempdir().unwrap();
        // Leak the tempdir for the test's lifetime (db holds an open handle).
        let path = td.keep().join("lineage.db");
        LineageDb::open_at(path).unwrap()
    }

    fn art(job: &str, idx: i64, hash: &str) -> ArtifactRow {
        ArtifactRow {
            job_id: job.into(),
            stage_idx: idx,
            stage_name: format!("s{idx}"),
            content_hash: hash.into(),
            kind: "ckpt".into(),
            schema_ver: 1,
            sidecar_path: Some(format!("/j/{job}/stages/{idx}/output.metadata.json")),
            produced_unix: Some(1000 + idx),
        }
    }

    #[test]
    fn open_is_idempotent_and_migrates() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("lineage.db");
        let _ = LineageDb::open_at(&p).unwrap();
        // Re-open the same file: schema is CREATE IF NOT EXISTS → no error.
        let db = LineageDb::open_at(&p).unwrap();
        assert_eq!(db.run_count().unwrap(), 0);
    }

    #[test]
    fn record_and_query_run() {
        let db = db();
        let r = RunRow {
            job_id: "job-1".into(),
            recipe: "lamquant_joint_codec".into(),
            git_sha: Some("abc123".into()),
            outcome: Some("done".into()),
            ..Default::default()
        };
        db.record_run(&r).unwrap();
        assert_eq!(db.get_run("job-1").unwrap().unwrap().git_sha.as_deref(), Some("abc123"));
        assert!(db.get_run("nope").unwrap().is_none());
    }

    #[test]
    fn record_run_is_idempotent_upsert_preserving_provenance() {
        let db = db();
        db.record_run(&RunRow {
            job_id: "j".into(),
            recipe: "r".into(),
            git_sha: Some("sha".into()),
            ..Default::default()
        })
        .unwrap();
        // A later partial record (e.g. the Done stamp) must NOT clobber git_sha.
        db.record_run(&RunRow {
            job_id: "j".into(),
            recipe: "r".into(),
            ended_unix: Some(42),
            outcome: Some("done".into()),
            ..Default::default()
        })
        .unwrap();
        let got = db.get_run("j").unwrap().unwrap();
        assert_eq!(got.git_sha.as_deref(), Some("sha"), "COALESCE preserves prior provenance");
        assert_eq!(got.ended_unix, Some(42));
        assert_eq!(db.run_count().unwrap(), 1, "upsert, not duplicate");
    }

    #[test]
    fn find_artifacts_by_hash_prefix() {
        let db = db();
        db.record_artifact(&art("j", 0, "deadbeefcafe")).unwrap();
        db.record_artifact(&art("j", 1, "deadc0de0000")).unwrap();
        assert_eq!(db.find_artifacts("deadbeef").unwrap().len(), 1);
        assert_eq!(db.find_artifacts("dead").unwrap().len(), 2);
        assert_eq!(db.find_artifacts("ffff").unwrap().len(), 0);
        // Case-insensitive (to_hex is lowercase; tolerate an uppercase query).
        assert_eq!(db.find_artifacts("DEADBEEF").unwrap().len(), 1);
    }

    #[test]
    fn trace_walks_upstream_chain() {
        let db = db();
        // corpus(C) → train(T) → gate(G): three stages, two edges.
        db.record_run(&RunRow { job_id: "j".into(), recipe: "r".into(), ..Default::default() })
            .unwrap();
        db.record_artifact(&art("j", 0, "corpushash")).unwrap();
        db.record_artifact(&art("j", 1, "trainhash")).unwrap();
        db.record_artifact(&art("j", 2, "gatehash")).unwrap();
        db.record_edge(&EdgeRow { job_id: "j".into(), to_idx: 1, input_hash: "corpushash".into(), output_hash: "trainhash".into() }).unwrap();
        db.record_edge(&EdgeRow { job_id: "j".into(), to_idx: 2, input_hash: "trainhash".into(), output_hash: "gatehash".into() }).unwrap();

        let chain = db.trace("gatehash").unwrap();
        let hashes: Vec<&str> = chain.iter().map(|s| s.artifact.content_hash.as_str()).collect();
        assert_eq!(hashes, vec!["gatehash", "trainhash", "corpushash"], "upstream order");
        assert!(chain[0].run.is_some(), "trace annotates each hop with its run");
    }

    #[test]
    fn stage_idx_parsed_from_sidecar_path() {
        assert_eq!(stage_idx_of(Path::new("/j/stages/0-make/output.metadata.json")), 0);
        assert_eq!(stage_idx_of(Path::new("/j/stages/12-train_joint/output.metadata.json")), 12);
        assert_eq!(stage_idx_of(Path::new("/j/stages/garbage/output.metadata.json")), 0);
    }

    #[test]
    fn trace_is_cycle_guarded() {
        let db = db();
        db.record_artifact(&art("j", 0, "a")).unwrap();
        db.record_artifact(&art("j", 1, "b")).unwrap();
        // A pathological cycle a→b→a must terminate, not hang.
        db.record_edge(&EdgeRow { job_id: "j".into(), to_idx: 1, input_hash: "b".into(), output_hash: "a".into() }).unwrap();
        db.record_edge(&EdgeRow { job_id: "j".into(), to_idx: 0, input_hash: "a".into(), output_hash: "b".into() }).unwrap();
        let chain = db.trace("a").unwrap();
        assert!(chain.len() <= 2, "cycle guard bounds the walk");
    }
}
