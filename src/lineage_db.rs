// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
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

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

/// Bump when the schema changes in a non-additive way (forces a `reindex`).
/// v2 adds the additive `metrics` (E1) + `gauges` (E2) tables — they
/// materialize on existing v1 DBs via `CREATE TABLE IF NOT EXISTS`, so the
/// 1→2 bump needs NO data migration. v3 (ADR 0096) adds `runs.tenant`; v4
/// (ADR 0090) adds `runs.experiment` + `runs.args_json` so comparisons use an
/// explicit campaign key and report real parameter deltas instead of only an
/// opaque fingerprint. v5 (ADR 0101) adds the partition-status matrix index.
/// The open path uses idempotent column checks before each `ALTER TABLE`.
const SCHEMA_VERSION: i64 = 5;

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
    vram_mib           INTEGER,
    -- ADR 0096: the owning tenant. `default` = the flat/pre-tenancy namespace.
    -- A `clinical`/`restricted` tenant's rows are fail-closed excluded from any
    -- exported graph/card (ADR 0061/0099).
    tenant             TEXT NOT NULL DEFAULT 'default',
    -- Explicit experiment/campaign grouping. NULL legacy rows fall back to the
    -- recipe name in experiment queries.
    experiment         TEXT,
    -- ADR 0090: canonical recipe args persisted at <job>/args.json. Nullable
    -- for legacy/imported runs whose source file predates this index column.
    args_json          TEXT
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
-- PROVENANCE EDGE (data + code + parent, all captured — despite the bare
-- two-hash schema). A node's `input_hash` is the cache key's content digest:
-- it FOLDS its parents' `output_hash`es AND its data-source content hashes,
-- so a single `input_hash`→`output_hash` edge encodes BOTH the parent edge
-- (lineage) AND the data dependency. CODE provenance is carried separately,
-- per-run, on `runs.git_sha` (the cache key also busts on `code_sha`, surfaced
-- as `code_freshness`). So (data, code, parent) are each represented; only the
-- FIELD NAMING is terse — no extra columns are needed to recover any of them.
CREATE TABLE IF NOT EXISTS lineage_edges (
    job_id      TEXT NOT NULL,
    to_idx      INTEGER NOT NULL,
    input_hash  TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    PRIMARY KEY (job_id, to_idx, input_hash)
);
CREATE INDEX IF NOT EXISTS idx_edges_output ON lineage_edges(output_hash);
-- v2: the queryable metric store (E1). Folded from status.jsonl StageStep
-- events — a rebuildable INDEX, never the source of truth. `step = -1` is the
-- per-(job,node) FINAL value (the run's headline).
CREATE TABLE IF NOT EXISTS metrics (
    job_id    TEXT NOT NULL,
    node_idx  INTEGER NOT NULL,
    step      INTEGER NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL NOT NULL,
    wall_unix INTEGER,
    PRIMARY KEY (job_id, node_idx, step, metric)
);
CREATE INDEX IF NOT EXISTS idx_metrics_metric_val ON metrics(metric, value);
-- v2: system + GPU gauges sampled during a run (E2). `gpu_util` drives the
-- first-class GPU-saturation metric (gpu_saturation / gpu_wasted).
CREATE TABLE IF NOT EXISTS gauges (
    job_id            TEXT NOT NULL,
    node_idx          INTEGER NOT NULL,
    wall_unix         INTEGER NOT NULL,
    gpu_util          REAL,
    gpu_mem_mib       REAL,
    gpu_temp_c        REAL,
    gpu_power_w       REAL,
    host_ram_mib      REAL,
    host_disk_free_mib REAL,
    PRIMARY KEY (job_id, node_idx, wall_unix)
);
-- v5 / ADR 0101: latest materialization record per tenant/set/key. This is a
-- rebuildable index over the append-only partition status log + job lineage,
-- not a new source of truth.
CREATE TABLE IF NOT EXISTS partition_status (
    tenant             TEXT NOT NULL,
    recipe             TEXT NOT NULL,
    partition_set      TEXT NOT NULL,
    partition_key      TEXT NOT NULL,
    job_id              TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    resolved_args_json  TEXT NOT NULL,
    input_fingerprint   TEXT NOT NULL,
    recorded_unix       INTEGER NOT NULL,
    PRIMARY KEY (tenant, recipe, partition_set, partition_key)
);
CREATE INDEX IF NOT EXISTS idx_partition_status_job ON partition_status(job_id);
";

fn table_has_column(conn: &Connection, table: &str, expected: &str) -> Result<bool> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|e| TrainError::other(format!("table_info({table}): {e}")))?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| TrainError::other(format!("table_info rows: {e}")))?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| TrainError::other(format!("table_info collect: {e}")))?;
    Ok(columns.iter().any(|column| column == expected))
}

fn runs_has_column(conn: &Connection, expected: &str) -> Result<bool> {
    table_has_column(conn, "runs", expected)
}

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
    /// ADR 0096 owning tenant (`project[/domain]`). Empty ⇒ recorded as
    /// `default` (the flat namespace). A `clinical`/`restricted` tenant is
    /// fail-closed excluded from any export (ADR 0061/0099).
    #[serde(default)]
    pub tenant: String,
    /// Explicit experiment/campaign name. Legacy rows fall back to `recipe`.
    #[serde(default)]
    pub experiment: Option<String>,
    /// Canonical recipe args JSON. Missing on legacy/imported runs.
    #[serde(default)]
    pub args_json: Option<String>,
}

/// Latest indexed materialization for one partition cell (ADR 0101). The
/// append-only `<recipe>~<set>.status.jsonl` plus the referenced job remain the
/// canonical record; this row is the query-optimized projection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartitionStatusRow {
    pub tenant: String,
    pub recipe: String,
    pub partition_set: String,
    pub partition_key: String,
    pub job_id: String,
    pub outcome: String,
    pub resolved_args_json: String,
    pub input_fingerprint: String,
    pub recorded_unix: i64,
}

/// FRESHNESS verdict for a run's code (Phase G): did the code that built
/// this output drift from the current `HEAD`? `Stale` means a re-run would
/// re-execute (the cache key includes `code_sha`); `Unknown` means there
/// was no recorded git SHA or no git HEAD to compare against (not stale —
/// just unverifiable).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "freshness", rename_all = "snake_case")]
pub enum CodeFreshness {
    Fresh { git_sha: String },
    Stale { built_sha: String, head: String },
    Unknown,
}

/// Pure freshness verdict (testable without a real git HEAD): compare a
/// run's recorded git SHA against the current HEAD.
fn freshness_verdict(recorded: Option<String>, head: Option<String>) -> CodeFreshness {
    match (recorded, head) {
        (Some(r), Some(h)) if r == h => CodeFreshness::Fresh { git_sha: r },
        (Some(r), Some(h)) => CodeFreshness::Stale {
            built_sha: r,
            head: h,
        },
        // No recorded SHA, or no git HEAD to compare against.
        _ => CodeFreshness::Unknown,
    }
}

impl CodeFreshness {
    pub fn tag(&self) -> &'static str {
        match self {
            CodeFreshness::Fresh { .. } => "FRESH",
            CodeFreshness::Stale { .. } => "STALE",
            CodeFreshness::Unknown => "UNKNOWN",
        }
    }
    pub fn is_stale(&self) -> bool {
        matches!(self, CodeFreshness::Stale { .. })
    }
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

/// One metric sample (E1). `step = -1` marks the per-(job,node) FINAL value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MetricRow {
    pub job_id: String,
    pub node_idx: i64,
    pub step: i64,
    pub metric: String,
    pub value: f64,
    pub wall_unix: Option<i64>,
}

/// One system/GPU gauge sample (E2). `gpu_util` is the first-class saturation
/// signal.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GaugeRow {
    pub job_id: String,
    pub node_idx: i64,
    pub wall_unix: i64,
    pub gpu_util: Option<f64>,
    pub gpu_mem_mib: Option<f64>,
    pub gpu_temp_c: Option<f64>,
    pub gpu_power_w: Option<f64>,
    pub host_ram_mib: Option<f64>,
    pub host_disk_free_mib: Option<f64>,
}

/// The first-class GPU-saturation summary for a run (owner directive): the mean
/// utilization over its samples + the fraction of samples below the
/// "wasted" floor. `gpu_wasted ≈ 0` is the optimization target.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GpuSaturation {
    pub samples: usize,
    /// Mean `gpu_util` (%) over the run's gauge samples.
    pub saturation: f64,
    /// Fraction of samples with `gpu_util` below the floor — wasted GPU.
    pub wasted: f64,
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
        // v2→v3 (ADR 0096): add `runs.tenant` to a pre-existing v2 db. CREATE
        // TABLE IF NOT EXISTS can't alter an extant table, so ADD COLUMN here.
        // Probe `PRAGMA table_info` for the column rather than string-matching the
        // "duplicate column name" error (locale/SQLite-version fragile).
        let has_tenant = runs_has_column(&conn, "tenant")?;
        if !has_tenant {
            conn.execute(
                "ALTER TABLE runs ADD COLUMN tenant TEXT NOT NULL DEFAULT 'default'",
                [],
            )
            .map_err(|e| TrainError::other(format!("migrate runs.tenant: {e}")))?;
        }
        // v3→v4 (ADR 0090): preserve canonical recipe args for experiment
        // comparisons. Legacy rows remain NULL and compare as "unknown".
        let has_args_json = runs_has_column(&conn, "args_json")?;
        if !has_args_json {
            conn.execute("ALTER TABLE runs ADD COLUMN args_json TEXT", [])
                .map_err(|e| TrainError::other(format!("migrate runs.args_json: {e}")))?;
        }
        if !runs_has_column(&conn, "experiment")? {
            conn.execute("ALTER TABLE runs ADD COLUMN experiment TEXT", [])
                .map_err(|e| TrainError::other(format!("migrate runs.experiment: {e}")))?;
        }
        // Pre-landing v5 development databases may have the additive table
        // without the later input fingerprint. Empty means unverifiable and is
        // therefore surfaced as stale until rematerialized.
        if !table_has_column(&conn, "partition_status", "input_fingerprint")? {
            conn.execute(
                "ALTER TABLE partition_status ADD COLUMN input_fingerprint TEXT NOT NULL DEFAULT ''",
                [],
            )
            .map_err(|e| {
                TrainError::other(format!("migrate partition_status.input_fingerprint: {e}"))
            })?;
        }
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .map_err(|e| TrainError::other(format!("set user_version: {e}")))?;
        Ok(Self { conn })
    }

    /// Idempotent upsert of a run row (re-ingest = no-op-equivalent overwrite).
    pub fn record_run(&self, r: &RunRow) -> Result<()> {
        let incoming_tenant = if r.tenant.is_empty() {
            "default"
        } else {
            r.tenant.as_str()
        };
        if r.experiment
            .as_deref()
            .is_some_and(|name| !crate::experiment_registry::is_safe_experiment_name(name))
        {
            return Err(TrainError::other(format!(
                "record run {}: invalid experiment name",
                r.job_id
            )));
        }
        let incoming_args: Option<serde_json::Value> = r
            .args_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| {
                TrainError::other(format!(
                    "record run {}: incoming args_json is malformed: {e}",
                    r.job_id
                ))
            })?;
        if let Some(existing) = self.get_run(&r.job_id)? {
            let existing_tenant = if existing.tenant.is_empty() {
                "default"
            } else {
                existing.tenant.as_str()
            };
            if existing_tenant != incoming_tenant {
                return Err(TrainError::other(format!(
                    "record run {} refused: tenant is immutable ('{}' != '{}')",
                    r.job_id, existing_tenant, incoming_tenant
                )));
            }
            if existing.recipe != r.recipe {
                return Err(TrainError::other(format!(
                    "record run {} refused: recipe is immutable ('{}' != '{}')",
                    r.job_id, existing.recipe, r.recipe
                )));
            }
            if let (Some(current), Some(incoming)) = (&existing.experiment, &r.experiment)
                && current != incoming
            {
                return Err(TrainError::other(format!(
                    "record run {} refused: experiment is immutable ('{}' != '{}')",
                    r.job_id, current, incoming
                )));
            }
            if let (Some(current), Some(incoming)) = (&existing.args_json, &incoming_args) {
                let current: serde_json::Value = serde_json::from_str(current).map_err(|e| {
                    TrainError::other(format!(
                        "record run {}: stored args_json is malformed: {e}",
                        r.job_id
                    ))
                })?;
                if &current != incoming {
                    return Err(TrainError::other(format!(
                        "record run {} refused: canonical recipe args are immutable",
                        r.job_id
                    )));
                }
            }
        }
        self.conn
            .execute(
                "INSERT INTO runs (job_id, recipe, config_fingerprint, git_sha,
                    started_unix, ended_unix, outcome, host, gpu_name, ram_gib, vram_mib,
                    tenant, experiment, args_json)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)
                 ON CONFLICT(job_id) DO UPDATE SET
                    recipe=runs.recipe,
                    config_fingerprint=COALESCE(excluded.config_fingerprint, runs.config_fingerprint),
                    git_sha=COALESCE(excluded.git_sha, runs.git_sha),
                    started_unix=COALESCE(excluded.started_unix, runs.started_unix),
                    ended_unix=COALESCE(excluded.ended_unix, runs.ended_unix),
                    outcome=COALESCE(excluded.outcome, runs.outcome),
                    host=COALESCE(excluded.host, runs.host),
                    gpu_name=COALESCE(excluded.gpu_name, runs.gpu_name),
                    ram_gib=COALESCE(excluded.ram_gib, runs.ram_gib),
                    vram_mib=COALESCE(excluded.vram_mib, runs.vram_mib),
                    tenant=runs.tenant,
                    experiment=COALESCE(runs.experiment, excluded.experiment),
                    args_json=COALESCE(runs.args_json, excluded.args_json)",
                params![
                    r.job_id, r.recipe, r.config_fingerprint, r.git_sha, r.started_unix,
                    r.ended_unix, r.outcome, r.host, r.gpu_name, r.ram_gib, r.vram_mib,
                    incoming_tenant,
                    r.experiment,
                    r.args_json,
                ],
            )
            .map_err(|e| TrainError::other(format!("record run {}: {e}", r.job_id)))?;
        Ok(())
    }

    /// Upsert the latest derived partition materialization. Older concurrent
    /// completions cannot overwrite a newer row.
    pub fn record_partition_status(&self, row: &PartitionStatusRow) -> Result<()> {
        let args: serde_json::Value =
            serde_json::from_str(&row.resolved_args_json).map_err(|e| {
                TrainError::other(format!(
                    "record partition status {}/{} {}: malformed resolved args: {e}",
                    row.recipe, row.partition_set, row.partition_key
                ))
            })?;
        let canonical_args = serde_json::to_string(&args)
            .map_err(|e| TrainError::other(format!("canonicalize partition args: {e}")))?;
        self.conn
            .execute(
                "INSERT INTO partition_status
                    (tenant, recipe, partition_set, partition_key, job_id, outcome,
                     resolved_args_json, input_fingerprint, recorded_unix)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
                 ON CONFLICT(tenant, recipe, partition_set, partition_key) DO UPDATE SET
                    job_id=excluded.job_id,
                    outcome=excluded.outcome,
                    resolved_args_json=excluded.resolved_args_json,
                    input_fingerprint=excluded.input_fingerprint,
                    recorded_unix=excluded.recorded_unix
                 WHERE excluded.recorded_unix >= partition_status.recorded_unix",
                params![
                    row.tenant,
                    row.recipe,
                    row.partition_set,
                    row.partition_key,
                    row.job_id,
                    row.outcome,
                    canonical_args,
                    row.input_fingerprint,
                    row.recorded_unix,
                ],
            )
            .map_err(|e| TrainError::other(format!("record partition status: {e}")))?;
        Ok(())
    }

    /// Replace one disposable projection row from the canonical append-only
    /// partition log. Unlike [`Self::record_partition_status`], this does not
    /// compare timestamps: an index row can be corrupt or clock-skewed into
    /// the future, but it must never outrank canonical JSONL history.
    pub(crate) fn reconcile_partition_status(&self, row: &PartitionStatusRow) -> Result<()> {
        let args: serde_json::Value =
            serde_json::from_str(&row.resolved_args_json).map_err(|e| {
                TrainError::other(format!(
                    "reconcile partition status {}/{} {}: malformed resolved args: {e}",
                    row.recipe, row.partition_set, row.partition_key
                ))
            })?;
        let canonical_args = serde_json::to_string(&args)
            .map_err(|e| TrainError::other(format!("canonicalize partition args: {e}")))?;
        self.conn
            .execute(
                "INSERT INTO partition_status
                    (tenant, recipe, partition_set, partition_key, job_id, outcome,
                     resolved_args_json, input_fingerprint, recorded_unix)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)
                 ON CONFLICT(tenant, recipe, partition_set, partition_key) DO UPDATE SET
                    job_id=excluded.job_id,
                    outcome=excluded.outcome,
                    resolved_args_json=excluded.resolved_args_json,
                    input_fingerprint=excluded.input_fingerprint,
                    recorded_unix=excluded.recorded_unix",
                params![
                    row.tenant,
                    row.recipe,
                    row.partition_set,
                    row.partition_key,
                    row.job_id,
                    row.outcome,
                    canonical_args,
                    row.input_fingerprint,
                    row.recorded_unix,
                ],
            )
            .map_err(|e| TrainError::other(format!("reconcile partition status: {e}")))?;
        Ok(())
    }

    /// Latest indexed rows for exactly one tenant and named partition set.
    pub fn partition_statuses(
        &self,
        tenant: &str,
        recipe: &str,
        partition_set: &str,
    ) -> Result<std::collections::BTreeMap<String, PartitionStatusRow>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT tenant, recipe, partition_set, partition_key, job_id, outcome,
                        resolved_args_json, input_fingerprint, recorded_unix
                 FROM partition_status
                 WHERE tenant=?1 AND recipe=?2 AND partition_set=?3
                 ORDER BY partition_key",
            )
            .map_err(|e| TrainError::other(format!("prepare partition statuses: {e}")))?;
        let rows = stmt
            .query_map(params![tenant, recipe, partition_set], |row| {
                Ok(PartitionStatusRow {
                    tenant: row.get(0)?,
                    recipe: row.get(1)?,
                    partition_set: row.get(2)?,
                    partition_key: row.get(3)?,
                    job_id: row.get(4)?,
                    outcome: row.get(5)?,
                    resolved_args_json: row.get(6)?,
                    input_fingerprint: row.get(7)?,
                    recorded_unix: row.get(8)?,
                })
            })
            .map_err(|e| TrainError::other(format!("query partition statuses: {e}")))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("collect partition statuses: {e}")))
            .map(|rows| {
                rows.into_iter()
                    .map(|row| (row.partition_key.clone(), row))
                    .collect()
            })
    }

    /// Derive the operator-visible matrix state from the indexed cell record,
    /// its owning run, and live content-addressed output. Any missing or
    /// contradictory lineage is stale, never silently materialized.
    pub fn derive_partition_status(
        &self,
        row: Option<&PartitionStatusRow>,
        tenant: &str,
        recipe: &str,
        current_resolved_args: &serde_json::Value,
        current_input_fingerprint: &str,
        restricted: bool,
    ) -> Result<crate::config::partition::CellStatus> {
        use crate::config::partition::CellStatus;
        if restricted {
            return Ok(CellStatus::Restricted);
        }
        let Some(row) = row else {
            return Ok(CellStatus::Missing);
        };
        if row.outcome != "done" {
            return Ok(CellStatus::Failed);
        }
        let recorded_args: serde_json::Value = match serde_json::from_str(&row.resolved_args_json) {
            Ok(args) => args,
            Err(_) => return Ok(CellStatus::Stale),
        };
        if row.tenant != tenant || row.recipe != recipe || recorded_args != *current_resolved_args {
            return Ok(CellStatus::Stale);
        }
        if row.input_fingerprint != current_input_fingerprint {
            return Ok(CellStatus::Stale);
        }
        let Some(run) = self.get_run(&row.job_id)? else {
            return Ok(CellStatus::Stale);
        };
        let run_args = run
            .args_json
            .as_deref()
            .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok());
        if run.tenant != tenant
            || run.recipe != recipe
            || run.outcome.as_deref() != Some("done")
            || run_args.as_ref() != Some(current_resolved_args)
        {
            return Ok(CellStatus::Stale);
        }
        if matches!(
            self.code_freshness(&row.job_id)?,
            CodeFreshness::Stale { .. }
        ) {
            return Ok(CellStatus::Stale);
        }
        let live = self
            .terminal_artifact(&row.job_id)?
            .is_some_and(|artifact| {
                let Some(path) = artifact.sidecar_path else {
                    return false;
                };
                let path = Path::new(&path);
                let metadata_live = crate::framework::ArtifactMetadata::read_from(path)
                    .map(|metadata| metadata.content_hash.to_hex() == artifact.content_hash)
                    .unwrap_or(false);
                let cache_live = path
                    .parent()
                    .and_then(|parent| {
                        crate::framework::cache::CacheProof::read_from(
                            &parent.join("cache-proof.json"),
                        )
                        .ok()
                    })
                    .is_some_and(|proof| proof.is_live());
                metadata_live && cache_live
            });
        Ok(if live {
            CellStatus::Materialized
        } else {
            CellStatus::Stale
        })
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
                    a.job_id,
                    a.stage_idx,
                    a.stage_name,
                    content_hash,
                    a.kind,
                    a.schema_ver,
                    a.sidecar_path,
                    a.produced_unix
                ],
            )
            .map_err(|e| TrainError::other(format!("record artifact {}: {e}", a.content_hash)))?;
        Ok(())
    }

    /// Idempotent upsert of a lineage edge. Hashes lowercased so the upstream
    /// `trace` walk (which lowercases) matches `artifact_by_hash`'s `=` lookup.
    /// The `input_hash` is a fold of (parent `output_hash`es + data-source
    /// content hashes), so this two-hash edge encodes BOTH the parent edge and
    /// the data dependency; code provenance lives on `runs.git_sha`. See the
    /// `lineage_edges` CREATE comment for the full (data, code, parent) mapping.
    pub fn record_edge(&self, edge: &EdgeRow) -> Result<()> {
        let (input_hash, output_hash) = (
            edge.input_hash.to_lowercase(),
            edge.output_hash.to_lowercase(),
        );
        self.conn
            .execute(
                "INSERT OR REPLACE INTO lineage_edges (job_id, to_idx, input_hash, output_hash)
                 VALUES (?1,?2,?3,?4)",
                params![edge.job_id, edge.to_idx, input_hash, output_hash],
            )
            .map_err(|err| TrainError::other(format!("record edge {}: {err}", edge.job_id)))?;
        Ok(())
    }

    /// Idempotent bulk upsert of metric samples (E1) in one transaction.
    pub fn record_metrics(&self, rows: &[MetricRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| TrainError::other(format!("metrics tx: {e}")))?;
        for m in rows {
            tx.execute(
                "INSERT OR REPLACE INTO metrics (job_id, node_idx, step, metric, value, wall_unix)
                 VALUES (?1,?2,?3,?4,?5,?6)",
                params![m.job_id, m.node_idx, m.step, m.metric, m.value, m.wall_unix],
            )
            .map_err(|e| TrainError::other(format!("record metric {}: {e}", m.metric)))?;
        }
        tx.commit()
            .map_err(|e| TrainError::other(format!("metrics commit: {e}")))?;
        Ok(())
    }

    /// Idempotent bulk upsert of gauge samples (E2) in one transaction.
    pub fn record_gauges(&self, rows: &[GaugeRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| TrainError::other(format!("gauges tx: {e}")))?;
        for g in rows {
            tx.execute(
                "INSERT OR REPLACE INTO gauges
                    (job_id, node_idx, wall_unix, gpu_util, gpu_mem_mib, gpu_temp_c,
                     gpu_power_w, host_ram_mib, host_disk_free_mib)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![
                    g.job_id,
                    g.node_idx,
                    g.wall_unix,
                    g.gpu_util,
                    g.gpu_mem_mib,
                    g.gpu_temp_c,
                    g.gpu_power_w,
                    g.host_ram_mib,
                    g.host_disk_free_mib
                ],
            )
            .map_err(|e| TrainError::other(format!("record gauge {}: {e}", g.job_id)))?;
        }
        tx.commit()
            .map_err(|e| TrainError::other(format!("gauges commit: {e}")))?;
        Ok(())
    }

    /// The FINAL value of `metric` for a job (max `step`), across all its nodes —
    /// the headline number `blut compare` shows. `None` if the metric was never
    /// recorded for the job.
    pub fn final_metric(&self, job_id: &str, metric: &str) -> Result<Option<f64>> {
        self.conn
            .query_row(
                // `step = -1` is the explicit FINAL marker (NOT the max step — a
                // metric can collapse after its peak, so the largest step value
                // is not the final one).
                "SELECT value FROM metrics WHERE job_id=?1 AND metric=?2 AND step=-1 LIMIT 1",
                params![job_id, metric],
                |r| r.get::<_, f64>(0),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(TrainError::other(format!("final_metric: {other}"))),
            })
    }

    /// Every metric's FINAL value for a job (`step = -1`), across its nodes —
    /// the `(metric, value)` panel `blut compare` shows. Aggregated MAX per
    /// metric so a multi-node job reports one headline per metric.
    pub fn final_metrics(&self, job_id: &str) -> Result<Vec<(String, f64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT metric, MAX(value) FROM metrics WHERE job_id=?1 AND step=-1
                 GROUP BY metric ORDER BY metric",
            )
            .map_err(|e| TrainError::other(format!("final_metrics prepare: {e}")))?;
        let rows = stmt
            .query_map(params![job_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
            })
            .map_err(|e| TrainError::other(format!("final_metrics query: {e}")))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| TrainError::other(format!("final_metrics collect: {e}")))
    }

    /// The per-step TRAJECTORY of `metric` for a job (real samples, `step >= 0`),
    /// ordered by step — for `blut results` (ADR 0071 A3). Excludes the `step = -1`
    /// final marker so the series is the live curve, not the headline.
    ///
    /// Job-level aggregate (no `node_idx` filter), consistent with `final_metrics`:
    /// a single train node emits the headline (val_r), so this is the curve as
    /// the run reports it. A future multi-node-per-metric layout would interleave
    /// samples — add a `node_idx` clause then.
    pub fn metric_series(&self, job_id: &str, metric: &str) -> Result<Vec<(i64, f64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT step, value FROM metrics
                 WHERE job_id=?1 AND metric=?2 AND step>=0 ORDER BY step",
            )
            .map_err(|e| TrainError::other(format!("metric_series prepare: {e}")))?;
        let rows = stmt
            .query_map(params![job_id, metric], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })
            .map_err(|e| TrainError::other(format!("metric_series query: {e}")))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| TrainError::other(format!("metric_series collect: {e}")))
    }

    /// The BEST (peak) value of `metric` over the real trajectory (`step >= 0`) —
    /// the best-EVER, distinct from `final_metric`'s ended value (which can collapse
    /// after the peak). `maximize` picks MAX else MIN; ties → earliest step. `None`
    /// if the metric was never sampled. Job-level aggregate (no `node_idx` filter),
    /// consistent with `metric_series` / `final_metrics`.
    pub fn best_metric(
        &self,
        job_id: &str,
        metric: &str,
        maximize: bool,
    ) -> Result<Option<(i64, f64)>> {
        let sql = if maximize {
            "SELECT step, value FROM metrics
             WHERE job_id=?1 AND metric=?2 AND step>=0 ORDER BY value DESC, step ASC LIMIT 1"
        } else {
            "SELECT step, value FROM metrics
             WHERE job_id=?1 AND metric=?2 AND step>=0 ORDER BY value ASC, step ASC LIMIT 1"
        };
        self.conn
            .query_row(sql, params![job_id, metric], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(TrainError::other(format!("best_metric: {other}"))),
            })
    }

    /// The TERMINAL artifact (highest `stage_idx`) for a job — its `sidecar_path`
    /// holds the produced ckpt's location. `None` if no artifacts were recorded.
    pub fn terminal_artifact(&self, job_id: &str) -> Result<Option<ArtifactRow>> {
        self.conn
            .query_row(
                "SELECT job_id, stage_idx, stage_name, content_hash, kind, schema_ver,
                        sidecar_path, produced_unix
                 FROM artifacts WHERE job_id=?1 ORDER BY stage_idx DESC LIMIT 1",
                params![job_id],
                |r| {
                    Ok(ArtifactRow {
                        job_id: r.get(0)?,
                        stage_idx: r.get(1)?,
                        stage_name: r.get(2)?,
                        content_hash: r.get(3)?,
                        kind: r.get(4)?,
                        schema_ver: r.get(5)?,
                        sidecar_path: r.get(6)?,
                        produced_unix: r.get(7)?,
                    })
                },
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(TrainError::other(format!("terminal_artifact: {other}"))),
            })
    }

    /// Top runs by their FINAL `metric` value (HPO ranking / leaderboard) — the
    /// `step = -1` row per (job, node), so an overfit run that peaked then
    /// collapsed ranks by where it ENDED, not its best-ever intermediate.
    /// Ordered `DESC` when `maximize`, else `ASC`. Returns `(job_id, node_idx,
    /// final_value)` best-first, capped at `limit`.
    pub fn top_runs_by_metric(
        &self,
        metric: &str,
        maximize: bool,
        limit: usize,
    ) -> Result<Vec<(String, i64, f64)>> {
        // Static SQL per direction (no `format!` into a query) — clearer + leaves
        // no injection-shaped pattern to copy.
        let sql = if maximize {
            "SELECT job_id, node_idx, value FROM metrics
             WHERE metric=?1 AND step=-1 ORDER BY value DESC LIMIT ?2"
        } else {
            "SELECT job_id, node_idx, value FROM metrics
             WHERE metric=?1 AND step=-1 ORDER BY value ASC LIMIT ?2"
        };
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| TrainError::other(format!("top_runs_by_metric prepare: {e}")))?;
        let rows = stmt
            .query_map(params![metric, limit as i64], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, f64>(2)?,
                ))
            })
            .map_err(|e| TrainError::other(format!("top_runs_by_metric query: {e}")))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| TrainError::other(format!("top_runs_by_metric collect: {e}")))
    }

    /// The first-class GPU-saturation summary for a job (owner directive): mean
    /// `gpu_util` + the fraction of samples below `util_floor` (wasted GPU).
    /// `None` if the run recorded no gauge samples with a `gpu_util`.
    pub fn gpu_saturation(&self, job_id: &str, util_floor: f64) -> Result<Option<GpuSaturation>> {
        let row = self
            .conn
            .query_row(
                "SELECT COUNT(*), AVG(gpu_util),
                        AVG(CASE WHEN gpu_util < ?2 THEN 1.0 ELSE 0.0 END)
                 FROM gauges WHERE job_id=?1 AND gpu_util IS NOT NULL",
                params![job_id, util_floor],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<f64>>(1)?,
                        r.get::<_, Option<f64>>(2)?,
                    ))
                },
            )
            .map_err(|e| TrainError::other(format!("gpu_saturation: {e}")))?;
        match row {
            (n, Some(mean), Some(wasted)) if n > 0 => Ok(Some(GpuSaturation {
                samples: n as usize,
                saturation: mean,
                wasted,
            })),
            _ => Ok(None),
        }
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

    /// FRESHNESS (Phase G): is a job's output still current, or was it
    /// built with code that has since drifted? The cache key already busts
    /// on `code_sha` + `input_hash` (so a re-run RE-EXECUTES a stale node);
    /// this just SURFACES that verdict without re-running — by comparing the
    /// git SHA recorded for the run against the current `HEAD`.
    pub fn code_freshness(&self, job_id: &str) -> Result<CodeFreshness> {
        let recorded = self.get_run(job_id)?.and_then(|r| r.git_sha);
        Ok(freshness_verdict(recorded, git_head_sha()))
    }

    /// A run's provenance row by job id.
    pub fn get_run(&self, job_id: &str) -> Result<Option<RunRow>> {
        self.conn
            .query_row(
                "SELECT job_id, recipe, config_fingerprint, git_sha, started_unix, ended_unix,
                        outcome, host, gpu_name, ram_gib, vram_mib, tenant, experiment, args_json
                 FROM runs WHERE job_id = ?1",
                params![job_id],
                row_to_run,
            )
            .optional()
            .map_err(|e| TrainError::other(format!("get_run {job_id}: {e}")))
    }

    /// All recorded runs (newest started first). Used by `blut sla check` (ADR
    /// 0094) to evaluate SLA rules over every run's timing — a pure read.
    pub fn all_runs(&self) -> Result<Vec<RunRow>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT job_id, recipe, config_fingerprint, git_sha, started_unix, ended_unix,
                        outcome, host, gpu_name, ram_gib, vram_mib, tenant, experiment, args_json
                 FROM runs ORDER BY started_unix DESC",
            )
            .map_err(|e| TrainError::other(format!("all_runs prepare: {e}")))?;
        let rows = stmt
            .query_map([], row_to_run)
            .map_err(|e| TrainError::other(format!("all_runs query: {e}")))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("all_runs collect: {e}")))
    }

    /// Newest runs for one explicit experiment in exactly one tenant. Legacy
    /// rows with no marker use their recipe name as the experiment key.
    /// The tenant predicate is part of the SQL, not a caller-side filter, so a
    /// restricted run cannot enter another tenant's comparison candidate set.
    pub fn runs_for_experiment_tenant(
        &self,
        experiment: &str,
        tenant: &str,
        limit: usize,
    ) -> Result<Vec<RunRow>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT job_id, recipe, config_fingerprint, git_sha, started_unix, ended_unix,
                        outcome, host, gpu_name, ram_gib, vram_mib, tenant, experiment, args_json
                 FROM runs
                 WHERE COALESCE(experiment, recipe) = ?1 AND tenant = ?2
                 ORDER BY COALESCE(started_unix, 0) DESC, job_id DESC
                 LIMIT ?3",
            )
            .map_err(|e| TrainError::other(format!("prepare experiment runs: {e}")))?;
        stmt.query_map(params![experiment, tenant, limit as i64], row_to_run)
            .map_err(|e| TrainError::other(format!("query experiment runs: {e}")))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("collect experiment runs: {e}")))
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
            let run = self.get_run(&artifact.job_id)?.ok_or_else(|| {
                TrainError::other(format!(
                    "lineage trace refused: artifact {} has no run custody metadata",
                    artifact.content_hash
                ))
            })?;
            let tenant = crate::tenant::Tenant::parse(&run.tenant).ok_or_else(|| {
                TrainError::other(format!(
                    "lineage trace refused: run {} has invalid tenant metadata",
                    run.job_id
                ))
            })?;
            if tenant.is_restricted() {
                return Err(TrainError::other(
                    "lineage trace refused: Restricted provenance requires an exact tenant-scoped view",
                ));
            }
            chain.push(TraceStep {
                artifact,
                run: Some(run),
            });
            // The input hash of the edge that produced `current`.
            match self.input_hash_for_output(&current)? {
                Some(input) => current = input,
                None => break,
            }
        }
        Ok(chain)
    }

    /// The stage that produced the artifact at `content_hash` (ADR 0100 catalog
    /// lineage neighborhood — "produced_by"). `None` if the hash isn't indexed.
    pub fn producing_stage(&self, content_hash: &str) -> Result<Option<String>> {
        Ok(self
            .artifact_by_hash(&content_hash.to_lowercase())?
            .map(|a| a.stage_name))
    }

    /// The DOWNSTREAM consumer artifact hashes of `content_hash` — the outputs of
    /// every edge that took it as an input (ADR 0100 catalog neighborhood).
    /// Distinct, sorted.
    pub fn consumers_of(&self, content_hash: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT output_hash FROM lineage_edges WHERE input_hash = ?1")
            .map_err(|e| TrainError::other(format!("consumers_of prepare: {e}")))?;
        let rows = stmt
            .query_map(params![content_hash.to_lowercase()], |r| {
                r.get::<_, String>(0)
            })
            .map_err(|e| TrainError::other(format!("consumers_of query: {e}")))?;
        let mut out = rows
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("consumers_of collect: {e}")))?;
        out.sort();
        Ok(out)
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

    /// ALL input hashes feeding an output (a fan-in/merge stage has several) —
    /// the multi-branch generalisation of `input_hash_for_output` the
    /// provenance graph walks. Distinct, lowercased.
    pub fn inputs_for_output(&self, output_hash: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT input_hash FROM lineage_edges WHERE output_hash = ?1")
            .map_err(|e| TrainError::other(format!("inputs_for_output prepare: {e}")))?;
        let rows = stmt
            .query_map(params![output_hash.to_lowercase()], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| TrainError::other(format!("inputs_for_output query: {e}")))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("inputs_for_output collect: {e}")))
    }

    /// The FULL transitive upstream provenance DAG of `hash` (ADR 0099): every
    /// artifact reachable by walking input edges backward to the source inputs,
    /// plus the edges between them. Unlike [`trace`](Self::trace) (a single-input
    /// linear walk) this follows EVERY input at each fan-in, so a merge node's
    /// whole ancestry is captured. Cycle-guarded (a content-addressed DAG can't
    /// truly cycle, but a corrupt db must not loop). `exclude_restricted` drops
    /// any artifact produced by a `clinical`/`restricted`-tenant run and every
    /// edge touching it — the fail-closed export boundary (ADR 0061).
    pub fn graph_upstream(
        &self,
        hash: &str,
        exclude_restricted: bool,
    ) -> Result<crate::lineage_report::ProvenanceGraph> {
        use std::collections::VecDeque;
        let mut nodes: Vec<crate::lineage_report::GraphNode> = Vec::new();
        let mut edges: Vec<(String, String)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(hash.to_lowercase());

        while let Some(cur) = queue.pop_front() {
            if !seen.insert(cur.clone()) {
                continue;
            }
            // Resolve the artifact + its run; skip a restricted-tenant node (and
            // any edge into it) when exporting.
            let artifact = self.artifact_by_hash(&cur)?;
            let run = match &artifact {
                Some(a) => self.get_run(&a.job_id)?,
                None => None,
            };
            if exclude_restricted
                && let Some(r) = &run
                && crate::tenant::Tenant::parse(&r.tenant).is_some_and(|t| t.is_restricted())
            {
                // Drop this node entirely — do NOT enqueue its inputs, so
                // its whole subtree stays out of the export.
                continue;
            }
            nodes.push(crate::lineage_report::GraphNode {
                content_hash: cur.clone(),
                stage_name: artifact.as_ref().map(|a| a.stage_name.clone()),
                kind: artifact.as_ref().map(|a| a.kind.clone()),
                job_id: artifact.as_ref().map(|a| a.job_id.clone()),
            });
            for input in self.inputs_for_output(&cur)? {
                edges.push((input.clone(), cur.clone()));
                if !seen.contains(&input) {
                    queue.push_back(input);
                }
            }
        }
        // A restricted node is dropped, but an edge INTO it was recorded while
        // walking its consumer (before the exclusion fired). Drop every edge with
        // an excluded endpoint so no `clinical → …` edge leaks into the export.
        let kept: HashSet<&str> = nodes.iter().map(|n| n.content_hash.as_str()).collect();
        edges.retain(|(from, to)| kept.contains(from.as_str()) && kept.contains(to.as_str()));
        // Deterministic order so a DOT/JSON export is byte-stable across runs.
        nodes.sort_by(|a, b| a.content_hash.cmp(&b.content_hash));
        edges.sort();
        Ok(crate::lineage_report::ProvenanceGraph {
            root: hash.to_lowercase(),
            nodes,
            edges,
        })
    }

    /// Collate a model artifact's card (ADR 0099): its transitive DATA sources,
    /// producing recipe + config fingerprint (args identity), headline metrics,
    /// and gate outcome — a DETERMINISTIC content-addressed card (rebuilding on
    /// the same rows yields a byte-identical `card_hash`). `exclude_restricted`
    /// (the export path) drops clinical/restricted data sources; if the model
    /// itself is restricted, its root is excluded and this returns `None`.
    /// `None` also for an unknown hash.
    pub fn model_card(
        &self,
        hash: &str,
        exclude_restricted: bool,
    ) -> Result<Option<crate::lineage_report::ModelCard>> {
        let root = hash.to_lowercase();
        let graph = self.graph_upstream(&root, exclude_restricted)?;
        // The model's own node must survive (present + indexed) to build a card.
        let Some(node) = graph.nodes.iter().find(|n| n.content_hash == root) else {
            return Ok(None);
        };
        if node.stage_name.is_none() && graph.edges.is_empty() {
            return Ok(None); // unknown hash (no indexed artifact, no lineage)
        }
        let job_id = node.job_id.clone();
        let run = match &job_id {
            Some(j) => self.get_run(j)?,
            None => None,
        };
        let mut metrics = match &job_id {
            Some(j) => self.final_metrics(j)?,
            None => Vec::new(),
        };
        metrics.sort_by(|a, b| a.0.cmp(&b.0));
        let mut data_sources: Vec<String> = graph
            .sources()
            .into_iter()
            .filter(|s| *s != root) // the model itself isn't its own data source
            .map(|s| s.to_string())
            .collect();
        // `sources()` already returns sorted, but sort explicitly so the card's
        // content hash is provably determined HERE (not by an upstream invariant)
        // — cross-process byte-identical regardless of HashMap iteration order.
        data_sources.sort();
        let content = crate::lineage_report::CardContent {
            model_hash: root,
            job_id,
            recipe: run.as_ref().map(|r| r.recipe.clone()),
            config_fingerprint: run.as_ref().and_then(|r| r.config_fingerprint.clone()),
            gate_outcome: run.as_ref().and_then(|r| r.outcome.clone()),
            data_sources,
            metrics,
        };
        Ok(Some(crate::lineage_report::ModelCard::new(content)))
    }

    /// The symmetric difference of two runs (ADR 0099): only the recipe /
    /// config-fingerprint / input-hash / argument / gate-outcome / metric fields
    /// that DIFFER. Errors if either run is unknown or persisted args are
    /// malformed. This generic export refuses cross-tenant and Restricted runs;
    /// tenant-scoped engine views use [`Self::run_diff_for_tenant`].
    pub fn run_diff(&self, a: &str, b: &str) -> Result<crate::lineage_report::RunDiff> {
        self.run_diff_scoped(a, b, None)
    }

    /// Tenant-authorized comparison for the experiment registry. Restricted
    /// runs may be compared only through this exact-tenant seam; the generic
    /// lineage export remains fail-closed.
    pub(crate) fn run_diff_for_tenant(
        &self,
        a: &str,
        b: &str,
        tenant: &crate::tenant::Tenant,
    ) -> Result<crate::lineage_report::RunDiff> {
        self.run_diff_scoped(a, b, Some(tenant))
    }

    fn run_diff_scoped(
        &self,
        a: &str,
        b: &str,
        expected_tenant: Option<&crate::tenant::Tenant>,
    ) -> Result<crate::lineage_report::RunDiff> {
        let ra = self
            .get_run(a)?
            .ok_or_else(|| TrainError::other(format!("run diff: unknown run {a}")))?;
        let rb = self
            .get_run(b)?
            .ok_or_else(|| TrainError::other(format!("run diff: unknown run {b}")))?;
        let tenant_a = crate::tenant::Tenant::parse(&ra.tenant).ok_or_else(|| {
            TrainError::other(format!("run diff refused: run {a} has an invalid tenant"))
        })?;
        let tenant_b = crate::tenant::Tenant::parse(&rb.tenant).ok_or_else(|| {
            TrainError::other(format!("run diff refused: run {b} has an invalid tenant"))
        })?;
        if tenant_a != tenant_b {
            return Err(TrainError::other(
                "run diff refused: runs belong to different tenants",
            ));
        }
        match expected_tenant {
            Some(expected) if &tenant_a != expected => {
                return Err(TrainError::other(
                    "run diff refused: runs do not belong to the requested tenant",
                ));
            }
            None if tenant_a.is_restricted() => {
                return Err(TrainError::other(
                    "run diff refused: Restricted run details require an exact tenant-scoped experiment comparison",
                ));
            }
            _ => {}
        }

        let diff_opt = |x: &Option<String>, y: &Option<String>| {
            if x != y {
                Some((x.clone(), y.clone()))
            } else {
                None
            }
        };
        let recipe = if ra.recipe != rb.recipe {
            Some((Some(ra.recipe.clone()), Some(rb.recipe.clone())))
        } else {
            None
        };
        let inputs_a = self.input_hashes_for_run(a)?;
        let inputs_b = self.input_hashes_for_run(b)?;
        let input_hashes = (inputs_a != inputs_b).then_some((inputs_a, inputs_b));
        let arg_deltas = diff_args_json(ra.args_json.as_deref(), rb.args_json.as_deref(), a, b)?;

        // Metric deltas: union of names, keep only where a and b differ.
        let ma: std::collections::HashMap<String, f64> =
            self.final_metrics(a)?.into_iter().collect();
        let mb: std::collections::HashMap<String, f64> =
            self.final_metrics(b)?.into_iter().collect();
        let mut names: Vec<String> = ma.keys().chain(mb.keys()).cloned().collect();
        names.sort();
        names.dedup();
        let mut metric_deltas = Vec::new();
        for n in names {
            let (va, vb) = (ma.get(&n).copied(), mb.get(&n).copied());
            // Differ if either is missing or the values are not bit-equal
            // (bit-compare avoids the float-cmp lint + treats NaN consistently).
            if va.map(f64::to_bits) != vb.map(f64::to_bits) {
                metric_deltas.push((n, va, vb));
            }
        }

        Ok(crate::lineage_report::RunDiff {
            run_a: a.to_string(),
            run_b: b.to_string(),
            recipe,
            config_fingerprint: diff_opt(&ra.config_fingerprint, &rb.config_fingerprint),
            input_hashes,
            arg_deltas,
            gate_outcome: diff_opt(&ra.outcome, &rb.outcome),
            metric_deltas,
        })
    }

    /// Sorted unique input identities recorded for a run. Each input hash folds
    /// parent and data-source content identity per the lineage-edge contract.
    pub fn input_hashes_for_run(&self, job_id: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT DISTINCT input_hash FROM lineage_edges
                 WHERE job_id = ?1 ORDER BY input_hash",
            )
            .map_err(|e| TrainError::other(format!("prepare run inputs: {e}")))?;
        stmt.query_map(params![job_id], |row| row.get::<_, String>(0))
            .map_err(|e| TrainError::other(format!("query run inputs: {e}")))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("collect run inputs: {e}")))
    }

    /// Number of indexed runs (for `reindex` reporting + tests).
    pub fn run_count(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM runs", [], |r| r.get(0))
            .map_err(|e| TrainError::other(format!("run_count: {e}")))
    }
}

fn diff_args_json(
    raw_a: Option<&str>,
    raw_b: Option<&str>,
    run_a: &str,
    run_b: &str,
) -> Result<Vec<crate::lineage_report::ArgDelta>> {
    let parse = |raw: Option<&str>, run: &str| -> Result<Option<serde_json::Value>> {
        raw.map(|body| {
            serde_json::from_str(body).map_err(|e| {
                TrainError::other(format!("run diff: malformed args_json for {run}: {e}"))
            })
        })
        .transpose()
    };
    let parsed_a = parse(raw_a, run_a)?;
    let parsed_b = parse(raw_b, run_b)?;
    let mut leaves_a = std::collections::BTreeMap::new();
    let mut leaves_b = std::collections::BTreeMap::new();
    if let Some(value) = parsed_a {
        flatten_arg_leaves(&value, "$", &mut leaves_a);
    }
    if let Some(value) = parsed_b {
        flatten_arg_leaves(&value, "$", &mut leaves_b);
    }
    let paths: std::collections::BTreeSet<String> =
        leaves_a.keys().chain(leaves_b.keys()).cloned().collect();
    Ok(paths
        .into_iter()
        .filter_map(|path| {
            let a = leaves_a.get(&path).cloned();
            let b = leaves_b.get(&path).cloned();
            (a != b).then_some(crate::lineage_report::ArgDelta { path, a, b })
        })
        .collect())
}

fn flatten_arg_leaves(
    value: &serde_json::Value,
    path: &str,
    out: &mut std::collections::BTreeMap<String, serde_json::Value>,
) {
    match value {
        serde_json::Value::Object(object) if !object.is_empty() => {
            for (key, child) in object {
                let escaped = key.replace('~', "~0").replace('/', "~1");
                let child_path = if path == "$" {
                    format!("$/{escaped}")
                } else {
                    format!("{path}/{escaped}")
                };
                flatten_arg_leaves(child, &child_path, out);
            }
        }
        // Arrays remain one leaf. Treating index-level changes as parameter
        // paths would imply list order is a set; it is not.
        leaf => {
            out.insert(path.to_string(), leaf.clone());
        }
    }
}

/// Ingest a COMPLETED job into the index: run provenance (git SHA, hardware,
/// outcome, timestamp) + its artifacts (from the sidecars) + its lineage edges
/// (from `status.jsonl`). Idempotent — re-ingesting (a `reindex`) is safe.
/// Reads the content-addressed filesystem; the DB is the derived index.
pub fn ingest_job(job_id: &str, recipe: &str, outcome: &str) -> Result<()> {
    let db = LineageDb::open()?;
    let snap = crate::broker::ResourceSnapshot::probe();
    let tenant = crate::jobs::read_tenant(job_id)?;
    let experiment = crate::jobs::read_experiment(job_id)?.unwrap_or_else(|| recipe.to_string());
    let args_path = crate::paths::job_dir(job_id)?.join("args.json");
    let args_json = match std::fs::read_to_string(&args_path) {
        Ok(body) => {
            serde_json::from_str::<serde_json::Value>(&body).map_err(|e| {
                TrainError::other(format!("parse persisted args {}: {e}", args_path.display()))
            })?;
            Some(body)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            return Err(TrainError::Io {
                path: args_path,
                source: e,
            });
        }
    };
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
        tenant: tenant.to_string(),
        experiment: Some(experiment),
        args_json,
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
    // E1: fold this run's StageStep metrics into the queryable store so
    // `final_metric` / `top_runs_by_metric` / `blut compare` see it.
    db.record_metrics(&crate::framework::lineage::fold_metrics(job_id)?)?;
    // E2: fold the GPU sampler's `gpu_gauge` samples into the gauges
    // table so `gpu_saturation` / `gpu_wasted` are queryable — the
    // first-class "GPU not wasted" number (owner directive).
    db.record_gauges(&crate::framework::lineage::fold_gauges(job_id)?)?;
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
        tenant: row.get(11)?,
        experiment: row.get(12)?,
        args_json: row.get(13)?,
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
    fn metric_store_records_ranks_and_finals() {
        let db = db();
        db.record_metrics(&[
            // j1 PEAKS at step 2 (0.8) then COLLAPSES to a final 0.4 (step=-1).
            // final/ranking must use 0.4, NOT the max-step 0.8 — the bug guard.
            MetricRow {
                job_id: "j1".into(),
                node_idx: 0,
                step: 1,
                metric: "val_r".into(),
                value: 0.3,
                wall_unix: None,
            },
            MetricRow {
                job_id: "j1".into(),
                node_idx: 0,
                step: 2,
                metric: "val_r".into(),
                value: 0.8,
                wall_unix: None,
            },
            MetricRow {
                job_id: "j1".into(),
                node_idx: 0,
                step: -1,
                metric: "val_r".into(),
                value: 0.4,
                wall_unix: None,
            },
            MetricRow {
                job_id: "j2".into(),
                node_idx: 0,
                step: -1,
                metric: "val_r".into(),
                value: 0.6,
                wall_unix: None,
            },
        ])
        .unwrap();
        // final = the step=-1 headline (0.4), not the peak (0.8).
        assert_eq!(db.final_metric("j1", "val_r").unwrap(), Some(0.4));
        assert_eq!(db.final_metric("j1", "absent").unwrap(), None);
        // ranking by FINAL: j2 (0.6) beats j1 (0.4) — j1's peak 0.8 is ignored.
        let top = db.top_runs_by_metric("val_r", true, 10).unwrap();
        assert_eq!(top[0].0, "j2");
        assert_eq!(top[0].2, 0.6);
        assert_eq!(top[1].0, "j1");
        assert_eq!(top[1].2, 0.4, "ranks by final, not best-ever");
    }

    #[test]
    fn results_queries_best_series_and_terminal_artifact() {
        let db = db();
        // val_r peaks at step 2 (0.8) then collapses to a final 0.4 (step=-1).
        db.record_metrics(&[
            MetricRow {
                job_id: "j".into(),
                node_idx: 0,
                step: 0,
                metric: "val_r".into(),
                value: 0.3,
                wall_unix: None,
            },
            MetricRow {
                job_id: "j".into(),
                node_idx: 0,
                step: 1,
                metric: "val_r".into(),
                value: 0.5,
                wall_unix: None,
            },
            MetricRow {
                job_id: "j".into(),
                node_idx: 0,
                step: 2,
                metric: "val_r".into(),
                value: 0.8,
                wall_unix: None,
            },
            MetricRow {
                job_id: "j".into(),
                node_idx: 0,
                step: -1,
                metric: "val_r".into(),
                value: 0.4,
                wall_unix: None,
            },
        ])
        .unwrap();
        // best (peak) is the step=2 0.8, NOT the collapsed final 0.4.
        assert_eq!(db.best_metric("j", "val_r", true).unwrap(), Some((2, 0.8)));
        // trajectory excludes the step=-1 marker → exactly the 3 real samples.
        let series = db.metric_series("j", "val_r").unwrap();
        assert_eq!(series, vec![(0, 0.3), (1, 0.5), (2, 0.8)]);
        // a never-sampled metric → None / empty.
        assert_eq!(db.best_metric("j", "absent", true).unwrap(), None);
        assert!(db.metric_series("j", "absent").unwrap().is_empty());

        // terminal artifact = highest stage_idx.
        db.record_artifact(&art("j", 0, "h0")).unwrap();
        db.record_artifact(&art("j", 3, "h3")).unwrap();
        let term = db.terminal_artifact("j").unwrap().unwrap();
        assert_eq!(term.stage_idx, 3);
        assert_eq!(
            term.sidecar_path.as_deref(),
            Some("/j/j/stages/3/output.metadata.json")
        );
        // a job with no artifacts → None.
        assert!(db.terminal_artifact("other").unwrap().is_none());
    }

    #[test]
    fn gpu_saturation_summarizes_gauges() {
        let db = db();
        // 4 samples: util 90,95,20,80 → mean 71.25; below floor(50) = 1/4 = 0.25.
        let g = |t: i64, u: f64| GaugeRow {
            job_id: "j".into(),
            node_idx: 0,
            wall_unix: t,
            gpu_util: Some(u),
            ..Default::default()
        };
        db.record_gauges(&[g(1, 90.0), g(2, 95.0), g(3, 20.0), g(4, 80.0)])
            .unwrap();
        let s = db.gpu_saturation("j", 50.0).unwrap().unwrap();
        assert_eq!(s.samples, 4);
        assert!(
            (s.saturation - 71.25).abs() < 1e-9,
            "mean util {}",
            s.saturation
        );
        assert!((s.wasted - 0.25).abs() < 1e-9, "wasted {}", s.wasted);
        // a job with no gauges → None.
        assert!(db.gpu_saturation("other", 50.0).unwrap().is_none());
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
    fn v3_run_rows_migrate_to_experiment_and_args_json() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("lineage-v3.db");
        let legacy = rusqlite::Connection::open(&path).unwrap();
        legacy
            .execute_batch(
                "CREATE TABLE runs (
                    job_id TEXT PRIMARY KEY,
                    recipe TEXT NOT NULL,
                    config_fingerprint TEXT,
                    git_sha TEXT,
                    started_unix INTEGER,
                    ended_unix INTEGER,
                    outcome TEXT,
                    host TEXT,
                    gpu_name TEXT,
                    ram_gib INTEGER,
                    vram_mib INTEGER,
                    tenant TEXT NOT NULL DEFAULT 'default'
                 );
                 INSERT INTO runs (job_id, recipe, tenant)
                    VALUES ('legacy', 'demo', 'research/dev');
                 PRAGMA user_version=3;",
            )
            .unwrap();
        drop(legacy);

        let db = LineageDb::open_at(&path).unwrap();
        let legacy = db.get_run("legacy").unwrap().unwrap();
        assert_eq!(legacy.tenant, "research/dev");
        assert_eq!(legacy.experiment, None);
        assert_eq!(legacy.args_json, None);
        db.record_run(&RunRow {
            job_id: "new".into(),
            recipe: "demo".into(),
            experiment: Some("campaign-a".into()),
            args_json: Some("{\"lr\":0.01}".into()),
            ..RunRow::default()
        })
        .unwrap();
        assert_eq!(
            db.get_run("new").unwrap().unwrap().args_json.as_deref(),
            Some("{\"lr\":0.01}")
        );
        assert_eq!(
            db.get_run("new").unwrap().unwrap().experiment.as_deref(),
            Some("campaign-a")
        );
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
        assert_eq!(
            db.get_run("job-1").unwrap().unwrap().git_sha.as_deref(),
            Some("abc123")
        );
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
        assert_eq!(
            got.git_sha.as_deref(),
            Some("sha"),
            "COALESCE preserves prior provenance"
        );
        assert_eq!(got.ended_unix, Some(42));
        assert_eq!(db.run_count().unwrap(), 1, "upsert, not duplicate");
    }

    #[test]
    fn record_run_refuses_experiment_or_args_reassignment() {
        let db = db();
        db.record_run(&RunRow {
            job_id: "immutable".into(),
            recipe: "r".into(),
            experiment: Some("campaign-a".into()),
            args_json: Some("{\"lr\":0.01}".into()),
            ..RunRow::default()
        })
        .unwrap();
        // Formatting-only JSON differences remain idempotent.
        db.record_run(&RunRow {
            job_id: "immutable".into(),
            recipe: "r".into(),
            experiment: Some("campaign-a".into()),
            args_json: Some("{ \"lr\" : 0.01 }".into()),
            ..RunRow::default()
        })
        .unwrap();
        assert!(
            db.record_run(&RunRow {
                job_id: "immutable".into(),
                recipe: "r".into(),
                experiment: Some("campaign-b".into()),
                ..RunRow::default()
            })
            .is_err()
        );
        assert!(
            db.record_run(&RunRow {
                job_id: "immutable".into(),
                recipe: "r".into(),
                args_json: Some("{\"lr\":0.02}".into()),
                ..RunRow::default()
            })
            .is_err()
        );
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
        db.record_run(&RunRow {
            job_id: "j".into(),
            recipe: "r".into(),
            ..Default::default()
        })
        .unwrap();
        db.record_artifact(&art("j", 0, "corpushash")).unwrap();
        db.record_artifact(&art("j", 1, "trainhash")).unwrap();
        db.record_artifact(&art("j", 2, "gatehash")).unwrap();
        db.record_edge(&EdgeRow {
            job_id: "j".into(),
            to_idx: 1,
            input_hash: "corpushash".into(),
            output_hash: "trainhash".into(),
        })
        .unwrap();
        db.record_edge(&EdgeRow {
            job_id: "j".into(),
            to_idx: 2,
            input_hash: "trainhash".into(),
            output_hash: "gatehash".into(),
        })
        .unwrap();

        let chain = db.trace("gatehash").unwrap();
        let hashes: Vec<&str> = chain
            .iter()
            .map(|s| s.artifact.content_hash.as_str())
            .collect();
        assert_eq!(
            hashes,
            vec!["gatehash", "trainhash", "corpushash"],
            "upstream order"
        );
        assert!(
            chain[0].run.is_some(),
            "trace annotates each hop with its run"
        );
    }

    #[test]
    fn generic_trace_refuses_restricted_provenance() {
        let db = db();
        db.record_run(&RunRow {
            job_id: "clinical-job".into(),
            recipe: "r".into(),
            tenant: "clinical/prod".into(),
            ..Default::default()
        })
        .unwrap();
        db.record_artifact(&art("clinical-job", 0, "restrictedhash"))
            .unwrap();
        let error = db.trace("restrictedhash").unwrap_err().to_string();
        assert!(error.contains("Restricted provenance"));
    }

    #[test]
    fn generic_trace_refuses_artifact_without_run_custody() {
        let db = db();
        db.record_artifact(&art("orphan-job", 0, "orphanhash"))
            .unwrap();
        let error = db.trace("orphanhash").unwrap_err().to_string();
        assert!(error.contains("no run custody metadata"));
    }

    #[test]
    fn stage_idx_parsed_from_sidecar_path() {
        assert_eq!(
            stage_idx_of(Path::new("/j/stages/0-make/output.metadata.json")),
            0
        );
        assert_eq!(
            stage_idx_of(Path::new("/j/stages/12-train_joint/output.metadata.json")),
            12
        );
        assert_eq!(
            stage_idx_of(Path::new("/j/stages/garbage/output.metadata.json")),
            0
        );
    }

    #[test]
    fn trace_is_cycle_guarded() {
        let db = db();
        db.record_run(&RunRow {
            job_id: "j".into(),
            recipe: "r".into(),
            ..Default::default()
        })
        .unwrap();
        db.record_artifact(&art("j", 0, "a")).unwrap();
        db.record_artifact(&art("j", 1, "b")).unwrap();
        // A pathological cycle a→b→a must terminate, not hang.
        db.record_edge(&EdgeRow {
            job_id: "j".into(),
            to_idx: 1,
            input_hash: "b".into(),
            output_hash: "a".into(),
        })
        .unwrap();
        db.record_edge(&EdgeRow {
            job_id: "j".into(),
            to_idx: 0,
            input_hash: "a".into(),
            output_hash: "b".into(),
        })
        .unwrap();
        let chain = db.trace("a").unwrap();
        assert!(chain.len() <= 2, "cycle guard bounds the walk");
    }

    #[test]
    fn freshness_verdict_classifies_code_drift() {
        // Same SHA → FRESH.
        assert_eq!(
            freshness_verdict(Some("abc123".into()), Some("abc123".into())),
            CodeFreshness::Fresh {
                git_sha: "abc123".into()
            }
        );
        // Different SHA → STALE (built_sha vs head).
        assert_eq!(
            freshness_verdict(Some("old".into()), Some("new".into())),
            CodeFreshness::Stale {
                built_sha: "old".into(),
                head: "new".into()
            }
        );
        // Missing either side → UNKNOWN (not stale — just unverifiable).
        assert_eq!(
            freshness_verdict(None, Some("h".into())),
            CodeFreshness::Unknown
        );
        assert_eq!(
            freshness_verdict(Some("r".into()), None),
            CodeFreshness::Unknown
        );
        assert_eq!(freshness_verdict(None, None), CodeFreshness::Unknown);
        // is_stale only true for Stale.
        assert!(freshness_verdict(Some("a".into()), Some("b".into())).is_stale());
        assert!(!freshness_verdict(Some("a".into()), Some("a".into())).is_stale());
    }
}
