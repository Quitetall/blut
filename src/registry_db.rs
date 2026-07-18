// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Local **plan deployment registry** (ADR 0085) — a `~/.blut/registry.db`
//! SQLite store that turns a content-addressed `PlanSpec` into a promotable,
//! rollback-able deploy identity, all pure-local (charter-safe: no server).
//!
//! Two tables:
//!   * `deployments` — IMMUTABLE rows keyed by the PlanSpec's ADR-0078
//!     provenance fingerprint: the canonical spec bytes + publisher + tenant +
//!     source provenance. Re-publishing identical bytes is a no-op (same key).
//!   * `deployment_pointers` — MUTABLE named pointers `(tenant, name) →
//!     fingerprint`, with an append-only `pointer_history` audit trail so
//!     `rollback` can pop a pointer to its previous target atomically.
//!
//! The fingerprint IS the deploy identity: `blut run registry://plan@prod`
//! resolves a pointer to a frozen `deployments` row and dispatches that exact
//! graph. Publish typechecks fail-closed (a PlanSpec that does not resolve
//! against the compiled stage registry never enters `deployments`), and a
//! Restricted-tenant fingerprint can never be promoted onto another tenant's
//! pointer (ADR 0061/0096 clinical boundary).

use rusqlite::{Connection, OpenFlags, params};

use crate::error::{Result, TrainError};
use crate::framework::Registry;
use crate::framework::plan_spec::PlanSpec;

/// The clinical / PHI tenant tag. A deployment under this tenant can never be
/// promoted onto a pointer owned by a different tenant (ADR 0061 hard-block).
/// A minimal stand-in until full multi-tenancy (ADR 0096) lands.
pub const RESTRICTED_TENANT: &str = "restricted";
/// The default (non-clinical) tenant.
pub const SHARED_TENANT: &str = "shared";

const CREATE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS deployments (
    plan_fingerprint TEXT PRIMARY KEY,
    spec_bytes       BLOB NOT NULL,
    publisher        TEXT NOT NULL,
    tenant           TEXT NOT NULL,
    source           TEXT,
    created_at       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS deployment_pointers (
    tenant           TEXT NOT NULL,
    name             TEXT NOT NULL,
    plan_fingerprint TEXT NOT NULL,
    updated_at       INTEGER NOT NULL,
    PRIMARY KEY (tenant, name)
);
CREATE TABLE IF NOT EXISTS pointer_history (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant           TEXT NOT NULL,
    name             TEXT NOT NULL,
    plan_fingerprint TEXT NOT NULL,
    moved_at         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_ptr ON pointer_history(tenant, name, seq);
";

/// A published, immutable deployment row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deployment {
    pub plan_fingerprint: String,
    pub publisher: String,
    pub tenant: String,
    pub source: Option<String>,
    pub created_at: i64,
}

/// One entry in a pointer's audit trail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub plan_fingerprint: String,
    pub moved_at: i64,
}

/// Path to the deployment registry DB (`~/.blut/registry.db`).
/// `$BLUT_REGISTRY_DB` overrides it (tests, alternate homes).
pub fn registry_db_path() -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BLUT_REGISTRY_DB") {
        return Ok(std::path::PathBuf::from(p));
    }
    let dir = dirs::home_dir()
        .ok_or_else(|| TrainError::other("home_dir() unavailable; set $BLUT_REGISTRY_DB"))?
        .join(".blut");
    std::fs::create_dir_all(&dir).map_err(|e| TrainError::Io {
        path: dir.clone(),
        source: e,
    })?;
    Ok(dir.join("registry.db"))
}

/// Open (create-if-absent) the registry DB. Read+write, WAL.
pub fn open() -> Result<Connection> {
    open_at(&registry_db_path()?)
}

pub fn open_at(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )
    .map_err(|e| TrainError::other(format!("open {}: {e}", path.display())))?;
    // Best-effort WAL; a failure just leaves the default journal mode.
    let _ =
        conn.query_row::<rusqlite::types::Value, _, _>("PRAGMA journal_mode=WAL", [], |r| r.get(0));
    conn.execute_batch(CREATE_SCHEMA)
        .map_err(|e| TrainError::other(format!("create registry schema: {e}")))?;
    Ok(conn)
}

/// The ADR-0078 fingerprint that keys a spec in `deployments` — deterministic
/// over the spec's canonical bytes, so re-publishing identical bytes yields the
/// same id (idempotent).
pub fn fingerprint(spec: &PlanSpec) -> String {
    spec.provenance_fingerprint("", &serde_json::Value::Null)
        .to_hex()
}

/// Publish a PlanSpec: TYPECHECK it against `reg` (fail-closed — a spec that does
/// not compile never enters the table), then insert an immutable `deployments`
/// row keyed by its fingerprint. Idempotent: re-publishing identical bytes
/// returns the existing fingerprint without a second insert. Returns the id.
pub fn publish(
    conn: &Connection,
    reg: &Registry,
    spec: &PlanSpec,
    publisher: &str,
    tenant: &str,
    source: Option<&str>,
    now_unix: i64,
) -> Result<String> {
    // Fail-closed typecheck: the spec must resolve against the compiled stages.
    spec.compile(reg).map_err(|e| {
        TrainError::other(format!(
            "publish refused — PlanSpec does not typecheck: {e}"
        ))
    })?;
    let fp = fingerprint(spec);

    // The fingerprint is over spec CONTENT only, so identical bytes published
    // under two tenants would collide on one immutable row — silently binding the
    // content to whichever tenant published first. Reject a cross-tenant
    // re-publish (tenant isolation); a same-tenant re-publish is the idempotent
    // no-op the ADR promises.
    if let Some(existing) = get_deployment(conn, &fp)? {
        if existing.tenant != tenant {
            return Err(TrainError::other(format!(
                "publish refused — fingerprint {fp} already published under tenant \
                 '{}' (not '{tenant}')",
                existing.tenant
            )));
        }
        return Ok(fp);
    }
    // Known-absent → plain INSERT so real errors (disk-full, constraint) surface
    // instead of being swallowed. A concurrent same-fp insert races to the PK
    // constraint; treat that lone case as the idempotent no-op (content is equal).
    let bytes = spec.canonical_bytes();
    match conn.execute(
        "INSERT INTO deployments \
         (plan_fingerprint, spec_bytes, publisher, tenant, source, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![fp, bytes, publisher, tenant, source, now_unix],
    ) {
        Ok(_) => Ok(fp),
        Err(rusqlite::Error::SqliteFailure(e, _))
            if e.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            // Lost an insert race for this fingerprint. Re-verify the winner's
            // tenant: identical content + same tenant is the idempotent no-op;
            // a DIFFERENT tenant is the very cross-tenant collision we refuse
            // (the narrow race window that a pre-INSERT check alone can't close).
            match get_deployment(conn, &fp)? {
                Some(d) if d.tenant == tenant => Ok(fp),
                _ => Err(TrainError::other(format!(
                    "publish refused — fingerprint {fp} already published under a different tenant"
                ))),
            }
        }
        Err(e) => Err(TrainError::other(format!("insert deployment: {e}"))),
    }
}

/// Look up an immutable deployment by fingerprint.
pub fn get_deployment(conn: &Connection, fp: &str) -> Result<Option<Deployment>> {
    let r = conn.query_row(
        "SELECT plan_fingerprint, publisher, tenant, source, created_at \
         FROM deployments WHERE plan_fingerprint = ?1",
        params![fp],
        |row| {
            Ok(Deployment {
                plan_fingerprint: row.get(0)?,
                publisher: row.get(1)?,
                tenant: row.get(2)?,
                source: row.get(3)?,
                created_at: row.get(4)?,
            })
        },
    );
    match r {
        Ok(d) => Ok(Some(d)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(TrainError::other(format!("get deployment: {e}"))),
    }
}

/// Promote `fp` onto the pointer `(tenant, name)`, appending the move to the
/// audit trail (single transaction). Fail-closed boundary: the deployment must
/// exist AND its own tenant must equal `tenant` — a Restricted-tenant
/// fingerprint can never be promoted onto another tenant's pointer (ADR 0061).
pub fn promote(
    conn: &mut Connection,
    fp: &str,
    tenant: &str,
    name: &str,
    now_unix: i64,
) -> Result<()> {
    let dep = get_deployment(conn, fp)?
        .ok_or_else(|| TrainError::other(format!("promote refused — no deployment {fp}")))?;
    if dep.tenant != tenant {
        return Err(TrainError::other(format!(
            "promote refused — cross-tenant boundary: deployment tenant '{}' != pointer tenant '{}'",
            dep.tenant, tenant
        )));
    }
    // IMMEDIATE (not DEFERRED), matching `rollback` and the model registry: both
    // verbs MUTATE the same pointer, so take the write lock up front and
    // serialize cleanly — a concurrent double-promote waits instead of surfacing
    // a raw SQLITE_BUSY.
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| TrainError::other(format!("promote txn: {e}")))?;
    tx.execute(
        "INSERT INTO deployment_pointers (tenant, name, plan_fingerprint, updated_at) \
         VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(tenant, name) DO UPDATE SET plan_fingerprint = ?3, updated_at = ?4",
        params![tenant, name, fp, now_unix],
    )
    .map_err(|e| TrainError::other(format!("upsert pointer: {e}")))?;
    tx.execute(
        "INSERT INTO pointer_history (tenant, name, plan_fingerprint, moved_at) \
         VALUES (?1, ?2, ?3, ?4)",
        params![tenant, name, fp, now_unix],
    )
    .map_err(|e| TrainError::other(format!("append history: {e}")))?;
    tx.commit()
        .map_err(|e| TrainError::other(format!("promote commit: {e}")))?;
    Ok(())
}

/// Roll a pointer back to its previous target atomically (a single
/// transaction): reads the two most-recent history entries, sets the pointer to
/// the second-most-recent, and records the rollback in the trail. Errors if the
/// pointer has no prior target. Returns the fingerprint rolled back to.
pub fn rollback(conn: &mut Connection, tenant: &str, name: &str, now_unix: i64) -> Result<String> {
    // BEGIN IMMEDIATE: take the write lock up front so the two-most-recent read
    // and the pointer/history writes see ONE consistent snapshot — a concurrent
    // promote/rollback can't slip between the read and the write and stale `prev`.
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| TrainError::other(format!("rollback txn: {e}")))?;
    let recent: Vec<String> = {
        let mut stmt = tx
            .prepare(
                "SELECT plan_fingerprint FROM pointer_history \
                 WHERE tenant = ?1 AND name = ?2 ORDER BY seq DESC LIMIT 2",
            )
            .map_err(|e| TrainError::other(format!("history query: {e}")))?;
        let rows = stmt
            .query_map(params![tenant, name], |row| row.get::<_, String>(0))
            .map_err(|e| TrainError::other(format!("history rows: {e}")))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("history collect: {e}")))?
    };
    if recent.len() < 2 {
        return Err(TrainError::other(format!(
            "rollback refused — pointer 'registry://plan@{name}' has no prior deployment"
        )));
    }
    let prev = recent[1].clone();
    // Defense-in-depth (matches promote): the target must belong to THIS tenant.
    // promote already blocks a foreign fingerprint from entering the trail, so
    // this can only fail on a corrupted/hand-edited DB — fail closed rather than
    // silently re-point across the clinical boundary (ADR 0061).
    match get_deployment(&tx, &prev)? {
        Some(d) if d.tenant == tenant => {}
        _ => {
            return Err(TrainError::other(format!(
                "rollback refused — prior target {prev} is not a '{tenant}'-tenant deployment"
            )));
        }
    }
    tx.execute(
        "UPDATE deployment_pointers SET plan_fingerprint = ?3, updated_at = ?4 \
         WHERE tenant = ?1 AND name = ?2",
        params![tenant, name, prev, now_unix],
    )
    .map_err(|e| TrainError::other(format!("rollback pointer: {e}")))?;
    tx.execute(
        "INSERT INTO pointer_history (tenant, name, plan_fingerprint, moved_at) \
         VALUES (?1, ?2, ?3, ?4)",
        params![tenant, name, prev, now_unix],
    )
    .map_err(|e| TrainError::other(format!("rollback history: {e}")))?;
    tx.commit()
        .map_err(|e| TrainError::other(format!("rollback commit: {e}")))?;
    Ok(prev)
}

/// The fingerprint a pointer currently resolves to (`None` if unset).
pub fn resolve_pointer(conn: &Connection, tenant: &str, name: &str) -> Result<Option<String>> {
    let r = conn.query_row(
        "SELECT plan_fingerprint FROM deployment_pointers WHERE tenant = ?1 AND name = ?2",
        params![tenant, name],
        |row| row.get::<_, String>(0),
    );
    match r {
        Ok(fp) => Ok(Some(fp)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(TrainError::other(format!("resolve pointer: {e}"))),
    }
}

/// Resolve a pointer all the way to its frozen `PlanSpec` (the run-dispatch
/// path for `registry://plan@<name>`).
pub fn resolve_spec(conn: &Connection, tenant: &str, name: &str) -> Result<PlanSpec> {
    let fp = resolve_pointer(conn, tenant, name)?.ok_or_else(|| {
        TrainError::other(format!("no deployment pointer 'registry://plan@{name}'"))
    })?;
    let bytes: Vec<u8> = conn
        .query_row(
            "SELECT spec_bytes FROM deployments WHERE plan_fingerprint = ?1",
            params![fp],
            |row| row.get(0),
        )
        .map_err(|e| TrainError::other(format!("load deployment {fp}: {e}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| TrainError::other(format!("deserialize deployment {fp}: {e}")))
}

/// A pointer's full audit trail, oldest first.
pub fn history(conn: &Connection, tenant: &str, name: &str) -> Result<Vec<HistoryEntry>> {
    let mut stmt = conn
        .prepare(
            "SELECT plan_fingerprint, moved_at FROM pointer_history \
             WHERE tenant = ?1 AND name = ?2 ORDER BY seq ASC",
        )
        .map_err(|e| TrainError::other(format!("history prepare: {e}")))?;
    let rows = stmt
        .query_map(params![tenant, name], |row| {
            Ok(HistoryEntry {
                plan_fingerprint: row.get(0)?,
                moved_at: row.get(1)?,
            })
        })
        .map_err(|e| TrainError::other(format!("history query: {e}")))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| TrainError::other(format!("history collect: {e}")))
}

/// Parse a `registry://plan@<name>` deploy URI into its pointer name. Returns
/// `None` for any non-registry string (so the run dispatcher can fall through to
/// file-path handling) OR a name with characters outside the safe identifier set
/// `[A-Za-z0-9_.-]` — so a pointer name can never carry whitespace, slashes, or
/// control bytes into an audit log or a future filesystem context.
pub fn parse_pointer_uri(uri: &str) -> Option<&str> {
    let name = uri.strip_prefix("registry://plan@")?;
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
    {
        return None;
    }
    Some(name)
}
