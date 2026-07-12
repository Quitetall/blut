// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Local **model registry** (ADR 0090, increment 1) — a named, promotable,
//! rollback-able surface over checkpoint content hashes, sharing the frozen
//! ADR-0085 `~/.blut/registry.db` and mirroring its pattern verb-for-verb.
//!
//! Where the 0085 plan registry binds a name to a PlanSpec provenance
//! fingerprint, this binds a model *name* + a mutable *alias* (`prod`,
//! `staging`, `v7`) to an immutable checkpoint `sha256`. Same audit semantics:
//!
//!   * `models` — IMMUTABLE rows keyed by the checkpoint `model_hash` (a
//!     sha256 the cache/lineage already computed): the name it belongs to, its
//!     tenant, and where it came from. Re-registering the same hash under the
//!     same (tenant, name) is the idempotent no-op; a cross-tenant or
//!     rename collision is refused.
//!   * `model_pointers` — MUTABLE `(tenant, name, alias) → model_hash`.
//!   * `model_pointer_history` — append-only audit trail, so `rollback` pops a
//!     pointer to its previous target atomically.
//!
//! `model://<name>@<alias>` resolves a pointer to a checkpoint hash; the hash
//! then feeds a recipe's typed args (the artifact itself lives in the cache/fs
//! by that hash — there are no bytes to store here, unlike the plan registry).
//! A Restricted-tenant checkpoint can never be promoted onto another tenant's
//! pointer (ADR 0061/0096 clinical boundary), exactly as in 0085.
//!
//! Increment 2 (own gate) adds the PCCP fail-close: promotion to a governed
//! alias (`@prod`) will shell out to `pccp_gate.py --change-id` so `@prod` can
//! never point at an unpromoted checkpoint. This increment is the pure-engine
//! registry mechanism; it deliberately does not yet call Python.

use rusqlite::{Connection, params};

use crate::error::{Result, TrainError};
// Share the ADR-0085 registry DB file + its clinical tenant tags, so `blut plan`
// and `blut model` live in ONE `~/.blut/registry.db` with one boundary policy.
use crate::registry_db::registry_db_path;
pub use crate::registry_db::{RESTRICTED_TENANT, SHARED_TENANT};

const CREATE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS models (
    model_hash   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    tenant       TEXT NOT NULL,
    source       TEXT,
    created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name, tenant);
CREATE TABLE IF NOT EXISTS model_pointers (
    tenant       TEXT NOT NULL,
    name         TEXT NOT NULL,
    alias        TEXT NOT NULL,
    model_hash   TEXT NOT NULL,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (tenant, name, alias)
);
CREATE TABLE IF NOT EXISTS model_pointer_history (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant       TEXT NOT NULL,
    name         TEXT NOT NULL,
    alias        TEXT NOT NULL,
    model_hash   TEXT NOT NULL,
    moved_at     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_history_ptr
    ON model_pointer_history(tenant, name, alias, seq);
";

/// A registered, immutable model checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelReg {
    pub model_hash: String,
    pub name: String,
    pub tenant: String,
    pub source: Option<String>,
    pub created_at: i64,
}

/// One entry in a model pointer's audit trail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub model_hash: String,
    pub moved_at: i64,
}

/// Open (create-if-absent) the shared registry DB and ensure the model tables
/// exist. Reuses the ADR-0085 `~/.blut/registry.db` path (`$BLUT_REGISTRY_DB`
/// override), so both registries share one file + one clinical boundary.
pub fn open() -> Result<Connection> {
    open_at(&registry_db_path()?)
}

pub fn open_at(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_CREATE,
    )
    .map_err(|e| TrainError::other(format!("open {}: {e}", path.display())))?;
    let _ =
        conn.query_row::<rusqlite::types::Value, _, _>("PRAGMA journal_mode=WAL", [], |r| r.get(0));
    conn.execute_batch(CREATE_SCHEMA)
        .map_err(|e| TrainError::other(format!("create model registry schema: {e}")))?;
    Ok(conn)
}

/// A checkpoint hash is a lowercase 64-char sha256 hex (what
/// `datasets_db::compute_file_sha256`'s `{:x}` and the cache both emit). Reject
/// anything else fail-closed — a malformed hash must never enter the registry or
/// an audit row.
fn is_model_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A pointer `name`/`alias` is a safe identifier `[A-Za-z0-9_.-]`, non-empty —
/// so it can never carry whitespace, slashes, `@`, or control bytes into an
/// audit log, a URI, or a future filesystem context.
fn is_safe_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// Validate a `tenant` tag by the FROZEN ADR-0096 `Tenant` grammar (the single
/// canonical validator: `project[/domain]`, rejects empty segments / `.`/`..`
/// traversal / unsafe chars). A garbled `--tenant` must never create a stray
/// namespace or a garbled audit row — the clinical boundary is only meaningful
/// if a tenant string is well-formed. `Err` fail-closed on a bad tag.
fn check_tenant(tenant: &str) -> Result<()> {
    if crate::tenant::Tenant::parse(tenant).is_none() {
        return Err(TrainError::other(format!(
            "refused — '{tenant}' is not a valid tenant (project[/domain], [A-Za-z0-9_.-])"
        )));
    }
    Ok(())
}

/// Register a checkpoint hash under a model name (immutable candidate row).
/// Idempotent: re-registering the same hash under the SAME (tenant, name) is a
/// no-op. Refused fail-closed if the hash is malformed, the name is unsafe, or
/// the hash was already registered under a different tenant OR a different name
/// (a content hash belongs to exactly one model identity).
pub fn register(
    conn: &Connection,
    model_hash: &str,
    name: &str,
    tenant: &str,
    source: Option<&str>,
    now_unix: i64,
) -> Result<()> {
    check_tenant(tenant)?;
    if !is_model_hash(model_hash) {
        return Err(TrainError::other(format!(
            "register refused — '{model_hash}' is not a lowercase 64-char sha256"
        )));
    }
    if !is_safe_ident(name) {
        return Err(TrainError::other(format!(
            "register refused — model name '{name}' is not [A-Za-z0-9_.-]"
        )));
    }
    if let Some(existing) = get_model(conn, model_hash)? {
        if existing.tenant != tenant {
            return Err(TrainError::other(format!(
                "register refused — checkpoint {model_hash} already registered under tenant \
                 '{}' (not '{tenant}')",
                existing.tenant
            )));
        }
        if existing.name != name {
            return Err(TrainError::other(format!(
                "register refused — checkpoint {model_hash} already registered as model \
                 '{}' (not '{name}')",
                existing.name
            )));
        }
        return Ok(()); // same hash, tenant, name → idempotent no-op
    }
    // Known-absent → plain INSERT so real errors surface. A concurrent same-hash
    // insert races to the PK; treat that lone case as the idempotent no-op only
    // when the winner's (tenant, name) match — else it's the collision we refuse.
    match conn.execute(
        "INSERT INTO models (model_hash, name, tenant, source, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![model_hash, name, tenant, source, now_unix],
    ) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(e, _))
            if e.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            match get_model(conn, model_hash)? {
                Some(m) if m.tenant == tenant && m.name == name => Ok(()),
                _ => Err(TrainError::other(format!(
                    "register refused — checkpoint {model_hash} already registered under a \
                     different model identity"
                ))),
            }
        }
        Err(e) => Err(TrainError::other(format!("insert model: {e}"))),
    }
}

/// Look up an immutable model row by checkpoint hash.
pub fn get_model(conn: &Connection, model_hash: &str) -> Result<Option<ModelReg>> {
    let r = conn.query_row(
        "SELECT model_hash, name, tenant, source, created_at FROM models WHERE model_hash = ?1",
        params![model_hash],
        |row| {
            Ok(ModelReg {
                model_hash: row.get(0)?,
                name: row.get(1)?,
                tenant: row.get(2)?,
                source: row.get(3)?,
                created_at: row.get(4)?,
            })
        },
    );
    match r {
        Ok(m) => Ok(Some(m)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(TrainError::other(format!("get model: {e}"))),
    }
}

/// Promote `model_hash` onto the pointer `(tenant, name, alias)`, appending the
/// move to the audit trail in one transaction. Fail-closed boundary: the model
/// must be registered AND its own tenant must equal `tenant` — a Restricted
/// checkpoint can never be promoted onto another tenant's pointer (ADR 0061).
/// The name is also verified to match the registered model, so an alias can't be
/// pointed at a hash that belongs to a different model line.
pub fn promote(
    conn: &mut Connection,
    model_hash: &str,
    tenant: &str,
    name: &str,
    alias: &str,
    now_unix: i64,
) -> Result<()> {
    check_tenant(tenant)?;
    if !is_safe_ident(alias) {
        return Err(TrainError::other(format!(
            "promote refused — alias '{alias}' is not [A-Za-z0-9_.-]"
        )));
    }
    let m = get_model(conn, model_hash)?
        .ok_or_else(|| TrainError::other(format!("promote refused — no model {model_hash}")))?;
    if m.tenant != tenant {
        return Err(TrainError::other(format!(
            "promote refused — cross-tenant boundary: model tenant '{}' != pointer tenant '{}'",
            m.tenant, tenant
        )));
    }
    if m.name != name {
        return Err(TrainError::other(format!(
            "promote refused — checkpoint {model_hash} is model '{}', not '{name}'",
            m.name
        )));
    }
    // IMMEDIATE (not DEFERRED): both promote and rollback MUTATE the same pointer,
    // so they take the write lock up front and serialize cleanly — a concurrent
    // double-promote waits instead of surfacing a raw SQLITE_BUSY.
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| TrainError::other(format!("promote txn: {e}")))?;
    tx.execute(
        "INSERT INTO model_pointers (tenant, name, alias, model_hash, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5) \
         ON CONFLICT(tenant, name, alias) DO UPDATE SET model_hash = ?4, updated_at = ?5",
        params![tenant, name, alias, model_hash, now_unix],
    )
    .map_err(|e| TrainError::other(format!("upsert model pointer: {e}")))?;
    tx.execute(
        "INSERT INTO model_pointer_history (tenant, name, alias, model_hash, moved_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![tenant, name, alias, model_hash, now_unix],
    )
    .map_err(|e| TrainError::other(format!("append model history: {e}")))?;
    tx.commit()
        .map_err(|e| TrainError::other(format!("promote commit: {e}")))?;
    Ok(())
}

/// Roll an alias back to its previous target atomically (one `BEGIN IMMEDIATE`
/// transaction): read the two most-recent history entries, set the pointer to
/// the second-most-recent, record the rollback. Errors if the alias has no prior
/// target. Returns the hash rolled back to.
pub fn rollback(
    conn: &mut Connection,
    tenant: &str,
    name: &str,
    alias: &str,
    now_unix: i64,
) -> Result<String> {
    // Take the write lock up front so the two-most-recent read and the writes see
    // ONE snapshot — a concurrent promote/rollback can't stale `prev`.
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| TrainError::other(format!("rollback txn: {e}")))?;
    let recent: Vec<String> = {
        let mut stmt = tx
            .prepare(
                "SELECT model_hash FROM model_pointer_history \
                 WHERE tenant = ?1 AND name = ?2 AND alias = ?3 ORDER BY seq DESC LIMIT 2",
            )
            .map_err(|e| TrainError::other(format!("model history query: {e}")))?;
        let rows = stmt
            .query_map(params![tenant, name, alias], |row| row.get::<_, String>(0))
            .map_err(|e| TrainError::other(format!("model history rows: {e}")))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| TrainError::other(format!("model history collect: {e}")))?
    };
    if recent.len() < 2 {
        return Err(TrainError::other(format!(
            "rollback refused — 'model://{name}@{alias}' has no prior target"
        )));
    }
    let prev = recent[1].clone();
    // Defense-in-depth (matches promote): the prior target must belong to THIS
    // tenant + model. promote already blocks a foreign hash from the trail, so
    // this only fails on a corrupted/hand-edited DB — fail closed rather than
    // silently re-point across the clinical boundary (ADR 0061).
    match get_model(&tx, &prev)? {
        Some(m) if m.tenant == tenant && m.name == name => {}
        _ => {
            return Err(TrainError::other(format!(
                "rollback refused — prior target {prev} is not a '{tenant}'/'{name}' model"
            )));
        }
    }
    tx.execute(
        "UPDATE model_pointers SET model_hash = ?4, updated_at = ?5 \
         WHERE tenant = ?1 AND name = ?2 AND alias = ?3",
        params![tenant, name, alias, prev, now_unix],
    )
    .map_err(|e| TrainError::other(format!("rollback model pointer: {e}")))?;
    tx.execute(
        "INSERT INTO model_pointer_history (tenant, name, alias, model_hash, moved_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![tenant, name, alias, prev, now_unix],
    )
    .map_err(|e| TrainError::other(format!("rollback model history: {e}")))?;
    tx.commit()
        .map_err(|e| TrainError::other(format!("rollback commit: {e}")))?;
    Ok(prev)
}

/// The checkpoint hash an alias currently resolves to (`None` if unset).
pub fn resolve_pointer(
    conn: &Connection,
    tenant: &str,
    name: &str,
    alias: &str,
) -> Result<Option<String>> {
    let r = conn.query_row(
        "SELECT model_hash FROM model_pointers WHERE tenant = ?1 AND name = ?2 AND alias = ?3",
        params![tenant, name, alias],
        |row| row.get::<_, String>(0),
    );
    match r {
        Ok(h) => Ok(Some(h)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(TrainError::other(format!("resolve model pointer: {e}"))),
    }
}

/// An alias's full audit trail, oldest first.
pub fn history(
    conn: &Connection,
    tenant: &str,
    name: &str,
    alias: &str,
) -> Result<Vec<HistoryEntry>> {
    let mut stmt = conn
        .prepare(
            "SELECT model_hash, moved_at FROM model_pointer_history \
             WHERE tenant = ?1 AND name = ?2 AND alias = ?3 ORDER BY seq ASC",
        )
        .map_err(|e| TrainError::other(format!("model history prepare: {e}")))?;
    let rows = stmt
        .query_map(params![tenant, name, alias], |row| {
            Ok(HistoryEntry {
                model_hash: row.get(0)?,
                moved_at: row.get(1)?,
            })
        })
        .map_err(|e| TrainError::other(format!("model history query: {e}")))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| TrainError::other(format!("model history collect: {e}")))
}

/// Parse a `model://<name>@<alias>` URI into `(name, alias)`. Returns `None` for
/// any non-model string (so a dispatcher can fall through) or a name/alias with
/// characters outside `[A-Za-z0-9_.-]`, or a missing `@alias`. Exactly one `@`
/// separates the two (neither part may contain `@`).
pub fn parse_model_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri.strip_prefix("model://")?;
    let (name, alias) = rest.split_once('@')?;
    if is_safe_ident(name) && is_safe_ident(alias) {
        Some((name, alias))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn db() -> Connection {
        // A private in-memory DB per test — no shared file, no env.
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(CREATE_SCHEMA).unwrap();
        c
    }

    #[test]
    fn register_validates_hash_and_name() {
        let c = db();
        assert!(register(&c, "nothex", "enc", SHARED_TENANT, None, 1).is_err());
        assert!(register(&c, HASH_A, "bad name", SHARED_TENANT, None, 1).is_err());
        assert!(register(&c, HASH_A, "enc-v1", SHARED_TENANT, None, 1).is_ok());
    }

    #[test]
    fn register_is_idempotent_but_refuses_collisions() {
        let c = db();
        register(&c, HASH_A, "enc", SHARED_TENANT, None, 1).unwrap();
        // Same hash+tenant+name → idempotent.
        assert!(register(&c, HASH_A, "enc", SHARED_TENANT, None, 2).is_ok());
        // Same hash, different NAME → refused (a hash is one model identity).
        assert!(register(&c, HASH_A, "dec", SHARED_TENANT, None, 3).is_err());
        // Same hash, different TENANT → refused (ADR 0061 boundary).
        assert!(register(&c, HASH_A, "enc", RESTRICTED_TENANT, None, 4).is_err());
    }

    #[test]
    fn garbled_tenant_is_refused() {
        // ADR 0096 tenant grammar: a stray/garbled tenant must never create a
        // namespace or a garbled audit row.
        let mut c = db();
        assert!(register(&c, HASH_A, "enc", "with space", None, 1).is_err());
        assert!(register(&c, HASH_A, "enc", "../evil", None, 1).is_err());
        // A well-formed project/domain tenant is accepted.
        assert!(register(&c, HASH_A, "enc", "research/prod", None, 1).is_ok());
        assert!(promote(&mut c, HASH_A, "bad tenant", "enc", "prod", 2).is_err());
    }

    #[test]
    fn promote_resolve_rollback_roundtrip() {
        let mut c = db();
        register(&c, HASH_A, "enc", SHARED_TENANT, None, 1).unwrap();
        register(&c, HASH_B, "enc", SHARED_TENANT, None, 2).unwrap();
        // Promote A→@prod, then B→@prod.
        promote(&mut c, HASH_A, SHARED_TENANT, "enc", "prod", 10).unwrap();
        promote(&mut c, HASH_B, SHARED_TENANT, "enc", "prod", 20).unwrap();
        assert_eq!(
            resolve_pointer(&c, SHARED_TENANT, "enc", "prod")
                .unwrap()
                .as_deref(),
            Some(HASH_B)
        );
        // Rollback pops @prod back to A.
        let rolled = rollback(&mut c, SHARED_TENANT, "enc", "prod", 30).unwrap();
        assert_eq!(rolled, HASH_A);
        assert_eq!(
            resolve_pointer(&c, SHARED_TENANT, "enc", "prod")
                .unwrap()
                .as_deref(),
            Some(HASH_A)
        );
        // History is the full audit trail, oldest first: A, B, A.
        let h: Vec<String> = history(&c, SHARED_TENANT, "enc", "prod")
            .unwrap()
            .into_iter()
            .map(|e| e.model_hash)
            .collect();
        assert_eq!(h, vec![HASH_A, HASH_B, HASH_A]);
    }

    #[test]
    fn aliases_are_independent() {
        let mut c = db();
        register(&c, HASH_A, "enc", SHARED_TENANT, None, 1).unwrap();
        register(&c, HASH_B, "enc", SHARED_TENANT, None, 2).unwrap();
        promote(&mut c, HASH_A, SHARED_TENANT, "enc", "staging", 10).unwrap();
        promote(&mut c, HASH_B, SHARED_TENANT, "enc", "prod", 11).unwrap();
        assert_eq!(
            resolve_pointer(&c, SHARED_TENANT, "enc", "staging")
                .unwrap()
                .as_deref(),
            Some(HASH_A)
        );
        assert_eq!(
            resolve_pointer(&c, SHARED_TENANT, "enc", "prod")
                .unwrap()
                .as_deref(),
            Some(HASH_B)
        );
    }

    #[test]
    fn promote_refuses_cross_tenant_and_wrong_name() {
        let mut c = db();
        register(&c, HASH_A, "enc", RESTRICTED_TENANT, None, 1).unwrap();
        // A restricted checkpoint can't be promoted onto a shared-tenant pointer.
        assert!(promote(&mut c, HASH_A, SHARED_TENANT, "enc", "prod", 10).is_err());
        // Nor onto a pointer whose name doesn't match the registered model.
        assert!(promote(&mut c, HASH_A, RESTRICTED_TENANT, "other", "prod", 11).is_err());
        // The matching promote succeeds.
        assert!(promote(&mut c, HASH_A, RESTRICTED_TENANT, "enc", "prod", 12).is_ok());
    }

    #[test]
    fn rollback_without_prior_is_refused() {
        let mut c = db();
        register(&c, HASH_A, "enc", SHARED_TENANT, None, 1).unwrap();
        promote(&mut c, HASH_A, SHARED_TENANT, "enc", "prod", 10).unwrap();
        // Only ONE entry in the trail → no prior to roll back to.
        assert!(rollback(&mut c, SHARED_TENANT, "enc", "prod", 20).is_err());
    }

    #[test]
    fn parse_model_uri_grammar() {
        assert_eq!(
            parse_model_uri("model://enc-v1@prod"),
            Some(("enc-v1", "prod"))
        );
        assert_eq!(parse_model_uri("model://enc@v7"), Some(("enc", "v7")));
        assert_eq!(parse_model_uri("registry://plan@prod"), None); // wrong scheme
        assert_eq!(parse_model_uri("model://enc"), None); // no @alias
        assert_eq!(parse_model_uri("model://enc@"), None); // empty alias
        assert_eq!(parse_model_uri("model://a b@prod"), None); // unsafe name
        assert_eq!(parse_model_uri("model://enc@pr od"), None); // unsafe alias
    }
}
