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
/// The tenant deployments were published under by default before the CLI
/// moved to `default` (the tenant every other verb uses). Records made under it
/// stay addressable with an explicit `--tenant shared`.
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
    moved_at         INTEGER NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'promote'
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
/// `$BLUT_REGISTRY_DB` overrides it (tests, alternate homes), else
/// `$LAMU_TRAIN_DATA_DIR/registry.db` (see [`crate::paths`]).
pub fn registry_db_path() -> Result<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("BLUT_REGISTRY_DB") {
        return Ok(std::path::PathBuf::from(p));
    }
    if let Some(root) = crate::paths::state_root() {
        std::fs::create_dir_all(&root).map_err(|e| TrainError::Io {
            path: root.clone(),
            source: e,
        })?;
        return Ok(root.join("registry.db"));
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
    // Concurrent CLI invocations (two `promote`s, a promote racing a rollback,
    // or a web reader) must WAIT for the writer lock, not fail with a raw
    // "database is locked": without a busy timeout SQLite returns SQLITE_BUSY
    // immediately, before the IMMEDIATE transactions below ever get to serialize.
    conn.busy_timeout(std::time::Duration::from_secs(10))
        .map_err(|e| TrainError::other(format!("busy_timeout: {e}")))?;
    // Best-effort WAL; a failure just leaves the default journal mode.
    let _ =
        conn.query_row::<rusqlite::types::Value, _, _>("PRAGMA journal_mode=WAL", [], |r| r.get(0));
    conn.execute_batch(CREATE_SCHEMA)
        .map_err(|e| TrainError::other(format!("create registry schema: {e}")))?;
    migrate_history_kind(&conn)?;
    Ok(conn)
}

/// Add `pointer_history.kind` (`promote` | `rollback`) to a registry created
/// before rollbacks walked the history. Rows written before it are counted as
/// promotions: the old trail did not record which moves were rollbacks.
fn migrate_history_kind(conn: &Connection) -> Result<()> {
    add_move_kind_column(conn, "pointer_history")
}

/// Add a `kind TEXT NOT NULL DEFAULT 'promote'` column to an audit-trail
/// `table` that predates it; a no-op when the column exists. Shared by the plan
/// trail and the model registry's (`model_pointer_history`), which live in the
/// same database. `table` is a fixed name from this crate, never user input.
pub(crate) fn add_move_kind_column(conn: &Connection, table: &'static str) -> Result<()> {
    let has_kind = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .and_then(|mut stmt| {
            stmt.query_map([], |row| row.get::<_, String>(1))?
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .map_err(|e| TrainError::other(format!("{table} columns: {e}")))?
        .iter()
        .any(|c| c == "kind");
    if !has_kind {
        conn.execute(
            &format!("ALTER TABLE {table} ADD COLUMN kind TEXT NOT NULL DEFAULT 'promote'"),
            [],
        )
        .map_err(|e| TrainError::other(format!("add {table}.kind: {e}")))?;
    }
    Ok(())
}

/// A pointer's promotion stack, replayed from its audit trail: a promotion
/// pushes its fingerprint (unless it is already on top), a rollback pops. The
/// top is what the pointer resolves to; a rollback moves to the entry below.
fn promotion_stack(conn: &Connection, tenant: &str, name: &str) -> Result<Vec<String>> {
    let moves = history_moves(conn, tenant, name)?;
    Ok(replay_promotion_stack(
        moves
            .into_iter()
            .map(|(e, kind)| (e.plan_fingerprint, kind)),
    ))
}

/// Replay an audit trail (oldest first) as an undo stack: a promotion pushes
/// its target unless that target is already on top (re-promoting the current
/// target is not a new step), a rollback pops. The top is the current target.
/// Shared by plan pointers and model aliases.
pub(crate) fn replay_promotion_stack(
    moves: impl IntoIterator<Item = (String, MoveKind)>,
) -> Vec<String> {
    let mut stack: Vec<String> = Vec::new();
    for (target, kind) in moves {
        match kind {
            MoveKind::Rollback => {
                stack.pop();
            }
            MoveKind::Promote => {
                if stack.last() != Some(&target) {
                    stack.push(target);
                }
            }
        }
    }
    stack
}

/// How a pointer moved, as recorded in `pointer_history.kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoveKind {
    Promote,
    Rollback,
}

impl MoveKind {
    /// The kind a trail row's `kind` column names; anything but `rollback`
    /// (including rows migrated from before the column existed) is a promotion.
    pub(crate) fn from_column(kind: &str) -> Self {
        if kind == "rollback" {
            MoveKind::Rollback
        } else {
            MoveKind::Promote
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            MoveKind::Promote => "promote",
            MoveKind::Rollback => "rollback",
        }
    }
}

/// A pointer's audit trail with the kind of each move, oldest first.
pub(crate) fn history_moves(
    conn: &Connection,
    tenant: &str,
    name: &str,
) -> Result<Vec<(HistoryEntry, MoveKind)>> {
    let mut stmt = conn
        .prepare(
            "SELECT plan_fingerprint, moved_at, kind FROM pointer_history \
             WHERE tenant = ?1 AND name = ?2 ORDER BY seq ASC",
        )
        .map_err(|e| TrainError::other(format!("history prepare: {e}")))?;
    let rows = stmt
        .query_map(params![tenant, name], |row| {
            let kind: String = row.get(2)?;
            Ok((
                HistoryEntry {
                    plan_fingerprint: row.get(0)?,
                    moved_at: row.get(1)?,
                },
                MoveKind::from_column(&kind),
            ))
        })
        .map_err(|e| TrainError::other(format!("history query: {e}")))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| TrainError::other(format!("history collect: {e}")))
}

/// The ADR-0078 fingerprint that keys a spec in `deployments` — deterministic
/// over the spec's canonical bytes, so re-publishing identical bytes yields the
/// same id (idempotent).
///
/// This must stay a pure function of `spec`. `blut plan check` reports it
/// without holding a database handle, so if it ever came to depend on stored
/// state — a registry version, a per-tenant salt — check would report an id
/// that publish does not key by, and every recorded id would be a dead
/// reference. Keying on anything but spec content also breaks the idempotence
/// this function exists to provide.
pub fn fingerprint(spec: &PlanSpec) -> String {
    spec.provenance_fingerprint("", &serde_json::Value::Null)
        .to_hex()
}

/// The fail-closed typecheck that gates entry to the `deployments` table: a
/// spec must resolve against the compiled-in stages and kind-check as a graph.
///
/// Returns the compiled plan, so a caller that intends to RUN what it just
/// checked executes that object rather than compiling a second time and hoping
/// the two agree. `publish` discards it; `plan run` uses it.
///
/// Factored out so `blut plan check` can ask exactly this question instead of
/// a copy of it. That is the whole point — a pre-flight check that
/// *approximates* the deploy-time gate is worse than none, because a spec can
/// then pass review and be refused at deploy, which is the failure a pre-flight
/// check exists to prevent. Anything that must hold before a spec is publishable
/// belongs in here, not in [`publish`]'s body, or the two drift apart again.
pub fn typecheck(reg: &Registry, spec: &PlanSpec) -> Result<crate::framework::plan::CompiledPlan> {
    spec.compile(reg)
        .map_err(|e| TrainError::other(format!("PlanSpec does not typecheck: {e}")))
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
    // The compiled plan is discarded: publishing stores bytes, not a plan.
    typecheck(reg, spec).map_err(|e| TrainError::other(format!("publish refused — {e}")))?;
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

/// Resolve a deployment fingerprint from the full id or an unambiguous prefix,
/// the way other commands take job ids and content hashes. A full id matches
/// itself; a prefix must match exactly one published deployment (across all
/// tenants, so a prefix never silently means different deployments for
/// different tenants). No match, several matches, or a query that is not hex
/// is refused: guessing here would promote the wrong graph.
pub fn resolve_fingerprint(conn: &Connection, query: &str) -> Result<String> {
    let query = query.trim();
    if query.is_empty() || !query.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(TrainError::other(format!(
            "'{query}' is not a deployment fingerprint or prefix (expected hex digits)"
        )));
    }
    let query = query.to_ascii_lowercase();
    if get_deployment(conn, &query)?.is_some() {
        return Ok(query);
    }
    let mut stmt = conn
        .prepare(
            "SELECT plan_fingerprint FROM deployments \
             WHERE substr(plan_fingerprint, 1, length(?1)) = ?1 \
             ORDER BY plan_fingerprint LIMIT 2",
        )
        .map_err(|e| TrainError::other(format!("fingerprint prefix prepare: {e}")))?;
    let matches = stmt
        .query_map(params![query], |row| row.get::<_, String>(0))
        .map_err(|e| TrainError::other(format!("fingerprint prefix query: {e}")))?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| TrainError::other(format!("fingerprint prefix collect: {e}")))?;
    match matches.as_slice() {
        [] => Err(TrainError::other(format!(
            "no deployment matches fingerprint '{query}' (see `blut plan publish`)"
        ))),
        [only] => Ok(only.clone()),
        _ => Err(TrainError::other(format!(
            "fingerprint prefix '{query}' is ambiguous: more than one deployment matches; \
             give more characters"
        ))),
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

/// Roll a pointer back one step through its promotion history, atomically (a
/// single `BEGIN IMMEDIATE` transaction). Rollbacks walk the history like an
/// undo stack: each one moves to the promotion before the current one, so
/// promote A, B, C then roll back twice lands on A. A promotion after a
/// rollback starts from where the rollback left the pointer (the rolled-back
/// promotions are not reachable again by rolling back). Errors when the
/// pointer is at its first promotion: there is nothing earlier. The move is
/// recorded in the trail as a `rollback`. Returns the fingerprint rolled back
/// to.
///
/// It used to read only the two most recent trail entries, so a second
/// rollback returned to the target the first one had left: repeated rollbacks
/// toggled between the last two deployments.
pub fn rollback(conn: &mut Connection, tenant: &str, name: &str, now_unix: i64) -> Result<String> {
    // BEGIN IMMEDIATE: take the write lock up front so the stack replay and the
    // pointer/history writes see ONE consistent snapshot — a concurrent
    // promote/rollback can't slip between the read and the write.
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| TrainError::other(format!("rollback txn: {e}")))?;
    let stack = promotion_stack(&tx, tenant, name)?;
    let prev = match stack.as_slice() {
        [] => {
            return Err(TrainError::other(format!(
                "rollback refused — pointer 'registry://plan@{name}' has no deployment for \
                 tenant '{tenant}'"
            )));
        }
        [only] => {
            return Err(TrainError::other(format!(
                "rollback refused — pointer 'registry://plan@{name}' is at its first \
                 promotion ({only}); there is nothing earlier to roll back to"
            )));
        }
        [.., prev, _current] => prev.clone(),
    };
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
        "INSERT INTO pointer_history (tenant, name, plan_fingerprint, moved_at, kind) \
         VALUES (?1, ?2, ?3, ?4, 'rollback')",
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

/// The error for a pointer absent under `tenant`. When the same name exists
/// under another non-restricted tenant (typically `shared`, the former
/// default), say which and how to address it, instead of a bare "no pointer".
/// Restricted tenants are never named: that would disclose a clinical record
/// across the boundary.
fn missing_pointer(conn: &Connection, tenant: &str, name: &str) -> TrainError {
    let others: Vec<String> = conn
        .prepare("SELECT tenant FROM deployment_pointers WHERE name = ?1 AND tenant != ?2")
        .and_then(|mut stmt| {
            stmt.query_map(params![name, tenant], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|t| crate::tenant::Tenant::parse(t).is_some_and(|t| !t.is_restricted()))
        .collect();
    match others.as_slice() {
        [] => TrainError::other(format!(
            "no deployment pointer 'registry://plan@{name}' for tenant '{tenant}'"
        )),
        [other, ..] => TrainError::other(format!(
            "no deployment pointer 'registry://plan@{name}' for tenant '{tenant}'; tenant \
             '{other}' has one (pass --tenant {other})"
        )),
    }
}

/// Resolve a pointer all the way to its frozen `PlanSpec` (the run-dispatch
/// path for `registry://plan@<name>`).
pub fn resolve_spec(conn: &Connection, tenant: &str, name: &str) -> Result<PlanSpec> {
    let fp = match resolve_pointer(conn, tenant, name)? {
        Some(fp) => fp,
        None => return Err(missing_pointer(conn, tenant, name)),
    };
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
