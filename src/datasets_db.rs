//! Datasets registry — `datasets` table alongside LAMU's
//! `conversations.db`.
//!
//! Lamu-train owns this table's schema. Lamu-mcp owns
//! `conversations` + `turns` and is unaware of `datasets`. Sharing
//! the file is convenient (one SQLite file in
//! `~/.local/share/lamu/`); each crate runs its own
//! `CREATE TABLE IF NOT EXISTS` at open so neither has to depend
//! on the other.
//!
//! Concurrency: SQLite's WAL journal mode (set by lamu-mcp at DB
//! creation) tolerates concurrent readers + a single writer.
//! Lamu-train writes during `data add` / auto-register from
//! `--from-conversations`; lamu-mcp writes during chat sessions.
//! They rarely overlap in time, and on overlap SQLite's lockfile
//! serializes the writes — no corruption, just a brief block.
//!
//! Why sha256 dedup is a warning (not a refusal): the user might
//! legitimately register the same JSONL under multiple names
//! (e.g. before/after a curation pass). Refusing would force
//! `--force` flags. Warning preserves the lineage record while
//! flagging the duplicate.

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Result, TrainError};

const CREATE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS datasets (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    n_examples   INTEGER NOT NULL,
    n_tokens     INTEGER,
    created_at   INTEGER NOT NULL,
    metadata     TEXT
);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);
CREATE INDEX IF NOT EXISTS idx_datasets_sha256 ON datasets(sha256);
";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub id: String,
    pub name: String,
    /// Free-form tag (`"jsonl"`, `"conversations"`, `"hf-pull"`, ...).
    pub kind: String,
    pub source_path: PathBuf,
    pub sha256: String,
    pub n_examples: i64,
    pub n_tokens: Option<i64>,
    /// UNIX seconds.
    pub created_at: i64,
    /// Freeform JSON object as a string. Caller's responsibility to
    /// stay valid JSON; we don't parse here.
    pub metadata: Option<String>,
}

/// Compute sha256 of the file at `path`. Streams in 64 KiB blocks
/// so a 4 GiB JSONL doesn't balloon RSS.
pub fn compute_file_sha256(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| TrainError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| TrainError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Count newline-terminated examples in a JSONL file. One line =
/// one example; trailing empty lines ignored. Streams; doesn't
/// load the file into memory.
pub fn count_jsonl_examples(path: &Path) -> Result<i64> {
    use std::io::{BufRead, BufReader};
    let f = std::fs::File::open(path).map_err(|e| TrainError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let reader = BufReader::new(f);
    let mut n = 0i64;
    for line in reader.lines() {
        let line = line.map_err(|e| TrainError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if !line.trim().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

/// Path to the datasets registry — same file LAMU's memory uses.
/// Override via `$LAMU_MEMORY_DB` (matches `conversations::memory_db_path`).
pub fn registry_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LAMU_MEMORY_DB") {
        return Ok(PathBuf::from(p));
    }
    let dir = dirs::data_local_dir()
        .ok_or_else(|| TrainError::other(
            "data_local_dir() unavailable; set $LAMU_MEMORY_DB",
        ))?
        .join("lamu");
    std::fs::create_dir_all(&dir).map_err(|e| TrainError::Io {
        path: dir.clone(),
        source: e,
    })?;
    Ok(dir.join("conversations.db"))
}

/// Open the datasets table (creates it idempotently). Read+write.
/// Caller is responsible for ensuring `path`'s parent dir exists
/// when using `open_at`; the canonical `open()` handles that.
pub fn open() -> Result<Connection> {
    open_at(&registry_path()?)
}

pub fn open_at(path: &Path) -> Result<Connection> {
    // Use the default mutex mode (SERIALIZED) — connections may
    // be shared across threads in future callers, and the perf
    // hit of an extra rwlock per query is negligible for a
    // few-rows-per-minute table.
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )
    .map_err(|e| TrainError::other(format!("open {}: {e}", path.display())))?;
    // WAL — same mode lamu-mcp::memory sets. Idempotent. Retry
    // on SQLITE_BUSY: a concurrent startup with lamu-mcp can
    // briefly hold the writer lock during its own pragma calls.
    apply_pragma_with_retry(&conn, "PRAGMA journal_mode=WAL");
    apply_pragma_with_retry(&conn, "PRAGMA synchronous=NORMAL");
    conn.execute_batch(CREATE_SCHEMA)
        .map_err(|e| TrainError::other(format!("create datasets schema: {e}")))?;
    Ok(conn)
}

fn apply_pragma_with_retry(conn: &Connection, sql: &str) {
    use std::time::Duration;
    let mut tries = 0u32;
    loop {
        let r: rusqlite::Result<rusqlite::types::Value> =
            conn.query_row(sql, [], |row| row.get(0));
        match r {
            Ok(_) => return,
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::DatabaseBusy && tries < 5 =>
            {
                std::thread::sleep(Duration::from_millis(50 << tries));
                tries += 1;
            }
            Err(e) => {
                tracing::warn!("pragma '{sql}' failed: {e}; continuing with defaults");
                return;
            }
        }
    }
}

/// Build a `DatasetRecord` for a JSONL file on disk. Computes
/// sha256, counts examples, leaves `n_tokens` as `None` (the
/// trainer fills it in post-tokenize if it cares).
pub fn record_from_jsonl(
    name: impl Into<String>,
    path: &Path,
    kind: impl Into<String>,
    metadata: Option<String>,
) -> Result<DatasetRecord> {
    let name = name.into();
    if !is_safe_dataset_name(&name) {
        return Err(TrainError::other(format!(
            "dataset name '{name}' must match [A-Za-z0-9_.-]+ \
             with no leading '.' or '-' and no '..' substring"
        )));
    }
    if !path.exists() {
        return Err(TrainError::DatasetUnresolvable(format!(
            "dataset file not found: {}",
            path.display()
        )));
    }
    Ok(DatasetRecord {
        id: uuid::Uuid::new_v4().to_string(),
        name,
        kind: kind.into(),
        source_path: path.to_path_buf(),
        sha256: compute_file_sha256(path)?,
        n_examples: count_jsonl_examples(path)?,
        n_tokens: None,
        created_at: now_unix(),
        metadata,
    })
}

/// Insert a record. Refuses on duplicate `name`. If `sha256`
/// already exists under a different name, logs a warning and
/// inserts anyway (different name = different lineage record;
/// the user might want to register the same content twice
/// deliberately).
pub fn add(conn: &Connection, rec: &DatasetRecord) -> Result<()> {
    if let Some(existing) = get_by_sha256(conn, &rec.sha256)? {
        if existing.name != rec.name {
            tracing::warn!(
                "dataset sha256 {} already registered as '{}' \
                 (now also as '{}'); accepting both",
                rec.sha256,
                existing.name,
                rec.name
            );
        }
    }
    let path_str = rec.source_path.to_string_lossy().into_owned();
    let r = conn.execute(
        "INSERT INTO datasets \
            (id, name, kind, source_path, sha256, n_examples, n_tokens, created_at, metadata) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        params![
            rec.id,
            rec.name,
            rec.kind,
            path_str,
            rec.sha256,
            rec.n_examples,
            rec.n_tokens,
            rec.created_at,
            rec.metadata,
        ],
    );
    match r {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(e, _))
            if e.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Err(TrainError::other(format!(
                "dataset '{}' already registered. Use `lamu-train data rm {}` first \
                 or pick a unique name.",
                rec.name, rec.name
            )))
        }
        Err(e) => Err(TrainError::other(format!("INSERT dataset: {e}"))),
    }
}

pub fn get_by_name(conn: &Connection, name: &str) -> Result<Option<DatasetRecord>> {
    let r = conn.query_row(
        "SELECT id, name, kind, source_path, sha256, n_examples, n_tokens, created_at, metadata \
         FROM datasets WHERE name = ?",
        params![name],
        row_to_record,
    );
    match r {
        Ok(rec) => Ok(Some(rec)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(TrainError::other(format!("get_by_name: {e}"))),
    }
}

pub fn get_by_sha256(conn: &Connection, sha256: &str) -> Result<Option<DatasetRecord>> {
    let r = conn.query_row(
        "SELECT id, name, kind, source_path, sha256, n_examples, n_tokens, created_at, metadata \
         FROM datasets WHERE sha256 = ? LIMIT 1",
        params![sha256],
        row_to_record,
    );
    match r {
        Ok(rec) => Ok(Some(rec)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(TrainError::other(format!("get_by_sha256: {e}"))),
    }
}

pub fn list(conn: &Connection) -> Result<Vec<DatasetRecord>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, name, kind, source_path, sha256, n_examples, n_tokens, created_at, metadata \
             FROM datasets ORDER BY created_at DESC, name ASC",
        )
        .map_err(|e| TrainError::other(format!("prepare list: {e}")))?;
    let rows = stmt
        .query_map([], row_to_record)
        .map_err(|e| TrainError::other(format!("query list: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| TrainError::other(format!("row decode: {e}")))?);
    }
    Ok(out)
}

pub fn remove(conn: &Connection, name: &str) -> Result<bool> {
    let n = conn
        .execute("DELETE FROM datasets WHERE name = ?", params![name])
        .map_err(|e| TrainError::other(format!("DELETE: {e}")))?;
    Ok(n > 0)
}

fn row_to_record(r: &rusqlite::Row) -> rusqlite::Result<DatasetRecord> {
    Ok(DatasetRecord {
        id: r.get(0)?,
        name: r.get(1)?,
        kind: r.get(2)?,
        source_path: PathBuf::from(r.get::<_, String>(3)?),
        sha256: r.get(4)?,
        n_examples: r.get(5)?,
        n_tokens: r.get(6)?,
        created_at: r.get(7)?,
        metadata: r.get(8)?,
    })
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn is_safe_dataset_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.starts_with('-')
        && !name.contains("..")
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_jsonl(path: &Path, n_lines: usize) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        for i in 0..n_lines {
            writeln!(
                f,
                "{{\"messages\":[{{\"role\":\"user\",\"content\":\"line {i}\"}}]}}"
            )
            .unwrap();
        }
    }

    #[test]
    fn open_creates_schema() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        // Insert a record; if the schema isn't there this errors.
        let n = list(&conn).unwrap();
        assert!(n.is_empty());
    }

    #[test]
    fn add_then_list_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        let f = td.path().join("data.jsonl");
        make_jsonl(&f, 3);
        let rec =
            record_from_jsonl("ds-a", &f, "jsonl", Some("{}".into())).unwrap();
        add(&conn, &rec).unwrap();
        let listed = list(&conn).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "ds-a");
        assert_eq!(listed[0].n_examples, 3);
        assert_eq!(listed[0].sha256.len(), 64);
    }

    #[test]
    fn add_refuses_duplicate_name() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        let f1 = td.path().join("data1.jsonl");
        let f2 = td.path().join("data2.jsonl");
        make_jsonl(&f1, 2);
        make_jsonl(&f2, 5);
        let r1 = record_from_jsonl("dup", &f1, "jsonl", None).unwrap();
        let r2 = record_from_jsonl("dup", &f2, "jsonl", None).unwrap();
        add(&conn, &r1).unwrap();
        let err = add(&conn, &r2).expect_err("dup name must reject");
        assert!(format!("{err}").contains("'dup'"));
    }

    #[test]
    fn add_accepts_same_sha_under_different_name() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        let f = td.path().join("data.jsonl");
        make_jsonl(&f, 4);
        // Build two records with same content (so same sha) but
        // different names. record_from_jsonl assigns a fresh uuid
        // each call so the primary keys don't collide.
        let r1 = record_from_jsonl("alpha", &f, "jsonl", None).unwrap();
        let r2 = record_from_jsonl("bravo", &f, "jsonl", None).unwrap();
        assert_eq!(r1.sha256, r2.sha256);

        add(&conn, &r1).unwrap();
        add(&conn, &r2).expect("same sha + different name = warn + accept");
        assert_eq!(list(&conn).unwrap().len(), 2);
    }

    #[test]
    fn get_by_name_returns_record() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        let f = td.path().join("data.jsonl");
        make_jsonl(&f, 2);
        let r = record_from_jsonl("findme", &f, "jsonl", None).unwrap();
        add(&conn, &r).unwrap();
        let got = get_by_name(&conn, "findme").unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().n_examples, 2);
    }

    #[test]
    fn get_by_name_missing_returns_none() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        assert!(get_by_name(&conn, "nope").unwrap().is_none());
    }

    #[test]
    fn remove_returns_true_when_exists() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        let f = td.path().join("data.jsonl");
        make_jsonl(&f, 1);
        let r = record_from_jsonl("kill", &f, "jsonl", None).unwrap();
        add(&conn, &r).unwrap();
        assert!(remove(&conn, "kill").unwrap());
        assert!(get_by_name(&conn, "kill").unwrap().is_none());
    }

    #[test]
    fn remove_returns_false_when_missing() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        assert!(!remove(&conn, "nope").unwrap());
    }

    #[test]
    fn record_from_jsonl_rejects_unsafe_name() {
        let td = tempfile::tempdir().unwrap();
        let f = td.path().join("data.jsonl");
        make_jsonl(&f, 1);
        for bad in ["", ".hidden", "-leading", "a..b", "a/b"] {
            let r = record_from_jsonl(bad, &f, "jsonl", None);
            assert!(r.is_err(), "name '{bad}' should reject");
        }
    }

    #[test]
    fn record_from_jsonl_rejects_missing_file() {
        let r = record_from_jsonl(
            "test",
            Path::new("/tmp/lamu-train-defs-nonexistent-xyz"),
            "jsonl",
            None,
        );
        assert!(r.is_err());
    }

    #[test]
    fn empty_jsonl_counts_zero_examples() {
        let td = tempfile::tempdir().unwrap();
        let f = td.path().join("empty.jsonl");
        std::fs::write(&f, "\n\n   \n").unwrap();
        assert_eq!(count_jsonl_examples(&f).unwrap(), 0);
    }

    #[test]
    fn sha256_stable_across_calls() {
        let td = tempfile::tempdir().unwrap();
        let f = td.path().join("data.jsonl");
        make_jsonl(&f, 10);
        let h1 = compute_file_sha256(&f).unwrap();
        let h2 = compute_file_sha256(&f).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn list_orders_newest_first() {
        let td = tempfile::tempdir().unwrap();
        let conn = open_at(&td.path().join("test.db")).unwrap();
        let f = td.path().join("data.jsonl");
        make_jsonl(&f, 1);
        for (name, ts) in [("a", 100i64), ("b", 200), ("c", 50)] {
            let mut r = record_from_jsonl(name, &f, "jsonl", None).unwrap();
            r.created_at = ts;
            add(&conn, &r).unwrap();
        }
        let listed = list(&conn).unwrap();
        let names: Vec<&str> = listed.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["b", "a", "c"]);
    }
}
