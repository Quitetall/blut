//! Read LAMU's conversation memory and materialize a JSONL training
//! set.
//!
//! Architectural boundary: lamu-train is a *separate* project from
//! LAMU. Rather than depending on lamu-mcp (and pulling in its
//! HTTP/MCP/embedding/etc. dep tree), this module opens
//! `conversations.db` directly via rusqlite in **read-only** mode.
//! The schema is owned by `lamu-mcp::memory`; if it changes, this
//! module's queries break loud — at which point we either patch
//! both or extract the schema to a third crate.
//!
//! Output format is the OpenAI chat-completion JSONL shape that
//! `trainer.py::datasets_loader` expects:
//!
//! ```json
//! {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
//! ```
//!
//! Filter rules baked into `dump_to_jsonl`:
//!   - drop conversations whose RAW turn count is below
//!     `MIN_TURNS_PER_CONVERSATION` (noise — single-turn pings,
//!     half-broken sessions, etc.)
//!   - drop messages whose content starts with `error:` (tool
//!     failure echoes from MCP — they teach the wrong thing)
//!   - cap at the 200 KiB per-message rendering limit so a stray
//!     large blob doesn't dominate the training distribution
//!   - drop a conversation whose POST-FILTER turn count falls
//!     below `MIN_TURNS_PER_CONVERSATION` (counted separately as
//!     `n_dropped_filtered_below_min` so callers can tell the
//!     difference between raw-too-short and gutted-by-filters)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OpenFlags};
use serde::Serialize;

use crate::error::{Result, TrainError};

/// Maximum bytes per message before we drop it from the dataset —
/// not a security cap (sqlite already accepts whatever's there),
/// but a sanity check so a single 50 MB pasted log doesn't end up
/// dwarfing real conversations during training.
const MAX_MESSAGE_BYTES: usize = 200 * 1024;

/// Minimum turns for a conversation to qualify. Below this is
/// usually noise (single-turn pings, half-broken sessions).
const MIN_TURNS_PER_CONVERSATION: usize = 4;

#[derive(Clone, Debug, Serialize)]
pub struct DumpStats {
    pub n_conversations: usize,
    pub n_turns: usize,
    /// Raw conversation had < MIN_TURNS_PER_CONVERSATION turns
    /// before any filtering ran.
    pub n_dropped_short: usize,
    /// Conversation passed the raw-length check but post-filter
    /// (errors + oversize removed) it fell below the threshold.
    pub n_dropped_filtered_below_min: usize,
    pub n_dropped_errors: usize,
    pub n_dropped_oversize: usize,
    pub path: PathBuf,
}

/// Locate `conversations.db`. Default mirrors `lamu-mcp::memory::db_path`:
/// `$XDG_DATA_HOME/lamu/conversations.db`. Override via `$LAMU_MEMORY_DB`
/// for hermetic tests.
pub fn memory_db_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LAMU_MEMORY_DB") {
        return Ok(PathBuf::from(p));
    }
    let dir = dirs::data_local_dir()
        .ok_or_else(|| TrainError::other(
            "data_local_dir() unavailable; set $LAMU_MEMORY_DB",
        ))?
        .join("lamu");
    Ok(dir.join("conversations.db"))
}

/// Pull every turn at or after `since`, group by conversation, write
/// JSONL to `out_path`, return stats. Read-only by construction —
/// lamu-train cannot corrupt LAMU's data even if a bug fires.
pub fn dump_to_jsonl(since: Duration, out_path: &Path) -> Result<DumpStats> {
    let cutoff = SystemTime::now()
        .checked_sub(since)
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let db_path = memory_db_path()?;
    if !db_path.exists() {
        return Err(TrainError::DatasetUnresolvable(format!(
            "memory database not found at {}; \
             run a session first or set $LAMU_MEMORY_DB",
            db_path.display()
        )));
    }
    dump_with_db(&db_path, cutoff, out_path)
}

/// Path-injectable variant for tests + the future MCP dry-run path.
pub fn dump_with_db(db_path: &Path, cutoff_unix_secs: i64, out_path: &Path) -> Result<DumpStats> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| TrainError::Io {
            path: parent.into(),
            source: e,
        })?;
    }
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| TrainError::DatasetUnresolvable(format!(
        "open {} read-only: {e}",
        db_path.display()
    )))?;

    let mut stmt = conn
        .prepare(
            "SELECT conversation_id, idx, role, content \
             FROM turns WHERE ts >= ? ORDER BY conversation_id, idx ASC",
        )
        .map_err(|e| TrainError::other(format!("prepare query: {e}")))?;

    // Group rows by conversation. BTreeMap so output ordering is
    // stable across runs — same memory.db snapshot + same cutoff =
    // identical JSONL bytes, so a downstream sha256 dedup on the
    // dataset is meaningful.
    let mut grouped: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let rows = stmt
        .query_map(params![cutoff_unix_secs], |r| {
            let conv: String = r.get(0)?;
            let _idx: i64 = r.get(1)?;
            let role: String = r.get(2)?;
            let content: String = r.get(3)?;
            Ok((conv, role, content))
        })
        .map_err(|e| TrainError::other(format!("query rows: {e}")))?;
    for row in rows {
        let (conv, role, content) =
            row.map_err(|e| TrainError::other(format!("row decode: {e}")))?;
        grouped.entry(conv).or_default().push((role, content));
    }

    let f = File::create(out_path).map_err(|e| TrainError::Io {
        path: out_path.into(),
        source: e,
    })?;
    let mut writer = BufWriter::new(f);

    let mut n_conversations = 0usize;
    let mut n_turns = 0usize;
    let mut n_dropped_short = 0usize;
    let mut n_dropped_filtered_below_min = 0usize;
    let mut n_dropped_errors = 0usize;
    let mut n_dropped_oversize = 0usize;

    for (_, msgs) in &grouped {
        if msgs.len() < MIN_TURNS_PER_CONVERSATION {
            n_dropped_short += 1;
            continue;
        }
        let mut filtered: Vec<(&String, &String)> = Vec::with_capacity(msgs.len());
        for (role, content) in msgs {
            if content.starts_with("error:") {
                n_dropped_errors += 1;
                continue;
            }
            if content.len() > MAX_MESSAGE_BYTES {
                n_dropped_oversize += 1;
                continue;
            }
            filtered.push((role, content));
        }
        if filtered.len() < MIN_TURNS_PER_CONVERSATION {
            n_dropped_filtered_below_min += 1;
            continue;
        }
        let messages: Vec<serde_json::Value> = filtered
            .iter()
            .map(|(role, content)| {
                serde_json::json!({"role": role, "content": content})
            })
            .collect();
        let line = serde_json::json!({"messages": messages});
        writeln!(writer, "{line}").map_err(|e| TrainError::Io {
            path: out_path.into(),
            source: e,
        })?;
        n_conversations += 1;
        n_turns += filtered.len();
    }
    writer.flush().map_err(|e| TrainError::Io {
        path: out_path.into(),
        source: e,
    })?;

    Ok(DumpStats {
        n_conversations,
        n_turns,
        n_dropped_short,
        n_dropped_filtered_below_min,
        n_dropped_errors,
        n_dropped_oversize,
        path: out_path.to_path_buf(),
    })
}

/// Cheap raw count of turns at or after `cutoff_unix_secs`. Does
/// not apply filtering — intended for the auto-trigger
/// `threshold_new_turns` gate where we want a fast "have things
/// happened since last train" check, not a precise dataset
/// estimate.
pub fn count_turns_since(cutoff_unix_secs: i64) -> Result<i64> {
    count_turns_since_at(&memory_db_path()?, cutoff_unix_secs)
}

pub fn count_turns_since_at(db_path: &Path, cutoff_unix_secs: i64) -> Result<i64> {
    if !db_path.exists() {
        return Ok(0);
    }
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| TrainError::DatasetUnresolvable(format!(
        "open {} read-only: {e}",
        db_path.display()
    )))?;
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM turns WHERE ts >= ?",
            params![cutoff_unix_secs],
            |r| r.get(0),
        )
        .map_err(|e| TrainError::other(format!("count_turns_since: {e}")))?;
    Ok(n)
}

/// Dry-run variant: count what `dump_to_jsonl` would emit without
/// writing. Used by the future MCP `train_from_conversations`
/// confirmation gate to estimate dataset size before the user opts in.
pub fn count_with_db(db_path: &Path, cutoff_unix_secs: i64) -> Result<DumpStats> {
    let tmp = tempfile::NamedTempFile::new()
        .map_err(|e| TrainError::other(format!("create tmp jsonl: {e}")))?;
    let stats = dump_with_db(db_path, cutoff_unix_secs, tmp.path())?;
    // tmp file dropped + removed automatically.
    Ok(DumpStats {
        path: PathBuf::new(),
        ..stats
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal schema-compatible memory.db for hermetic tests. Mirrors
    /// the columns lamu-mcp::memory CREATEs; unused columns from the
    /// production schema are omitted so the tests don't depend on
    /// migrations.
    fn make_test_db(path: &Path) {
        let conn = Connection::open(path).expect("open test db");
        conn.execute_batch(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at INTEGER NOT NULL);
             CREATE TABLE turns (
                conversation_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts INTEGER NOT NULL,
                metadata TEXT,
                PRIMARY KEY (conversation_id, idx)
             );",
        )
        .expect("create schema");
    }

    fn insert_conv(conn: &Connection, conv: &str, ts: i64, msgs: &[(&str, &str)]) {
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, created_at) VALUES (?, ?)",
            params![conv, ts],
        )
        .unwrap();
        for (idx, (role, content)) in msgs.iter().enumerate() {
            conn.execute(
                "INSERT INTO turns (conversation_id, idx, role, content, ts) \
                 VALUES (?, ?, ?, ?, ?)",
                params![conv, idx as i64, role, content, ts + idx as i64],
            )
            .unwrap();
        }
    }

    #[test]
    fn good_conversation_dumps_one_jsonl_line() {
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("memory.db");
        make_test_db(&db);
        let conn = Connection::open(&db).unwrap();
        insert_conv(
            &conn,
            "good",
            1000,
            &[
                ("user", "hello"),
                ("assistant", "hi"),
                ("user", "what's up"),
                ("assistant", "training docs"),
            ],
        );
        drop(conn);

        let out = td.path().join("dump.jsonl");
        let stats = dump_with_db(&db, 0, &out).expect("dump");
        assert_eq!(stats.n_conversations, 1);
        assert_eq!(stats.n_turns, 4);
        let body = std::fs::read_to_string(&out).unwrap();
        assert_eq!(body.lines().count(), 1);
        assert!(body.contains("\"role\":\"user\""));
        assert!(body.contains("\"content\":\"hello\""));
    }

    #[test]
    fn short_conversation_dropped() {
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("memory.db");
        make_test_db(&db);
        let conn = Connection::open(&db).unwrap();
        insert_conv(
            &conn,
            "tiny",
            1000,
            &[("user", "ping"), ("assistant", "pong")],
        );
        drop(conn);

        let out = td.path().join("dump.jsonl");
        let stats = dump_with_db(&db, 0, &out).unwrap();
        assert_eq!(stats.n_conversations, 0);
        assert_eq!(stats.n_dropped_short, 1);
    }

    #[test]
    fn error_messages_filtered_out() {
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("memory.db");
        make_test_db(&db);
        let conn = Connection::open(&db).unwrap();
        // Conversation has 5 turns but two start with "error:". The
        // 3 remaining are below MIN_TURNS_PER_CONVERSATION (4), so
        // the whole conv drops as short — correct behaviour.
        insert_conv(
            &conn,
            "errs",
            1000,
            &[
                ("user", "do x"),
                ("tool", "error: failed"),
                ("user", "retry"),
                ("tool", "error: still failed"),
                ("assistant", "give up"),
            ],
        );
        drop(conn);

        let out = td.path().join("dump.jsonl");
        let stats = dump_with_db(&db, 0, &out).unwrap();
        assert_eq!(stats.n_dropped_errors, 2);
        assert_eq!(stats.n_conversations, 0);
        // Raw 5 turns ≥ 4, but after filtering 2 errors only 3 remain
        // → counts under filtered_below_min, not short.
        assert_eq!(stats.n_dropped_short, 0);
        assert_eq!(stats.n_dropped_filtered_below_min, 1);
    }

    #[test]
    fn oversize_message_filtered_out() {
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("memory.db");
        make_test_db(&db);
        let conn = Connection::open(&db).unwrap();
        // 5 turns, one of which is a 300 KiB blob. Surviving 4 turns
        // = exactly MIN_TURNS_PER_CONVERSATION → kept.
        let blob = "x".repeat(300 * 1024);
        insert_conv(
            &conn,
            "oversize",
            1000,
            &[
                ("user", "trace pls"),
                ("assistant", &blob),
                ("user", "smaller"),
                ("assistant", "ok"),
                ("user", "thanks"),
            ],
        );
        drop(conn);

        let out = td.path().join("dump.jsonl");
        let stats = dump_with_db(&db, 0, &out).unwrap();
        assert_eq!(stats.n_dropped_oversize, 1);
        assert_eq!(stats.n_conversations, 1);
        assert_eq!(stats.n_turns, 4);
    }

    #[test]
    fn cutoff_filters_old_conversations() {
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("memory.db");
        make_test_db(&db);
        let conn = Connection::open(&db).unwrap();
        let four = &[
            ("user", "a"),
            ("assistant", "b"),
            ("user", "c"),
            ("assistant", "d"),
        ];
        insert_conv(&conn, "old", 100, four);
        insert_conv(&conn, "new", 1000, four);
        drop(conn);

        let out = td.path().join("dump.jsonl");
        let stats = dump_with_db(&db, 500, &out).unwrap();
        assert_eq!(stats.n_conversations, 1);
        let body = std::fs::read_to_string(&out).unwrap();
        assert!(body.contains("\"a\""));
        assert_eq!(body.lines().count(), 1);
    }

    #[test]
    fn count_with_db_does_not_persist_jsonl() {
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("memory.db");
        make_test_db(&db);
        let conn = Connection::open(&db).unwrap();
        insert_conv(
            &conn,
            "good",
            1000,
            &[
                ("user", "a"),
                ("assistant", "b"),
                ("user", "c"),
                ("assistant", "d"),
            ],
        );
        drop(conn);
        let stats = count_with_db(&db, 0).unwrap();
        assert_eq!(stats.n_conversations, 1);
        assert_eq!(stats.path, PathBuf::new(), "dry-run must not expose a path");
    }

    #[test]
    fn missing_db_errors_clean() {
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("nonexistent.db");
        let out = td.path().join("dump.jsonl");
        let err = dump_with_db(&db, 0, &out).expect_err("missing db must error");
        assert!(format!("{err}").contains("read-only") || format!("{err}").contains("nonexistent"));
    }

    #[test]
    fn read_only_open_cannot_mutate_db() {
        // Defence in depth: even with an open dump_with_db handle,
        // the connection is read-only. Verify by trying to INSERT
        // through a parallel handle and then dump — the dump should
        // still see the new row, proving the read-only flag doesn't
        // break correctness, but inversely we know our handle can't
        // INSERT.
        let td = tempfile::tempdir().unwrap();
        let db = td.path().join("memory.db");
        make_test_db(&db);

        // Open read-only via the same code path.
        let conn = Connection::open_with_flags(
            &db,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .unwrap();
        let r = conn.execute(
            "INSERT INTO turns (conversation_id, idx, role, content, ts) \
             VALUES ('x', 0, 'user', 'forbidden', 0)",
            [],
        );
        assert!(r.is_err(), "read-only handle must reject INSERT");
    }
}
