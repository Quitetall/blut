// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Sweep-completion index (E4).
//!
//! A sweep combo is identified by its composed-config `fingerprint`
//! (`ResolvedConfig.fingerprint`). When a sweep job finishes successfully
//! the runner appends one line to a global JSONL index; before launching a
//! combo the sweep engine consults the index and SKIPS a combo whose
//! fingerprint is present AND whose recorded output sidecar still exists on
//! disk with a matching content hash (so a pruned / GC'd output correctly
//! re-runs instead of being silently treated as done).
//!
//! Append-only JSONL (one object per line) is crash-safe under the single-box,
//! scheduler-lock-serialized run model: a torn final line is skipped on read,
//! and last-writer-wins resolves a re-run of the same combo.
//!
//! WIRING NOTE: the read side (`is_complete` → `sweep::cache_skip`) and the
//! record API (`record_completion`) are live + tested here, but appending on
//! Done belongs to a sweep RUNNER that does not exist yet (the sweep engine is
//! not hooked into `framework::executor` / `jobs.rs` — see `sweep.rs`). The
//! runner calls `record_completion` when it lands; until then the index is
//! simply always empty and every combo runs, exactly as before.

use std::path::{Path, PathBuf};

use crate::error::{Result, TrainError};
use crate::framework::artifact::{ArtifactMetadata, ContentHash};
use crate::framework::cache::CacheHandle;

/// One completed sweep combo.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SweepRecord {
    /// `ResolvedConfig.fingerprint` of the combo (lowercase hex).
    pub fingerprint: String,
    /// Job that produced it (for lineage / `blut runs diff`).
    pub job_id: String,
    /// Content hash of the combo's final output artifact (lowercase hex).
    pub final_output_hash: String,
    /// Sidecar (`output.metadata.json`) of that final output. Its existence
    /// + matching `content_hash` is the "still on disk" proof.
    pub sidecar_path: PathBuf,
    /// Unix-seconds completion stamp.
    pub completed_at: i64,
}

/// Default index location: `<global-cache>/sweep-index.jsonl`
/// (`$LAMU_TRAIN_CACHE_DIR` overrides the root, same as the train cache).
pub fn default_index_path() -> Option<PathBuf> {
    CacheHandle::default_global_path().map(|d| d.join("sweep-index.jsonl"))
}

/// Append one completion record. Atomic per line (a single `\n`-terminated
/// `write_all`); concurrent appends are serialized by the scheduler lock.
pub fn record_to(index_path: &Path, rec: &SweepRecord) -> Result<()> {
    if let Some(parent) = index_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| TrainError::other(format!("mkdir sweep-index dir: {e}")))?;
        }
    }
    let mut line = serde_json::to_string(rec)
        .map_err(|e| TrainError::other(format!("encode sweep record: {e}")))?;
    line.push('\n');
    // No fsync: a record lost to a crash just causes a re-run (the skip is an
    // optimization, never a correctness gate), and the scheduler lock prevents
    // interleaved writers — so the durability cost isn't worth paying here.
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(index_path)
        .map_err(|e| {
            TrainError::other(format!("open sweep-index {}: {e}", index_path.display()))
        })?;
    f.write_all(line.as_bytes())
        .map_err(|e| TrainError::other(format!("append sweep-index: {e}")))
}

/// Append to the default index, stamping `completed_at` from the wall clock.
pub fn record_completion(
    fingerprint: ContentHash,
    job_id: &str,
    final_output_hash: ContentHash,
    sidecar_path: PathBuf,
) -> Result<()> {
    let path = default_index_path()
        .ok_or_else(|| TrainError::other("cannot resolve global cache for sweep-index"))?;
    record_to(
        &path,
        &SweepRecord {
            fingerprint: fingerprint.to_hex(),
            job_id: job_id.to_string(),
            final_output_hash: final_output_hash.to_hex(),
            sidecar_path,
            completed_at: chrono::Utc::now().timestamp(),
        },
    )
}

/// Read all records, last-writer-wins per fingerprint. Tolerant: an empty,
/// torn, or malformed line is skipped (append-only crash safety). A missing
/// file yields an empty map (no completions recorded yet). Load this ONCE and
/// reuse it across a whole sweep — every combo's check is then a map lookup,
/// not a re-parse of the index.
pub fn load_index(index_path: &Path) -> std::collections::HashMap<String, SweepRecord> {
    let mut latest = std::collections::HashMap::new();
    let Ok(body) = std::fs::read_to_string(index_path) else {
        return latest;
    };
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<SweepRecord>(line) {
            latest.insert(rec.fingerprint.clone(), rec); // last write wins
        }
    }
    latest
}

/// Does a record's output still exist on disk with the content hash we
/// recorded? A pruned output or a path reused by a different run both read as
/// `false` (re-run).
pub fn is_record_live(rec: &SweepRecord) -> bool {
    match ArtifactMetadata::read_from(&rec.sidecar_path) {
        Ok(meta) => meta.content_hash.to_hex() == rec.final_output_hash,
        Err(_) => false,
    }
}

/// Is this combo already complete in `index_path`? True only when the
/// fingerprint is recorded AND `is_record_live`.
pub fn is_complete_in(index_path: &Path, fingerprint: ContentHash) -> bool {
    let fp = fingerprint.to_hex();
    load_index(index_path).get(&fp).is_some_and(is_record_live)
}

/// `is_complete_in` against the default global index. Resolves to `false`
/// (re-run) when the cache root can't be located.
pub fn is_complete(fingerprint: ContentHash) -> bool {
    default_index_path().is_some_and(|p| is_complete_in(&p, fingerprint))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_sidecar(dir: &Path, hash: ContentHash) -> PathBuf {
        let p = dir.join("output.metadata.json");
        ArtifactMetadata::new("ckpt".to_string(), 1, hash)
            .write_to(&p)
            .unwrap();
        p
    }

    #[test]
    fn absent_fingerprint_is_not_complete() {
        let td = tempfile::tempdir().unwrap();
        let idx = td.path().join("sweep-index.jsonl");
        assert!(!is_complete_in(&idx, ContentHash([1u8; 32])));
    }

    #[test]
    fn recorded_with_live_sidecar_is_complete() {
        let td = tempfile::tempdir().unwrap();
        let idx = td.path().join("sweep-index.jsonl");
        let fp = ContentHash([2u8; 32]);
        let out = ContentHash([3u8; 32]);
        let sidecar = write_sidecar(td.path(), out);
        record_to(
            &idx,
            &SweepRecord {
                fingerprint: fp.to_hex(),
                job_id: "job-a".into(),
                final_output_hash: out.to_hex(),
                sidecar_path: sidecar,
                completed_at: 1,
            },
        )
        .unwrap();
        assert!(is_complete_in(&idx, fp));
    }

    #[test]
    fn recorded_but_sidecar_pruned_re_runs() {
        let td = tempfile::tempdir().unwrap();
        let idx = td.path().join("sweep-index.jsonl");
        let fp = ContentHash([4u8; 32]);
        let out = ContentHash([5u8; 32]);
        let sidecar = write_sidecar(td.path(), out);
        record_to(
            &idx,
            &SweepRecord {
                fingerprint: fp.to_hex(),
                job_id: "job-b".into(),
                final_output_hash: out.to_hex(),
                sidecar_path: sidecar.clone(),
                completed_at: 1,
            },
        )
        .unwrap();
        assert!(is_complete_in(&idx, fp));
        // GC the output: skip must flip back to re-run.
        std::fs::remove_file(&sidecar).unwrap();
        assert!(!is_complete_in(&idx, fp));
    }

    #[test]
    fn content_hash_mismatch_re_runs() {
        // Same sidecar path reused by a DIFFERENT run (different content).
        let td = tempfile::tempdir().unwrap();
        let idx = td.path().join("sweep-index.jsonl");
        let fp = ContentHash([6u8; 32]);
        let sidecar = write_sidecar(td.path(), ContentHash([0xAA; 32]));
        record_to(
            &idx,
            &SweepRecord {
                fingerprint: fp.to_hex(),
                job_id: "job-c".into(),
                final_output_hash: ContentHash([0xBB; 32]).to_hex(), // not what's on disk
                sidecar_path: sidecar,
                completed_at: 1,
            },
        )
        .unwrap();
        assert!(!is_complete_in(&idx, fp));
    }

    #[test]
    fn last_writer_wins_and_torn_line_skipped() {
        let td = tempfile::tempdir().unwrap();
        let idx = td.path().join("sweep-index.jsonl");
        let fp = ContentHash([7u8; 32]);
        let out = ContentHash([8u8; 32]);
        let sidecar = write_sidecar(td.path(), out);
        // First record points at a now-pruned sidecar; second (later) record
        // points at the live one. Last-writer-wins must pick the live one.
        record_to(
            &idx,
            &SweepRecord {
                fingerprint: fp.to_hex(),
                job_id: "old".into(),
                final_output_hash: out.to_hex(),
                sidecar_path: td.path().join("gone.metadata.json"),
                completed_at: 1,
            },
        )
        .unwrap();
        // Inject a torn/garbage line between valid records.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(&idx).unwrap();
            f.write_all(b"{not valid json\n").unwrap();
        }
        record_to(
            &idx,
            &SweepRecord {
                fingerprint: fp.to_hex(),
                job_id: "new".into(),
                final_output_hash: out.to_hex(),
                sidecar_path: sidecar,
                completed_at: 2,
            },
        )
        .unwrap();
        assert!(
            is_complete_in(&idx, fp),
            "live last-writer record must win over torn + stale lines"
        );
    }
}
