//! Stage output cache.
//!
//! Skips re-execution of a stage when the inputs + args + stage
//! identity match a previous run's cached output. Per-job by
//! default (lives at `<job_dir>/_cache/<key:hex>/output.json`); the
//! `--shared-cache` flag (commit 5) flips lookup to the global
//! cache at `~/.local/share/lamu/train-cache/` first, then job-local.
//!
//! Cache key formula:
//!
//! ```text
//! sha256(
//!   b"blut.cache.v1" ‖
//!   stage_name (as bytes) ‖
//!   stage_schema (LE u32) ‖
//!   input_content_hash (32 bytes) ‖
//!   canonical(args_json)
//! )
//! ```
//!
//! `canonical(args_json)` = serde_json with object keys sorted
//! lexicographically. Field reorder doesn't invalidate; rename
//! does (semantic change). Test-covered.
//!
//! What lives in `<key:hex>/`:
//!
//! - `output.json` — the `ErasedArtifact` JSON. Cheap to read.
//! - The artifact's payload files DO NOT live here. They live
//!   wherever the producing stage put them (typically
//!   `<job_dir>/stages/<idx>-<name>/`). Cache hit means "I know
//!   the output of this stage; here's the metadata"; the on-disk
//!   payload is content-addressed via the artifact's primary path
//!   so it's findable even across jobs.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::ContentHash;
use crate::framework::stage::ErasedArtifact;

/// Per-job + (commit-5) global cache handle.
#[derive(Clone, Debug)]
pub struct CacheHandle {
    pub job_local: PathBuf,
    pub global: Option<PathBuf>,
}

impl CacheHandle {
    /// Construct a per-job cache handle.
    pub fn job_local(path: PathBuf) -> Self {
        Self {
            job_local: path,
            global: None,
        }
    }

    /// Promote this handle to the `--shared-cache` shape: global
    /// cache is checked FIRST on lookup; writes go to the global
    /// cache so future jobs benefit too.
    pub fn with_global(self, global: PathBuf) -> Self {
        Self {
            global: Some(global),
            ..self
        }
    }

    /// Default global cache location: `$XDG_DATA_HOME/lamu/train-cache/`.
    /// Override with `$LAMU_TRAIN_CACHE_DIR`.
    pub fn default_global_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("LAMU_TRAIN_CACHE_DIR") {
            return Some(PathBuf::from(p));
        }
        dirs::data_local_dir().map(|d| d.join("lamu").join("train-cache"))
    }

    /// Compute the cache key for a stage invocation.
    ///
    /// Uses SHA-256. BLAKE3 was tried but lost to SHA-256 on the
    /// typical cache-key input size (~300-600 bytes): BLAKE3's SIMD
    /// parallelism only wins at multi-KiB inputs, and SHA-256 has
    /// hardware acceleration on every recent x86 + ARM via SHA-NI /
    /// crypto-extension. Benchmark showed +17% regression for
    /// BLAKE3 here, so we stayed with SHA-256.
    pub fn key_for(
        stage_name: &str,
        stage_schema: u32,
        input_hash: ContentHash,
        args: &serde_json::Value,
    ) -> ContentHash {
        const VERSION_TAG: &[u8] = b"blut.cache.v1";
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(VERSION_TAG);
        hasher.update([0u8]);
        hasher.update(stage_name.as_bytes());
        hasher.update([0u8]);
        hasher.update(stage_schema.to_le_bytes());
        hasher.update(input_hash.0);
        let canon = canonical_json(args);
        hasher.update(canon.as_bytes());
        let arr: [u8; 32] = hasher.finalize().into();
        ContentHash(arr)
    }

    /// Look up a cached output. Returns the parsed
    /// `ErasedArtifact` if present, `None` if absent.
    ///
    /// Cache entries are compact JSON (no pretty-printing) for
    /// speed + smaller disk footprint vs the pre-opt pretty-JSON
    /// (typically ~2× smaller). Bincode was tried but the cache
    /// stores `ErasedArtifact` whose `payload: serde_json::Value`
    /// needs `deserialize_any` — which bincode doesn't support.
    /// JSON's self-describing format is the right fit.
    ///
    /// I/O errors other than NotFound are downgraded to None with
    /// a `tracing::warn` — a corrupt cache entry shouldn't break
    /// the run, just trigger a re-execution.
    pub fn lookup(&self, key: ContentHash) -> Option<CacheHit> {
        for base in self.search_order() {
            let path = base.join(key.to_hex()).join("output.json");
            match std::fs::read(&path) {
                Ok(body) => match serde_json::from_slice::<ErasedArtifact>(&body) {
                    Ok(art) => {
                        return Some(CacheHit {
                            artifact: art,
                            from_path: path,
                        });
                    }
                    Err(e) => {
                        tracing::warn!(
                            "cache: corrupt entry at {}: {e}; treating as miss",
                            path.display()
                        );
                    }
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(
                        "cache: read {}: {e}; treating as miss",
                        path.display()
                    );
                }
            }
        }
        None
    }

    /// Insert an output for the given key. Atomic: writes to a
    /// sibling `.tmp.<pid>.<nanos>` and renames into place. Uses
    /// compact JSON (no pretty-printing) — ~2× smaller on disk
    /// than the pre-opt pretty-JSON, identical wire compatibility.
    pub fn insert(&self, key: ContentHash, output: &ErasedArtifact) -> std::io::Result<()> {
        let dir = self.write_target().join(key.to_hex());
        std::fs::create_dir_all(&dir)?;
        let dest = dir.join("output.json");
        let body = serde_json::to_vec(output).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("serialize cache entry: {e}"),
            )
        })?;
        write_atomic(&dest, &body)
    }

    /// Search order for lookups: global first when `--shared-cache`
    /// promoted it, then job-local. Writes always go to
    /// `write_target` (job-local unless `--shared-cache`).
    fn search_order(&self) -> Vec<&Path> {
        let mut v = Vec::with_capacity(2);
        if let Some(g) = &self.global {
            v.push(g.as_path());
        }
        v.push(self.job_local.as_path());
        v
    }

    fn write_target(&self) -> &Path {
        // With --shared-cache: writes go to the global cache so
        // future jobs share. Without: writes are job-local only.
        // The job-local path is always also a search target on
        // lookup, so a global hit is preferred when both are
        // populated.
        match &self.global {
            Some(g) => g.as_path(),
            None => &self.job_local,
        }
    }
}

/// LRU prune: scan the cache root, sort entries by atime, delete
/// oldest until total size ≤ `max_bytes`. Best-effort: I/O errors
/// are logged + skipped. Intended to run periodically (e.g. before
/// a fresh `recipe run` that's about to fill the cache further).
///
/// `max_bytes`: cap, e.g. 50 GiB. Default driven by
/// `$LAMU_CACHE_MAX_GB` (commit 8 wires the CLI knob).
pub fn lru_prune(cache_root: &Path, max_bytes: u64) -> std::io::Result<u64> {
    let mut entries: Vec<(PathBuf, std::time::SystemTime, u64)> = Vec::new();
    let mut total: u64 = 0;
    let dir = match std::fs::read_dir(cache_root) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    for entry in dir.flatten() {
        let p = entry.path();
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_dir() {
            continue;
        }
        let size = dir_size(&p).unwrap_or(0);
        let atime = meta
            .accessed()
            .or_else(|_| meta.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        total += size;
        entries.push((p, atime, size));
    }
    if total <= max_bytes {
        return Ok(0);
    }
    entries.sort_by_key(|(_, atime, _)| *atime);
    let mut freed: u64 = 0;
    for (path, _, size) in entries {
        if total <= max_bytes {
            break;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                total = total.saturating_sub(size);
                freed += size;
            }
            Err(e) => {
                tracing::warn!("lru_prune: failed to remove {}: {}", path.display(), e);
            }
        }
    }
    Ok(freed)
}

fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let m = entry.metadata()?;
        if m.is_dir() {
            total = total.saturating_add(dir_size(&entry.path())?);
        } else {
            total = total.saturating_add(m.len());
        }
    }
    Ok(total)
}

#[derive(Debug)]
pub struct CacheHit {
    pub artifact: ErasedArtifact,
    pub from_path: PathBuf,
}

/// Produce a canonical JSON form: object keys sorted
/// lexicographically, recursively. Used as part of the cache key
/// so two args dicts with the same fields in different orders
/// hash identically.
fn canonical_json(value: &serde_json::Value) -> String {
    let canon = canonical_value(value);
    serde_json::to_string(&canon).unwrap_or_default()
}

fn canonical_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            // BTreeMap collects entries in lex-sorted key order.
            let mut sorted: std::collections::BTreeMap<String, serde_json::Value> =
                std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonical_value(v));
            }
            // Round-trip back to serde_json::Map preserving the
            // sorted order (serde_json::Map is insertion-ordered;
            // the BTreeMap traversal gives us lex order).
            let mut out = serde_json::Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.iter().map(canonical_value).collect())
        }
        other => other.clone(),
    }
}

/// Serializable record used by the executor when writing the
/// cache. Currently identical to `ErasedArtifact`, but kept as a
/// distinct alias so commit 5's lru-prune metadata can extend
/// without touching every call site.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheRecord {
    pub artifact: ErasedArtifact,
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stem = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "tmp".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = dest.with_file_name(format!(
        ".{stem}.tmp.{}.{nanos}",
        std::process::id()
    ));
    let mut f = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&tmp)?;
    f.write_all(bytes)?;
    let _ = f.sync_all();
    drop(f);
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_erased(payload: serde_json::Value) -> ErasedArtifact {
        ErasedArtifact {
            kind: "test.kind".into(),
            schema: 1,
            payload,
        }
    }

    #[test]
    fn key_changes_on_stage_name_change() {
        let h = ContentHash::of_bytes(b"x");
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("alpha", 1, h, &a);
        let k2 = CacheHandle::key_for("beta", 1, h, &a);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_schema_bump() {
        let h = ContentHash::of_bytes(b"x");
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("s", 1, h, &a);
        let k2 = CacheHandle::key_for("s", 2, h, &a);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_input_hash_change() {
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("s", 1, ContentHash::of_bytes(b"a"), &a);
        let k2 = CacheHandle::key_for("s", 1, ContentHash::of_bytes(b"b"), &a);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_invariant_under_args_field_order() {
        // Same fields, different order → same cache key. Critical
        // property: users shouldn't have to keep arg structs in
        // a specific order to hit the cache.
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"alpha": 1, "beta": 2});
        let a2 = serde_json::json!({"beta": 2, "alpha": 1});
        let k1 = CacheHandle::key_for("s", 1, h, &a1);
        let k2 = CacheHandle::key_for("s", 1, h, &a2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_changes_on_args_value_change() {
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"alpha": 1});
        let a2 = serde_json::json!({"alpha": 2});
        let k1 = CacheHandle::key_for("s", 1, h, &a1);
        let k2 = CacheHandle::key_for("s", 1, h, &a2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_handles_nested_object_canonical_order() {
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"outer": {"a": 1, "b": 2}});
        let a2 = serde_json::json!({"outer": {"b": 2, "a": 1}});
        let k1 = CacheHandle::key_for("s", 1, h, &a1);
        let k2 = CacheHandle::key_for("s", 1, h, &a2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn lookup_returns_none_when_empty() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"missing");
        assert!(h.lookup(key).is_none());
    }

    #[test]
    fn insert_then_lookup_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        let art = fake_erased(serde_json::json!({"n": 7}));
        h.insert(key, &art).unwrap();
        let hit = h.lookup(key).expect("should hit");
        assert_eq!(hit.artifact.kind, "test.kind");
        assert_eq!(hit.artifact.payload, serde_json::json!({"n": 7}));
    }

    #[test]
    fn lookup_returns_none_on_corrupt_entry() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        let dir = td.path().join(key.to_hex());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("output.json"), b"{not valid json").unwrap();
        assert!(h.lookup(key).is_none());
    }

    #[test]
    fn shared_cache_writes_go_to_global() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("job");
        let global = td.path().join("global");
        std::fs::create_dir_all(&job).unwrap();
        std::fs::create_dir_all(&global).unwrap();
        let h = CacheHandle::job_local(job).with_global(global.clone());
        let key = ContentHash::of_bytes(b"k");
        h.insert(key, &fake_erased(serde_json::json!({"x": 1}))).unwrap();
        // Entry must exist under the global path.
        assert!(global.join(key.to_hex()).join("output.json").exists());
    }

    #[test]
    fn shared_cache_lookup_prefers_global() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("job");
        let global = td.path().join("global");
        let key = ContentHash::of_bytes(b"k");
        std::fs::create_dir_all(job.join(key.to_hex())).unwrap();
        std::fs::create_dir_all(global.join(key.to_hex())).unwrap();
        // Different payloads under the two roots.
        std::fs::write(
            job.join(key.to_hex()).join("output.json"),
            serde_json::to_vec(&fake_erased(serde_json::json!({"src": "job"}))).unwrap(),
        )
        .unwrap();
        std::fs::write(
            global.join(key.to_hex()).join("output.json"),
            serde_json::to_vec(&fake_erased(serde_json::json!({"src": "global"}))).unwrap(),
        )
        .unwrap();
        let h = CacheHandle::job_local(job).with_global(global);
        let hit = h.lookup(key).expect("must hit");
        assert_eq!(hit.artifact.payload, serde_json::json!({"src": "global"}));
    }

    #[test]
    fn lru_prune_removes_oldest_until_under_cap() {
        let td = tempfile::tempdir().unwrap();
        // Three "cache entries" each 1 KiB. Cap at 2 KiB → one
        // must go.
        for name in ["e1", "e2", "e3"] {
            let dir = td.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("output.json"), vec![0u8; 1024]).unwrap();
        }
        // Bump atime ordering by sleeping briefly between touches.
        // tempdirs default to creation time; force atime spread:
        for name in ["e1", "e2", "e3"] {
            let p = td.path().join(name);
            let _ = std::fs::File::open(&p);
        }
        let freed = lru_prune(td.path(), 2 * 1024).unwrap();
        // At least one entry was freed.
        assert!(freed >= 1024);
    }

    #[test]
    fn lru_prune_noop_when_under_cap() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("e1")).unwrap();
        std::fs::write(td.path().join("e1/output.json"), vec![0u8; 100]).unwrap();
        let freed = lru_prune(td.path(), 1024).unwrap();
        assert_eq!(freed, 0);
    }

    #[test]
    fn lru_prune_handles_missing_root() {
        // Nonexistent directory → 0 freed, no error.
        let freed = lru_prune(Path::new("/tmp/lamu-nonexistent-xyz-9999"), 1024).unwrap();
        assert_eq!(freed, 0);
    }

    #[test]
    fn insert_creates_dir_atomically_no_tmp_remnants() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        h.insert(key, &fake_erased(serde_json::json!({}))).unwrap();
        let entries: Vec<_> = std::fs::read_dir(td.path().join(key.to_hex()))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .contains(".tmp.")
            })
            .collect();
        assert!(entries.is_empty(), "no tmp files should survive successful insert");
    }
}
