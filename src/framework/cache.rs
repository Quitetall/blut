// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
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

/// Per-job + (commit-5) global + (Tier-4) remote cache handle.
#[derive(Clone, Debug)]
pub struct CacheHandle {
    pub job_local: PathBuf,
    pub global: Option<PathBuf>,
    /// Optional content-addressed REMOTE tier (ADR 0067 T4.2): checked LAST on
    /// lookup (after the local dirs), and — on a remote hit — written through
    /// to `job_local` so the entry is a real local `CacheHit`. `insert` writes
    /// through to it too (best-effort). A shared cache across machines / pods.
    pub remote: Option<std::sync::Arc<dyn crate::framework::object_store::BlobStore>>,
}

impl CacheHandle {
    /// Construct a per-job cache handle.
    pub fn job_local(path: PathBuf) -> Self {
        Self {
            job_local: path,
            global: None,
            remote: None,
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

    /// Namespace the GLOBAL cache tier by tenant (ADR 0096). The tenant is a path
    /// PREFIX on the store — the ADR-0078 key algorithm is unchanged, so two
    /// tenants get DISJOINT global roots (neither reads the other's entries) while
    /// a graph's fingerprint stays byte-identical across tenants. The `default`
    /// tenant is the flat store, so this is a NO-OP then (single-tenant and every
    /// existing cache path are byte-identical). `job_local` is per-job and already
    /// isolated, so it is left untouched.
    pub fn with_tenant(mut self, tenant: &crate::tenant::Tenant) -> Self {
        if !tenant.is_default() {
            self.global = self.global.map(|g| g.join(tenant.as_path()));
        }
        self
    }

    /// Attach a content-addressed remote tier (a shared object store / RWX
    /// PVC). Checked after the local dirs on lookup; written through on insert.
    pub fn with_remote(
        self,
        remote: std::sync::Arc<dyn crate::framework::object_store::BlobStore>,
    ) -> Self {
        Self {
            remote: Some(remote),
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
        code_sha: &[u8],
    ) -> ContentHash {
        Self::key_for_partitioned(stage_name, stage_schema, input_hash, args, code_sha, None)
    }

    /// Partition-aware cache identity (ADR 0101). `None` is byte-identical to
    /// [`Self::key_for`]; a concrete key is appended as the final, domain-
    /// separated input so cells cannot collide.
    pub fn key_for_partitioned(
        stage_name: &str,
        stage_schema: u32,
        input_hash: ContentHash,
        args: &serde_json::Value,
        code_sha: &[u8],
        partition: Option<&blut_types::partition::PartitionKey>,
    ) -> ContentHash {
        let canon = canonical_json(args);
        Self::key_for_canon_bytes_partitioned(
            stage_name,
            stage_schema,
            input_hash,
            canon.as_bytes(),
            code_sha,
            partition,
        )
    }

    /// Variant that accepts precomputed canonical-JSON bytes. The
    /// executor uses this on every stage invocation by caching the
    /// canonical bytes in the `PlanNode` at compile time — avoids
    /// re-walking the args `Value` tree on every cache lookup.
    pub fn key_for_canon_bytes(
        stage_name: &str,
        stage_schema: u32,
        input_hash: ContentHash,
        canon_args: &[u8],
        code_sha: &[u8],
    ) -> ContentHash {
        Self::key_for_canon_bytes_partitioned(
            stage_name,
            stage_schema,
            input_hash,
            canon_args,
            code_sha,
            None,
        )
    }

    pub(crate) fn key_for_canon_bytes_partitioned(
        stage_name: &str,
        stage_schema: u32,
        input_hash: ContentHash,
        canon_args: &[u8],
        code_sha: &[u8],
        partition: Option<&blut_types::partition::PartitionKey>,
    ) -> ContentHash {
        // v1→v2 (S4): `code_sha` (build git hash + the stage's script content
        // hash) now keys the cache, so editing a kernel with identical args
        // re-runs instead of reusing the stale checkpoint (closes G9). This is a
        // ONE-TIME global cache-bust — every pre-v2 entry re-keys; BLUT is
        // pre-1.0 so we carry no migration (the .json→.bin bust set the
        // precedent). Length-prefix code_sha so it can't ambiguate with the
        // trailing canon_args.
        const VERSION_TAG: &[u8] = b"blut.cache.v2";
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(VERSION_TAG);
        hasher.update([0u8]);
        hasher.update(stage_name.as_bytes());
        hasher.update([0u8]);
        hasher.update(stage_schema.to_le_bytes());
        hasher.update(input_hash.0);
        hasher.update((code_sha.len() as u64).to_le_bytes());
        hasher.update(code_sha);
        hasher.update(canon_args);
        if let Some(partition) = partition {
            let value = serde_json::to_value(partition).expect("PartitionKey serializes");
            let canonical = canonical_json(&value);
            hasher.update([0u8]);
            hasher.update(b"partition");
            hasher.update((canonical.len() as u64).to_le_bytes());
            hasher.update(canonical.as_bytes());
        }
        let arr: [u8; 32] = hasher.finalize().into();
        ContentHash(arr)
    }

    /// Expose `canonical_json` for the executor / plan compiler so
    /// the canonical bytes can be precomputed once per stage at
    /// plan-compile time.
    pub fn canonical_json_bytes(args: &serde_json::Value) -> Vec<u8> {
        canonical_json(args).into_bytes()
    }

    /// Look up a cached output. Returns the parsed
    /// `ErasedArtifact` if present, `None` if absent.
    ///
    /// Cache entries are bincode-encoded (opt-4). The original opt-2
    /// bincode attempt failed because `ErasedArtifact.payload` was
    /// `serde_json::Value` and bincode rejects `deserialize_any`;
    /// after refactoring payload to `Vec<u8>` (bincode bytes of the
    /// typed inner artifact), the wrapper itself is now safely
    /// bincode-able too. Result: smaller on-disk size + ~2-3×
    /// faster parse on cache hits.
    ///
    /// File extension is `.bin` (was `.json`) — old caches need a
    /// one-shot purge; BLUT is pre-v1 so we don't carry a migration.
    ///
    /// I/O errors other than NotFound are downgraded to None with
    /// a `tracing::warn` — a corrupt cache entry shouldn't break
    /// the run, just trigger a re-execution.
    pub fn lookup(&self, key: ContentHash) -> Option<CacheHit> {
        for base in self.search_order() {
            let path = base.join(key.to_hex()).join("output.bin");
            match std::fs::read(&path) {
                Ok(body) => match bincode::deserialize::<ErasedArtifact>(&body) {
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
                    tracing::warn!("cache: read {}: {e}; treating as miss", path.display());
                }
            }
        }
        // Remote tier (T4.2): local dirs missed — try the shared object store.
        // On a hit, write the bytes through to `job_local` so this becomes a
        // real local CacheHit (with a `from_path` a stage can read), and later
        // lookups in this job skip the network. A remote error degrades to a
        // miss (never a wrong answer).
        if let Some(remote) = &self.remote {
            match remote.get(key) {
                Ok(Some(body)) => match bincode::deserialize::<ErasedArtifact>(&body) {
                    Ok(art) => {
                        // Write through so `from_path` names a file that
                        // EXISTS. If that write fails, fall through to a miss
                        // rather than return a hit whose `from_path` points at
                        // nothing — every returned CacheHit has a readable
                        // path, and the entry is still on the remote for a
                        // later attempt.
                        let dest = self.job_local.join(key.to_hex()).join("output.bin");
                        match write_atomic(&dest, &body) {
                            Ok(()) => {
                                return Some(CacheHit {
                                    artifact: art,
                                    from_path: dest,
                                });
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "cache: remote hit but local write-through failed at {}: \
                                     {e}; treating as miss",
                                    dest.display()
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "cache: corrupt remote entry for {}: {e}; miss",
                            key.to_hex()
                        );
                    }
                },
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(
                        "cache: remote lookup for {}: {e}; treating as miss",
                        key.to_hex()
                    );
                }
            }
        }
        None
    }

    /// Insert an output for the given key. Atomic: writes to a
    /// sibling `.tmp.<pid>.<nanos>` and renames into place. Encoded
    /// as bincode — see `lookup` for rationale.
    pub fn insert(&self, key: ContentHash, output: &ErasedArtifact) -> std::io::Result<()> {
        let dir = self.write_target().join(key.to_hex());
        std::fs::create_dir_all(&dir)?;
        let dest = dir.join("output.bin");
        let body = bincode::serialize(output).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("serialize cache entry: {e}"),
            )
        })?;
        write_atomic(&dest, &body)?;
        // Write through to the remote tier (T4.2) so other machines/pods share
        // this result. Best-effort: a remote failure is logged, not fatal — the
        // local write already succeeded, so the run is unaffected.
        if let Some(remote) = &self.remote
            && let Err(e) = remote.put(key, &body)
        {
            tracing::warn!("cache: remote write-through for {}: {e}", key.to_hex());
        }
        Ok(())
    }

    /// Exact local entry path an insert writes for this handle/key.
    pub(crate) fn entry_path_for_write(&self, key: ContentHash) -> PathBuf {
        self.write_target().join(key.to_hex()).join("output.bin")
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

/// Durable proof tying a completed stage to the cache entry that made it
/// skippable. Written inside the current job's stage dir on both hits and
/// misses so lineage remains complete for all-cache-hit jobs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheProof {
    pub key: ContentHash,
    pub entry_path: PathBuf,
}

impl CacheProof {
    pub fn write_to(&self, path: &Path) -> std::io::Result<()> {
        let body = serde_json::to_vec(self).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("serialize cache proof: {e}"),
            )
        })?;
        write_atomic(path, &body)
    }

    pub fn read_from(path: &Path) -> std::io::Result<Self> {
        let body = std::fs::read(path)?;
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("parse cache proof: {e}"),
            )
        })
    }

    pub fn is_live(&self) -> bool {
        let key_hex = self.key.to_hex();
        if self.entry_path.file_name().and_then(|name| name.to_str()) != Some("output.bin")
            || self
                .entry_path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                != Some(key_hex.as_str())
        {
            return false;
        }
        std::fs::read(&self.entry_path)
            .ok()
            .and_then(|body| bincode::deserialize::<ErasedArtifact>(&body).ok())
            .is_some()
    }
}

/// Produce a canonical JSON form: object keys sorted
/// lexicographically, recursively. Used as part of the cache key
/// so two args dicts with the same fields in different orders
/// hash identically.
///
/// Performance: streams directly into the output String, no
/// intermediate `Value` tree. Sorts object keys via `Vec` + `sort`
/// rather than `BTreeMap` to avoid allocating a separate map per
/// object. For a typical ~500-byte recipe-args dict, ~3-4× faster
/// than the previous "build canonical Value, then to_string" path
/// because we skip the intermediate clones + Map round-trip.
fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::with_capacity(256);
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            // Reuse serde_json's number formatter — handles ints,
            // floats, scientific notation correctly.
            out.push_str(&n.to_string());
        }
        Value::String(s) => {
            // serde_json::to_string on a Value::String emits a
            // properly-escaped JSON literal (quotes + escapes).
            // Cheaper than reimplementing the escape state machine
            // here; the allocation is amortized across the whole
            // canonical buffer.
            if let Ok(rendered) = serde_json::to_string(s) {
                out.push_str(&rendered);
            } else {
                // Unreachable: serializing a &str cannot fail.
                out.push_str("\"\"");
            }
        }
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Borrow keys; sort references; no per-entry clone.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if let Ok(rendered) = serde_json::to_string(k.as_str()) {
                    out.push_str(&rendered);
                }
                out.push(':');
                write_canonical(&map[*k], out);
            }
            out.push('}');
        }
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

pub(crate) fn write_atomic(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
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
    let tmp = dest.with_file_name(format!(".{stem}.tmp.{}.{nanos}", std::process::id()));
    // Write+sync+rename in one fallible step; clean up the tmp on ANY
    // failure (mirrors `broker/footprint.rs::save`) — a sync error must
    // not leave an orphaned tmp file behind, same as a rename error. A
    // dropped `sync_all` error would let `insert()` report `Ok(())` even
    // though the bytes may not be durable: a crash before background
    // writeback flushes the page leaves a truncated/garbage file at
    // `dest` after the rename, which `lookup()` would only catch later
    // via the corrupt-entry downgrade-to-miss path.
    let result = (|| -> std::io::Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, dest)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_erased(payload: serde_json::Value) -> ErasedArtifact {
        // Payload bytes are bincode of the JSON STRING form of the
        // value. `serde_json::Value` itself requires `deserialize_any`
        // which bincode rejects; encoding the string side-steps it
        // and keeps test fixtures ergonomic with `json!(...)`.
        let s = payload.to_string();
        ErasedArtifact {
            kind: "test.kind".into(),
            schema: 1,
            payload: bincode::serialize(&s).unwrap(),
        }
    }

    fn decode_payload(art: &ErasedArtifact) -> serde_json::Value {
        let s: String = bincode::deserialize(&art.payload).unwrap();
        serde_json::from_str(&s).unwrap()
    }

    const CS: &[u8] = b"code-sha-fixture";

    #[test]
    fn key_changes_on_stage_name_change() {
        let h = ContentHash::of_bytes(b"x");
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("alpha", 1, h, &a, CS);
        let k2 = CacheHandle::key_for("beta", 1, h, &a, CS);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_schema_bump() {
        let h = ContentHash::of_bytes(b"x");
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("s", 1, h, &a, CS);
        let k2 = CacheHandle::key_for("s", 2, h, &a, CS);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_input_hash_change() {
        let a = serde_json::json!({});
        let k1 = CacheHandle::key_for("s", 1, ContentHash::of_bytes(b"a"), &a, CS);
        let k2 = CacheHandle::key_for("s", 1, ContentHash::of_bytes(b"b"), &a, CS);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_changes_on_code_sha_change() {
        // S4: same name/schema/input/args, DIFFERENT code → different key. This
        // is the data-loss gap (G9): editing a kernel must re-run, not reuse the
        // stale checkpoint.
        let h = ContentHash::of_bytes(b"x");
        let a = serde_json::json!({"lr": 0.1});
        let k1 = CacheHandle::key_for("train", 1, h, &a, b"code-v1");
        let k2 = CacheHandle::key_for("train", 1, h, &a, b"code-v2");
        assert_ne!(k1, k2, "a code edit must change the cache key");
    }

    #[test]
    fn key_invariant_under_args_field_order() {
        // Same fields, different order → same cache key. Critical
        // property: users shouldn't have to keep arg structs in
        // a specific order to hit the cache.
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"alpha": 1, "beta": 2});
        let a2 = serde_json::json!({"beta": 2, "alpha": 1});
        let k1 = CacheHandle::key_for("s", 1, h, &a1, CS);
        let k2 = CacheHandle::key_for("s", 1, h, &a2, CS);
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_changes_on_args_value_change() {
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"alpha": 1});
        let a2 = serde_json::json!({"alpha": 2});
        let k1 = CacheHandle::key_for("s", 1, h, &a1, CS);
        let k2 = CacheHandle::key_for("s", 1, h, &a2, CS);
        assert_ne!(k1, k2);
    }

    #[test]
    fn key_handles_nested_object_canonical_order() {
        let h = ContentHash::of_bytes(b"x");
        let a1 = serde_json::json!({"outer": {"a": 1, "b": 2}});
        let a2 = serde_json::json!({"outer": {"b": 2, "a": 1}});
        let k1 = CacheHandle::key_for("s", 1, h, &a1, CS);
        let k2 = CacheHandle::key_for("s", 1, h, &a2, CS);
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
        assert_eq!(decode_payload(&hit.artifact), serde_json::json!({"n": 7}));
    }

    #[test]
    fn lookup_returns_none_on_corrupt_entry() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        let dir = td.path().join(key.to_hex());
        std::fs::create_dir_all(&dir).unwrap();
        // Truncated bincode header → deserialize fails.
        std::fs::write(dir.join("output.bin"), [0xFFu8; 3]).unwrap();
        assert!(h.lookup(key).is_none());
    }

    /// §5.1 "Cache CORRUPT `.bin`": a `output.bin` that is a *truncated*
    /// copy of a once-valid bincode `ErasedArtifact` must DOWNGRADE to a
    /// miss (return `None`, triggering a re-run) — never panic and never
    /// hand back a garbage / partially-decoded artifact.
    #[test]
    fn lookup_downgrades_truncated_valid_entry_to_miss() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"truncated");
        let dir = td.path().join(key.to_hex());
        std::fs::create_dir_all(&dir).unwrap();

        // Serialize a real, well-formed cache entry first…
        let good = bincode::serialize(&fake_erased(serde_json::json!({"n": 42}))).unwrap();
        assert!(good.len() > 4, "fixture must be long enough to truncate");
        // …then write only its first few bytes (the length prefix +
        // partial payload) so the on-disk record is a torn write.
        std::fs::write(dir.join("output.bin"), &good[..good.len() / 2]).unwrap();

        // No panic, and the lookup reports a clean miss.
        assert!(
            h.lookup(key).is_none(),
            "truncated bincode must downgrade to a cache miss"
        );
    }

    /// §5.1 "Cache CORRUPT `.bin`": pure garbage (not even a valid
    /// bincode prefix) must also downgrade to a miss without panicking.
    #[test]
    fn lookup_downgrades_garbage_entry_to_miss() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"garbage");
        let dir = td.path().join(key.to_hex());
        std::fs::create_dir_all(&dir).unwrap();
        // A bincode length prefix claiming a huge string, followed by no
        // data — the classic "allocator bomb" corrupt-frame shape. The
        // deserializer must error (not OOM / panic), and lookup returns
        // None.
        let mut garbage = Vec::new();
        garbage.extend_from_slice(&u64::MAX.to_le_bytes()); // bogus length
        garbage.extend_from_slice(b"\x00not-a-valid-record\xff\xfe");
        std::fs::write(dir.join("output.bin"), &garbage).unwrap();
        assert!(
            h.lookup(key).is_none(),
            "garbage bytes must downgrade to a cache miss, not panic"
        );

        // Empty file is also corrupt-shaped (truncated to zero) → miss.
        std::fs::write(dir.join("output.bin"), b"").unwrap();
        assert!(
            h.lookup(key).is_none(),
            "empty output.bin must downgrade to a cache miss"
        );
    }

    /// A *valid* entry written right after a corrupt one was evicted /
    /// overwritten still hits — i.e. the downgrade-to-miss path doesn't
    /// poison the key. Guards against a regression where a corrupt read
    /// might cache a negative result.
    #[test]
    fn corrupt_then_valid_entry_hits() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"recover");
        let dir = td.path().join(key.to_hex());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("output.bin"), [0x01u8, 0x02, 0x03]).unwrap();
        assert!(h.lookup(key).is_none(), "corrupt first read → miss");
        // Overwrite with a valid record (insert uses atomic rename).
        h.insert(key, &fake_erased(serde_json::json!({"ok": true})))
            .unwrap();
        let hit = h.lookup(key).expect("valid entry must now hit");
        assert_eq!(
            decode_payload(&hit.artifact),
            serde_json::json!({"ok": true})
        );
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
        h.insert(key, &fake_erased(serde_json::json!({"x": 1})))
            .unwrap();
        // Entry must exist under the global path.
        assert!(global.join(key.to_hex()).join("output.bin").exists());
    }

    #[test]
    fn remote_tier_write_through_and_hit() {
        use crate::framework::object_store::{BlobStore, FsBlobStore};
        let td = tempfile::tempdir().unwrap();
        let remote = std::sync::Arc::new(FsBlobStore::new(td.path().join("remote")));
        let key = ContentHash::of_bytes(b"k");

        // Machine A: insert → writes local AND through to the remote store.
        let a_job = td.path().join("a");
        let h_a = CacheHandle::job_local(a_job).with_remote(remote.clone());
        h_a.insert(key, &fake_erased(serde_json::json!({ "v": 1 })))
            .unwrap();
        assert!(remote.head(key).unwrap(), "insert wrote through to remote");

        // Machine B: cold local, same remote → lookup hits the remote and
        // writes it through to B's job dir (a real CacheHit with a path).
        let b_job = td.path().join("b");
        let h_b = CacheHandle::job_local(b_job.clone()).with_remote(remote.clone());
        let hit = h_b.lookup(key).expect("remote tier serves the entry");
        assert_eq!(hit.artifact.kind, fake_erased(serde_json::json!({})).kind);
        assert!(
            b_job.join(key.to_hex()).join("output.bin").exists(),
            "remote hit was written through to the local job dir"
        );
    }

    #[test]
    fn remote_error_degrades_to_a_miss() {
        // A remote whose root can't be read → lookup is a miss, not a panic.
        use crate::framework::object_store::FsBlobStore;
        let td = tempfile::tempdir().unwrap();
        // FsBlobStore over a missing dir returns None (a miss), never errors on
        // get; the handle must simply report no hit.
        let remote = std::sync::Arc::new(FsBlobStore::new(PathBuf::from("/no-such-remote-xyz")));
        let h = CacheHandle::job_local(td.path().join("job")).with_remote(remote);
        assert!(h.lookup(ContentHash::of_bytes(b"absent")).is_none());
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
            job.join(key.to_hex()).join("output.bin"),
            bincode::serialize(&fake_erased(serde_json::json!({"src": "job"}))).unwrap(),
        )
        .unwrap();
        std::fs::write(
            global.join(key.to_hex()).join("output.bin"),
            bincode::serialize(&fake_erased(serde_json::json!({"src": "global"}))).unwrap(),
        )
        .unwrap();
        let h = CacheHandle::job_local(job).with_global(global);
        let hit = h.lookup(key).expect("must hit");
        assert_eq!(
            decode_payload(&hit.artifact),
            serde_json::json!({"src": "global"})
        );
    }

    #[test]
    fn lru_prune_removes_oldest_until_under_cap() {
        let td = tempfile::tempdir().unwrap();
        // Three "cache entries" each 1 KiB. Cap at 2 KiB → one
        // must go.
        for name in ["e1", "e2", "e3"] {
            let dir = td.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("output.bin"), vec![0u8; 1024]).unwrap();
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
        std::fs::write(td.path().join("e1/output.bin"), vec![0u8; 100]).unwrap();
        let freed = lru_prune(td.path(), 1024).unwrap();
        assert_eq!(freed, 0);
    }

    #[test]
    fn lru_prune_handles_missing_root() {
        // Nonexistent directory → 0 freed, no error.
        let freed = lru_prune(Path::new("/tmp/lamu-nonexistent-xyz-9999"), 1024).unwrap();
        assert_eq!(freed, 0);
    }

    /// Happy path for `write_atomic` itself (not just via `insert`):
    /// bytes land at `dest`, and no sibling `.tmp.<pid>.<nanos>` file
    /// survives. Direct regression test for the write→sync→rename
    /// refactor that now propagates `sync_all()` errors (previously
    /// `let _ = f.sync_all();` silently dropped a failed fsync, so
    /// `insert()` could report `Ok(())` for bytes that were never made
    /// durable — see `broker/footprint.rs::save()` for the identical
    /// fix applied earlier to the footprint store).
    #[test]
    fn write_atomic_success_writes_bytes_and_leaves_no_tmp() {
        let td = tempfile::tempdir().unwrap();
        let dest = td.path().join("out.bin");
        write_atomic(&dest, b"hello").unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
        let tmp_remnants: Vec<_> = std::fs::read_dir(td.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            tmp_remnants.is_empty(),
            "no tmp file should remain after a successful write_atomic"
        );
    }

    /// A genuine fsync-failure injection (ENOSPC/EIO at fsync time) isn't
    /// portably reachable from a `#[test]` without OS-level tricks or new
    /// dependencies, and `broker/footprint.rs`'s own tests for the
    /// identical fix don't attempt it either — so this instead forces a
    /// *different* failure (`rename(tmp, dest)` onto an existing
    /// directory) that routes through the SAME cleanup branch
    /// (`if result.is_err() { remove_file(&tmp) }`) that a propagated
    /// `sync_all()` error now also takes. Confirms the refactor didn't
    /// regress tmp cleanup on error.
    #[test]
    fn write_atomic_cleans_up_tmp_on_failure() {
        let td = tempfile::tempdir().unwrap();
        let dest = td.path().join("out.bin");
        std::fs::create_dir_all(&dest).unwrap(); // dest occupied by a dir → rename fails
        let err = write_atomic(&dest, b"hello");
        assert!(err.is_err(), "rename onto an existing dir must fail");
        let tmp_remnants: Vec<_> = std::fs::read_dir(td.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            tmp_remnants.is_empty(),
            "tmp file must be cleaned up when write_atomic fails, not orphaned"
        );
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
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            entries.is_empty(),
            "no tmp files should survive successful insert"
        );
    }
}
