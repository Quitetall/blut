// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Stage output cache.
//!
//! Skips re-execution of a stage when the inputs + args + stage
//! identity match a previous run's cached output. Per-job by
//! default; the
//! `--shared-cache` flag (commit 5) flips lookup to the global
//! cache at `~/.local/share/lamu/train-cache/` first, then job-local.
//!
//! Cache key formula:
//!
//! ```text
//! sha256(
//!   b"blut.cache.v2" ‖ 0x00 ‖
//!   stage_name (as bytes) ‖ 0x00 ‖
//!   stage_schema (LE u32) ‖
//!   input_content_hash (32 bytes) ‖
//!   code_sha_len (LE u64) ‖ code_sha ‖
//!   canonical(args_json)
//! )
//! ```
//!
//! `canonical(args_json)` = serde_json with object keys sorted
//! lexicographically. Field reorder doesn't invalidate; rename
//! does (semantic change). Test-covered.
//!
//! Cache roots contain two disjoint namespaces:
//!
//! - `v1/cache-invocations/<InvocationKey>` maps one invocation to a
//!   [`ArtifactContentId`] plus the expected artifact kind/schema.
//! - `v1/artifacts/<ArtifactContentId>` owns the canonical payload, portable
//!   handle, integrity metadata, and validation material.
//!
//! A lookup is not a metadata read. It restores and independently validates the
//! object beneath the consumer's stage directory before returning a handle.
//! Pre-A09 `<key>/output.bin` entries intentionally cold-miss: cache data is
//! non-authoritative and cannot be migrated safely because it owns no payload.

use std::path::{Path, PathBuf};

use bincode::Options;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::framework::artifact::{ArtifactContentId, ContentHash, InvocationKey};
use crate::framework::artifact_store::{ArtifactRole, StoredArtifact, capture, restore};
use crate::framework::object_store::{
    BlockingObjectStore, MAX_OBJECT_SIZE, ObjectKey, ObjectNamespace,
};
use crate::framework::stage::{ErasedArtifact, StageDyn};

const CACHE_RECORD_VERSION: u16 = 1;
const MAX_CACHE_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CACHE_RECORD_BYTES: u64 = 1024 * 1024;
const MAX_CACHE_PROOF_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheRecord {
    pub version: u16,
    pub invocation_key: InvocationKey,
    pub content_id: ArtifactContentId,
    pub kind: String,
    pub schema: u32,
}

/// Canonical object and invocation bytes held until optional execution commits.
pub(crate) struct OptionalCacheWrite {
    key: InvocationKey,
    pub(crate) content_id: ArtifactContentId,
    object_bytes: Vec<u8>,
    record_bytes: Vec<u8>,
}

#[cfg(test)]
type OptionalLocalInsertHook = std::sync::Arc<dyn Fn(&Path) + Send + Sync>;

#[cfg(test)]
fn optional_local_insert_hook() -> &'static std::sync::Mutex<Option<OptionalLocalInsertHook>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<OptionalLocalInsertHook>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
pub(crate) fn set_optional_local_insert_hook(hook: Option<OptionalLocalInsertHook>) {
    *optional_local_insert_hook()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = hook;
}

#[cfg(test)]
fn run_optional_local_insert_hook(path: &Path) {
    let hook = optional_local_insert_hook()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    if let Some(hook) = hook {
        hook(path);
    }
}

/// Per-job + (commit-5) global + (Tier-4) remote cache handle.
#[derive(Clone, Debug)]
pub struct CacheHandle {
    pub job_local: PathBuf,
    pub global: Option<PathBuf>,
    /// Optional content-addressed REMOTE tier (ADR 0067 T4.2): checked LAST on
    /// lookup (after the local dirs), and — on a remote hit — written through
    /// to the active local write target so the entry is a real local `CacheHit`.
    /// `insert` writes through to it too (best-effort). A shared cache across
    /// machines / pods.
    pub remote: Option<BlockingObjectStore>,
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
    pub fn with_remote(self, remote: BlockingObjectStore) -> Self {
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
    ) -> InvocationKey {
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
    ) -> InvocationKey {
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
    ) -> InvocationKey {
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
    ) -> InvocationKey {
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
        InvocationKey::from_digest(ContentHash(arr))
    }

    /// Expose `canonical_json` for the executor / plan compiler so
    /// the canonical bytes can be precomputed once per stage at
    /// plan-compile time.
    pub fn canonical_json_bytes(args: &serde_json::Value) -> Vec<u8> {
        canonical_json(args).into_bytes()
    }

    /// Resolve an invocation to a content object, restore that object beneath
    /// `into_stage_dir`, and validate it as this stage's output. Any missing,
    /// corrupt, mismatched, or unrehydratable value is a cache miss. Restore
    /// removes its private `.artifact-import/<ArtifactContentId>` subtree on failure;
    /// the executor's cold path removes the enclosing final stage directory
    /// before atomic promotion, so a failed lookup cannot poison a rerun.
    pub fn lookup(
        &self,
        key: InvocationKey,
        stage: &dyn StageDyn,
        into_stage_dir: &Path,
    ) -> Option<CacheHit> {
        for base in self.search_order() {
            let store = BlockingObjectStore::filesystem(base);
            let record_key = ObjectKey::CacheInvocation(key);
            let record_path = store
                .filesystem_path(record_key)
                .expect("filesystem store exposes paths");
            let record_bytes = match store.get(record_key) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => continue,
                Err(error) => {
                    tracing::warn!(
                        "cache: read {}: {error}; treating as miss",
                        record_path.display()
                    );
                    continue;
                }
            };
            let Some(record) = decode_record(&record_bytes, key, stage, &record_path) else {
                continue;
            };
            let object_key = ObjectKey::Artifact(record.content_id);
            let object_path = store
                .filesystem_path(object_key)
                .expect("filesystem store exposes paths");
            let object_bytes = match store.get(object_key) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => continue,
                Err(error) => {
                    tracing::warn!(
                        "cache: read {}: {error}; treating as miss",
                        object_path.display()
                    );
                    continue;
                }
            };
            if let Some(hit) =
                materialize_hit(&record, object_bytes, stage, into_stage_dir, &object_path)
            {
                return Some(hit);
            }
        }

        self.lookup_remote(key, stage, into_stage_dir)
    }

    fn lookup_remote(
        &self,
        key: InvocationKey,
        stage: &dyn StageDyn,
        into_stage_dir: &Path,
    ) -> Option<CacheHit> {
        let remote = self.remote.as_ref()?;
        let record_bytes = match remote.get(ObjectKey::CacheInvocation(key)) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return None,
            Err(error) => {
                tracing::warn!(
                    "cache: remote invocation lookup for {}: {error}; treating as miss",
                    key.to_hex()
                );
                return None;
            }
        };
        let diagnostic_path = PathBuf::from(format!("remote:invocations/{}", key.to_hex()));
        let record = decode_record(&record_bytes, key, stage, &diagnostic_path)?;
        let object_bytes = match remote.get(ObjectKey::Artifact(record.content_id)) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return None,
            Err(error) => {
                tracing::warn!(
                    "cache: remote object lookup for {}: {error}; treating as miss",
                    record.content_id
                );
                return None;
            }
        };
        // Publish an object orphan before consuming its allocation. Invocation
        // record remains the visibility marker and lands only after typed restore
        // validates the remote object.
        let local = self.write_store();
        let local_object_ready = match put_local_repairing_corrupt(
            &local,
            ObjectKey::Artifact(record.content_id),
            &object_bytes,
        ) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    "cache: remote object verified by storage envelope but local write-through failed: {error}"
                );
                false
            }
        };
        let hit = materialize_hit(
            &record,
            object_bytes,
            stage,
            into_stage_dir,
            &PathBuf::from(format!("remote:objects/{}", record.content_id)),
        )?;

        if local_object_ready
            && let Err(error) =
                put_local_repairing_corrupt(&local, ObjectKey::CacheInvocation(key), &record_bytes)
        {
            tracing::warn!(
                "cache: remote hit verified but invocation write-through failed: {error}"
            );
        }
        Some(hit)
    }

    /// Capture `output` into the canonical object representation, write the
    /// content object first, then publish the invocation record atomically.
    pub fn insert(
        &self,
        key: InvocationKey,
        stage: &dyn StageDyn,
        output: &ErasedArtifact,
        src_root: &Path,
    ) -> std::io::Result<ArtifactContentId> {
        let stored = capture(stage, output.clone(), src_root, ArtifactRole::Output, None)
            .map_err(artifact_store_io)?;
        self.insert_stored(key, stored)
    }

    /// Persist an artifact already captured by the canonical store. The executor
    /// uses this path so lineage identity and cache identity come from the exact
    /// same captured bytes without reading large artifacts twice.
    pub fn insert_stored(
        &self,
        key: InvocationKey,
        stored: StoredArtifact,
    ) -> std::io::Result<ArtifactContentId> {
        let content_id = stored.manifest.content_id;
        let record = CacheRecord {
            version: CACHE_RECORD_VERSION,
            invocation_key: key,
            content_id,
            kind: stored.manifest.kind.clone(),
            schema: stored.manifest.schema,
        };
        let object_bytes = encode_stored(stored)?;
        let record_bytes = bincode::serialize(&record).map_err(cache_encode_io)?;
        ensure_cache_bytes_bound(&object_bytes, MAX_OBJECT_SIZE, "content object")?;
        ensure_cache_bytes_bound(&record_bytes, MAX_CACHE_RECORD_BYTES, "invocation record")?;
        let target = self.write_store();
        put_local_repairing_corrupt(&target, ObjectKey::Artifact(content_id), &object_bytes)?;
        put_local_repairing_corrupt(&target, ObjectKey::CacheInvocation(key), &record_bytes)?;

        if let Some(remote) = &self.remote {
            match remote.put(ObjectKey::Artifact(content_id), &object_bytes) {
                Ok(_) => {
                    if let Err(error) = remote.put(ObjectKey::CacheInvocation(key), &record_bytes) {
                        tracing::warn!(
                            "cache: remote invocation write-through for {}: {error}",
                            key.to_hex()
                        );
                    }
                }
                Err(error) => tracing::warn!(
                    "cache: remote object write-through for {}: {error}",
                    content_id
                ),
            }
        }
        Ok(content_id)
    }

    /// Side-effect-free presence probe used by optional scheduler work.
    ///
    /// This deliberately does not deserialize, fetch, hydrate, or contact a
    /// shared provider. A false positive (for example, a corrupt local file or
    /// merely having a global/remote tier configured) only suppresses optional
    /// speculation; the ordinary path still performs the authoritative
    /// [`lookup`](Self::lookup). Shared tiers are treated as "possibly present"
    /// because even a metadata/HEAD request may block the single coordinator.
    pub(crate) fn probe_presence(&self, key: InvocationKey) -> std::io::Result<bool> {
        match BlockingObjectStore::filesystem(&self.job_local)
            .contains(ObjectKey::CacheInvocation(key))
        {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(error) => return Err(store_io(error)),
        }
        if self.global.is_some() || self.remote.is_some() {
            return Ok(true);
        }
        Ok(false)
    }

    /// Commit only the local/global filesystem tier for selected optional work.
    /// The executor keeps this write behind its publication rollback guard and
    /// calls [`replicate_optional`](Self::replicate_optional) only after the
    /// canonical stage/lifecycle commit.
    pub(crate) fn insert_optional_local(
        &self,
        key: InvocationKey,
        stored: StoredArtifact,
    ) -> std::io::Result<OptionalCacheWrite> {
        let content_id = stored.manifest.content_id;
        let record = CacheRecord {
            version: CACHE_RECORD_VERSION,
            invocation_key: key,
            content_id,
            kind: stored.manifest.kind.clone(),
            schema: stored.manifest.schema,
        };
        let object_bytes = encode_stored(stored)?;
        let record_bytes = bincode::serialize(&record).map_err(cache_encode_io)?;
        ensure_cache_bytes_bound(&object_bytes, MAX_OBJECT_SIZE, "content object")?;
        ensure_cache_bytes_bound(&record_bytes, MAX_CACHE_RECORD_BYTES, "invocation record")?;
        let target = self.write_store();
        put_local_repairing_corrupt(&target, ObjectKey::Artifact(content_id), &object_bytes)?;
        put_local_repairing_corrupt(&target, ObjectKey::CacheInvocation(key), &record_bytes)?;
        #[cfg(test)]
        run_optional_local_insert_hook(&self.entry_path_for_write(key));
        Ok(OptionalCacheWrite {
            key,
            content_id,
            object_bytes,
            record_bytes,
        })
    }

    /// Best-effort remote replication after an optional result is canonical.
    /// Plugin panics and remote errors remain contained; the committed local
    /// entry is already sufficient for correctness.
    pub(crate) fn replicate_optional(&self, write: &OptionalCacheWrite) {
        if let Some(remote) = &self.remote {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                remote.put(ObjectKey::Artifact(write.content_id), &write.object_bytes)?;
                remote.put(ObjectKey::CacheInvocation(write.key), &write.record_bytes)
            })) {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => {
                    tracing::warn!(
                        "cache: optional remote write-through for {}: {error}",
                        write.key.to_hex()
                    );
                }
                Err(_) => {
                    tracing::warn!(
                        "cache: optional remote write-through for {} panicked; local entry retained",
                        write.key.to_hex()
                    );
                }
            }
        }
    }

    /// Exact local entry path an insert writes for this handle/key.
    pub(crate) fn entry_path_for_write(&self, key: InvocationKey) -> PathBuf {
        self.write_store()
            .filesystem_path(ObjectKey::CacheInvocation(key))
            .expect("local/global cache targets are filesystem stores")
    }

    pub(crate) fn remove_invocation(&self, key: InvocationKey) -> std::io::Result<bool> {
        self.write_store()
            .remove(ObjectKey::CacheInvocation(key))
            .map_err(store_io)
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

    fn write_store(&self) -> BlockingObjectStore {
        BlockingObjectStore::filesystem(self.write_target())
    }
}

#[cfg(test)]
fn record_path(base: &Path, key: InvocationKey) -> PathBuf {
    base.join(ObjectKey::CacheInvocation(key).relative_path())
}

#[cfg(test)]
fn object_path(base: &Path, content_id: ArtifactContentId) -> PathBuf {
    base.join(ObjectKey::Artifact(content_id).relative_path())
}

fn ensure_cache_bytes_bound(bytes: &[u8], max_bytes: u64, label: &str) -> std::io::Result<()> {
    if bytes.len() as u64 > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cache {label} is {} bytes, above its {max_bytes}-byte bound",
                bytes.len()
            ),
        ));
    }
    Ok(())
}

fn read_file_capped(path: &Path, max_bytes: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read;

    let file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    if size > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cache file {} is {size} bytes, above its {max_bytes}-byte bound",
                path.display()
            ),
        ));
    }
    let capacity = usize::try_from(size).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache file {} does not fit address space", path.display()),
        )
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|error| {
        std::io::Error::other(format!(
            "reserve {capacity} bytes for cache file {}: {error}",
            path.display()
        ))
    })?;
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache file {} grew beyond its bound", path.display()),
        ));
    }
    Ok(bytes)
}

fn deserialize_capped<T: DeserializeOwned>(bytes: &[u8], max_bytes: u64) -> bincode::Result<T> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(max_bytes)
        .deserialize(bytes)
}

/// Encode metadata ahead of the existing pack allocation. `StoredArtifact`'s
/// bincode layout is `manifest || vec_length || pack`, so serializing only the
/// manifest and writing the vector length preserves byte compatibility while
/// avoiding a second full-pack allocation.
fn encode_stored(stored: StoredArtifact) -> std::io::Result<Vec<u8>> {
    let manifest_bytes = bincode::serialize(&stored.manifest).map_err(cache_encode_io)?;
    ensure_cache_bytes_bound(
        &manifest_bytes,
        MAX_CACHE_MANIFEST_BYTES,
        "artifact manifest",
    )?;
    let prefix_len = std::mem::size_of::<u64>()
        .checked_add(manifest_bytes.len())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cache object size overflow",
            )
        })?;
    let final_len = prefix_len.checked_add(stored.pack.len()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "cache object size overflow",
        )
    })?;
    if final_len as u64 > MAX_OBJECT_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cache content object is {final_len} bytes, above its {MAX_OBJECT_SIZE}-byte bound"
            ),
        ));
    }

    let mut object = stored.pack;
    object.try_reserve_exact(prefix_len).map_err(|error| {
        std::io::Error::other(format!(
            "reserve {prefix_len} bytes for cache object header: {error}"
        ))
    })?;
    let pack_len = object.len();
    object.resize(final_len, 0);
    object.copy_within(..pack_len, prefix_len);
    object[..manifest_bytes.len()].copy_from_slice(&manifest_bytes);
    object[manifest_bytes.len()..prefix_len].copy_from_slice(&(pack_len as u64).to_le_bytes());
    Ok(object)
}

/// Decode the manifest, then compact pack bytes over the header in the same
/// allocation returned by the object store.
fn decode_stored(mut object: Vec<u8>) -> std::io::Result<StoredArtifact> {
    use std::io::Cursor;

    ensure_cache_bytes_bound(&object, MAX_OBJECT_SIZE, "content object")?;
    let mut cursor = Cursor::new(object.as_slice());
    let manifest = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(MAX_CACHE_MANIFEST_BYTES)
        .deserialize_from::<_, crate::framework::artifact_store::ArtifactManifest>(&mut cursor)
        .map_err(cache_encode_io)?;
    let manifest_end = usize::try_from(cursor.position()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "cache manifest does not fit address space",
        )
    })?;
    let pack_offset = manifest_end
        .checked_add(std::mem::size_of::<u64>())
        .filter(|offset| *offset <= object.len())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cache pack length is truncated",
            )
        })?;
    let declared_pack_len = u64::from_le_bytes(
        object[manifest_end..pack_offset]
            .try_into()
            .expect("fixed length slice"),
    );
    let pack_len = object.len() - pack_offset;
    if declared_pack_len != pack_len as u64 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache pack length {declared_pack_len} != stored {pack_len}"),
        ));
    }
    object.copy_within(pack_offset.., 0);
    object.truncate(pack_len);
    Ok(StoredArtifact {
        manifest,
        pack: object,
    })
}

fn decode_record(
    bytes: &[u8],
    key: InvocationKey,
    stage: &dyn StageDyn,
    source: &Path,
) -> Option<CacheRecord> {
    let record = match deserialize_capped::<CacheRecord>(bytes, MAX_CACHE_RECORD_BYTES) {
        Ok(record) => record,
        Err(error) => {
            tracing::warn!(
                "cache: corrupt invocation record at {}: {error}; treating as miss",
                source.display()
            );
            return None;
        }
    };
    if record.version != CACHE_RECORD_VERSION
        || record.invocation_key != key
        || record.kind != stage.output_kind()
        || record.schema != stage.output_schema()
    {
        tracing::warn!(
            "cache: invocation record contract mismatch at {}; treating as miss",
            source.display()
        );
        return None;
    }
    Some(record)
}

fn materialize_hit(
    record: &CacheRecord,
    object_bytes: Vec<u8>,
    stage: &dyn StageDyn,
    into_stage_dir: &Path,
    source: &Path,
) -> Option<CacheHit> {
    let stored = match decode_stored(object_bytes) {
        Ok(stored) => stored,
        Err(error) => {
            tracing::warn!(
                "cache: corrupt content object at {}: {error}; treating as miss",
                source.display()
            );
            return None;
        }
    };
    if stored.manifest.content_id != record.content_id
        || stored.manifest.kind != record.kind
        || stored.manifest.schema != record.schema
    {
        tracing::warn!(
            "cache: content object contract mismatch at {}; treating as miss",
            source.display()
        );
        return None;
    }
    match restore(
        stage,
        &stored,
        into_stage_dir,
        ArtifactRole::Output,
        Some(record.content_id),
    ) {
        Ok(artifact) => Some(CacheHit {
            artifact,
            content_id: record.content_id,
        }),
        Err(error) => {
            tracing::warn!(
                "cache: failed to validate content object at {}: {error}; treating as miss",
                source.display()
            );
            None
        }
    }
}

fn cache_encode_io(error: bincode::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error)
}

fn store_io(error: crate::framework::object_store::StoreError) -> std::io::Error {
    std::io::Error::other(error)
}

fn put_local_repairing_corrupt(
    store: &BlockingObjectStore,
    key: ObjectKey,
    bytes: &[u8],
) -> std::io::Result<()> {
    match store.put(key, bytes) {
        Ok(_) => Ok(()),
        Err(crate::framework::object_store::StoreError::Corrupt { .. }) => {
            store.remove(key).map_err(store_io)?;
            store.put(key, bytes).map(|_| ()).map_err(store_io)
        }
        Err(error) => Err(store_io(error)),
    }
}

fn artifact_store_io(
    error: crate::framework::artifact_store::ArtifactStoreError,
) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error)
}

/// LRU prune invocation records until the complete cache root is within the
/// limit. A content object is removed only after its final local invocation
/// reference disappears; unreferenced objects from interrupted writes are
/// eligible first. Best-effort I/O failures are logged and skipped.
///
/// Callers must not run pruning concurrently with writers targeting the same
/// root. Addressing, listing, validation, and deletion all flow through the
/// canonical typed object store; cache owns only reference/LRU decisions.
///
/// `max_bytes`: cap, e.g. 50 GiB. Default driven by
/// `$LAMU_CACHE_MAX_GB` (commit 8 wires the CLI knob).
pub fn lru_prune(cache_root: &Path, max_bytes: u64) -> std::io::Result<u64> {
    let store = BlockingObjectStore::filesystem(cache_root);
    let invocation_objects = store
        .list_namespace(ObjectNamespace::CacheInvocation)
        .map_err(store_io)?;
    let artifact_objects = store
        .list_namespace(ObjectNamespace::Artifact)
        .map_err(store_io)?;
    let mut total = invocation_objects
        .iter()
        .chain(&artifact_objects)
        .fold(0_u64, |sum, object| sum.saturating_add(object.stored_size));
    if total <= max_bytes {
        return Ok(0);
    }

    let mut entries = Vec::new();
    let mut references: std::collections::HashMap<ArtifactContentId, usize> =
        std::collections::HashMap::new();
    for entry in invocation_objects {
        let content_id = store
            .get(entry.key)
            .ok()
            .flatten()
            .and_then(|bytes| {
                deserialize_capped::<CacheRecord>(&bytes, MAX_CACHE_RECORD_BYTES).ok()
            })
            .filter(|record| record.version == CACHE_RECORD_VERSION)
            .map(|record| record.content_id);
        if let Some(content_id) = content_id {
            *references.entry(content_id).or_default() += 1;
        }
        entries.push((entry, content_id));
    }

    let mut freed = 0u64;
    let mut artifact_by_id: std::collections::HashMap<ArtifactContentId, _> = artifact_objects
        .into_iter()
        .filter_map(|object| match object.key {
            ObjectKey::Artifact(content_id) => Some((content_id, object)),
            _ => None,
        })
        .collect();
    let mut orphans: Vec<_> = artifact_by_id
        .iter()
        .filter(|(content_id, _)| !references.contains_key(content_id))
        .map(|(content_id, object)| (*content_id, *object))
        .collect();
    orphans.sort_by_key(|(_, object)| object.accessed);
    for (content_id, object) in orphans {
        if total <= max_bytes {
            break;
        }
        match store.remove(object.key) {
            Ok(true) => {
                total = total.saturating_sub(object.stored_size);
                freed += object.stored_size;
                artifact_by_id.remove(&content_id);
            }
            Ok(false) => {}
            Err(error) => {
                tracing::warn!("lru_prune: failed to remove {}: {error}", object.key);
            }
        }
    }

    entries.sort_by_key(|(object, _)| object.accessed);
    for (object, content_id) in entries {
        if total <= max_bytes {
            break;
        }
        match store.remove(object.key) {
            Ok(true) => {
                total = total.saturating_sub(object.stored_size);
                freed += object.stored_size;
                if let Some(content_id) = content_id
                    && let Some(count) = references.get_mut(&content_id)
                {
                    *count -= 1;
                    if *count == 0
                        && let Some(artifact) = artifact_by_id.remove(&content_id)
                    {
                        match store.remove(artifact.key) {
                            Ok(true) => {
                                total = total.saturating_sub(artifact.stored_size);
                                freed += artifact.stored_size;
                            }
                            Ok(false) => {}
                            Err(error) => tracing::warn!(
                                "lru_prune: failed to remove {}: {error}",
                                artifact.key
                            ),
                        }
                    }
                }
            }
            Ok(false) => {}
            Err(error) => {
                tracing::warn!("lru_prune: failed to remove {}: {error}", object.key);
            }
        }
    }
    Ok(freed)
}

#[cfg(test)]
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

#[derive(Clone, Debug)]
pub struct CacheHit {
    pub artifact: ErasedArtifact,
    pub content_id: ArtifactContentId,
}

/// Durable proof tying a completed stage to the cache entry that made it
/// skippable. Written inside the current job's stage dir on both hits and
/// misses so lineage remains complete for all-cache-hit jobs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheProof {
    pub key: InvocationKey,
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
        let body = read_file_capped(path, MAX_CACHE_PROOF_BYTES)?;
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("parse cache proof: {e}"),
            )
        })
    }

    pub fn is_live(&self) -> bool {
        let Some(base) = self.entry_path.ancestors().nth(3) else {
            return false;
        };
        let store = BlockingObjectStore::filesystem(base);
        let record_key = ObjectKey::CacheInvocation(self.key);
        if store.filesystem_path(record_key).as_deref() != Some(self.entry_path.as_path()) {
            return false;
        }
        let Some(record) = store
            .get(record_key)
            .ok()
            .flatten()
            .and_then(|body| deserialize_capped::<CacheRecord>(&body, MAX_CACHE_RECORD_BYTES).ok())
            .filter(|record| {
                record.version == CACHE_RECORD_VERSION && record.invocation_key == self.key
            })
        else {
            return false;
        };
        store
            .get(ObjectKey::Artifact(record.content_id))
            .ok()
            .flatten()
            .and_then(|body| decode_stored(body).ok())
            .is_some_and(|stored| stored.manifest.content_id == record.content_id)
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

pub(crate) fn write_atomic(dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stem = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "tmp".into());
    let tmp = dest.with_file_name(format!(".{stem}.tmp.{}", uuid::Uuid::new_v4()));
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
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};

    use crate::framework::artifact::Artifact;
    use crate::framework::error::StageError;
    use crate::framework::resource::Resource;
    use crate::framework::stage::{Stage, StageContext};

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct FileArtifact {
        path: PathBuf,
        content_hash: ContentHash,
    }

    impl Artifact for FileArtifact {
        const KIND: &'static str = "test.cache-file";
        const SCHEMA: u32 = 1;

        fn content_hash(&self) -> ContentHash {
            self.content_hash
        }

        fn primary_path(&self) -> &Path {
            &self.path
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct NoArgs;

    struct FileStage;

    #[async_trait]
    impl Stage for FileStage {
        const NAME: &'static str = "cache_file_stage";
        const SCHEMA: u32 = 7;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = FileArtifact;
        type Output = FileArtifact;
        type Args = NoArgs;

        async fn run(
            &self,
            _ctx: &StageContext,
            input: Self::Input,
            _args: &Self::Args,
        ) -> Result<Self::Output, StageError> {
            Ok(input)
        }
    }

    fn file_artifact(root: &Path, body: &[u8]) -> (ErasedArtifact, ContentHash) {
        std::fs::create_dir_all(root).unwrap();
        let path = root.join("payload.bin");
        std::fs::write(&path, body).unwrap();
        let hash = ContentHash::hash_file(&path).unwrap();
        let erased = ErasedArtifact::from_typed(&FileArtifact {
            path,
            content_hash: hash,
        })
        .unwrap();
        (erased, hash)
    }

    fn restored_file(hit: &CacheHit) -> FileArtifact {
        hit.artifact.clone().into_typed().unwrap()
    }

    const CS: &[u8] = b"code-sha-fixture";

    fn invocation(bytes: &[u8]) -> InvocationKey {
        InvocationKey::from_digest(ContentHash::of_bytes(bytes))
    }

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
        let key = invocation(b"missing");
        assert!(
            h.lookup(key, &FileStage, &td.path().join("consumer"))
                .is_none()
        );
    }

    #[test]
    fn insert_then_lookup_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let cache = td.path().join("cache");
        let producer = td.path().join("producer");
        let consumer = td.path().join("consumer");
        let h = CacheHandle::job_local(cache);
        let key = invocation(b"k");
        let (artifact, logical_hash) = file_artifact(&producer, b"portable bytes");
        let content_id = h.insert(key, &FileStage, &artifact, &producer).unwrap();
        assert_ne!(content_id.digest(), logical_hash);
        std::fs::remove_dir_all(&producer).unwrap();

        let hit = h
            .lookup(key, &FileStage, &consumer)
            .expect("should rehydrate after producer deletion");
        let restored = restored_file(&hit);
        assert_eq!(hit.content_id, content_id);
        assert!(restored.path.starts_with(&consumer));
        assert_eq!(std::fs::read(restored.path).unwrap(), b"portable bytes");
    }

    #[test]
    fn lookup_returns_none_on_corrupt_entry() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = invocation(b"k");
        let path = record_path(td.path(), key);
        write_atomic(&path, &[0xFFu8; 3]).unwrap();
        assert!(
            h.lookup(key, &FileStage, &td.path().join("consumer"))
                .is_none()
        );
    }

    #[test]
    fn lookup_rejects_oversized_local_record_before_reading_it() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = invocation(b"oversized-record");
        let path = record_path(td.path(), key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::File::create(&path)
            .unwrap()
            .set_len(MAX_CACHE_RECORD_BYTES + 1)
            .unwrap();

        assert!(
            h.lookup(key, &FileStage, &td.path().join("consumer"))
                .is_none(),
            "oversized sparse record must be rejected from metadata"
        );
    }

    #[test]
    fn lookup_rejects_oversized_local_object_before_reading_it() {
        let td = tempfile::tempdir().unwrap();
        let cache = td.path().join("cache");
        let producer = td.path().join("producer");
        let h = CacheHandle::job_local(cache.clone());
        let key = invocation(b"oversized-object");
        let (artifact, _) = file_artifact(&producer, b"small valid payload");
        let content_id = h.insert(key, &FileStage, &artifact, &producer).unwrap();
        // MAX_STORED_SIZE + 1, not MAX_OBJECT_SIZE + 1. The metadata guard in
        // `BlockingObjectStore::get` compares the FILE length against
        // MAX_STORED_SIZE (payload + header), so a file of MAX_OBJECT_SIZE + 1
        // is BELOW it: the cheap check passes and execution falls through to
        // `read_to_end`, which pulls all 16 GiB into memory before the payload
        // check rejects it. The assertion still held — the object was rejected
        // — so the test looked fine on a machine with the RAM to absorb it, and
        // hung the CI runner, taking the whole suite's summary with it. The
        // name says "before reading it"; this is the size that makes that true.
        std::fs::OpenOptions::new()
            .write(true)
            .open(object_path(&cache, content_id))
            .unwrap()
            .set_len(crate::framework::object_store::MAX_STORED_SIZE + 1)
            .unwrap();

        assert!(
            h.lookup(key, &FileStage, &td.path().join("consumer"))
                .is_none(),
            "oversized sparse object must be rejected from metadata"
        );
    }

    #[test]
    fn cache_decode_reuses_object_allocation_for_pack() {
        let td = tempfile::tempdir().unwrap();
        let producer = td.path().join("producer");
        let (artifact, _) = file_artifact(&producer, b"allocation identity fixture");
        let stored = capture(&FileStage, artifact, &producer, ArtifactRole::Output, None).unwrap();
        let object = encode_stored(stored).unwrap();
        let allocation = object.as_ptr();

        let decoded = decode_stored(object).unwrap();

        assert_eq!(decoded.pack.as_ptr(), allocation);
    }

    /// A truncated once-valid invocation record must downgrade to a
    /// miss (return `None`, triggering a re-run) — never panic and never
    /// hand back a garbage / partially-decoded artifact.
    #[test]
    fn lookup_downgrades_truncated_valid_entry_to_miss() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = invocation(b"truncated");
        let record = CacheRecord {
            version: CACHE_RECORD_VERSION,
            invocation_key: key,
            content_id: ArtifactContentId::from_digest(ContentHash::of_bytes(b"content")),
            kind: FileArtifact::KIND.into(),
            schema: FileArtifact::SCHEMA,
        };
        let good = bincode::serialize(&record).unwrap();
        assert!(good.len() > 4, "fixture must be long enough to truncate");
        write_atomic(&record_path(td.path(), key), &good[..good.len() / 2]).unwrap();

        // No panic, and the lookup reports a clean miss.
        assert!(
            h.lookup(key, &FileStage, &td.path().join("consumer"))
                .is_none(),
            "truncated bincode must downgrade to a cache miss"
        );
    }

    /// §5.1 "Cache CORRUPT `.bin`": pure garbage (not even a valid
    /// bincode prefix) must also downgrade to a miss without panicking.
    #[test]
    fn lookup_downgrades_garbage_entry_to_miss() {
        let td = tempfile::tempdir().unwrap();
        let h = CacheHandle::job_local(td.path().to_path_buf());
        let key = invocation(b"garbage");
        let path = record_path(td.path(), key);
        // A bincode length prefix claiming a huge string, followed by no
        // data — the classic "allocator bomb" corrupt-frame shape. The
        // deserializer must error (not OOM / panic), and lookup returns
        // None.
        let mut garbage = Vec::new();
        garbage.extend_from_slice(&u64::MAX.to_le_bytes()); // bogus length
        garbage.extend_from_slice(b"\x00not-a-valid-record\xff\xfe");
        write_atomic(&path, &garbage).unwrap();
        assert!(
            h.lookup(key, &FileStage, &td.path().join("consumer"))
                .is_none(),
            "garbage bytes must downgrade to a cache miss, not panic"
        );

        // Empty file is also corrupt-shaped (truncated to zero) → miss.
        write_atomic(&path, b"").unwrap();
        assert!(
            h.lookup(key, &FileStage, &td.path().join("consumer"))
                .is_none(),
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
        let cache = td.path().join("cache");
        let producer = td.path().join("producer");
        let h = CacheHandle::job_local(cache.clone());
        let key = invocation(b"recover");
        write_atomic(&record_path(&cache, key), &[0x01u8, 0x02, 0x03]).unwrap();
        assert!(
            h.lookup(key, &FileStage, &td.path().join("miss")).is_none(),
            "corrupt first read -> miss"
        );
        // Overwrite with a valid record (insert uses atomic rename).
        let (artifact, _) = file_artifact(&producer, b"recovered");
        h.insert(key, &FileStage, &artifact, &producer).unwrap();
        let hit = h
            .lookup(key, &FileStage, &td.path().join("consumer"))
            .expect("valid entry must now hit");
        assert_eq!(
            std::fs::read(restored_file(&hit).path).unwrap(),
            b"recovered"
        );
    }

    #[test]
    fn shared_cache_writes_go_to_global() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("job");
        let global = td.path().join("global");
        let producer = td.path().join("producer");
        std::fs::create_dir_all(&job).unwrap();
        std::fs::create_dir_all(&global).unwrap();
        let h = CacheHandle::job_local(job).with_global(global.clone());
        let key = invocation(b"k");
        let (artifact, _) = file_artifact(&producer, b"global");
        let content_id = h.insert(key, &FileStage, &artifact, &producer).unwrap();
        assert!(record_path(&global, key).exists());
        assert!(object_path(&global, content_id).exists());
    }

    #[test]
    fn remote_tier_write_through_and_hit() {
        use crate::framework::object_store::{BlockingObjectStore, ObjectKey};
        let td = tempfile::tempdir().unwrap();
        let remote = BlockingObjectStore::filesystem(td.path().join("remote"));
        let key = invocation(b"k");

        // Machine A: insert → writes local AND through to the remote store.
        let a_job = td.path().join("a");
        let a_producer = td.path().join("a-producer");
        let h_a = CacheHandle::job_local(a_job).with_remote(remote.clone());
        let (artifact, _) = file_artifact(&a_producer, b"remote payload");
        let content_id = h_a.insert(key, &FileStage, &artifact, &a_producer).unwrap();
        assert!(
            remote.contains(ObjectKey::CacheInvocation(key)).unwrap(),
            "invocation record wrote through to remote"
        );
        assert!(
            remote.contains(ObjectKey::Artifact(content_id)).unwrap(),
            "content object wrote through to remote"
        );
        std::fs::remove_dir_all(&a_producer).unwrap();

        // Machine B: cold local, same remote -> verified B-local materialization.
        let b_job = td.path().join("b");
        let b_consumer = td.path().join("b-consumer");
        let h_b = CacheHandle::job_local(b_job.clone()).with_remote(remote.clone());
        let hit = h_b
            .lookup(key, &FileStage, &b_consumer)
            .expect("remote tier serves the entry");
        let restored = restored_file(&hit);
        assert!(restored.path.starts_with(&b_consumer));
        assert_eq!(std::fs::read(restored.path).unwrap(), b"remote payload");
        assert!(record_path(&b_job, key).exists());
        assert!(object_path(&b_job, content_id).exists());
    }

    #[test]
    fn remote_hit_writes_through_to_global_when_shared_cache_is_enabled() {
        use crate::framework::object_store::BlockingObjectStore;
        let td = tempfile::tempdir().unwrap();
        let remote = BlockingObjectStore::filesystem(td.path().join("remote"));
        let key = invocation(b"remote-global");
        let producer = td.path().join("producer");
        let (artifact, _) = file_artifact(&producer, b"shared remote payload");
        let content_id = CacheHandle::job_local(td.path().join("seed"))
            .with_remote(remote.clone())
            .insert(key, &FileStage, &artifact, &producer)
            .unwrap();

        let job = td.path().join("job");
        let global = td.path().join("global");
        let handle = CacheHandle::job_local(job.clone())
            .with_global(global.clone())
            .with_remote(remote);
        handle
            .lookup(key, &FileStage, &td.path().join("consumer"))
            .expect("remote hit must materialize");

        assert!(record_path(&global, key).exists());
        assert!(object_path(&global, content_id).exists());
        assert!(!record_path(&job, key).exists());
        assert!(!object_path(&job, content_id).exists());
    }

    #[test]
    fn presence_probe_sees_remote_without_hydrating_job_cache() {
        use crate::framework::object_store::BlockingObjectStore;
        let td = tempfile::tempdir().unwrap();
        let remote = BlockingObjectStore::filesystem(td.path().join("remote"));
        let key = invocation(b"probe-remote");
        let producer = td.path().join("producer");
        let (artifact, _) = file_artifact(&producer, b"presence probe payload");
        let content_id = CacheHandle::job_local(td.path().join("seed"))
            .with_remote(remote.clone())
            .insert(key, &FileStage, &artifact, &producer)
            .unwrap();

        let consumer = td.path().join("consumer");
        let job_cache = td.path().join("consumer-cache");
        let handle = CacheHandle::job_local(job_cache.clone()).with_remote(remote);
        assert!(handle.probe_presence(key).unwrap());
        assert!(
            !record_path(&job_cache, key).exists() && !object_path(&job_cache, content_id).exists(),
            "an optional presence check must not hydrate canonical job state"
        );
        assert!(handle.lookup(key, &FileStage, &consumer).is_some());
        assert!(record_path(&job_cache, key).is_file());
        assert!(object_path(&job_cache, content_id).is_file());
    }

    #[test]
    fn presence_probe_treats_shared_tier_as_unknown_without_provider_io() {
        let td = tempfile::tempdir().unwrap();
        let key = invocation(b"probe-remote-provider");
        let remote_root = td.path().join("remote-is-a-file");
        std::fs::write(&remote_root, b"not a store root").unwrap();
        let handle = CacheHandle::job_local(td.path().join("consumer"))
            .with_remote(BlockingObjectStore::filesystem(remote_root.clone()));

        assert!(
            handle.probe_presence(key).unwrap(),
            "unknown shared state must conservatively suppress optional work"
        );
        assert_eq!(std::fs::read(&remote_root).unwrap(), b"not a store root");
        assert!(
            handle
                .lookup(key, &FileStage, &td.path().join("restore"))
                .is_none()
        );
    }

    #[test]
    fn presence_probe_suppresses_optional_work_without_parsing_corrupt_bytes() {
        let td = tempfile::tempdir().unwrap();
        let key = invocation(b"probe-corrupt");
        let path = record_path(td.path(), key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = [0xFF, 0x01, 0x02];
        std::fs::write(&path, bytes).unwrap();
        let handle = CacheHandle::job_local(td.path().to_path_buf());

        assert!(handle.probe_presence(key).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(
            handle
                .lookup(key, &FileStage, &td.path().join("restore"))
                .is_none(),
            "ordinary lookup remains the authoritative validity check"
        );
    }

    #[test]
    fn remote_error_degrades_to_a_miss() {
        // A remote whose root can't be read → lookup is a miss, not a panic.
        use crate::framework::object_store::BlockingObjectStore;
        let td = tempfile::tempdir().unwrap();
        // A filesystem store over a missing dir returns None, never errors on
        // get; the handle must simply report no hit.
        let remote = BlockingObjectStore::filesystem(PathBuf::from("/no-such-remote-xyz"));
        let h = CacheHandle::job_local(td.path().join("job")).with_remote(remote);
        assert!(
            h.lookup(
                invocation(b"absent"),
                &FileStage,
                &td.path().join("consumer")
            )
            .is_none()
        );
    }

    #[test]
    fn shared_cache_lookup_prefers_global() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("job");
        let global = td.path().join("global");
        let key = invocation(b"k");
        let job_producer = td.path().join("job-producer");
        let global_producer = td.path().join("global-producer");
        let (job_artifact, _) = file_artifact(&job_producer, b"job");
        CacheHandle::job_local(job.clone())
            .insert(key, &FileStage, &job_artifact, &job_producer)
            .unwrap();
        let (global_artifact, _) = file_artifact(&global_producer, b"global");
        CacheHandle::job_local(global.clone())
            .insert(key, &FileStage, &global_artifact, &global_producer)
            .unwrap();
        let h = CacheHandle::job_local(job).with_global(global);
        let hit = h
            .lookup(key, &FileStage, &td.path().join("consumer"))
            .expect("must hit");
        assert_eq!(std::fs::read(restored_file(&hit).path).unwrap(), b"global");
    }

    #[test]
    fn lru_prune_removes_oldest_until_under_cap() {
        let td = tempfile::tempdir().unwrap();
        let cache = CacheHandle::job_local(td.path().join("cache"));
        for (index, byte) in [1u8, 2, 3].into_iter().enumerate() {
            let producer = td.path().join(format!("producer-{index}"));
            let (artifact, _) = file_artifact(&producer, &vec![byte; 1024]);
            cache
                .insert(
                    invocation(format!("entry-{index}").as_bytes()),
                    &FileStage,
                    &artifact,
                    &producer,
                )
                .unwrap();
        }
        let root = td.path().join("cache");
        let before = dir_size(&root).unwrap();
        let freed = lru_prune(&root, before - 1).unwrap();
        assert!(freed > 0);
        assert!(dir_size(&root).unwrap() < before);
    }

    #[test]
    fn lru_prune_noop_when_under_cap() {
        let td = tempfile::tempdir().unwrap();
        let cache_root = td.path().join("cache");
        let producer = td.path().join("producer");
        let cache = CacheHandle::job_local(cache_root.clone());
        let (artifact, _) = file_artifact(&producer, b"small");
        cache
            .insert(invocation(b"entry"), &FileStage, &artifact, &producer)
            .unwrap();
        let freed = lru_prune(&cache_root, dir_size(&cache_root).unwrap() + 1).unwrap();
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
        let cache = td.path().join("cache");
        let producer = td.path().join("producer");
        let h = CacheHandle::job_local(cache.clone());
        let key = invocation(b"k");
        let (artifact, _) = file_artifact(&producer, b"atomic");
        let content_id = h.insert(key, &FileStage, &artifact, &producer).unwrap();
        let entries: Vec<_> = [record_path(&cache, key), object_path(&cache, content_id)]
            .into_iter()
            .flat_map(|path| {
                std::fs::read_dir(path.parent().unwrap())
                    .unwrap()
                    .filter_map(|entry| entry.ok())
                    .collect::<Vec<_>>()
            })
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            entries.is_empty(),
            "no tmp files should survive successful insert"
        );
    }
}
