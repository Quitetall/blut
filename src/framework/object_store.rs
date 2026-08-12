// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Canonical object-storage policy for cache and distributed execution.
//!
//! [`ObjectStore`] is the asynchronous production facade. It owns typed key
//! namespaces, immutable create semantics, maximum object size, integrity
//! envelopes, errors, and the filesystem/provider adapters. Sync cache and
//! chunk callers use [`BlockingObjectStore`]; they do not define a second
//! storage policy.
//!
//! `CacheInvocation` and `Artifact` are semantic addresses whose payloads are
//! validated by their owner modules. `DispatchBundle` and `Chunk` are direct
//! SHA-256 content addresses and are additionally bound to their payload bytes
//! here. Every namespace still carries an envelope checksum, so storage-level
//! corruption is detected before a caller decodes the value.

use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "cloud")]
use object_store::ObjectStoreExt;

use crate::framework::artifact::{ContentHash, ContentId, InvocationKey};

/// Maximum payload accepted by every production object-store adapter.
pub const MAX_OBJECT_SIZE: u64 = 16 * 1024 * 1024 * 1024;

const OBJECT_MAGIC: &[u8; 8] = b"BLUTOS01";
const OBJECT_FORMAT_VERSION: u16 = 1;
const OBJECT_HEADER_LEN: usize = 8 + 2 + 1 + 32 + 8 + 32;
const MAX_STORED_SIZE: u64 = MAX_OBJECT_SIZE + OBJECT_HEADER_LEN as u64;

/// Closed storage namespaces. A digest cannot alias a different kind of value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ObjectNamespace {
    CacheInvocation,
    Artifact,
    DispatchBundle,
    Chunk,
}

impl ObjectNamespace {
    fn tag(self) -> u8 {
        match self {
            Self::CacheInvocation => 1,
            Self::Artifact => 2,
            Self::DispatchBundle => 3,
            Self::Chunk => 4,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::CacheInvocation),
            2 => Some(Self::Artifact),
            3 => Some(Self::DispatchBundle),
            4 => Some(Self::Chunk),
            _ => None,
        }
    }

    fn path_segment(self) -> &'static str {
        match self {
            Self::CacheInvocation => "cache-invocations",
            Self::Artifact => "artifacts",
            Self::DispatchBundle => "dispatch-bundles",
            Self::Chunk => "chunks",
        }
    }

    fn payload_is_content_addressed(self) -> bool {
        matches!(self, Self::DispatchBundle | Self::Chunk)
    }
}

/// Typed address for an immutable stored object.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ObjectKey {
    CacheInvocation(InvocationKey),
    Artifact(ContentId),
    DispatchBundle(ContentHash),
    Chunk(ContentHash),
}

impl ObjectKey {
    pub fn namespace(self) -> ObjectNamespace {
        match self {
            Self::CacheInvocation(_) => ObjectNamespace::CacheInvocation,
            Self::Artifact(_) => ObjectNamespace::Artifact,
            Self::DispatchBundle(_) => ObjectNamespace::DispatchBundle,
            Self::Chunk(_) => ObjectNamespace::Chunk,
        }
    }

    pub fn digest(self) -> ContentHash {
        match self {
            Self::CacheInvocation(key) => key.digest(),
            Self::Artifact(content_id) => content_id.digest(),
            Self::DispatchBundle(hash) | Self::Chunk(hash) => hash,
        }
    }

    /// Canonical backend-relative address. The layout is part of storage policy,
    /// not selected independently by cache, P2P, or cloud callers.
    pub fn relative_path(self) -> String {
        format!(
            "v{OBJECT_FORMAT_VERSION}/{}/{}",
            self.namespace().path_segment(),
            self.digest().to_hex()
        )
    }

    fn from_parts(namespace: ObjectNamespace, digest: ContentHash) -> Self {
        match namespace {
            ObjectNamespace::CacheInvocation => {
                Self::CacheInvocation(InvocationKey::from_digest(digest))
            }
            ObjectNamespace::Artifact => Self::Artifact(ContentId::from_digest(digest)),
            ObjectNamespace::DispatchBundle => Self::DispatchBundle(digest),
            ObjectNamespace::Chunk => Self::Chunk(digest),
        }
    }
}

impl std::fmt::Display for ObjectKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}",
            self.namespace().path_segment(),
            self.digest().to_hex()
        )
    }
}

/// Canonical storage failures shared by every adapter and caller mode.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("object {key} is too large ({size} bytes; maximum {max})")]
    TooLarge { key: ObjectKey, size: u64, max: u64 },
    #[error("object {key} content hash mismatch (expected {expected}, got {actual})")]
    HashMismatch {
        key: ObjectKey,
        expected: ContentHash,
        actual: ContentHash,
    },
    #[error("object {key} envelope is corrupt: {reason}")]
    Corrupt { key: ObjectKey, reason: String },
    #[error("immutable object {key} already contains different bytes")]
    Conflict { key: ObjectKey },
    #[error("invalid object-store address: {0}")]
    InvalidAddress(String),
    #[error("object-store {operation} failed for {key}: {source}")]
    Backend {
        operation: &'static str,
        key: ObjectKey,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("object-store blocking adapter failed: {0}")]
    Runtime(String),
}

impl StoreError {
    fn backend(
        operation: &'static str,
        key: ObjectKey,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Backend {
            operation,
            key,
            source: Box::new(source),
        }
    }
}

/// Outcome of an immutable create operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PutOutcome {
    Stored,
    AlreadyPresent,
}

#[derive(Clone, Debug)]
enum Backend {
    Filesystem {
        root: PathBuf,
    },
    #[cfg(feature = "cloud")]
    Provider {
        inner: Arc<dyn object_store::ObjectStore>,
        prefix: String,
    },
}

/// Canonical asynchronous object-storage policy facade.
#[derive(Clone, Debug)]
pub struct ObjectStore {
    backend: Backend,
}

impl ObjectStore {
    /// Open a store rooted on a filesystem. Directories are created lazily.
    pub fn filesystem(root: impl Into<PathBuf>) -> Self {
        Self {
            backend: Backend::Filesystem { root: root.into() },
        }
    }

    /// Wrap a production `object_store` provider behind canonical addressing and
    /// validation. An empty prefix is valid.
    #[cfg(feature = "cloud")]
    pub fn provider(
        inner: Arc<dyn object_store::ObjectStore>,
        prefix: impl AsRef<str>,
    ) -> Result<Self, StoreError> {
        let prefix = prefix.as_ref().trim_matches('/');
        let prefix = if prefix.is_empty() {
            String::new()
        } else {
            object_store::path::Path::parse(prefix)
                .map_err(|error| StoreError::InvalidAddress(error.to_string()))?
                .to_string()
        };
        Ok(Self {
            backend: Backend::Provider { inner, prefix },
        })
    }

    /// Local `object_store` provider adapter for development and conformance
    /// tests. The directory must already exist.
    #[cfg(feature = "cloud")]
    pub fn local_provider(
        root: impl AsRef<Path>,
        prefix: impl AsRef<str>,
    ) -> Result<Self, StoreError> {
        let inner = object_store::local::LocalFileSystem::new_with_prefix(root)
            .map_err(|error| StoreError::InvalidAddress(error.to_string()))?;
        Self::provider(Arc::new(inner), prefix)
    }

    /// Fetch and validate an object. Absence is uniformly `Ok(None)`.
    pub async fn get(&self, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
        let Some(stored) = self.read_raw(key).await? else {
            return Ok(None);
        };
        decode_object(key, &stored).map(Some)
    }

    /// Atomically publish immutable bytes. Repeating the same write is
    /// idempotent; different bytes under the same typed key are rejected.
    pub async fn put(
        &self,
        key: ObjectKey,
        payload: impl Into<Vec<u8>>,
    ) -> Result<PutOutcome, StoreError> {
        let payload = payload.into();
        let stored = encode_object(key, &payload)?;
        match self.create_raw(key, stored).await? {
            RawCreate::Created => Ok(PutOutcome::Stored),
            RawCreate::AlreadyExists => match self.get(key).await? {
                Some(existing) if existing == payload => Ok(PutOutcome::AlreadyPresent),
                Some(_) => Err(StoreError::Conflict { key }),
                None => Err(StoreError::Corrupt {
                    key,
                    reason: "object disappeared after create conflict".into(),
                }),
            },
        }
    }

    /// Check physical presence without downloading the payload.
    pub async fn contains(&self, key: ObjectKey) -> Result<bool, StoreError> {
        self.contains_raw(key).await
    }

    /// Cross into synchronous code through the one explicit blocking adapter.
    pub fn blocking(&self) -> BlockingObjectStore {
        BlockingObjectStore::new(self.clone())
    }

    async fn read_raw(&self, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
        match &self.backend {
            Backend::Filesystem { root } => {
                let path = root.join(key.relative_path());
                tokio::task::spawn_blocking(move || read_file_capped(&path, key))
                    .await
                    .map_err(|error| StoreError::Runtime(error.to_string()))?
            }
            #[cfg(feature = "cloud")]
            Backend::Provider { inner, prefix } => {
                read_provider_capped(inner, provider_path(prefix, key), key).await
            }
        }
    }

    async fn create_raw(&self, key: ObjectKey, stored: Vec<u8>) -> Result<RawCreate, StoreError> {
        match &self.backend {
            Backend::Filesystem { root } => {
                let path = root.join(key.relative_path());
                tokio::task::spawn_blocking(move || create_file_atomic(&path, key, &stored))
                    .await
                    .map_err(|error| StoreError::Runtime(error.to_string()))?
            }
            #[cfg(feature = "cloud")]
            Backend::Provider { inner, prefix } => {
                let path = provider_path(prefix, key);
                let payload = object_store::PutPayload::from_bytes(bytes::Bytes::from(stored));
                let options = object_store::PutOptions::from(object_store::PutMode::Create);
                match inner.put_opts(&path, payload, options).await {
                    Ok(_) => Ok(RawCreate::Created),
                    Err(object_store::Error::AlreadyExists { .. }) => Ok(RawCreate::AlreadyExists),
                    Err(error) => Err(StoreError::backend("put", key, error)),
                }
            }
        }
    }

    async fn contains_raw(&self, key: ObjectKey) -> Result<bool, StoreError> {
        match &self.backend {
            Backend::Filesystem { root } => {
                let path = root.join(key.relative_path());
                tokio::task::spawn_blocking(move || match std::fs::metadata(&path) {
                    Ok(metadata) if metadata.is_file() => Ok(true),
                    Ok(_) => Err(StoreError::Corrupt {
                        key,
                        reason: "address is not a regular file".into(),
                    }),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
                    Err(error) => Err(StoreError::backend("head", key, error)),
                })
                .await
                .map_err(|error| StoreError::Runtime(error.to_string()))?
            }
            #[cfg(feature = "cloud")]
            Backend::Provider { inner, prefix } => {
                match inner.head(&provider_path(prefix, key)).await {
                    Ok(_) => Ok(true),
                    Err(object_store::Error::NotFound { .. }) => Ok(false),
                    Err(error) => Err(StoreError::backend("head", key, error)),
                }
            }
        }
    }
}

/// The sole synchronous view of [`ObjectStore`]. It centralizes runtime
/// isolation for cache and chunk callers that cannot be async.
#[derive(Clone, Debug)]
pub struct BlockingObjectStore {
    inner: ObjectStore,
}

impl BlockingObjectStore {
    pub fn new(inner: ObjectStore) -> Self {
        Self { inner }
    }

    pub fn filesystem(root: impl Into<PathBuf>) -> Self {
        Self::new(ObjectStore::filesystem(root))
    }

    pub fn get(&self, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
        let store = self.inner.clone();
        block_on_isolated(async move { store.get(key).await })
    }

    pub fn put(&self, key: ObjectKey, payload: &[u8]) -> Result<PutOutcome, StoreError> {
        let store = self.inner.clone();
        let payload = payload.to_vec();
        block_on_isolated(async move { store.put(key, payload).await })
    }

    pub fn contains(&self, key: ObjectKey) -> Result<bool, StoreError> {
        let store = self.inner.clone();
        block_on_isolated(async move { store.contains(key).await })
    }

    pub fn asynchronous(&self) -> &ObjectStore {
        &self.inner
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RawCreate {
    Created,
    AlreadyExists,
}

fn encode_object(key: ObjectKey, payload: &[u8]) -> Result<Vec<u8>, StoreError> {
    if payload.len() as u64 > MAX_OBJECT_SIZE {
        return Err(StoreError::TooLarge {
            key,
            size: payload.len() as u64,
            max: MAX_OBJECT_SIZE,
        });
    }
    let payload_hash = ContentHash::of_bytes(payload);
    if key.namespace().payload_is_content_addressed() && payload_hash != key.digest() {
        return Err(StoreError::HashMismatch {
            key,
            expected: key.digest(),
            actual: payload_hash,
        });
    }

    let mut stored = Vec::with_capacity(OBJECT_HEADER_LEN + payload.len());
    stored.extend_from_slice(OBJECT_MAGIC);
    stored.extend_from_slice(&OBJECT_FORMAT_VERSION.to_le_bytes());
    stored.push(key.namespace().tag());
    stored.extend_from_slice(&key.digest().0);
    stored.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    stored.extend_from_slice(&payload_hash.0);
    stored.extend_from_slice(payload);
    Ok(stored)
}

fn decode_object(key: ObjectKey, stored: &[u8]) -> Result<Vec<u8>, StoreError> {
    if stored.len() < OBJECT_HEADER_LEN {
        return Err(StoreError::Corrupt {
            key,
            reason: format!("truncated header ({} bytes)", stored.len()),
        });
    }
    if &stored[..8] != OBJECT_MAGIC {
        return Err(StoreError::Corrupt {
            key,
            reason: "bad magic".into(),
        });
    }
    let version = u16::from_le_bytes(stored[8..10].try_into().expect("fixed version slice"));
    if version != OBJECT_FORMAT_VERSION {
        return Err(StoreError::Corrupt {
            key,
            reason: format!("unsupported version {version}"),
        });
    }
    let namespace = ObjectNamespace::from_tag(stored[10]).ok_or_else(|| StoreError::Corrupt {
        key,
        reason: format!("unknown namespace tag {}", stored[10]),
    })?;
    let digest = ContentHash(stored[11..43].try_into().expect("fixed digest slice"));
    let envelope_key = ObjectKey::from_parts(namespace, digest);
    if envelope_key != key {
        return Err(StoreError::Corrupt {
            key,
            reason: format!("envelope is bound to {envelope_key}"),
        });
    }
    let declared_len = u64::from_le_bytes(stored[43..51].try_into().expect("fixed length slice"));
    if declared_len > MAX_OBJECT_SIZE {
        return Err(StoreError::TooLarge {
            key,
            size: declared_len,
            max: MAX_OBJECT_SIZE,
        });
    }
    let payload = &stored[OBJECT_HEADER_LEN..];
    if payload.len() as u64 != declared_len {
        return Err(StoreError::Corrupt {
            key,
            reason: format!(
                "declared payload length {declared_len} != stored {}",
                payload.len()
            ),
        });
    }
    let expected_hash = ContentHash(stored[51..83].try_into().expect("fixed hash slice"));
    let actual_hash = ContentHash::of_bytes(payload);
    if actual_hash != expected_hash {
        return Err(StoreError::Corrupt {
            key,
            reason: format!("payload checksum {actual_hash} != envelope {expected_hash}"),
        });
    }
    if key.namespace().payload_is_content_addressed() && actual_hash != key.digest() {
        return Err(StoreError::HashMismatch {
            key,
            expected: key.digest(),
            actual: actual_hash,
        });
    }
    Ok(payload.to_vec())
}

fn read_file_capped(path: &Path, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
    use std::io::Read;

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(StoreError::backend("get", key, error)),
    };
    let size = file
        .metadata()
        .map_err(|error| StoreError::backend("head", key, error))?
        .len();
    if size > MAX_STORED_SIZE {
        return Err(StoreError::TooLarge {
            key,
            size: size.saturating_sub(OBJECT_HEADER_LEN as u64),
            max: MAX_OBJECT_SIZE,
        });
    }
    let mut stored = Vec::with_capacity(size.min(8 * 1024 * 1024) as usize);
    file.take(MAX_STORED_SIZE + 1)
        .read_to_end(&mut stored)
        .map_err(|error| StoreError::backend("get", key, error))?;
    if stored.len() as u64 > MAX_STORED_SIZE {
        return Err(StoreError::TooLarge {
            key,
            size: stored.len() as u64 - OBJECT_HEADER_LEN as u64,
            max: MAX_OBJECT_SIZE,
        });
    }
    Ok(Some(stored))
}

fn create_file_atomic(path: &Path, key: ObjectKey, stored: &[u8]) -> Result<RawCreate, StoreError> {
    use std::io::Write;

    let parent = path
        .parent()
        .ok_or_else(|| StoreError::InvalidAddress(path.display().to_string()))?;
    std::fs::create_dir_all(parent).map_err(|error| StoreError::backend("mkdir", key, error))?;
    let tmp = parent.join(format!(".object-{}.tmp", uuid::Uuid::new_v4()));
    let write_result = (|| -> Result<RawCreate, StoreError> {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)
            .map_err(|error| StoreError::backend("create-temp", key, error))?;
        file.write_all(stored)
            .map_err(|error| StoreError::backend("write", key, error))?;
        file.sync_all()
            .map_err(|error| StoreError::backend("sync", key, error))?;
        match std::fs::hard_link(&tmp, path) {
            Ok(()) => Ok(RawCreate::Created),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                Ok(RawCreate::AlreadyExists)
            }
            Err(error) => Err(StoreError::backend("publish", key, error)),
        }
    })();
    let _ = std::fs::remove_file(&tmp);
    write_result
}

#[cfg(feature = "cloud")]
fn provider_path(prefix: &str, key: ObjectKey) -> object_store::path::Path {
    let relative = key.relative_path();
    if prefix.is_empty() {
        object_store::path::Path::from(relative)
    } else {
        object_store::path::Path::from(format!("{prefix}/{relative}"))
    }
}

#[cfg(feature = "cloud")]
async fn read_provider_capped(
    inner: &Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
    key: ObjectKey,
) -> Result<Option<Vec<u8>>, StoreError> {
    use futures::StreamExt;

    let meta = match inner.head(&path).await {
        Ok(meta) => meta,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(StoreError::backend("head", key, error)),
    };
    if meta.size as u64 > MAX_STORED_SIZE {
        return Err(StoreError::TooLarge {
            key,
            size: (meta.size as u64).saturating_sub(OBJECT_HEADER_LEN as u64),
            max: MAX_OBJECT_SIZE,
        });
    }
    let result = match inner.get(&path).await {
        Ok(result) => result,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(StoreError::backend("get", key, error)),
    };
    let mut stream = result.into_stream();
    let mut stored = Vec::with_capacity((meta.size as u64).min(8 * 1024 * 1024) as usize);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| StoreError::backend("read", key, error))?;
        let next_len = stored
            .len()
            .checked_add(chunk.len())
            .ok_or(StoreError::TooLarge {
                key,
                size: u64::MAX,
                max: MAX_OBJECT_SIZE,
            })?;
        if next_len as u64 > MAX_STORED_SIZE {
            return Err(StoreError::TooLarge {
                key,
                size: next_len as u64 - OBJECT_HEADER_LEN as u64,
                max: MAX_OBJECT_SIZE,
            });
        }
        stored.extend_from_slice(&chunk);
    }
    Ok(Some(stored))
}

fn block_on_isolated<T: Send + 'static>(
    future: impl std::future::Future<Output = Result<T, StoreError>> + Send + 'static,
) -> Result<T, StoreError> {
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| StoreError::Runtime(error.to_string()))?
            .block_on(future)
    })
    .join()
    .map_err(|_| StoreError::Runtime("blocking worker thread panicked".into()))?
}

/// A content-addressed blob backend. Implementations must be cheap to share
/// (`Arc`ed) across the executor's tasks.
pub trait BlobStore: Send + Sync + std::fmt::Debug {
    /// Fetch the bytes stored under `key`, or `None` if absent. An I/O error
    /// (network blip, permission) is surfaced — the caller treats it as a miss
    /// but logs it, so a broken remote degrades to "no shared cache" rather
    /// than a wrong answer.
    fn get(&self, key: ContentHash) -> std::io::Result<Option<Vec<u8>>>;

    /// Store `bytes` under `key`. Idempotent — content-addressed, so a
    /// re-`put` of the same key is a no-op-equivalent overwrite of identical
    /// bytes.
    fn put(&self, key: ContentHash, bytes: &[u8]) -> std::io::Result<()>;

    /// Presence check without transferring the bytes.
    fn head(&self, key: ContentHash) -> std::io::Result<bool>;
}

/// A shared-filesystem blob store: one file per key at
/// `<root>/<hex-key>`. This is the plain NFS/Lustre backend AND the
/// Kubernetes RWX-PVC backend for a pod-shared cache. Writes are atomic
/// (write-temp + rename), so concurrent writers never observe a torn blob.
#[derive(Debug, Clone)]
pub struct FsBlobStore {
    root: PathBuf,
}

impl FsBlobStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn path_for(&self, key: ContentHash) -> PathBuf {
        self.root.join(key.to_hex())
    }
}

impl BlobStore for FsBlobStore {
    fn get(&self, key: ContentHash) -> std::io::Result<Option<Vec<u8>>> {
        match std::fs::read(self.path_for(key)) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn put(&self, key: ContentHash, bytes: &[u8]) -> std::io::Result<()> {
        crate::framework::cache::write_atomic(&self.path_for(key), bytes)
    }

    fn head(&self, key: ContentHash) -> std::io::Result<bool> {
        // `is_file` (not `exists`): a key resolves to a regular file, never a
        // directory. Presence is advisory — a concurrent delete can race a
        // following `get`, which then simply misses (never a wrong answer).
        Ok(self.path_for(key).is_file())
    }
}

/// Convenience: `Arc` a store for the cache's remote tier.
pub fn shared(store: impl BlobStore + 'static) -> Arc<dyn BlobStore> {
    Arc::new(store)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fs_blob_store_round_trips_and_head() {
        let td = tempfile::tempdir().unwrap();
        let store = FsBlobStore::new(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"payload");
        assert!(!store.head(key).unwrap());
        assert_eq!(store.get(key).unwrap(), None);
        store.put(key, b"payload-bytes").unwrap();
        assert!(store.head(key).unwrap());
        assert_eq!(
            store.get(key).unwrap().as_deref(),
            Some(&b"payload-bytes"[..])
        );
    }

    #[test]
    fn fs_blob_store_put_is_idempotent() {
        let td = tempfile::tempdir().unwrap();
        let store = FsBlobStore::new(td.path().to_path_buf());
        let key = ContentHash::of_bytes(b"k");
        store.put(key, b"v").unwrap();
        store.put(key, b"v").unwrap(); // no torn write, no error
        assert_eq!(store.get(key).unwrap().as_deref(), Some(&b"v"[..]));
    }

    #[test]
    fn missing_root_get_is_a_miss_not_an_error() {
        let store = FsBlobStore::new(PathBuf::from("/nonexistent-blob-root-xyz"));
        assert_eq!(store.get(ContentHash::of_bytes(b"x")).unwrap(), None);
        assert!(!store.head(ContentHash::of_bytes(b"x")).unwrap());
    }
}
