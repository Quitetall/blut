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

use async_trait::async_trait;
#[cfg(feature = "cloud")]
use object_store::ObjectStoreExt;

use crate::framework::artifact::{ArtifactContentId, ContentHash, InvocationKey};

/// Maximum payload accepted by every production object-store adapter.
pub const MAX_OBJECT_SIZE: u64 = 16 * 1024 * 1024 * 1024;

const OBJECT_MAGIC: &[u8; 8] = b"BLUTOS01";
const OBJECT_FORMAT_VERSION: u16 = 1;
const OBJECT_HEADER_LEN: usize = 8 + 2 + 1 + 32 + 8 + 32;
/// Largest legitimate ON-DISK object file: a full payload plus its header.
/// This — not `MAX_OBJECT_SIZE` — is the bound the metadata guard in `get`
/// compares a file's length against, so it is what a test must exceed to
/// exercise rejection BEFORE the read.
pub(crate) const MAX_STORED_SIZE: u64 = MAX_OBJECT_SIZE + OBJECT_HEADER_LEN as u64;

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
    Artifact(ArtifactContentId),
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
            ObjectNamespace::Artifact => Self::Artifact(ArtifactContentId::from_digest(digest)),
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
    #[error("object-store {adapter} adapter initialization failed: {source}")]
    Initialization {
        adapter: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("object-store {operation} failed for {key}: {source}")]
    Backend {
        operation: &'static str,
        key: ObjectKey,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("object-store blocking adapter failed: {0}")]
    Runtime(String),
    #[error("object-store operation {operation} is unsupported by this adapter")]
    Unsupported { operation: &'static str },
}

impl StoreError {
    #[cfg(feature = "cloud")]
    fn initialization(
        adapter: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Initialization {
            adapter,
            source: Box::new(source),
        }
    }

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

/// Physical adapter seam beneath canonical addressing and validation policy.
///
/// Implementations only move already-enveloped bytes. [`ObjectStore`] remains
/// responsible for namespaces, size limits, hashes, immutable-conflict checks,
/// and error semantics. `create_raw` returns `true` only when this call created
/// the key; `false` means an object already existed and must be verified by the
/// facade. Production callers normally use the filesystem or provider
/// constructors; this seam also permits deterministic fault-injection tests.
#[async_trait]
pub trait ObjectStoreAdapter: Send + Sync + std::fmt::Debug {
    async fn read_raw(&self, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError>;
    async fn create_raw(&self, key: ObjectKey, stored: Vec<u8>) -> Result<bool, StoreError>;
    async fn contains_raw(&self, key: ObjectKey) -> Result<bool, StoreError>;
    async fn delete_raw(&self, _key: ObjectKey) -> Result<bool, StoreError> {
        Err(StoreError::Unsupported {
            operation: "delete",
        })
    }
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
    Adapter {
        inner: Arc<dyn ObjectStoreAdapter>,
    },
}

/// Canonical asynchronous object-storage policy facade.
#[derive(Clone, Debug)]
pub struct ObjectStore {
    backend: Backend,
}

/// Shared ownership lets asynchronous adapters retain incoming bytes through a
/// conflict read-back without cloning the full payload. Provider uploads wrap
/// this owner in `Bytes`; filesystem writes borrow it inside `spawn_blocking`.
#[derive(Clone, Debug)]
struct SharedPayload(Arc<Vec<u8>>);

impl SharedPayload {
    fn new(payload: Vec<u8>) -> Self {
        Self(Arc::new(payload))
    }
}

impl AsRef<[u8]> for SharedPayload {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl ObjectStore {
    /// Open a store rooted on a filesystem. Directories are created lazily.
    pub fn filesystem(root: impl Into<PathBuf>) -> Self {
        Self {
            backend: Backend::Filesystem { root: root.into() },
        }
    }

    /// Wrap a physical adapter while retaining every canonical policy gate.
    pub fn adapter(inner: Arc<dyn ObjectStoreAdapter>) -> Self {
        Self {
            backend: Backend::Adapter { inner },
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

    /// Open a store from a URL, choosing the provider from the scheme.
    ///
    /// | scheme | provider |
    /// |---|---|
    /// | `s3://bucket/prefix` | S3, and any S3-compatible endpoint |
    /// | `file:///abs/path` | local filesystem provider |
    ///
    /// Credentials and endpoint come from the environment, the same variables
    /// the AWS CLI reads (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    /// `AWS_REGION`, `AWS_ENDPOINT`/`AWS_ENDPOINT_URL`, `AWS_ALLOW_HTTP`). They
    /// are deliberately NOT accepted as arguments: a credential passed on a
    /// command line lands in the shell history, the process table and any log
    /// that records argv.
    ///
    /// The returned store is a [`Backend::Provider`], so every canonical
    /// address rule, size ceiling and content check still applies. A provider
    /// is wrapped, never trusted.
    #[cfg(feature = "s3")]
    pub fn from_url(url: &str, prefix: impl AsRef<str>) -> Result<Self, StoreError> {
        let parsed = url::Url::parse(url)
            .map_err(|error| StoreError::InvalidAddress(format!("{url}: {error}")))?;
        match parsed.scheme() {
            "s3" => {
                let bucket = parsed.host_str().ok_or_else(|| {
                    StoreError::InvalidAddress(format!("{url}: no bucket in s3 URL"))
                })?;
                let inner = object_store::aws::AmazonS3Builder::from_env()
                    .with_bucket_name(bucket)
                    .build()
                    .map_err(|error| StoreError::initialization("s3", error))?;
                // A path inside the URL prefixes the caller's prefix, so
                // `s3://bucket/runs` + "cloud" addresses `runs/cloud/…`.
                let url_prefix = parsed.path().trim_matches('/');
                let prefix = match (url_prefix, prefix.as_ref().trim_matches('/')) {
                    ("", p) => p.to_string(),
                    (u, "") => u.to_string(),
                    (u, p) => format!("{u}/{p}"),
                };
                Self::provider(Arc::new(inner), prefix)
            }
            "file" => {
                let path = parsed.to_file_path().map_err(|()| {
                    StoreError::InvalidAddress(format!("{url}: not a valid file path"))
                })?;
                Self::local_provider(path, prefix)
            }
            other => Err(StoreError::InvalidAddress(format!(
                "unsupported object-store scheme {other:?} in {url}"
            ))),
        }
    }

    /// Local `object_store` provider adapter for development and conformance
    /// tests. The directory must already exist.
    #[cfg(feature = "cloud")]
    pub fn local_provider(
        root: impl AsRef<Path>,
        prefix: impl AsRef<str>,
    ) -> Result<Self, StoreError> {
        let inner = object_store::local::LocalFileSystem::new_with_prefix(root)
            .map_err(|error| StoreError::initialization("local-provider", error))?;
        Self::provider(Arc::new(inner), prefix)
    }

    /// Fetch and validate an object. Absence is uniformly `Ok(None)`.
    pub async fn get(&self, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
        let Some(stored) = self.read_raw(key).await? else {
            return Ok(None);
        };
        decode_object(key, stored).map(Some)
    }

    /// Atomically publish immutable bytes. Repeating the same write is
    /// idempotent; different bytes under the same typed key are rejected.
    pub async fn put(
        &self,
        key: ObjectKey,
        payload: impl Into<Vec<u8>>,
    ) -> Result<PutOutcome, StoreError> {
        let payload = SharedPayload::new(payload.into());
        let header = encode_header(key, payload.as_ref())?;
        match self.create_raw(key, header, payload.clone()).await? {
            RawCreate::Created => Ok(PutOutcome::Stored),
            RawCreate::AlreadyExists => {
                resolve_existing(key, payload.as_ref(), self.get(key).await?)
            }
        }
    }

    /// Check physical presence without downloading the payload.
    pub async fn contains(&self, key: ObjectKey) -> Result<bool, StoreError> {
        self.contains_raw(key).await
    }

    /// Remove one typed object. Cache maintenance uses this only after proving
    /// no retained invocation references the object.
    pub async fn remove(&self, key: ObjectKey) -> Result<bool, StoreError> {
        match &self.backend {
            Backend::Filesystem { root } => {
                let path = root.join(key.relative_path());
                tokio::task::spawn_blocking(move || remove_file(&path, key))
                    .await
                    .map_err(|error| StoreError::Runtime(error.to_string()))?
            }
            #[cfg(feature = "cloud")]
            Backend::Provider { inner, prefix } => {
                match inner.delete(&provider_path(prefix, key)).await {
                    Ok(()) => Ok(true),
                    Err(object_store::Error::NotFound { .. }) => Ok(false),
                    Err(error) => Err(StoreError::backend("delete", key, error)),
                }
            }
            Backend::Adapter { inner } => inner.delete_raw(key).await,
        }
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
            Backend::Adapter { inner } => {
                let stored = inner.read_raw(key).await?;
                if stored
                    .as_ref()
                    .is_some_and(|bytes| bytes.len() as u64 > MAX_STORED_SIZE)
                {
                    return Err(StoreError::TooLarge {
                        key,
                        size: stored
                            .as_ref()
                            .map_or(0, |bytes| bytes.len() as u64)
                            .saturating_sub(OBJECT_HEADER_LEN as u64),
                        max: MAX_OBJECT_SIZE,
                    });
                }
                Ok(stored)
            }
        }
    }

    async fn create_raw(
        &self,
        key: ObjectKey,
        header: [u8; OBJECT_HEADER_LEN],
        payload: SharedPayload,
    ) -> Result<RawCreate, StoreError> {
        match &self.backend {
            Backend::Filesystem { root } => {
                let path = root.join(key.relative_path());
                tokio::task::spawn_blocking(move || {
                    create_file_atomic(&path, key, &header, payload.as_ref())
                })
                .await
                .map_err(|error| StoreError::Runtime(error.to_string()))?
            }
            #[cfg(feature = "cloud")]
            Backend::Provider { inner, prefix } => {
                let path = provider_path(prefix, key);
                let payload = object_store::PutPayload::from_iter([
                    bytes::Bytes::copy_from_slice(&header),
                    bytes::Bytes::from_owner(payload),
                ]);
                let options = object_store::PutOptions::from(object_store::PutMode::Create);
                match inner.put_opts(&path, payload, options).await {
                    Ok(_) => Ok(RawCreate::Created),
                    Err(object_store::Error::AlreadyExists { .. }) => Ok(RawCreate::AlreadyExists),
                    Err(error) => Err(StoreError::backend("put", key, error)),
                }
            }
            Backend::Adapter { inner } => inner
                .create_raw(key, join_envelope(header, payload.as_ref().to_vec()))
                .await
                .map(|created| {
                    if created {
                        RawCreate::Created
                    } else {
                        RawCreate::AlreadyExists
                    }
                }),
        }
    }

    async fn contains_raw(&self, key: ObjectKey) -> Result<bool, StoreError> {
        match &self.backend {
            Backend::Filesystem { root } => {
                let path = root.join(key.relative_path());
                tokio::task::spawn_blocking(move || contains_file(&path, key))
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
            Backend::Adapter { inner } => inner.contains_raw(key).await,
        }
    }
}

/// The sole synchronous view of [`ObjectStore`]. It centralizes runtime
/// isolation for cache and chunk callers that cannot be async.
#[derive(Clone, Debug)]
pub struct BlockingObjectStore {
    inner: ObjectStore,
}

/// Physical usage metadata exposed only for typed cache maintenance.
#[derive(Clone, Copy, Debug)]
pub struct StoredObjectInfo {
    pub key: ObjectKey,
    pub stored_size: u64,
    pub accessed: std::time::SystemTime,
}

impl BlockingObjectStore {
    pub fn new(inner: ObjectStore) -> Self {
        Self { inner }
    }

    pub fn filesystem(root: impl Into<PathBuf>) -> Self {
        Self::new(ObjectStore::filesystem(root))
    }

    pub fn get(&self, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
        match &self.inner.backend {
            Backend::Filesystem { root } => {
                let Some(stored) = read_file_capped(&root.join(key.relative_path()), key)? else {
                    return Ok(None);
                };
                decode_object(key, stored).map(Some)
            }
            #[cfg(feature = "cloud")]
            Backend::Provider { .. } => {
                let store = self.inner.clone();
                block_on_isolated(async move { store.get(key).await })
            }
            Backend::Adapter { .. } => {
                let store = self.inner.clone();
                block_on_isolated(async move { store.get(key).await })
            }
        }
    }

    pub fn put(&self, key: ObjectKey, payload: &[u8]) -> Result<PutOutcome, StoreError> {
        match &self.inner.backend {
            Backend::Filesystem { root } => {
                let header = encode_header(key, payload)?;
                match create_file_atomic(&root.join(key.relative_path()), key, &header, payload)? {
                    RawCreate::Created => Ok(PutOutcome::Stored),
                    RawCreate::AlreadyExists => resolve_existing(key, payload, self.get(key)?),
                }
            }
            #[cfg(feature = "cloud")]
            Backend::Provider { .. } => {
                let store = self.inner.clone();
                let payload = payload.to_vec();
                block_on_isolated(async move { store.put(key, payload).await })
            }
            Backend::Adapter { .. } => {
                let store = self.inner.clone();
                let payload = payload.to_vec();
                block_on_isolated(async move { store.put(key, payload).await })
            }
        }
    }

    pub fn contains(&self, key: ObjectKey) -> Result<bool, StoreError> {
        match &self.inner.backend {
            Backend::Filesystem { root } => contains_file(&root.join(key.relative_path()), key),
            #[cfg(feature = "cloud")]
            Backend::Provider { .. } => {
                let store = self.inner.clone();
                block_on_isolated(async move { store.contains(key).await })
            }
            Backend::Adapter { .. } => {
                let store = self.inner.clone();
                block_on_isolated(async move { store.contains(key).await })
            }
        }
    }

    pub fn remove(&self, key: ObjectKey) -> Result<bool, StoreError> {
        match &self.inner.backend {
            Backend::Filesystem { root } => remove_file(&root.join(key.relative_path()), key),
            #[cfg(feature = "cloud")]
            Backend::Provider { .. } => {
                let store = self.inner.clone();
                block_on_isolated(async move { store.remove(key).await })
            }
            Backend::Adapter { .. } => {
                let store = self.inner.clone();
                block_on_isolated(async move { store.remove(key).await })
            }
        }
    }

    /// List canonical files in one namespace. Provider listing is deliberately
    /// outside synchronous cache-prune scope.
    pub fn list_namespace(
        &self,
        namespace: ObjectNamespace,
    ) -> Result<Vec<StoredObjectInfo>, StoreError> {
        let Backend::Filesystem { root } = &self.inner.backend else {
            return Err(StoreError::Unsupported { operation: "list" });
        };
        let dir = root
            .join(format!("v{OBJECT_FORMAT_VERSION}"))
            .join(namespace.path_segment());
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                let placeholder = ObjectKey::from_parts(namespace, ContentHash([0; 32]));
                return Err(StoreError::backend("list", placeholder, error));
            }
        };
        let mut objects = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                let placeholder = ObjectKey::from_parts(namespace, ContentHash([0; 32]));
                StoreError::backend("list", placeholder, error)
            })?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(".object-") {
                continue;
            }
            let digest = ContentHash::from_hex(&name)
                .map_err(|_| StoreError::InvalidAddress(entry.path().display().to_string()))?;
            let metadata = entry.metadata().map_err(|error| {
                StoreError::backend("head", ObjectKey::from_parts(namespace, digest), error)
            })?;
            if !metadata.is_file() {
                return Err(StoreError::Corrupt {
                    key: ObjectKey::from_parts(namespace, digest),
                    reason: "canonical address is not a regular file".into(),
                });
            }
            objects.push(StoredObjectInfo {
                key: ObjectKey::from_parts(namespace, digest),
                stored_size: metadata.len(),
                accessed: metadata
                    .accessed()
                    .or_else(|_| metadata.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
            });
        }
        Ok(objects)
    }

    /// Exact physical address for proofs and diagnostics. Non-filesystem stores
    /// intentionally expose no host path.
    pub fn filesystem_path(&self, key: ObjectKey) -> Option<PathBuf> {
        match &self.inner.backend {
            Backend::Filesystem { root } => Some(root.join(key.relative_path())),
            #[cfg(feature = "cloud")]
            Backend::Provider { .. } => None,
            Backend::Adapter { .. } => None,
        }
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

fn resolve_existing(
    key: ObjectKey,
    payload: &[u8],
    existing: Option<Vec<u8>>,
) -> Result<PutOutcome, StoreError> {
    match existing {
        Some(existing) if existing == payload => Ok(PutOutcome::AlreadyPresent),
        Some(_) => Err(StoreError::Conflict { key }),
        None => Err(StoreError::Corrupt {
            key,
            reason: "object disappeared after create conflict".into(),
        }),
    }
}

fn encode_header(key: ObjectKey, payload: &[u8]) -> Result<[u8; OBJECT_HEADER_LEN], StoreError> {
    encode_header_with_limit(key, payload, MAX_OBJECT_SIZE)
}

fn encode_header_with_limit(
    key: ObjectKey,
    payload: &[u8],
    max_size: u64,
) -> Result<[u8; OBJECT_HEADER_LEN], StoreError> {
    if payload.len() as u64 > max_size {
        return Err(StoreError::TooLarge {
            key,
            size: payload.len() as u64,
            max: max_size,
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

    let mut header = [0_u8; OBJECT_HEADER_LEN];
    header[..8].copy_from_slice(OBJECT_MAGIC);
    header[8..10].copy_from_slice(&OBJECT_FORMAT_VERSION.to_le_bytes());
    header[10] = key.namespace().tag();
    header[11..43].copy_from_slice(&key.digest().0);
    header[43..51].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    header[51..83].copy_from_slice(&payload_hash.0);
    Ok(header)
}

fn join_envelope(header: [u8; OBJECT_HEADER_LEN], mut payload: Vec<u8>) -> Vec<u8> {
    payload.reserve_exact(OBJECT_HEADER_LEN);
    let payload_len = payload.len();
    payload.resize(payload_len + OBJECT_HEADER_LEN, 0);
    payload.copy_within(..payload_len, OBJECT_HEADER_LEN);
    payload[..OBJECT_HEADER_LEN].copy_from_slice(&header);
    payload
}

fn decode_object(key: ObjectKey, mut stored: Vec<u8>) -> Result<Vec<u8>, StoreError> {
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
    stored.copy_within(OBJECT_HEADER_LEN.., 0);
    stored.truncate(declared_len as usize);
    Ok(stored)
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

fn contains_file(path: &Path, key: ObjectKey) -> Result<bool, StoreError> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(true),
        Ok(_) => Err(StoreError::Corrupt {
            key,
            reason: "address is not a regular file".into(),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(StoreError::backend("head", key, error)),
    }
}

fn remove_file(path: &Path, key: ObjectKey) -> Result<bool, StoreError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(StoreError::backend("delete", key, error)),
    }
}

fn create_file_atomic(
    path: &Path,
    key: ObjectKey,
    header: &[u8; OBJECT_HEADER_LEN],
    payload: &[u8],
) -> Result<RawCreate, StoreError> {
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
        file.write_all(header)
            .and_then(|()| file.write_all(payload))
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
    if let Err(error) = std::fs::remove_file(&tmp)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            path = %tmp.display(),
            object = %key,
            "object-store temporary-file cleanup failed: {error}"
        );
    }
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
    if meta.size > MAX_STORED_SIZE {
        return Err(StoreError::TooLarge {
            key,
            size: meta.size.saturating_sub(OBJECT_HEADER_LEN as u64),
            max: MAX_OBJECT_SIZE,
        });
    }
    let result = match inner.get(&path).await {
        Ok(result) => result,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(StoreError::backend("get", key, error)),
    };
    let mut stream = result.into_stream();
    let mut stored = Vec::with_capacity(meta.size.min(8 * 1024 * 1024) as usize);
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

#[cfg(test)]
mod tests {

    #[cfg(feature = "s3")]
    mod from_url_tests {
        use super::super::{ObjectStore, StoreError};

        #[test]
        fn an_unsupported_scheme_is_refused_by_name() {
            let err = ObjectStore::from_url("gs://bucket/x", "cloud").unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("gs"), "error should name the scheme: {msg}");
        }

        #[test]
        fn an_s3_url_without_a_bucket_is_refused() {
            let err = ObjectStore::from_url("s3:///just-a-path", "cloud").unwrap_err();
            assert!(matches!(err, StoreError::InvalidAddress(_)));
        }

        #[test]
        fn a_malformed_url_is_refused_rather_than_treated_as_a_path() {
            // No scheme at all: from_url is only reached for URLs, and this must
            // not silently become a relative directory.
            let err = ObjectStore::from_url("not a url", "cloud").unwrap_err();
            assert!(matches!(err, StoreError::InvalidAddress(_)));
        }

        #[test]
        fn a_file_url_opens_a_local_provider() {
            let dir = tempfile::tempdir().expect("tempdir");
            let url = format!("file://{}", dir.path().display());
            ObjectStore::from_url(&url, "cloud").expect("file:// opens a local provider");
        }
    }
    use super::*;

    #[test]
    fn payload_limit_is_inclusive_and_reports_exact_oversize_fields() {
        let key = ObjectKey::CacheInvocation(InvocationKey::from_digest(ContentHash::of_bytes(
            b"bounded-write",
        )));
        assert!(encode_header_with_limit(key, b"four", 4).is_ok());

        let error = encode_header_with_limit(key, b"five!", 4).unwrap_err();
        assert!(matches!(
            error,
            StoreError::TooLarge {
                key: actual_key,
                size: 5,
                max: 4,
            } if actual_key == key
        ));
    }

    #[test]
    fn decoding_reuses_envelope_allocation() {
        let payload = b"decode-without-full-payload-copy".to_vec();
        let key = ObjectKey::DispatchBundle(ContentHash::of_bytes(&payload));
        let stored = join_envelope(encode_header(key, &payload).unwrap(), payload.clone());
        let allocation = stored.as_ptr();

        let decoded = decode_object(key, stored).unwrap();
        assert_eq!(decoded, payload);
        assert_eq!(decoded.as_ptr(), allocation);
    }
}
