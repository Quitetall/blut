//! Content-addressed blob transport for cloud dispatch — the object-store analog
//! of `crate::p2p::transport::{send_blob, recv_blob}`.
//!
//! A "blob" is an artifact-bundle pack (the `Vec<u8>` from `crate::p2p::bundle::bundle`).
//! It is keyed by its `ContentHash`, so:
//!   - puts are **idempotent** (re-uploading the same bundle is a no-op),
//!   - a worker can **skip a download** it already holds,
//!   - the address is verified independently on the receiving side by the bundle's
//!     own four fail-closed gates — the store is dumb bytes, not a trust anchor.
//!
//! Unlike QUIC's chunked uni-streams (a consequence of QUIC's per-message cap),
//! an object store handles arbitrarily large objects natively, so this transport
//! is a plain put/get. The `MAX_BLOB_SIZE` ceiling is kept as a download guard so
//! a corrupt/hostile object can't OOM the worker.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use super::CloudError;
use crate::framework::artifact::ContentHash;

/// Maximum blob accepted on download — mirrors the p2p transport ceiling.
pub const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB

/// Content-addressed blob store. Implementations move opaque bundle packs to/from
/// a backing store; verification of the bytes is the bundle layer's job, not theirs.
#[async_trait]
pub trait BlobStore: Send + Sync {
    /// Upload `pack` under `hash`. Takes `Bytes` by value so a multi-GiB bundle is
    /// moved, not copied. **Best-effort skip** if the object already exists: two
    /// concurrent puts of the same hash may both upload — benign, since the bytes
    /// are content-addressed (identical), only a little wasted egress.
    async fn put_blob(&self, hash: &ContentHash, pack: Bytes) -> Result<(), CloudError>;
    /// Download the blob stored under `hash`. A cheap `head()` rejects objects over
    /// `MAX_BLOB_SIZE` before the body is read.
    ///
    /// Residual TOCTOU: the object could change between `head` and `get`. For
    /// content-addressed immutable keys on a store you control this is benign — a
    /// swapped object fails the bundle's downstream content-hash gates, the real
    /// integrity guarantee. Tightening to a conditional `if_match`/streaming-capped
    /// read is deferred to T3.2 (untrusted multi-tenant providers).
    async fn get_blob(&self, hash: &ContentHash) -> Result<Vec<u8>, CloudError>;
    /// Whether a blob exists under `hash` (lets a producer skip an upload and a
    /// worker skip a download).
    async fn has_blob(&self, hash: &ContentHash) -> Result<bool, CloudError>;
}

/// `object_store`-backed [`BlobStore`]: one trait over the local filesystem and
/// any S3-compatible store (AWS S3 / Cloudflare R2 / MinIO) with the `aws` feature.
/// The provider is chosen at construction, so the cloud queue is provider-agnostic
/// and dev-testable on the local filesystem.
pub struct ObjStore {
    inner: Arc<dyn object_store::ObjectStore>,
    /// Key prefix under which blobs live, e.g. `"blobs"` → `blobs/<hash-hex>`.
    prefix: String,
}

impl ObjStore {
    /// Wrap any `object_store::ObjectStore` (S3, R2, …) with a key prefix.
    pub fn new(inner: Arc<dyn object_store::ObjectStore>, prefix: impl Into<String>) -> Self {
        Self { inner, prefix: prefix.into() }
    }

    /// Local-filesystem backend for dev / loopback tests (no cloud account). The
    /// `root` directory must already exist (`object_store` requires it).
    pub fn local(root: impl AsRef<std::path::Path>) -> Result<Self, CloudError> {
        let fs = object_store::local::LocalFileSystem::new_with_prefix(root)
            .map_err(|e| CloudError::Store(format!("local fs init: {e}")))?;
        Ok(Self::new(Arc::new(fs), "blobs"))
    }

    fn path_for(&self, hash: &ContentHash) -> object_store::path::Path {
        let hex = hash.to_hex();
        let prefix = self.prefix.trim_matches('/');
        // Guard an empty/`/`-only prefix → don't emit a leading-slash key.
        if prefix.is_empty() {
            object_store::path::Path::from(hex)
        } else {
            object_store::path::Path::from(format!("{prefix}/{hex}"))
        }
    }
}

#[async_trait]
impl BlobStore for ObjStore {
    async fn put_blob(&self, hash: &ContentHash, pack: Bytes) -> Result<(), CloudError> {
        let path = self.path_for(hash);
        // Best-effort skip: avoid re-PUTting a large EEG/checkpoint bundle that's
        // already present. Content-addressed, so a racing duplicate put is harmless.
        if self.inner.head(&path).await.is_ok() {
            return Ok(());
        }
        // Bytes → PutPayload is zero-copy (no whole-blob copy for a 16 GiB bundle).
        let payload = object_store::PutPayload::from_bytes(pack);
        self.inner
            .put(&path, payload)
            .await
            .map_err(|e| CloudError::Store(format!("put {}: {e}", hash.to_hex())))?;
        Ok(())
    }

    async fn get_blob(&self, hash: &ContentHash) -> Result<Vec<u8>, CloudError> {
        let path = self.path_for(hash);
        // Bound BEFORE reading the body (head is cheap) so an oversize object is
        // rejected without buffering it.
        let meta = self
            .inner
            .head(&path)
            .await
            .map_err(|e| CloudError::Store(format!("head {}: {e}", hash.to_hex())))?;
        if meta.size as u64 > MAX_BLOB_SIZE {
            return Err(CloudError::Store(format!(
                "blob {} exceeds MAX_BLOB_SIZE ({} > {MAX_BLOB_SIZE})",
                hash.to_hex(),
                meta.size
            )));
        }
        let res = self
            .inner
            .get(&path)
            .await
            .map_err(|e| CloudError::Store(format!("get {}: {e}", hash.to_hex())))?;
        let bytes = res
            .bytes()
            .await
            .map_err(|e| CloudError::Store(format!("read {}: {e}", hash.to_hex())))?;
        Ok(bytes.to_vec())
    }

    async fn has_blob(&self, hash: &ContentHash) -> Result<bool, CloudError> {
        match self.inner.head(&self.path_for(hash)).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(CloudError::Store(format!("head {}: {e}", hash.to_hex()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn blob_round_trips_on_local_fs() {
        let dir = tempfile::tempdir().unwrap();
        let store = ObjStore::local(dir.path()).unwrap();
        let pack = b"hello cloud bundle".to_vec();
        let hash = ContentHash::of_bytes(&pack);

        assert!(!store.has_blob(&hash).await.unwrap(), "absent before put");
        store.put_blob(&hash, Bytes::from(pack.clone())).await.unwrap();
        assert!(store.has_blob(&hash).await.unwrap(), "present after put");
        assert_eq!(store.get_blob(&hash).await.unwrap(), pack, "round-trips");

        // Idempotent re-put of identical content-addressed bytes.
        store.put_blob(&hash, Bytes::from(pack.clone())).await.unwrap();
        assert_eq!(store.get_blob(&hash).await.unwrap(), pack);
    }

    #[tokio::test]
    async fn missing_blob_get_errors_and_has_is_false() {
        let dir = tempfile::tempdir().unwrap();
        let store = ObjStore::local(dir.path()).unwrap();
        let hash = ContentHash::of_bytes(b"nope");
        assert!(!store.has_blob(&hash).await.unwrap());
        assert!(store.get_blob(&hash).await.is_err());
    }
}
