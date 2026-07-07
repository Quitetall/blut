// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `BlobStore` — a content-addressed blob backend for the cache's remote tier
//! (ADR 0067 Tier 4 / ADR 0079 data plane).
//!
//! The engine's cache stores each stage output under its content hash. A
//! `BlobStore` is a pluggable backend for those bytes that outlives a single
//! job dir: a shared filesystem (`FsBlobStore` — also the Kubernetes RWX-PVC
//! backend), or an object store (S3/R2/GCS/MinIO via the `s3` feature). Keys
//! are `ContentHash`es, so writes are idempotent and multi-writer-safe by
//! construction (the same content always hashes to the same key).
//!
//! The trait is deliberately SYNC (the cache path is sync); an async backend
//! bridges internally.

use std::path::PathBuf;
use std::sync::Arc;

use crate::framework::artifact::ContentHash;

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

/// An S3-compatible object store (S3 / R2 / GCS / MinIO) behind the `s3`
/// feature. A CLIENT, not a server — honors the ADR 0034 no-HTTP-server
/// charter. Bytes are keyed `<prefix>/<hex-key>`, so it is a drop-in remote
/// tier for the content-addressed cache shared across machines and pods.
#[cfg(feature = "s3")]
#[derive(Clone)]
pub struct S3BlobStore {
    store: Arc<dyn object_store::ObjectStore>,
    prefix: String,
}

#[cfg(feature = "s3")]
impl std::fmt::Debug for S3BlobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3BlobStore")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "s3")]
impl S3BlobStore {
    /// Build from the ambient AWS environment (`AWS_ACCESS_KEY_ID`,
    /// `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT` for MinIO/R2, …).
    /// `bucket` names the container; `prefix` namespaces the cache within it.
    pub fn from_env(bucket: &str, prefix: impl Into<String>) -> std::io::Result<Self> {
        let store = object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(to_io)?;
        Ok(Self {
            store: Arc::new(store),
            prefix: prefix.into(),
        })
    }

    fn obj_path(&self, key: ContentHash) -> object_store::path::Path {
        object_store::path::Path::from(format!("{}/{}", self.prefix, key.to_hex()))
    }
}

/// Run a `Send` future to completion from a SYNC context that may itself be
/// inside a Tokio runtime (the executor's async coordinator calls the cache
/// synchronously). `block_on` on the calling thread would panic with "Cannot
/// start a runtime from within a runtime", so the future runs on a fresh
/// non-runtime thread with its own current-thread runtime. Cache-tier calls
/// are infrequent and S3 latency dominates, so the per-call thread is cheap.
#[cfg(feature = "s3")]
fn block_on_isolated<T: Send>(fut: impl std::future::Future<Output = T> + Send) -> T {
    std::thread::scope(|s| {
        s.spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build current-thread runtime for S3 blob op")
                .block_on(fut)
        })
        .join()
        .expect("S3 blob worker thread panicked")
    })
}

#[cfg(feature = "s3")]
fn to_io(e: object_store::Error) -> std::io::Error {
    std::io::Error::other(e)
}

#[cfg(feature = "s3")]
impl BlobStore for S3BlobStore {
    fn get(&self, key: ContentHash) -> std::io::Result<Option<Vec<u8>>> {
        let path = self.obj_path(key);
        let store = self.store.clone();
        block_on_isolated(async move {
            match store.get(&path).await {
                Ok(res) => {
                    let bytes = res.bytes().await.map_err(to_io)?;
                    Ok(Some(bytes.to_vec()))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(to_io(e)),
            }
        })
    }

    fn put(&self, key: ContentHash, bytes: &[u8]) -> std::io::Result<()> {
        let path = self.obj_path(key);
        let store = self.store.clone();
        let payload = object_store::PutPayload::from(bytes::Bytes::copy_from_slice(bytes));
        block_on_isolated(async move { store.put(&path, payload).await.map(|_| ()).map_err(to_io) })
    }

    fn head(&self, key: ContentHash) -> std::io::Result<bool> {
        let path = self.obj_path(key);
        let store = self.store.clone();
        block_on_isolated(async move {
            match store.head(&path).await {
                Ok(_) => Ok(true),
                Err(object_store::Error::NotFound { .. }) => Ok(false),
                Err(e) => Err(to_io(e)),
            }
        })
    }
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

    // MinIO round-trip for the S3 backend. `#[ignore]` — needs a running
    // S3-compatible endpoint; `scripts/minio_smoke.sh` stands one up and runs
    // this. Exercises the same get/put/head contract as FsBlobStore, plus the
    // nested-runtime-safe async bridge from a sync (here: `#[tokio::test]`)
    // caller.
    #[cfg(feature = "s3")]
    #[tokio::test]
    #[ignore = "needs a MinIO/S3 endpoint (AWS_* env); run via scripts/minio_smoke.sh"]
    async fn s3_blob_store_round_trip() {
        let bucket = std::env::var("BLUT_S3_TEST_BUCKET").expect("BLUT_S3_TEST_BUCKET");
        let store = super::S3BlobStore::from_env(&bucket, "blut-test").unwrap();
        let key = ContentHash::of_bytes(b"s3-payload");
        // Runs inside a #[tokio::test] runtime — proves the isolated-thread
        // bridge doesn't panic with "runtime within a runtime".
        assert!(!store.head(key).unwrap());
        assert_eq!(store.get(key).unwrap(), None);
        store.put(key, b"s3-bytes").unwrap();
        assert!(store.head(key).unwrap());
        assert_eq!(store.get(key).unwrap().as_deref(), Some(&b"s3-bytes"[..]));
    }
}
