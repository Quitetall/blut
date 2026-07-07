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
