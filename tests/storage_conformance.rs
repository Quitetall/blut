// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0092 A10: one storage contract across adapters and caller modes.

use std::path::PathBuf;
use std::sync::Arc;

use blut::framework::artifact::{ContentHash, ContentId, InvocationKey};
use blut::framework::object_store::{
    BlockingObjectStore, MAX_OBJECT_SIZE, ObjectKey, ObjectStore, PutOutcome, StoreError,
};

struct Fixture {
    _root: tempfile::TempDir,
    store: ObjectStore,
    raw_prefix: PathBuf,
}

impl Fixture {
    fn filesystem() -> Self {
        let root = tempfile::tempdir().unwrap();
        let raw_prefix = root.path().to_path_buf();
        let store = ObjectStore::filesystem(&raw_prefix);
        Self {
            _root: root,
            store,
            raw_prefix,
        }
    }

    #[cfg(feature = "cloud")]
    fn provider() -> Self {
        let root = tempfile::tempdir().unwrap();
        let prefix = "provider-contract";
        let store = ObjectStore::local_provider(root.path(), prefix).unwrap();
        let raw_prefix = root.path().join(prefix);
        Self {
            _root: root,
            store,
            raw_prefix,
        }
    }

    fn raw_path(&self, key: ObjectKey) -> PathBuf {
        self.raw_prefix.join(key.relative_path())
    }
}

fn content_key(payload: &[u8]) -> ObjectKey {
    ObjectKey::DispatchBundle(ContentHash::of_bytes(payload))
}

async fn assert_async_contract(fixture: Fixture) {
    let missing = content_key(b"async-missing");
    assert_eq!(fixture.store.get(missing).await.unwrap(), None);
    assert!(!fixture.store.contains(missing).await.unwrap());

    let payload = b"async-round-trip".to_vec();
    let key = content_key(&payload);
    assert_eq!(
        fixture.store.put(key, payload.clone()).await.unwrap(),
        PutOutcome::Stored
    );
    assert!(fixture.store.contains(key).await.unwrap());
    assert_eq!(fixture.store.get(key).await.unwrap(), Some(payload.clone()));
    assert_eq!(
        fixture.store.put(key, payload.clone()).await.unwrap(),
        PutOutcome::AlreadyPresent
    );

    let wrong_key = content_key(b"async-claimed-content");
    let error = fixture
        .store
        .put(wrong_key, b"async-wrong-content".to_vec())
        .await
        .unwrap_err();
    assert!(matches!(error, StoreError::HashMismatch { .. }));
    assert!(!fixture.store.contains(wrong_key).await.unwrap());

    let invocation = ObjectKey::CacheInvocation(InvocationKey::from_digest(ContentHash::of_bytes(
        b"async-conflict-key",
    )));
    fixture
        .store
        .put(invocation, b"first".to_vec())
        .await
        .unwrap();
    let error = fixture
        .store
        .put(invocation, b"second".to_vec())
        .await
        .unwrap_err();
    assert!(matches!(error, StoreError::Conflict { .. }));
    assert_eq!(
        fixture.store.get(invocation).await.unwrap().as_deref(),
        Some(&b"first"[..])
    );

    let concurrent_payload = b"async-concurrent-put".to_vec();
    let concurrent_key = content_key(&concurrent_payload);
    let shared = Arc::new(fixture.store.clone());
    let mut writes = Vec::new();
    for _ in 0..12 {
        let store = shared.clone();
        let payload = concurrent_payload.clone();
        writes.push(tokio::spawn(async move {
            store.put(concurrent_key, payload).await
        }));
    }
    let mut stored = 0;
    let mut already_present = 0;
    for write in writes {
        match write.await.unwrap().unwrap() {
            PutOutcome::Stored => stored += 1,
            PutOutcome::AlreadyPresent => already_present += 1,
        }
    }
    assert_eq!(stored, 1);
    assert_eq!(already_present, 11);

    let corrupt_payload = b"async-corruption".to_vec();
    let corrupt_key = content_key(&corrupt_payload);
    fixture
        .store
        .put(corrupt_key, corrupt_payload)
        .await
        .unwrap();
    let corrupt_path = fixture.raw_path(corrupt_key);
    let mut raw = std::fs::read(&corrupt_path).unwrap();
    *raw.last_mut().unwrap() ^= 0x01;
    std::fs::write(&corrupt_path, raw).unwrap();
    let error = fixture.store.get(corrupt_key).await.unwrap_err();
    assert!(matches!(error, StoreError::Corrupt { .. }));

    let oversize_payload = b"async-oversize".to_vec();
    let oversize_key = content_key(&oversize_payload);
    fixture
        .store
        .put(oversize_key, oversize_payload)
        .await
        .unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.raw_path(oversize_key))
        .unwrap()
        .set_len(MAX_OBJECT_SIZE + 1024)
        .unwrap();
    let error = fixture.store.get(oversize_key).await.unwrap_err();
    assert!(matches!(error, StoreError::TooLarge { .. }));

    let digest = ContentHash::of_bytes(b"typed-namespace-digest");
    let cache_key = ObjectKey::CacheInvocation(InvocationKey::from_digest(digest));
    let artifact_key = ObjectKey::Artifact(ContentId::from_digest(digest));
    fixture
        .store
        .put(cache_key, b"invocation".to_vec())
        .await
        .unwrap();
    fixture
        .store
        .put(artifact_key, b"artifact".to_vec())
        .await
        .unwrap();
    assert_ne!(cache_key.relative_path(), artifact_key.relative_path());
    assert_eq!(
        fixture.store.get(cache_key).await.unwrap().as_deref(),
        Some(&b"invocation"[..])
    );
    assert_eq!(
        fixture.store.get(artifact_key).await.unwrap().as_deref(),
        Some(&b"artifact"[..])
    );
}

fn assert_blocking_contract(fixture: Fixture) {
    let store = fixture.store.blocking();
    let missing = content_key(b"blocking-missing");
    assert_eq!(store.get(missing).unwrap(), None);
    assert!(!store.contains(missing).unwrap());

    let payload = b"blocking-round-trip".to_vec();
    let key = content_key(&payload);
    assert_eq!(store.put(key, &payload).unwrap(), PutOutcome::Stored);
    assert!(store.contains(key).unwrap());
    assert_eq!(store.get(key).unwrap(), Some(payload.clone()));
    assert_eq!(
        store.put(key, &payload).unwrap(),
        PutOutcome::AlreadyPresent
    );

    let wrong_key = content_key(b"blocking-claimed-content");
    let error = store.put(wrong_key, b"blocking-wrong-content").unwrap_err();
    assert!(matches!(error, StoreError::HashMismatch { .. }));
    assert!(!store.contains(wrong_key).unwrap());

    let invocation = ObjectKey::CacheInvocation(InvocationKey::from_digest(ContentHash::of_bytes(
        b"blocking-conflict-key",
    )));
    store.put(invocation, b"first").unwrap();
    let error = store.put(invocation, b"second").unwrap_err();
    assert!(matches!(error, StoreError::Conflict { .. }));
    assert_eq!(
        store.get(invocation).unwrap().as_deref(),
        Some(&b"first"[..])
    );

    let digest = ContentHash::of_bytes(b"blocking-typed-namespace-digest");
    let cache_key = ObjectKey::CacheInvocation(InvocationKey::from_digest(digest));
    let artifact_key = ObjectKey::Artifact(ContentId::from_digest(digest));
    store.put(cache_key, b"invocation").unwrap();
    store.put(artifact_key, b"artifact").unwrap();
    assert_ne!(cache_key.relative_path(), artifact_key.relative_path());
    assert_eq!(
        store.get(cache_key).unwrap().as_deref(),
        Some(&b"invocation"[..])
    );
    assert_eq!(
        store.get(artifact_key).unwrap().as_deref(),
        Some(&b"artifact"[..])
    );

    let concurrent_payload = b"blocking-concurrent-put".to_vec();
    let concurrent_key = content_key(&concurrent_payload);
    let outcomes = std::thread::scope(|scope| {
        let mut writes = Vec::new();
        for _ in 0..12 {
            let store = store.clone();
            let payload = concurrent_payload.clone();
            writes.push(scope.spawn(move || store.put(concurrent_key, &payload)));
        }
        writes
            .into_iter()
            .map(|write| write.join().unwrap().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| **outcome == PutOutcome::Stored)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| **outcome == PutOutcome::AlreadyPresent)
            .count(),
        11
    );

    let corrupt_payload = b"blocking-corruption".to_vec();
    let corrupt_key = content_key(&corrupt_payload);
    store.put(corrupt_key, &corrupt_payload).unwrap();
    let corrupt_path = fixture.raw_path(corrupt_key);
    let mut raw = std::fs::read(&corrupt_path).unwrap();
    *raw.last_mut().unwrap() ^= 0x01;
    std::fs::write(&corrupt_path, raw).unwrap();
    let error = store.get(corrupt_key).unwrap_err();
    assert!(matches!(error, StoreError::Corrupt { .. }));

    let oversize_payload = b"blocking-oversize".to_vec();
    let oversize_key = content_key(&oversize_payload);
    store.put(oversize_key, &oversize_payload).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.raw_path(oversize_key))
        .unwrap()
        .set_len(MAX_OBJECT_SIZE + 1024)
        .unwrap();
    let error = store.get(oversize_key).unwrap_err();
    assert!(matches!(error, StoreError::TooLarge { .. }));
}

#[tokio::test(flavor = "multi_thread")]
async fn filesystem_async_conformance() {
    assert_async_contract(Fixture::filesystem()).await;
}

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "cloud")]
async fn provider_async_conformance() {
    assert_async_contract(Fixture::provider()).await;
}

#[test]
fn filesystem_blocking_conformance() {
    assert_blocking_contract(Fixture::filesystem());
}

#[test]
#[cfg(feature = "cloud")]
fn provider_blocking_conformance() {
    assert_blocking_contract(Fixture::provider());
}

#[test]
fn blocking_adapter_is_explicit_and_concrete() {
    let fixture = Fixture::filesystem();
    let _: BlockingObjectStore = fixture.store.blocking();
}
