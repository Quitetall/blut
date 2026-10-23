// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Stage data over the mesh: content-addressed artifacts move peer to peer
//! (ADR 0079 data plane).
//!
//! Without this, a mesh worker running a real stage ([`SharedCacheRunner`])
//! could only read an input that was already in its object store, so every node
//! had to mount one shared directory or bucket. Here a node that lacks an
//! object PULLS it, by content id, from the peer that dispatched the task, and
//! the dispatcher pulls the output back from the worker the same way.
//!
//! **Pull, not push.** A node only ever writes an object it asked for, after it
//! has verified the signed task that names it. A push protocol would let any
//! connected peer write into a node's disk unasked.
//!
//! **Grants.** A node serves an object only to a peer it was granted to: the
//! dispatcher grants the task's input to the worker it dispatches to, and the
//! worker grants a task's output to the peer that dispatched it. Knowing a
//! content id is not enough to read an object, and an ungranted request is
//! refused with nothing sent. Grants expire ([`GRANT_TTL`]) and are capped per
//! peer.
//!
//! **Verification.** Every chunk is hash-checked as it lands (the
//! [`ChunkStore`]); the reassembled object must decode as a stored artifact
//! whose manifest names the requested content id and whose pack matches the
//! manifest's SHA-256 and length. The full content-id derivation then runs in
//! `unbundle` before anything is used, and only then is the object written to
//! the local store.
//!
//! **Bounds.** The requester caps the object size before any chunk is
//! requested ([`DEFAULT_MAX_OBJECT_BYTES`]), bounds every read by a stall
//! timeout, and bounds the whole fetch by the caller's deadline. The server
//! bounds concurrent sends ([`MAX_CONCURRENT_SENDS`]).
//!
//! **Wire.** One object per QUIC bi-stream, opened by the requester:
//! `[OBJECT_STREAM_TAG][OBJECT_PLANE_VERSION][32-byte content id]`. The server
//! answers `[0]` and then runs the deduplicated chunk transfer
//! (`send_blob_deduped`'s protocol: index, needed set, chunks, ack), or `[1]`
//! and a length-prefixed reason. The tag byte cannot collide with a
//! [`MeshFrame`](crate::p2p::mesh_wire::MeshFrame) stream, which starts with
//! the protocol version.
//!
//! [`SharedCacheRunner`]: crate::p2p::mesh_runner::SharedCacheRunner

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::error::TrainError;
use crate::framework::artifact::{ArtifactContentId, ContentHash};
use crate::framework::artifact_store::StoredArtifact;
use crate::framework::object_store::{ObjectKey, ObjectStore};
use crate::p2p::chunkstore::ChunkStore;
use crate::p2p::peer::PeerId;
use crate::p2p::transport::{recv_blob_deduped_on, send_blob_deduped_on};

/// First byte of an object-plane stream. Mesh frames start with their protocol
/// version (2), so the node's stream router can tell the two apart.
pub const OBJECT_STREAM_TAG: u8 = 0xB7;

/// Version of the object-plane request layout.
pub const OBJECT_PLANE_VERSION: u8 = 1;

/// Default cap on one fetched object. The receiver checks the offered size
/// against it before requesting a single chunk.
pub const DEFAULT_MAX_OBJECT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Default bound on any single read during a transfer.
pub const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a grant lets a peer read an object.
pub const GRANT_TTL: Duration = Duration::from_secs(6 * 3600);

/// Max live grants per peer (a bound on the grant table, not on the store).
pub const MAX_GRANTS_PER_PEER: usize = 1024;

/// Max objects a node sends at once, across all peers (each send holds the
/// object in memory).
pub const MAX_CONCURRENT_SENDS: usize = 4;

const STATUS_OK: u8 = 0;
const STATUS_REFUSED: u8 = 1;
const MAX_REASON: usize = 1024;

/// A node's object plane: the store it serves from and fetches into, a chunk
/// store for deduplicated transfer, and the grant table.
pub struct MeshObjects {
    store: ObjectStore,
    chunks: ChunkStore,
    max_object_bytes: u64,
    stall: Duration,
    grants: std::sync::Mutex<HashMap<PeerId, HashMap<ArtifactContentId, Instant>>>,
    sends: tokio::sync::Semaphore,
}

impl MeshObjects {
    /// An object plane over `store`, with transfer chunks kept under
    /// `chunk_dir`. Default limits.
    pub fn new(store: ObjectStore, chunk_dir: PathBuf) -> Self {
        Self {
            store,
            chunks: ChunkStore::new(chunk_dir),
            max_object_bytes: DEFAULT_MAX_OBJECT_BYTES,
            stall: DEFAULT_STALL_TIMEOUT,
            grants: std::sync::Mutex::new(HashMap::new()),
            sends: tokio::sync::Semaphore::new(MAX_CONCURRENT_SENDS),
        }
    }

    /// Set the largest object this node will fetch.
    pub fn with_max_object_bytes(mut self, max: u64) -> Self {
        self.max_object_bytes = max;
        self
    }

    /// Set the per-read stall bound for transfers in both directions.
    pub fn with_stall_timeout(mut self, stall: Duration) -> Self {
        self.stall = stall;
        self
    }

    /// The object store this plane serves from and fetches into.
    pub fn store(&self) -> &ObjectStore {
        &self.store
    }

    /// The largest object this node will fetch.
    pub fn max_object_bytes(&self) -> u64 {
        self.max_object_bytes
    }

    /// Let `peer` read artifact `id` from this node for [`GRANT_TTL`].
    pub fn grant(&self, peer: &PeerId, id: ArtifactContentId) {
        let now = Instant::now();
        let mut grants = self.grants.lock().unwrap_or_else(|p| p.into_inner());
        let per_peer = grants.entry(peer.clone()).or_default();
        per_peer.retain(|_, expires| *expires > now);
        if per_peer.len() >= MAX_GRANTS_PER_PEER && !per_peer.contains_key(&id) {
            // Drop the grant closest to expiry to make room: the newest task's
            // objects matter more than the oldest.
            if let Some(oldest) = per_peer
                .iter()
                .min_by_key(|(_, expires)| **expires)
                .map(|(k, _)| *k)
            {
                per_peer.remove(&oldest);
            }
        }
        per_peer.insert(id, now + GRANT_TTL);
    }

    /// Whether `peer` currently holds a grant for `id`.
    pub fn is_granted(&self, peer: &PeerId, id: ArtifactContentId) -> bool {
        let grants = self.grants.lock().unwrap_or_else(|p| p.into_inner());
        grants
            .get(peer)
            .and_then(|m| m.get(&id))
            .is_some_and(|expires| *expires > Instant::now())
    }

    /// Fetch artifact `id` from the peer at the other end of `conn`, verify it,
    /// and return it. Nothing is written to the object store here: the caller
    /// stores the object once `unbundle` has verified its full content id.
    /// `deadline` bounds the whole fetch; every read is also bounded by the
    /// stall timeout.
    pub async fn fetch_artifact(
        &self,
        conn: &quinn::Connection,
        id: ArtifactContentId,
        deadline: Duration,
    ) -> Result<StoredArtifact, TrainError> {
        let started = Instant::now();
        let fetch = async {
            let (mut send, mut recv) = conn
                .open_bi()
                .await
                .map_err(|e| TrainError::other(format!("open object stream: {e}")))?;
            let mut req = Vec::with_capacity(34);
            req.push(OBJECT_STREAM_TAG);
            req.push(OBJECT_PLANE_VERSION);
            req.extend_from_slice(&id.to_bytes());
            send.write_all(&req)
                .await
                .map_err(|e| TrainError::other(format!("write object request: {e}")))?;

            let mut status = [0u8; 1];
            read_bounded(self.stall, "the object status", &mut recv, &mut status).await?;
            match status[0] {
                STATUS_OK => {}
                STATUS_REFUSED => {
                    let mut len = [0u8; 2];
                    read_bounded(self.stall, "the refusal", &mut recv, &mut len).await?;
                    let n = (u16::from_le_bytes(len) as usize).min(MAX_REASON);
                    let mut reason = vec![0u8; n];
                    read_bounded(self.stall, "the refusal", &mut recv, &mut reason).await?;
                    return Err(TrainError::other(format!(
                        "peer refused object {}: {}",
                        id.to_hex(),
                        String::from_utf8_lossy(&reason)
                    )));
                }
                other => {
                    return Err(TrainError::other(format!(
                        "object stream: unknown status byte {other}"
                    )));
                }
            }
            tracing::debug!(
                "object {}: accepted by peer after {} ms",
                id.to_hex(),
                started.elapsed().as_millis()
            );
            recv_blob_deduped_on(
                &mut send,
                &mut recv,
                &self.chunks,
                self.max_object_bytes,
                Some(self.stall),
            )
            .await
        };
        let bytes = tokio::time::timeout(deadline, fetch).await.map_err(|_| {
            TrainError::other(format!(
                "fetching object {} did not finish within {}s",
                id.to_hex(),
                deadline.as_secs()
            ))
        })??;
        let transferred = started.elapsed();
        let stored = verify_stored_artifact(&bytes, id)?;
        tracing::info!(
            "fetched object {} from {}: {} bytes in {} ms (verified in {} ms)",
            id.to_hex(),
            conn.remote_address(),
            bytes.len(),
            transferred.as_millis(),
            (started.elapsed() - transferred).as_millis()
        );
        Ok(stored)
    }

    /// Serve one object-plane stream whose tag byte the router already read.
    /// Refuses an ungranted, absent or malformed request with a reason.
    pub(crate) async fn serve_stream(
        &self,
        peer: &PeerId,
        mut send: quinn::SendStream,
        mut recv: quinn::RecvStream,
    ) -> Result<(), TrainError> {
        let mut req = [0u8; 33];
        read_bounded(self.stall, "the object request", &mut recv, &mut req).await?;
        if req[0] != OBJECT_PLANE_VERSION {
            return refuse(&mut send, "unsupported object-plane version").await;
        }
        let id = ArtifactContentId::from_bytes(req[1..].try_into().expect("32 bytes"));
        if !self.is_granted(peer, id) {
            return refuse(&mut send, "not granted to this peer").await;
        }
        let _permit = self
            .sends
            .acquire()
            .await
            .map_err(|_| TrainError::other("object plane closed"))?;
        let bytes = match self.store.get(ObjectKey::Artifact(id)).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return refuse(&mut send, "not in this node's store").await,
            Err(e) => return refuse(&mut send, &format!("store read failed: {e}")).await,
        };
        send.write_all(&[STATUS_OK])
            .await
            .map_err(|e| TrainError::other(format!("write object status: {e}")))?;
        send_blob_deduped_on(&mut send, &mut recv, &self.chunks, &bytes, Some(self.stall)).await
    }
}

/// Decode a fetched object and bind it to the id that was asked for: it must be
/// a stored artifact whose manifest names `id` and whose pack matches the
/// manifest's SHA-256 and length. (The content-id derivation from the files
/// themselves runs in `unbundle`, which needs the stage.)
pub fn verify_stored_artifact(
    bytes: &[u8],
    id: ArtifactContentId,
) -> Result<StoredArtifact, TrainError> {
    let stored: StoredArtifact = bincode::deserialize(bytes)
        .map_err(|e| TrainError::other(format!("fetched object does not decode: {e}")))?;
    if stored.manifest.content_id != id {
        return Err(TrainError::other(format!(
            "fetched object names content id {}, but {} was requested (rejected)",
            stored.manifest.content_id.to_hex(),
            id.to_hex()
        )));
    }
    if stored.pack.len() as u64 != stored.manifest.blob_len
        || ContentHash::of_bytes(&stored.pack) != stored.manifest.blob_sha256
    {
        return Err(TrainError::other(format!(
            "fetched object {}: pack does not match its manifest hash (rejected)",
            id.to_hex()
        )));
    }
    Ok(stored)
}

async fn read_bounded(
    stall: Duration,
    what: &str,
    recv: &mut quinn::RecvStream,
    buf: &mut [u8],
) -> Result<(), TrainError> {
    tokio::time::timeout(stall, recv.read_exact(buf))
        .await
        .map_err(|_| {
            TrainError::other(format!(
                "peer stalled: nothing received for {}s while reading {what}",
                stall.as_secs()
            ))
        })?
        .map_err(|e| TrainError::other(format!("read {what}: {e}")))
}

pub(crate) async fn refuse(send: &mut quinn::SendStream, reason: &str) -> Result<(), TrainError> {
    let reason = &reason.as_bytes()[..reason.len().min(MAX_REASON)];
    let mut out = Vec::with_capacity(3 + reason.len());
    out.push(STATUS_REFUSED);
    out.extend_from_slice(&(reason.len() as u16).to_le_bytes());
    out.extend_from_slice(reason);
    send.write_all(&out)
        .await
        .map_err(|e| TrainError::other(format!("write refusal: {e}")))?;
    let _ = send.finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::cookbook::Registry;
    use crate::framework::stage::ErasedArtifact;
    use crate::p2p::chunkstore::ChunkIndex;
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::mesh_runner::SharedCacheRunner;
    use crate::p2p::node::{MeshNode, NodeCapabilities};
    use crate::p2p::registry::PeerRegistry;
    use crate::p2p::smoke::{SMOKE_STAGE, SmokeText};
    use crate::p2p::transport::{P2pClient, P2pServer};
    use std::sync::Arc;

    fn tmp() -> PathBuf {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().to_path_buf();
        std::mem::forget(d);
        p
    }

    async fn scheduler_node(objects: Option<Arc<MeshObjects>>) -> (Arc<MeshNode>, Arc<KeyPair>) {
        let kp = Arc::new(KeyPair::generate());
        let peers = PeerRegistry::load(&tmp().join("peers.json")).unwrap();
        let node = MeshNode::bind(
            "127.0.0.1:0".parse().unwrap(),
            kp.clone(),
            peers,
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
            false,
        )
        .await
        .unwrap();
        if let Some(o) = objects {
            node.set_objects(o);
        }
        (node, kp)
    }

    /// Publish a `p2p-echo` input of `text` into `store`; return its id.
    async fn publish(store: &ObjectStore, text: &str) -> ArtifactContentId {
        let mut registry = Registry::new();
        crate::p2p::smoke::register(&mut registry);
        let src = tmp();
        let path = src.join("p2p-echo-in.txt");
        std::fs::write(&path, text).unwrap();
        let input = SmokeText {
            content_hash: ContentHash::hash_file(&path).unwrap(),
            path,
        };
        SharedCacheRunner::publish_input(
            store,
            &registry,
            SMOKE_STAGE,
            ErasedArtifact::from_typed(&input).unwrap(),
            &src,
        )
        .await
        .unwrap()
    }

    /// Serving node S (objects attached) + client node C dialed into it.
    async fn pair() -> (
        Arc<MeshNode>,
        Arc<MeshObjects>,
        Arc<MeshNode>,
        MeshObjects,
        quinn::Connection,
    ) {
        let s_objects = Arc::new(MeshObjects::new(ObjectStore::filesystem(tmp()), tmp()));
        let (s, s_kp) = scheduler_node(Some(s_objects.clone())).await;
        let (c, _) = scheduler_node(None).await;
        let conn = c
            .connect_to(s.local_addr().unwrap(), s_kp.verifying.to_bytes())
            .await
            .unwrap();
        let c_objects = MeshObjects::new(ObjectStore::filesystem(tmp()), tmp());
        (s, s_objects, c, c_objects, conn)
    }

    const DL: Duration = Duration::from_secs(30);

    #[tokio::test]
    async fn only_a_granted_peer_can_fetch_and_the_object_verifies() {
        let (_s, s_objects, c, c_objects, conn) = pair().await;
        // 3 MiB spans more than one transfer chunk only if CHUNK_MAX is small;
        // either way the bytes must round-trip exactly.
        let text = "mesh object plane ".repeat(3 * 1024 * 1024 / 18);
        let id = publish(s_objects.store(), &text).await;

        let err = c_objects.fetch_artifact(&conn, id, DL).await.unwrap_err();
        assert!(format!("{err}").contains("not granted"), "{err}");

        s_objects.grant(c.node_id(), id);
        let got = c_objects.fetch_artifact(&conn, id, DL).await.unwrap();
        assert_eq!(got.manifest.content_id, id);
        // Nothing was written to the fetcher's store by the fetch itself.
        assert!(
            !c_objects
                .store()
                .contains(ObjectKey::Artifact(id))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn an_object_that_is_not_the_requested_id_is_rejected() {
        let (_s, s_objects, c, c_objects, conn) = pair().await;
        let real = publish(s_objects.store(), "real").await;
        let other = publish(s_objects.store(), "other").await;
        // Plant `other`'s bytes under `real`'s key on the serving node.
        let other_bytes = s_objects
            .store()
            .get(ObjectKey::Artifact(other))
            .await
            .unwrap()
            .unwrap();
        // The artifact namespace is not payload-addressed, so a corrupted or
        // hostile store can hold the wrong bytes under a key.
        let bad_store = ObjectStore::filesystem(tmp());
        bad_store
            .put(ObjectKey::Artifact(real), other_bytes)
            .await
            .expect("plant mislabelled bytes");
        let (s2, s2_kp) = scheduler_node(Some(Arc::new(MeshObjects::new(bad_store, tmp())))).await;
        s2.objects().unwrap().grant(c.node_id(), real);
        let conn2 = c
            .connect_to(s2.local_addr().unwrap(), s2_kp.verifying.to_bytes())
            .await
            .unwrap();
        let err = c_objects
            .fetch_artifact(&conn2, real, DL)
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("was requested"), "{err}");
        let other_bytes = s_objects
            .store()
            .get(ObjectKey::Artifact(other))
            .await
            .unwrap()
            .unwrap();
        let err = verify_stored_artifact(&other_bytes, real).unwrap_err();
        assert!(format!("{err}").contains("was requested"), "{err}");
        let _ = conn;
    }

    #[tokio::test]
    async fn an_object_over_the_cap_is_refused_before_any_chunk() {
        let (_s, s_objects, c, _c_objects, conn) = pair().await;
        let id = publish(s_objects.store(), &"x".repeat(64 * 1024)).await;
        s_objects.grant(c.node_id(), id);
        let small =
            MeshObjects::new(ObjectStore::filesystem(tmp()), tmp()).with_max_object_bytes(1024);
        let err = small.fetch_artifact(&conn, id, DL).await.unwrap_err();
        assert!(format!("{err}").contains("exceeds cap"), "{err}");
    }

    #[tokio::test]
    async fn a_node_without_an_object_plane_refuses() {
        let (s, s_kp) = scheduler_node(None).await;
        let (c, _) = scheduler_node(None).await;
        let conn = c
            .connect_to(s.local_addr().unwrap(), s_kp.verifying.to_bytes())
            .await
            .unwrap();
        let objects = MeshObjects::new(ObjectStore::filesystem(tmp()), tmp());
        let id = ArtifactContentId::from_digest(ContentHash::of_bytes(b"x"));
        let err = objects.fetch_artifact(&conn, id, DL).await.unwrap_err();
        assert!(format!("{err}").contains("serves no objects"), "{err}");
    }

    /// How a hostile server behaves after offering an index.
    #[derive(Clone, Copy)]
    enum Hostile {
        WrongChunkBytes,
        StallAfterIndex,
    }

    /// A raw server that answers one object request with a one-chunk index and
    /// then misbehaves. Returns the client's connection.
    async fn hostile_server(mode: Hostile) -> quinn::Connection {
        let server_kp = Arc::new(KeyPair::generate());
        let peers = PeerRegistry::load(&tmp().join("peers.json")).unwrap();
        let server = P2pServer::bind("127.0.0.1:0".parse().unwrap(), server_kp.clone(), peers)
            .await
            .unwrap();
        let addr = server.local_addr().unwrap();
        tokio::spawn(async move {
            let (_peer, conn) = server.accept_peer().await.unwrap();
            let (mut send, mut recv) = conn.accept_bi().await.unwrap();
            let mut req = [0u8; 34];
            recv.read_exact(&mut req).await.unwrap();
            assert_eq!(req[0], OBJECT_STREAM_TAG);
            send.write_all(&[STATUS_OK]).await.unwrap();
            let blob = vec![7u8; 100];
            let index = ChunkIndex::of(&blob);
            let idx = bincode::serialize(&index).unwrap();
            send.write_all(&(idx.len() as u32).to_le_bytes())
                .await
                .unwrap();
            send.write_all(&idx).await.unwrap();
            // Read the needed set.
            let mut len = [0u8; 4];
            recv.read_exact(&mut len).await.unwrap();
            let mut needed = vec![0u8; u32::from_le_bytes(len) as usize];
            recv.read_exact(&mut needed).await.unwrap();
            match mode {
                Hostile::WrongChunkBytes => {
                    let bad = vec![8u8; 100];
                    send.write_all(&0u32.to_le_bytes()).await.unwrap();
                    send.write_all(&(bad.len() as u32).to_le_bytes())
                        .await
                        .unwrap();
                    send.write_all(&bad).await.unwrap();
                }
                Hostile::StallAfterIndex => {}
            }
            let _ = tokio::time::timeout(crate::TEST_HANG_GUARD, conn.closed()).await;
        });
        let client = P2pClient::with_coordinator_pin(
            Arc::new(KeyPair::generate()),
            server_kp.verifying.to_bytes(),
        );
        client.connect(addr).await.unwrap().0
    }

    #[tokio::test]
    async fn a_chunk_that_does_not_match_its_hash_is_rejected() {
        let conn = hostile_server(Hostile::WrongChunkBytes).await;
        let objects = MeshObjects::new(ObjectStore::filesystem(tmp()), tmp());
        let id = ArtifactContentId::from_digest(ContentHash::of_bytes(b"x"));
        let err = objects.fetch_artifact(&conn, id, DL).await.unwrap_err();
        assert!(format!("{err}").contains("store chunk"), "{err}");
    }

    #[tokio::test]
    async fn a_peer_that_stalls_mid_transfer_times_out() {
        let conn = hostile_server(Hostile::StallAfterIndex).await;
        let objects = MeshObjects::new(ObjectStore::filesystem(tmp()), tmp())
            .with_stall_timeout(Duration::from_millis(500));
        let id = ArtifactContentId::from_digest(ContentHash::of_bytes(b"x"));
        let started = Instant::now();
        let err = objects.fetch_artifact(&conn, id, DL).await.unwrap_err();
        assert!(format!("{err}").contains("stalled"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn grants_expire_per_peer_and_are_capped() {
        let objects = MeshObjects::new(ObjectStore::filesystem(tmp()), tmp());
        let a = PeerId::from_pubkey(&KeyPair::generate().verifying);
        let b = PeerId::from_pubkey(&KeyPair::generate().verifying);
        let id = |i: u32| ArtifactContentId::from_digest(ContentHash::of_bytes(&i.to_le_bytes()));
        objects.grant(&a, id(0));
        assert!(objects.is_granted(&a, id(0)));
        assert!(!objects.is_granted(&b, id(0)), "grants are per peer");
        for i in 1..=(MAX_GRANTS_PER_PEER as u32 + 10) {
            objects.grant(&a, id(i));
        }
        let held = (0..=(MAX_GRANTS_PER_PEER as u32 + 10))
            .filter(|i| objects.is_granted(&a, id(*i)))
            .count();
        assert_eq!(held, MAX_GRANTS_PER_PEER);
        assert!(objects.is_granted(&a, id(MAX_GRANTS_PER_PEER as u32 + 10)));
    }
}
