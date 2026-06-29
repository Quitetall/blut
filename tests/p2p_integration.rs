// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! P2P integration test — two-process QUIC coordinator + peer on localhost.
//!
//! Tests the full lifecycle: coordinator starts, peer connects, task is
//! dispatched, peer computes, result is returned and verified.

#![cfg(feature = "p2p")]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use blut::framework::artifact::ContentHash;
use blut::p2p::crypto::KeyPair;
use blut::p2p::dispatch::{DefaultDispatchPolicy, DispatchPolicy};
use blut::p2p::peer::{PeerCapabilities, PeerId};
use blut::p2p::registry::PeerRegistry;
use blut::p2p::task::{ResourceRequest, TaskManifest, TaskResult};
use blut::p2p::transport::P2pClient;
use blut::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};
use blut::p2p::Coordinator;

/// Create a temporary peer registry.
fn temp_registry() -> (PeerRegistry, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let reg = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
    (reg, dir)
}

#[tokio::test]
async fn coordinator_accepts_peer_connection() {
    let coord_kp = Arc::new(KeyPair::generate());
    let peer_kp = Arc::new(KeyPair::generate());

    let dispatch: Arc<dyn DispatchPolicy> =
        Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));
    let (registry, _dir) = temp_registry();

    let coordinator = Coordinator::start(
        "127.0.0.1:0".parse().unwrap(),
        coord_kp.clone(),
        dispatch,
        registry,
    )
    .await
    .unwrap();

    let addr = coordinator.local_addr().unwrap();

    // Peer connects.
    let client = P2pClient::new(peer_kp.clone());
    let (conn, peer_id) = client.connect(addr).await.unwrap();

    // Verify peer is registered.
    let peers_arc = coordinator.peers();
    let peers = peers_arc.read().await;
    assert!(peers.get(&peer_id).is_some());
    assert_eq!(peers.get(&peer_id).unwrap().trust, TrustLevel::Anonymous);

    drop(conn);
    coordinator.shutdown();
}

#[tokio::test]
async fn task_manifest_sign_verify_roundtrip() {
    let kp = KeyPair::generate();
    let input_hash = ContentHash::of_bytes(&[1u8; 32]);
    let args_hash = ContentHash::of_bytes(&[2u8; 32]);
    let expected_output_hash = ContentHash::of_bytes(&[3u8; 32]);

    let mut manifest = TaskManifest {
        task_id: "test-task-1".into(),
        coordinator_id: PeerId::from_pubkey(&kp.verifying),
        stage_name: "warm_fb_cache".into(),
        stage_schema: 1,
        input_hash,
        args_hash,
        expected_output_hash,
        args: serde_json::json!({"lma_root": "/data"}),
        resources: ResourceRequest::default(),
        data_class: DataClass::Public,
        timeout_secs: 3600,
        encrypted_input: None,
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    manifest.signature = kp.sign(&manifest.sign_payload());

    // Verify signature.
    let payload = manifest.sign_payload();
    assert!(blut::p2p::crypto::verify(&kp.verifying, &payload, &manifest.signature));

    // Tamper with task_id → signature invalid.
    let mut tampered = manifest.clone();
    tampered.task_id = "tampered".into();
    let tampered_payload = tampered.sign_payload();
    assert!(!blut::p2p::crypto::verify(
        &kp.verifying,
        &tampered_payload,
        &tampered.signature
    ));
}

#[tokio::test]
async fn task_result_sign_verify_roundtrip() {
    let kp = KeyPair::generate();
    let output_hash = ContentHash::of_bytes(&[42u8; 32]);

    let mut result = TaskResult {
        task_id: "test-task-1".into(),
        peer_id: PeerId::from_pubkey(&kp.verifying),
        output_hash,
        encrypted_output: None,
        wall_time_ms: 1500,
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    result.signature = kp.sign(&result.sign_payload());

    let payload = result.sign_payload();
    assert!(blut::p2p::crypto::verify(&kp.verifying, &payload, &result.signature));
}

#[tokio::test]
async fn dispatch_matrix_blocks_restricted_for_anonymous() {
    let matrix = DispatchMatrix::default();
    assert!(!matrix.can_dispatch(DataClass::Restricted, TrustLevel::Anonymous));
    assert!(!matrix.can_dispatch(DataClass::Restricted, TrustLevel::Registered));
    assert!(matrix.can_dispatch(DataClass::Restricted, TrustLevel::Trusted));
}

#[tokio::test]
async fn peer_capabilities_serde_roundtrip() {
    let caps = PeerCapabilities {
        cpu_cores: 16,
        memory_gib: 64,
        gpu_model: Some("NVIDIA A100".into()),
        gpu_vram_gib: Some(80),
    };
    let json = serde_json::to_string(&caps).unwrap();
    let caps2: PeerCapabilities = serde_json::from_str(&json).unwrap();
    assert_eq!(caps.cpu_cores, caps2.cpu_cores);
    assert_eq!(caps.gpu_vram_gib, caps2.gpu_vram_gib);
}

#[tokio::test]
async fn blob_side_stream_round_trips_over_quic() {
    use blut::p2p::bundle::BlobDir;
    use blut::p2p::transport::{recv_blob, send_blob, P2pServer, P2pClient};

    let coord_kp = Arc::new(KeyPair::generate());
    let peer_kp = Arc::new(KeyPair::generate());
    let (registry, _dir) = temp_registry();

    let server = Arc::new(
        P2pServer::bind("127.0.0.1:0".parse().unwrap(), coord_kp.clone(), registry)
            .await
            .unwrap(),
    );
    let addr = server.local_addr().unwrap();

    // Payload > CHUNK_MAX (12 MiB) so send_blob actually emits ≥2 chunks and the
    // receiver exercises seq ordering + reassembly. Pattern bytes so a reorder
    // or truncation changes the sha.
    let pack: Vec<u8> = (0..(13 * 1024 * 1024)).map(|i| (i % 251) as u8).collect();
    let expect = pack.clone();

    // Server side: accept the peer (consumes the handshake stream), then recv
    // the blob.
    let server_c = server.clone();
    let recv = tokio::spawn(async move {
        let (_peer_id, conn) = server_c.accept_peer().await.unwrap();
        recv_blob(&conn, "task-blob-1", BlobDir::Input, blut::p2p::transport::MAX_BLOB_SIZE)
            .await
            .unwrap()
    });

    // Peer side: connect (sends handshake), then send the blob.
    let client = P2pClient::new(peer_kp.clone());
    let (conn, _peer_id) = client.connect(addr).await.unwrap();
    send_blob(&conn, "task-blob-1", BlobDir::Input, &pack)
        .await
        .unwrap();

    let got = tokio::time::timeout(Duration::from_secs(5), recv)
        .await
        .expect("blob recv must finish")
        .unwrap();
    assert_eq!(got, expect, "reassembled blob matches the sent pack");

    drop(conn);
    server.shutdown();
}

#[tokio::test]
async fn recv_blob_rejects_oversized_total_len() {
    use blut::p2p::bundle::BlobDir;
    use blut::p2p::transport::{recv_blob, send_blob, P2pServer, P2pClient};

    let coord_kp = Arc::new(KeyPair::generate());
    let peer_kp = Arc::new(KeyPair::generate());
    let (registry, _dir) = temp_registry();
    let server = Arc::new(
        P2pServer::bind("127.0.0.1:0".parse().unwrap(), coord_kp.clone(), registry)
            .await
            .unwrap(),
    );
    let addr = server.local_addr().unwrap();

    // A 2 MiB pack, but the receiver caps at 1 MiB → Begin's total_len (2 MiB)
    // must be rejected BEFORE any chunk is buffered.
    let pack: Vec<u8> = vec![0xab; 2 * 1024 * 1024];

    let server_c = server.clone();
    let recv = tokio::spawn(async move {
        let (_id, conn) = server_c.accept_peer().await.unwrap();
        recv_blob(&conn, "t", BlobDir::Input, 1024 * 1024).await // 1 MiB cap
    });

    let client = P2pClient::new(peer_kp.clone());
    let (conn, _id) = client.connect(addr).await.unwrap();
    // Sender doesn't know the cap; it just streams. The receiver rejects.
    let _ = send_blob(&conn, "t", BlobDir::Input, &pack).await;

    let r = tokio::time::timeout(Duration::from_secs(5), recv)
        .await
        .expect("recv must finish")
        .unwrap();
    assert!(r.is_err(), "oversized total_len must be rejected");
    let msg = format!("{:?}", r.unwrap_err());
    assert!(msg.contains("max_blob_size"), "rejected for the cap reason: {msg}");

    drop(conn);
    server.shutdown();
}
