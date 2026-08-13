// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! P2P integration test — two-process QUIC coordinator + peer on localhost.
//!
//! Tests the full lifecycle: coordinator starts, peer connects, task is
//! dispatched, peer computes, result is returned and verified.

#![cfg(feature = "p2p")]

use std::sync::Arc;
use std::time::Duration;

use blut::framework::artifact::{ContentHash, ContentId, InvocationKey};
use blut::p2p::Coordinator;
use blut::p2p::crypto::KeyPair;
use blut::p2p::dispatch::{DefaultDispatchPolicy, DispatchPolicy};
use blut::p2p::peer::{PeerCapabilities, PeerId};
use blut::p2p::registry::PeerRegistry;
use blut::p2p::task::{ResourceRequest, TaskManifest, TaskResult};
use blut::p2p::transport::P2pClient;
use blut::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};

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
async fn coordinator_reconnect_closes_replaced_peer_session() {
    let coord_kp = Arc::new(KeyPair::generate());
    let peer_kp = Arc::new(KeyPair::generate());
    let dispatch: Arc<dyn DispatchPolicy> =
        Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));
    let (registry, _dir) = temp_registry();
    let coordinator =
        Coordinator::start("127.0.0.1:0".parse().unwrap(), coord_kp, dispatch, registry)
            .await
            .unwrap();
    let client = P2pClient::new(peer_kp);
    let (first, _peer_id) = client
        .connect(coordinator.local_addr().unwrap())
        .await
        .unwrap();
    let (_second, _peer_id) = client
        .connect(coordinator.local_addr().unwrap())
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(2), first.closed())
        .await
        .expect("replaced peer session must close its prior connection");
    coordinator.shutdown();
}

#[tokio::test]
async fn task_manifest_sign_verify_roundtrip() {
    let kp = KeyPair::generate();
    let input_hash = ContentHash::of_bytes(&[1u8; 32]);
    let args_hash = ContentHash::of_bytes(&[2u8; 32]);
    let expected_output_hash = ContentHash::of_bytes(&[3u8; 32]);

    let mut manifest = TaskManifest {
        protocol_version: blut::p2p::task::TASK_PROTOCOL_VERSION,
        task_id: "test-task-1".into(),
        coordinator_id: PeerId::from_pubkey(&kp.verifying),
        stage_name: "warm_fb_cache".into(),
        stage_schema: 1,
        input_content_id: ContentId::from_digest(input_hash),
        invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"test-task-1")),
        args_hash,
        expected_content_id: Some(ContentId::from_digest(expected_output_hash)),
        args: serde_json::json!({"lma_root": "/data"}),
        resources: ResourceRequest::default(),
        data_class: DataClass::Public,
        timeout_secs: 3600,
        deadline: blut::framework::execution::ExecutionDeadline::from_now(
            None,
            Duration::from_secs(3600),
        ),
        encrypted_input: None,
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    manifest.signature = kp.sign(&manifest.sign_payload());

    // Verify signature.
    let payload = manifest.sign_payload();
    assert!(blut::p2p::crypto::verify(
        &kp.verifying,
        &payload,
        &manifest.signature
    ));

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
        protocol_version: blut::p2p::task::TASK_PROTOCOL_VERSION,
        task_id: "test-task-1".into(),
        peer_id: PeerId::from_pubkey(&kp.verifying),
        content_id: ContentId::from_digest(output_hash),
        encrypted_output: None,
        wall_time_ms: 1500,
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    result.signature = kp.sign(&result.sign_payload());

    let payload = result.sign_payload();
    assert!(blut::p2p::crypto::verify(
        &kp.verifying,
        &payload,
        &result.signature
    ));
}

#[tokio::test]
async fn dispatch_matrix_keeps_restricted_node_local_for_every_trust_tier() {
    let matrix = DispatchMatrix::default();
    assert!(!matrix.can_dispatch(DataClass::Restricted, TrustLevel::Anonymous));
    assert!(!matrix.can_dispatch(DataClass::Restricted, TrustLevel::Registered));
    assert!(!matrix.can_dispatch(DataClass::Restricted, TrustLevel::Trusted));
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
    use blut::p2p::transport::{P2pClient, P2pServer, recv_blob, send_blob};

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
        recv_blob(
            &conn,
            "task-blob-1",
            BlobDir::Input,
            blut::p2p::transport::MAX_BLOB_SIZE,
        )
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

// ── End-to-end: a real dispatchable stage runs on a peer (C1d) ───────────────

mod e2e {
    use super::*;
    use async_trait::async_trait;
    use blut::framework::artifact::Artifact;
    use blut::framework::artifact_store::{ArtifactRole, capture};
    use blut::framework::cookbook::{Cookbook, Registry};
    use blut::framework::error::StageError;
    use blut::framework::execution::{
        DataClassification, EXECUTION_PROTOCOL_VERSION, ExecutionDeadline, ExecutionRequest,
        ExecutionResources, ExecutionResult, drive_execution,
    };
    use blut::framework::resource::Resource;
    use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
    use blut::p2p::PeerInfo;
    use blut::p2p::dispatch::{DispatchPolicy, DispatchVerdict};
    use blut::p2p::peer_exec::{CoordinatorKeys, dispatch_to_peer, run_peer_loop};
    use blut::p2p::task::ResourceRequest;
    use blut::p2p::transport::P2pServer;
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};
    use tokio_util::sync::CancellationToken;

    // A file-backed artifact: one text file on disk.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TextFile {
        path: PathBuf,
        content_hash: ContentHash,
    }
    impl Artifact for TextFile {
        const KIND: &'static str = "test.textfile";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            self.content_hash
        }
        fn primary_path(&self) -> &Path {
            &self.path
        }
    }

    // A deterministic dispatchable stage: read the input file, uppercase it,
    // write the output file. Real work that opens primary_path().
    struct Upper;
    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct UpperArgs {}
    #[async_trait]
    impl Stage for Upper {
        const NAME: &'static str = "upper";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        const DETERMINISTIC: bool = true;
        type Input = TextFile;
        type Output = TextFile;
        type Args = UpperArgs;
        async fn run(
            &self,
            ctx: &StageContext,
            input: Self::Input,
            _args: &Self::Args,
        ) -> Result<Self::Output, StageError> {
            let body = std::fs::read_to_string(&input.path)
                .map_err(|e| StageError::Backend(anyhow::anyhow!("read input: {e}")))?;
            let upper = body.to_uppercase();
            let out = ctx.stage_dir.join("out.txt");
            std::fs::write(&out, upper.as_bytes())
                .map_err(|e| StageError::Backend(anyhow::anyhow!("write output: {e}")))?;
            Ok(TextFile {
                content_hash: ContentHash::hash_file(&out).unwrap(),
                path: out,
            })
        }
    }

    struct UpperCookbook;
    impl Cookbook for UpperCookbook {
        fn name(&self) -> &'static str {
            "upper-cookbook"
        }
        fn recipes(&self) -> &'static [&'static blut::recipes::recipe::RecipeDef] {
            &[]
        }
        fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
            static S: &[(&str, ErasedStageCtor)] = &[("upper", || std::sync::Arc::new(Upper))];
            S
        }
    }

    fn registry() -> Registry {
        let mut r = Registry::new();
        r.register(Box::new(UpperCookbook));
        r
    }

    // A permissive policy that allows the `upper` stage to all peers.
    struct AllowUpper;
    impl DispatchPolicy for AllowUpper {
        fn is_dispatchable(&self, stage_name: &str) -> bool {
            stage_name == "upper"
        }
        fn classify_stage(&self, _s: &str, _a: &serde_json::Value) -> DataClass {
            DataClass::Public
        }
        fn select_peer(
            &self,
            _s: &str,
            _r: &ResourceRequest,
            _d: DataClass,
            peers: &[PeerInfo],
        ) -> Option<PeerId> {
            peers.first().map(|p| p.id.clone())
        }
        fn verify_result(
            &self,
            _r: &TaskResult,
            _e: Option<ContentId>,
            _k: &ed25519_dalek::VerifyingKey,
        ) -> DispatchVerdict {
            DispatchVerdict::Accept
        }
    }

    #[tokio::test]
    async fn peer_runs_real_stage_end_to_end() {
        let coord_kp = Arc::new(KeyPair::generate());
        let peer_kp = Arc::new(KeyPair::generate());
        let (registry_store, _dir) = temp_registry();

        let server = Arc::new(
            P2pServer::bind(
                "127.0.0.1:0".parse().unwrap(),
                coord_kp.clone(),
                registry_store,
            )
            .await
            .unwrap(),
        );
        let addr = server.local_addr().unwrap();

        // Produce the input artifact on the COORDINATOR's disk (its src_root).
        let src_root = tempfile::tempdir().unwrap();
        let in_path = src_root.path().join("in.txt");
        std::fs::write(&in_path, b"hello p2p world").unwrap();
        let input = TextFile {
            content_hash: ContentHash::hash_file(&in_path).unwrap(),
            path: in_path.clone(),
        };
        let input_hash = input.content_hash;
        let input_erased = blut::framework::stage::ErasedArtifact::from_typed(&input).unwrap();

        // ── Peer side: accept the connection, run the loop for one task. ──
        let server_c = server.clone();
        let peer_kp_c = peer_kp.clone();
        let coord_verifying = coord_kp.verifying;
        let coord_x = coord_kp.x25519_public;
        let peer_work = tempfile::tempdir().unwrap();
        let peer_work_path = peer_work.path().to_path_buf();
        let peer = tokio::spawn(async move {
            let (_peer_id, conn) = server_c.accept_peer().await.unwrap();
            let reg = registry();
            let policy = AllowUpper;
            let ck = CoordinatorKeys {
                verifying: coord_verifying,
                x25519_pub: coord_x,
            };
            // Run exactly one task then return (recv_task errs when the conn
            // closes, ending the loop).
            let _ = tokio::time::timeout(
                Duration::from_secs(8),
                run_peer_loop(&conn, &peer_kp_c, &ck, &reg, &policy, &peer_work_path),
            )
            .await;
        });

        // ── Coordinator side (this test). ──
        // The QUIC connection is bidirectional. The peer task above accepted the
        // SERVER end (`conn`) and runs the peer loop on it (recv_task = accept_uni,
        // send_result = open_uni). Here we hold the CLIENT end (`coord_conn`) and
        // drive the coordinator half: dispatch_to_peer does send_task (open_uni)
        // + recv_result (accept_uni), which pair with the peer's opposite ends.
        let client = P2pClient::new(peer_kp.clone());
        let (coord_conn, _coord_peer_id) = client.connect(addr).await.unwrap();

        // Poll for the server-side accept_peer to register the peer (no fixed
        // sleep — retry up to ~2s), then fetch its info.
        let peer_id = PeerId::from_pubkey(&peer_kp.verifying);
        let mut peer_info = None;
        for _ in 0..40 {
            if let Some(info) = server.peers.read().await.get(&peer_id).cloned() {
                peer_info = Some(info);
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let peer_info = peer_info.expect("peer must register within 2s");

        let out_dir = tempfile::tempdir().unwrap();
        let dispatched = dispatch_to_peer(
            &coord_conn,
            &coord_kp,
            &peer_info,
            &registry(),
            "task-e2e-1",
            "upper",
            input_erased,
            src_root.path(),
            serde_json::json!({}),
            blut::framework::CacheHandle::key_for(
                "upper",
                1,
                input_hash,
                &serde_json::json!({}),
                b"p2p-integration-v1",
            ),
            None,
            blut::tenant::Tenant::default(),
            DataClass::Public,
            60, // timeout_secs
            out_dir.path(),
        )
        .await
        .expect("dispatch must succeed");

        // The returned output is a local handle; its file holds the uppercased
        // text and verifies against expected_output_hash (the bundle gates ran).
        let out: TextFile = dispatched.output.into_typed().unwrap();
        let body = std::fs::read_to_string(&out.path).unwrap();
        assert_eq!(body, "HELLO P2P WORLD", "peer ran the real stage");
        assert!(
            out.path.starts_with(out_dir.path()),
            "output materialized locally"
        );

        drop(coord_conn);
        let _ = tokio::time::timeout(Duration::from_secs(2), peer).await;
        server.shutdown();
    }

    #[tokio::test]
    async fn coordinator_adapter_restores_portable_output() {
        let coord_kp = Arc::new(KeyPair::generate());
        let peer_kp = Arc::new(KeyPair::generate());
        let policy: Arc<dyn DispatchPolicy> = Arc::new(AllowUpper);
        let (registry_store, _dir) = temp_registry();
        let coordinator = Coordinator::start(
            "127.0.0.1:0".parse().unwrap(),
            coord_kp.clone(),
            policy,
            registry_store,
        )
        .await
        .unwrap();
        let addr = coordinator.local_addr().unwrap();

        let peer_kp_c = peer_kp.clone();
        let coord_verifying = coord_kp.verifying;
        let coord_x = coord_kp.x25519_public;
        let peer_work = tempfile::tempdir().unwrap();
        let peer_work_path = peer_work.path().to_path_buf();
        let peer_stop = CancellationToken::new();
        let peer_stop_c = peer_stop.clone();
        let peer = tokio::spawn(async move {
            let client = P2pClient::new(peer_kp_c.clone());
            let (conn, _peer_id) = client.connect(addr).await.unwrap();
            let keys = CoordinatorKeys {
                verifying: coord_verifying,
                x25519_pub: coord_x,
            };
            let peer_registry = registry();
            let peer_policy = AllowUpper;
            tokio::select! {
                _ = peer_stop_c.cancelled() => {}
                _ = run_peer_loop(
                    &conn,
                    &peer_kp_c,
                    &keys,
                    &peer_registry,
                    &peer_policy,
                    &peer_work_path,
                ) => {}
            }
        });

        let peer_id = PeerId::from_pubkey(&peer_kp.verifying);
        for _ in 0..40 {
            if coordinator.peers().read().await.get(&peer_id).is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(coordinator.peers().read().await.get(&peer_id).is_some());

        let src_root = tempfile::tempdir().unwrap();
        let input_path = src_root.path().join("in.txt");
        std::fs::write(&input_path, b"adapter roundtrip").unwrap();
        let input = TextFile {
            content_hash: ContentHash::hash_file(&input_path).unwrap(),
            path: input_path,
        };
        let stage: Arc<dyn blut::framework::stage::StageDyn> = Arc::new(Upper);
        let stored = capture(
            stage.as_ref(),
            blut::framework::stage::ErasedArtifact::from_typed(&input).unwrap(),
            src_root.path(),
            ArtifactRole::Input,
            None,
        )
        .unwrap();
        let request = ExecutionRequest {
            protocol_version: EXECUTION_PROTOCOL_VERSION,
            execution_id: "adapter-roundtrip-1".into(),
            tenant: blut::tenant::Tenant::default(),
            stage_name: "upper".into(),
            stage_schema: 1,
            invocation_key: blut::framework::CacheHandle::key_for(
                "upper",
                1,
                input.content_hash,
                &serde_json::json!({}),
                b"p2p-adapter-v1",
            ),
            args_hash: ContentHash::of_bytes(b"{}"),
            args: serde_json::json!({}),
            input: Some(stored),
            expected_content_id: None,
            resources: ExecutionResources::default(),
            data_class: DataClassification::Public,
            deadline: ExecutionDeadline::from_now(None, Duration::from_secs(5)),
        };
        let output_root = tempfile::tempdir().unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            drive_execution(
                &coordinator,
                request,
                &CancellationToken::new(),
                stage,
                output_root.path(),
                Duration::from_millis(10),
            ),
        )
        .await
        .expect("canonical driver must respect P2P deadline");
        let artifact = match result {
            ExecutionResult::Succeeded { artifact, .. } => artifact,
            ExecutionResult::Failed(failure) => {
                panic!("expected restored P2P output, failed: {failure}")
            }
            ExecutionResult::Cancelled => panic!("expected restored P2P output, got cancellation"),
            ExecutionResult::TimedOut { .. } => panic!("expected restored P2P output, timed out"),
        };
        let output: TextFile = artifact.into_typed().unwrap();
        assert_eq!(
            std::fs::read_to_string(&output.path).unwrap(),
            "ADAPTER ROUNDTRIP"
        );
        assert!(output.path.starts_with(output_root.path()));

        coordinator.shutdown();
        peer_stop.cancel();
        peer.abort();
    }
}

#[tokio::test]
async fn recv_blob_rejects_oversized_total_len() {
    use blut::p2p::bundle::BlobDir;
    use blut::p2p::transport::{P2pClient, P2pServer, recv_blob, send_blob};

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
    assert!(
        msg.contains("max_blob_size"),
        "rejected for the cap reason: {msg}"
    );

    drop(conn);
    server.shutdown();
}

/// T1.6 (ADR 0067): the CLI smoke path — coordinator BINDS (`serve --smoke-stage`),
/// worker DIALS (`connect`) — dispatches the shipped `p2p-echo` stage over the full
/// data plane and content-verifies the result. Mirrors `peer_runs_real_stage_end_to_end`
/// but with the bind/dial roles matching the CLI and using the real shipped stage +
/// `DefaultDispatchPolicy` (which now allow-lists `p2p-echo`).
#[tokio::test]
async fn smoke_serve_dispatches_echo_over_loopback() {
    use blut::framework::cookbook::Registry;
    use blut::p2p::crypto::KeyPair;
    use blut::p2p::dispatch::DefaultDispatchPolicy;
    use blut::p2p::peer::PeerId;
    use blut::p2p::peer_exec::{CoordinatorKeys, run_peer_loop};
    use blut::p2p::smoke;
    use blut::p2p::transport::{P2pClient, P2pServer};
    use blut::p2p::trust::DispatchMatrix;

    let coord_kp = Arc::new(KeyPair::generate());
    let peer_kp = Arc::new(KeyPair::generate());
    let (peer_reg_store, _dir) = temp_registry();

    // Coordinator binds (the `serve` role).
    let server = Arc::new(
        P2pServer::bind(
            "127.0.0.1:0".parse().unwrap(),
            coord_kp.clone(),
            peer_reg_store,
        )
        .await
        .unwrap(),
    );
    let addr = server.local_addr().unwrap();

    // Worker dials in (the `connect` role) and runs the peer loop for one task.
    let coord_verifying = coord_kp.verifying;
    let coord_x = coord_kp.x25519_public;
    let peer_kp_c = peer_kp.clone();
    let peer_work = tempfile::tempdir().unwrap();
    let peer_work_path = peer_work.path().to_path_buf();
    let worker = tokio::spawn(async move {
        let mut pin = [0u8; 32];
        pin.copy_from_slice(coord_verifying.as_bytes());
        let client = P2pClient::with_coordinator_pin(peer_kp_c.clone(), pin);
        let (conn, _my_id) = client.connect(addr).await.unwrap();
        let mut reg = Registry::new();
        smoke::register(&mut reg);
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let ck = CoordinatorKeys {
            verifying: coord_verifying,
            x25519_pub: coord_x,
        };
        let _ = tokio::time::timeout(
            Duration::from_secs(8),
            run_peer_loop(&conn, &peer_kp_c, &ck, &reg, &policy, &peer_work_path),
        )
        .await;
    });

    // Coordinator accepts the worker, then drives the smoke dispatch.
    let (peer_id, conn) = server.accept_peer().await.unwrap();
    assert_eq!(peer_id, PeerId::from_pubkey(&peer_kp.verifying));
    let mut reg = Registry::new();
    smoke::register(&mut reg);
    let out = smoke::smoke_dispatch_once(
        &server,
        &conn,
        &coord_kp,
        &reg,
        &peer_id,
        smoke::SMOKE_STAGE,
        "hello smoke",
        30,
    )
    .await
    .expect("smoke dispatch must succeed");
    assert_eq!(out, "HELLO SMOKE", "peer ran the shipped p2p-echo stage");

    drop(conn);
    let _ = tokio::time::timeout(Duration::from_secs(2), worker).await;
    server.shutdown();
}
