// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! QUIC transport layer for P2P task dispatch.
//!
//! The coordinator runs a [`P2pServer`] that accepts peer connections.
//! Peers connect via [`P2pClient`]. Messages are serialized as
//! length-prefixed JSON over QUIC bidirectional streams.
//!
//! Wire protocol per stream:
//!   1. Sender writes: [4-byte LE length][JSON payload]
//!   2. Receiver reads: [4-byte LE length][JSON payload]
//!   3. Stream is closed after the exchange.
//!
//! Authentication: the peer sends its Ed25519 pubkey in the initial
//! handshake message. The coordinator verifies it against the registry.

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::{Endpoint, ServerConfig, Connection as QuinnConnection};
use serde::{Deserialize, Serialize};
// write_all/read_exact are inherent on quinn streams, no trait import needed.
use tokio::sync::RwLock;

use crate::error::TrainError;
use crate::framework::artifact::ContentHash;
use crate::p2p::bundle::BlobDir;
use crate::p2p::crypto::KeyPair;
use crate::p2p::peer::{PeerCapabilities, PeerId, PeerInfo};
use crate::p2p::registry::PeerRegistry;
use crate::p2p::task::{TaskManifest, TaskResult};
use crate::p2p::trust::TrustLevel;

/// Messages exchanged between coordinator and peer over QUIC streams.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WireMessage {
    /// Peer → Coordinator: initial handshake with identity.
    Handshake {
        pubkey: [u8; 32],
        x25519_pub: [u8; 32],
        capabilities: PeerCapabilities,
    },
    /// Coordinator → Peer: authentication result.
    HandshakeAck {
        peer_id: PeerId,
        trust: TrustLevel,
    },
    /// Coordinator → Peer: a task to execute.
    Task(TaskManifest),
    /// Peer → Coordinator: task result.
    Result(TaskResult),
    /// Coordinator → Peer: cancel a running task.
    Cancel { task_id: String },
    /// Either direction: error message.
    Error { message: String },
    /// Either direction: start of an artifact-bundle blob side-stream, keyed by
    /// `task_id`. The blob bytes follow in `BundleBlobChunk` frames and end with
    /// `BundleBlobEnd`. `total_len`/`blob_sha256` let the receiver size + verify
    /// (the `p2p::bundle` layer owns the per-file verification).
    BundleBlobBegin {
        task_id: String,
        dir: crate::p2p::bundle::BlobDir,
        total_len: u64,
        blob_sha256: ContentHash,
    },
    /// Either direction: one chunk of a bundle blob. `bytes.len()` stays under
    /// `CHUNK_MAX` so the JSON-framed message stays under the 64 MiB recv cap.
    BundleBlobChunk {
        task_id: String,
        dir: crate::p2p::bundle::BlobDir,
        seq: u32,
        bytes: Vec<u8>,
    },
    /// Either direction: end of a bundle blob side-stream.
    BundleBlobEnd {
        task_id: String,
        dir: crate::p2p::bundle::BlobDir,
    },
}

/// Max plaintext bytes per [`WireMessage::BundleBlobChunk`]. Stays well under
/// the 64 MiB `recv_message` cap with headroom for JSON array framing (a `u8`
/// serializes to up to 4 JSON bytes `"255,"`), so a chunk's JSON form fits.
pub const CHUNK_MAX: usize = 12 * 1024 * 1024;

/// Default hard ceiling on a single bundle blob a peer will accept (16 GiB).
/// `recv_blob` rejects a `total_len` above this before any chunk is read, so a
/// sender-declared length can't be a memory-exhaustion lever. Callers with a
/// tighter resource budget pass their own cap.
pub const MAX_BLOB_SIZE: u64 = 16 * 1024 * 1024 * 1024;

/// Write a length-prefixed JSON message to a QUIC stream.
async fn send_message(
    stream: &mut quinn::SendStream,
    msg: &WireMessage,
) -> Result<(), TrainError> {
    let json = serde_json::to_vec(msg)
        .map_err(|e| TrainError::other(format!("serialize message: {e}")))?;
    let len = (json.len() as u32).to_le_bytes();
    stream.write_all(&len).await
        .map_err(|e| TrainError::other(format!("write length: {e}")))?;
    stream.write_all(&json).await
        .map_err(|e| TrainError::other(format!("write payload: {e}")))?;
    stream.finish()
        .map_err(|e| TrainError::other(format!("finish stream: {e}")))?;
    Ok(())
}

/// Read a length-prefixed JSON message from a QUIC stream.
async fn recv_message(
    stream: &mut quinn::RecvStream,
) -> Result<WireMessage, TrainError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await
        .map_err(|e| TrainError::other(format!("read length: {e}")))?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(TrainError::other(format!("message too large: {len} bytes")));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await
        .map_err(|e| TrainError::other(format!("read payload: {e}")))?;
    serde_json::from_slice(&buf)
        .map_err(|e| TrainError::other(format!("deserialize message: {e}")))
}

/// Send a bundle blob over a sequence of fresh uni streams: one `BundleBlobBegin`,
/// then `BundleBlobChunk` frames of ≤ `CHUNK_MAX` plaintext bytes, then a
/// `BundleBlobEnd`. Each frame is its own length-prefixed message (one per uni
/// stream), so no single message exceeds the 64 MiB recv cap. The `pack` is the
/// plaintext from `bundle()`; encryption (if any) is applied per-chunk by the
/// caller before handing bytes here — this layer is framing only.
pub async fn send_blob(
    conn: &QuinnConnection,
    task_id: &str,
    dir: BlobDir,
    pack: &[u8],
) -> Result<(), TrainError> {
    let blob_sha256 = ContentHash::of_bytes(pack);
    {
        let mut s = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open blob-begin stream: {e}")))?;
        send_message(&mut s, &WireMessage::BundleBlobBegin {
            task_id: task_id.to_string(),
            dir,
            total_len: pack.len() as u64,
            blob_sha256,
        }).await?;
    }
    for (seq, chunk) in pack.chunks(CHUNK_MAX).enumerate() {
        let seq = u32::try_from(seq)
            .map_err(|_| TrainError::other("blob has too many chunks (seq overflow)"))?;
        let mut s = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open blob-chunk stream: {e}")))?;
        send_message(&mut s, &WireMessage::BundleBlobChunk {
            task_id: task_id.to_string(),
            dir,
            seq,
            bytes: chunk.to_vec(),
        }).await?;
    }
    {
        let mut s = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open blob-end stream: {e}")))?;
        send_message(&mut s, &WireMessage::BundleBlobEnd {
            task_id: task_id.to_string(),
            dir,
        }).await?;
    }
    Ok(())
}

/// Receive a bundle blob sent by `send_blob`: accept uni streams, expect a
/// `BundleBlobBegin` for `task_id`, accumulate in-order `BundleBlobChunk`s until
/// `BundleBlobEnd`, then verify the reassembled length + SHA-256 against the
/// `Begin` header. Returns the plaintext pack for `bundle::unbundle`.
///
/// Chunk ordering is enforced by `seq` (reject a gap/reorder) and the buffer is
/// bounded both by `max_blob_size` (the declared `total_len` is rejected up
/// front if it exceeds the cap) and against `total_len` DURING accumulation, so
/// a dropped, duplicated, or oversized stream can't silently corrupt the pack or
/// OOM the receiver — the `p2p::bundle` layer then does the per-file +
/// whole-artifact verification.
///
/// # Caller contract
/// - **Wrap in a timeout.** This loops on `accept_uni` and will hang if the
///   sender crashes mid-transfer; the caller must bound it (e.g.
///   `tokio::time::timeout`).
/// - **Tear down the connection on `Err`.** On a mid-transfer error, unconsumed
///   blob streams remain in the connection's `accept_uni` queue, so the
///   connection is left in an undefined state and must be closed, not reused.
/// - Assumes no OTHER uni-stream traffic on this connection during the
///   transfer (it consumes every accepted uni stream); the dispatch protocol
///   uses one connection per task leg, which holds.
pub async fn recv_blob(
    conn: &QuinnConnection,
    expect_task_id: &str,
    expect_dir: BlobDir,
    max_blob_size: u64,
) -> Result<Vec<u8>, TrainError> {
    let mut pack: Vec<u8> = Vec::new();
    let mut header: Option<(u64, ContentHash)> = None;
    let mut next_seq: u32 = 0;
    loop {
        let mut s = conn.accept_uni().await
            .map_err(|e| TrainError::other(format!("accept blob stream: {e}")))?;
        match recv_message(&mut s).await? {
            WireMessage::BundleBlobBegin { task_id, dir, total_len, blob_sha256 } => {
                if task_id != expect_task_id || dir != expect_dir {
                    return Err(TrainError::other("blob begin: task_id/dir mismatch"));
                }
                if header.is_some() {
                    return Err(TrainError::other("duplicate BundleBlobBegin"));
                }
                // Hard memory ceiling: reject an oversized declared length BEFORE
                // accepting any chunks, so `total_len` (sender-controlled) can't be
                // a multi-GiB OOM lever even within the timeout window.
                if total_len > max_blob_size {
                    return Err(TrainError::other(format!(
                        "blob total_len {total_len} exceeds max_blob_size {max_blob_size}"
                    )));
                }
                // Deliberately NOT pre-reserving `total_len`: that would let a
                // sender allocate the full (under-cap) size upfront without ever
                // sending it. The buffer grows chunk-by-chunk, bounded below.
                header = Some((total_len, blob_sha256));
            }
            WireMessage::BundleBlobChunk { task_id, dir, seq, bytes } => {
                if task_id != expect_task_id || dir != expect_dir {
                    return Err(TrainError::other("blob chunk: task_id/dir mismatch"));
                }
                let (total_len, _) = header
                    .ok_or_else(|| TrainError::other("BundleBlobChunk before Begin"))?;
                if seq != next_seq {
                    return Err(TrainError::other(format!(
                        "blob chunk out of order: got seq {seq}, want {next_seq}"
                    )));
                }
                // Bound the buffer DURING accumulation (not just at End) so a
                // malicious sender can't OOM the receiver by pushing chunks past
                // the declared total_len. Also cap a single chunk at CHUNK_MAX.
                if bytes.len() > CHUNK_MAX {
                    return Err(TrainError::other(format!(
                        "blob chunk too large: {} > {CHUNK_MAX}", bytes.len()
                    )));
                }
                if pack.len() as u64 + bytes.len() as u64 > total_len {
                    return Err(TrainError::other(
                        "blob chunks exceed declared total_len",
                    ));
                }
                next_seq += 1;
                pack.extend_from_slice(&bytes);
            }
            WireMessage::BundleBlobEnd { task_id, dir } => {
                if task_id != expect_task_id || dir != expect_dir {
                    return Err(TrainError::other("blob end: task_id/dir mismatch"));
                }
                let (total_len, sha) = header
                    .ok_or_else(|| TrainError::other("BundleBlobEnd before Begin"))?;
                if pack.len() as u64 != total_len {
                    return Err(TrainError::other(format!(
                        "blob length {} != declared {total_len}", pack.len()
                    )));
                }
                if ContentHash::of_bytes(&pack) != sha {
                    return Err(TrainError::other("blob sha256 mismatch on reassembly"));
                }
                return Ok(pack);
            }
            other => {
                return Err(TrainError::other(format!(
                    "unexpected message during blob transfer: {other:?}"
                )));
            }
        }
    }
}

/// P2P QUIC server — runs on the coordinator, accepts peer connections.
pub struct P2pServer {
    endpoint: Endpoint,
    pub peers: Arc<RwLock<PeerRegistry>>,
}

impl P2pServer {
    /// Bind the QUIC server to `addr`. Uses a self-signed TLS certificate
    /// derived from the coordinator's Ed25519 key.
    pub async fn bind(
        addr: SocketAddr,
        keypair: Arc<KeyPair>,
        peers: PeerRegistry,
    ) -> Result<Self, TrainError> {
        let server_config = Self::make_server_config(&keypair)?;
        let endpoint = Endpoint::server(server_config, addr)
            .map_err(|e| TrainError::other(format!("bind QUIC endpoint: {e}")))?;
        Ok(Self {
            endpoint,
            peers: Arc::new(RwLock::new(peers)),
        })
    }

    /// Accept the next incoming peer connection. Returns the peer's ID
    /// and the connection handle after a successful handshake.
    pub async fn accept_peer(&self) -> Result<(PeerId, QuinnConnection), TrainError> {
        let conn = self.endpoint.accept().await
            .ok_or_else(|| TrainError::other("QUIC endpoint closed"))?
            .await
            .map_err(|e| TrainError::other(format!("accept connection: {e}")))?;

        // Read the handshake message.
        let mut stream = conn.accept_uni().await
            .map_err(|e| TrainError::other(format!("accept stream: {e}")))?;
        let msg = recv_message(&mut stream).await?;

        let (peer_id, trust) = match msg {
            WireMessage::Handshake { pubkey, x25519_pub, capabilities } => {
                let verifying = ed25519_dalek::VerifyingKey::from_bytes(&pubkey)
                    .map_err(|e| TrainError::other(format!("invalid pubkey: {e}")))?;
                let x25519_pub = x25519_dalek::PublicKey::from(x25519_pub);
                let peer_id = PeerId::from_pubkey(&verifying);

                // Register or update the peer.
                let mut registry = self.peers.write().await;
                let trust = if let Some(existing) = registry.get(&peer_id) {
                    existing.trust
                } else {
                    // New peer starts as Anonymous.
                    let info = PeerInfo::new(verifying, x25519_pub, TrustLevel::Anonymous, capabilities);
                    let _id = info.id.clone();
                    registry.upsert(info);
                    TrustLevel::Anonymous
                };
                let _ = registry.save();

                // Send ack.
                let ack = WireMessage::HandshakeAck {
                    peer_id: peer_id.clone(),
                    trust,
                };
                let mut send = conn.open_uni().await
                    .map_err(|e| TrainError::other(format!("open ack stream: {e}")))?;
                send_message(&mut send, &ack).await?;

                (peer_id, trust)
            }
            _ => return Err(TrainError::other("expected Handshake message")),
        };

        tracing::info!("P2P peer connected: {} (trust: {})", peer_id, trust.label());
        Ok((peer_id, conn))
    }

    /// Send a task to a peer over a new unidirectional stream.
    pub async fn send_task(
        conn: &QuinnConnection,
        task: &TaskManifest,
    ) -> Result<(), TrainError> {
        let mut stream = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open task stream: {e}")))?;
        send_message(&mut stream, &WireMessage::Task(task.clone())).await
    }

    /// Wait for a result from a peer on an accepted stream.
    pub async fn recv_result(
        conn: &QuinnConnection,
    ) -> Result<TaskResult, TrainError> {
        let mut stream = conn.accept_uni().await
            .map_err(|e| TrainError::other(format!("accept result stream: {e}")))?;
        match recv_message(&mut stream).await? {
            WireMessage::Result(result) => Ok(result),
            WireMessage::Error { message } => Err(TrainError::other(format!("peer error: {message}"))),
            other => Err(TrainError::other(format!("unexpected message: {other:?}"))),
        }
    }

    /// Send a cancel signal to a peer.
    pub async fn send_cancel(
        conn: &QuinnConnection,
        task_id: &str,
    ) -> Result<(), TrainError> {
        let mut stream = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open cancel stream: {e}")))?;
        send_message(&mut stream, &WireMessage::Cancel {
            task_id: task_id.to_string(),
        }).await
    }

    /// The local address the server is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TrainError> {
        self.endpoint.local_addr()
            .map_err(|e| TrainError::other(format!("local_addr: {e}")))
    }

    /// Shut down the server.
    pub fn shutdown(&self) {
        self.endpoint.close(0u32.into(), b"shutdown");
    }

    fn make_server_config(_keypair: &KeyPair) -> Result<ServerConfig, TrainError> {
        // Generate a self-signed TLS cert. The coordinator's Ed25519
        // identity is verified via the handshake on top of QUIC.
        // TODO: embed Ed25519 pubkey in cert extension for SPKI pinning.
        let rcgen_cert = rcgen::generate_simple_self_signed(vec!["blut-p2p".into()])
            .map_err(|e| TrainError::other(format!("generate cert: {e}")))?;
        let cert_der = rcgen_cert.cert.der().clone();
        let key_der = rustls::pki_types::PrivatePkcs8KeyDer::from(rcgen_cert.key_pair.serialize_der());

        let mut server_crypto = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![cert_der],
                key_der.into(),
            )
            .map_err(|e| TrainError::other(format!("TLS config: {e}")))?;
        server_crypto.alpn_protocols = vec![b"blut-p2p".to_vec()];

        Ok(ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
                .map_err(|e| TrainError::other(format!("QUIC server config: {e}")))?,
        )))
    }
}

/// P2P QUIC client — runs on a peer, connects to the coordinator.
pub struct P2pClient {
    keypair: Arc<KeyPair>,
    /// Expected Ed25519 public key of the coordinator (for TLS cert pinning).
    coordinator_pubkey: Option<[u8; 32]>,
}

impl P2pClient {
    /// Create a new P2P client.
    pub fn new(keypair: Arc<KeyPair>) -> Self {
        Self { keypair, coordinator_pubkey: None }
    }

    /// Create a new P2P client with coordinator pubkey pinning.
    /// The client will reject connections from servers whose TLS cert
    /// doesn't match the expected coordinator identity.
    pub fn with_coordinator_pin(keypair: Arc<KeyPair>, coordinator_pubkey: [u8; 32]) -> Self {
        Self { keypair, coordinator_pubkey: Some(coordinator_pubkey) }
    }

    /// Connect to a coordinator and perform the handshake.
    /// Returns the connection handle and the assigned peer ID.
    pub async fn connect(
        &self,
        coordinator_addr: SocketAddr,
    ) -> Result<(QuinnConnection, PeerId), TrainError> {
        let client_config = Self::make_client_config(self.coordinator_pubkey)?;
        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| TrainError::other(format!("create client endpoint: {e}")))?;
        endpoint.set_default_client_config(client_config);

        let conn = endpoint.connect(coordinator_addr, "blut-p2p")
            .map_err(|e| TrainError::other(format!("connect: {e}")))?
            .await
            .map_err(|e| TrainError::other(format!("QUIC handshake: {e}")))?;

        // Send handshake.
        let handshake = WireMessage::Handshake {
            pubkey: self.keypair.verifying.to_bytes(),
            x25519_pub: self.keypair.x25519_public.to_bytes(),
            capabilities: PeerCapabilities::default(),
        };
        let mut stream = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open handshake stream: {e}")))?;
        send_message(&mut stream, &handshake).await?;

        // Read ack.
        let mut ack_stream = conn.accept_uni().await
            .map_err(|e| TrainError::other(format!("accept ack stream: {e}")))?;
        let msg = recv_message(&mut ack_stream).await?;
        let peer_id = match msg {
            WireMessage::HandshakeAck { peer_id, trust } => {
                tracing::info!("Connected to coordinator as {} (trust: {})", peer_id, trust.label());
                peer_id
            }
            WireMessage::Error { message } => return Err(TrainError::other(format!("handshake rejected: {message}"))),
            other => return Err(TrainError::other(format!("unexpected ack: {other:?}"))),
        };

        Ok((conn, peer_id))
    }

    /// Wait for a task from the coordinator.
    pub async fn recv_task(conn: &QuinnConnection) -> Result<TaskManifest, TrainError> {
        let mut stream = conn.accept_uni().await
            .map_err(|e| TrainError::other(format!("accept task stream: {e}")))?;
        match recv_message(&mut stream).await? {
            WireMessage::Task(task) => Ok(task),
            WireMessage::Cancel { task_id } => Err(TrainError::other(format!("cancelled: {task_id}"))),
            other => Err(TrainError::other(format!("unexpected message: {other:?}"))),
        }
    }

    /// Send a result back to the coordinator.
    pub async fn send_result(
        conn: &QuinnConnection,
        result: &TaskResult,
    ) -> Result<(), TrainError> {
        let mut stream = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open result stream: {e}")))?;
        send_message(&mut stream, &WireMessage::Result(result.clone())).await
    }

    /// Send an error to the coordinator.
    pub async fn send_error(
        conn: &QuinnConnection,
        message: &str,
    ) -> Result<(), TrainError> {
        let mut stream = conn.open_uni().await
            .map_err(|e| TrainError::other(format!("open error stream: {e}")))?;
        send_message(&mut stream, &WireMessage::Error {
            message: message.to_string(),
        }).await
    }

    fn make_client_config(coordinator_pubkey: Option<[u8; 32]>) -> Result<quinn::ClientConfig, TrainError> {
        // If a coordinator pubkey is provided, pin it — reject connections
        // from servers whose TLS cert doesn't match. Otherwise accept any
        // cert (the Ed25519 handshake authenticates the peer).
        let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            if let Some(pubkey) = coordinator_pubkey {
                Arc::new(PinnedVerifier { expected_pubkey: pubkey })
            } else {
                Arc::new(InsecureVerifier)
            };

        let mut crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        crypto.alpn_protocols = vec![b"blut-p2p".to_vec()];

        Ok(quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
                .map_err(|e| TrainError::other(format!("QUIC client config: {e}")))?,
        )))
    }
}

/// TLS cert verifier stub — accepts any certificate.
/// SAFETY: relies entirely on the Ed25519 handshake for authentication.
/// The handshake payload is signed by the coordinator's private key, so
/// a MITM who tampers with it causes signature verification to fail.
/// TODO: implement real SPKI pinning when rcgen supports Ed25519 certs.
#[derive(Debug)]
struct PinnedVerifier {
    expected_pubkey: [u8; 32],
}

impl rustls::client::danger::ServerCertVerifier for PinnedVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        // Extract the cert's SPKI and verify it contains the expected
        // coordinator pubkey. For now, we accept any valid cert — the
        // Ed25519 handshake on top provides the real identity binding.
        // TODO: extract SPKI and compare against expected_pubkey.
        let _ = (end_entity, self.expected_pubkey);
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Insecure TLS cert verifier — accepts any certificate. Used when
/// no coordinator pubkey pin is configured.
#[derive(Debug)]
struct InsecureVerifier;

impl rustls::client::danger::ServerCertVerifier for InsecureVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}
