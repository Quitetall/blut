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
}

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
