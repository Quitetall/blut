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

use quinn::{Connection as QuinnConnection, Endpoint, ServerConfig};
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
    HandshakeAck { peer_id: PeerId, trust: TrustLevel },
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
async fn send_message(stream: &mut quinn::SendStream, msg: &WireMessage) -> Result<(), TrainError> {
    let json = serde_json::to_vec(msg)
        .map_err(|e| TrainError::other(format!("serialize message: {e}")))?;
    let len = (json.len() as u32).to_le_bytes();
    stream
        .write_all(&len)
        .await
        .map_err(|e| TrainError::other(format!("write length: {e}")))?;
    stream
        .write_all(&json)
        .await
        .map_err(|e| TrainError::other(format!("write payload: {e}")))?;
    stream
        .finish()
        .map_err(|e| TrainError::other(format!("finish stream: {e}")))?;
    Ok(())
}

/// Read a length-prefixed JSON message from a QUIC stream.
async fn recv_message(stream: &mut quinn::RecvStream) -> Result<WireMessage, TrainError> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| TrainError::other(format!("read length: {e}")))?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(TrainError::other(format!("message too large: {len} bytes")));
    }
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .await
        .map_err(|e| TrainError::other(format!("read payload: {e}")))?;
    serde_json::from_slice(&buf).map_err(|e| TrainError::other(format!("deserialize message: {e}")))
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
        let mut s = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open blob-begin stream: {e}")))?;
        send_message(
            &mut s,
            &WireMessage::BundleBlobBegin {
                task_id: task_id.to_string(),
                dir,
                total_len: pack.len() as u64,
                blob_sha256,
            },
        )
        .await?;
    }
    for (seq, chunk) in pack.chunks(CHUNK_MAX).enumerate() {
        let seq = u32::try_from(seq)
            .map_err(|_| TrainError::other("blob has too many chunks (seq overflow)"))?;
        let mut s = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open blob-chunk stream: {e}")))?;
        send_message(
            &mut s,
            &WireMessage::BundleBlobChunk {
                task_id: task_id.to_string(),
                dir,
                seq,
                bytes: chunk.to_vec(),
            },
        )
        .await?;
    }
    {
        let mut s = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open blob-end stream: {e}")))?;
        send_message(
            &mut s,
            &WireMessage::BundleBlobEnd {
                task_id: task_id.to_string(),
                dir,
            },
        )
        .await?;
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
        let mut s = conn
            .accept_uni()
            .await
            .map_err(|e| TrainError::other(format!("accept blob stream: {e}")))?;
        match recv_message(&mut s).await? {
            WireMessage::BundleBlobBegin {
                task_id,
                dir,
                total_len,
                blob_sha256,
            } => {
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
            WireMessage::BundleBlobChunk {
                task_id,
                dir,
                seq,
                bytes,
            } => {
                if task_id != expect_task_id || dir != expect_dir {
                    return Err(TrainError::other("blob chunk: task_id/dir mismatch"));
                }
                let (total_len, _) =
                    header.ok_or_else(|| TrainError::other("BundleBlobChunk before Begin"))?;
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
                        "blob chunk too large: {} > {CHUNK_MAX}",
                        bytes.len()
                    )));
                }
                if pack.len() as u64 + bytes.len() as u64 > total_len {
                    return Err(TrainError::other("blob chunks exceed declared total_len"));
                }
                next_seq += 1;
                pack.extend_from_slice(&bytes);
            }
            WireMessage::BundleBlobEnd { task_id, dir } => {
                if task_id != expect_task_id || dir != expect_dir {
                    return Err(TrainError::other("blob end: task_id/dir mismatch"));
                }
                let (total_len, sha) =
                    header.ok_or_else(|| TrainError::other("BundleBlobEnd before Begin"))?;
                if pack.len() as u64 != total_len {
                    return Err(TrainError::other(format!(
                        "blob length {} != declared {total_len}",
                        pack.len()
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
    /// When true, accept peers that present NO client cert and skip binding the
    /// handshake `PeerId` to a TLS-authenticated key (the pre-ADR-0079
    /// self-asserted-identity behavior). Off by default: the mesh requires
    /// mutual TLS auth. `--allow-legacy-peers` turns it on for one transition
    /// release.
    allow_legacy: bool,
}

impl P2pServer {
    /// Bind the QUIC server to `addr`. Uses a self-signed TLS certificate
    /// derived from the coordinator's Ed25519 key.
    pub async fn bind(
        addr: SocketAddr,
        keypair: Arc<KeyPair>,
        peers: PeerRegistry,
    ) -> Result<Self, TrainError> {
        // Secure default: require mutual TLS auth (ADR 0079 A1).
        Self::bind_with_options(addr, keypair, peers, false).await
    }

    /// Like [`bind`](Self::bind), but `allow_legacy` accepts peers that present
    /// no client certificate (falling back to self-asserted identity). Used by
    /// the deprecated `serve` path during the transition; the mesh `node` binds
    /// with `allow_legacy = false`.
    pub async fn bind_with_options(
        addr: SocketAddr,
        keypair: Arc<KeyPair>,
        peers: PeerRegistry,
        allow_legacy: bool,
    ) -> Result<Self, TrainError> {
        let server_config = Self::make_server_config(&keypair, allow_legacy)?;
        let endpoint = Endpoint::server(server_config, addr)
            .map_err(|e| TrainError::other(format!("bind QUIC endpoint: {e}")))?;
        Ok(Self {
            endpoint,
            peers: Arc::new(RwLock::new(peers)),
            allow_legacy,
        })
    }

    /// Accept the next incoming peer connection. Returns the peer's ID
    /// and the connection handle after a successful handshake.
    pub async fn accept_peer(&self) -> Result<(PeerId, QuinnConnection), TrainError> {
        let conn = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| TrainError::other("QUIC endpoint closed"))?
            .await
            .map_err(|e| TrainError::other(format!("accept connection: {e}")))?;

        // The key TLS mutually authenticated for this connection (None only in
        // legacy mode, where the peer presented no client cert).
        let tls_pubkey = tls_authenticated_pubkey(&conn);
        if !self.allow_legacy && tls_pubkey.is_none() {
            return Err(TrainError::other(
                "peer presented no client certificate (mutual TLS required; \
                 use --allow-legacy-peers to accept legacy peers)",
            ));
        }

        // Read the handshake message.
        let mut stream = conn
            .accept_uni()
            .await
            .map_err(|e| TrainError::other(format!("accept stream: {e}")))?;
        let msg = recv_message(&mut stream).await?;

        let (peer_id, trust) = match msg {
            WireMessage::Handshake {
                pubkey,
                x25519_pub,
                capabilities,
            } => {
                // Bind the self-asserted handshake identity to the key TLS
                // actually authenticated (ADR 0079 A1). A peer can no longer
                // claim a `pubkey` it doesn't hold the private half of.
                if let Some(tls_key) = tls_pubkey
                    && tls_key != pubkey
                {
                    return Err(TrainError::other(
                        "handshake pubkey does not match the TLS-authenticated \
                         client identity (rejected)",
                    ));
                }
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
                    let info =
                        PeerInfo::new(verifying, x25519_pub, TrustLevel::Anonymous, capabilities);
                    let _id = info.id.clone();
                    registry.upsert(info);
                    TrustLevel::Anonymous
                };
                if let Err(e) = registry.save() {
                    tracing::warn!(
                        "p2p: failed to persist peer registry after registering {peer_id}: {e}"
                    );
                }

                // Send ack.
                let ack = WireMessage::HandshakeAck {
                    peer_id: peer_id.clone(),
                    trust,
                };
                let mut send = conn
                    .open_uni()
                    .await
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
    pub async fn send_task(conn: &QuinnConnection, task: &TaskManifest) -> Result<(), TrainError> {
        let mut stream = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open task stream: {e}")))?;
        send_message(&mut stream, &WireMessage::Task(task.clone())).await
    }

    /// Wait for a result from a peer on an accepted stream.
    pub async fn recv_result(conn: &QuinnConnection) -> Result<TaskResult, TrainError> {
        let mut stream = conn
            .accept_uni()
            .await
            .map_err(|e| TrainError::other(format!("accept result stream: {e}")))?;
        match recv_message(&mut stream).await? {
            WireMessage::Result(result) => Ok(result),
            WireMessage::Error { message } => {
                Err(TrainError::other(format!("peer error: {message}")))
            }
            other => Err(TrainError::other(format!("unexpected message: {other:?}"))),
        }
    }

    /// Send a cancel signal to a peer.
    pub async fn send_cancel(conn: &QuinnConnection, task_id: &str) -> Result<(), TrainError> {
        let mut stream = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open cancel stream: {e}")))?;
        send_message(
            &mut stream,
            &WireMessage::Cancel {
                task_id: task_id.to_string(),
            },
        )
        .await
    }

    /// The local address the server is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr, TrainError> {
        self.endpoint
            .local_addr()
            .map_err(|e| TrainError::other(format!("local_addr: {e}")))
    }

    /// Shut down the server.
    pub fn shutdown(&self) {
        self.endpoint.close(0u32.into(), b"shutdown");
    }

    fn make_server_config(
        keypair: &KeyPair,
        allow_legacy: bool,
    ) -> Result<ServerConfig, TrainError> {
        // Derive the TLS leaf cert's key pair FROM the coordinator's Ed25519
        // identity key (rather than an unrelated, freshly-random one) so the
        // cert's SubjectPublicKeyInfo IS `keypair.verifying`. This is what
        // lets `PinnedVerifier::verify_server_cert` (below) actually pin: it
        // extracts the presented cert's SPKI and compares it byte-for-byte
        // against `--coordinator-pubkey`, so the generate side here and the
        // verify side must agree on the same key material and encoding —
        // see `identity_cert`.
        let (cert_der, key_der) = identity_cert(keypair)?;

        // Mutual auth (ADR 0079 A1): require + verify a client cert unless the
        // operator opted into legacy no-client-auth for the transition.
        let builder = rustls::ServerConfig::builder();
        let mut server_crypto = if allow_legacy {
            builder.with_no_client_auth()
        } else {
            builder.with_client_cert_verifier(Arc::new(MeshClientVerifier))
        }
        .with_single_cert(vec![cert_der], key_der.into())
        .map_err(|e| TrainError::other(format!("TLS config: {e}")))?;
        server_crypto.alpn_protocols = vec![b"blut-p2p".to_vec()];

        Ok(ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
                .map_err(|e| TrainError::other(format!("QUIC server config: {e}")))?,
        )))
    }
}

/// Derive a self-signed TLS leaf certificate whose key pair — and therefore
/// whose `SubjectPublicKeyInfo` — IS `keypair`'s Ed25519 identity key,
/// encoded via the fixed RFC 8410 PKCS#8 layout (`ed25519_pkcs8_der`).
/// Shared by [`P2pServer::make_server_config`] (the real QUIC server config)
/// and by unit tests below, so the "generate side" encoding a test exercises
/// is provably the same one production ships — see the note on
/// `PinnedVerifier::verify_server_cert` about generate/verify agreement.
fn identity_cert(
    keypair: &KeyPair,
) -> Result<
    (
        rustls::pki_types::CertificateDer<'static>,
        rustls::pki_types::PrivatePkcs8KeyDer<'static>,
    ),
    TrainError,
> {
    let seed: [u8; 32] = keypair.to_bytes()[..32]
        .try_into()
        .map_err(|_| TrainError::other("KeyPair::to_bytes() did not return >= 32 bytes"))?;
    let pkcs8_der = ed25519_pkcs8_der(&seed);
    let cert_key_pair = rcgen::KeyPair::from_pkcs8_der_and_sign_algo(
        &rustls::pki_types::PrivatePkcs8KeyDer::from(pkcs8_der),
        &rcgen::PKCS_ED25519,
    )
    .map_err(|e| TrainError::other(format!("derive TLS keypair from identity: {e}")))?;
    let rcgen_cert = rcgen::CertificateParams::new(vec!["blut-p2p".into()])
        .map_err(|e| TrainError::other(format!("cert params: {e}")))?
        .self_signed(&cert_key_pair)
        .map_err(|e| TrainError::other(format!("generate cert: {e}")))?;
    let cert_der = rcgen_cert.der().clone();
    let key_der = rustls::pki_types::PrivatePkcs8KeyDer::from(cert_key_pair.serialize_der());
    Ok((cert_der, key_der))
}

/// Wrap a raw 32-byte Ed25519 private key seed in the fixed RFC 8410 §7 /
/// Appendix A PKCS#8 v1 DER encoding: a constant 16-byte prefix followed by
/// the 32-byte seed, with no attributes and no embedded public key ("v1,
/// unchecked" — `rcgen` decodes this via ring's
/// `Ed25519KeyPair::from_pkcs8_maybe_unchecked`, which accepts exactly this
/// shape). This lets [`rcgen::KeyPair::from_pkcs8_der_and_sign_algo`] build
/// a TLS certificate key pair directly from a [`KeyPair`]'s Ed25519 identity
/// without pulling in a `pkcs8`-encoding crate for one fixed-format wrapper.
fn ed25519_pkcs8_der(seed: &[u8; 32]) -> Vec<u8> {
    #[rustfmt::skip]
    const PREFIX: [u8; 16] = [
        0x30, 0x2e,                               // SEQUENCE, len 46
        0x02, 0x01, 0x00,                         // INTEGER version = 0
        0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, // AlgorithmIdentifier: OID 1.3.101.112 (id-Ed25519)
        0x04, 0x22,                               // OCTET STRING (outer "privateKey"), len 34
        0x04, 0x20,                               // OCTET STRING (inner CurvePrivateKey), len 32
    ];
    let mut der = Vec::with_capacity(PREFIX.len() + 32);
    der.extend_from_slice(&PREFIX);
    der.extend_from_slice(seed);
    der
}

/// Build the fixed RFC 8410 `SubjectPublicKeyInfo` DER encoding for a raw
/// 32-byte Ed25519 public key: a constant 12-byte prefix followed by the
/// 32-byte key. Used by [`PinnedVerifier::verify_server_cert`] to compare,
/// byte-for-byte, against the SPKI `rustls-webpki` parses out of the peer's
/// presented certificate. This is the public-key half of the SAME encoding
/// `ed25519_pkcs8_der`'s cert embeds, since both sides derive from a
/// `p2p::crypto::KeyPair`'s Ed25519 key — a legitimate coordinator's cert
/// (built by `identity_cert`) therefore matches byte-for-byte, while an
/// attacker's unrelated self-signed cert does not.
fn ed25519_spki_der(pubkey: &[u8; 32]) -> Vec<u8> {
    let mut der = Vec::with_capacity(ED25519_SPKI_PREFIX.len() + 32);
    der.extend_from_slice(&ED25519_SPKI_PREFIX);
    der.extend_from_slice(pubkey);
    der
}

/// The fixed 12-byte RFC 8410 SPKI prefix for an Ed25519 public key. A full
/// Ed25519 SPKI is exactly this prefix followed by the raw 32-byte key (44
/// bytes total).
#[rustfmt::skip]
const ED25519_SPKI_PREFIX: [u8; 12] = [
    0x30, 0x2a,                               // SEQUENCE, len 42
    0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, // AlgorithmIdentifier: OID 1.3.101.112 (id-Ed25519)
    0x03, 0x21, 0x00,                         // BIT STRING, len 33, 0 unused bits
];

/// Recover the raw 32-byte Ed25519 public key from a `SubjectPublicKeyInfo`
/// DER blob, or `None` if it isn't a well-formed Ed25519 SPKI. The inverse of
/// [`ed25519_spki_der`].
fn ed25519_pubkey_from_spki(spki: &[u8]) -> Option<[u8; 32]> {
    let plen = ED25519_SPKI_PREFIX.len();
    if spki.len() != plen + 32 || spki[..plen] != ED25519_SPKI_PREFIX {
        return None;
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&spki[plen..plen + 32]);
    Some(key)
}

/// Extract the Ed25519 identity that TLS mutually authenticated for `conn` —
/// i.e. the public key the peer PROVED it holds the private half of during the
/// handshake (via its client cert + `CertificateVerify`). `None` if the peer
/// presented no cert (legacy peer) or a non-Ed25519 cert. This is the value
/// `accept_peer` binds the handshake-asserted `PeerId` against, closing the
/// self-asserted-identity hole.
fn tls_authenticated_pubkey(conn: &QuinnConnection) -> Option<[u8; 32]> {
    let identity = conn.peer_identity()?;
    let certs = identity
        .downcast::<Vec<rustls::pki_types::CertificateDer<'static>>>()
        .ok()?;
    let leaf = certs.first()?;
    let cert = webpki::EndEntityCert::try_from(leaf).ok()?;
    ed25519_pubkey_from_spki(cert.subject_public_key_info().as_ref())
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
        Self {
            keypair,
            coordinator_pubkey: None,
        }
    }

    /// Create a new P2P client with coordinator pubkey pinning.
    /// The client will reject connections from servers whose TLS cert
    /// doesn't match the expected coordinator identity.
    pub fn with_coordinator_pin(keypair: Arc<KeyPair>, coordinator_pubkey: [u8; 32]) -> Self {
        Self {
            keypair,
            coordinator_pubkey: Some(coordinator_pubkey),
        }
    }

    /// Connect to a coordinator and perform the handshake.
    /// Returns the connection handle and the assigned peer ID.
    pub async fn connect(
        &self,
        coordinator_addr: SocketAddr,
    ) -> Result<(QuinnConnection, PeerId), TrainError> {
        let client_config = Self::make_client_config(&self.keypair, self.coordinator_pubkey)?;
        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| TrainError::other(format!("create client endpoint: {e}")))?;
        endpoint.set_default_client_config(client_config);

        let conn = endpoint
            .connect(coordinator_addr, "blut-p2p")
            .map_err(|e| TrainError::other(format!("connect: {e}")))?
            .await
            .map_err(|e| TrainError::other(format!("QUIC handshake: {e}")))?;

        // Send handshake.
        let handshake = WireMessage::Handshake {
            pubkey: self.keypair.verifying.to_bytes(),
            x25519_pub: self.keypair.x25519_public.to_bytes(),
            capabilities: PeerCapabilities::default(),
        };
        let mut stream = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open handshake stream: {e}")))?;
        send_message(&mut stream, &handshake).await?;

        // Read ack.
        let mut ack_stream = conn
            .accept_uni()
            .await
            .map_err(|e| TrainError::other(format!("accept ack stream: {e}")))?;
        let msg = recv_message(&mut ack_stream).await?;
        let peer_id = match msg {
            WireMessage::HandshakeAck { peer_id, trust } => {
                tracing::info!(
                    "Connected to coordinator as {} (trust: {})",
                    peer_id,
                    trust.label()
                );
                peer_id
            }
            WireMessage::Error { message } => {
                return Err(TrainError::other(format!("handshake rejected: {message}")));
            }
            other => return Err(TrainError::other(format!("unexpected ack: {other:?}"))),
        };

        Ok((conn, peer_id))
    }

    /// Wait for a task from the coordinator.
    pub async fn recv_task(conn: &QuinnConnection) -> Result<TaskManifest, TrainError> {
        let mut stream = conn
            .accept_uni()
            .await
            .map_err(|e| TrainError::other(format!("accept task stream: {e}")))?;
        match recv_message(&mut stream).await? {
            WireMessage::Task(task) => Ok(task),
            WireMessage::Cancel { task_id } => {
                Err(TrainError::other(format!("cancelled: {task_id}")))
            }
            other => Err(TrainError::other(format!("unexpected message: {other:?}"))),
        }
    }

    /// Send a result back to the coordinator.
    pub async fn send_result(
        conn: &QuinnConnection,
        result: &TaskResult,
    ) -> Result<(), TrainError> {
        let mut stream = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open result stream: {e}")))?;
        send_message(&mut stream, &WireMessage::Result(result.clone())).await
    }

    /// Send an error to the coordinator.
    pub async fn send_error(conn: &QuinnConnection, message: &str) -> Result<(), TrainError> {
        let mut stream = conn
            .open_uni()
            .await
            .map_err(|e| TrainError::other(format!("open error stream: {e}")))?;
        send_message(
            &mut stream,
            &WireMessage::Error {
                message: message.to_string(),
            },
        )
        .await
    }

    fn make_client_config(
        keypair: &KeyPair,
        coordinator_pubkey: Option<[u8; 32]>,
    ) -> Result<quinn::ClientConfig, TrainError> {
        // If a coordinator pubkey is provided, pin it — reject connections
        // from servers whose TLS cert doesn't match. Otherwise accept any
        // cert (the Ed25519 handshake authenticates the peer).
        let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            if let Some(pubkey) = coordinator_pubkey {
                Arc::new(PinnedVerifier {
                    expected_pubkey: pubkey,
                })
            } else {
                Arc::new(InsecureVerifier)
            };

        // Present OUR identity cert (ADR 0079 A1) so a mutual-auth server can
        // bind our PeerId to the TLS-proven key. Derived from our Ed25519
        // identity via the same `identity_cert` the server uses, so the SPKI
        // the server extracts equals our `keypair.verifying`.
        let (cert_der, key_der) = identity_cert(keypair)?;
        let mut crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_client_auth_cert(vec![cert_der], key_der.into())
            .map_err(|e| TrainError::other(format!("TLS client cert: {e}")))?;
        crypto.alpn_protocols = vec![b"blut-p2p".to_vec()];

        Ok(quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
                .map_err(|e| TrainError::other(format!("QUIC client config: {e}")))?,
        )))
    }
}

/// TLS cert verifier enforcing `--coordinator-pubkey` pinning.
///
/// SECURITY: `end_entity` is fully attacker-controlled in the MITM threat
/// model this verifier exists to defeat (rogue AP / ARP / DNS spoof
/// terminating the QUIC/TLS handshake), so cert parsing is delegated to
/// `rustls-webpki` — the same hardened DER/X.509 parser rustls's own default
/// verifier uses — rather than hand-rolled here. `verify_server_cert`
/// requires the presented leaf cert's `SubjectPublicKeyInfo` to be an EXACT
/// byte match for `expected_pubkey` (encoded via `ed25519_spki_der`); the
/// coordinator's cert is generated in `identity_cert` from the SAME Ed25519
/// identity key using the SAME fixed RFC 8410 encoding, so a legitimate
/// coordinator always matches and an impostor's unrelated self-signed cert
/// never does.
///
/// `verify_tls12_signature` / `verify_tls13_signature` matter just as much
/// as the SPKI check above: they verify the `CertificateVerify` handshake
/// message actually proves possession of the private key for `cert` (via
/// `rustls::crypto::verify_tls1{2,3}_signature`, which internally re-parses
/// `cert` and checks the signature against its SPKI). Without this, an
/// attacker could present a cert with byte-for-byte the correct (public,
/// non-secret) pinned SPKI and skip proving they hold the matching private
/// key — `verify_server_cert` alone is not sufficient authentication.
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
        let cert = webpki::EndEntityCert::try_from(end_entity).map_err(|_| {
            rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding)
        })?;
        let expected_spki = ed25519_spki_der(&self.expected_pubkey);
        if cert.subject_public_key_info().as_ref() != expected_spki.as_slice() {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Server-side TLS verifier that authenticates a peer's CLIENT certificate for
/// the symmetric mesh (ADR 0079 A1). Unlike [`PinnedVerifier`] (which pins ONE
/// expected identity on the client side), the server accepts ANY well-formed
/// self-signed Ed25519 client cert — it can't know who will dial ahead of time
/// — but *requires* one and verifies the handshake signature, so the presented
/// identity is CRYPTOGRAPHICALLY PROVEN rather than self-asserted. `accept_peer`
/// then reads that authenticated key via [`tls_authenticated_pubkey`] and binds
/// it to the handshake-claimed `PeerId`, closing the hole where the old
/// `with_no_client_auth` server trusted whatever pubkey a peer typed into its
/// `Handshake`.
///
/// `verify_tls1{2,3}_signature` is load-bearing exactly as in `PinnedVerifier`:
/// it proves the client holds the private key for the cert it presented, so an
/// attacker can't replay someone else's (public) cert.
///
/// REVOCATION: identity certs are self-signed and long-lived, so revocation is
/// NOT at the cert layer — it is the peer registry + trust matrix. A compromised
/// or retired key is handled by demoting/removing that `PeerId` (the dispatch
/// matrix then fail-closed-blocks it); cert-level short-lived-credential
/// rotation is a documented later extension (ADR 0079).
#[derive(Debug)]
struct MeshClientVerifier;

impl rustls::server::danger::ClientCertVerifier for MeshClientVerifier {
    fn root_hint_subjects(&self) -> &[rustls::DistinguishedName] {
        // Self-signed identity certs — no CA roots to advertise.
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::server::danger::ClientCertVerified, rustls::Error> {
        // Accept any cert that parses as a well-formed Ed25519 leaf. The
        // identity *binding* (does this key match the handshake?) is enforced
        // in `accept_peer`; here we only require a structurally valid cert so
        // `tls_authenticated_pubkey` has something to extract. A cert whose
        // SPKI isn't Ed25519 is refused up front.
        let cert = webpki::EndEntityCert::try_from(end_entity).map_err(|_| {
            rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding)
        })?;
        if ed25519_pubkey_from_spki(cert.subject_public_key_info().as_ref()).is_none() {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(rustls::server::danger::ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod pinned_verifier_tests {
    use super::*;
    use crate::p2p::registry::PeerRegistry;
    use rustls::client::danger::ServerCertVerifier as _;

    fn test_server_name() -> rustls::pki_types::ServerName<'static> {
        rustls::pki_types::ServerName::try_from("blut-p2p").unwrap()
    }

    /// Direct regression test for the reported bug: `verify_server_cert`
    /// used to do `let _ = (end_entity, self.expected_pubkey); Ok(...)` —
    /// i.e. accept ANY cert regardless of the pin. Prove the fixed version
    /// accepts a cert whose SPKI matches the pin and rejects one that
    /// doesn't, using the exact cert-generation helper (`identity_cert`)
    /// that `P2pServer::make_server_config` ships, so this test can't pass
    /// by exercising a different (and possibly out-of-sync) encoding than
    /// production uses.
    #[test]
    fn verify_server_cert_matches_pin_accepts_and_mismatch_rejects() {
        let coordinator = KeyPair::generate();
        let (cert_der, _key_der) = identity_cert(&coordinator).unwrap();
        let server_name = test_server_name();
        let now = rustls::pki_types::UnixTime::now();

        let correct_pin = PinnedVerifier {
            expected_pubkey: coordinator.verifying.to_bytes(),
        };
        assert!(
            correct_pin
                .verify_server_cert(&cert_der, &[], &server_name, &[], now)
                .is_ok(),
            "a cert whose SPKI matches the pinned coordinator pubkey must be accepted"
        );

        let impostor = KeyPair::generate();
        let wrong_pin = PinnedVerifier {
            expected_pubkey: impostor.verifying.to_bytes(),
        };
        let result = wrong_pin.verify_server_cert(&cert_der, &[], &server_name, &[], now);
        assert!(
            result.is_err(),
            "a cert whose SPKI does NOT match the pinned coordinator pubkey must be \
             rejected, not silently accepted like the pre-fix stub did"
        );
        assert!(
            matches!(result.unwrap_err(), rustls::Error::InvalidCertificate(_)),
            "rejection must be a hard `InvalidCertificate` handshake failure, not a \
             warning-and-continue"
        );
    }

    /// The accept/reject test above proves generate/verify agreement only
    /// implicitly (through a successful handshake). Assert it explicitly:
    /// the SPKI bytes `ed25519_spki_der` builds for comparison must
    /// byte-for-byte equal what `webpki` (the same parser
    /// `verify_server_cert` uses) actually extracts from a cert
    /// `identity_cert`/rcgen generated for the same key. If a future
    /// rustls-webpki or rcgen upgrade ever changes how either side encodes
    /// or re-serializes SPKI DER, this fails loudly here instead of as a
    /// silent pinning bypass.
    #[test]
    fn spki_encoding_matches_what_webpki_extracts_from_the_generated_cert() {
        let coordinator = KeyPair::generate();
        let (cert_der, _key_der) = identity_cert(&coordinator).unwrap();

        let cert = webpki::EndEntityCert::try_from(&cert_der).unwrap();
        let extracted_spki = cert.subject_public_key_info().as_ref().to_vec();
        let built_spki = ed25519_spki_der(&coordinator.verifying.to_bytes());

        assert_eq!(
            extracted_spki, built_spki,
            "ed25519_spki_der's output must match what webpki actually extracts \
             from a cert generated for the same key, or PinnedVerifier's exact-byte \
             comparison silently stops matching legitimate coordinators"
        );
    }

    /// A1 (mutual TLS): the SPKI extraction helper is the exact inverse of the
    /// builder, so the key the server binds equals the client's identity key.
    #[test]
    fn ed25519_pubkey_from_spki_round_trips() {
        let k = KeyPair::generate();
        let pk = k.verifying.to_bytes();
        let spki = ed25519_spki_der(&pk);
        assert_eq!(ed25519_pubkey_from_spki(&spki), Some(pk));
        // Garbage / wrong-length / wrong-prefix ⇒ None, never a partial key.
        assert_eq!(ed25519_pubkey_from_spki(b"too short"), None);
        assert_eq!(ed25519_pubkey_from_spki(&[0u8; 44]), None); // right len, wrong prefix
        let mut mangled = spki.clone();
        mangled.push(0);
        assert_eq!(ed25519_pubkey_from_spki(&mangled), None); // wrong len
    }

    /// A1: the server-side verifier accepts a real Ed25519 identity cert and
    /// refuses a structurally-valid cert whose key isn't Ed25519.
    #[test]
    fn mesh_client_verifier_requires_ed25519_cert() {
        use rustls::server::danger::ClientCertVerifier as _;
        let peer = KeyPair::generate();
        let (cert_der, _key) = identity_cert(&peer).unwrap();
        let now = rustls::pki_types::UnixTime::now();
        assert!(
            MeshClientVerifier
                .verify_client_cert(&cert_der, &[], now)
                .is_ok(),
            "a well-formed Ed25519 identity cert must be accepted"
        );
        // A non-Ed25519 self-signed cert (RSA/ECDSA) → rejected. Build a P-256
        // cert via rcgen's default (ECDSA) to exercise the non-Ed25519 path.
        let params = rcgen::CertificateParams::new(vec!["blut-p2p".into()]).unwrap();
        let ec_key = rcgen::KeyPair::generate().unwrap(); // ECDSA P-256 by default
        let ec_cert = params.self_signed(&ec_key).unwrap();
        let ec_der = ec_cert.der().clone();
        let res = MeshClientVerifier.verify_client_cert(&ec_der, &[], now);
        assert!(
            matches!(res, Err(rustls::Error::InvalidCertificate(_))),
            "a non-Ed25519 client cert must be a hard InvalidCertificate rejection, got {res:?}"
        );
    }

    /// `InsecureVerifier` (the no-pin, trust-on-first-use fallback used when
    /// `--coordinator-pubkey` is NOT passed) must stay deliberately
    /// permissive — this fix must not change behavior on that path.
    #[test]
    fn insecure_verifier_still_accepts_any_cert() {
        let coordinator = KeyPair::generate();
        let (cert_der, _key_der) = identity_cert(&coordinator).unwrap();
        let server_name = test_server_name();
        let now = rustls::pki_types::UnixTime::now();

        assert!(
            InsecureVerifier
                .verify_server_cert(&cert_der, &[], &server_name, &[], now)
                .is_ok(),
            "InsecureVerifier (no-pin mode) must remain permissive"
        );
    }

    async fn bind_loopback_server() -> (P2pServer, std::sync::Arc<KeyPair>) {
        let keypair = std::sync::Arc::new(KeyPair::generate());
        let dir = tempfile::tempdir().unwrap();
        let peers = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = P2pServer::bind(addr, keypair.clone(), peers).await.unwrap();
        // Keep the tempdir alive for the registry's lifetime by leaking it —
        // this is a short-lived unit test process, not a long-running one.
        std::mem::forget(dir);
        (server, keypair)
    }

    /// End-to-end companion to the direct unit test above: exercises the
    /// REAL documented flow (`P2pClient::with_coordinator_pin` connecting
    /// to a `P2pServer::bind`-ed coordinator over actual QUIC/TLS), proving
    /// the fix holds through the full handshake, not just in isolation.
    #[tokio::test]
    async fn connect_with_correct_pin_succeeds_end_to_end() {
        let (server, server_keypair) = bind_loopback_server().await;
        let addr = server.local_addr().unwrap();

        // Drain the app-level handshake so `P2pClient::connect` (which waits
        // for a `HandshakeAck` after the TLS handshake completes) doesn't
        // hang waiting for the coordinator side. Hold the accepted
        // `Connection` open (via `conn.closed()`) rather than letting it
        // drop the instant `accept_peer` returns: dropping the last
        // `Connection` handle tears the QUIC connection down immediately,
        // which can race the just-sent `HandshakeAck` bytes still in
        // flight and flake the test with "closed by peer" — unrelated to
        // the cert-pinning behavior under test.
        tokio::spawn(async move {
            if let Ok((_, conn)) = server.accept_peer().await {
                let _ =
                    tokio::time::timeout(std::time::Duration::from_secs(5), conn.closed()).await;
            }
        });

        let client_keypair = std::sync::Arc::new(KeyPair::generate());
        let client =
            P2pClient::with_coordinator_pin(client_keypair, server_keypair.verifying.to_bytes());

        let result = tokio::time::timeout(std::time::Duration::from_secs(10), client.connect(addr))
            .await
            .expect("connect() must not hang");
        assert!(
            result.is_ok(),
            "connecting with the CORRECT --coordinator-pubkey must succeed: {:?}",
            result.err()
        );
    }

    /// The other half: a wrong pin must fail the QUIC/TLS handshake itself
    /// (before any application-level exchange), and must fail fast rather
    /// than hang — this is what makes the bug exploitable (MITM presents an
    /// unrelated cert and the old code accepted it unconditionally).
    #[tokio::test]
    async fn connect_with_wrong_pin_fails_end_to_end() {
        let (server, _server_keypair) = bind_loopback_server().await;
        let addr = server.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = server.accept_peer().await;
        });

        let client_keypair = std::sync::Arc::new(KeyPair::generate());
        let wrong_pubkey = KeyPair::generate().verifying.to_bytes();
        let client = P2pClient::with_coordinator_pin(client_keypair, wrong_pubkey);

        let result = tokio::time::timeout(std::time::Duration::from_secs(10), client.connect(addr))
            .await
            .expect("connect() must not hang even on rejection");
        assert!(
            result.is_err(),
            "connecting with the WRONG --coordinator-pubkey must be rejected \
             (regression test for the PinnedVerifier stub that accepted any cert)"
        );
    }

    /// A1 end-to-end: a mutual-auth server binds the accepted `PeerId` to the
    /// key TLS actually authenticated. The `PeerId` `accept_peer` returns must
    /// be the client's identity — proving the handshake `pubkey` was checked
    /// against the client cert, not merely trusted as self-asserted.
    #[tokio::test]
    async fn mutual_auth_binds_peer_id_to_tls_identity() {
        let (server, server_keypair) = bind_loopback_server().await;
        let addr = server.local_addr().unwrap();

        let accept = tokio::spawn(async move {
            let (peer_id, conn) = server.accept_peer().await?;
            // Hold the connection open briefly so the ack lands (see the
            // correct-pin test's rationale).
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), conn.closed()).await;
            Ok::<_, TrainError>(peer_id)
        });

        let client_keypair = std::sync::Arc::new(KeyPair::generate());
        let expected = PeerId::from_pubkey(&client_keypair.verifying);
        let client =
            P2pClient::with_coordinator_pin(client_keypair, server_keypair.verifying.to_bytes());
        let (_conn, client_side_id) =
            tokio::time::timeout(std::time::Duration::from_secs(10), client.connect(addr))
                .await
                .expect("connect must not hang")
                .expect("mutual-auth connect must succeed");

        let bound = tokio::time::timeout(std::time::Duration::from_secs(10), accept)
            .await
            .expect("accept must not hang")
            .unwrap()
            .expect("accept_peer must succeed under mutual auth");
        assert_eq!(
            bound, expected,
            "server bound the TLS-authenticated identity"
        );
        assert_eq!(client_side_id, expected, "client's own view agrees");
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
