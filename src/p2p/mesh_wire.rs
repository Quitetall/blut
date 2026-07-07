// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Symmetric bi-stream wire protocol for the mesh (ADR 0079 A2).
//!
//! The legacy [`transport`](crate::p2p::transport) protocol is role-locked:
//! workers only ever *send* `Handshake`/`Result` and *receive* `Task`, over
//! **uni-directional** streams whose direction is fixed by role. A symmetric
//! mesh — where any node dispatches to, and accepts work from, any other —
//! needs a request/response protocol that either side can initiate.
//!
//! This module defines that protocol: a versioned [`MeshFrame`] envelope
//! carried over a QUIC **bi-directional** stream (`open_bi()`), one
//! request+response per stream. Encoding is `[version:u8][len:u32-le][JSON]`
//! (JSON, not bincode, because [`TaskManifest`] carries a `serde_json::Value`
//! that bincode's `deserialize_any` rejects — and it matches the legacy frame
//! encoding).
//!
//! **Version negotiation.** A node advertises [`PROTOCOL_VERSION`] in its
//! [`MeshHello`]. A peer that speaks the legacy protocol advertises a lower
//! version (or none, over the uni-stream path); [`negotiate`] maps a peer's
//! advertised version to the [`Wire`] path to use, so v1 peers keep working
//! (the A3 node loop routes on that decision).
//!
//! Bulk bytes (bundle chunks) are NOT carried inline as JSON arrays — that
//! would 4× the payload. [`MeshFrame`] carries the chunk *control* frames
//! ([`MeshFrame::ChunkIndex`] / [`MeshFrame::ChunkRequest`]); the chunk bytes
//! themselves stream raw, wired in the D1.1 transport-v2 work.

use serde::{Deserialize, Serialize};

use crate::error::TrainError;
use crate::p2p::chunkstore::ChunkIndex;
use crate::p2p::peer::{PeerCapabilities, PeerId};
use crate::p2p::task::{TaskManifest, TaskResult};
use crate::p2p::trust::TrustLevel;

/// The mesh protocol version this build speaks. Bumped only on a
/// wire-incompatible change; additive [`MeshFrame`] variants do NOT bump it
/// (an old peer simply never sends/receives the new variant).
pub const PROTOCOL_VERSION: u8 = 2;

/// The lowest version whose *frame encoding* this build can still decode on a
/// bi-stream. Kept equal to `PROTOCOL_VERSION` until a real encoding change
/// forces a spread; the negotiation to the legacy uni-stream path is handled
/// by [`negotiate`], not by decoding an old bi-stream frame.
pub const MIN_BISTREAM_VERSION: u8 = 2;

/// Max JSON body of a single frame (mirrors the legacy `recv_message` cap).
const MAX_FRAME_BODY: usize = 64 * 1024 * 1024;

/// Accept a frame's version byte only if this build can decode it: at least
/// [`MIN_BISTREAM_VERSION`] (not the legacy uni-stream protocol) and at most
/// [`PROTOCOL_VERSION`] (a NEWER peer's frame may use an encoding we don't
/// understand, so reject it cleanly instead of feeding it to the v2 serde
/// codec and silently corrupting). Shared by the slice and stream decoders.
fn check_frame_version(version: u8) -> Result<(), TrainError> {
    if version < MIN_BISTREAM_VERSION {
        return Err(TrainError::other(format!(
            "unsupported frame version {version} (min {MIN_BISTREAM_VERSION}; \
             use the legacy uni-stream path)"
        )));
    }
    if version > PROTOCOL_VERSION {
        return Err(TrainError::other(format!(
            "frame version {version} too new to decode (max {PROTOCOL_VERSION})"
        )));
    }
    Ok(())
}

/// Node identity + capabilities advertised on connect. Carries the
/// `protocol_version` so peers can negotiate v1 (legacy uni-stream) vs v2
/// (this bi-stream protocol).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshHello {
    pub protocol_version: u8,
    pub pubkey: [u8; 32],
    pub x25519_pub: [u8; 32],
    pub capabilities: PeerCapabilities,
}

/// A symmetric mesh message. Sent in EITHER direction on a bi-stream: the
/// initiator writes a request variant and reads a response variant on the same
/// stream.
///
/// Variants reserved for later slices (`PeerExchange`/`Introduce` A4–A5,
/// `Status` D5, `FedRound`/`FedUpdate` D4) land WITH those slices rather than
/// as empty stubs — adding a variant is additive and does not bump
/// [`PROTOCOL_VERSION`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MeshFrame {
    /// Identity + capability advertisement (request).
    Hello(MeshHello),
    /// Response to `Hello`: the responder's assigned view of the peer.
    HelloAck { peer_id: PeerId, trust: TrustLevel },
    /// Liveness probe.
    Ping { nonce: u64 },
    /// Response to `Ping` echoing the nonce.
    Pong { nonce: u64 },
    /// Dispatch a task (scheduler → worker).
    Task(Box<TaskManifest>),
    /// A completed task's result (worker → scheduler).
    Result(Box<TaskResult>),
    /// Ask a running task to stop.
    Cancel { task_id: String },
    /// Offer the content-addressed index of a blob (data plane, D1.1).
    ChunkIndex(ChunkIndex),
    /// Request the chunks at these index positions (the receiver's missing set).
    ChunkRequest { needed: Vec<u32> },
    /// A signed peer-address gossip (A5). One-way — the responder just `Ack`s.
    PeerExchange(Box<crate::p2p::gossip::PeerExchange>),
    /// Generic acknowledgement.
    Ack,
    /// A protocol-level error (either direction).
    Error { message: String },
}

impl MeshFrame {
    /// Short static label for logs/metrics (no payload).
    pub fn kind(&self) -> &'static str {
        match self {
            MeshFrame::Hello(_) => "Hello",
            MeshFrame::HelloAck { .. } => "HelloAck",
            MeshFrame::Ping { .. } => "Ping",
            MeshFrame::Pong { .. } => "Pong",
            MeshFrame::Task(_) => "Task",
            MeshFrame::Result(_) => "Result",
            MeshFrame::Cancel { .. } => "Cancel",
            MeshFrame::ChunkIndex(_) => "ChunkIndex",
            MeshFrame::ChunkRequest { .. } => "ChunkRequest",
            MeshFrame::PeerExchange(_) => "PeerExchange",
            MeshFrame::Ack => "Ack",
            MeshFrame::Error { .. } => "Error",
        }
    }
}

/// Which wire path to use with a peer, decided from its advertised version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wire {
    /// This bi-stream `MeshFrame` protocol.
    V2Mesh,
    /// The legacy uni-stream `transport::WireMessage` protocol.
    V1Legacy,
}

/// Decide the wire path for a peer advertising `their_version`. A peer at or
/// above [`MIN_BISTREAM_VERSION`] speaks the mesh protocol; anything lower
/// falls back to the legacy uni-stream path (kept, not dropped), so a v1 peer
/// keeps working through the transition.
pub fn negotiate(their_version: u8) -> Wire {
    if their_version >= MIN_BISTREAM_VERSION {
        Wire::V2Mesh
    } else {
        Wire::V1Legacy
    }
}

/// Serialize a frame to its `[version][len][json]` wire bytes. Exposed for
/// unit tests + any non-QUIC transport; the QUIC path uses [`write_frame`].
pub fn encode_frame(frame: &MeshFrame) -> Result<Vec<u8>, TrainError> {
    let json =
        serde_json::to_vec(frame).map_err(|e| TrainError::other(format!("encode frame: {e}")))?;
    if json.len() > MAX_FRAME_BODY {
        return Err(TrainError::other(format!(
            "frame too large: {} > {MAX_FRAME_BODY}",
            json.len()
        )));
    }
    let mut out = Vec::with_capacity(1 + 4 + json.len());
    out.push(PROTOCOL_VERSION);
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    Ok(out)
}

/// Parse `[version][len][json]` bytes back into a frame, rejecting an
/// unsupported version or an over-cap length before allocating the body.
pub fn decode_frame(bytes: &[u8]) -> Result<MeshFrame, TrainError> {
    if bytes.len() < 5 {
        return Err(TrainError::other("frame too short for header"));
    }
    check_frame_version(bytes[0])?;
    let len = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
    if len > MAX_FRAME_BODY {
        return Err(TrainError::other(format!("frame too large: {len}")));
    }
    let body = bytes.get(5..5 + len).ok_or_else(|| {
        TrainError::other(format!(
            "frame truncated: need {len} body bytes, have {}",
            bytes.len() - 5
        ))
    })?;
    serde_json::from_slice(body).map_err(|e| TrainError::other(format!("decode frame: {e}")))
}

/// Write a frame to a QUIC send stream and finish it (signals request-complete
/// on a bi-stream's send half).
pub async fn write_frame(
    stream: &mut quinn::SendStream,
    frame: &MeshFrame,
) -> Result<(), TrainError> {
    let bytes = encode_frame(frame)?;
    stream
        .write_all(&bytes)
        .await
        .map_err(|e| TrainError::other(format!("write frame: {e}")))?;
    stream
        .finish()
        .map_err(|e| TrainError::other(format!("finish frame stream: {e}")))?;
    Ok(())
}

/// Read one frame from a QUIC recv stream: version byte, length prefix (capped
/// before allocation), then the JSON body.
///
/// Contract: ONE frame per stream. A stream carries exactly one frame (the
/// sender [`write_frame`]s + `finish()`es it), and each request opens a fresh
/// bi-stream, so there is no stream reuse to desync — any trailing bytes would
/// be on a stream that is dropped immediately after.
pub async fn read_frame(stream: &mut quinn::RecvStream) -> Result<MeshFrame, TrainError> {
    let mut version = [0u8; 1];
    stream
        .read_exact(&mut version)
        .await
        .map_err(|e| TrainError::other(format!("read frame version: {e}")))?;
    check_frame_version(version[0])?;
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| TrainError::other(format!("read frame length: {e}")))?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_BODY {
        return Err(TrainError::other(format!("frame too large: {len} bytes")));
    }
    let mut body = vec![0u8; len];
    stream
        .read_exact(&mut body)
        .await
        .map_err(|e| TrainError::other(format!("read frame body: {e}")))?;
    serde_json::from_slice(&body).map_err(|e| TrainError::other(format!("decode frame: {e}")))
}

/// Initiator side of a bi-stream exchange: open a bi-stream, write `request`,
/// read the response frame. Symmetric — any node can call this against any
/// connected peer.
pub async fn mesh_request(
    conn: &quinn::Connection,
    request: &MeshFrame,
) -> Result<MeshFrame, TrainError> {
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| TrainError::other(format!("open bi-stream: {e}")))?;
    write_frame(&mut send, request).await?;
    read_frame(&mut recv).await
}

/// Responder side: accept the next bi-stream and read its request frame,
/// returning the frame plus the send half to reply on. The caller handles the
/// request and writes a response via [`write_frame`].
pub async fn accept_request(
    conn: &quinn::Connection,
) -> Result<(MeshFrame, quinn::SendStream), TrainError> {
    let (send, mut recv) = conn
        .accept_bi()
        .await
        .map_err(|e| TrainError::other(format!("accept bi-stream: {e}")))?;
    let frame = read_frame(&mut recv).await?;
    Ok((frame, send))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::crypto::KeyPair;

    fn sample_hello() -> MeshFrame {
        let kp = KeyPair::generate();
        MeshFrame::Hello(MeshHello {
            protocol_version: PROTOCOL_VERSION,
            pubkey: kp.verifying.to_bytes(),
            x25519_pub: kp.x25519_public.to_bytes(),
            capabilities: PeerCapabilities::default(),
        })
    }

    #[test]
    fn encode_decode_round_trip_all_control_frames() {
        let frames = vec![
            sample_hello(),
            MeshFrame::Ping { nonce: 42 },
            MeshFrame::Pong { nonce: 42 },
            MeshFrame::Cancel {
                task_id: "t-1".into(),
            },
            MeshFrame::ChunkRequest {
                needed: vec![1, 3, 7],
            },
            MeshFrame::ChunkIndex(ChunkIndex {
                total_len: 0,
                chunk_hashes: Vec::new(),
            }),
            MeshFrame::PeerExchange(Box::new(crate::p2p::gossip::PeerExchange::create(
                &KeyPair::generate(),
                1000,
                Vec::new(),
            ))),
            MeshFrame::Ack,
            MeshFrame::Error {
                message: "boom".into(),
            },
        ];
        for f in &frames {
            let bytes = encode_frame(f).unwrap();
            assert_eq!(bytes[0], PROTOCOL_VERSION, "version byte prefix");
            let back = decode_frame(&bytes).unwrap();
            assert_eq!(back.kind(), f.kind(), "kind survives round-trip");
        }
    }

    #[test]
    fn decode_rejects_out_of_range_version_and_truncation() {
        let bytes = encode_frame(&MeshFrame::Ack).unwrap();
        // Downgrade the version byte (legacy peer) → rejected.
        let mut old = bytes.clone();
        old[0] = MIN_BISTREAM_VERSION - 1;
        assert!(decode_frame(&old).is_err());
        // A NEWER version we can't decode → rejected cleanly, not fed to the
        // v2 serde codec.
        let mut newer = bytes.clone();
        newer[0] = PROTOCOL_VERSION + 1;
        assert!(decode_frame(&newer).is_err());
        // Truncate the body → rejected, not a panic.
        assert!(decode_frame(&bytes[..bytes.len() - 1]).is_err());
        // Too short for even the header.
        assert!(decode_frame(&[2, 0, 0]).is_err());
    }

    #[test]
    fn negotiate_picks_mesh_or_legacy() {
        assert_eq!(negotiate(PROTOCOL_VERSION), Wire::V2Mesh);
        assert_eq!(negotiate(PROTOCOL_VERSION + 9), Wire::V2Mesh); // newer still meshes
        assert_eq!(negotiate(1), Wire::V1Legacy);
        assert_eq!(negotiate(0), Wire::V1Legacy);
    }

    /// End-to-end: the bi-stream request/response works over a real QUIC
    /// connection, in BOTH directions (the whole point of a symmetric mesh —
    /// the node that accepted the connection can also initiate a request back).
    #[tokio::test]
    async fn bistream_request_response_echoes_over_quic() {
        use crate::p2p::registry::PeerRegistry;
        use crate::p2p::transport::{P2pClient, P2pServer};
        use std::net::SocketAddr;
        use std::sync::Arc;

        let server_kp = Arc::new(KeyPair::generate());
        let dir = tempfile::tempdir().unwrap();
        let peers = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = P2pServer::bind(addr, server_kp.clone(), peers)
            .await
            .unwrap();
        let bound = server.local_addr().unwrap();

        // Server: accept the peer (legacy handshake), then serve one bi-stream
        // request by echoing a Ping's nonce back as a Pong.
        let server_task = tokio::spawn(async move {
            let (_peer, conn) = server.accept_peer().await.expect("accept_peer");
            let (req, mut send) = accept_request(&conn).await.expect("accept_request");
            let resp = match req {
                MeshFrame::Ping { nonce } => MeshFrame::Pong { nonce },
                other => MeshFrame::Error {
                    message: format!("unexpected {}", other.kind()),
                },
            };
            write_frame(&mut send, &resp).await.expect("write response");
            // Keep the connection alive until the client is done reading.
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), conn.closed()).await;
        });

        let client_kp = Arc::new(KeyPair::generate());
        let client = P2pClient::with_coordinator_pin(client_kp, server_kp.verifying.to_bytes());
        let (conn, _id) =
            tokio::time::timeout(std::time::Duration::from_secs(10), client.connect(bound))
                .await
                .expect("connect must not hang")
                .expect("mutual-auth connect");

        let resp = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            mesh_request(&conn, &MeshFrame::Ping { nonce: 99 }),
        )
        .await
        .expect("request must not hang")
        .expect("mesh_request");
        assert!(
            matches!(resp, MeshFrame::Pong { nonce: 99 }),
            "bi-stream echo returned the nonce, got {}",
            resp.kind()
        );

        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server_task).await;
    }
}
