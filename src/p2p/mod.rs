// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! P2P distributed compute — dispatch DAG nodes to peer GPUs over the
//! network with encrypted inputs and content-addressed verification.
//!
//! BLUT is a DAG orchestrator. This module extends "where a node runs"
//! from local / Slurm / Ray to a P2P network of GPU peers. The trust
//! model ensures peers cannot steal model weights or data; the
//! content-addressed cache ensures peer output is correct.
//!
//! Gated behind `#[cfg(feature = "p2p")]`.

pub mod bundle;
pub mod chunkstore;
pub mod coordinator;
pub mod crypto;
pub mod dispatch;
pub mod gossip;
pub mod introduction;
pub mod job;
pub mod mesh_wire;
pub mod node;
pub mod peer;
pub mod peer_exec;
pub mod privacy;
pub mod registry;
pub mod smoke;
pub mod task;
pub mod transport;
pub mod trust;

pub use bundle::{BlobDir, BundleError, BundleManifest, bundle, unbundle};
pub use chunkstore::{ChunkError, ChunkIndex, ChunkStore};
pub use coordinator::Coordinator;
pub use crypto::{EncryptedPayload, KeyPair};
pub use dispatch::{DefaultDispatchPolicy, DispatchPolicy, DispatchVerdict};
pub use gossip::{PeerExchange, PeerRecord};
pub use introduction::{IntroError, Introduction, MAX_INTRODUCED_TRUST};
pub use job::P2pJob;
pub use mesh_wire::{
    MeshFrame, MeshHello, PROTOCOL_VERSION, Wire, accept_request, mesh_request, negotiate,
};
pub use node::{MeshNode, MeshTaskRunner, NodeCapabilities};
pub use peer::{PeerCapabilities, PeerId, PeerInfo};
pub use privacy::{DpConfig, GradientStatus, PrivacyLedger, gate_gradient_dispatch};
pub use registry::PeerRegistry;
pub use task::{ResourceRequest, TaskManifest, TaskResult};
pub use transport::{P2pClient, P2pServer};
pub use trust::{DataClass, DispatchMatrix, TrustLevel};
