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
pub mod coordinator;
pub mod crypto;
pub mod dispatch;
pub mod job;
pub mod peer;
pub mod registry;
pub mod task;
pub mod transport;
pub mod trust;

pub use bundle::{bundle, unbundle, BlobDir, BundleError, BundleManifest};
pub use coordinator::Coordinator;
pub use crypto::{EncryptedPayload, KeyPair};
pub use dispatch::{DefaultDispatchPolicy, DispatchPolicy, DispatchVerdict};
pub use job::P2pJob;
pub use peer::{PeerCapabilities, PeerId, PeerInfo};
pub use registry::PeerRegistry;
pub use task::{ResourceRequest, TaskManifest, TaskResult};
pub use transport::{P2pClient, P2pServer};
pub use trust::{DataClass, DispatchMatrix, TrustLevel};
