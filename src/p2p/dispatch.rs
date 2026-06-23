// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Dispatch policy — decides which DAG nodes go to peers vs run locally.
//!
//! The `DispatchPolicy` trait is consulted by the executor before spawning
//! each DAG node. If the policy returns `Some(peer_id)`, the node is
//! offloaded to that peer. If it returns `None`, the node runs locally.

use std::collections::HashSet;

use crate::framework::artifact::ContentHash;
use crate::p2p::peer::{PeerId, PeerInfo};
use crate::p2p::task::{ResourceRequest, TaskResult};
use crate::p2p::trust::{DataClass, DispatchMatrix};
#[cfg(test)]
use crate::p2p::trust::TrustLevel;

/// Trait for dispatch policies. Implementors decide whether a DAG node
/// should run locally or be offloaded to a peer.
pub trait DispatchPolicy: Send + Sync {
    /// Is this stage eligible for remote dispatch at all?
    /// Training stages should return false.
    fn is_dispatchable(&self, stage_name: &str) -> bool;

    /// What data class does this stage's input belong to?
    fn classify_stage(&self, stage_name: &str, args: &serde_json::Value) -> DataClass;

    /// Select a peer for this task. Returns `None` if no suitable peer
    /// is available (the node runs locally).
    fn select_peer(
        &self,
        stage_name: &str,
        resources: &ResourceRequest,
        data_class: DataClass,
        peers: &[PeerInfo],
    ) -> Option<PeerId>;

    /// Verify a task result. The coordinator calls this after receiving
    /// a result from a peer.
    fn verify_result(&self, result: &TaskResult, expected: &ContentHash) -> DispatchVerdict;
}

/// Verdict on a peer's task result.
#[derive(Clone, Debug)]
pub enum DispatchVerdict {
    /// Result accepted — output hash matches expected.
    Accept,
    /// Result rejected — output hash mismatch or invalid signature.
    Reject(String),
    /// Result rejected but retry on a different peer might help.
    RetryOnDifferentPeer,
}

/// The default dispatch policy: trust-matrix based, with a hardcoded list
/// of dispatchable stages.
pub struct DefaultDispatchPolicy {
    /// The trust matrix (data class × trust level → allowed).
    pub matrix: DispatchMatrix,
    /// Stage names eligible for remote dispatch.
    pub dispatchable_stages: HashSet<String>,
}

impl DefaultDispatchPolicy {
    /// Create with the default dispatchable stages (preprocessing, encoding,
    /// evaluation, export — NOT training).
    pub fn new(matrix: DispatchMatrix) -> Self {
        let dispatchable = HashSet::from([
            "warm_fb_cache".into(),
            "precompute_l3".into(),
            "precompute_fullband".into(),
            "encode_lma".into(),
            "pccp_gate_encoder".into(),
            "pccp_gate_snn".into(),
            "export_firmware".into(),
            "build_manifest".into(),
            "build_split_manifest".into(),
        ]);
        Self {
            matrix,
            dispatchable_stages: dispatchable,
        }
    }

    /// Create with a custom set of dispatchable stages.
    pub fn with_stages(matrix: DispatchMatrix, stages: HashSet<String>) -> Self {
        Self {
            matrix,
            dispatchable_stages: stages,
        }
    }
}

impl DispatchPolicy for DefaultDispatchPolicy {
    fn is_dispatchable(&self, stage_name: &str) -> bool {
        self.dispatchable_stages.contains(stage_name)
    }

    fn classify_stage(&self, _stage_name: &str, _args: &serde_json::Value) -> DataClass {
        // Default: all stages are Public. Override in a domain-specific policy.
        DataClass::Public
    }

    fn select_peer(
        &self,
        _stage_name: &str,
        resources: &ResourceRequest,
        data_class: DataClass,
        peers: &[PeerInfo],
    ) -> Option<PeerId> {
        // Filter by trust matrix + capability match.
        let candidates: Vec<&PeerInfo> = peers
            .iter()
            .filter(|p| self.matrix.can_dispatch(data_class, p.trust))
            .filter(|p| p.capabilities.cpu_cores >= resources.cpu_cores)
            .filter(|p| p.capabilities.memory_gib >= resources.memory_gib)
            .filter(|p| !resources.gpu || p.capabilities.gpu_model.is_some())
            .filter(|p| {
                resources
                    .gpu_vram_gib
                    .map(|v| p.capabilities.gpu_vram_gib.unwrap_or(0) >= v)
                    .unwrap_or(true)
            })
            .collect();

        // Pick the peer with the highest reputation.
        candidates
            .into_iter()
            .max_by(|a, b| a.reputation.partial_cmp(&b.reputation).unwrap())
            .map(|p| p.id.clone())
    }

    fn verify_result(&self, result: &TaskResult, expected: &ContentHash) -> DispatchVerdict {
        if result.output_hash == *expected {
            DispatchVerdict::Accept
        } else {
            DispatchVerdict::Reject(format!(
                "output hash mismatch: got {}, expected {}",
                result.output_hash.to_hex(),
                expected.to_hex()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::peer::PeerCapabilities;

    fn make_peer(trust: TrustLevel, reputation: f64) -> PeerInfo {
        let kp = KeyPair::generate();
        let mut peer = PeerInfo::new(kp.verifying, kp.x25519_public, trust, PeerCapabilities::default());
        peer.reputation = reputation;
        peer
    }

    fn make_gpu_peer(trust: TrustLevel, reputation: f64, vram: u32) -> PeerInfo {
        let kp = KeyPair::generate();
        let mut caps = PeerCapabilities::default();
        caps.gpu_model = Some("RTX 4090".into());
        caps.gpu_vram_gib = Some(vram);
        let mut peer = PeerInfo::new(kp.verifying, kp.x25519_public, trust, caps);
        peer.reputation = reputation;
        peer
    }

    #[test]
    fn default_dispatchable_stages() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        assert!(policy.is_dispatchable("warm_fb_cache"));
        assert!(policy.is_dispatchable("precompute_l3"));
        assert!(policy.is_dispatchable("pccp_gate_encoder"));
        assert!(!policy.is_dispatchable("train_joint"));
        assert!(!policy.is_dispatchable("train_snn"));
        assert!(!policy.is_dispatchable("train_l3_teacher"));
    }

    #[test]
    fn select_peer_respects_trust_matrix() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let anonymous = make_peer(TrustLevel::Anonymous, 0.9);
        let registered = make_peer(TrustLevel::Registered, 0.5);

        // Internal data → anonymous rejected, registered selected.
        let peers = vec![anonymous.clone(), registered.clone()];
        let resources = ResourceRequest::default();
        let selected = policy.select_peer("warm_fb_cache", &resources, DataClass::Internal, &peers);
        assert_eq!(selected, Some(registered.id.clone()));
    }

    #[test]
    fn select_peer_picks_highest_reputation() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let peer_a = make_peer(TrustLevel::Anonymous, 0.3);
        let peer_b = make_peer(TrustLevel::Anonymous, 0.9);
        let peer_c = make_peer(TrustLevel::Anonymous, 0.6);

        let peers = vec![peer_a, peer_b.clone(), peer_c];
        let resources = ResourceRequest::default();
        let selected = policy.select_peer("warm_fb_cache", &resources, DataClass::Public, &peers);
        assert_eq!(selected, Some(peer_b.id.clone()));
    }

    #[test]
    fn select_peer_respects_resource_requirements() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let small_peer = make_peer(TrustLevel::Anonymous, 0.9); // default: 1 core, 4 GiB

        // Need 16 cores.
        let resources = ResourceRequest { cpu_cores: 16, ..Default::default() };
        let peers = vec![small_peer.clone()];
        let selected = policy.select_peer("warm_fb_cache", &resources, DataClass::Public, &peers);
        assert!(selected.is_none()); // no suitable peer
    }

    #[test]
    fn select_peer_gpu_requirement() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let cpu_peer = make_peer(TrustLevel::Anonymous, 0.9);
        let gpu_peer = make_gpu_peer(TrustLevel::Anonymous, 0.8, 24);

        // Need GPU with 16 GiB VRAM.
        let resources = ResourceRequest { gpu: true, gpu_vram_gib: Some(16), ..Default::default() };
        let peers = vec![cpu_peer, gpu_peer.clone()];
        let selected = policy.select_peer("warm_fb_cache", &resources, DataClass::Public, &peers);
        assert_eq!(selected, Some(gpu_peer.id.clone()));
    }

    #[test]
    fn verify_result_matching_hash() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let kp = KeyPair::generate();
        let hash = ContentHash::of_bytes(&[42u8; 32]);
        let result = TaskResult {
            task_id: "test".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash: hash.clone(),
            encrypted_output: None,
            wall_time_secs: 1.0,
            signature: kp.sign(b"test"),
        };
        assert!(matches!(policy.verify_result(&result, &hash), DispatchVerdict::Accept));
    }

    #[test]
    fn verify_result_mismatch() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let kp = KeyPair::generate();
        let expected = ContentHash::of_bytes(&[42u8; 32]);
        let actual = ContentHash::of_bytes(&[99u8; 32]);
        let result = TaskResult {
            task_id: "test".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash: actual,
            encrypted_output: None,
            wall_time_secs: 1.0,
            signature: kp.sign(b"test"),
        };
        assert!(matches!(policy.verify_result(&result, &expected), DispatchVerdict::Reject(_)));
    }
}
