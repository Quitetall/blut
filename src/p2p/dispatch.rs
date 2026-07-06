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
#[cfg(test)]
use crate::p2p::trust::TrustLevel;
use crate::p2p::trust::{DataClass, DispatchMatrix};

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
    /// a result from a peer. Checks both the Ed25519 signature and the
    /// output hash.
    fn verify_result(
        &self,
        result: &TaskResult,
        expected: &ContentHash,
        peer_pubkey: &ed25519_dalek::VerifyingKey,
    ) -> DispatchVerdict;
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
    /// Explicit per-stage data classifications. A stage NOT in this map
    /// classifies as `DataClass::Restricted` (fail-closed): its data can
    /// only reach Trusted peers until an operator explicitly declares it
    /// less sensitive via [`classify`](Self::classify).
    pub stage_classes: std::collections::HashMap<String, DataClass>,
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
            // Built-in connectivity probe (deterministic, Public, dependency-free).
            // Always dispatchable so `blut p2p serve --smoke-stage` works out of the
            // box on any worker. See crate::p2p::smoke.
            crate::p2p::smoke::SMOKE_STAGE.into(),
        ]);
        // The ONLY default classification: the smoke probe's payload is
        // synthetic by construction (no corpus data), so it is safely
        // Public — connectivity tests work out of the box against
        // anonymous peers. Every other stage touches corpus data whose
        // sensitivity the engine cannot know, so it inherits the
        // fail-closed Restricted default until the operator classifies it.
        let stage_classes = std::collections::HashMap::from([(
            crate::p2p::smoke::SMOKE_STAGE.to_string(),
            DataClass::Public,
        )]);
        Self {
            matrix,
            dispatchable_stages: dispatchable,
            stage_classes,
        }
    }

    /// Create with a custom set of dispatchable stages. Like [`new`](Self::new),
    /// only the smoke probe is pre-classified (Public — its payload is
    /// synthetic by construction, independent of the operator's stage set);
    /// every custom stage classifies Restricted until [`classify`](Self::classify)d.
    pub fn with_stages(matrix: DispatchMatrix, stages: HashSet<String>) -> Self {
        let stage_classes = std::collections::HashMap::from([(
            crate::p2p::smoke::SMOKE_STAGE.to_string(),
            DataClass::Public,
        )]);
        Self {
            matrix,
            dispatchable_stages: stages,
            stage_classes,
        }
    }

    /// Declare a stage's data classification (builder-style). Stages
    /// without a declaration classify as `Restricted` — fail-closed.
    pub fn classify(mut self, stage: impl Into<String>, class: DataClass) -> Self {
        self.stage_classes.insert(stage.into(), class);
        self
    }
}

impl DispatchPolicy for DefaultDispatchPolicy {
    fn is_dispatchable(&self, stage_name: &str) -> bool {
        self.dispatchable_stages.contains(stage_name)
    }

    fn classify_stage(&self, stage_name: &str, _args: &serde_json::Value) -> DataClass {
        // FAIL-CLOSED: an unclassified stage is treated as Restricted
        // (Trusted peers only). Defaulting to Public here would silently
        // ship potentially-clinical corpus data to anonymous peers the
        // moment a stage is marked dispatchable. Domain policies override
        // per stage via `classify` (or their own DispatchPolicy impl).
        self.stage_classes
            .get(stage_name)
            .copied()
            .unwrap_or(DataClass::Restricted)
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
            .max_by(|a, b| a.reputation.total_cmp(&b.reputation))
            .map(|p| p.id.clone())
    }

    fn verify_result(
        &self,
        result: &TaskResult,
        expected: &ContentHash,
        peer_pubkey: &ed25519_dalek::VerifyingKey,
    ) -> DispatchVerdict {
        // Verify Ed25519 signature first.
        let payload = result.sign_payload();
        if !crate::p2p::crypto::verify(peer_pubkey, &payload, &result.signature) {
            return DispatchVerdict::Reject("invalid Ed25519 signature".into());
        }
        // Then verify output hash.
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
        let mut peer = PeerInfo::new(
            kp.verifying,
            kp.x25519_public,
            trust,
            PeerCapabilities::default(),
        );
        peer.reputation = reputation;
        peer
    }

    fn make_gpu_peer(trust: TrustLevel, reputation: f64, vram: u32) -> PeerInfo {
        let kp = KeyPair::generate();
        let caps = PeerCapabilities {
            gpu_model: Some("RTX 4090".into()),
            gpu_vram_gib: Some(vram),
            ..Default::default()
        };
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
        // Built-in connectivity probe is always dispatchable.
        assert!(policy.is_dispatchable(crate::p2p::smoke::SMOKE_STAGE));
        assert!(!policy.is_dispatchable("train_joint"));
        assert!(!policy.is_dispatchable("train_snn"));
        assert!(!policy.is_dispatchable("train_l3_teacher"));
    }

    #[test]
    fn unclassified_stage_is_restricted_fail_closed() {
        // An unclassified stage must NOT default to Public — that would
        // ship potentially-clinical corpus data to anonymous peers.
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let args = serde_json::json!({});
        assert_eq!(
            policy.classify_stage("warm_fb_cache", &args),
            DataClass::Restricted
        );
        assert_eq!(
            policy.classify_stage("some_future_stage", &args),
            DataClass::Restricted
        );
        // The smoke probe's payload is synthetic by construction — the one
        // default Public classification, so out-of-the-box connectivity
        // tests still reach anonymous peers.
        assert_eq!(
            policy.classify_stage(crate::p2p::smoke::SMOKE_STAGE, &args),
            DataClass::Public
        );
    }

    #[test]
    fn classify_builder_overrides_default() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default())
            .classify("warm_fb_cache", DataClass::Internal);
        let args = serde_json::json!({});
        assert_eq!(
            policy.classify_stage("warm_fb_cache", &args),
            DataClass::Internal
        );
        // Everything else stays fail-closed.
        assert_eq!(
            policy.classify_stage("precompute_l3", &args),
            DataClass::Restricted
        );
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
        let resources = ResourceRequest {
            cpu_cores: 16,
            ..Default::default()
        };
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
        let resources = ResourceRequest {
            gpu: true,
            gpu_vram_gib: Some(16),
            ..Default::default()
        };
        let peers = vec![cpu_peer, gpu_peer.clone()];
        let selected = policy.select_peer("warm_fb_cache", &resources, DataClass::Public, &peers);
        assert_eq!(selected, Some(gpu_peer.id.clone()));
    }

    #[test]
    fn verify_result_matching_hash() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let kp = KeyPair::generate();
        let hash = ContentHash::of_bytes(&[42u8; 32]);
        let mut result = TaskResult {
            task_id: "test".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash: hash,
            encrypted_output: None,
            wall_time_ms: 1000,
            signature: kp.sign(b"placeholder"),
        };
        result.signature = kp.sign(&result.sign_payload());
        assert!(matches!(
            policy.verify_result(&result, &hash, &kp.verifying),
            DispatchVerdict::Accept
        ));
    }

    #[test]
    fn verify_result_mismatch() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let kp = KeyPair::generate();
        let expected = ContentHash::of_bytes(&[42u8; 32]);
        let actual = ContentHash::of_bytes(&[99u8; 32]);
        let mut result = TaskResult {
            task_id: "test".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash: actual,
            encrypted_output: None,
            wall_time_ms: 1000,
            signature: kp.sign(b"placeholder"),
        };
        result.signature = kp.sign(&result.sign_payload());
        assert!(matches!(
            policy.verify_result(&result, &expected, &kp.verifying),
            DispatchVerdict::Reject(_)
        ));
    }

    #[test]
    fn verify_result_bad_signature() {
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let kp = KeyPair::generate();
        let hash = ContentHash::of_bytes(&[42u8; 32]);
        let result = TaskResult {
            task_id: "test".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash: hash,
            encrypted_output: None,
            wall_time_ms: 1000,
            signature: kp.sign(b"wrong payload"),
        };
        // Signature was over "wrong payload", not sign_payload() → rejects.
        assert!(matches!(
            policy.verify_result(&result, &hash, &kp.verifying),
            DispatchVerdict::Reject(_)
        ));
    }

    #[test]
    fn verify_result_wrong_key() {
        // Signed by a key that is NOT the peer's registered key — must
        // reject even though the signed payload bytes are exactly right
        // (an impersonating peer can produce a valid-looking signature
        // with its own key; only the registered pubkey may verify).
        let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
        let kp = KeyPair::generate();
        let kp_other = KeyPair::generate();
        let hash = ContentHash::of_bytes(&[42u8; 32]);
        let mut result = TaskResult {
            task_id: "test".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash: hash,
            encrypted_output: None,
            wall_time_ms: 1000,
            signature: kp_other.sign(b"placeholder"),
        };
        result.signature = kp_other.sign(&result.sign_payload());
        assert!(matches!(
            policy.verify_result(&result, &hash, &kp.verifying),
            DispatchVerdict::Reject(_)
        ));
    }
}
