// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Peer identity and capabilities.
//!
//! A peer is identified by the blake3 hash of its Ed25519 public key.
//! The coordinator tracks each peer's trust level, hardware capabilities,
//! and reputation score.

use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};

use crate::p2p::trust::TrustLevel;

/// Unique peer identifier — blake3 hash of the peer's Ed25519 public key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PeerId(#[serde(with = "hex_serde")] pub [u8; 32]);

impl PeerId {
    /// Derive a PeerId from an Ed25519 public key.
    pub fn from_pubkey(pubkey: &VerifyingKey) -> Self {
        let hash = blake3::hash(&pubkey.to_bytes());
        Self(*hash.as_bytes())
    }

    /// Hex-encoded short form (first 8 bytes = 16 hex chars) for display.
    pub fn short(&self) -> String {
        faster_hex::hex_string(&self.0[..8])
    }
}

impl std::fmt::Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short())
    }
}

/// Hardware capabilities reported by a peer during registration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerCapabilities {
    /// CPU cores available for compute.
    pub cpu_cores: u32,
    /// System memory in GiB.
    pub memory_gib: u32,
    /// GPU model name (e.g. "NVIDIA RTX 4090"). None = CPU-only.
    pub gpu_model: Option<String>,
    /// GPU VRAM in GiB. None = CPU-only.
    pub gpu_vram_gib: Option<u32>,
}

impl Default for PeerCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_gib: 4,
            gpu_model: None,
            gpu_vram_gib: None,
        }
    }
}

/// Full peer record — identity, trust, capabilities, reputation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Unique peer ID (blake3 hash of pubkey).
    pub id: PeerId,
    /// Ed25519 public key (for signature verification).
    #[serde(with = "pubkey_serde")]
    pub pubkey: VerifyingKey,
    /// X25519 public key (for encryption / key exchange).
    #[serde(with = "x25519_serde")]
    pub x25519_pub: x25519_dalek::PublicKey,
    /// Trust level assigned by the coordinator.
    pub trust: TrustLevel,
    /// Hardware capabilities reported by the peer.
    pub capabilities: PeerCapabilities,
    /// Reputation score (0.0–1.0). Starts at 0.5.
    /// Updated by the coordinator after each task.
    pub reputation: f64,
    /// Total tasks completed successfully.
    pub tasks_completed: u64,
    /// Total tasks failed (timeout, bad hash, error).
    pub tasks_failed: u64,
    /// Last time the coordinator heard from this peer.
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

impl PeerInfo {
    /// Create a new peer record with default reputation.
    pub fn new(
        pubkey: VerifyingKey,
        x25519_pub: x25519_dalek::PublicKey,
        trust: TrustLevel,
        capabilities: PeerCapabilities,
    ) -> Self {
        Self {
            id: PeerId::from_pubkey(&pubkey),
            pubkey,
            x25519_pub,
            trust,
            capabilities,
            reputation: 0.5,
            tasks_completed: 0,
            tasks_failed: 0,
            last_seen: chrono::Utc::now(),
        }
    }

    /// Update reputation after a task completes. Exponential moving average:
    /// success → reputation moves toward 1.0, failure → toward 0.0.
    pub fn record_outcome(&mut self, success: bool) {
        if success {
            self.tasks_completed += 1;
            self.reputation = (self.reputation * 0.95) + 0.05;
        } else {
            self.tasks_failed += 1;
            self.reputation *= 0.95;
        }
        self.reputation = self.reputation.clamp(0.0, 1.0);
        self.last_seen = chrono::Utc::now();
    }

    /// Total tasks attempted.
    pub fn total_tasks(&self) -> u64 {
        self.tasks_completed + self.tasks_failed
    }

    /// Success rate (0.0–1.0). Returns 0.5 if no tasks yet (prior).
    pub fn success_rate(&self) -> f64 {
        let total = self.total_tasks();
        if total == 0 {
            return 0.5;
        }
        self.tasks_completed as f64 / total as f64
    }
}

/// Hex serde for PeerId's inner bytes.
mod hex_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        faster_hex::hex_string(bytes).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let hex: String = Deserialize::deserialize(d)?;
        let mut bytes = [0u8; 32];
        faster_hex::hex_decode(hex.as_bytes(), &mut bytes)
            .map_err(|e| serde::de::Error::custom(format!("invalid hex: {e}")))?;
        Ok(bytes)
    }
}

/// Serde for VerifyingKey (serialize as hex-encoded 32 bytes).
mod pubkey_serde {
    use ed25519_dalek::VerifyingKey;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(key: &VerifyingKey, s: S) -> Result<S::Ok, S::Error> {
        faster_hex::hex_string(&key.to_bytes()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<VerifyingKey, D::Error> {
        let hex: String = Deserialize::deserialize(d)?;
        let mut bytes = [0u8; 32];
        faster_hex::hex_decode(hex.as_bytes(), &mut bytes)
            .map_err(|e| serde::de::Error::custom(format!("invalid hex: {e}")))?;
        VerifyingKey::from_bytes(&bytes)
            .map_err(|e| serde::de::Error::custom(format!("invalid pubkey: {e}")))
    }
}

/// Serde for x25519_dalek::PublicKey (serialize as hex-encoded 32 bytes).
mod x25519_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(key: &x25519_dalek::PublicKey, s: S) -> Result<S::Ok, S::Error> {
        faster_hex::hex_string(&key.to_bytes()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<x25519_dalek::PublicKey, D::Error> {
        let hex: String = Deserialize::deserialize(d)?;
        let mut bytes = [0u8; 32];
        faster_hex::hex_decode(hex.as_bytes(), &mut bytes)
            .map_err(|e| serde::de::Error::custom(format!("invalid hex: {e}")))?;
        Ok(x25519_dalek::PublicKey::from(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::crypto::KeyPair;

    fn make_test_peer(trust: TrustLevel) -> PeerInfo {
        let kp = KeyPair::generate();
        PeerInfo::new(kp.verifying, kp.x25519_public, trust, PeerCapabilities::default())
    }

    #[test]
    fn peer_id_from_pubkey_deterministic() {
        let kp = KeyPair::generate();
        let id1 = PeerId::from_pubkey(&kp.verifying);
        let id2 = PeerId::from_pubkey(&kp.verifying);
        assert_eq!(id1, id2);
    }

    #[test]
    fn peer_id_different_keys() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::generate();
        let id1 = PeerId::from_pubkey(&kp1.verifying);
        let id2 = PeerId::from_pubkey(&kp2.verifying);
        assert_ne!(id1, id2);
    }

    #[test]
    fn peer_id_display_short() {
        let kp = KeyPair::generate();
        let id = PeerId::from_pubkey(&kp.verifying);
        let s = id.to_string();
        assert_eq!(s.len(), 16); // 8 bytes = 16 hex chars
    }

    #[test]
    fn peer_info_reputation_update() {
        let mut peer = make_test_peer(TrustLevel::Anonymous);
        assert_eq!(peer.reputation, 0.5);

        peer.record_outcome(true);
        assert!(peer.reputation > 0.5);
        assert_eq!(peer.tasks_completed, 1);

        peer.record_outcome(false);
        assert!(peer.reputation < 0.5 + 0.05); // moved down
        assert_eq!(peer.tasks_failed, 1);
    }

    #[test]
    fn peer_info_reputation_bounds() {
        let mut peer = make_test_peer(TrustLevel::Anonymous);

        for _ in 0..1000 {
            peer.record_outcome(true);
        }
        assert!(peer.reputation <= 1.0);
        assert!(peer.reputation > 0.99);

        for _ in 0..2000 {
            peer.record_outcome(false);
        }
        assert!(peer.reputation >= 0.0);
        assert!(peer.reputation < 0.01);
    }

    #[test]
    fn peer_info_success_rate() {
        let mut peer = make_test_peer(TrustLevel::Registered);
        assert_eq!(peer.success_rate(), 0.5); // prior

        peer.record_outcome(true);
        peer.record_outcome(true);
        peer.record_outcome(false);
        assert!((peer.success_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn peer_info_serialization_roundtrip() {
        let kp = KeyPair::generate();
        let peer = PeerInfo::new(kp.verifying, kp.x25519_public, TrustLevel::Trusted, PeerCapabilities {
            cpu_cores: 16,
            memory_gib: 64,
            gpu_model: Some("NVIDIA A100".into()),
            gpu_vram_gib: Some(80),
        });
        let json = serde_json::to_string(&peer).unwrap();
        let peer2: PeerInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(peer.id, peer2.id);
        assert_eq!(peer.pubkey.to_bytes(), peer2.pubkey.to_bytes());
        assert_eq!(peer.x25519_pub.to_bytes(), peer2.x25519_pub.to_bytes());
        assert_eq!(peer.trust, peer2.trust);
        assert_eq!(peer.capabilities.cpu_cores, peer2.capabilities.cpu_cores);
    }
}
