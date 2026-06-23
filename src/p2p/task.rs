// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! P2P task manifest and result types.
//!
//! A `TaskManifest` is what the coordinator sends to a peer: the stage
//! to execute, its inputs (optionally encrypted), resource requirements,
//! and a timeout. A `TaskResult` is what the peer sends back: the
//! output hash and (optionally) encrypted output data.

use ed25519_dalek::Signature;
use serde::{Deserialize, Serialize};

use crate::framework::artifact::ContentHash;
use crate::p2p::crypto::EncryptedPayload;
use crate::p2p::peer::PeerId;
use crate::p2p::trust::DataClass;

/// A task to be executed by a remote peer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskManifest {
    /// Unique task identifier: "blut-<job>-<node_idx>".
    pub task_id: String,
    /// Coordinator's peer ID (so the peer knows who sent this).
    pub coordinator_id: PeerId,
    /// Name of the stage to execute (must match a known stage).
    pub stage_name: String,
    /// Stage schema version.
    pub stage_schema: u32,
    /// Content hash of the input artifact.
    pub input_hash: ContentHash,
    /// Content hash of the serialized args.
    pub args_hash: ContentHash,
    /// JSON-encoded stage args.
    pub args: serde_json::Value,
    /// Resource requirements for this task.
    pub resources: ResourceRequest,
    /// Data sensitivity classification.
    pub data_class: DataClass,
    /// Maximum wall-clock time (seconds) for the task.
    pub timeout_secs: u64,
    /// Encrypted input data (None for Public data on a shared filesystem).
    pub encrypted_input: Option<EncryptedPayload>,
    /// Ed25519 signature over (task_id + stage_name + input_hash + args_hash).
    /// The peer verifies this to confirm the task came from the coordinator.
    #[serde(with = "sig_serde")]
    pub signature: Signature,
}

/// Resource requirements for a task.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// CPU cores needed.
    pub cpu_cores: u32,
    /// System memory in GiB.
    pub memory_gib: u32,
    /// Whether a GPU is required.
    pub gpu: bool,
    /// Minimum GPU VRAM in GiB (None = any GPU).
    pub gpu_vram_gib: Option<u32>,
}

impl Default for ResourceRequest {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_gib: 4,
            gpu: false,
            gpu_vram_gib: None,
        }
    }
}

/// The data the coordinator signs to authenticate a task.
fn sign_payload(task_id: &str, stage_name: &str, input_hash: &ContentHash, args_hash: &ContentHash) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(task_id.as_bytes());
    buf.push(0);
    buf.extend_from_slice(stage_name.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&input_hash.0);
    buf.extend_from_slice(&args_hash.0);
    buf
}

impl TaskManifest {
    /// Build the signing payload for this manifest.
    pub fn sign_payload(&self) -> Vec<u8> {
        sign_payload(&self.task_id, &self.stage_name, &self.input_hash, &self.args_hash)
    }
}

/// Result returned by a peer after executing a task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task this result is for.
    pub task_id: String,
    /// The peer that computed this result.
    pub peer_id: PeerId,
    /// Content hash of the output artifact.
    pub output_hash: ContentHash,
    /// Encrypted output data (None if output is on shared filesystem).
    pub encrypted_output: Option<EncryptedPayload>,
    /// Wall-clock time for the task (seconds).
    pub wall_time_secs: f64,
    /// Ed25519 signature over (task_id + output_hash + wall_time_secs).
    #[serde(with = "sig_serde")]
    pub signature: Signature,
}

impl TaskResult {
    /// Build the signing payload for this result.
    pub fn sign_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(self.task_id.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&self.output_hash.0);
        buf.extend_from_slice(&self.wall_time_secs.to_le_bytes());
        buf
    }
}

/// Serde for ed25519_dalek::Signature (64 bytes, hex-encoded).
mod sig_serde {
    use ed25519_dalek::Signature;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(sig: &Signature, s: S) -> Result<S::Ok, S::Error> {
        faster_hex::hex_string(&sig.to_bytes()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Signature, D::Error> {
        let hex: String = Deserialize::deserialize(d)?;
        let mut bytes = [0u8; 64];
        faster_hex::hex_decode(hex.as_bytes(), &mut bytes)
            .map_err(|e| serde::de::Error::custom(format!("invalid hex: {e}")))?;
        Ok(Signature::from_bytes(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::crypto::{KeyPair, verify};

    fn make_manifest(kp: &KeyPair) -> TaskManifest {
        let input_hash = ContentHash::of_bytes(&[1u8; 32]);
        let args_hash = ContentHash::of_bytes(&[2u8; 32]);
        let payload = sign_payload("test-task-1", "warm_fb_cache", &input_hash, &args_hash);
        let sig = kp.sign(&payload);
        TaskManifest {
            task_id: "test-task-1".into(),
            coordinator_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            stage_name: "warm_fb_cache".into(),
            stage_schema: 1,
            input_hash,
            args_hash,
            args: serde_json::json!({"lma_root": "/data"}),
            resources: ResourceRequest::default(),
            data_class: DataClass::Public,
            timeout_secs: 3600,
            encrypted_input: None,
            signature: sig,
        }
    }

    #[test]
    fn manifest_sign_verify() {
        let kp = KeyPair::generate();
        let manifest = make_manifest(&kp);
        let payload = manifest.sign_payload();
        assert!(verify(&kp.verifying, &payload, &manifest.signature));
    }

    #[test]
    fn manifest_serialization_roundtrip() {
        let kp = KeyPair::generate();
        let manifest = make_manifest(&kp);
        let json = serde_json::to_string(&manifest).unwrap();
        let manifest2: TaskManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.task_id, manifest2.task_id);
        assert_eq!(manifest.stage_name, manifest2.stage_name);
        assert_eq!(manifest.input_hash, manifest2.input_hash);
    }

    #[test]
    fn result_sign_verify() {
        let kp = KeyPair::generate();
        let output_hash = ContentHash::of_bytes(&[3u8; 32]);
        let mut result = TaskResult {
            task_id: "test-task-1".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash,
            encrypted_output: None,
            wall_time_secs: 42.5,
            signature: Signature::from_bytes(&[0u8; 64]), // placeholder
        };
        let payload = result.sign_payload();
        result.signature = kp.sign(&payload);
        assert!(verify(&kp.verifying, &payload, &result.signature));
    }

    #[test]
    fn result_serialization_roundtrip() {
        let kp = KeyPair::generate();
        let result = TaskResult {
            task_id: "test-task-1".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            output_hash: ContentHash::of_bytes(&[3u8; 32]),
            encrypted_output: None,
            wall_time_secs: 42.5,
            signature: kp.sign(b"test"),
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.task_id, result2.task_id);
        assert_eq!(result.output_hash, result2.output_hash);
    }

    #[test]
    fn manifest_tampered_signature_fails() {
        let kp = KeyPair::generate();
        let mut manifest = make_manifest(&kp);
        // Tamper with the task_id after signing.
        manifest.task_id = "tampered".into();
        let payload = manifest.sign_payload();
        assert!(!verify(&kp.verifying, &payload, &manifest.signature));
    }
}
