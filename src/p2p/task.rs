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

use crate::framework::artifact::{ContentHash, ContentId, InvocationKey};
use crate::p2p::crypto::EncryptedPayload;
use crate::p2p::peer::PeerId;
use crate::p2p::trust::DataClass;

pub const TASK_PROTOCOL_VERSION: u16 = 2;

fn legacy_protocol_version() -> u16 {
    1
}

/// A task to be executed by a remote peer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskManifest {
    /// Signed wire-contract version. Missing on pre-A09 records and therefore
    /// deserializes as v1, which current workers reject before execution.
    #[serde(default = "legacy_protocol_version")]
    pub protocol_version: u16,
    /// Unique task identifier: `blut-<job>-<node_idx>`.
    pub task_id: String,
    /// Coordinator's peer ID (so the peer knows who sent this).
    pub coordinator_id: PeerId,
    /// Name of the stage to execute (must match a known stage).
    pub stage_name: String,
    /// Stage schema version.
    pub stage_schema: u32,
    /// Content identity of the input artifact.
    #[serde(rename = "input_hash")]
    pub input_content_id: ContentId,
    /// Invocation identity used only to correlate the returned content object
    /// with the coordinator cache. It is never an expected content hash.
    pub invocation_key: InvocationKey,
    /// Content hash of the serialized args.
    pub args_hash: ContentHash,
    /// Expected output identity only when the stage contract can derive it
    /// analytically before execution. Generic dispatch leaves this `None`.
    #[serde(rename = "expected_output_hash")]
    pub expected_content_id: Option<ContentId>,
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
    /// Ed25519 signature over (task_id + stage_name + input_hash + args_hash +
    /// args + expected_output_hash + coordinator_id + resources + data_class +
    /// timeout_secs). The peer verifies this to confirm the task came from the
    /// coordinator. Note the signature covers `args` directly (not just
    /// `args_hash`) — see `sign_payload` and `verify_args`.
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

/// Length-prefix raw bytes into a buffer (4-byte LE length + bytes).
fn write_lp_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    let len = bytes.len() as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Length-prefix a string into a buffer (4-byte LE length + bytes).
fn write_lp_string(buf: &mut Vec<u8>, s: &str) {
    write_lp_bytes(buf, s.as_bytes());
}

impl TaskManifest {
    /// Build the signing payload for this manifest. Covers all mutable
    /// fields to prevent MITM tampering — INCLUDING `args` itself, not just
    /// `args_hash`.
    ///
    /// `args_hash` alone previously gave the signature no real leverage over
    /// `args`: the hash was inside the signed payload, but nothing on the
    /// peer side recomputed it against the `args` bytes actually received
    /// and rejected on mismatch, so a swapped `args` field still carried a
    /// verifying signature. Signing the args bytes directly closes that
    /// class of bug outright — the signature itself no longer verifies if
    /// `args` changes in transit. `args_hash` stays on the struct (and in
    /// this payload) because it's also used as a cache/dedup key elsewhere
    /// (see `framework::executor`); `verify_args` below is a second,
    /// independent belt-and-suspenders check for callers that want to
    /// reconcile received args against `args_hash` specifically.
    pub fn sign_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.protocol_version.to_le_bytes());
        write_lp_string(&mut buf, &self.task_id);
        write_lp_string(&mut buf, &self.stage_name);
        buf.extend_from_slice(&self.input_content_id.digest().0);
        buf.extend_from_slice(&self.invocation_key.digest().0);
        buf.extend_from_slice(&self.args_hash.0);
        // `args` is recipe-arg JSON (small), not the artifact payload, so
        // signing it directly is cheap. serde_json's `Value` serialization
        // is infallible (it's already-parsed JSON; no user Serialize impls
        // that could fail), and uses a `BTreeMap`-backed `Map` by default
        // (no `preserve_order` feature enabled — see Cargo.toml), so object
        // key order is canonical/deterministic across identical `Value`s.
        let args_bytes =
            serde_json::to_vec(&self.args).expect("serde_json::Value serialization is infallible");
        write_lp_bytes(&mut buf, &args_bytes);
        match self.expected_content_id {
            Some(content_id) => {
                buf.push(1);
                buf.extend_from_slice(&content_id.digest().0);
            }
            None => buf.push(0),
        }
        buf.extend_from_slice(&self.coordinator_id.0);
        buf.extend_from_slice(&self.resources.cpu_cores.to_le_bytes());
        buf.extend_from_slice(&self.resources.memory_gib.to_le_bytes());
        buf.push(self.resources.gpu as u8);
        buf.push(match self.data_class {
            DataClass::Public => 0,
            DataClass::Internal => 1,
            DataClass::Restricted => 2,
        });
        buf.extend_from_slice(&self.timeout_secs.to_le_bytes());
        buf
    }

    /// Verify that `received_args` — the args a peer is about to run a
    /// stage with — actually matches what `args_hash` commits to.
    ///
    /// This is defense-in-depth alongside `sign_payload` now covering
    /// `args` directly: callers (e.g. `peer_exec::execute_one`) MUST call
    /// this (or rely on `sign_payload` covering `args`) after signature
    /// verification and before running the stage with `received_args`.
    /// Before this method existed, `args_hash` sat inside the signed
    /// payload but nothing ever recomputed it against the args a peer
    /// actually received, so the hash provided no real protection — only
    /// its own presence was checked, not its correctness relative to the
    /// executed args.
    pub fn verify_args(&self, received_args: &serde_json::Value) -> crate::error::Result<()> {
        let bytes = serde_json::to_vec(received_args)
            .map_err(|e| crate::error::TrainError::other(format!("serialize args: {e}")))?;
        let actual = ContentHash::of_bytes(&bytes);
        if actual != self.args_hash {
            return Err(crate::error::TrainError::other(format!(
                "args_hash mismatch for task '{}': manifest commits to {}, received args hash to {}",
                self.task_id,
                self.args_hash.to_hex(),
                actual.to_hex(),
            )));
        }
        Ok(())
    }
}

/// Result returned by a peer after executing a task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskResult {
    /// Signed wire-contract version; see [`TaskManifest::protocol_version`].
    #[serde(default = "legacy_protocol_version")]
    pub protocol_version: u16,
    /// The task this result is for.
    pub task_id: String,
    /// The peer that computed this result.
    pub peer_id: PeerId,
    /// Content identity of the output artifact, independently verified by the
    /// receiving artifact store before the result is accepted.
    #[serde(rename = "output_hash")]
    pub content_id: ContentId,
    /// Encrypted output data (None if output is on shared filesystem).
    pub encrypted_output: Option<EncryptedPayload>,
    /// Wall-clock time for the task (milliseconds, integer for deterministic signing).
    pub wall_time_ms: u64,
    /// Ed25519 signature over all fields.
    #[serde(with = "sig_serde")]
    pub signature: Signature,
}

impl TaskResult {
    /// Build the signing payload for this result. Covers all mutable
    /// fields to prevent MITM tampering.
    pub fn sign_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.protocol_version.to_le_bytes());
        write_lp_string(&mut buf, &self.task_id);
        buf.extend_from_slice(&self.peer_id.0);
        buf.extend_from_slice(&self.content_id.digest().0);
        buf.extend_from_slice(&self.wall_time_ms.to_le_bytes());
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
        let mut manifest = TaskManifest {
            protocol_version: TASK_PROTOCOL_VERSION,
            task_id: "test-task-1".into(),
            coordinator_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            stage_name: "warm_fb_cache".into(),
            stage_schema: 1,
            input_content_id: ContentId::from_digest(input_hash),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"invocation")),
            args_hash,
            expected_content_id: Some(ContentId::from_digest(ContentHash::of_bytes(&[3u8; 32]))),
            args: serde_json::json!({"lma_root": "/data"}),
            resources: ResourceRequest::default(),
            data_class: DataClass::Public,
            timeout_secs: 3600,
            encrypted_input: None,
            signature: kp.sign(b"placeholder"),
        };
        manifest.signature = kp.sign(&manifest.sign_payload());
        manifest
    }

    #[test]
    fn manifest_sign_verify() {
        let kp = KeyPair::generate();
        let manifest = make_manifest(&kp);
        let payload = manifest.sign_payload();
        assert!(verify(&kp.verifying, &payload, &manifest.signature));
    }

    #[test]
    fn manifest_without_expected_identity_is_signed_and_round_trips() {
        let kp = KeyPair::generate();
        let mut manifest = make_manifest(&kp);
        manifest.expected_content_id = None;
        manifest.signature = kp.sign(&manifest.sign_payload());
        assert!(verify(
            &kp.verifying,
            &manifest.sign_payload(),
            &manifest.signature
        ));
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let decoded: TaskManifest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.expected_content_id, None);
    }

    #[test]
    fn manifest_serialization_roundtrip() {
        let kp = KeyPair::generate();
        let manifest = make_manifest(&kp);
        let json = serde_json::to_string(&manifest).unwrap();
        let manifest2: TaskManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.task_id, manifest2.task_id);
        assert_eq!(manifest.stage_name, manifest2.stage_name);
        assert_eq!(manifest.input_content_id, manifest2.input_content_id);
        assert_eq!(manifest.invocation_key, manifest2.invocation_key);
    }

    #[test]
    fn content_identity_keeps_the_legacy_hash_json_shape() {
        let kp = KeyPair::generate();
        let manifest = make_manifest(&kp);
        let mut value = serde_json::to_value(&manifest).unwrap();

        let legacy_input: ContentHash =
            serde_json::from_value(value["input_hash"].clone()).unwrap();
        let legacy_expected: ContentHash =
            serde_json::from_value(value["expected_output_hash"].clone()).unwrap();
        assert_eq!(legacy_input, manifest.input_content_id.digest());
        assert_eq!(
            legacy_expected,
            manifest.expected_content_id.unwrap().digest()
        );

        value.as_object_mut().unwrap().remove("protocol_version");
        let legacy: TaskManifest = serde_json::from_value(value).unwrap();
        assert_eq!(legacy.protocol_version, 1);
    }

    #[test]
    fn result_sign_verify() {
        let kp = KeyPair::generate();
        let content_id = ContentId::from_digest(ContentHash::of_bytes(&[3u8; 32]));
        let mut result = TaskResult {
            protocol_version: TASK_PROTOCOL_VERSION,
            task_id: "test-task-1".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            content_id,
            encrypted_output: None,
            wall_time_ms: 42500,
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
            protocol_version: TASK_PROTOCOL_VERSION,
            task_id: "test-task-1".into(),
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            content_id: ContentId::from_digest(ContentHash::of_bytes(&[3u8; 32])),
            encrypted_output: None,
            wall_time_ms: 42500,
            signature: kp.sign(b"test"),
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.task_id, result2.task_id);
        assert_eq!(result.content_id, result2.content_id);
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

    /// Build a manifest whose `args_hash` is genuinely derived from `args`
    /// (unlike `make_manifest`'s fixed dummy `args_hash`), for the
    /// args-integrity tests below.
    fn make_manifest_with_args(kp: &KeyPair, args: serde_json::Value) -> TaskManifest {
        let input_hash = ContentHash::of_bytes(&[1u8; 32]);
        let args_hash = ContentHash::of_bytes(&serde_json::to_vec(&args).unwrap());
        let mut manifest = TaskManifest {
            protocol_version: TASK_PROTOCOL_VERSION,
            task_id: "test-task-1".into(),
            coordinator_id: crate::p2p::peer::PeerId::from_pubkey(&kp.verifying),
            stage_name: "warm_fb_cache".into(),
            stage_schema: 1,
            input_content_id: ContentId::from_digest(input_hash),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"invocation")),
            args_hash,
            expected_content_id: Some(ContentId::from_digest(ContentHash::of_bytes(&[3u8; 32]))),
            args,
            resources: ResourceRequest::default(),
            data_class: DataClass::Public,
            timeout_secs: 3600,
            encrypted_input: None,
            signature: kp.sign(b"placeholder"),
        };
        manifest.signature = kp.sign(&manifest.sign_payload());
        manifest
    }

    /// (a) `verify_args` succeeds when the received args genuinely match
    /// what `args_hash` commits to.
    #[test]
    fn verify_args_accepts_matching_args() {
        let kp = KeyPair::generate();
        let args = serde_json::json!({"lma_root": "/data", "epochs": 3});
        let manifest = make_manifest_with_args(&kp, args.clone());
        assert!(manifest.verify_args(&args).is_ok());
    }

    /// (b) `verify_args` rejects args that don't hash to `args_hash` — the
    /// scenario from the finding: a peer about to run tampered args must be
    /// stopped even though the signature over the manifest itself still
    /// verifies (args_hash didn't change, only the args bytes did).
    #[test]
    fn verify_args_rejects_tampered_args() {
        let kp = KeyPair::generate();
        let args = serde_json::json!({"lma_root": "/data", "epochs": 3});
        let manifest = make_manifest_with_args(&kp, args);
        let tampered = serde_json::json!({"lma_root": "/data", "epochs": 9_999_999});
        let err = manifest
            .verify_args(&tampered)
            .expect_err("tampered args must be rejected");
        assert!(err.to_string().contains("args_hash mismatch"), "{err}");
    }

    /// (c) `sign_payload` now covers `args` directly, so swapping `args`
    /// changes the signed payload (and therefore invalidates any signature
    /// computed before the swap) even if `args_hash` is left untouched.
    /// Before this fix, `args` wasn't in the payload at all, so this
    /// assertion would have failed (the two payloads were identical
    /// whenever `args_hash` matched, regardless of `args`).
    #[test]
    fn sign_payload_changes_when_args_change() {
        let kp = KeyPair::generate();
        let args_a = serde_json::json!({"lma_root": "/data"});
        let manifest_a = make_manifest_with_args(&kp, args_a);

        // Simulate an attacker swapping `args` in transit without being
        // able to recompute `args_hash` or re-sign (both would require the
        // coordinator's private key / knowledge, which the attacker lacks).
        let mut manifest_b = manifest_a.clone();
        manifest_b.args = serde_json::json!({"lma_root": "/evil"});

        assert_ne!(manifest_a.sign_payload(), manifest_b.sign_payload());
        // The original signature (over manifest_a's payload) must NOT
        // verify against the tampered manifest's payload.
        assert!(!verify(
            &kp.verifying,
            &manifest_b.sign_payload(),
            &manifest_a.signature
        ));
    }
}
