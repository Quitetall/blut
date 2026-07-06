// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Cryptographic primitives for P2P task dispatch.
//!
//! - Ed25519 signing / verification (task manifest authenticity)
//! - X25519 key exchange (per-task key wrapping)
//! - AES-256-GCM encryption / decryption (data in transit)
//!
//! Each peer has TWO keypairs:
//! - Ed25519 for signing (identity, authentication)
//! - X25519 for encryption (key exchange, data protection)
//!
//! This avoids the complex Ed25519→X25519 birational map. The X25519
//! public key is shared alongside the Ed25519 public key during
//! registration.

use aes_gcm::aead::Aead;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::error::TrainError;

/// A keypair for P2P operations: Ed25519 for signing, X25519 for encryption.
pub struct KeyPair {
    signing: SigningKey,
    /// Ed25519 public key (for signature verification).
    pub verifying: VerifyingKey,
    /// X25519 secret key (for decryption / key exchange).
    x25519_secret: x25519_dalek::StaticSecret,
    /// X25519 public key (share with peers for encryption).
    pub x25519_public: x25519_dalek::PublicKey,
}

impl KeyPair {
    /// Generate a fresh random keypair (Ed25519 + X25519).
    pub fn generate() -> Self {
        let signing = SigningKey::generate(&mut rand::thread_rng());
        let verifying = signing.verifying_key();
        let x25519_secret = x25519_dalek::StaticSecret::random_from_rng(rand::thread_rng());
        let x25519_public = x25519_dalek::PublicKey::from(&x25519_secret);
        Self {
            signing,
            verifying,
            x25519_secret,
            x25519_public,
        }
    }

    /// Load a keypair from stored bytes: [32 Ed25519 seed | 32 X25519 secret].
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let signing = SigningKey::from_bytes(&bytes[..32].try_into().unwrap());
        let verifying = signing.verifying_key();
        let x25519_bytes: [u8; 32] = bytes[32..64].try_into().unwrap();
        let x25519_secret = x25519_dalek::StaticSecret::from(x25519_bytes);
        let x25519_public = x25519_dalek::PublicKey::from(&x25519_secret);
        Self {
            signing,
            verifying,
            x25519_secret,
            x25519_public,
        }
    }

    /// Export the 64-byte secret material (for persistence).
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&self.signing.to_bytes());
        out[32..].copy_from_slice(&self.x25519_secret.to_bytes());
        out
    }

    /// Sign a message with Ed25519.
    pub fn sign(&self, msg: &[u8]) -> Signature {
        self.signing.sign(msg)
    }

    /// Decrypt a payload sealed to THIS keypair's X25519 public key. Keeps the
    /// X25519 secret encapsulated (no accessor) — a peer decrypts its dispatched
    /// input, a coordinator decrypts a returned output, both via this method.
    pub fn decrypt(&self, payload: &EncryptedPayload) -> Result<Vec<u8>, TrainError> {
        decrypt(payload, &self.x25519_secret)
    }
}

/// Verify a detached Ed25519 signature.
pub fn verify(pubkey: &VerifyingKey, msg: &[u8], sig: &Signature) -> bool {
    pubkey.verify(msg, sig).is_ok()
}

/// An AES-256-GCM encrypted payload with the key sealed for the recipient.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptedPayload {
    /// AES-256-GCM ciphertext (includes the 16-byte authentication tag).
    pub ciphertext: Vec<u8>,
    /// The AES key sealed for the recipient.
    /// Format: [32-byte ephemeral X25519 pubkey | 12-byte nonce | sealed AES key].
    pub sealed_key: Vec<u8>,
    /// 12-byte nonce for AES-256-GCM.
    pub nonce: [u8; 12],
}

/// Encrypt plaintext for a recipient identified by their X25519 public key.
///
/// A random AES-256 key is generated, used to encrypt the plaintext, then
/// sealed with the recipient's X25519 public key via ECDH + AES-256-GCM.
pub fn encrypt(
    plaintext: &[u8],
    recipient_x25519_pub: &x25519_dalek::PublicKey,
) -> EncryptedPayload {
    use rand::RngCore;

    // Generate random AES-256 key + nonces.
    let mut aes_key = [0u8; 32];
    let mut nonce_bytes = [0u8; 12];
    let mut seal_nonce = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut aes_key);
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    rand::thread_rng().fill_bytes(&mut seal_nonce);

    // Encrypt plaintext with AES-256-GCM.
    let cipher = Aes256Gcm::new_from_slice(&aes_key).expect("32-byte key");
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .expect("AES-256-GCM encryption failed");

    // Seal the AES key with the recipient's X25519 pubkey.
    // Ephemeral ECDH → shared secret → SHA-256 → AES key for sealing.
    let eph_secret = x25519_dalek::EphemeralSecret::random_from_rng(rand::thread_rng());
    let eph_public = x25519_dalek::PublicKey::from(&eph_secret);
    let shared = eph_secret.diffie_hellman(recipient_x25519_pub);
    let seal_key = hkdf_derive(shared.as_bytes(), b"blut-p2p-seal-v1");
    let seal_cipher = Aes256Gcm::new_from_slice(&seal_key).expect("32-byte key");
    let seal_nonce_ref = Nonce::from_slice(&seal_nonce);
    let sealed_aes = seal_cipher
        .encrypt(seal_nonce_ref, aes_key.as_ref())
        .expect("seal encryption failed");

    // Pack: [ephemeral pubkey (32) | seal nonce (12) | sealed AES key (48)]
    let mut sealed_with_meta = eph_public.as_bytes().to_vec();
    sealed_with_meta.extend_from_slice(&seal_nonce);
    sealed_with_meta.extend_from_slice(&sealed_aes);

    EncryptedPayload {
        ciphertext,
        sealed_key: sealed_with_meta,
        nonce: nonce_bytes,
    }
}

/// Decrypt an encrypted payload using the recipient's X25519 secret key.
pub fn decrypt(
    payload: &EncryptedPayload,
    recipient_x25519_secret: &x25519_dalek::StaticSecret,
) -> Result<Vec<u8>, TrainError> {
    // Extract ephemeral public key + seal nonce from sealed_key.
    if payload.sealed_key.len() < 44 {
        return Err(TrainError::other("sealed_key too short"));
    }
    let eph_pub_bytes: [u8; 32] = payload.sealed_key[..32]
        .try_into()
        .map_err(|_| TrainError::other("invalid ephemeral pubkey"))?;
    let seal_nonce_bytes: [u8; 12] = payload.sealed_key[32..44]
        .try_into()
        .map_err(|_| TrainError::other("invalid seal nonce"))?;
    let sealed_aes = &payload.sealed_key[44..];

    // ECDH → shared secret → SHA-256 → AES key for unsealing.
    let eph_pub = x25519_dalek::PublicKey::from(eph_pub_bytes);
    let shared = recipient_x25519_secret.diffie_hellman(&eph_pub);
    let seal_key = hkdf_derive(shared.as_bytes(), b"blut-p2p-seal-v1");
    let seal_cipher =
        Aes256Gcm::new_from_slice(&seal_key).map_err(|_| TrainError::other("invalid seal key"))?;
    let seal_nonce = Nonce::from_slice(&seal_nonce_bytes);
    let aes_key = seal_cipher
        .decrypt(seal_nonce, sealed_aes)
        .map_err(|_| TrainError::other("failed to unseal AES key — wrong key or tampered"))?;

    // Decrypt the ciphertext with the recovered AES key.
    let cipher = Aes256Gcm::new_from_slice(&aes_key)
        .map_err(|_| TrainError::other("invalid recovered AES key"))?;
    let nonce = Nonce::from_slice(&payload.nonce);
    cipher
        .decrypt(nonce, payload.ciphertext.as_ref())
        .map_err(|_| TrainError::other("AES-256-GCM decryption failed — wrong key or tampered"))
}

/// HKDF-SHA256 key derivation with domain separation.
fn hkdf_derive(ikm: &[u8], info: &[u8]) -> [u8; 32] {
    let hk = hkdf::Hkdf::<sha2::Sha256>::new(None, ikm);
    let mut okm = [0u8; 32];
    hk.expand(info, &mut okm)
        .expect("32 bytes is valid for SHA-256");
    okm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypair_generate_and_roundtrip() {
        let kp = KeyPair::generate();
        let bytes = kp.to_bytes();
        let kp2 = KeyPair::from_bytes(&bytes);
        assert_eq!(kp.verifying.to_bytes(), kp2.verifying.to_bytes());
        assert_eq!(kp.x25519_public.to_bytes(), kp2.x25519_public.to_bytes());
    }

    #[test]
    fn sign_verify_roundtrip() {
        let kp = KeyPair::generate();
        let msg = b"hello, BLUT P2P";
        let sig = kp.sign(msg);
        assert!(verify(&kp.verifying, msg, &sig));
    }

    #[test]
    fn sign_verify_tampered() {
        let kp = KeyPair::generate();
        let msg = b"hello, BLUT P2P";
        let sig = kp.sign(msg);
        let tampered = b"hello, BLUT P3P";
        assert!(!verify(&kp.verifying, tampered, &sig));
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let kp = KeyPair::generate();
        let plaintext = b"secret EEG data";
        let payload = encrypt(plaintext, &kp.x25519_public);
        let decrypted = decrypt(&payload, &kp.x25519_secret).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_wrong_key() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::generate();
        let plaintext = b"secret EEG data";
        let payload = encrypt(plaintext, &kp1.x25519_public);
        assert!(decrypt(&payload, &kp2.x25519_secret).is_err());
    }

    #[test]
    fn encrypt_decrypt_tampered_ciphertext() {
        let kp = KeyPair::generate();
        let plaintext = b"secret EEG data";
        let mut payload = encrypt(plaintext, &kp.x25519_public);
        if let Some(byte) = payload.ciphertext.first_mut() {
            *byte ^= 0xFF;
        }
        assert!(decrypt(&payload, &kp.x25519_secret).is_err());
    }

    #[test]
    fn encrypt_decrypt_tampered_sealed_key() {
        let kp = KeyPair::generate();
        let plaintext = b"secret EEG data";
        let mut payload = encrypt(plaintext, &kp.x25519_public);
        if let Some(byte) = payload.sealed_key.first_mut() {
            *byte ^= 0xFF;
        }
        assert!(decrypt(&payload, &kp.x25519_secret).is_err());
    }

    #[test]
    fn encrypt_different_nonces() {
        let kp = KeyPair::generate();
        let plaintext = b"same data twice";
        let p1 = encrypt(plaintext, &kp.x25519_public);
        let p2 = encrypt(plaintext, &kp.x25519_public);
        assert_ne!(p1.nonce, p2.nonce);
        assert_ne!(p1.ciphertext, p2.ciphertext);
        assert_eq!(decrypt(&p1, &kp.x25519_secret).unwrap(), plaintext);
        assert_eq!(decrypt(&p2, &kp.x25519_secret).unwrap(), plaintext);
    }

    #[test]
    fn sign_task_manifest() {
        let kp = KeyPair::generate();
        let manifest = br#"{"task_id":"test-1","stage_name":"warm_fb_cache"}"#;
        let sig = kp.sign(manifest);
        assert!(verify(&kp.verifying, manifest, &sig));
    }
}
