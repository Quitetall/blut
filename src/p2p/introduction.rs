// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Signed peer introductions (ADR 0079 A4).
//!
//! A mesh needs a way to onboard a peer you haven't met without an operator
//! typing its key on every node. An [`Introduction`] is a signed voucher: a
//! node you already trust says "this subject key is real, treat it up to
//! `max_trust`". It lets trust propagate without a central authority — but on
//! a TIGHT leash, because vouching is a Sybil lever if unbounded:
//!
//! - **Only a locally-`Trusted` introducer counts.** A voucher signed by a
//!   peer you rate `Anonymous`/`Registered` is ignored — trust is not
//!   transitive through untrusted nodes.
//! - **The ceiling is `Registered`, never `Trusted`.** An introduction can get
//!   a stranger onto the mesh (enough to run `Public`/`Internal` work per the
//!   dispatch matrix) but CANNOT confer the `Trusted` level that clinical
//!   (`Restricted`) routing needs — that stays an operator-CLI decision.
//! - **Expiry is mandatory.** A voucher past `expires_at` is dead; there is no
//!   "never expires".
//! - **The signature is verified** against the introducer's own key, so a
//!   forged or tampered voucher is rejected.
//!
//! This module is the decision logic ([`Introduction::evaluate`]); persisting
//! the resulting promotion into the registry is the caller's job.

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::p2p::crypto::{KeyPair, verify};
use crate::p2p::peer::{PeerId, PeerInfo};
use crate::p2p::trust::TrustLevel;

/// The highest trust an introduction can ever confer. Promotion to `Trusted`
/// is an operator-CLI decision, never an automatic consequence of a voucher.
pub const MAX_INTRODUCED_TRUST: TrustLevel = TrustLevel::Registered;

/// A signed voucher: `introducer` attests that `subject_pubkey` is a real peer
/// that may be trusted up to `max_trust` until `expires_at`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Introduction {
    /// The vouching peer's id (must be locally `Trusted` to have any effect).
    pub introducer: PeerId,
    /// The vouched-for peer's id (derived from `subject_pubkey`).
    pub subject: PeerId,
    /// The subject's Ed25519 identity key, so the registry can create its entry.
    #[serde(with = "pubkey_hex")]
    pub subject_pubkey: VerifyingKey,
    /// Trust ceiling requested — clamped to [`MAX_INTRODUCED_TRUST`] on apply.
    pub max_trust: TrustLevel,
    /// Hard expiry. A voucher past this instant is ignored.
    pub expires_at: DateTime<Utc>,
    /// Ed25519 signature by `introducer` over [`Introduction::sign_payload`].
    #[serde(with = "sig_hex")]
    pub signature: Signature,
}

/// Domain-separation tag prefixing the signed payload, so an introduction
/// signature can never be confused with a signature over any other blut
/// structure (task manifest, result, …).
const INTRO_SIG_DOMAIN: &[u8] = b"blut-introduction-v1";

/// Why an introduction did not grant trust.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntroError {
    /// The introducer isn't the peer whose `PeerInfo` was supplied.
    IntroducerMismatch,
    /// The introducer is not locally `Trusted` — trust is not transitive.
    UntrustedIntroducer,
    /// The subject id doesn't match `subject_pubkey`.
    SubjectMismatch,
    /// The voucher's signature failed to verify.
    BadSignature,
    /// The voucher has expired.
    Expired,
}

impl std::fmt::Display for IntroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            IntroError::IntroducerMismatch => "introducer id does not match the supplied peer",
            IntroError::UntrustedIntroducer => "introducer is not locally Trusted",
            IntroError::SubjectMismatch => "subject id does not match subject_pubkey",
            IntroError::BadSignature => "introduction signature failed to verify",
            IntroError::Expired => "introduction has expired",
        };
        f.write_str(s)
    }
}

impl std::error::Error for IntroError {}

impl Introduction {
    /// Create + sign an introduction for `subject_pubkey`, valid until
    /// `expires_at`, requesting up to `max_trust` (clamped on apply).
    pub fn create(
        introducer_kp: &KeyPair,
        subject_pubkey: VerifyingKey,
        max_trust: TrustLevel,
        expires_at: DateTime<Utc>,
    ) -> Self {
        // Build with a zero signature, then sign the real payload — no wasted
        // signing pass. `sign_payload` never reads `self.signature`.
        let mut intro = Self {
            introducer: PeerId::from_pubkey(&introducer_kp.verifying),
            subject: PeerId::from_pubkey(&subject_pubkey),
            subject_pubkey,
            max_trust,
            expires_at,
            signature: Signature::from_bytes(&[0u8; 64]),
        };
        intro.signature = introducer_kp.sign(&intro.sign_payload());
        intro
    }

    /// Bytes the signature covers: a domain tag + every field except the
    /// signature itself, so tampering with any of them invalidates the voucher.
    /// All fields are fixed-length (32/32/1/8 bytes), so concatenation is
    /// unambiguous.
    pub fn sign_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(INTRO_SIG_DOMAIN);
        buf.extend_from_slice(&self.introducer.0);
        buf.extend_from_slice(&self.subject.0);
        buf.extend_from_slice(self.subject_pubkey.as_bytes());
        buf.push(self.max_trust.level());
        buf.extend_from_slice(self.expires_at.timestamp().to_le_bytes().as_slice());
        buf
    }

    /// Decide the trust this voucher grants, given the introducer's local
    /// `PeerInfo` and the current time. Returns the (clamped) granted trust or
    /// an [`IntroError`]. Does NOT mutate any registry — the caller promotes
    /// the subject only if the grant EXCEEDS the subject's current trust (an
    /// introduction never demotes an operator-set `Trusted` peer).
    pub fn evaluate(
        &self,
        introducer: &PeerInfo,
        now: DateTime<Utc>,
    ) -> Result<TrustLevel, IntroError> {
        // Bind the key we'll verify with to the claimed introducer id: derive
        // the id from the introducer's OWN pubkey rather than trusting the
        // `PeerInfo.id` field, so a corrupt/mismatched registry entry can't let
        // a wrong key pass the signature check.
        if PeerId::from_pubkey(&introducer.pubkey) != self.introducer {
            return Err(IntroError::IntroducerMismatch);
        }
        if introducer.trust != TrustLevel::Trusted {
            return Err(IntroError::UntrustedIntroducer);
        }
        if PeerId::from_pubkey(&self.subject_pubkey) != self.subject {
            return Err(IntroError::SubjectMismatch);
        }
        if now > self.expires_at {
            return Err(IntroError::Expired);
        }
        if !verify(&introducer.pubkey, &self.sign_payload(), &self.signature) {
            return Err(IntroError::BadSignature);
        }
        // Clamp to the ceiling: an introduction can reach Registered, no higher.
        let granted = if self.max_trust.level() > MAX_INTRODUCED_TRUST.level() {
            MAX_INTRODUCED_TRUST
        } else {
            self.max_trust
        };
        Ok(granted)
    }
}

/// Hex serde for an Ed25519 `VerifyingKey`.
mod pubkey_hex {
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

/// Hex serde for an Ed25519 `Signature`.
mod sig_hex {
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
    use crate::p2p::peer::PeerCapabilities;

    fn peer_info(kp: &KeyPair, trust: TrustLevel) -> PeerInfo {
        PeerInfo::new(
            kp.verifying,
            kp.x25519_public,
            trust,
            PeerCapabilities::default(),
        )
    }

    fn future() -> DateTime<Utc> {
        Utc::now() + chrono::Duration::hours(1)
    }

    #[test]
    fn trusted_introducer_promotes_to_registered() {
        let introducer_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Registered,
            future(),
        );
        let info = peer_info(&introducer_kp, TrustLevel::Trusted);
        assert_eq!(
            intro.evaluate(&info, Utc::now()).unwrap(),
            TrustLevel::Registered
        );
    }

    #[test]
    fn ceiling_clamps_requested_trusted_down_to_registered() {
        // Even if the voucher REQUESTS Trusted, the grant is capped.
        let introducer_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Trusted, // request the top level…
            future(),
        );
        let info = peer_info(&introducer_kp, TrustLevel::Trusted);
        assert_eq!(
            intro.evaluate(&info, Utc::now()).unwrap(),
            TrustLevel::Registered, // …but only Registered is granted
        );
    }

    #[test]
    fn non_trusted_introducer_is_ignored() {
        let introducer_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Registered,
            future(),
        );
        for t in [TrustLevel::Anonymous, TrustLevel::Registered] {
            let info = peer_info(&introducer_kp, t);
            assert_eq!(
                intro.evaluate(&info, Utc::now()),
                Err(IntroError::UntrustedIntroducer)
            );
        }
    }

    #[test]
    fn expired_introduction_is_ignored() {
        let introducer_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let past = Utc::now() - chrono::Duration::minutes(1);
        let intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Registered,
            past,
        );
        let info = peer_info(&introducer_kp, TrustLevel::Trusted);
        assert_eq!(intro.evaluate(&info, Utc::now()), Err(IntroError::Expired));
    }

    #[test]
    fn forged_signature_is_rejected() {
        let introducer_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let mut intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Registered,
            future(),
        );
        // Tamper with a signed field → signature no longer matches.
        intro.max_trust = TrustLevel::Trusted;
        let info = peer_info(&introducer_kp, TrustLevel::Trusted);
        assert_eq!(
            intro.evaluate(&info, Utc::now()),
            Err(IntroError::BadSignature)
        );
    }

    #[test]
    fn introducer_identity_mismatch_is_rejected() {
        let introducer_kp = KeyPair::generate();
        let impostor_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Registered,
            future(),
        );
        // Supply a DIFFERENT peer as the introducer → id mismatch.
        let info = peer_info(&impostor_kp, TrustLevel::Trusted);
        assert_eq!(
            intro.evaluate(&info, Utc::now()),
            Err(IntroError::IntroducerMismatch)
        );
    }

    #[test]
    fn subject_pubkey_swap_is_rejected() {
        let introducer_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let other_kp = KeyPair::generate();
        let mut intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Registered,
            future(),
        );
        // Swap the subject key but keep the old subject id → mismatch (and the
        // signature also breaks, but the id check fires first, deterministically).
        intro.subject_pubkey = other_kp.verifying;
        let info = peer_info(&introducer_kp, TrustLevel::Trusted);
        assert_eq!(
            intro.evaluate(&info, Utc::now()),
            Err(IntroError::SubjectMismatch)
        );
    }

    #[test]
    fn introduction_json_round_trips() {
        let introducer_kp = KeyPair::generate();
        let subject_kp = KeyPair::generate();
        let intro = Introduction::create(
            &introducer_kp,
            subject_kp.verifying,
            TrustLevel::Registered,
            future(),
        );
        let json = serde_json::to_string(&intro).unwrap();
        let back: Introduction = serde_json::from_str(&json).unwrap();
        let info = peer_info(&introducer_kp, TrustLevel::Trusted);
        // The round-tripped voucher still verifies + grants.
        assert_eq!(
            back.evaluate(&info, Utc::now()).unwrap(),
            TrustLevel::Registered
        );
    }
}
