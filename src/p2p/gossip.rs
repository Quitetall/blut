// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Signed peer-exchange gossip (ADR 0079 A5).
//!
//! Static `--seed` addresses get a node onto the mesh; gossip is how it then
//! LEARNS the rest without a central directory. A node periodically sends a
//! [`PeerExchange`] — a signed list of the peers (identity + address) it knows
//! — to its connections. A receiver verifies the signature, then learns each
//! address and admits each identity as **`Anonymous`** (the fail-closed
//! dispatch matrix still gates what that peer may actually do, so learning an
//! address grants no trust).
//!
//! Safety rests on A4's machinery: the exchange is signed by the sender (a
//! forged/tampered list is dropped), and admitting at `Anonymous` means gossip
//! can widen *reachability* but never *trust* — promotion still needs an
//! operator or a signed [`Introduction`](crate::p2p::introduction::Introduction).
//! mDNS is deliberately NOT here: gossip over the authenticated mesh covers the
//! WAN case; mDNS would only help a LAN and is a later convenience.

use std::net::SocketAddr;

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::p2p::crypto::{KeyPair, verify};
use crate::p2p::peer::PeerId;

/// Domain-separation tag for a peer-exchange signature.
const GOSSIP_SIG_DOMAIN: &[u8] = b"blut-peer-exchange-v1";

/// Max records a single exchange may carry — a bound so one (authenticated but
/// only `Anonymous`) peer can't flood a receiver's address book in one message.
pub const MAX_RECORDS_PER_EXCHANGE: usize = 256;

/// How old (seconds) an exchange may be and still be acted on. Past this it is
/// a stale replay and ignored; also rejects timestamps implausibly in the
/// future (clock-skew tolerance below).
pub const MAX_GOSSIP_AGE_SECS: i64 = 24 * 3600;

/// Allowed clock skew (seconds) for a future-dated exchange.
const GOSSIP_FUTURE_SKEW_SECS: i64 = 300;

/// A single shareable peer record: identity key + last-known address. The
/// `PeerId` is derivable from `pubkey`, so it isn't carried separately.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerRecord {
    /// The peer's Ed25519 identity key (hex).
    #[serde(with = "pubkey_hex")]
    pub pubkey: VerifyingKey,
    /// Where the peer was last reachable.
    pub addr: SocketAddr,
}

impl PeerRecord {
    pub fn peer_id(&self) -> PeerId {
        PeerId::from_pubkey(&self.pubkey)
    }
}

/// A signed set of peer records shared by `from`. `issued_at` (unix seconds)
/// lets a receiver prefer fresher exchanges and ignore stale replays.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerExchange {
    /// The sharing peer's id (its `pubkey` is looked up in the local registry
    /// to verify the signature).
    pub from: PeerId,
    /// Unix-seconds timestamp the exchange was built.
    pub issued_at: i64,
    /// The peers `from` is advertising.
    pub records: Vec<PeerRecord>,
    /// Ed25519 signature by `from` over [`PeerExchange::sign_payload`].
    #[serde(with = "sig_hex")]
    pub signature: Signature,
}

impl PeerExchange {
    /// Build + sign a peer exchange advertising `records`, stamped `issued_at`
    /// (unix seconds — passed in, since wall-clock isn't available everywhere).
    pub fn create(from_kp: &KeyPair, issued_at: i64, records: Vec<PeerRecord>) -> Self {
        let mut ex = Self {
            from: PeerId::from_pubkey(&from_kp.verifying),
            issued_at,
            records,
            signature: Signature::from_bytes(&[0u8; 64]),
        };
        ex.signature = from_kp.sign(&ex.sign_payload());
        ex
    }

    /// Bytes the signature covers: a domain tag + `from` + `issued_at` + every
    /// record (length-prefixed address string, since a `SocketAddr` renders to
    /// a variable-length string).
    pub fn sign_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(GOSSIP_SIG_DOMAIN);
        buf.extend_from_slice(&self.from.0);
        buf.extend_from_slice(&self.issued_at.to_le_bytes());
        buf.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        for r in &self.records {
            buf.extend_from_slice(r.pubkey.as_bytes());
            let addr = r.addr.to_string();
            buf.extend_from_slice(&(addr.len() as u64).to_le_bytes());
            buf.extend_from_slice(addr.as_bytes());
        }
        buf
    }

    /// Verify the exchange was signed by `from` using `from`'s pubkey. The
    /// derived-id bind (`PeerId::from_pubkey(from_pubkey) == self.from`) is the
    /// actual security gate — the `from` field is untrusted until this passes.
    /// The caller looks `from_pubkey` up in its own registry by `self.from`.
    pub fn verify(&self, from_pubkey: &VerifyingKey) -> bool {
        if PeerId::from_pubkey(from_pubkey) != self.from {
            return false;
        }
        verify(from_pubkey, &self.sign_payload(), &self.signature)
    }

    /// Whether the exchange is fresh enough to act on at `now` (unix seconds):
    /// no older than [`MAX_GOSSIP_AGE_SECS`] and not further than the allowed
    /// skew into the future. Bounds replay of a captured valid exchange to a
    /// finite window.
    pub fn is_fresh(&self, now: i64) -> bool {
        let age = now - self.issued_at;
        (-GOSSIP_FUTURE_SKEW_SECS..=MAX_GOSSIP_AGE_SECS).contains(&age)
    }

    /// Whether the record count is within [`MAX_RECORDS_PER_EXCHANGE`].
    pub fn within_size_limit(&self) -> bool {
        self.records.len() <= MAX_RECORDS_PER_EXCHANGE
    }
}

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

    fn record(addr: &str) -> PeerRecord {
        let kp = KeyPair::generate();
        PeerRecord {
            pubkey: kp.verifying,
            addr: addr.parse().unwrap(),
        }
    }

    #[test]
    fn create_and_verify_round_trip() {
        let kp = KeyPair::generate();
        let ex = PeerExchange::create(
            &kp,
            1000,
            vec![record("127.0.0.1:9000"), record("[::1]:9001")],
        );
        assert!(ex.verify(&kp.verifying), "own signature verifies");
    }

    #[test]
    fn wrong_signer_is_rejected() {
        let kp = KeyPair::generate();
        let other = KeyPair::generate();
        let ex = PeerExchange::create(&kp, 1000, vec![record("127.0.0.1:9000")]);
        // Verifying with a different key fails (the derived-id bind trips first).
        assert!(!ex.verify(&other.verifying));
    }

    #[test]
    fn tampered_record_is_rejected() {
        let kp = KeyPair::generate();
        let mut ex = PeerExchange::create(&kp, 1000, vec![record("127.0.0.1:9000")]);
        // Mutate an advertised address after signing → signature breaks.
        ex.records[0].addr = "10.0.0.1:1".parse().unwrap();
        assert!(!ex.verify(&kp.verifying));
    }

    #[test]
    fn tampered_issued_at_is_rejected() {
        let kp = KeyPair::generate();
        let mut ex = PeerExchange::create(&kp, 1000, vec![record("127.0.0.1:9000")]);
        ex.issued_at = 9999;
        assert!(!ex.verify(&kp.verifying));
    }

    #[test]
    fn peer_id_derives_from_record_pubkey() {
        let kp = KeyPair::generate();
        let r = PeerRecord {
            pubkey: kp.verifying,
            addr: "127.0.0.1:9000".parse().unwrap(),
        };
        assert_eq!(r.peer_id(), PeerId::from_pubkey(&kp.verifying));
    }

    #[test]
    fn json_round_trips_and_still_verifies() {
        let kp = KeyPair::generate();
        let ex = PeerExchange::create(&kp, 1234, vec![record("127.0.0.1:9000")]);
        let json = serde_json::to_string(&ex).unwrap();
        let back: PeerExchange = serde_json::from_str(&json).unwrap();
        assert!(back.verify(&kp.verifying));
    }

    #[test]
    fn empty_exchange_verifies() {
        let kp = KeyPair::generate();
        let ex = PeerExchange::create(&kp, 1, Vec::new());
        assert!(ex.verify(&kp.verifying));
    }

    #[test]
    fn freshness_window_bounds_replay() {
        let kp = KeyPair::generate();
        let now = 1_000_000;
        let fresh = PeerExchange::create(&kp, now, Vec::new());
        assert!(fresh.is_fresh(now));
        assert!(fresh.is_fresh(now + MAX_GOSSIP_AGE_SECS)); // at the edge
        assert!(!fresh.is_fresh(now + MAX_GOSSIP_AGE_SECS + 1)); // stale
        assert!(!fresh.is_fresh(now - GOSSIP_FUTURE_SKEW_SECS - 1)); // too future
    }

    #[test]
    fn size_limit_flags_oversized_exchange() {
        let kp = KeyPair::generate();
        let small = PeerExchange::create(&kp, 1, vec![record("127.0.0.1:1")]);
        assert!(small.within_size_limit());
        let records: Vec<_> = (0..MAX_RECORDS_PER_EXCHANGE + 1)
            .map(|_| record("127.0.0.1:1"))
            .collect();
        let big = PeerExchange::create(&kp, 1, records);
        assert!(!big.within_size_limit());
    }
}
