// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Federated round protocol — initiator-side driver (ADR 0080 C2).
//!
//! One FedAvg round: the initiator seals the global weights per participant
//! (the existing X25519 blob path), dispatches N `fed-local-train` tasks over
//! the mesh, collects signed delta results, hash-verifies them, and hands the
//! deltas to the Python aggregator. **Aggregation ALWAYS runs on the
//! initiator** (ADR 0080 invariant 4) — a worker only computes + returns its
//! own DP-noised delta.
//!
//! This module is the ROUND STATE MACHINE (mesh-transport-agnostic + testable):
//! the straggler policy (`min_clients` + a round deadline), delta integrity
//! verification, and the ε bookkeeping into the [`PrivacyLedger`]. The actual
//! sealing / dispatch / aggregation are the caller's (they use the mesh +
//! `fed-aggregate` Python stage). Every gradient-bearing dispatch is gated by
//! [`gate_gradient_dispatch`](crate::p2p::privacy::gate_gradient_dispatch)
//! first (C1) — Restricted never dispatches.

use crate::error::TrainError;
use crate::framework::artifact::ContentHash;
use crate::p2p::crypto::KeyPair;
use crate::p2p::peer::PeerId;
use crate::p2p::privacy::{DpConfig, PrivacyLedger};

/// Configuration for one federated round.
#[derive(Clone, Debug)]
pub struct FedRoundConfig {
    /// The corpus whose ε budget this round debits.
    pub corpus_id: String,
    /// Minimum participants whose deltas must arrive for the round to aggregate.
    /// Below this, the round is abandoned (no partial aggregate). ≥ 2.
    pub min_clients: usize,
    /// The DP config every participant must train under.
    pub dp: DpConfig,
}

/// A delta result collected from one participant (already arrived before the
/// deadline). The bytes are the participant's DP-noised model update.
#[derive(Clone, Debug)]
pub struct CollectedDelta {
    pub peer: PeerId,
    /// The hash the participant claims for its delta bytes.
    pub claimed_hash: ContentHash,
    /// ε the participant reports spending (Opacus-accounted).
    pub epsilon_cost: f64,
    /// The delta bytes (verified against `claimed_hash`).
    pub bytes: Vec<u8>,
}

impl CollectedDelta {
    /// Whether the bytes actually hash to the claimed value (content integrity —
    /// a participant can't send bytes that don't match its claimed hash).
    pub fn verify(&self) -> bool {
        ContentHash::of_bytes(&self.bytes) == self.claimed_hash
    }
}

/// The outcome of finalizing a round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoundOutcome {
    /// Enough valid deltas arrived — the caller aggregates these participants.
    Aggregate {
        participants: Vec<PeerId>,
        /// Participants that were dispatched but didn't deliver a valid delta.
        dropped: Vec<PeerId>,
    },
    /// Fewer than `min_clients` valid deltas arrived; the round is abandoned.
    Insufficient { got: usize, needed: usize },
}

/// Decide a round's outcome from the deltas that arrived before the deadline,
/// and — if aggregating — debit the round's ε from the ledger (a signed round).
///
/// `dispatched` is the full participant set (to compute who was dropped).
/// `collected` are the deltas that arrived; each is hash-verified here (a
/// mismatch drops that participant rather than failing the whole round). The
/// ε debit is the SUM of the surviving participants' costs, recorded as ONE
/// ledger round — if that would exhaust the budget, the whole round errors
/// (ADR 0080 invariant 2, ledger exhaustion is a hard stop) and nothing is
/// recorded.
pub fn finalize_round(
    config: &FedRoundConfig,
    dispatched: &[PeerId],
    collected: Vec<CollectedDelta>,
    ledger: &mut PrivacyLedger,
    owner: &KeyPair,
) -> Result<RoundOutcome, TrainError> {
    if config.min_clients < 2 {
        return Err(TrainError::other(
            "fed round: min_clients must be >= 2 (a federation of one isn't federated)",
        ));
    }
    // Keep only integrity-valid deltas; a bad-hash delta drops that peer.
    let valid: Vec<CollectedDelta> = collected.into_iter().filter(|d| d.verify()).collect();

    if valid.len() < config.min_clients {
        return Ok(RoundOutcome::Insufficient {
            got: valid.len(),
            needed: config.min_clients,
        });
    }

    // Debit the round's ε as a single signed ledger round. This RE-CHECKS the
    // budget (record → admit); exhaustion aborts the round with nothing spent.
    let total_epsilon: f64 = valid.iter().map(|d| d.epsilon_cost).sum();
    ledger.record(&config.corpus_id, total_epsilon, owner)?;

    let participants: Vec<PeerId> = valid.iter().map(|d| d.peer.clone()).collect();
    let dropped: Vec<PeerId> = dispatched
        .iter()
        .filter(|p| !participants.contains(p))
        .cloned()
        .collect();

    Ok(RoundOutcome::Aggregate {
        participants,
        dropped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dp() -> DpConfig {
        DpConfig {
            noise_multiplier: 1.1,
            max_grad_norm: 1.0,
            delta: 1e-5,
        }
    }

    fn config(min_clients: usize) -> FedRoundConfig {
        FedRoundConfig {
            corpus_id: "c".into(),
            min_clients,
            dp: dp(),
        }
    }

    fn ledger(budget: f64) -> (PrivacyLedger, KeyPair) {
        let kp = KeyPair::generate();
        let mut l = PrivacyLedger::new(PathBuf::from("/nonexistent/ledger.json"));
        l.set_budget("c", budget);
        (l, kp)
    }

    fn delta(peer: PeerId, bytes: &[u8], eps: f64) -> CollectedDelta {
        CollectedDelta {
            peer,
            claimed_hash: ContentHash::of_bytes(bytes),
            epsilon_cost: eps,
            bytes: bytes.to_vec(),
        }
    }

    /// A distinct random peer id.
    fn pid() -> PeerId {
        PeerId::from_pubkey(&KeyPair::generate().verifying)
    }

    #[test]
    fn aggregates_when_enough_valid_deltas_and_debits_epsilon() {
        let (mut l, kp) = ledger(10.0);
        let a = pid();
        let b = pid();
        let dispatched = vec![a.clone(), b.clone()];
        let collected = vec![
            delta(a.clone(), b"delta-a", 0.5),
            delta(b.clone(), b"delta-b", 0.5),
        ];
        let out = finalize_round(&config(2), &dispatched, collected, &mut l, &kp).unwrap();
        match out {
            RoundOutcome::Aggregate {
                participants,
                dropped,
            } => {
                assert_eq!(participants.len(), 2);
                assert!(dropped.is_empty());
            }
            other => panic!("expected Aggregate, got {other:?}"),
        }
        // ε debited (0.5 + 0.5 = 1.0).
        assert!((l.remaining("c") - 9.0).abs() < 1e-9);
    }

    #[test]
    fn insufficient_below_min_clients_does_not_debit() {
        let (mut l, kp) = ledger(10.0);
        let a = pid();
        let dispatched = vec![a.clone(), pid()];
        let collected = vec![delta(a, b"delta-a", 0.5)]; // only 1 arrived
        let out = finalize_round(&config(2), &dispatched, collected, &mut l, &kp).unwrap();
        assert_eq!(out, RoundOutcome::Insufficient { got: 1, needed: 2 });
        // No spend on an abandoned round.
        assert!((l.remaining("c") - 10.0).abs() < 1e-9);
    }

    #[test]
    fn corrupt_delta_is_dropped_not_fatal() {
        let (mut l, kp) = ledger(10.0);
        let a = pid();
        let b = pid();
        let c = pid();
        let dispatched = vec![a.clone(), b.clone(), c.clone()];
        // c's bytes don't match its claimed hash → dropped.
        let mut bad = delta(c.clone(), b"real", 0.5);
        bad.claimed_hash = ContentHash::of_bytes(b"a lie");
        let collected = vec![
            delta(a.clone(), b"da", 0.4),
            delta(b.clone(), b"db", 0.4),
            bad,
        ];
        let out = finalize_round(&config(2), &dispatched, collected, &mut l, &kp).unwrap();
        match out {
            RoundOutcome::Aggregate {
                participants,
                dropped,
            } => {
                assert_eq!(participants.len(), 2, "corrupt delta dropped");
                assert_eq!(dropped, vec![c], "c dropped as a straggler/corrupt");
            }
            other => panic!("expected Aggregate, got {other:?}"),
        }
        // Only the two valid deltas' ε debited.
        assert!((l.remaining("c") - 9.2).abs() < 1e-9);
    }

    #[test]
    fn ledger_exhaustion_aborts_the_round() {
        let (mut l, kp) = ledger(0.5); // budget too small for the round's cost
        let a = pid();
        let b = pid();
        let dispatched = vec![a.clone(), b.clone()];
        let collected = vec![
            delta(a, b"da", 0.4),
            delta(b, b"db", 0.4), // total 0.8 > 0.5
        ];
        let err = finalize_round(&config(2), &dispatched, collected, &mut l, &kp).unwrap_err();
        assert!(format!("{err}").contains("budget"));
        // Nothing spent — the whole round aborted (invariant 2).
        assert!((l.remaining("c") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn min_clients_below_two_is_rejected() {
        let (mut l, kp) = ledger(10.0);
        let a = pid();
        let err =
            finalize_round(&config(1), std::slice::from_ref(&a), vec![], &mut l, &kp).unwrap_err();
        assert!(format!("{err}").contains("min_clients"));
    }

    #[test]
    fn collected_delta_verify_detects_tamper() {
        let a = pid();
        let mut d = delta(a, b"payload", 0.1);
        assert!(d.verify());
        d.bytes = b"tampered".to_vec();
        assert!(!d.verify());
    }
}
