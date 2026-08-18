// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Differential-privacy budget ledger + gradient-dispatch gate (ADR 0080 C1).
//!
//! This is the BRAKE for async federated training — built first, before any
//! gradient ever moves. Two pieces:
//!
//! - [`PrivacyLedger`]: a per-corpus ε (epsilon) budget with an append-only,
//!   Ed25519-signed round chain (tamper-evident). [`admit`] refuses a round
//!   that would exceed the budget; [`record`] appends a signed round and debits
//!   the spend. The ε COST of a round is computed in Python by Opacus's RDP
//!   accountant — Rust only STORES + ENFORCES the accounted numbers (charter:
//!   delegate the DP math to the domain layer; the engine owns the gate).
//! - [`gate_gradient_dispatch`]: the fail-closed policy. A gradient-bearing
//!   task is DENIED unless `data_class ∈ {Public, Internal}` AND a DP config is
//!   present with non-zero noise AND the ledger admits the cost. `Restricted`
//!   gradients NEVER dispatch — no DP override (HIPAA; ADR 0061). Unknown
//!   gradient status is treated as carrying gradients (fail-closed).
//!
//! [`admit`]: PrivacyLedger::admit
//! [`record`]: PrivacyLedger::record

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::error::TrainError;
use crate::framework::artifact::ContentHash;
use crate::p2p::crypto::{KeyPair, verify};
use crate::p2p::trust::DataClass;

/// Domain-separation tag for a ledger round's signature.
const LEDGER_SIG_DOMAIN: &[u8] = b"blut-privacy-ledger-v1";
const LEDGER_SIG_DOMAIN_V2: &[u8] = b"blut-privacy-ledger-v2";
const LEDGER_SCHEMA_V2: u32 = 2;

/// The DP configuration a gradient-bearing task must carry. `noise_multiplier`
/// is the Gaussian-mechanism σ; it MUST be > 0 (DP is mandatory — zero noise is
/// no privacy). The accounted ε is produced by Opacus, not from these fields.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DpConfig {
    /// Gaussian noise multiplier (σ). Must be > 0.
    pub noise_multiplier: f64,
    /// Per-sample gradient clipping norm.
    pub max_grad_norm: f64,
    /// Target δ for the (ε, δ) guarantee.
    pub delta: f64,
}

/// Whether a task carries gradients / model-update state. `Unknown` is treated
/// as `Yes` by the gate (fail-closed) — a task that can't prove it's
/// gradient-free is assumed to carry gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientStatus {
    Yes,
    No,
    Unknown,
}

/// One append-only, signed ledger round.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LedgerRound {
    pub corpus_id: String,
    /// ε spent this round (Opacus-accounted).
    pub epsilon_cost: f64,
    /// Monotonic per-ledger index (0-based).
    pub index: u64,
    /// Hash of the previous round (chains the log; the genesis prev is zero).
    pub prev_hash: ContentHash,
    /// Ed25519 signature by the ledger owner over this round.
    #[serde(with = "sig_hex")]
    pub signature: Signature,
}

impl LedgerRound {
    fn signing_bytes(
        corpus_id: &str,
        epsilon_cost: f64,
        index: u64,
        prev_hash: &ContentHash,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(LEDGER_SIG_DOMAIN);
        buf.extend_from_slice(&(corpus_id.len() as u64).to_le_bytes());
        buf.extend_from_slice(corpus_id.as_bytes());
        buf.extend_from_slice(&epsilon_cost.to_le_bytes());
        buf.extend_from_slice(&index.to_le_bytes());
        buf.extend_from_slice(&prev_hash.0);
        buf
    }

    fn signing_bytes_v2(
        tenant: &str,
        corpus_id: &str,
        epsilon_cost: f64,
        index: u64,
        prev_hash: &ContentHash,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(LEDGER_SIG_DOMAIN_V2);
        buf.extend_from_slice(&(tenant.len() as u64).to_le_bytes());
        buf.extend_from_slice(tenant.as_bytes());
        buf.extend_from_slice(&(corpus_id.len() as u64).to_le_bytes());
        buf.extend_from_slice(corpus_id.as_bytes());
        buf.extend_from_slice(&epsilon_cost.to_le_bytes());
        buf.extend_from_slice(&index.to_le_bytes());
        buf.extend_from_slice(&prev_hash.0);
        buf
    }

    fn signing_bytes_for(
        schema_version: u32,
        tenant: &str,
        corpus_id: &str,
        epsilon_cost: f64,
        index: u64,
        prev_hash: &ContentHash,
    ) -> Result<Vec<u8>, TrainError> {
        match schema_version {
            1 => Ok(Self::signing_bytes(
                corpus_id,
                epsilon_cost,
                index,
                prev_hash,
            )),
            LEDGER_SCHEMA_V2 => Ok(Self::signing_bytes_v2(
                tenant,
                corpus_id,
                epsilon_cost,
                index,
                prev_hash,
            )),
            other => Err(TrainError::other(format!(
                "privacy ledger schema v{other} is unsupported"
            ))),
        }
    }

    /// This round's hash, chaining the next round's `prev_hash`.
    fn hash(&self, schema_version: u32, tenant: &str) -> Result<ContentHash, TrainError> {
        let mut bytes = Self::signing_bytes_for(
            schema_version,
            tenant,
            &self.corpus_id,
            self.epsilon_cost,
            self.index,
            &self.prev_hash,
        )?;
        bytes.extend_from_slice(&self.signature.to_bytes());
        Ok(ContentHash::of_bytes(&bytes))
    }
}

/// Per-corpus ε budget.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CorpusBudget {
    /// Total ε the corpus may ever spend.
    pub epsilon_budget: f64,
    /// ε spent so far (derived from the round chain; persisted for convenience).
    pub epsilon_spent: f64,
}

/// A per-corpus differential-privacy budget ledger, tamper-evident via a signed
/// append-only round chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrivacyLedger {
    #[serde(skip)]
    path: PathBuf,
    /// v1 omitted tenant from signed rounds. New ledgers use v2, which binds
    /// every round and hash-chain edge to the owning tenant. Missing = legacy v1.
    #[serde(default = "legacy_ledger_schema")]
    schema_version: u32,
    /// Owning tenant (`project[/domain]`). Missing on legacy ledgers means the
    /// flat `default` namespace. Enforcement loaders require an exact match.
    #[serde(default = "default_tenant_label")]
    tenant: String,
    budgets: HashMap<String, CorpusBudget>,
    rounds: Vec<LedgerRound>,
}

fn default_tenant_label() -> String {
    crate::tenant::Tenant::default().to_string()
}

fn legacy_ledger_schema() -> u32 {
    1
}

impl PrivacyLedger {
    /// A new empty ledger backed by `path`.
    pub fn new(path: PathBuf) -> Self {
        Self::new_for_tenant(path, &crate::tenant::Tenant::default())
    }

    /// A new empty ledger bound to `tenant`.
    pub fn new_for_tenant(path: PathBuf, tenant: &crate::tenant::Tenant) -> Self {
        Self {
            path,
            schema_version: LEDGER_SCHEMA_V2,
            tenant: tenant.to_string(),
            budgets: HashMap::new(),
            rounds: Vec::new(),
        }
    }

    /// Canonical ledger path below a P2P state root. `default` keeps the legacy
    /// flat path; every other tenant gets a disjoint namespace prefix.
    pub fn path_for_tenant(root: &Path, tenant: &crate::tenant::Tenant) -> PathBuf {
        if tenant.is_default() {
            root.join("privacy_ledger.json")
        } else {
            root.join(tenant.as_path()).join("privacy_ledger.json")
        }
    }

    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    /// Load a ledger from disk (empty if absent), VERIFYING the signed chain
    /// against `owner`. A tampered or forged chain is rejected fail-closed.
    pub fn load(path: PathBuf, owner: &VerifyingKey) -> Result<Self, TrainError> {
        Self::load_for_tenant(path, owner, &crate::tenant::Tenant::default())
    }

    /// Load and verify a ledger for an exact tenant. A valid signed chain under
    /// another tenant is still refused: cryptographic ownership does not grant a
    /// cross-namespace read.
    pub fn load_for_tenant(
        path: PathBuf,
        owner: &VerifyingKey,
        expected_tenant: &crate::tenant::Tenant,
    ) -> Result<Self, TrainError> {
        if !path.exists() {
            return Ok(Self::new_for_tenant(path, expected_tenant));
        }
        let data = std::fs::read_to_string(&path).map_err(|e| TrainError::Io {
            path: path.clone(),
            source: e,
        })?;
        let mut ledger: Self = serde_json::from_str(&data).map_err(|e| {
            TrainError::other(format!("corrupt privacy ledger {}: {e}", path.display()))
        })?;
        ledger.path = path;
        ledger.verify_tenant(expected_tenant)?;
        ledger.verify_chain(owner)?;
        Ok(ledger)
    }

    /// Set a corpus's ε CEILING. This adjusts only `epsilon_budget`; it does
    /// NOT reset `epsilon_spent` and does NOT touch the signed round chain (so
    /// raising the ceiling can't erase history). A fresh privacy epoch (reset
    /// spend) is a deliberate, separate operator action, not provided here.
    pub fn set_budget(&mut self, corpus_id: &str, epsilon_budget: f64) {
        let entry = self
            .budgets
            .entry(corpus_id.to_string())
            .or_insert(CorpusBudget {
                epsilon_budget,
                epsilon_spent: 0.0,
            });
        entry.epsilon_budget = epsilon_budget;
    }

    /// Load a ledger for READ-ONLY display (e.g. the console) WITHOUT verifying
    /// the signed chain — the numbers are shown, not enforced. Any code that
    /// ENFORCES the budget must go through [`load`](Self::load) +
    /// [`verify_chain`](Self::verify_chain) instead.
    pub fn load_readonly(path: PathBuf) -> Result<Self, TrainError> {
        Self::load_readonly_for_tenant(path, &crate::tenant::Tenant::default())
    }

    /// Read-only display load with the same tenant boundary as enforcement.
    pub fn load_readonly_for_tenant(
        path: PathBuf,
        expected_tenant: &crate::tenant::Tenant,
    ) -> Result<Self, TrainError> {
        if !path.exists() {
            return Ok(Self::new_for_tenant(path, expected_tenant));
        }
        let data = std::fs::read_to_string(&path).map_err(|e| TrainError::Io {
            path: path.clone(),
            source: e,
        })?;
        let mut ledger: Self = serde_json::from_str(&data).map_err(|e| {
            TrainError::other(format!("corrupt privacy ledger {}: {e}", path.display()))
        })?;
        ledger.path = path;
        ledger.verify_tenant(expected_tenant)?;
        Ok(ledger)
    }

    fn verify_tenant(&self, expected: &crate::tenant::Tenant) -> Result<(), TrainError> {
        let actual = crate::tenant::Tenant::parse(&self.tenant)
            .filter(|tenant| tenant.to_string() == self.tenant)
            .ok_or_else(|| {
                TrainError::other(format!(
                    "privacy ledger has invalid tenant '{}' (fail-closed)",
                    self.tenant
                ))
            })?;
        if &actual != expected {
            return Err(TrainError::other(format!(
                "privacy ledger tenant mismatch: stored '{actual}' != requested '{expected}' (cross-tenant read denied)"
            )));
        }
        Ok(())
    }

    /// `(corpus_id, ε spent, ε budget)` for every known corpus — for read-only
    /// display. Order is unspecified (a `HashMap`); callers sort if needed.
    pub fn corpus_summaries(&self) -> Vec<(String, f64, f64)> {
        self.budgets
            .iter()
            .map(|(id, b)| (id.clone(), b.epsilon_spent, b.epsilon_budget))
            .collect()
    }

    /// ε remaining for a corpus (0 if unknown — fail-closed: no budget set means
    /// nothing may be spent).
    pub fn remaining(&self, corpus_id: &str) -> f64 {
        self.budgets
            .get(corpus_id)
            .map(|b| (b.epsilon_budget - b.epsilon_spent).max(0.0))
            .unwrap_or(0.0)
    }

    /// Whether spending `cost` more ε on `corpus_id` stays within budget. Refuses
    /// a non-finite / negative cost and an unknown corpus (fail-closed).
    pub fn admit(&self, corpus_id: &str, cost: f64) -> Result<(), TrainError> {
        if !cost.is_finite() || cost < 0.0 {
            return Err(TrainError::other(format!(
                "privacy: invalid epsilon cost {cost}"
            )));
        }
        let Some(b) = self.budgets.get(corpus_id) else {
            return Err(TrainError::other(format!(
                "privacy: no budget set for corpus '{corpus_id}' (fail-closed)"
            )));
        };
        if b.epsilon_spent + cost > b.epsilon_budget {
            return Err(TrainError::other(format!(
                "privacy: corpus '{corpus_id}' budget exhausted \
                 (spent {:.4} + {:.4} > {:.4})",
                b.epsilon_spent, cost, b.epsilon_budget
            )));
        }
        Ok(())
    }

    /// Admit + append a signed round + debit the spend. Atomic in memory; call
    /// [`save`](Self::save) to persist. A round that isn't admitted is NOT
    /// recorded (the budget is never over-spent).
    pub fn record(
        &mut self,
        corpus_id: &str,
        cost: f64,
        owner: &KeyPair,
    ) -> Result<(), TrainError> {
        self.admit(corpus_id, cost)?;
        let index = self.rounds.len() as u64;
        let prev_hash = if let Some(round) = self.rounds.last() {
            round.hash(self.schema_version, &self.tenant)?
        } else {
            ContentHash([0u8; 32])
        };
        let signing_bytes = LedgerRound::signing_bytes_for(
            self.schema_version,
            &self.tenant,
            corpus_id,
            cost,
            index,
            &prev_hash,
        )?;
        let sig = owner.sign(&signing_bytes);
        self.rounds.push(LedgerRound {
            corpus_id: corpus_id.to_string(),
            epsilon_cost: cost,
            index,
            prev_hash,
            signature: sig,
        });
        // Debit (the corpus is known — admit() checked).
        if let Some(b) = self.budgets.get_mut(corpus_id) {
            b.epsilon_spent += cost;
        }
        Ok(())
    }

    /// Verify the whole signed chain against `owner`: every round's signature,
    /// its monotonic index, and its `prev_hash` linkage. Any break ⇒ error.
    pub fn verify_chain(&self, owner: &VerifyingKey) -> Result<(), TrainError> {
        if !matches!(self.schema_version, 1 | LEDGER_SCHEMA_V2) {
            return Err(TrainError::other(format!(
                "privacy ledger schema v{} is unsupported",
                self.schema_version
            )));
        }
        let mut prev = ContentHash([0u8; 32]);
        for (i, round) in self.rounds.iter().enumerate() {
            if round.index != i as u64 {
                return Err(TrainError::other(format!(
                    "privacy ledger: round {i} has wrong index {}",
                    round.index
                )));
            }
            if round.prev_hash != prev {
                return Err(TrainError::other(format!(
                    "privacy ledger: round {i} prev_hash chain break"
                )));
            }
            let bytes = LedgerRound::signing_bytes_for(
                self.schema_version,
                &self.tenant,
                &round.corpus_id,
                round.epsilon_cost,
                round.index,
                &round.prev_hash,
            )?;
            if !verify(owner, &bytes, &round.signature) {
                return Err(TrainError::other(format!(
                    "privacy ledger: round {i} signature invalid (tampered?)"
                )));
            }
            prev = round.hash(self.schema_version, &self.tenant)?;
        }
        Ok(())
    }

    /// Persist to disk (atomic temp + rename).
    pub fn save(&self) -> Result<(), TrainError> {
        let tenant = crate::tenant::Tenant::parse(&self.tenant).ok_or_else(|| {
            TrainError::other(format!(
                "privacy ledger has invalid tenant '{}' (refusing save)",
                self.tenant
            ))
        })?;
        self.verify_tenant(&tenant)?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| TrainError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TrainError::other(format!("serialize privacy ledger: {e}")))?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, &json).map_err(|e| TrainError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        // fsync before rename: a security ledger must be durable, so a crash
        // between write and rename can't silently drop committed rounds (same
        // discipline as the peer registry / cache write-atomic).
        let f = std::fs::File::open(&tmp).map_err(|e| TrainError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.sync_all().map_err(|e| TrainError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp, &self.path).map_err(|e| TrainError::Io {
            path: self.path.clone(),
            source: e,
        })
    }
}

/// The fail-closed gradient-dispatch gate (ADR 0080). Returns `Ok(())` only if
/// the gradient-bearing task may leave the box. See the module docs.
///
/// CONTRACT: this is a PRE-FLIGHT policy check against a shared `&ledger` — it
/// does NOT debit the budget. The authoritative atomic admit+debit is
/// [`PrivacyLedger::record`], which RE-CHECKS the budget under `&mut self`; call
/// it once the round's ε cost is known so two concurrent gates can't both pass
/// and then over-spend. The ledger is never over-drawn because `record`
/// re-admits — the gate is the policy gate (data class / DP), `record` is the
/// budget gate.
pub fn gate_gradient_dispatch(
    data_class: DataClass,
    carries_gradients: GradientStatus,
    dp: Option<&DpConfig>,
    ledger: &PrivacyLedger,
    corpus_id: &str,
    epsilon_cost: f64,
) -> Result<(), TrainError> {
    // A task that isn't (provably) carrying gradients isn't gated here. Unknown
    // ⇒ assume it does (fail-closed).
    let carries = matches!(
        carries_gradients,
        GradientStatus::Yes | GradientStatus::Unknown
    );
    if !carries {
        return Ok(());
    }

    // HARD block: Restricted gradients NEVER dispatch — no DP override (HIPAA;
    // ADR 0061 / 0080 invariant 1).
    if data_class == DataClass::Restricted {
        return Err(TrainError::other(
            "gradient dispatch DENIED: Restricted-class gradients never leave the box \
             (ADR 0080/0061 clinical hard-block; no DP override)",
        ));
    }
    if !matches!(data_class, DataClass::Public | DataClass::Internal) {
        return Err(TrainError::other(
            "gradient dispatch DENIED: only Public/Internal data classes are eligible",
        ));
    }

    // DP is mandatory for any gradient-bearing dispatch.
    let Some(dp) = dp else {
        return Err(TrainError::other(
            "gradient dispatch DENIED: a DP config is required (ADR 0080 invariant 5)",
        ));
    };
    // Fail-closed on NaN/inf too (a non-finite σ is not a valid noise level).
    if !dp.noise_multiplier.is_finite() || dp.noise_multiplier <= 0.0 {
        return Err(TrainError::other(
            "gradient dispatch DENIED: noise_multiplier must be finite and > 0 (DP is mandatory)",
        ));
    }
    // δ must be a valid (finite, positive) DP parameter — δ ≤ 0 is meaningless.
    if !dp.delta.is_finite() || dp.delta <= 0.0 {
        return Err(TrainError::other(
            "gradient dispatch DENIED: delta must be finite and > 0",
        ));
    }

    // The ledger must admit the ε cost.
    ledger
        .admit(corpus_id, epsilon_cost)
        .map_err(|e| TrainError::other(format!("gradient dispatch DENIED: {e}")))
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

    fn dp() -> DpConfig {
        DpConfig {
            noise_multiplier: 1.1,
            max_grad_norm: 1.0,
            delta: 1e-5,
        }
    }

    fn ledger_with(corpus: &str, budget: f64) -> (PrivacyLedger, KeyPair) {
        // These tests never call save(), so the path is never written — no
        // tempdir needed (and nothing to leak).
        let kp = KeyPair::generate();
        let mut l = PrivacyLedger::new(PathBuf::from("/nonexistent/ledger.json"));
        l.set_budget(corpus, budget);
        (l, kp)
    }

    #[test]
    fn admit_and_record_debits_budget() {
        let (mut l, kp) = ledger_with("c", 1.0);
        assert!(l.admit("c", 0.4).is_ok());
        l.record("c", 0.4, &kp).unwrap();
        assert!((l.remaining("c") - 0.6).abs() < 1e-9);
        l.record("c", 0.5, &kp).unwrap();
        assert!((l.remaining("c") - 0.1).abs() < 1e-9);
    }

    #[test]
    fn exhaustion_is_a_hard_stop() {
        let (mut l, kp) = ledger_with("c", 1.0);
        l.record("c", 0.9, &kp).unwrap();
        // 0.9 + 0.2 > 1.0 → refused, and NOT recorded.
        assert!(l.admit("c", 0.2).is_err());
        assert!(l.record("c", 0.2, &kp).is_err());
        assert!(
            (l.remaining("c") - 0.1).abs() < 1e-9,
            "spend unchanged after refusal"
        );
    }

    #[test]
    fn unknown_corpus_is_fail_closed() {
        let (l, _kp) = ledger_with("c", 1.0);
        assert!(l.admit("other", 0.01).is_err());
        assert_eq!(l.remaining("other"), 0.0);
    }

    #[test]
    fn restricted_gradients_denied_even_with_dp() {
        let (l, _kp) = ledger_with("c", 100.0);
        let v = gate_gradient_dispatch(
            DataClass::Restricted,
            GradientStatus::Yes,
            Some(&dp()),
            &l,
            "c",
            0.1,
        );
        assert!(
            v.is_err(),
            "Restricted + gradients must be denied even with DP + budget"
        );
        assert!(format!("{}", v.unwrap_err()).contains("Restricted"));
    }

    #[test]
    fn unknown_gradient_status_is_treated_as_carrying() {
        let (l, _kp) = ledger_with("c", 100.0);
        // Unknown + Restricted → denied (fail-closed: assumed to carry).
        assert!(
            gate_gradient_dispatch(
                DataClass::Restricted,
                GradientStatus::Unknown,
                Some(&dp()),
                &l,
                "c",
                0.1
            )
            .is_err()
        );
        // Unknown + Public without DP → denied.
        assert!(
            gate_gradient_dispatch(
                DataClass::Public,
                GradientStatus::Unknown,
                None,
                &l,
                "c",
                0.1
            )
            .is_err()
        );
    }

    #[test]
    fn non_gradient_task_is_not_gated() {
        let (l, _kp) = ledger_with("c", 0.0); // no budget
        assert!(
            gate_gradient_dispatch(
                DataClass::Restricted,
                GradientStatus::No,
                None,
                &l,
                "c",
                0.0
            )
            .is_ok()
        );
    }

    #[test]
    fn dp_is_mandatory_and_noise_must_be_positive() {
        let (l, _kp) = ledger_with("c", 100.0);
        // No DP config → denied.
        assert!(
            gate_gradient_dispatch(DataClass::Public, GradientStatus::Yes, None, &l, "c", 0.1)
                .is_err()
        );
        // Zero noise → denied.
        let zero = DpConfig {
            noise_multiplier: 0.0,
            ..dp()
        };
        assert!(
            gate_gradient_dispatch(
                DataClass::Public,
                GradientStatus::Yes,
                Some(&zero),
                &l,
                "c",
                0.1
            )
            .is_err()
        );
    }

    #[test]
    fn public_gradient_within_budget_is_allowed() {
        let (l, _kp) = ledger_with("c", 1.0);
        assert!(
            gate_gradient_dispatch(
                DataClass::Public,
                GradientStatus::Yes,
                Some(&dp()),
                &l,
                "c",
                0.5
            )
            .is_ok()
        );
        // But over-budget is denied.
        assert!(
            gate_gradient_dispatch(
                DataClass::Public,
                GradientStatus::Yes,
                Some(&dp()),
                &l,
                "c",
                1.5
            )
            .is_err()
        );
    }

    #[test]
    fn tampered_chain_is_detected_on_verify() {
        let (mut l, kp) = ledger_with("c", 10.0);
        l.record("c", 1.0, &kp).unwrap();
        l.record("c", 2.0, &kp).unwrap();
        assert!(l.verify_chain(&kp.verifying).is_ok());
        // Tamper with a recorded cost → signature no longer matches.
        l.rounds[0].epsilon_cost = 0.0;
        assert!(l.verify_chain(&kp.verifying).is_err());
    }

    #[test]
    fn round_chain_persists_and_reloads_verified() {
        let kp = KeyPair::generate();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("ledger.json");
        {
            let mut l = PrivacyLedger::new(path.clone());
            l.set_budget("c", 5.0);
            l.record("c", 1.0, &kp).unwrap();
            l.record("c", 1.5, &kp).unwrap();
            l.save().unwrap();
        }
        // Reload verifies the chain against the owner (wrong-owner rejection is
        // covered by `reload_with_wrong_owner_is_rejected`).
        let l = PrivacyLedger::load(path, &kp.verifying).unwrap();
        assert_eq!(l.rounds.len(), 2);
    }

    #[test]
    fn legacy_v1_default_ledger_still_reloads() {
        let kp = KeyPair::generate();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("legacy-ledger.json");
        let mut ledger = PrivacyLedger::new(path.clone());
        ledger.schema_version = 1;
        ledger.set_budget("c", 2.0);
        ledger.record("c", 0.5, &kp).unwrap();
        ledger.save().unwrap();

        let loaded = PrivacyLedger::load(path, &kp.verifying).unwrap();
        assert_eq!(loaded.schema_version, 1);
        assert_eq!(loaded.tenant(), "default");
        assert_eq!(loaded.rounds.len(), 1);
    }

    #[test]
    fn reload_with_wrong_owner_is_rejected() {
        let kp = KeyPair::generate();
        let wrong = KeyPair::generate();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("ledger.json");
        let mut l = PrivacyLedger::new(path.clone());
        l.set_budget("c", 5.0);
        l.record("c", 1.0, &kp).unwrap();
        l.save().unwrap();
        assert!(PrivacyLedger::load(path, &wrong.verifying).is_err());
    }

    #[test]
    fn ledger_is_tenant_bound_and_cross_tenant_load_is_refused() {
        let kp = KeyPair::generate();
        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("p2p");
        let clinical = crate::tenant::Tenant::parse("clinical/prod").unwrap();
        let research = crate::tenant::Tenant::parse("research/dev").unwrap();
        let path = PrivacyLedger::path_for_tenant(&root, &clinical);
        assert!(path.ends_with("clinical/prod/privacy_ledger.json"));

        let mut ledger = PrivacyLedger::new_for_tenant(path.clone(), &clinical);
        ledger.set_budget("corpus", 5.0);
        ledger.record("corpus", 1.0, &kp).unwrap();
        ledger.save().unwrap();

        let loaded = PrivacyLedger::load_for_tenant(path.clone(), &kp.verifying, &clinical)
            .expect("owning tenant can load its ledger");
        assert_eq!(loaded.tenant(), "clinical/prod");
        let mut relabeled = loaded.clone();
        relabeled.tenant = "research/dev".into();
        assert!(
            relabeled.verify_chain(&kp.verifying).is_err(),
            "v2 round signatures must bind the tenant label"
        );
        assert!(
            PrivacyLedger::load_for_tenant(path, &kp.verifying, &research).is_err(),
            "a tenant ledger must never load into another tenant's namespace"
        );
    }
}
