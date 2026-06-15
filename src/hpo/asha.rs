//! ASHA — Asynchronous Successive Halving (v0.20 Phase 4), the headline
//! adaptive search.
//!
//! Trials run as parallel nodes; at each RUNG (a budget milestone
//! `min_budget · eta^k`), a trial reaching the rung is compared against all
//! trials that have reached it and KILL-ed unless it is in the top `1/eta`.
//! Survivors are simply NOT killed, so they keep training to `max_budget`
//! ("promotion = not killed") — no resume, no executor surgery; ASHA rides the
//! existing `KillBranch`. The per-rung "top `1/eta`" cut is exactly a percentile
//! threshold at `(1 − 1/eta)·100`, so it reuses [`super::median::percentile`].
//!
//! Async (vs synchronous SHA): a trial is judged the moment it reaches a rung
//! against whoever has reached it so far — no global barrier, so a slow trial
//! never stalls the cohort. This makes the decision mildly order-sensitive at
//! the cohort margin (a trial reaching a rung early, before enough peers, is not
//! culled until the quorum exists), which is the standard ASHA behavior.

use super::median;
use super::scheduler::EarlyStop;

/// ASHA early-stop strategy. Acts ONLY at rung budgets; elsewhere it never
/// stops (the trial keeps reporting between rungs).
pub struct AshaStop {
    /// Budget milestones at which to cull (strictly below max_budget — the top
    /// survivors run to max_budget without further culling).
    pub rungs: Vec<u64>,
    /// Reduction factor: keep the top `1/eta` at each rung (≥ 2).
    pub eta: u32,
    /// Need at least this many peers at a rung before culling anyone.
    pub min_peers: usize,
}

impl AshaStop {
    /// The rung ladder: `min_budget · eta^k` for k = 0,1,… while strictly below
    /// `max_budget`. `max_budget` itself is NOT a rung (survivors run to it).
    pub fn rung_ladder(min_budget: u64, max_budget: u64, eta: u32) -> Vec<u64> {
        let eta = eta.max(2) as u64;
        let mut rungs = Vec::new();
        let mut b = min_budget.max(1);
        // Guard against a degenerate ladder (eta or budgets that don't grow).
        while b < max_budget && rungs.len() < 64 {
            rungs.push(b);
            let next = b.saturating_mul(eta);
            if next == b {
                break;
            }
            b = next;
        }
        rungs
    }

    /// Convenience: build an `AshaStop` from the CLI knobs.
    pub fn from_budgets(min_budget: u64, max_budget: u64, eta: u32) -> Self {
        let eta = eta.max(2);
        AshaStop {
            rungs: Self::rung_ladder(min_budget, max_budget, eta),
            eta,
            min_peers: eta as usize,
        }
    }
}

impl EarlyStop for AshaStop {
    fn should_stop(&self, budget: u64, score: f64, peer_scores: &[f64]) -> bool {
        // Only cull at a rung milestone.
        if !self.rungs.contains(&budget) {
            return false;
        }
        let eta = self.eta.max(2) as f64;
        // Need a real cohort (≥ eta, so keeping 1/eta keeps ≥ 1).
        if peer_scores.len() < self.min_peers.max(eta as usize) {
            return false;
        }
        // Keep the top 1/eta ⇒ kill below the (1 − 1/eta) percentile.
        let keep_pct = (1.0 - 1.0 / eta) * 100.0;
        let thr = median::percentile(peer_scores, keep_pct);
        score < thr
    }

    fn name(&self) -> &'static str {
        "asha"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rung_ladder_geometric_below_max() {
        // min=1, max=27, eta=3 → 1, 3, 9 (27 is max, not a rung).
        assert_eq!(AshaStop::rung_ladder(1, 27, 3), vec![1, 3, 9]);
        // min=2, max=16, eta=2 → 2, 4, 8.
        assert_eq!(AshaStop::rung_ladder(2, 16, 2), vec![2, 4, 8]);
        // max <= min → empty ladder.
        assert!(AshaStop::rung_ladder(8, 8, 3).is_empty());
    }

    #[test]
    fn culls_bottom_at_rung_keeps_top_fraction() {
        // eta=3: keep top 1/3, kill below the 66.7th percentile. 6 peers at a rung.
        let a = AshaStop::from_budgets(1, 9, 3); // rungs [1,3]
        let peers = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]; // scores, higher=better
        // At rung 1: a bottom trial (0.2) is below the keep threshold → killed.
        assert!(a.should_stop(1, 0.2, &peers), "bottom culled at rung");
        // A top trial (0.9) survives.
        assert!(!a.should_stop(1, 0.9, &peers), "top survives at rung");
        // NOT at a rung (budget 2 ∉ {1,3}) → never cull.
        assert!(!a.should_stop(2, 0.2, &peers), "off-rung never culls");
        // Too few peers → no cull even at a rung.
        assert!(!a.should_stop(1, 0.2, &[0.2, 0.9]));
    }
}
