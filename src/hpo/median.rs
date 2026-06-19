//! Median / percentile early-stop (v0.20 Phase 3).
//!
//! Kill a trial whose score (higher = better; the scheduler already negated for
//! `minimize`) is below the p-th percentile of its peers at the same budget. A
//! grace period (in the scheduler) protects early, noisy budgets; `min_peers`
//! guards against deciding on too little data.

use super::scheduler::EarlyStop;

/// `percentile = 50` is the classic median rule; higher prunes more
/// aggressively (keep only the top `100 - percentile`%).
pub struct MedianStop {
    pub percentile: f64,
    pub min_peers: usize,
}

impl EarlyStop for MedianStop {
    fn should_stop(&self, _budget: u64, score: f64, peer_scores: &[f64]) -> bool {
        // Need a real cohort (and never < 2) before stopping anyone.
        if peer_scores.len() < self.min_peers.max(2) {
            return false;
        }
        let thr = percentile(peer_scores, self.percentile);
        score < thr // strictly below the percentile → kill; ties survive
    }

    fn name(&self) -> &'static str {
        "median"
    }
}

/// Nearest-rank p-th percentile of `xs` (ascending), `p` in `[0, 100]`. NaN
/// values are dropped first — a NaN would make `partial_cmp` non-total and the
/// sort (hence the threshold + the kill decision) non-deterministic. Shared with
/// ASHA (its per-rung cut is a percentile at `(1 - 1/eta)`).
pub(crate) fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = p.clamp(0.0, 100.0);
    let idx = ((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize;
    v[idx.min(v.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_endpoints_and_median() {
        let xs = [0.1, 0.5, 0.9];
        assert_eq!(percentile(&xs, 50.0), 0.5);
        assert_eq!(percentile(&xs, 0.0), 0.1);
        assert_eq!(percentile(&xs, 100.0), 0.9);
    }

    #[test]
    fn below_median_stops_at_or_above_survives() {
        let m = MedianStop {
            percentile: 50.0,
            min_peers: 2,
        };
        let peers = [0.1, 0.5, 0.9];
        assert!(m.should_stop(1, 0.1, &peers), "below median → kill");
        assert!(!m.should_stop(1, 0.5, &peers), "at median → survive");
        assert!(!m.should_stop(1, 0.9, &peers), "above median → survive");
        assert!(!m.should_stop(1, 0.1, &[0.1]), "too few peers → never stop");
    }

    #[test]
    fn higher_percentile_prunes_more() {
        // p=75 keeps only the top 25%; 0.5 (the median value) is now below the
        // 75th-percentile threshold (0.9) → killed.
        let m = MedianStop {
            percentile: 75.0,
            min_peers: 2,
        };
        let peers = [0.1, 0.5, 0.9];
        assert!(m.should_stop(1, 0.5, &peers));
    }
}
