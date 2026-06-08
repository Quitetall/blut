//! Sweep engine: expand sweep overrides into a cartesian product of
//! concrete configs, fingerprint each, and (stub) decide whether a combo
//! can be skipped because its output is already cached.
//!
//! The cartesian expansion is delegated to lerna's `expand_simple_sweeps`,
//! which handles `a=1,2,3` choice sweeps and `a=range(1,10)` range sweeps.

use lerna::expand_simple_sweeps;

use crate::config::{compose, ResolvedConfig};
use crate::error::Result;
use crate::framework::artifact::ContentHash;

/// One concrete point in a sweep: the merged overrides that produced it,
/// the composed config, its fingerprint, and whether it can be skipped.
#[derive(Clone, Debug)]
pub struct SweepEntry {
    /// `base_overrides ++ <this combo>` — the exact overrides composed.
    pub overrides: Vec<String>,
    /// The composed + frozen config for this combo.
    pub config: ResolvedConfig,
    /// `config.fingerprint`, lifted for convenience at the entry level.
    pub fingerprint: ContentHash,
    /// Whether this combo's output is already cached (STUB: always false).
    pub cache_skip: bool,
}

/// Cartesian product of sweep override strings.
///
/// Thin wrapper over `lerna::expand_simple_sweeps`, which takes `&[&str]`.
/// `expand_simple_sweeps(&["lr=1e-3,1e-4", "bs=8,16"])` returns the 4-element
/// cartesian product directly.
///
/// Upgrade path for a validating sweep: `OverrideParser::parse_many` +
/// `lerna::expand_sweeps(&[Override])` gives typed parse + float/step-aware
/// `RangeSweep`. The simple form is sufficient for this skeleton.
pub fn cartesian(sweep_overrides: &[String]) -> Vec<Vec<String>> {
    let refs: Vec<&str> = sweep_overrides.iter().map(String::as_str).collect();
    expand_simple_sweeps(&refs)
}

/// Expand a sweep into one fingerprinted [`SweepEntry`] per combo.
///
/// For each combo in `cartesian(sweep_overrides)`, the combo is appended to
/// `base_overrides` and composed via [`compose`].
pub fn expand(
    config_dir: &str,
    config_name: &str,
    base_overrides: &[String],
    sweep_overrides: &[String],
) -> Result<Vec<SweepEntry>> {
    let mut entries = Vec::new();
    for combo in cartesian(sweep_overrides) {
        let mut merged = base_overrides.to_vec();
        merged.extend(combo);
        let config = compose(config_dir, config_name, &merged)?;
        let fingerprint = config.fingerprint;
        entries.push(SweepEntry {
            overrides: merged,
            cache_skip: cache_skip_stub(fingerprint),
            fingerprint,
            config,
        });
    }
    Ok(entries)
}

/// STUB — always re-runs (returns false).
///
/// The real implementation checks `framework::cache::CacheHandle::lookup(fp)`
/// against the global train-cache. Wiring the skip into the run loop is OUT OF
/// LANE (the `framework::executor` + `jobs.rs` seam); nobody should assume the
/// sweep engine is hooked into execution yet.
pub fn cache_skip_stub(_fp: ContentHash) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_config(dir: &std::path::Path, name: &str, body: &str) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    #[test]
    fn cartesian_matches_lerna() {
        let combos = cartesian(&["lr=1e-3,1e-4".to_string(), "bs=8,16".to_string()]);
        assert_eq!(combos.len(), 4);
        assert!(combos.contains(&vec!["lr=1e-3".to_string(), "bs=8".to_string()]));
        assert!(combos.contains(&vec!["lr=1e-4".to_string(), "bs=16".to_string()]));
    }

    #[test]
    fn expand_produces_entry_per_combo() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "opt:\n  lr: 0.1\n  bs: 4\n");
        // Dotted keys -> lerna treats these as VALUE overrides (group=config
        // selection only applies to dot-less keys against the defaults list).
        let entries = expand(
            dir.path().to_str().unwrap(),
            "config",
            &[],
            &["opt.lr=1e-3,1e-4".to_string(), "opt.bs=8,16".to_string()],
        )
        .unwrap();
        assert_eq!(entries.len(), 4);
        for e in &entries {
            // Every entry has a populated (non-zero) fingerprint.
            assert_ne!(e.fingerprint.0, [0u8; 32]);
            assert!(!e.cache_skip);
        }
        // Distinct combos yield distinct fingerprints.
        let mut fps: Vec<[u8; 32]> = entries.iter().map(|e| e.fingerprint.0).collect();
        fps.sort();
        fps.dedup();
        assert_eq!(fps.len(), 4);
    }

    #[test]
    fn cache_skip_stub_false() {
        assert!(!cache_skip_stub(ContentHash([7u8; 32])));
    }
}
