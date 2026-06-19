// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Sweep engine: expand sweep overrides into a cartesian product of
//! concrete configs, fingerprint each, and (stub) decide whether a combo
//! can be skipped because its output is already cached.
//!
//! The cartesian expansion is delegated to the native `hydra::expand_simple_sweeps`,
//! which handles `a=1,2,3` choice sweeps and `a=range(1,10)` range sweeps.

use crate::config::hydra::expand_simple_sweeps;

use crate::config::{ResolvedConfig, compose};
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
/// Thin wrapper over `hydra::expand_simple_sweeps`, which takes `&[&str]`.
/// `expand_simple_sweeps(&["lr=1e-3,1e-4", "bs=8,16"])` returns the 4-element
/// cartesian product directly.
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
    use crate::config::sweep_index;
    // Load the completion index ONCE for the whole sweep — every combo's skip
    // check is then a map lookup + sidecar stat, not a re-parse of the JSONL.
    let index = sweep_index::default_index_path()
        .map(|p| sweep_index::load_index(&p))
        .unwrap_or_default();
    let mut entries = Vec::new();
    for combo in cartesian(sweep_overrides) {
        let mut merged = base_overrides.to_vec();
        merged.extend(combo);
        let config = compose(config_dir, config_name, &merged)?;
        let fingerprint = config.fingerprint;
        let cache_skip = index
            .get(&fingerprint.to_hex())
            .is_some_and(sweep_index::is_record_live);
        entries.push(SweepEntry {
            overrides: merged,
            cache_skip,
            fingerprint,
            config,
        });
    }
    Ok(entries)
}

/// Whether this combo's output is already complete + still on disk, per the
/// global sweep-completion index (see [`crate::config::sweep_index`]). Returns
/// `false` (re-run) when the fingerprint was never recorded OR its recorded
/// output sidecar is gone / content-mismatched.
///
/// Single-fingerprint convenience — re-reads the index each call. [`expand`]
/// loads the index once and checks all combos against it; prefer that for a
/// whole sweep.
///
/// Wiring the SKIP into a run loop is still OUT OF LANE (the sweep engine is
/// not hooked into `framework::executor` / `jobs.rs`), and the index is only
/// populated once a sweep runner calls `sweep_index::record_completion` on
/// Done — until then this is always `false`, same observable behavior as the
/// old stub, but now backed by a real, tested index.
pub fn cache_skip(fp: ContentHash) -> bool {
    crate::config::sweep_index::is_complete(fp)
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
    fn cartesian_matches_hydra() {
        let combos = cartesian(&["lr=1e-3,1e-4".to_string(), "bs=8,16".to_string()]);
        assert_eq!(combos.len(), 4);
        assert!(combos.contains(&vec!["lr=1e-3".to_string(), "bs=8".to_string()]));
        assert!(combos.contains(&vec!["lr=1e-4".to_string(), "bs=16".to_string()]));
    }

    #[test]
    fn expand_produces_entry_per_combo() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "config.yaml", "opt:\n  lr: 0.1\n  bs: 4\n");
        // Dotted keys -> hydra treats these as VALUE overrides (group=config
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
    fn cache_skip_false_for_unrecorded_fingerprint() {
        // A fresh random fingerprint is never in the global index → re-run.
        // (Full skip semantics — record + sidecar liveness — are covered in
        // `sweep_index::tests`.)
        assert!(!cache_skip(ContentHash([0x5a; 32])));
    }

    #[test]
    fn expand_cache_skip_reflects_the_live_index() {
        // End-to-end: compose → expand reads the GLOBAL sweep-index, so a combo
        // recorded as complete (with a live sidecar) flips to cache_skip=true
        // while its sweep siblings still run. Points the index root at a tempdir
        // via $LAMU_TRAIN_CACHE_DIR (serialized by TEST_ENV_LOCK).
        use crate::config::sweep_index;
        use crate::framework::artifact::{ArtifactMetadata, ContentHash};

        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let cache = tempfile::tempdir().unwrap();
        let prev = std::env::var("LAMU_TRAIN_CACHE_DIR").ok();
        // SAFETY: TEST_ENV_LOCK serializes env mutation; restored below.
        unsafe {
            std::env::set_var("LAMU_TRAIN_CACHE_DIR", cache.path());
        }

        let cfg = tempfile::tempdir().unwrap();
        write_config(cfg.path(), "config.yaml", "opt:\n  lr: 0.1\n");
        let dir = cfg.path().to_str().unwrap();
        let sweep = ["opt.lr=1e-3,1e-4".to_string()];

        let before = expand(dir, "config", &[], &sweep).unwrap();
        assert_eq!(before.len(), 2);
        assert!(before.iter().all(|e| !e.cache_skip), "nothing recorded yet");
        let fp0 = before[0].fingerprint;

        // Record combo[0] complete with a live sidecar (hash must match).
        let sidecar = cache.path().join("out.metadata.json");
        ArtifactMetadata::new("ckpt", 1, ContentHash([9u8; 32]))
            .write_to(&sidecar)
            .unwrap();
        sweep_index::record_completion(fp0, "job-x", ContentHash([9u8; 32]), sidecar).unwrap();

        let after = expand(dir, "config", &[], &sweep).unwrap();
        let skip0 = after
            .iter()
            .find(|e| e.fingerprint.0 == fp0.0)
            .unwrap()
            .cache_skip;
        assert!(skip0, "recorded combo must now skip");
        assert_eq!(
            after.iter().filter(|e| !e.cache_skip).count(),
            1,
            "the sweep sibling still runs"
        );

        // SAFETY: restore under the same lock.
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_CACHE_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_CACHE_DIR"),
            }
        }
    }
}
