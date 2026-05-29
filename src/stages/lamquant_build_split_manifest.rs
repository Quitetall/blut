//! Stage — `lamquant_build_split_manifest` (RCP-3). LmaCorpus → SplitManifest.
//!
//! Wraps `ai_models/dataset_sim/build_seizure_split_manifest.py`,
//! which emits the patient-level, seizure-stratified train/val split
//! (`split_manifest.json`) that LMA-direct SNN training hard-requires.
//! Every `lamquant_train_mamba_snn` run consumes a `--split-manifest`;
//! before this stage existed there was no way to PRODUCE one inside a
//! recipe, so the full pipeline could not be self-contained.
//!
//! The split is patient-level (a subject's recordings land in exactly
//! one split — no seizure leakage), seizure-stratified (seizure-bearing
//! and non-seizure subjects split independently at `--val-fraction`),
//! and RNG-free (`sha1(subject_id) % 1000`), so it is reproducible
//! across machines. Only stems that BOTH have an encoded `.lma` under
//! `--lma-root` AND a `<stem>_labels.npz` under `--labels` are
//! included.
//!
//! Deterministic — the split assignment is a pure function of the
//! subject ids present in the corpus + labels and the val-fraction; no
//! seed, no RNG. `DETERMINISTIC = true`.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{LmaCorpus, SplitManifest};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, blut_python_script, blut_pythonpath, progress_forwarder, python_for, resolve_home,
};

pub struct LamquantBuildSplitManifest;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty → detected `ai_models_root`
    /// (the `LamQuant-Neural` submodule).
    #[serde(default)]
    pub lamquant_home: String,
    /// `--lma-root <dir>` — the encoded LMA corpus root. Empty = use
    /// the upstream `LmaCorpus` input's `root` (the corpus produced by
    /// an upstream `encode_lma` / `convert_lma` stage).
    #[serde(default)]
    pub lma_root: String,
    /// `--labels <dir>` — per-stem `<stem>_labels.npz` directory.
    /// Empty = the unified canonical labels root
    /// (`crate::paths::DEFAULT_LABELS_DIR`, RCP-6).
    #[serde(default)]
    pub labels_dir: String,
    /// `--out <path>` — where to write `split_manifest.json`. Required.
    pub out: PathBuf,
    /// `--val-fraction` — proportion of subjects assigned to val,
    /// applied independently within each seizure bucket.
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,
}

fn default_val_fraction() -> f32 {
    0.10
}

#[async_trait]
impl Stage for LamquantBuildSplitManifest {
    const NAME: &'static str = "lamquant_build_split_manifest";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu, Resource::Disk];
    const DETERMINISTIC: bool = true;
    type Input = LmaCorpus;
    type Output = SplitManifest;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: LmaCorpus,
        args: &Args,
    ) -> Result<SplitManifest, StageError> {
        if args.out.as_os_str().is_empty() {
            return Err(StageError::BadInput(
                "out is required for lamquant_build_split_manifest (path for split_manifest.json)"
                    .into(),
            ));
        }
        if !(args.val_fraction > 0.0 && args.val_fraction < 1.0 && args.val_fraction.is_finite()) {
            return Err(StageError::BadInput(format!(
                "val_fraction must be in (0,1); got {}",
                args.val_fraction
            )));
        }

        // MOVE-B (2026-05-29): the split-manifest builder now lives in
        // the PUBLIC BLUT submodule at
        // `<blut>/python/lamquant/dataset/build_seizure_split_manifest.py`,
        // resolved via `blut_python_root` ($BLUT_PYTHON). `home` is still
        // resolved for `python_for` interpreter selection.
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let (script, python_dir) = blut_python_script(&[
            "python",
            "lamquant",
            "dataset",
            "build_seizure_split_manifest.py",
        ])?;

        // `--lma-root`: explicit override, else the upstream corpus's
        // root (the LmaCorpus produced by encode_lma / convert_lma).
        let lma_root = if args.lma_root.is_empty() {
            input.root.display().to_string()
        } else {
            args.lma_root.clone()
        };
        // `--labels`: explicit override, else the unified canonical
        // labels root (RCP-6).
        let labels_dir = if args.labels_dir.is_empty() {
            crate::paths::DEFAULT_LABELS_DIR.to_string()
        } else {
            args.labels_dir.clone()
        };

        // Ensure the parent dir of the manifest exists before the
        // script writes it (the script opens --out for writing).
        if let Some(parent) = args.out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }

        let cmd_args: Vec<String> = vec![
            "--lma-root".into(),
            lma_root,
            "--labels".into(),
            labels_dir,
            "--out".into(),
            args.out.display().to_string(),
            "--val-fraction".into(),
            format!("{}", args.val_fraction),
        ];

        let mut env = blut_env(&ctx.job_dir, Self::NAME);
        env.push(("PYTHONPATH".into(), blut_pythonpath(&python_dir)));
        let inv = LamquantInvocation {
            python,
            script,
            cwd: python_dir.clone(),
            args: cmd_args,
            env,
            expected_outputs: vec![args.out.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(
                inv,
                Some(progress_forwarder(Self::NAME, ctx.status_tx.clone())),
            )
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        // Parse the manifest to populate train/val subject counts.
        // Schema: { "subjects": { sid: "train"|"val", ... }, ... }.
        let body = std::fs::read_to_string(&args.out).map_err(|source| StageError::Io {
            path: args.out.clone(),
            source,
        })?;
        let parsed: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            StageError::Backend(anyhow::anyhow!(
                "parse split_manifest JSON at {}: {e}",
                args.out.display()
            ))
        })?;
        let (n_train, n_val) = count_split_subjects(&parsed);

        let content_hash = ContentHash::hash_file(&args.out).map_err(|source| StageError::Io {
            path: args.out.clone(),
            source,
        })?;

        Ok(SplitManifest {
            path: args.out.clone(),
            content_hash,
            n_train_subjects: n_train,
            n_val_subjects: n_val,
        })
    }
}

/// Count subjects assigned to each split from the manifest's
/// `subjects` map (`{ sid: "train"|"val", ... }`). Returns
/// `(n_train, n_val)`; unknown labels are ignored. Missing map → (0,0).
fn count_split_subjects(manifest: &serde_json::Value) -> (i64, i64) {
    let mut n_train = 0i64;
    let mut n_val = 0i64;
    if let Some(map) = manifest.get("subjects").and_then(|v| v.as_object()) {
        for v in map.values() {
            match v.as_str() {
                Some("train") => n_train += 1,
                Some("val") => n_val += 1,
                _ => {}
            }
        }
    }
    (n_train, n_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::Artifact;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn corpus(p: &std::path::Path) -> LmaCorpus {
        LmaCorpus {
            root: p.to_path_buf(),
            n_archives: 1,
            content_hash: ContentHash::of_bytes(b"c"),
        }
    }

    #[tokio::test]
    async fn rejects_empty_out() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantBuildSplitManifest
            .run(
                &ctx(td.path()),
                corpus(td.path()),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    lma_root: String::new(),
                    labels_dir: String::new(),
                    out: PathBuf::new(),
                    val_fraction: 0.10,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_out_of_range_val_fraction() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantBuildSplitManifest
            .run(
                &ctx(td.path()),
                corpus(td.path()),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    lma_root: String::new(),
                    labels_dir: String::new(),
                    out: td.path().join("split.json"),
                    val_fraction: 1.5,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    // env-serialization guard held across the stage .await on purpose:
    // $BLUT_PYTHON is process-global; the lock serializes the whole
    // set/run/restore body against other env-mutating tests (same
    // pattern as recipe_paths_contract::env_guard).
    #[allow(clippy::await_holding_lock)]
    async fn rejects_missing_script() {
        // MOVE-B: the split-manifest builder resolves under
        // `blut_python_root` ($BLUT_PYTHON). Point it at a dir that
        // holds `python/` but not the script → clean BadInput.
        let _g = crate::TEST_ENV_LOCK.lock().unwrap();
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var("BLUT_PYTHON").ok();
        std::fs::create_dir_all(td.path().join("python")).unwrap();
        unsafe {
            std::env::set_var("BLUT_PYTHON", td.path());
        }
        let r = LamquantBuildSplitManifest
            .run(
                &ctx(td.path()),
                corpus(td.path()),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    lma_root: String::new(),
                    labels_dir: String::new(),
                    out: td.path().join("split.json"),
                    val_fraction: 0.10,
                },
            )
            .await;
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BLUT_PYTHON", v),
                None => std::env::remove_var("BLUT_PYTHON"),
            }
        }
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn stage_metadata_present() {
        assert_eq!(
            LamquantBuildSplitManifest::NAME,
            "lamquant_build_split_manifest"
        );
        assert_eq!(
            <<LamquantBuildSplitManifest as Stage>::Output as Artifact>::KIND,
            "lamquant.split_manifest"
        );
    }

    #[test]
    fn val_fraction_defaults_to_tenth() {
        let a: Args = serde_json::from_str(r#"{"out":"/tmp/split.json"}"#).unwrap();
        assert!((a.val_fraction - 0.10).abs() < 1e-6);
    }

    #[test]
    fn count_split_subjects_tallies_train_and_val() {
        let m = serde_json::json!({
            "subjects": {
                "aaaaaaaa": "train",
                "bbbbbbbb": "train",
                "cccccccc": "val",
                "dddddddd": "unknown",
            }
        });
        assert_eq!(count_split_subjects(&m), (2, 1));
    }

    #[test]
    fn count_split_subjects_zero_when_missing() {
        let m = serde_json::json!({ "meta": {} });
        assert_eq!(count_split_subjects(&m), (0, 0));
    }
}
