//! Stage — `lamquant_generate_snn_labels`. (LmaCorpus, Manifest) → SnnLabels.
//!
//! Wraps `ai_models/snn/generate_activity_labels.py`. Per dataset
//! variant (chbmit / tuh_seizure / tuh_artifact), produces the
//! `<stem>_labels.npz` files consumed by `lamquant_train_mamba_snn`.
//! `dataset_id` selects the corpus tree.
//!
//! Deterministic — labels are a pure function of the source
//! annotations + window stride config.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{LmaCorpus, Manifest, SnnLabels};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, progress_forwarder, python_for, resolve_home, safe_join, script_path,
};

pub struct LamquantGenerateSnnLabels;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// Dataset variant: chbmit | tuh_seizure | tuh_artifact | tuh_epilepsy.
    pub dataset_id: String,
    /// Input corpus directory (raw EDF source, relative to lamquant_home).
    /// Empty = builder default `ai_models/dataset_sim/datasets/<dataset_id>`.
    #[serde(default)]
    pub input_rel: String,
    /// Output labels directory relative to lamquant_home. Empty =
    /// builder default `ai_models/snn/labels`.
    #[serde(default)]
    pub output_rel: String,
    /// Manifest path relative to lamquant_home. Empty = builder default.
    #[serde(default)]
    pub manifest_rel: String,
    /// Restrict to training-split stems only. Required for some
    /// downstream eval setups.
    #[serde(default)]
    pub training_only: bool,
}

#[async_trait]
impl Stage for LamquantGenerateSnnLabels {
    const NAME: &'static str = "lamquant_generate_snn_labels";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu, Resource::Disk];
    const DETERMINISTIC: bool = true;
    type Input = (LmaCorpus, Manifest);
    type Output = SnnLabels;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: Self::Input,
        args: &Args,
    ) -> Result<SnnLabels, StageError> {
        if args.dataset_id.is_empty() {
            return Err(StageError::BadInput(
                "dataset_id is required (chbmit | tuh_seizure | tuh_artifact | tuh_epilepsy)"
                    .into(),
            ));
        }
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let script = script_path(&home, &["ai_models", "snn", "generate_activity_labels.py"])?;

        let input_dir = if args.input_rel.is_empty() {
            home.join("ai_models")
                .join("dataset_sim")
                .join("datasets")
                .join(&args.dataset_id)
        } else {
            safe_join(&home, &args.input_rel)?
        };
        let output_dir = if args.output_rel.is_empty() {
            home.join("ai_models").join("snn").join("labels")
        } else {
            safe_join(&home, &args.output_rel)?
        };
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;
        let manifest_path = if args.manifest_rel.is_empty() {
            home.join("ai_models")
                .join("dataset_sim")
                .join("manifest_v3.json")
        } else {
            safe_join(&home, &args.manifest_rel)?
        };

        let mut cmd_args = vec![
            "--input".into(),
            input_dir.display().to_string(),
            "--output".into(),
            output_dir.display().to_string(),
            "--manifest".into(),
            manifest_path.display().to_string(),
        ];
        if args.training_only {
            cmd_args.push("--training-only".into());
        }

        let inv = LamquantInvocation {
            python,
            script,
            cwd: home,
            args: cmd_args,
            env: blut_env(&ctx.job_dir, Self::NAME),
            expected_outputs: vec![output_dir.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(inv, Some(progress_forwarder(Self::NAME, ctx.status_tx.clone())))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let mut n_stems: i64 = 0;
        if let Ok(rd) = std::fs::read_dir(&output_dir) {
            for e in rd.flatten() {
                if e.path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with("_labels.npz"))
                    .unwrap_or(false)
                {
                    n_stems += 1;
                }
            }
        }
        let content_hash = stat_fingerprint(b"lamquant.snn_labels", &output_dir).map_err(
            |source| StageError::Io {
                path: output_dir.clone(),
                source,
            },
        )?;
        Ok(SnnLabels {
            dir: output_dir,
            dataset_id: args.dataset_id.clone(),
            n_stems,
            content_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use std::path::PathBuf;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn corpus(p: &std::path::Path) -> LmaCorpus {
        LmaCorpus {
            root: p.to_path_buf(),
            n_archives: 0,
            content_hash: ContentHash::of_bytes(b""),
        }
    }

    fn manifest(p: &std::path::Path) -> Manifest {
        Manifest {
            path: p.to_path_buf(),
            n_windows: 0,
            content_hash: ContentHash::of_bytes(b""),
            val_fraction: 0.05,
            seed: 42,
        }
    }

    #[tokio::test]
    async fn rejects_empty_dataset_id() {
        let td = tempfile::tempdir().unwrap();
        let m = td.path().join("m.json");
        std::fs::write(&m, "{}").unwrap();
        let r = LamquantGenerateSnnLabels
            .run(
                &ctx(td.path()),
                (corpus(td.path()), manifest(&m)),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    dataset_id: String::new(),
                    input_rel: String::new(),
                    output_rel: String::new(),
                    manifest_rel: String::new(),
                    training_only: false,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_missing_home() {
        let td = tempfile::tempdir().unwrap();
        let m = td.path().join("m.json");
        std::fs::write(&m, "{}").unwrap();
        let r = LamquantGenerateSnnLabels
            .run(
                &ctx(td.path()),
                (corpus(td.path()), manifest(&m)),
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    dataset_id: "chbmit".into(),
                    input_rel: String::new(),
                    output_rel: String::new(),
                    manifest_rel: String::new(),
                    training_only: false,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn content_hash_kind() {
        use crate::framework::artifact::Artifact;
        assert_eq!(SnnLabels::KIND, "lamquant.snn_labels");
        let _ = PathBuf::from("/tmp");
    }
}
