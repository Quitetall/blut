//! Stage — `lamquant_pretrain_mae`. LmaCorpus → MaeCkpt.
//! Wraps `ai_models/student/pretrain_mae.py`. Nondeterministic.
//!
//! Per ADR 0017 (BLUT-canonical + LMA-direct), Input is the LMA
//! corpus; split path flows via Args.split_manifest. Pre-ADR
//! `L3Cache` input is gone.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{LmaCorpus, MaeCkpt};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, progress_forwarder, push_opt_f32, push_opt_u32, python_for, resolve_home, safe_join,
    script_path,
};

pub struct LamquantPretrainMae;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// Output ckpt path. Empty = `ai_models/student/pretrained_mae.ckpt`.
    #[serde(default)]
    pub output_rel: String,
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub mask_ratio: Option<f32>,
    #[serde(default)]
    pub patch_size: Option<u32>,
    #[serde(default)]
    pub windows_per_epoch: Option<u32>,
    #[serde(default)]
    pub max_windows: Option<u32>,
    #[serde(default)]
    pub seed: Option<u32>,
    /// LMA-direct training root (BLUT canonical, ADR 0017). Empty falls
    /// back to the deprecated NPZ + L3 precompute path.
    #[serde(default)]
    pub lma_root: String,
    /// JSON split manifest path. Required when ``lma_root`` is set.
    #[serde(default)]
    pub split_manifest: String,
}

#[async_trait]
impl Stage for LamquantPretrainMae {
    const NAME: &'static str = "lamquant_pretrain_mae";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = LmaCorpus;
    type Output = MaeCkpt;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: LmaCorpus,
        args: &Args,
    ) -> Result<MaeCkpt, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let script = script_path(&home, &["ai_models", "student", "pretrain_mae.py"])?;
        let output_path = if args.output_rel.is_empty() {
            home.join("ai_models").join("student").join("pretrained_mae.ckpt")
        } else {
            safe_join(&home, &args.output_rel)?
        };
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let mut cmd_args = vec!["--output".into(), output_path.display().to_string()];
        push_opt_u32(&mut cmd_args, "--epochs", args.epochs);
        push_opt_f32(&mut cmd_args, "--lr", args.lr);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--mask-ratio", args.mask_ratio);
        push_opt_u32(&mut cmd_args, "--patch-size", args.patch_size);
        push_opt_u32(&mut cmd_args, "--windows-per-epoch", args.windows_per_epoch);
        push_opt_u32(&mut cmd_args, "--max-windows", args.max_windows);
        push_opt_u32(&mut cmd_args, "--seed", args.seed);
        if !args.lma_root.is_empty() {
            cmd_args.push("--lma-root".into());
            cmd_args.push(args.lma_root.clone());
        }
        if !args.split_manifest.is_empty() {
            cmd_args.push("--split-manifest".into());
            cmd_args.push(args.split_manifest.clone());
        }

        let inv = LamquantInvocation {
            python,
            script,
            cwd: home,
            args: cmd_args,
            env: blut_env(&ctx.job_dir, Self::NAME),
            expected_outputs: vec![output_path.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(inv, Some(progress_forwarder(Self::NAME, ctx.status_tx.clone())))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let content_hash = stat_fingerprint(b"lamquant.ckpt.mae", &output_path).map_err(
            |source| StageError::Io {
                path: output_path.clone(),
                source,
            },
        )?;
        Ok(MaeCkpt {
            path: output_path,
            content_hash,
            base_arch: "TernaryMobileNetV5_Subband".into(),
            final_loss: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn rejects_missing_home() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantPretrainMae
            .run(
                &ctx(td.path()),
                LmaCorpus {
                    root: td.path().to_path_buf(),
                    n_archives: 0,
                    content_hash: ContentHash::of_bytes(b""),
                },
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    output_rel: String::new(),
                    epochs: None,
                    lr: None,
                    batch_size: None,
                    mask_ratio: None,
                    patch_size: None,
                    windows_per_epoch: None,
                    max_windows: None,
                    seed: None,
                    lma_root: String::new(),
                    split_manifest: String::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn deterministic_false() {
        assert!(!<LamquantPretrainMae as Stage>::DETERMINISTIC);
    }
}
