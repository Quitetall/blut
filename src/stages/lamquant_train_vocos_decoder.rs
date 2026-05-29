//! Stage — `lamquant_train_vocos_decoder`.
//! LmaCorpus → JointCkpt (decoder-only — encoder_path points at
//! the upstream student ckpt referenced by --student-checkpoint,
//! decoder_path is the newly trained one). Wraps
//! `ai_models/decoder/train_vocos_decoder.py`.
//!
//! Per ADR 0017 (LMA-direct), the input is the LMA corpus; the
//! kernel resolves split via Args.split_manifest. Precompute
//! artifacts are no longer required inputs.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{JointCkpt, LmaCorpus};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, progress_forwarder, push_opt_f32, push_opt_u32, python_for, resolve_home, safe_join,
    script_path,
};

pub struct LamquantTrainVocosDecoder;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default = "default_tier")]
    pub tier: u32,
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub lr_min: Option<f32>,
    #[serde(default)]
    pub windows_per_epoch: Option<u32>,
    #[serde(default)]
    pub max_windows: Option<u32>,
    /// Path relative to lamquant_home of the student encoder ckpt
    /// to condition the decoder on. Required for the production
    /// flow; empty triggers the script's own default.
    #[serde(default)]
    pub student_checkpoint_rel: String,
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub adversarial: bool,
    #[serde(default)]
    pub adv_start_epoch: Option<u32>,
    #[serde(default)]
    pub adv_ramp_epochs: Option<u32>,
    #[serde(default)]
    pub disc_lr: Option<f32>,
    #[serde(default)]
    pub cfm_postfilter: bool,
    #[serde(default)]
    pub cfm_start_epoch: Option<u32>,
    #[serde(default)]
    pub perceptual_loss: bool,
    #[serde(default)]
    pub perceptual_weight: Option<f32>,
    #[serde(default)]
    pub dac_init: bool,
    /// LMA-direct training root (BLUT canonical, ADR 0017). Empty falls
    /// back to the deprecated NPZ + L3 precompute path.
    #[serde(default)]
    pub lma_root: String,
    /// JSON split manifest path. Required when ``lma_root`` is set.
    #[serde(default)]
    pub split_manifest: String,
}

fn default_tier() -> u32 {
    3
}

#[async_trait]
impl Stage for LamquantTrainVocosDecoder {
    const NAME: &'static str = "lamquant_train_vocos_decoder";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = LmaCorpus;
    type Output = JointCkpt;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: Self::Input,
        args: &Args,
    ) -> Result<JointCkpt, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let script = script_path(&home, &["ai_models", "decoder", "train_vocos_decoder.py"])?;

        let decoder_path = home
            .join("ai_models")
            .join("student")
            .join(format!("decoder_tier{}_stable_fast.ckpt", args.tier));
        if let Some(parent) = decoder_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let encoder_ref = if args.student_checkpoint_rel.is_empty() {
            PathBuf::from("(none)")
        } else {
            safe_join(&home, &args.student_checkpoint_rel)?
        };

        let mut cmd_args = vec!["--tier".into(), args.tier.to_string()];
        push_opt_u32(&mut cmd_args, "--epochs", args.epochs);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--lr", args.lr);
        push_opt_f32(&mut cmd_args, "--lr-min", args.lr_min);
        push_opt_u32(&mut cmd_args, "--windows-per-epoch", args.windows_per_epoch);
        push_opt_u32(&mut cmd_args, "--max-windows", args.max_windows);
        if !args.student_checkpoint_rel.is_empty() {
            cmd_args.push("--student-checkpoint".into());
            cmd_args.push(encoder_ref.display().to_string());
        }
        if !args.device.is_empty() {
            cmd_args.push("--device".into());
            cmd_args.push(args.device.clone());
        }
        if args.resume {
            cmd_args.push("--resume".into());
        }
        if args.adversarial {
            cmd_args.push("--adversarial".into());
        }
        push_opt_u32(&mut cmd_args, "--adv-start-epoch", args.adv_start_epoch);
        push_opt_u32(&mut cmd_args, "--adv-ramp-epochs", args.adv_ramp_epochs);
        push_opt_f32(&mut cmd_args, "--disc-lr", args.disc_lr);
        if args.cfm_postfilter {
            cmd_args.push("--cfm-postfilter".into());
        }
        push_opt_u32(&mut cmd_args, "--cfm-start-epoch", args.cfm_start_epoch);
        if args.perceptual_loss {
            cmd_args.push("--perceptual-loss".into());
        }
        push_opt_f32(&mut cmd_args, "--perceptual-weight", args.perceptual_weight);
        if args.dac_init {
            cmd_args.push("--dac-init".into());
        }
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
            expected_outputs: vec![decoder_path.clone()],
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

        let dec_fp =
            stat_fingerprint(b"lamquant.ckpt.decoder", &decoder_path).map_err(|source| {
                StageError::Io {
                    path: decoder_path.clone(),
                    source,
                }
            })?;
        Ok(JointCkpt {
            encoder_path: encoder_ref,
            decoder_path,
            content_hash: dec_fp,
            final_loss: 0.0,
            tier: args.tier,
            preset: "vocos_decoder_only".into(),
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
        let corpus = LmaCorpus {
            root: td.path().join("lma"),
            n_archives: 0,
            content_hash: ContentHash::of_bytes(b""),
        };
        let args = Args {
            lamquant_home: td.path().join("nope").display().to_string(),
            tier: 3,
            ..Default::default()
        };
        let r = LamquantTrainVocosDecoder
            .run(&ctx(td.path()), corpus, &args)
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
