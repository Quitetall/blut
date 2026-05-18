//! Stage — `lamquant_train_joint`.
//! LmaCorpus → JointCkpt. Wraps `ai_models/student/train_joint.py`.
//! Optional `encoder_init_rel` arg seeds the encoder from a prior
//! MAE pretrain — the `lamquant_encoder` recipe wires pretrain_mae
//! upstream and sets this. Nondeterministic.
//!
//! Per ADR 0017 (BLUT-canonical + LMA-direct), Input is the LMA
//! corpus directly; Manifest + split paths flow via Args.lma_root +
//! Args.split_manifest forwarded to the Python kernel. The
//! pre-ADR `(Manifest, FullbandMemmap, L3Cache)` tuple input is
//! gone — those artifacts are no longer required.

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

pub struct LamquantTrainJoint;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// `--config` preset (fast | medium | production). LamQuant
    /// uses TrainingConfig presets.
    #[serde(default = "default_preset")]
    pub preset: String,
    /// Decoder tier (1..4).
    #[serde(default = "default_tier")]
    pub tier: u32,
    #[serde(default = "default_seed")]
    pub seed: u32,
    /// MAE / SFT init path relative to lamquant_home. Empty = train
    /// from scratch.
    #[serde(default)]
    pub encoder_init_rel: String,
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub gan: Option<bool>,
    #[serde(default)]
    pub seizure_head: Option<bool>,
    #[serde(default)]
    pub infinite_lr: bool,
    /// Optional `--resume <path>` (or "auto"). Empty = no resume.
    #[serde(default)]
    pub resume: String,
    /// LMA-direct training root (BLUT canonical, ADR 0017). Empty falls
    /// back to the deprecated NPZ + L3 precompute path the Python kernel
    /// loads from manifest_v3.json.
    #[serde(default)]
    pub lma_root: String,
    /// JSON split manifest path. Required when ``lma_root`` is set.
    #[serde(default)]
    pub split_manifest: String,
}

fn default_preset() -> String {
    "production".into()
}
fn default_tier() -> u32 {
    3
}
fn default_seed() -> u32 {
    42
}

#[async_trait]
impl Stage for LamquantTrainJoint {
    const NAME: &'static str = "lamquant_train_joint";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    // Typed input is LmaCorpus only (ADR 0017). Manifest + split
    // paths flow via Args.split_manifest forwarded to the Python
    // kernel as `--split-manifest`; the LmaCorpus content hash
    // cascades cache invalidation through the corpus directory.
    type Input = LmaCorpus;
    type Output = JointCkpt;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: LmaCorpus,
        args: &Args,
    ) -> Result<JointCkpt, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let script = script_path(&home, &["ai_models", "student", "train_joint.py"])?;

        // train_joint writes a known pair of files to weights/.
        let weights_dir = home.join("weights");
        std::fs::create_dir_all(&weights_dir).map_err(|source| StageError::Io {
            path: weights_dir.clone(),
            source,
        })?;
        let encoder_path = weights_dir.join("student_encoder_joint.ckpt");
        let decoder_path = weights_dir.join(format!("decoder_tier{}_joint.ckpt", args.tier));

        let mut cmd_args = vec![
            "--config".into(),
            args.preset.clone(),
            "--tier".into(),
            args.tier.to_string(),
            "--seed".into(),
            args.seed.to_string(),
        ];
        if !args.encoder_init_rel.is_empty() {
            let init = safe_join(&home, &args.encoder_init_rel)?;
            cmd_args.push("--encoder-init".into());
            cmd_args.push(init.display().to_string());
        }
        push_opt_u32(&mut cmd_args, "--epochs", args.epochs);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--lr", args.lr);
        if let Some(gan) = args.gan {
            cmd_args.push(if gan { "--gan".into() } else { "--no-gan".into() });
        }
        if let Some(sh) = args.seizure_head {
            cmd_args.push(if sh {
                "--seizure-head".into()
            } else {
                "--no-seizure-head".into()
            });
        }
        if args.infinite_lr {
            cmd_args.push("--infinite-lr".into());
        }
        if !args.resume.is_empty() {
            cmd_args.push("--resume".into());
            cmd_args.push(args.resume.clone());
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
            expected_outputs: vec![encoder_path.clone(), decoder_path.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(inv, Some(progress_forwarder(Self::NAME, ctx.status_tx.clone())))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let enc_fp = stat_fingerprint(b"lamquant.ckpt.joint.encoder", &encoder_path).map_err(
            |source| StageError::Io {
                path: encoder_path.clone(),
                source,
            },
        )?;
        let dec_fp = stat_fingerprint(b"lamquant.ckpt.joint.decoder", &decoder_path).map_err(
            |source| StageError::Io {
                path: decoder_path.clone(),
                source,
            },
        )?;
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"lamquant.ckpt.joint");
        h.update(enc_fp.0);
        h.update(dec_fp.0);
        let arr: [u8; 32] = h.finalize().into();
        let content_hash = crate::framework::artifact::ContentHash(arr);

        Ok(JointCkpt {
            encoder_path,
            decoder_path,
            content_hash,
            final_loss: 0.0,
            tier: args.tier,
            preset: args.preset.clone(),
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
        let r = LamquantTrainJoint
            .run(
                &ctx(td.path()),
                LmaCorpus {
                    root: td.path().to_path_buf(),
                    n_archives: 0,
                    content_hash: ContentHash::of_bytes(b""),
                },
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    preset: "production".into(),
                    tier: 3,
                    seed: 42,
                    encoder_init_rel: String::new(),
                    epochs: None,
                    batch_size: None,
                    lr: None,
                    gan: None,
                    seizure_head: None,
                    infinite_lr: false,
                    resume: String::new(),
                    lma_root: String::new(),
                    split_manifest: String::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn deterministic_false() {
        assert!(!<LamquantTrainJoint as Stage>::DETERMINISTIC);
    }
}
