//! Stage — `lamquant_train_student`. (Manifest, LmaCorpus) → JointCkpt.
//!
//! Wraps `ai_models/student/train_student_subband.py` (subband-trained
//! ternary student encoder). Different from `lamquant_train_joint`:
//! student-only training, no decoder co-training. Output is a
//! single encoder ckpt; package it as `JointCkpt` with the decoder
//! path left as a placeholder so downstream consumers (harden,
//! export) can pick it up uniformly.
//!
//! LMA-direct from day one — caller passes `lma_root` + `split_manifest`
//! so the Python kernel reads via `lamquant_codec.training.LmaL3Dataset`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{JointCkpt, LmaCorpus, Manifest};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, progress_forwarder, push_opt_f32, push_opt_u32, python_for, resolve_home,
    script_path,
};

pub struct LamquantTrainStudent;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// `--config` preset name (fast / standard / production).
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    /// `--resume` flag (no value = auto-detect from recovery dir).
    #[serde(default)]
    pub resume: bool,
    /// LMA-direct training root (BLUT canonical, ADR 0017).
    #[serde(default)]
    pub lma_root: String,
    /// JSON split manifest. Required when ``lma_root`` is set.
    #[serde(default)]
    pub split_manifest: String,
}

fn default_preset() -> String {
    "production".into()
}

#[async_trait]
impl Stage for LamquantTrainStudent {
    const NAME: &'static str = "lamquant_train_student";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = (Manifest, LmaCorpus);
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
        let script = script_path(&home, &["ai_models", "student", "train_student_subband.py"])?;

        let weights_dir = home.join("weights");
        std::fs::create_dir_all(&weights_dir).map_err(|source| StageError::Io {
            path: weights_dir.clone(),
            source,
        })?;
        let encoder_path = weights_dir.join("student_encoder_subband.ckpt");

        let mut cmd_args = vec!["--config".into(), args.preset.clone()];
        push_opt_u32(&mut cmd_args, "--epochs", args.epochs);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--lr", args.lr);
        if args.resume {
            cmd_args.push("--resume".into());
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
            expected_outputs: vec![encoder_path.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(inv, Some(progress_forwarder(Self::NAME, ctx.status_tx.clone())))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let content_hash = stat_fingerprint(b"lamquant.ckpt.student", &encoder_path).map_err(
            |source| StageError::Io {
                path: encoder_path.clone(),
                source,
            },
        )?;
        // Pack as a JointCkpt with the decoder slot left empty —
        // downstream (harden / export) treats this as the encoder ckpt.
        Ok(JointCkpt {
            encoder_path,
            decoder_path: weights_dir.join("(no_decoder).ckpt"),
            content_hash,
            final_loss: 0.0,
            tier: 0,
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

    fn manifest(p: &std::path::Path) -> Manifest {
        Manifest {
            path: p.to_path_buf(),
            n_windows: 0,
            content_hash: ContentHash::of_bytes(b""),
            val_fraction: 0.05,
            seed: 42,
        }
    }

    fn corpus(p: &std::path::Path) -> LmaCorpus {
        LmaCorpus {
            root: p.to_path_buf(),
            n_archives: 0,
            content_hash: ContentHash::of_bytes(b""),
        }
    }

    #[tokio::test]
    async fn rejects_missing_home() {
        let td = tempfile::tempdir().unwrap();
        let m_path = td.path().join("m.json");
        std::fs::write(&m_path, "{}").unwrap();
        let r = LamquantTrainStudent
            .run(
                &ctx(td.path()),
                (manifest(&m_path), corpus(td.path())),
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    preset: "production".into(),
                    epochs: None,
                    batch_size: None,
                    lr: None,
                    resume: false,
                    lma_root: String::new(),
                    split_manifest: String::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
