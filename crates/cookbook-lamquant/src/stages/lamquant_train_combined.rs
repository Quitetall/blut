//! Stage — `lamquant_train_combined`.
//! LmaCorpus → (TeacherCkpt, JointCkpt). Wraps
//! `ai_models/decoder/train_combined.py` which trains teacher and
//! decoder simultaneously (~40% faster than separate runs).
//!
//! Per ADR 0017 (BLUT-canonical + LMA-direct dataset), this stage
//! reads `.lma` archives directly via Args.lma_root +
//! Args.split_manifest forwarded to the Python kernel. The
//! `Manifest` + `FullbandMemmap` precompute artifacts that the
//! v7.6.x recipe chain produced are no longer required inputs.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{JointCkpt, LmaCorpus, TeacherCkpt};
use crate::backends::lamquant::runner::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, blut_python_script, blut_pythonpath, progress_forwarder, push_opt_f32, push_opt_u32,
    python_for, resolve_home, safe_join,
};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct LamquantTrainCombined;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default = "default_decoder_tier")]
    pub decoder_tier: u32,
    #[serde(default)]
    pub teacher_epochs: Option<u32>,
    #[serde(default)]
    pub decoder_epochs: Option<u32>,
    #[serde(default)]
    pub teacher_width: Option<u32>,
    /// `--teacher-strides` like "1,2,2". Empty = script default.
    #[serde(default)]
    pub teacher_strides: String,
    #[serde(default)]
    pub channel_attn: bool,
    #[serde(default)]
    pub bottleneck_attn: bool,
    #[serde(default)]
    pub teacher_r_loss: Option<f32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub teacher_lr: Option<f32>,
    #[serde(default)]
    pub decoder_lr: Option<f32>,
    #[serde(default)]
    pub lr_min: Option<f32>,
    #[serde(default)]
    pub windows_per_epoch: Option<u32>,
    #[serde(default)]
    pub max_windows: Option<u32>,
    #[serde(default)]
    pub student_checkpoint_rel: String,
    /// LMA-direct training root (BLUT canonical, ADR 0017). Empty falls
    /// back to the deprecated NPZ + L3 precompute path.
    #[serde(default)]
    pub lma_root: String,
    /// JSON split manifest path. Required when ``lma_root`` is set.
    #[serde(default)]
    pub split_manifest: String,
}

fn default_decoder_tier() -> u32 {
    3
}

#[async_trait]
impl Stage for LamquantTrainCombined {
    const NAME: &'static str = "lamquant_train_combined";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = LmaCorpus;
    type Output = (TeacherCkpt, JointCkpt);
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: Self::Input,
        args: &Args,
    ) -> Result<Self::Output, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        // MOVE-B: script now under blut/python; resolve via $BLUT_PYTHON.
        let (script, python_dir) =
            blut_python_script(&["python", "lamquant", "decoder", "train_combined.py"])?;

        // Combined writes a teacher ckpt + a decoder ckpt to the
        // student/ dir per LamQuant convention.
        let teacher_path = home
            .join("ai_models")
            .join("oracle")
            .join("teacher_combined_best.ckpt");
        let decoder_path = home
            .join("ai_models")
            .join("student")
            .join(format!("decoder_tier{}_joint_fast.ckpt", args.decoder_tier));
        for p in [&teacher_path, &decoder_path] {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }

        let mut cmd_args = vec!["--decoder-tier".into(), args.decoder_tier.to_string()];
        push_opt_u32(&mut cmd_args, "--teacher-epochs", args.teacher_epochs);
        push_opt_u32(&mut cmd_args, "--decoder-epochs", args.decoder_epochs);
        push_opt_u32(&mut cmd_args, "--teacher-width", args.teacher_width);
        if !args.teacher_strides.is_empty() {
            cmd_args.push("--teacher-strides".into());
            cmd_args.push(args.teacher_strides.clone());
        }
        if args.channel_attn {
            cmd_args.push("--channel-attn".into());
        }
        if args.bottleneck_attn {
            cmd_args.push("--bottleneck-attn".into());
        }
        push_opt_f32(&mut cmd_args, "--teacher-r-loss", args.teacher_r_loss);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--teacher-lr", args.teacher_lr);
        push_opt_f32(&mut cmd_args, "--decoder-lr", args.decoder_lr);
        push_opt_f32(&mut cmd_args, "--lr-min", args.lr_min);
        push_opt_u32(&mut cmd_args, "--windows-per-epoch", args.windows_per_epoch);
        push_opt_u32(&mut cmd_args, "--max-windows", args.max_windows);
        if !args.student_checkpoint_rel.is_empty() {
            let p = safe_join(&home, &args.student_checkpoint_rel)?;
            cmd_args.push("--student-checkpoint".into());
            cmd_args.push(p.display().to_string());
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
            cwd: python_dir.clone(),
            args: cmd_args,
            env: {
                let mut e = blut_env(&ctx.job_dir, Self::NAME);
                e.push(("PYTHONPATH".into(), blut_pythonpath(&python_dir)));
                e
            },
            expected_outputs: vec![teacher_path.clone(), decoder_path.clone()],
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

        let teacher_fp =
            stat_fingerprint(b"lamquant.ckpt.teacher", &teacher_path).map_err(|source| {
                StageError::Io {
                    path: teacher_path.clone(),
                    source,
                }
            })?;
        let decoder_fp =
            stat_fingerprint(b"lamquant.ckpt.decoder", &decoder_path).map_err(|source| {
                StageError::Io {
                    path: decoder_path.clone(),
                    source,
                }
            })?;
        Ok((
            TeacherCkpt {
                path: teacher_path,
                content_hash: teacher_fp,
                gen_tag: "combined".into(),
                final_loss: 0.0,
            },
            JointCkpt {
                encoder_path: PathBuf::from("(combined-no-encoder)"),
                decoder_path,
                content_hash: decoder_fp,
                final_loss: 0.0,
                tier: args.decoder_tier,
                preset: "combined".into(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::artifact::ContentHash;

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
            ..Default::default()
        };
        let r = LamquantTrainCombined
            .run(&ctx(td.path()), corpus, &args)
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
