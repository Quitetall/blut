//! Stage — `lamquant_train_l3_teacher`.
//! LmaCorpus → TeacherCkpt. Wraps `ai_models/oracle/train_l3_teacher.py`.
//!
//! Per ADR 0017 (BLUT-canonical + LMA-direct), Input is the LMA
//! corpus; Manifest path flows via Args.split_manifest forwarded to
//! the Python kernel. Pre-ADR `(Manifest, L3Cache)` tuple input is
//! gone.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{LmaCorpus, TeacherCkpt};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, blut_python_script, blut_pythonpath, progress_forwarder, push_opt_f32, push_opt_u32,
    python_for, resolve_home,
};

pub struct LamquantTrainL3Teacher;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub lr_min: Option<f32>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub windows_per_epoch: Option<u32>,
    #[serde(default)]
    pub max_windows: Option<u32>,
    /// `--device` (default "auto"). Empty = pass nothing.
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub resume: bool,
    /// LMA-direct training root (BLUT canonical, ADR 0017). Empty falls
    /// back to the deprecated NPZ + L3 precompute path.
    #[serde(default)]
    pub lma_root: String,
    /// JSON split manifest path. Required when ``lma_root`` is set.
    #[serde(default)]
    pub split_manifest: String,
}

#[async_trait]
impl Stage for LamquantTrainL3Teacher {
    const NAME: &'static str = "lamquant_train_l3_teacher";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = LmaCorpus;
    type Output = TeacherCkpt;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: Self::Input,
        args: &Args,
    ) -> Result<TeacherCkpt, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        // MOVE-B: script now under blut/python; resolve via $BLUT_PYTHON.
        let (script, python_dir) =
            blut_python_script(&["python", "lamquant", "oracle", "train_l3_teacher.py"])?;

        let output_path = home
            .join("ai_models")
            .join("oracle")
            .join("teacher_strided_best.ckpt");
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let mut cmd_args: Vec<String> = Vec::new();
        push_opt_u32(&mut cmd_args, "--epochs", args.epochs);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--lr", args.lr);
        push_opt_f32(&mut cmd_args, "--lr-min", args.lr_min);
        push_opt_u32(&mut cmd_args, "--width", args.width);
        push_opt_u32(&mut cmd_args, "--windows-per-epoch", args.windows_per_epoch);
        push_opt_u32(&mut cmd_args, "--max-windows", args.max_windows);
        if !args.device.is_empty() {
            cmd_args.push("--device".into());
            cmd_args.push(args.device.clone());
        }
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
            cwd: python_dir.clone(),
            args: cmd_args,
            env: {
                let mut e = blut_env(&ctx.job_dir, Self::NAME);
                e.push(("PYTHONPATH".into(), blut_pythonpath(&python_dir)));
                e
            },
            expected_outputs: vec![output_path.clone()],
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

        let content_hash =
            stat_fingerprint(b"lamquant.ckpt.teacher_l3", &output_path).map_err(|source| {
                StageError::Io {
                    path: output_path.clone(),
                    source,
                }
            })?;
        Ok(TeacherCkpt {
            path: output_path,
            content_hash,
            gen_tag: "l3".into(),
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
        let r = LamquantTrainL3Teacher
            .run(
                &ctx(td.path()),
                LmaCorpus {
                    root: td.path().to_path_buf(),
                    n_archives: 0,
                    content_hash: ContentHash::of_bytes(b""),
                },
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    epochs: None,
                    batch_size: None,
                    lr: None,
                    lr_min: None,
                    width: None,
                    windows_per_epoch: None,
                    max_windows: None,
                    device: String::new(),
                    resume: false,
                    lma_root: String::new(),
                    split_manifest: String::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
