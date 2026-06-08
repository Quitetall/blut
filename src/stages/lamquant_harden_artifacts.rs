//! Stage — `lamquant_harden_artifacts`. (JointCkpt, TeacherCkpt) → HardenedCkpt.
//!
//! Wraps `ai_models/student/harden_artifacts.py`. Realigns the student
//! encoder's latents against the strided teacher so the decoder can
//! consume them at fullband scale (Route B deployment). Output is a
//! single hardened encoder ckpt with the latent alignment baked in.
//!
//! Nondeterministic — distillation step uses minibatch sampling.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{HardenedCkpt, JointCkpt, TeacherCkpt};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, blut_python_script, blut_pythonpath, progress_forwarder, push_opt_f32, push_opt_u32,
    python_for, resolve_home, safe_join,
};

pub struct LamquantHardenArtifacts;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// Output ckpt path relative to lamquant_home. Empty =
    /// builder default `ai_models/student/student_hardened.ckpt`.
    #[serde(default)]
    pub output_rel: String,
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    /// `--route-b` flag — produces a Route-B-compatible hardened ckpt
    /// (the typical case post-LMA pivot).
    #[serde(default = "default_true")]
    pub route_b: bool,
}

fn default_true() -> bool {
    true
}

#[async_trait]
impl Stage for LamquantHardenArtifacts {
    const NAME: &'static str = "lamquant_harden_artifacts";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = (JointCkpt, TeacherCkpt);
    type Output = HardenedCkpt;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Args,
    ) -> Result<HardenedCkpt, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        // MOVE-B: script now under blut/python; resolve via $BLUT_PYTHON.
        let (script, python_dir) =
            blut_python_script(&["python", "lamquant", "student", "harden_artifacts.py"])?;

        let (joint, teacher) = input;
        let output_path = if args.output_rel.is_empty() {
            home.join("ai_models")
                .join("student")
                .join("student_hardened.ckpt")
        } else {
            safe_join(&home, &args.output_rel)?
        };
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let mut cmd_args = vec![
            "--student-checkpoint".into(),
            joint.encoder_path.display().to_string(),
            "--teacher-checkpoint".into(),
            teacher.path.display().to_string(),
            "--output".into(),
            output_path.display().to_string(),
        ];
        push_opt_u32(&mut cmd_args, "--epochs", args.epochs);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--lr", args.lr);
        if args.route_b {
            cmd_args.push("--route-b".into());
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
            stat_fingerprint(b"lamquant.ckpt.hardened", &output_path).map_err(|source| {
                StageError::Io {
                    path: output_path.clone(),
                    source,
                }
            })?;
        Ok(HardenedCkpt {
            path: output_path,
            content_hash,
            route_b: args.route_b,
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

    #[tokio::test]
    async fn rejects_missing_home() {
        let td = tempfile::tempdir().unwrap();
        let joint = JointCkpt {
            encoder_path: PathBuf::from("/tmp/enc"),
            decoder_path: PathBuf::from("/tmp/dec"),
            content_hash: ContentHash::of_bytes(b""),
            final_loss: 0.0,
            tier: 0,
            preset: "production".into(),
        };
        let teacher = TeacherCkpt {
            path: PathBuf::from("/tmp/teacher"),
            content_hash: ContentHash::of_bytes(b""),
            gen_tag: "gen7".into(),
            final_loss: 0.0,
        };
        let r = LamquantHardenArtifacts
            .run(
                &ctx(td.path()),
                (joint, teacher),
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    output_rel: String::new(),
                    epochs: None,
                    batch_size: None,
                    lr: None,
                    route_b: true,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
