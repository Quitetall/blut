//! Stage — `lamquant_train_teacher`.
//! (Manifest, FullbandMemmap) → TeacherCkpt. Wraps
//! `ai_models/oracle/train_teacher.py`. Output ckpt lands at the
//! script-conventional path `ai_models/oracle/teacher_best.ckpt`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{FullbandMemmap, TeacherCkpt};
use crate::backends::lamquant::runner::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, blut_python_script, blut_pythonpath, progress_forwarder, push_opt_u32, python_for,
    resolve_home,
};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct LamquantTrainTeacher;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default = "default_true")]
    pub headless: bool,
    #[serde(default)]
    pub force_batch_size: Option<u32>,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default)]
    pub resume: bool,
    /// Optional `--logger wandb|mlflow`. Empty = skip.
    #[serde(default)]
    pub logger: String,
    #[serde(default)]
    pub freq_weighted_loss: bool,
}

fn default_true() -> bool {
    true
}
fn default_seed() -> u32 {
    42
}

#[async_trait]
impl Stage for LamquantTrainTeacher {
    const NAME: &'static str = "lamquant_train_teacher";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    // Typed input is FullbandMemmap only — Manifest is reached
    // upstream via path convention in lamquant_home. The Manifest's
    // identity flows through FullbandMemmap's logical hash since
    // precompute_fullband consumes Manifest as its input.
    type Input = FullbandMemmap;
    type Output = TeacherCkpt;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: FullbandMemmap,
        args: &Args,
    ) -> Result<TeacherCkpt, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        // MOVE-B: script now under blut/python; resolve via $BLUT_PYTHON.
        let (script, python_dir) =
            blut_python_script(&["python", "lamquant", "oracle", "train_teacher.py"])?;

        let output_path = home
            .join("ai_models")
            .join("oracle")
            .join("teacher_best.ckpt");
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let mut cmd_args: Vec<String> = vec!["--seed".into(), args.seed.to_string()];
        if args.headless {
            cmd_args.push("--headless".into());
        }
        if args.resume {
            cmd_args.push("--resume".into());
        }
        if args.freq_weighted_loss {
            cmd_args.push("--freq-weighted-loss".into());
        }
        push_opt_u32(&mut cmd_args, "--force_batch_size", args.force_batch_size);
        if !args.logger.is_empty() {
            cmd_args.push("--logger".into());
            cmd_args.push(args.logger.clone());
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
            stat_fingerprint(b"lamquant.ckpt.teacher", &output_path).map_err(|source| {
                StageError::Io {
                    path: output_path.clone(),
                    source,
                }
            })?;
        Ok(TeacherCkpt {
            path: output_path,
            content_hash,
            gen_tag: "gen6".into(),
            final_loss: 0.0,
        })
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
        let fb = FullbandMemmap {
            train_path: td.path().join("t.dat"),
            val_path: td.path().join("v.dat"),
            n_windows: 0,
            content_hash: ContentHash::of_bytes(b""),
        };
        let r = LamquantTrainTeacher
            .run(
                &ctx(td.path()),
                fb,
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    headless: true,
                    force_batch_size: None,
                    seed: 42,
                    resume: false,
                    logger: String::new(),
                    freq_weighted_loss: false,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
