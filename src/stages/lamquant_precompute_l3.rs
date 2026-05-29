//! Stage — `lamquant_precompute_l3`.
//!
//! Wraps `ai_models/student/precompute_l3_fast.py`. Updates Q31
//! NPZ files in-place with their L3-approximation arrays. Output
//! artifact is `L3Cache { dir: <q31_events_path>, ... }`.
//!
//! Deterministic — same input bytes produce the same L3.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint_dir;
use crate::artifacts::{FullbandMemmap, L3Cache};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{python_for, resolve_home, script_path};

pub struct LamquantPrecomputeL3;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// Override Q31 NPZ input dir. Empty = builder default
    /// `ai_models/dataset_sim/q31_events`.
    #[serde(default)]
    pub input_dir: String,
    #[serde(default = "default_workers")]
    pub workers: u32,
}

fn default_workers() -> u32 {
    8
}

#[async_trait]
impl Stage for LamquantPrecomputeL3 {
    const NAME: &'static str = "lamquant_precompute_l3";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu, Resource::Disk];
    // Typed input is FullbandMemmap so the recipe chain reads
    // manifest → fullband → l3 in linear topo order. The L3
    // builder doesn't need fullband; the typed edge exists only
    // for plan-wiring convenience. n_windows is inherited.
    type Input = FullbandMemmap;
    type Output = L3Cache;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: FullbandMemmap,
        args: &Args,
    ) -> Result<L3Cache, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let script = script_path(&home, &["ai_models", "student", "precompute_l3_fast.py"])?;

        let q31_dir = if args.input_dir.is_empty() {
            home.join("ai_models")
                .join("dataset_sim")
                .join("q31_events")
        } else {
            PathBuf::from(&args.input_dir)
        };
        if !q31_dir.exists() {
            return Err(StageError::BadInput(format!(
                "q31 input dir not found: {}",
                q31_dir.display()
            )));
        }

        let cmd_args = vec![
            "--input".into(),
            q31_dir.display().to_string(),
            "--workers".into(),
            args.workers.to_string(),
        ];
        let inv = LamquantInvocation {
            python,
            script,
            cwd: home,
            args: cmd_args,
            env: vec![],
            expected_outputs: vec![q31_dir.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(inv, None)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let content_hash =
            stat_fingerprint_dir(b"lamquant.l3_cache", &q31_dir).map_err(|source| {
                StageError::Io {
                    path: q31_dir.clone(),
                    source,
                }
            })?;
        Ok(L3Cache {
            dir: q31_dir,
            n_windows: input.n_windows,
            content_hash,
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
        let r = LamquantPrecomputeL3
            .run(
                &ctx(td.path()),
                FullbandMemmap {
                    train_path: td.path().join("t.dat"),
                    val_path: td.path().join("v.dat"),
                    n_windows: 0,
                    content_hash: ContentHash::of_bytes(b""),
                },
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    input_dir: String::new(),
                    workers: 4,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
