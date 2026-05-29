//! Stage — `lamquant_precompute_fullband`.
//!
//! Wraps `ai_models/dataset_sim/precompute_fullband_memmap.py`.
//! Manifest → FullbandMemmap (train + val `.dat` pair + meta.json).
//! Deterministic — same manifest hash + same script → same memmap
//! bytes.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{FullbandMemmap, Manifest};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{python_for, resolve_home, safe_join, script_path};

pub struct LamquantPrecomputeFullband;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// Output dir for `fullband_{split}.dat` + `.meta.json`.
    /// Empty = builder default `ai_models/dataset_sim/`.
    #[serde(default)]
    pub out_dir_rel: String,
    /// Splits to materialize; default ["train", "val"].
    #[serde(default = "default_splits")]
    pub splits: Vec<String>,
}

fn default_splits() -> Vec<String> {
    vec!["train".into(), "val".into()]
}

#[async_trait]
impl Stage for LamquantPrecomputeFullband {
    const NAME: &'static str = "lamquant_precompute_fullband";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = Manifest;
    type Output = FullbandMemmap;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: Manifest,
        args: &Args,
    ) -> Result<FullbandMemmap, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        let script = script_path(
            &home,
            &["ai_models", "dataset_sim", "precompute_fullband_memmap.py"],
        )?;

        let out_dir = if args.out_dir_rel.is_empty() {
            home.join("ai_models").join("dataset_sim")
        } else {
            safe_join(&home, &args.out_dir_rel)?
        };
        std::fs::create_dir_all(&out_dir).map_err(|source| StageError::Io {
            path: out_dir.clone(),
            source,
        })?;

        let mut cmd_args = vec![
            "--manifest".into(),
            input.path.display().to_string(),
            "--out".into(),
            out_dir.display().to_string(),
        ];
        if !args.splits.is_empty() {
            cmd_args.push("--splits".into());
            for s in &args.splits {
                cmd_args.push(s.clone());
            }
        }

        let train_path = out_dir.join("fullband_train.dat");
        let val_path = out_dir.join("fullband_val.dat");

        let inv = LamquantInvocation {
            python,
            script,
            cwd: home,
            args: cmd_args,
            env: vec![],
            expected_outputs: vec![train_path.clone(), val_path.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(inv, None)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        // Fingerprint over BOTH .dat files combined (domain-tagged).
        let train_fp =
            stat_fingerprint(b"lamquant.fullband.train", &train_path).map_err(|source| {
                StageError::Io {
                    path: train_path.clone(),
                    source,
                }
            })?;
        let val_fp = stat_fingerprint(b"lamquant.fullband.val", &val_path).map_err(|source| {
            StageError::Io {
                path: val_path.clone(),
                source,
            }
        })?;
        // Merkle of the two halves.
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"lamquant.fullband_memmap");
        h.update(train_fp.0);
        h.update(val_fp.0);
        let arr: [u8; 32] = h.finalize().into();
        let content_hash = crate::framework::artifact::ContentHash(arr);

        Ok(FullbandMemmap {
            train_path,
            val_path,
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

    fn manifest(p: &std::path::Path) -> Manifest {
        Manifest {
            path: p.to_path_buf(),
            content_hash: ContentHash::of_bytes(b"m"),
            n_windows: 100,
            val_fraction: 0.05,
            seed: 42,
        }
    }

    #[tokio::test]
    async fn rejects_missing_home() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantPrecomputeFullband
            .run(
                &ctx(td.path()),
                manifest(&td.path().join("m.json")),
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    out_dir_rel: String::new(),
                    splits: default_splits(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_missing_script() {
        let td = tempfile::tempdir().unwrap();
        // Empty repo — no precompute script in it.
        let r = LamquantPrecomputeFullband
            .run(
                &ctx(td.path()),
                manifest(&td.path().join("m.json")),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    out_dir_rel: String::new(),
                    splits: default_splits(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
