//! Stage — `lamquant_build_manifest`.
//!
//! Graph-input stage that runs LamQuant's
//! `ai_models/dataset_sim/build_manifest.py` to produce
//! `manifest_v3.json`. Shells out via `LamquantBackend`; reads
//! the resulting JSON to populate the typed `Manifest` artifact's
//! `n_windows` field.
//!
//! Deterministic — same `--seed` + same `--q31-dir` byte state
//! → identical manifest. Holdout patient selection is RNG-driven
//! but seeded, so `DETERMINISTIC = true` (default).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::Manifest;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation, resolve_lamquant_python};
use crate::stages::lamquant_helpers::{blut_python_script, blut_pythonpath, resolve_home};

pub struct LamquantBuildManifest;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty string → fall back to
    /// `$LAMQUANT_HOME` env, then `~/Desktop/LamQuant`.
    #[serde(default)]
    pub lamquant_home: String,
    /// Optional override for the q31_events dir. Empty = builder's
    /// own default (`ai_models/dataset_sim/q31_events`).
    #[serde(default)]
    pub q31_dir: String,
    /// Output manifest path, relative to `lamquant_home`. Empty =
    /// builder default `ai_models/dataset_sim/manifest_v3.json`.
    #[serde(default)]
    pub output_rel: String,
    /// Optional path to a prior v2 manifest.
    #[serde(default)]
    pub v2_path: String,
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_val_fraction() -> f32 {
    0.05
}
fn default_seed() -> u64 {
    42
}

#[async_trait]
impl Stage for LamquantBuildManifest {
    const NAME: &'static str = "lamquant_build_manifest";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = ();
    type Output = Manifest;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &Args,
    ) -> Result<Manifest, StageError> {
        // MOVE-B (2026-05-29): the build_manifest preprocessing script
        // now lives in the PUBLIC BLUT submodule at
        // `<blut>/python/lamquant/dataset/build_manifest.py`, resolved
        // via `blut_python_root` ($BLUT_PYTHON). `lamquant_home` is
        // still resolved for the (optional) output path that the
        // operator may anchor relative to the Neural repo.
        let lamquant_home = resolve_home(&args.lamquant_home)?;
        let python = resolve_lamquant_python(&lamquant_home);
        let (script, python_dir) = blut_python_script(&[
            "python",
            "lamquant",
            "dataset",
            "build_manifest.py",
        ])?;

        let output_path = if args.output_rel.is_empty() {
            // Builder default: the manifest_v3.json that moved alongside
            // the script into blut/python/lamquant/dataset/.
            python_dir
                .join("lamquant")
                .join("dataset")
                .join("manifest_v3.json")
        } else {
            lamquant_home.join(&args.output_rel)
        };

        let mut cmd_args: Vec<String> = vec![
            "--output".into(),
            output_path.display().to_string(),
            "--val-fraction".into(),
            format!("{}", args.val_fraction),
            "--seed".into(),
            format!("{}", args.seed),
        ];
        if !args.q31_dir.is_empty() {
            cmd_args.push("--q31-dir".into());
            cmd_args.push(args.q31_dir.clone());
        }
        if !args.v2_path.is_empty() {
            cmd_args.push("--v2".into());
            cmd_args.push(args.v2_path.clone());
        }

        let inv = LamquantInvocation {
            python,
            script,
            cwd: python_dir.clone(),
            args: cmd_args,
            env: vec![("PYTHONPATH".into(), blut_pythonpath(&python_dir))],
            expected_outputs: vec![output_path.clone()],
            run_manifest_path: None,
        };

        let mut backend = LamquantBackend::new();
        backend
            .run(inv, None)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        // Parse the JSON to populate the typed Manifest's n_windows
        // field. Manifest format is `{"n_windows": int, ...}` plus
        // various per-dataset breakdowns we don't need at this level.
        let body = std::fs::read_to_string(&output_path).map_err(|source| StageError::Io {
            path: output_path.clone(),
            source,
        })?;
        let parsed: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("parse manifest JSON: {e}")))?;
        let n_windows = parsed
            .get("n_windows")
            .and_then(|v| v.as_i64())
            .or_else(|| parsed.get("total_windows").and_then(|v| v.as_i64()))
            .ok_or_else(|| {
                StageError::Backend(anyhow::anyhow!(
                    "manifest JSON missing required `n_windows` / `total_windows` field at {}",
                    output_path.display()
                ))
            })?;

        let content_hash =
            ContentHash::hash_file(&output_path).map_err(|source| StageError::Io {
                path: output_path.clone(),
                source,
            })?;

        Ok(Manifest {
            path: output_path,
            content_hash,
            n_windows,
            val_fraction: args.val_fraction,
            seed: args.seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn rejects_missing_lamquant_home() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantBuildManifest
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    q31_dir: String::new(),
                    output_rel: String::new(),
                    v2_path: String::new(),
                    val_fraction: 0.05,
                    seed: 42,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    // env-serialization guard held across the stage .await on purpose:
    // $BLUT_PYTHON is process-global; the lock serializes the whole
    // set/run/restore body against other env-mutating tests (same
    // pattern as recipe_paths_contract::env_guard).
    #[allow(clippy::await_holding_lock)]
    async fn rejects_missing_build_script() {
        // MOVE-B: the build script resolves under `blut_python_root`
        // ($BLUT_PYTHON), not `lamquant_home`. Point $BLUT_PYTHON at a
        // dir with no `python/lamquant/dataset/build_manifest.py` so the
        // resolver returns a clean BadInput.
        let _g = crate::TEST_ENV_LOCK.lock().unwrap();
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var("BLUT_PYTHON").ok();
        // Lay a `python/` so the root validates but the script is absent.
        std::fs::create_dir_all(td.path().join("python")).unwrap();
        unsafe {
            std::env::set_var("BLUT_PYTHON", td.path());
        }
        let r = LamquantBuildManifest
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    q31_dir: String::new(),
                    output_rel: String::new(),
                    v2_path: String::new(),
                    val_fraction: 0.05,
                    seed: 42,
                },
            )
            .await;
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BLUT_PYTHON", v),
                None => std::env::remove_var("BLUT_PYTHON"),
            }
        }
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
