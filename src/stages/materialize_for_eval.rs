//! Materializer — `materialize_for_eval`.
//!
//! Graph-input stage for the `eval_suite` recipe: resolves a
//! registered model + a dataset (either registered or a path) and
//! outputs the tuple `(HfCheckpoint, DatasetJsonl)` that all three
//! eval stages consume. Having a single materialization point lets
//! the recipe `fork3` cleanly from one upstream into three parallel
//! branches.
//!
//! This commit's body is a thin shim — it looks up paths via the
//! existing registries / filesystem and synthesizes the artifact
//! handles. Hash computation is real (sha256 / merkle), so cache
//! keys downstream are correct.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, HfCheckpoint};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct MaterializeForEval;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path to the HuggingFace checkpoint directory.
    pub model_path: PathBuf,
    /// Recorded base model id (for provenance).
    #[serde(default)]
    pub base_model: String,
    /// Path to the eval JSONL.
    pub dataset_path: PathBuf,
}

#[async_trait]
impl Stage for MaterializeForEval {
    const NAME: &'static str = "materialize_for_eval";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = ();
    type Output = (HfCheckpoint, DatasetJsonl);
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &Args,
    ) -> Result<Self::Output, StageError> {
        // R30: reject path traversal — the kernel reads + may
        // re-emit these paths into downstream sidecar metadata.
        for (label, p) in [
            ("model_path", &args.model_path),
            ("dataset_path", &args.dataset_path),
        ] {
            if p.components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err(StageError::BadInput(format!(
                    "{label} '{}' contains '..' — refusing",
                    p.display()
                )));
            }
        }
        if !args.model_path.exists() {
            return Err(StageError::BadInput(format!(
                "model_path not found: {}",
                args.model_path.display()
            )));
        }
        if !args.dataset_path.exists() {
            return Err(StageError::BadInput(format!(
                "dataset_path not found: {}",
                args.dataset_path.display()
            )));
        }
        let ckpt_hash = if args.model_path.is_dir() {
            ContentHash::hash_dir(&args.model_path)
        } else {
            ContentHash::hash_file(&args.model_path)
        }
        .map_err(|source| StageError::Io {
            path: args.model_path.clone(),
            source,
        })?;
        let ckpt = HfCheckpoint {
            path: args.model_path.clone(),
            base_model: args.base_model.clone(),
            method_tag: "loaded".into(),
            content_hash: ckpt_hash,
            final_loss: 0.0,
        };

        let ds_hash =
            ContentHash::hash_file(&args.dataset_path).map_err(|source| StageError::Io {
                path: args.dataset_path.clone(),
                source,
            })?;
        // Cheap line count for the artifact; could be cached but
        // eval datasets are small (typically <50 MB).
        let n_examples = count_lines(&args.dataset_path)?;
        let ds = DatasetJsonl {
            path: args.dataset_path.clone(),
            content_hash: ds_hash,
            n_examples,
        };

        Ok((ckpt, ds))
    }
}

fn count_lines(path: &std::path::Path) -> Result<i64, StageError> {
    use std::io::BufRead;
    let f = std::fs::File::open(path).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut n = 0i64;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.map_err(|source| StageError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !line.trim().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trips_ckpt_and_dataset() {
        let td = tempfile::tempdir().unwrap();
        let model_dir = td.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        let ds_path = td.path().join("eval.jsonl");
        std::fs::write(&ds_path, "{\"a\":1}\n{\"b\":2}\n").unwrap();
        let ctx = StageContext::for_test(td.path().into(), td.path().join("stage"));
        let out = MaterializeForEval
            .run(
                &ctx,
                (),
                &Args {
                    model_path: model_dir,
                    base_model: "test/x".into(),
                    dataset_path: ds_path,
                },
            )
            .await
            .unwrap();
        assert_eq!(out.0.base_model, "test/x");
        assert_eq!(out.1.n_examples, 2);
    }

    #[tokio::test]
    async fn rejects_missing_model() {
        let td = tempfile::tempdir().unwrap();
        let ds_path = td.path().join("eval.jsonl");
        std::fs::write(&ds_path, "{}\n").unwrap();
        let ctx = StageContext::for_test(td.path().into(), td.path().join("stage"));
        let r = MaterializeForEval
            .run(
                &ctx,
                (),
                &Args {
                    model_path: td.path().join("nonexistent"),
                    base_model: "".into(),
                    dataset_path: ds_path,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
