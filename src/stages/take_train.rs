//! Adapter stage — `take_train`.
//!
//! Projects the train half of a `DatasetSplit` into a bare
//! `DatasetJsonl`. Exists because the sequential executor walks a
//! linear DAG and `sft_train` consumes `DatasetJsonl`, not
//! `DatasetSplit`. When the parallel executor lands (commit 6) and
//! gains real `fork`/`merge` support, recipes will fork the split
//! into separate train/eval edges and this adapter goes away.
//!
//! Pure projection — no work, no IO. Side-effect-free. The
//! returned artifact shares the train file's path + hash with the
//! upstream split, so the cache keys downstream are stable across
//! re-runs.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, DatasetSplit};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct TakeTrain;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {}

#[async_trait]
impl Stage for TakeTrain {
    const NAME: &'static str = "take_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = DatasetSplit;
    type Output = DatasetJsonl;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: DatasetSplit,
        _args: &Args,
    ) -> Result<DatasetJsonl, StageError> {
        // R21: structural pre/post on the projection.
        debug_assert!(input.train.n_examples > 0, "train half must be non-empty");
        debug_assert!(input.eval.n_examples > 0, "eval half must be non-empty");
        debug_assert!(
            input.train.n_examples >= input.eval.n_examples,
            "train should typically be larger than eval"
        );
        Ok(input.train)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use std::path::PathBuf;

    #[tokio::test]
    async fn projects_train_half() {
        let td = tempfile::tempdir().unwrap();
        let split = DatasetSplit {
            train: DatasetJsonl {
                path: PathBuf::from("/tmp/train.jsonl"),
                content_hash: ContentHash::of_bytes(b"train"),
                n_examples: 80,
            },
            eval: DatasetJsonl {
                path: PathBuf::from("/tmp/eval.jsonl"),
                content_hash: ContentHash::of_bytes(b"eval"),
                n_examples: 20,
            },
        };
        let out = TakeTrain
            .run(
                &StageContext::for_test(td.path().into(), td.path().join("stage")),
                split.clone(),
                &Args::default(),
            )
            .await
            .unwrap();
        assert_eq!(out.path, split.train.path);
        assert_eq!(out.content_hash, split.train.content_hash);
        assert_eq!(out.n_examples, 80);
    }
}
