//! Stage 5 — `split_train_eval`.
//!
//! Deterministic random split of a JSONL dataset into train/eval
//! files. Uses `rand::rngs::StdRng::seed_from_u64(args.seed)` so the
//! same input + seed always produce the same split — required for
//! cache correctness, reproducible experiments, and resumable plans.
//!
//! eval_ratio must be in (0, 1). Out-of-range fails fast with
//! `BadInput`. Empty input fails too — recipes shouldn't paper over
//! "we had zero examples" by producing an empty split.

use std::io::{BufRead, BufWriter, Write};

use async_trait::async_trait;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, DatasetSplit};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct SplitTrainEval;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Fraction of examples for the eval split. Must be in (0, 1).
    pub eval_ratio: f32,
    pub seed: u64,
}

#[async_trait]
impl Stage for SplitTrainEval {
    const NAME: &'static str = "split_train_eval";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = DatasetJsonl;
    type Output = DatasetSplit;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetJsonl,
        args: &Args,
    ) -> Result<DatasetSplit, StageError> {
        if !(args.eval_ratio > 0.0 && args.eval_ratio < 1.0) {
            return Err(StageError::BadInput(format!(
                "eval_ratio must be in (0, 1); got {}",
                args.eval_ratio
            )));
        }

        let file = std::fs::File::open(&input.path).map_err(|source| StageError::Io {
            path: input.path.clone(),
            source,
        })?;
        let mut lines: Vec<String> = Vec::new();
        for line in std::io::BufReader::new(file).lines() {
            let line = line.map_err(|source| StageError::Io {
                path: input.path.clone(),
                source,
            })?;
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }

        // Reject early on inputs that can't split into both halves —
        // catches both the empty case and the 1-line case (which
        // would otherwise hit `clamp(1, 0)` and panic).
        if lines.len() < 2 {
            return Err(StageError::BadInput(format!(
                "split_train_eval: need ≥2 examples to produce both train + eval, got {}",
                lines.len()
            )));
        }

        let mut indices: Vec<usize> = (0..lines.len()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        indices.shuffle(&mut rng);

        let n_eval = ((lines.len() as f32) * args.eval_ratio).round() as usize;
        // lines.len() ≥ 2 here, so the range [1, len-1] is non-empty.
        let n_eval = n_eval.clamp(1, lines.len() - 1);
        let (eval_idx, train_idx) = indices.split_at(n_eval);

        let train_path = ctx.stage_dir.join("train.jsonl");
        let eval_path = ctx.stage_dir.join("eval.jsonl");
        write_subset(&lines, train_idx, &train_path)?;
        write_subset(&lines, eval_idx, &eval_path)?;

        let train_hash = ContentHash::hash_file(&train_path).map_err(|source| StageError::Io {
            path: train_path.clone(),
            source,
        })?;
        let eval_hash = ContentHash::hash_file(&eval_path).map_err(|source| StageError::Io {
            path: eval_path.clone(),
            source,
        })?;

        Ok(DatasetSplit {
            train: DatasetJsonl {
                path: train_path,
                content_hash: train_hash,
                n_examples: train_idx.len() as i64,
            },
            eval: DatasetJsonl {
                path: eval_path,
                content_hash: eval_hash,
                n_examples: eval_idx.len() as i64,
            },
        })
    }
}

fn write_subset(
    lines: &[String],
    indices: &[usize],
    out_path: &std::path::Path,
) -> Result<(), StageError> {
    let f = std::fs::File::create(out_path).map_err(|source| StageError::Io {
        path: out_path.to_path_buf(),
        source,
    })?;
    let mut w = BufWriter::new(f);
    for &i in indices {
        writeln!(w, "{}", lines[i]).map_err(|source| StageError::Io {
            path: out_path.to_path_buf(),
            source,
        })?;
    }
    w.flush().map_err(|source| StageError::Io {
        path: out_path.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_n(td: &std::path::Path, n: usize) -> DatasetJsonl {
        let p = td.join("src.jsonl");
        let body: String = (0..n)
            .map(|i| format!(r#"{{"id":{i}}}"#))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&p, body).unwrap();
        DatasetJsonl {
            path: p,
            content_hash: ContentHash::of_bytes(b""),
            n_examples: n as i64,
        }
    }

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn deterministic_under_same_seed() {
        let td1 = tempfile::tempdir().unwrap();
        let td2 = tempfile::tempdir().unwrap();
        let args = Args {
            eval_ratio: 0.2,
            seed: 42,
        };
        let s1 = SplitTrainEval
            .run(&ctx(td1.path()), write_n(td1.path(), 100), &args)
            .await
            .unwrap();
        let s2 = SplitTrainEval
            .run(&ctx(td2.path()), write_n(td2.path(), 100), &args)
            .await
            .unwrap();
        // Hash equality means byte-equality of the train/eval files.
        assert_eq!(s1.train.content_hash, s2.train.content_hash);
        assert_eq!(s1.eval.content_hash, s2.eval.content_hash);
    }

    #[tokio::test]
    async fn different_seed_produces_different_split() {
        let td1 = tempfile::tempdir().unwrap();
        let td2 = tempfile::tempdir().unwrap();
        let s1 = SplitTrainEval
            .run(
                &ctx(td1.path()),
                write_n(td1.path(), 100),
                &Args {
                    eval_ratio: 0.2,
                    seed: 1,
                },
            )
            .await
            .unwrap();
        let s2 = SplitTrainEval
            .run(
                &ctx(td2.path()),
                write_n(td2.path(), 100),
                &Args {
                    eval_ratio: 0.2,
                    seed: 2,
                },
            )
            .await
            .unwrap();
        assert_ne!(s1.train.content_hash, s2.train.content_hash);
    }

    #[tokio::test]
    async fn rejects_out_of_range_ratio() {
        let td = tempfile::tempdir().unwrap();
        let r = SplitTrainEval
            .run(
                &ctx(td.path()),
                write_n(td.path(), 10),
                &Args {
                    eval_ratio: 1.5,
                    seed: 0,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn split_counts_sum_to_input() {
        let td = tempfile::tempdir().unwrap();
        let split = SplitTrainEval
            .run(
                &ctx(td.path()),
                write_n(td.path(), 100),
                &Args {
                    eval_ratio: 0.2,
                    seed: 0,
                },
            )
            .await
            .unwrap();
        assert_eq!(split.train.n_examples + split.eval.n_examples, 100);
        assert_eq!(split.eval.n_examples, 20);
    }

    #[tokio::test]
    async fn fails_on_single_example_input() {
        // Regression: clamp(1, 0) would panic on 1-line input
        // before the v2-commit-4a review fix.
        let td = tempfile::tempdir().unwrap();
        let r = SplitTrainEval
            .run(
                &ctx(td.path()),
                write_n(td.path(), 1),
                &Args {
                    eval_ratio: 0.2,
                    seed: 0,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn fails_on_empty_input() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("stage")).unwrap();
        let p = td.path().join("empty.jsonl");
        std::fs::write(&p, "").unwrap();
        let r = SplitTrainEval
            .run(
                &ctx(td.path()),
                DatasetJsonl {
                    path: p,
                    content_hash: ContentHash::of_bytes(b""),
                    n_examples: 0,
                },
                &Args {
                    eval_ratio: 0.2,
                    seed: 0,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
