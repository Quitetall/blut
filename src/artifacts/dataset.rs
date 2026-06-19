// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Dataset artifacts.
//!
//! A `DatasetJsonl` is one JSONL file on disk plus its content
//! hash and example count. Stages that produce datasets
//! (`materialize_conversations`, `materialize_dataset_path`,
//! `filter_dataset`, etc.) emit this. Stages that consume
//! datasets (`sft_train`, `eval_loss`) take it as input.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetJsonl {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub n_examples: i64,
}

impl Artifact for DatasetJsonl {
    const KIND: &'static str = "dataset.jsonl";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

/// Train/eval split produced by `split_train_eval`. The
/// primary_path returns the train file; consumers needing the eval
/// file destructure the struct directly. Content hash is the merkle
/// of (train_hash ‖ eval_hash) so the cache distinguishes splits
/// over the same source with different seeds/ratios.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetSplit {
    pub train: DatasetJsonl,
    pub eval: DatasetJsonl,
}

impl Artifact for DatasetSplit {
    const KIND: &'static str = "dataset.split";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"dataset.split");
        h.update(self.train.content_hash.0);
        h.update(self.eval.content_hash.0);
        let arr: [u8; 32] = h.finalize().into();
        ContentHash(arr)
    }
    fn primary_path(&self) -> &Path {
        &self.train.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_jsonl_round_trips_via_serde() {
        let d = DatasetJsonl {
            path: PathBuf::from("/tmp/data.jsonl"),
            content_hash: ContentHash::of_bytes(b"x"),
            n_examples: 7,
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: DatasetJsonl = serde_json::from_str(&json).unwrap();
        assert_eq!(back.n_examples, 7);
        assert_eq!(back.path, d.path);
        assert_eq!(back.content_hash, d.content_hash);
    }

    #[test]
    fn artifact_kind_is_stable() {
        assert_eq!(DatasetJsonl::KIND, "dataset.jsonl");
        assert_eq!(DatasetJsonl::SCHEMA, 1);
    }
}
