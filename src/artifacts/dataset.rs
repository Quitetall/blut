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

/// Domain separator for the `DatasetSplit` merkle. Shared by `content_hash`
/// (over the members' cached hashes) and `recompute_content_hash` (over the
/// members re-walked from disk) so the two can never silently drift — a drift
/// would make P2P verification reject valid splits or accept corrupt ones.
const DATASET_SPLIT_DOMAIN: &[u8] = b"dataset.split";

impl Artifact for DatasetSplit {
    const KIND: &'static str = "dataset.split";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(DATASET_SPLIT_DOMAIN);
        h.update(self.train.content_hash.0);
        h.update(self.eval.content_hash.0);
        let arr: [u8; 32] = h.finalize().into();
        ContentHash(arr)
    }
    fn primary_path(&self) -> &Path {
        &self.train.path
    }
    /// Composite: the address is a merkle over BOTH members, and
    /// `primary_path()` is only `train.path`. The default `recompute` would
    /// hash just the train file and never reproduce the merkle — so re-walk
    /// both files from disk and rebuild the same `b"dataset.split" ‖ train ‖
    /// eval` digest the producer used.
    fn recompute_content_hash(&self) -> std::io::Result<ContentHash> {
        use sha2::{Digest, Sha256};
        let t = ContentHash::hash_file(&self.train.path)?;
        let e = ContentHash::hash_file(&self.eval.path)?;
        let mut h = Sha256::new();
        h.update(DATASET_SPLIT_DOMAIN);
        h.update(t.0);
        h.update(e.0);
        Ok(ContentHash(h.finalize().into()))
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

    #[test]
    fn dataset_split_recompute_matches_merkle_from_disk() {
        // The composite override must re-walk BOTH member files and rebuild the
        // same merkle as content_hash() — the default (primary_path only) would
        // hash just the train file and never reproduce it.
        let td = tempfile::tempdir().unwrap();
        let tp = td.path().join("train.jsonl");
        let ep = td.path().join("eval.jsonl");
        std::fs::write(&tp, b"train rows").unwrap();
        std::fs::write(&ep, b"eval rows").unwrap();
        let split = DatasetSplit {
            train: DatasetJsonl {
                path: tp.clone(),
                content_hash: ContentHash::hash_file(&tp).unwrap(),
                n_examples: 2,
            },
            eval: DatasetJsonl {
                path: ep.clone(),
                content_hash: ContentHash::hash_file(&ep).unwrap(),
                n_examples: 1,
            },
        };
        // recompute (from disk) == the cached merkle.
        assert_eq!(split.recompute_content_hash().unwrap(), split.content_hash());
        // And it actually depends on the EVAL file (default would miss it):
        std::fs::write(&ep, b"eval rows CHANGED").unwrap();
        let drifted = DatasetSplit {
            eval: DatasetJsonl {
                path: ep.clone(),
                content_hash: ContentHash::hash_file(&ep).unwrap(),
                ..split.eval.clone()
            },
            ..split.clone()
        };
        assert_ne!(
            drifted.recompute_content_hash().unwrap(),
            split.content_hash(),
            "recompute must reflect the eval file, not just train"
        );
    }
}
