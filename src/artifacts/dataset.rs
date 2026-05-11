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
