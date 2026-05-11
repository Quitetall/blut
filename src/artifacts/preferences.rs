//! Preference dataset for DPO training.
//!
//! JSONL where each line carries a `prompt`, a `chosen` response,
//! and a `rejected` response. The DPO trainer optimises the model
//! to assign higher likelihood to `chosen` than `rejected` for the
//! same prompt.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreferenceJsonl {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub n_pairs: i64,
}

impl Artifact for PreferenceJsonl {
    const KIND: &'static str = "dataset.preferences";
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
    fn preference_jsonl_round_trip() {
        let p = PreferenceJsonl {
            path: PathBuf::from("/tmp/p.jsonl"),
            content_hash: ContentHash::of_bytes(b"p"),
            n_pairs: 100,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: PreferenceJsonl = serde_json::from_str(&json).unwrap();
        assert_eq!(back.n_pairs, 100);
    }
}
