//! Model checkpoint artifacts.
//!
//! `HfCheckpoint` is a HuggingFace-format directory (post-trainer.py
//! output). `GgufModel` is the converted + quantized GGUF file
//! (post-convert_gguf, registry-ready).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfCheckpoint {
    /// Directory path (HF format: config.json, weights, tokenizer).
    pub path: PathBuf,
    /// Recorded base model id (`Qwen/Qwen3-7B`, etc.). Provenance.
    pub base_model: String,
    /// LoRA / Full / etc., as a free-form tag for downstream.
    pub method_tag: String,
    /// SHA-256 over the directory's contents (merkle of files).
    /// Computed at production time and stored here so consumers
    /// don't re-walk the dir on every cache lookup.
    pub content_hash: ContentHash,
    /// Final training loss reported by the trainer. Diagnostic;
    /// not part of content_hash.
    pub final_loss: f32,
}

impl Artifact for HfCheckpoint {
    const KIND: &'static str = "checkpoint.hf";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GgufModel {
    pub path: PathBuf,
    /// `Q4_K_M`, `Q5_K_M`, `Q8_0`, `f16`, etc.
    pub quant: String,
    pub content_hash: ContentHash,
    /// Registry entry name once `register_model` has run; `None`
    /// when the GGUF is in-flight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registered_as: Option<String>,
}

impl Artifact for GgufModel {
    const KIND: &'static str = "model.gguf";
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
    fn hf_checkpoint_round_trips() {
        let h = HfCheckpoint {
            path: PathBuf::from("/tmp/ckpt"),
            base_model: "Qwen/Qwen3-7B".into(),
            method_tag: "qlora".into(),
            content_hash: ContentHash::of_bytes(b"ckpt"),
            final_loss: 0.42,
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: HfCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back.base_model, "Qwen/Qwen3-7B");
        assert_eq!(back.method_tag, "qlora");
    }

    #[test]
    fn gguf_model_skips_registered_as_when_none() {
        let g = GgufModel {
            path: PathBuf::from("/tmp/m.gguf"),
            quant: "Q4_K_M".into(),
            content_hash: ContentHash::of_bytes(b"m"),
            registered_as: None,
        };
        let json = serde_json::to_string(&g).unwrap();
        assert!(!json.contains("registered_as"));
    }

    #[test]
    fn artifact_kinds_are_stable() {
        assert_eq!(HfCheckpoint::KIND, "checkpoint.hf");
        assert_eq!(GgufModel::KIND, "model.gguf");
    }
}
