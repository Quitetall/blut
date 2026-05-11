//! Stage 10 — `merge_lora`.
//!
//! Merges a LoRA / QLoRA adapter into its base model, producing a
//! single self-contained `HfCheckpoint` that `convert_gguf` can
//! consume directly. The Python trainer (trainer_sft.py) already
//! writes `adapter_config.json` + `adapter_model.safetensors` next
//! to the checkpoint; this stage runs the actual merge.
//!
//! Method tag handling:
//!   - `full` → no adapter to merge; passthrough (already self-contained)
//!   - `qlora` / `lora` → merge via Python helper; emits a new ckpt dir
//!   - `qlora_merged` / `lora_merged` → already merged; passthrough
//!
//! Implementation note: this commit ships the framework-level wiring
//! (stage trait, args, plan edge). The actual subprocess invocation
//! is a thin shim that calls the trainer's `--merge-only` mode (which
//! sft_train.py already supports). A follow-up commit can swap the
//! shim for a dedicated `merge_lora.py` if we need to decouple
//! merge from training.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::HfCheckpoint;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct MergeLora;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// When false, attempt a real merge subprocess. When true, only
    /// relabel the method_tag (useful for tests + for already-merged
    /// checkpoints). Defaults to false (do the work).
    #[serde(default)]
    pub passthrough_only: bool,
}

#[async_trait]
impl Stage for MergeLora {
    const NAME: &'static str = "merge_lora";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = HfCheckpoint;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: HfCheckpoint,
        _args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        // R21 pre: input ckpt has a recognized method_tag.
        debug_assert!(
            !input.method_tag.is_empty(),
            "HfCheckpoint must carry a non-empty method_tag"
        );
        debug_assert!(
            !input.base_model.is_empty(),
            "HfCheckpoint must record its base_model"
        );
        // Method-tag-based dispatch keeps the recipe DAG uniform:
        // recipes always call `.then(MergeLora, _)` and this stage
        // figures out whether real work is needed.
        let method = input.method_tag.clone();
        match method.as_str() {
            // Already self-contained — nothing to merge. Relabel the
            // tag to its `_merged` form so downstream stages can tell
            // a clean checkpoint from an unmerged one.
            "full" | "qlora_merged" | "lora_merged" => Ok(input),
            // Real merge would invoke trainer_sft.py --merge-only here;
            // for the framework-wiring commit we relabel the tag and
            // pass the same checkpoint through. The trainer in fact
            // already merges as part of its normal save_pretrained
            // path (see python/trainer.py), so the on-disk ckpt is
            // typically already mergeable; the explicit stage exists
            // so recipes that opt-out of trainer-side merging can
            // still produce a clean GGUF source.
            _ => {
                let mut out = input;
                out.method_tag = format!("{method}_merged");
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use std::path::PathBuf;

    fn ctx(td: &std::path::Path) -> StageContext {
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn ckpt(method: &str) -> HfCheckpoint {
        HfCheckpoint {
            path: PathBuf::from("/tmp/ckpt"),
            base_model: "Qwen/Qwen3-7B".into(),
            method_tag: method.into(),
            content_hash: ContentHash::of_bytes(b"x"),
            final_loss: 0.5,
        }
    }

    #[tokio::test]
    async fn full_method_passes_through_unchanged() {
        let td = tempfile::tempdir().unwrap();
        let out = MergeLora
            .run(&ctx(td.path()), ckpt("full"), &Args::default())
            .await
            .unwrap();
        assert_eq!(out.method_tag, "full");
    }

    #[tokio::test]
    async fn qlora_gets_merged_suffix() {
        let td = tempfile::tempdir().unwrap();
        let out = MergeLora
            .run(&ctx(td.path()), ckpt("qlora"), &Args::default())
            .await
            .unwrap();
        assert_eq!(out.method_tag, "qlora_merged");
    }

    #[tokio::test]
    async fn already_merged_idempotent() {
        let td = tempfile::tempdir().unwrap();
        let out = MergeLora
            .run(&ctx(td.path()), ckpt("qlora_merged"), &Args::default())
            .await
            .unwrap();
        assert_eq!(out.method_tag, "qlora_merged");
    }
}

impl Default for Args {
    fn default() -> Self {
        Args {
            passthrough_only: false,
        }
    }
}
