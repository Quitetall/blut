//! Stage 11 — `convert_gguf`. HF checkpoint → GGUF (optionally quantized).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{GgufModel, HfCheckpoint};
use crate::convert;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct ConvertGguf;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// `Q4_K_M`, `Q5_K_M`, `Q8_0`, `f16`. f16 skips quantize.
    pub quant: String,
    /// Output filename stem (the gguf path becomes
    /// `<stem>.<quant>.gguf` next to the HF checkpoint dir).
    pub name: String,
}

#[async_trait]
impl Stage for ConvertGguf {
    const NAME: &'static str = "convert_gguf";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu, Resource::Disk];
    type Input = HfCheckpoint;
    type Output = GgufModel;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: HfCheckpoint,
        args: &Args,
    ) -> Result<GgufModel, StageError> {
        let gguf_path = convert::convert_to_gguf(&input.path, &args.name, &args.quant)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let hash = ContentHash::hash_file(&gguf_path).map_err(|source| StageError::Io {
            path: gguf_path.clone(),
            source,
        })?;

        Ok(GgufModel {
            path: gguf_path,
            quant: args.quant.clone(),
            content_hash: hash,
            registered_as: None,
        })
    }
}
