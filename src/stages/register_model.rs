//! Stage 12 — `register_model`.
//!
//! Side-effect passthrough: writes a registry entry via
//! `crate::registry::add_entry` and emits the same `GgufModel`
//! it received, with `registered_as` populated. Hash-stable so
//! the output is cacheable (a re-run finds the entry already
//! present and skips).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::GgufModel;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct RegisterModel;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub name: String,
    /// Free-form notes preserved with the registry entry for audit.
    #[serde(default)]
    pub notes: String,
    /// Architecture tag for the registry. Defaults to "trained".
    #[serde(default = "default_arch")]
    pub arch: String,
}

fn default_arch() -> String {
    "trained".into()
}

#[async_trait]
impl Stage for RegisterModel {
    const NAME: &'static str = "register_model";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = GgufModel;
    type Output = GgufModel;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: GgufModel,
        args: &Args,
    ) -> Result<GgufModel, StageError> {
        use crate::registry;
        use crate::registry::{
            BackendType, Capability, ModelEntry, ModelFormat, ModelStatus,
        };

        let registry_path = crate::config::registry_path();
        let entry = ModelEntry {
            name: args.name.clone(),
            path: input.path.clone(),
            format: ModelFormat::Gguf,
            backend: BackendType::LlamaCpp,
            arch: args.arch.clone(),
            params_b: 0.0,
            quant: input.quant.clone(),
            vram_mb: 0,
            context_max: 0,
            capabilities: vec![Capability::Chat],
            notes: args.notes.clone(),
            status: ModelStatus::default(),
        };
        registry::add_entry(entry, &registry_path, true)
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        Ok(GgufModel {
            registered_as: Some(args.name.clone()),
            ..input
        })
    }
}
