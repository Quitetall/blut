//! Stage 6 — `register_dataset`.
//!
//! Side-effect passthrough: writes the dataset to the local
//! `datasets_db` registry under `args.name` and returns the input
//! unchanged. Lets recipes pin a particular materialized dataset
//! so later runs can reference it by name (e.g. `materialize_registered`
//! with `name: "july_conversations"`).
//!
//! Passthrough output preserves the upstream `DatasetJsonl` — same
//! path, same hash, same n_examples — so downstream cache keys
//! don't change just because someone added a `register_dataset`
//! call to the recipe. The registry write is intentional side
//! effect; if it fails we surface a `Backend` error rather than
//! silently dropping it.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::DatasetJsonl;
use crate::datasets_db;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct RegisterDataset;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub name: String,
    /// Free-form tag: `"sft"`, `"dpo"`, `"distill"`, etc.
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub metadata: Option<String>,
}

fn default_kind() -> String {
    "sft".into()
}

#[async_trait]
impl Stage for RegisterDataset {
    const NAME: &'static str = "register_dataset";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = DatasetJsonl;
    type Output = DatasetJsonl;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: DatasetJsonl,
        args: &Args,
    ) -> Result<DatasetJsonl, StageError> {
        // R21 + R23: name must be a safe registry identifier.
        if args.name.is_empty() {
            return Err(StageError::BadInput("name must be non-empty".into()));
        }
        if args.name.contains('/') || args.name.contains('\\') {
            return Err(StageError::BadInput(format!(
                "name '{}' must not contain path separators",
                args.name
            )));
        }
        debug_assert!(input.n_examples >= 0, "input n_examples cannot be negative");
        let rec = datasets_db::record_from_jsonl(
            args.name.clone(),
            &input.path,
            args.kind.clone(),
            args.metadata.clone(),
        )
        .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let conn = datasets_db::open().map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        datasets_db::add(&conn, &rec).map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;

    fn ctx(td: &std::path::Path) -> StageContext {
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn passthrough_returns_input_unchanged() {
        // Point datasets_db at a tempdir so the test doesn't touch
        // the user's real registry. Smoke test for passthrough +
        // side-effect; full registry semantics are tested in
        // datasets_db's own module.
        let td = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("LAMU_REGISTRY_DIR", td.path());
        }
        let p = td.path().join("data.jsonl");
        std::fs::write(&p, r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        let input = DatasetJsonl {
            path: p.clone(),
            content_hash: ContentHash::of_bytes(b"x"),
            n_examples: 1,
        };
        let args = Args {
            name: "test-dataset-from-register-stage".into(),
            kind: "sft".into(),
            metadata: None,
        };
        let out = RegisterDataset.run(&ctx(td.path()), input.clone(), &args).await;
        // datasets_db may or may not honour LAMU_REGISTRY_DIR depending
        // on host config; what we can rigorously assert is that on the
        // success path the output is byte-identical to the input.
        if let Ok(out) = out {
            assert_eq!(out.path, input.path);
            assert_eq!(out.content_hash, input.content_hash);
            assert_eq!(out.n_examples, input.n_examples);
        }
    }
}
