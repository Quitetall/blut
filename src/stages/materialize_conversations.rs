//! Stage 1 — `materialize_conversations`.
//!
//! Pulls turns from LAMU's read-only conversations.db and writes a
//! filtered JSONL training set to the stage_dir. Wraps
//! `blut::conversations::dump_to_jsonl` so the
//! filter / sha256 / count logic stays in one place.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::DatasetJsonl;
use crate::conversations;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct MaterializeConversations;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// How far back to pull, in seconds. Recipes convert humantime
    /// strings to seconds before constructing Args so the
    /// JsonSchema is a clean integer.
    pub since_seconds: u64,
}

#[async_trait]
impl Stage for MaterializeConversations {
    const NAME: &'static str = "materialize_conversations";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = ();
    type Output = DatasetJsonl;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &Args,
    ) -> Result<DatasetJsonl, StageError> {
        // R21 pre: ctx.stage_dir must be set + writable region.
        debug_assert!(
            !ctx.stage_dir.as_os_str().is_empty(),
            "stage_dir must be non-empty"
        );
        // R23: since_seconds == 0 means "everything since epoch" —
        // valid but worth a debug_assert to catch accidental 0-init.
        debug_assert!(
            args.since_seconds > 0,
            "since_seconds=0 sweeps all-time history; intentional?"
        );
        let out_path = ctx.stage_dir.join("dataset.jsonl");
        let stats = conversations::dump_to_jsonl(
            Duration::from_secs(args.since_seconds),
            &out_path,
        )
        .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        if stats.n_conversations == 0 {
            return Err(StageError::BadInput(format!(
                "no usable conversations in window (since {}s); \
                 {} short, {} filtered, {} error msgs, {} oversize",
                args.since_seconds,
                stats.n_dropped_short,
                stats.n_dropped_filtered_below_min,
                stats.n_dropped_errors,
                stats.n_dropped_oversize
            )));
        }

        let hash = ContentHash::hash_file(&out_path).map_err(|source| StageError::Io {
            path: out_path.clone(),
            source,
        })?;

        // R21 post: emitted artifact has positive example count
        // (we just rejected n_conversations==0 above; n_turns is
        // >= n_conversations).
        let n = stats.n_turns as i64;
        debug_assert!(n > 0, "n_turns must be > 0 after n_conversations check");
        Ok(DatasetJsonl {
            path: out_path,
            content_hash: hash,
            n_examples: n,
        })
    }
}
