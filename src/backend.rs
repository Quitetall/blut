//! `TrainBackend` — the trait every concrete trainer implements.
//!
//! v1 ships only `PythonTrainBackend` (a subprocess runner that talks
//! to `trainer.py` via the protocol in `protocol.rs`). The trait
//! exists to keep the rest of `lamu-train` agnostic — a future
//! "rented GPU" backend or a "local Megakernel-style" backend plugs
//! in by implementing `run` + `cancel`.

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::protocol::StatusUpdate;
use crate::spec::TrainSpec;

/// What a successful training run leaves on disk. The HF checkpoint is
/// the canonical artifact; the GGUF path is populated only after the
/// (optional) convert step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainArtifact {
    /// HuggingFace-format checkpoint directory.
    pub checkpoint_dir: PathBuf,
    /// Quantized GGUF, populated post-convert. None if `skip_convert`.
    pub gguf_path: Option<PathBuf>,
    /// Final training loss as reported by the trainer.
    pub final_loss: f32,
    /// Wall-clock duration of the run.
    pub elapsed: Duration,
}

/// Callback that receives every `StatusUpdate` from the trainer.
///
/// Boxed because the runner spawns a reader task that owns the
/// callback; `'static` because we don't constrain the runner's
/// lifetime to a borrowed handler.
pub type StatusFn = Box<dyn Fn(StatusUpdate) + Send + Sync + 'static>;

#[async_trait]
pub trait TrainBackend: Send + Sync {
    /// Run a single training job to completion. Streams `StatusUpdate`s
    /// to `on_status` as they arrive. Returns the artifact on success
    /// or a `TrainError` on any failure (including a terminal `Failed`
    /// status from the trainer).
    async fn run(&mut self, spec: TrainSpec, on_status: StatusFn) -> Result<TrainArtifact>;

    /// Best-effort cancellation. Implementations should send SIGTERM
    /// then SIGKILL after a grace period — never leave a zombie
    /// holding the GPU. Returns when the process is reaped.
    async fn cancel(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_round_trips() {
        let a = TrainArtifact {
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            gguf_path: Some(PathBuf::from("/tmp/x.gguf")),
            final_loss: 0.42,
            elapsed: Duration::from_secs(120),
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: TrainArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }
}
