//! Wire protocol between the Rust trainer backend and the Python
//! trainer subprocess. One `StatusUpdate` per line on the trainer's
//! stdout, JSON, no envelope, no trailing fields.
//!
//! Why no envelope: a trainer that crashes mid-line should still
//! produce parseable lines up to the crash point. Each line stands
//! alone. The reader treats malformed lines as `BadStatus` and keeps
//! going (logging the offender) — a single bad print mustn't stall
//! the whole run.
//!
//! Why "kind"-tagged: matches the on-disk `status.jsonl` shape so the
//! same parser is used for live streams + post-hoc job inspection.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One progress event from the trainer subprocess. Streamed live;
/// also persisted to `status.jsonl` for post-hoc inspection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StatusUpdate {
    /// One training step completed. `step` and `total` are 1-indexed.
    Step {
        step: u32,
        total: u32,
        loss: f32,
        lr: f32,
        vram_mb: u32,
    },
    /// Eval pass results. `step` is the training step at which eval
    /// was triggered.
    Eval { step: u32, eval_loss: f32 },
    /// A checkpoint was written to `path`. Multiple `Saved` events
    /// per run are normal (epoch checkpoints, best-loss snapshots).
    Saved { path: PathBuf },
    /// Final success. `checkpoint_dir` is the canonical artifact the
    /// downstream convert step reads.
    Done {
        final_loss: f32,
        checkpoint_dir: PathBuf,
    },
    /// Final failure. The trainer exits with a non-zero code after
    /// emitting this. `error` is a human-readable explanation; do not
    /// parse it.
    Failed { error: String },
}

impl StatusUpdate {
    /// True iff this is a terminal event — no further updates will
    /// arrive after one of these.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Failed { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_round_trips() {
        let s = StatusUpdate::Step {
            step: 1,
            total: 100,
            loss: 1.234,
            lr: 0.0002,
            vram_mb: 8192,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"kind\":\"step\""));
        let back: StatusUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn done_round_trips() {
        let s = StatusUpdate::Done {
            final_loss: 0.42,
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"kind\":\"done\""));
        let back: StatusUpdate = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn is_terminal_marks_done_and_failed() {
        assert!(StatusUpdate::Done {
            final_loss: 0.0,
            checkpoint_dir: PathBuf::new()
        }
        .is_terminal());
        assert!(StatusUpdate::Failed {
            error: "x".into()
        }
        .is_terminal());
        assert!(!StatusUpdate::Step {
            step: 1,
            total: 1,
            loss: 0.0,
            lr: 0.0,
            vram_mb: 0
        }
        .is_terminal());
    }

    #[test]
    fn malformed_json_returns_error() {
        let bad = "{\"kind\": \"step\", \"step\": \"not_a_number\"}";
        let r: serde_json::Result<StatusUpdate> = serde_json::from_str(bad);
        assert!(r.is_err());
    }

    #[test]
    fn unknown_kind_returns_error() {
        let bad = "{\"kind\": \"telemetry\", \"foo\": 1}";
        let r: serde_json::Result<StatusUpdate> = serde_json::from_str(bad);
        assert!(r.is_err());
    }
}
