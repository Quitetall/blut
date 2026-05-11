//! Stage / Plan / Recipe error types.
//!
//! `StageError` is what a stage's `run` returns. `PlanError` is
//! what the executor returns to the caller. `RecipeError` is what
//! `Recipe::compile` returns. Each carries enough context that a
//! reader of `status.jsonl` can tell *what* failed and *where*
//! without the trace.

use std::path::PathBuf;

use crate::framework::resource::Resource;

#[derive(Debug, thiserror::Error)]
pub enum StageError {
    /// Input artifact didn't satisfy the stage's preconditions
    /// (e.g. a JSONL file with zero examples).
    #[error("input did not satisfy precondition: {0}")]
    BadInput(String),

    /// Backend-specific failure (Python subprocess crash, network
    /// timeout, OOM). Wraps `anyhow::Error` so backends can attach
    /// rich chained-source context without specializing the
    /// signature for every backend.
    #[error("backend: {0}")]
    Backend(#[source] anyhow::Error),

    /// I/O failure with the offending path. Stage callers usually
    /// have the path already; carrying it in the error type spares
    /// log readers from grepping for it.
    #[error("io at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Stage was cancelled mid-run. Distinct variant so the
    /// executor can decide not to mark a cancelled plan as failed
    /// (the caller asked for it).
    #[error("cancelled")]
    Cancelled,

    /// Resource semaphore acquisition timed out. Indicates upstream
    /// overcommit (rare); separate variant for triage.
    #[error("resource '{0}' acquisition timed out")]
    ResourceTimeout(Resource),

    /// Erased dispatch: input artifact arrived with the wrong
    /// `KIND` tag for what this stage expects. Should be unreachable
    /// when stages are wired through the typed `Plan` builder; can
    /// fire if a recipe constructs the plan dynamically and gets
    /// the kinds wrong, or if a Unix-style `lamu-train stage <x>`
    /// is fed an artifact of the wrong kind on stdin.
    #[error("input kind mismatch: stage '{stage}' expected '{expected}', got '{got}'")]
    KindMismatch {
        stage: &'static str,
        expected: &'static str,
        got: String,
    },

    /// Erased dispatch: input JSON did not deserialize into the
    /// stage's expected `Input` type even though the kind tag
    /// matched. Schema drift; bump `Artifact::SCHEMA` and the cache
    /// will invalidate downstream.
    #[error("input deserialize failed for stage '{stage}': {source}")]
    InputDeserialize {
        stage: &'static str,
        #[source]
        source: serde_json::Error,
    },

    /// Erased dispatch: args JSON did not deserialize into the
    /// stage's expected `Args` type. Distinct from `InputDeserialize`
    /// so log readers + the CLI can tell "bad artifact" from "bad
    /// recipe args". The latter is usually a recipe bug; the former
    /// is usually a schema mismatch between producer + consumer.
    #[error("args deserialize failed for stage '{stage}': {source}")]
    ArgsDeserialize {
        stage: &'static str,
        #[source]
        source: serde_json::Error,
    },

    /// Erased dispatch: stage produced an output that didn't
    /// serialize. Should be impossible if the output type derives
    /// `Serialize` correctly; here for completeness.
    #[error("output serialize failed for stage '{stage}': {source}")]
    OutputSerialize {
        stage: &'static str,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("plan has no nodes")]
    Empty,
    #[error("plan has a cycle (node {0:?})")]
    Cycle(u32),
    #[error("plan stage '{stage}' (idx {idx}) failed: {source}")]
    StageFailed {
        idx: u32,
        stage: String,
        #[source]
        source: StageError,
    },
    #[error("plan cancelled")]
    Cancelled,
    #[error("plan io: {0}")]
    Io(#[from] std::io::Error),
    #[error("plan: {0}")]
    Other(String),
}

#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("recipe args invalid: {0}")]
    InvalidArgs(String),
    #[error("recipe compile failed: {0}")]
    CompileFailed(String),
    #[error("recipe '{name}' not found in catalog")]
    NotFound { name: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_error_kind_mismatch_renders_clearly() {
        let e = StageError::KindMismatch {
            stage: "convert_gguf",
            expected: "checkpoint.hf",
            got: "dataset.jsonl".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("convert_gguf"));
        assert!(msg.contains("checkpoint.hf"));
        assert!(msg.contains("dataset.jsonl"));
    }

    #[test]
    fn stage_error_io_carries_path() {
        let e = StageError::Io {
            path: PathBuf::from("/tmp/x"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        };
        let msg = format!("{e}");
        assert!(msg.contains("/tmp/x"));
        assert!(msg.contains("missing"));
    }

    #[test]
    fn plan_error_stage_failed_chains_source() {
        let inner = StageError::Cancelled;
        let outer = PlanError::StageFailed {
            idx: 3,
            stage: "sft_train".into(),
            source: inner,
        };
        // The Display chain shows the wrapper without the inner
        // (thiserror renders `#[source]` on demand, not in
        // `Display`). Assert the wrapper is informative.
        let msg = format!("{outer}");
        assert!(msg.contains("3"));
        assert!(msg.contains("sft_train"));
        assert!(msg.contains("cancelled"));
    }

    #[test]
    fn recipe_error_not_found_names_recipe() {
        let e = RecipeError::NotFound { name: "frobulate".into() };
        assert!(format!("{e}").contains("frobulate"));
    }
}
