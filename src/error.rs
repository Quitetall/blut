use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("invalid TrainSpec: {0}")]
    InvalidSpec(String),

    #[error("dataset source not resolvable: {0}")]
    DatasetUnresolvable(String),

    #[error("trainer subprocess failed: {0}")]
    Trainer(String),

    #[error("trainer subprocess produced malformed status line: {0}")]
    BadStatus(String),

    #[error("conversion to GGUF failed: {0}")]
    Convert(String),

    #[error("registry update failed: {0}")]
    Registry(String),

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("operation cancelled")]
    Cancelled,

    #[error("{0}")]
    Other(String),
}

impl TrainError {
    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }

    pub fn invalid_spec(msg: impl Into<String>) -> Self {
        Self::InvalidSpec(msg.into())
    }
}

/// Convenience: lift bare `std::io::Error` (from `?` on stdlib
/// I/O ops) into `TrainError::Io` with an empty path. Callers
/// that have the path in scope should use the struct literal
/// directly for better messages, but this conversion keeps the
/// vendored scheduler_lock module compilable without rewriting
/// every `?` to be path-aware.
impl From<std::io::Error> for TrainError {
    fn from(source: std::io::Error) -> Self {
        TrainError::Io {
            path: PathBuf::new(),
            source,
        }
    }
}

impl From<serde_json::Error> for TrainError {
    fn from(e: serde_json::Error) -> Self {
        TrainError::Other(format!("json: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, TrainError>;
