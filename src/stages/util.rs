//! Shared stage utilities.
//!
//! Functions used by multiple stages live here so cross-module
//! `super::` calls don't have to reach into a specific stage's
//! private surface.

/// Pretty-print a JSON value to the given path. Creates the parent
/// directory if missing. Used by every eval stage + `merge_reports`.
pub(crate) fn write_report(
    path: &std::path::Path,
    metrics: &serde_json::Value,
) -> Result<(), crate::framework::error::StageError> {
    use crate::framework::error::StageError;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let body = serde_json::to_vec_pretty(metrics)
        .map_err(|e| StageError::Backend(anyhow::anyhow!("serialize eval report: {e}")))?;
    std::fs::write(path, body).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })
}
