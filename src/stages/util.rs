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

#[cfg(test)]
mod tests {
    use super::write_report;

    #[test]
    fn write_report_creates_parent_and_writes_pretty_json() {
        let dir = std::env::temp_dir().join(format!("blut_util_test_{}", std::process::id()));
        let path = dir.join("nested").join("report.json");
        let _ = std::fs::remove_dir_all(&dir);
        let metrics = serde_json::json!({"r": 0.5, "prd": 94.4});
        write_report(&path, &metrics).expect("write_report should succeed");
        assert!(path.exists(), "report not written");
        let back: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back, metrics, "round-trip mismatch");
        // pretty-printed → multi-line
        assert!(std::fs::read_to_string(&path).unwrap().contains('\n'));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
