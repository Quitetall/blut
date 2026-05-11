//! Shared helpers for LamQuant stages.
//!
//! Cuts boilerplate from the per-stage `run()` bodies: resolving
//! `lamquant_home`, checking script existence, building progress
//! fan-out, and the `safe_join` / `resolve_relative` path helpers.

use std::path::{Path, PathBuf};

use crate::framework::error::StageError;
use crate::framework::status::StageEvent;
use crate::lamquant_backend::{
    default_lamquant_home, resolve_lamquant_python, Progress,
};

/// Canonicalize and validate `lamquant_home`. Returns the absolute
/// path. `BadInput` if the path is missing or not canonicalizable.
pub fn resolve_home(raw: &str) -> Result<PathBuf, StageError> {
    let raw_path = if raw.is_empty() {
        default_lamquant_home()
    } else {
        PathBuf::from(raw)
    };
    std::fs::canonicalize(&raw_path).map_err(|e| {
        StageError::BadInput(format!(
            "lamquant_home not found or not canonicalizable: {} ({e})",
            raw_path.display()
        ))
    })
}

/// Resolve the LamQuant Python interpreter; wrapper for the
/// `lamquant_backend` helper so callers don't need to import both.
pub fn python_for(home: &Path) -> PathBuf {
    resolve_lamquant_python(home)
}

/// Build the absolute path to a LamQuant script and assert it
/// exists. `BadInput` on miss.
pub fn script_path(home: &Path, rel: &[&str]) -> Result<PathBuf, StageError> {
    let mut p = home.to_path_buf();
    for component in rel {
        p.push(component);
    }
    if !p.exists() {
        return Err(StageError::BadInput(format!(
            "{} not found",
            p.display()
        )));
    }
    Ok(p)
}

/// Reject `..` components + absolute paths in user-supplied
/// relative paths. Used for output paths the stage writes to.
pub fn safe_join(base: &Path, rel: &str) -> Result<PathBuf, StageError> {
    let p = Path::new(rel);
    if p.is_absolute() {
        return Err(StageError::BadInput(format!(
            "path '{rel}' must be relative to lamquant_home"
        )));
    }
    for component in p.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(StageError::BadInput(format!(
                "path '{rel}' contains '..' — refusing traversal"
            )));
        }
    }
    Ok(base.join(p))
}

/// Pass-through for absolute paths; join onto `base` for relative.
/// Used for read-side data dir args.
pub fn resolve_relative(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// Build the standard progress fan-out closure used by training
/// stages. Forwards parsed tqdm progress to the executor's status
/// broadcast as `StageEvent::StageStep`.
pub fn progress_forwarder(
    stage_name: &'static str,
    status_tx: tokio::sync::broadcast::Sender<StageEvent>,
) -> Box<dyn Fn(Progress) + Send + Sync> {
    Box::new(move |p| {
        let _ = status_tx.send(StageEvent::StageStep {
            node_idx: 0,
            stage_name: stage_name.to_string(),
            update: serde_json::json!({
                "kind": "tqdm",
                "current": p.current,
                "total": p.total,
            }),
        });
    })
}

/// Standard BLUT identity env vars for the RunManifest pre-hook
/// to read inside the subprocess.
pub fn blut_env(job_dir: &Path, stage_name: &str) -> Vec<(String, String)> {
    vec![
        ("BLUT_JOB_DIR".into(), job_dir.display().to_string()),
        ("BLUT_STAGE_NAME".into(), stage_name.to_string()),
    ]
}

/// Push `--flag <value>` if `v` is `Some`. Convenience helper for
/// stage `Args` with Option-shaped overrides.
pub fn push_opt_u32(out: &mut Vec<String>, flag: &str, v: Option<u32>) {
    if let Some(v) = v {
        out.push(flag.to_string());
        out.push(v.to_string());
    }
}

pub fn push_opt_f32(out: &mut Vec<String>, flag: &str, v: Option<f32>) {
    if let Some(v) = v {
        out.push(flag.to_string());
        out.push(v.to_string());
    }
}

pub fn push_opt_str(out: &mut Vec<String>, flag: &str, v: &str) {
    if !v.is_empty() {
        out.push(flag.to_string());
        out.push(v.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_join_rejects_traversal() {
        let r = safe_join(Path::new("/tmp/x"), "../../etc/passwd");
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn safe_join_rejects_absolute() {
        let r = safe_join(Path::new("/tmp/x"), "/etc/passwd");
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn safe_join_accepts_normal_relative() {
        let r = safe_join(Path::new("/tmp/x"), "a/b/c").unwrap();
        assert_eq!(r, PathBuf::from("/tmp/x/a/b/c"));
    }

    #[test]
    fn resolve_relative_passes_absolute() {
        let r = resolve_relative(Path::new("/tmp/x"), Path::new("/data"));
        assert_eq!(r, PathBuf::from("/data"));
    }

    #[test]
    fn resolve_relative_joins_relative() {
        let r = resolve_relative(Path::new("/tmp/x"), Path::new("sub"));
        assert_eq!(r, PathBuf::from("/tmp/x/sub"));
    }

    #[test]
    fn push_opt_appends_when_set() {
        let mut v = Vec::new();
        push_opt_u32(&mut v, "--epochs", Some(5));
        push_opt_f32(&mut v, "--lr", Some(0.1));
        push_opt_str(&mut v, "--name", "abc");
        push_opt_u32(&mut v, "--noop", None);
        push_opt_str(&mut v, "--empty", "");
        assert_eq!(v, vec!["--epochs", "5", "--lr", "0.1", "--name", "abc"]);
    }
}
