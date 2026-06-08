//! Shared helpers for LamQuant stages.
//!
//! Cuts boilerplate from the per-stage `run()` bodies: resolving
//! `lamquant_home`, checking script existence, building progress
//! fan-out, and the `safe_join` path helper.

use std::path::{Path, PathBuf};

use crate::backends::lamquant::runner::{Progress, default_lamquant_home, resolve_lamquant_python};
use crate::paths::LamquantRoots;
use blut::framework::error::StageError;
use blut::framework::status::StageEvent;

/// Resolve the multi-root LamQuant layout (RCP-1 / RCP-7), mapping a
/// resolution failure into `StageError::BadInput` so a missing root
/// surfaces as a clear preflight error rather than a low-level panic.
///
/// This is the structural fix for the submodule split: `ai_models/`
/// lives in the `LamQuant-Neural/` sibling submodule while `scripts/`
/// is at the meta-repo root, so a single `lamquant_home` can no longer
/// satisfy both. Stages call this once, then resolve their wrapped
/// script via `roots.ai_models_script(...)` / `roots.scripts_script(...)`.
pub fn resolve_roots() -> Result<LamquantRoots, StageError> {
    LamquantRoots::resolve().map_err(|e| StageError::BadInput(e.to_string()))
}

/// Resolve + existence-check a script under `scripts/` (at the
/// meta-repo root post-split). `rel` MUST start with `"scripts"`.
/// `BadInput` on miss.
///
/// (The `ai_models/` analogue isn't a separate helper here: the
/// `ai_models/*` stages keep using `resolve_home` + `script_path`,
/// where the resolved home already IS `ai_models_root`. Direct
/// callers / tests use `LamquantRoots::ai_models_script`.)
pub fn scripts_script(roots: &LamquantRoots, rel: &[&str]) -> Result<PathBuf, StageError> {
    roots
        .scripts_script(rel)
        .map_err(|e| StageError::BadInput(e.to_string()))
}

/// Resolve a BLUT-owned training/preprocess script under the
/// `blut_python_root` (`<blut>/python/lamquant/<area>/<file>.py`),
/// post MOVE-B (2026-05-29). `rel` MUST start with `"python"`.
/// `BadInput` on miss, naming the `$BLUT_PYTHON` override.
///
/// Returns `(script_path, blut_python_dir)` where `blut_python_dir` is
/// `<blut_python_root>/python` — the directory to put on `PYTHONPATH`
/// so the moved scripts resolve their `lamquant.*` package imports.
pub fn blut_python_script(rel: &[&str]) -> Result<(PathBuf, PathBuf), StageError> {
    let roots = resolve_roots()?;
    let script = roots
        .blut_python_script(rel)
        .map_err(|e| StageError::BadInput(e.to_string()))?;
    let python_dir = roots.blut_python_root.join("python");
    Ok((script, python_dir))
}

/// Build the `PYTHONPATH` value for a BLUT-owned script subprocess:
/// the `blut/python` dir (so `lamquant.*` resolves) layered ahead of
/// any caller-supplied `PYTHONPATH`. The PRIVATE `lamquant_neural` and
/// PUBLIC `lamquant_core` / `lamquant_codec` wheels are pip-installed
/// in the venv, so only the BLUT python root needs injecting here.
pub fn blut_pythonpath(python_dir: &Path) -> String {
    let existing = std::env::var("PYTHONPATH").unwrap_or_default();
    if existing.is_empty() {
        python_dir.display().to_string()
    } else {
        format!("{}:{}", python_dir.display(), existing)
    }
}

/// Canonicalize and validate `lamquant_home`. Returns the absolute
/// path. `BadInput` if the path is missing or not canonicalizable.
///
/// When `raw` is empty the home defaults to the resolved
/// `ai_models_root` (the `LamQuant-Neural` submodule post-split),
/// which is where the `ai_models/*` scripts live AND where they write
/// their checkpoints (`weights/…`, `ai_models/<sub>/…ckpt`). This is
/// the RCP-1/RCP-7 fix: the old `~/Desktop/LamQuant` default no longer
/// holds `ai_models/`. An explicit `raw` is still honored verbatim for
/// hermetic tests + bespoke layouts.
pub fn resolve_home(raw: &str) -> Result<PathBuf, StageError> {
    let raw_path = if raw.is_empty() {
        match resolve_roots() {
            Ok(roots) => roots.ai_models_root,
            // Fall back to the legacy env/Desktop default so the
            // error message points at a path the user recognizes
            // rather than an opaque detection failure.
            Err(_) => default_lamquant_home(),
        }
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
        return Err(StageError::BadInput(format!("{} not found", p.display())));
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
    fn push_opt_appends_when_set() {
        let mut v = Vec::new();
        push_opt_u32(&mut v, "--epochs", Some(5));
        push_opt_f32(&mut v, "--lr", Some(0.1));
        push_opt_u32(&mut v, "--noop", None);
        assert_eq!(v, vec!["--epochs", "5", "--lr", "0.1"]);
    }
}
