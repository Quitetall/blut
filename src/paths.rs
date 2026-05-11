//! Resolve runtime paths for the trainer subprocess + per-job state.
//!
//! Resolution policy is keep-it-discoverable: every input has an env
//! var the user can override. Defaults match the user's existing
//! local-llm layout (`~/local-llm/.venv` for python, `<crate>/python/`
//! for the bundled trainer.py during dev, XDG `data_local_dir/lamu/`
//! for everything else).

use std::path::PathBuf;

use crate::error::{Result, TrainError};

/// Directory holding all per-job state. One subdir per job id.
///
/// Default: `~/.local/share/lamu/train-jobs/`
/// Override: `$LAMU_TRAIN_JOBS_DIR`
pub fn jobs_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LAMU_TRAIN_JOBS_DIR") {
        return Ok(PathBuf::from(p));
    }
    let base = dirs::data_local_dir().ok_or_else(|| {
        TrainError::other("data_local_dir() unavailable; set $LAMU_TRAIN_JOBS_DIR")
    })?;
    Ok(base.join("lamu").join("train-jobs"))
}

/// Directory for one job. Created on demand.
pub fn job_dir(job_id: &str) -> Result<PathBuf> {
    let p = jobs_dir()?.join(job_id);
    std::fs::create_dir_all(&p).map_err(|e| TrainError::Io {
        path: p.clone(),
        source: e,
    })?;
    Ok(p)
}

/// Materialized JSONL data dir for `--from-conversations` etc.
///
/// Default: `~/.local/share/lamu/train-data/`
/// Override: `$LAMU_TRAIN_DATA_DIR`
pub fn data_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LAMU_TRAIN_DATA_DIR") {
        return Ok(PathBuf::from(p));
    }
    let base = dirs::data_local_dir().ok_or_else(|| {
        TrainError::other("data_local_dir() unavailable; set $LAMU_TRAIN_DATA_DIR")
    })?;
    Ok(base.join("lamu").join("train-data"))
}

/// Resolve the python interpreter to run trainer.py with.
///
/// Order:
///   1. `$LAMU_TRAIN_PYTHON` env (explicit override)
///   2. `~/local-llm/.venv/bin/python` (user's existing workhorse venv)
///   3. `~/.local/share/lamu/train-venv/bin/python` (managed venv, if
///      ever created — placeholder; venv bootstrap is a future step)
///   4. `python3` on `$PATH` (last resort; deps may be missing)
pub fn resolve_python() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LAMU_TRAIN_PYTHON") {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir().ok_or_else(|| {
        TrainError::other("home_dir() unavailable; set $LAMU_TRAIN_PYTHON")
    })?;
    let candidates = [
        home.join("local-llm/.venv/bin/python"),
        home.join(".local/share/lamu/train-venv/bin/python"),
    ];
    for c in candidates {
        if c.exists() {
            return Ok(c);
        }
    }
    // Last-ditch: rely on PATH. Spawn-time errors will surface
    // missing-deps clearly via trainer.py's lazy import.
    Ok(PathBuf::from("python3"))
}

/// Resolve trainer.py.
///
/// Order:
///   1. `$LAMU_TRAINER_PY` env (explicit override; for hermetic tests)
///   2. `<crate manifest dir>/python/trainer.py` (development /
///      cargo-run path; works when the binary is invoked from the
///      workspace).
///   3. Sibling-of-binary lookup: `<dir-of-current-exe>/../share/lamu/python/trainer.py`
///      (FHS-ish layout for a future `cargo install` deployment).
///   4. `~/.local/share/lamu/python/trainer.py` (user-installed copy).
///
/// First existing path wins. Errors with the env var name if none
/// resolve so the user has one sentence to fix.
/// Resolve a paradigm-specific trainer script
/// (`trainer_dpo.py`, `trainer_distill.py`, etc.). Same search
/// order as `resolve_trainer_script` but with a configurable
/// filename. Used by DPO + distill stages.
pub fn resolve_trainer_script_named(name: &str) -> Result<PathBuf> {
    if let Ok(p) = std::env::var(format!("LAMU_{}_PY", name.to_uppercase())) {
        return Ok(PathBuf::from(p));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python")
            .join(name),
    );
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent().and_then(|p| p.parent()) {
            candidates.push(dir.join("share/lamu/python").join(name));
        }
    }
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join(".local/share/lamu/python").join(name));
    }
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }
    Err(TrainError::other(format!(
        "{name} not found. Tried: {}. Set $LAMU_{}_PY to override.",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        name.to_uppercase()
    )))
}

pub fn resolve_trainer_script() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LAMU_TRAINER_PY") {
        return Ok(PathBuf::from(p));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python")
            .join("trainer.py"),
    );
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent().and_then(|p| p.parent()) {
            // <prefix>/bin/lamu-train → <prefix>/share/lamu/python/trainer.py
            candidates.push(dir.join("share").join("lamu").join("python").join("trainer.py"));
        }
    }
    if let Some(home) = dirs::home_dir() {
        candidates.push(
            home.join(".local/share/lamu/python/trainer.py"),
        );
    }
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }
    Err(TrainError::other(format!(
        "trainer.py not found. Tried: {}. \
         Set $LAMU_TRAINER_PY to override.",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::TEST_ENV_LOCK.lock().unwrap()
    }

    #[test]
    fn jobs_dir_respects_env() {
        let _g = lock();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::set_var("LAMU_TRAIN_JOBS_DIR", "/tmp/lamu-jobs-test");
        }
        assert_eq!(jobs_dir().unwrap(), PathBuf::from("/tmp/lamu-jobs-test"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
    }

    #[test]
    fn jobs_dir_default_under_data_local() {
        let _g = lock();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::remove_var("LAMU_TRAIN_JOBS_DIR");
        }
        let p = jobs_dir().unwrap();
        assert!(p.ends_with("lamu/train-jobs") || p.to_string_lossy().contains("lamu"));
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("LAMU_TRAIN_JOBS_DIR", v);
            }
        }
    }

    #[test]
    fn job_dir_creates_subdir() {
        let _g = lock();
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
        }
        let dir = job_dir("test-job-123").unwrap();
        assert!(dir.exists() && dir.is_dir());
        assert_eq!(dir.file_name().unwrap(), "test-job-123");
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
    }

    #[test]
    fn resolve_python_respects_env() {
        let _g = lock();
        let prev = std::env::var("LAMU_TRAIN_PYTHON").ok();
        unsafe {
            std::env::set_var("LAMU_TRAIN_PYTHON", "/usr/bin/python7");
        }
        assert_eq!(resolve_python().unwrap(), PathBuf::from("/usr/bin/python7"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_PYTHON", v),
                None => std::env::remove_var("LAMU_TRAIN_PYTHON"),
            }
        }
    }

    #[test]
    fn resolve_trainer_finds_crate_dev_path() {
        let _g = lock();
        let prev = std::env::var("LAMU_TRAINER_PY").ok();
        unsafe {
            std::env::remove_var("LAMU_TRAINER_PY");
        }
        // Inside the crate during cargo test, the dev path always
        // resolves because trainer.py is checked in at python/.
        let p = resolve_trainer_script().expect("dev trainer.py must resolve");
        assert!(p.ends_with("python/trainer.py"), "got: {}", p.display());
        assert!(p.exists(), "resolved path must exist on disk");
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("LAMU_TRAINER_PY", v);
            }
        }
    }

    #[test]
    fn resolve_trainer_respects_env() {
        let _g = lock();
        let prev = std::env::var("LAMU_TRAINER_PY").ok();
        unsafe {
            std::env::set_var("LAMU_TRAINER_PY", "/some/custom/trainer.py");
        }
        assert_eq!(
            resolve_trainer_script().unwrap(),
            PathBuf::from("/some/custom/trainer.py")
        );
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAINER_PY", v),
                None => std::env::remove_var("LAMU_TRAINER_PY"),
            }
        }
    }
}
