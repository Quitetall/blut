// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Resolve runtime paths for the trainer subprocess + per-job state.
//!
//! Resolution policy is keep-it-discoverable: every input has an env
//! var the user can override. Defaults match a conventional local
//! layout (`~/local-llm/.venv` for python, `<crate>/python/` for the
//! bundled trainer.py during dev, XDG `data_local_dir/lamu/` for
//! everything else); the `LAMU_*` env vars below are the real
//! back-compat override hooks.

use std::path::{Path, PathBuf};

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

// ─────────────────────────────────────────────────────────────────
// Reusable root primitives (domain-agnostic).
//
// The engine offers only the root-existence / path-join error-format
// primitives below. Locating any PARTICULAR repo (a meta-repo, a
// submodule that holds `ai_models/`/`scripts/`, etc.) is a COOKBOOK
// concern — a cookbook composes these primitives with its own roots /
// env-var names. The engine assumes nothing about the directory it
// lives in (the contained launcher embeds its own helper script).
// ─────────────────────────────────────────────────────────────────

/// Validate that `root` holds the `expects` subtree; clean `Err`
/// naming the env var to set if not. Generic primitive — domain
/// cookbooks pass their own roots / env-var names.
pub fn validate_holds(root: PathBuf, expects: &str, env_name: &str) -> Result<PathBuf> {
    if root.join(expects).is_dir() {
        Ok(root)
    } else {
        Err(TrainError::other(format!(
            "{} does not hold {}/ (from {})",
            root.display(),
            expects,
            env_name
        )))
    }
}

/// Join `rel` onto `root`, assert the leading component matches
/// `expect_first` (guards against call-site drift), and assert the
/// final path exists. Clean `Err` naming the override env var. Generic
/// primitive — domain cookbooks pass their own roots / rel paths.
pub fn join_existing(
    root: &Path,
    rel: &[&str],
    expect_first: &str,
    env_name: &str,
) -> Result<PathBuf> {
    debug_assert_eq!(
        rel.first().copied(),
        Some(expect_first),
        "ai_models_script/scripts_script rel must start with {expect_first}"
    );
    let mut p = root.to_path_buf();
    for c in rel {
        p.push(c);
    }
    if p.exists() {
        Ok(p)
    } else {
        Err(TrainError::other(format!(
            "{} not found (root {}; override with ${})",
            p.display(),
            root.display(),
            env_name
        )))
    }
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
    let home = dirs::home_dir()
        .ok_or_else(|| TrainError::other("home_dir() unavailable; set $LAMU_TRAIN_PYTHON"))?;
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
            candidates.push(
                dir.join("share")
                    .join("lamu")
                    .join("python")
                    .join("trainer.py"),
            );
        }
    }
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join(".local/share/lamu/python/trainer.py"));
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
    fn resolve_trainer_errors_without_python_in_engine() {
        // Engine carve (v1.0): the engine ships ZERO python — the
        // generic `trainer.py` moved to `blut-backends`. With no env
        // override and no checked-in `python/trainer.py`, resolution
        // must fail-closed with a clear error (not silently succeed).
        let _g = lock();
        let prev = std::env::var("LAMU_TRAINER_PY").ok();
        unsafe {
            std::env::remove_var("LAMU_TRAINER_PY");
        }
        let err = resolve_trainer_script()
            .expect_err("pure engine ships no trainer.py — resolution must error");
        assert!(
            err.to_string().contains("trainer.py not found"),
            "got: {err}"
        );
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
