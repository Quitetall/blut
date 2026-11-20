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
//!
//! # One root for a separate install: `$LAMU_TRAIN_DATA_DIR`
//!
//! When `$LAMU_TRAIN_DATA_DIR` is set, it is the root of ALL of a user's
//! BLUT state ([`state_root`]), so two values of it are two independent
//! installs. Each store resolves, in order:
//!
//! 1. its own explicit variable, if set (unchanged, e.g. `$BLUT_REGISTRY_DB`);
//! 2. `$LAMU_TRAIN_DATA_DIR/<name>` (the table below);
//! 3. its historical default under `$HOME` / XDG.
//!
//! With `$LAMU_TRAIN_DATA_DIR` unset, every path is exactly what it was.
//!
//! | Store | Explicit variable | Under the root | Default |
//! |---|---|---|---|
//! | jobs | `LAMU_TRAIN_JOBS_DIR` | `train-jobs/` | `~/.local/share/lamu/train-jobs/` |
//! | global stage cache | `LAMU_TRAIN_CACHE_DIR` | `train-cache/` | `~/.local/share/lamu/train-cache/` |
//! | model + plan registry | `BLUT_REGISTRY_DB` | `registry.db` | `~/.blut/registry.db` |
//! | dataset catalog tags | `BLUT_CATALOG_DB` | `catalog.db` | `~/.blut/catalog.db` |
//! | datasets (shared with LAMU memory) | `LAMU_MEMORY_DB` | `conversations.db` | `~/.local/share/lamu/conversations.db` |
//! | auto-train policy | `LAMU_TRAIN_POLICY` | `train-policy.toml` | `~/.config/lamu/train-policy.toml` |
//! | footprint calibration | `BLUT_FOOTPRINT_PATH` | `footprints.json` | `~/.config/blut/footprints.json` |
//! | backend registry config | `BLUT_REGISTRY_PATH` | `registry.yaml` | `~/.config/blut/registry.yaml` |
//! | partitions | `BLUT_PARTITIONS_DIR` | `partitions/` | `~/.config/blut/partitions/` |
//! | tenant quotas | `BLUT_TENANTS_CONFIG` | `tenants.toml` | `~/.config/blut/tenants.toml` |
//! | declarative recipes | `BLUT_USER_RECIPES_DIR` | `recipes/` | `~/.config/blut/recipes/` |
//! | `~/.blut` files (SLA, cost, budget, triggers, events) | per file | the root itself | `~/.blut/` |
//!
//! Not moved: the scheduler lock (it arbitrates the GPU between every
//! process on the machine, so separate installs must still see one lock),
//! the systemd timer units of `blut schedule` (systemd reads only
//! `~/.config/systemd/user`), and the materialized training data, which
//! already lived in `$LAMU_TRAIN_DATA_DIR`.
//!
//! Setting the variable on an existing install points it at a new, empty
//! state; set the per-store variables to keep using the old stores. The CLI
//! warns once per invocation when that has happened ([`stranded_stores`]):
//! a store whose own variable is unset, that is empty under the root, and
//! whose old default location holds data is named with the variable that
//! points back at it.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::error::{Result, TrainError};

/// The root of a separate install: `$LAMU_TRAIN_DATA_DIR` when set and
/// non-empty (see the module docs for what lives under it).
pub fn state_root() -> Option<PathBuf> {
    std::env::var_os("LAMU_TRAIN_DATA_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// `<state root>/<name>` when `$LAMU_TRAIN_DATA_DIR` is set, else `None`
/// (the store keeps its historical default).
pub fn under_state_root(name: &str) -> Option<PathBuf> {
    state_root().map(|root| root.join(name))
}

/// A store left behind by `$LAMU_TRAIN_DATA_DIR`: its own variable is unset,
/// it is empty under the root, and its old default location holds data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrandedStore {
    /// What the store holds, e.g. `jobs` or `model + plan registry`.
    pub store: &'static str,
    /// The store's own variable; set it to [`old`](Self::old) to keep the data.
    pub variable: &'static str,
    /// The old default location, which holds data.
    pub old: PathBuf,
}

/// Where an old default lives, relative to one of the user's base directories.
#[derive(Clone, Copy)]
enum Base {
    /// `$HOME`.
    Home,
    /// `dirs::data_local_dir()` (`$XDG_DATA_HOME`, `~/.local/share`).
    DataLocal,
    /// `dirs::config_dir()` (`$XDG_CONFIG_HOME`, `~/.config`).
    Config,
}

/// Every store `$LAMU_TRAIN_DATA_DIR` roots (the module table): what it is, its
/// own variables (the first is the one to suggest), its name under the root,
/// and its old default as a base directory plus a relative path.
#[rustfmt::skip]
const ROOTED_STORES: &[(&str, &[&str], &str, Base, &str)] = &[
    ("jobs", &["LAMU_TRAIN_JOBS_DIR"], "train-jobs", Base::DataLocal, "lamu/train-jobs"),
    ("global stage cache", &["LAMU_TRAIN_CACHE_DIR"], "train-cache", Base::DataLocal, "lamu/train-cache"),
    ("model + plan registry", &["BLUT_REGISTRY_DB"], "registry.db", Base::Home, ".blut/registry.db"),
    ("dataset catalog tags", &["BLUT_CATALOG_DB"], "catalog.db", Base::Home, ".blut/catalog.db"),
    ("datasets", &["LAMU_MEMORY_DB"], "conversations.db", Base::DataLocal, "lamu/conversations.db"),
    ("auto-train policy", &["LAMU_TRAIN_POLICY"], "train-policy.toml", Base::Config, "lamu/train-policy.toml"),
    ("footprint calibration", &["BLUT_FOOTPRINT_PATH"], "footprints.json", Base::Config, "blut/footprints.json"),
    ("backend registry config", &["BLUT_REGISTRY_PATH", "LAMU_REGISTRY_PATH"], "registry.yaml", Base::Config, "blut/registry.yaml"),
    ("partitions", &["BLUT_PARTITIONS_DIR"], "partitions", Base::Config, "blut/partitions"),
    ("tenant quotas", &["BLUT_TENANTS_CONFIG"], "tenants.toml", Base::Config, "blut/tenants.toml"),
    ("declarative recipes", &["BLUT_USER_RECIPES_DIR"], "recipes", Base::Config, "blut/recipes"),
    ("SLA log", &["BLUT_SLA_PATH"], "sla.jsonl", Base::Home, ".blut/sla.jsonl"),
    ("SLA rules", &["BLUT_SLA_RULES"], "sla.toml", Base::Home, ".blut/sla.toml"),
    ("cost ledger", &["BLUT_COST_LEDGER"], "cost.jsonl", Base::Home, ".blut/cost.jsonl"),
    ("budget caps", &["BLUT_BUDGET_CAP"], "budget.toml", Base::Home, ".blut/budget.toml"),
    ("trigger bindings", &["BLUT_TRIGGERS"], "triggers.toml", Base::Home, ".blut/triggers.toml"),
    ("trigger events", &["BLUT_EVENTS_DIR"], "events", Base::Home, ".blut/events"),
];

/// The stores `$LAMU_TRAIN_DATA_DIR` left behind (see [`StrandedStore`]);
/// empty when the variable is unset. An install that set the variable before
/// it rooted every store (it used to hold only materialized training data)
/// otherwise sees empty jobs and registries with no hint of where they went.
pub fn stranded_stores() -> Vec<StrandedStore> {
    stranded_stores_in(
        &|name| std::env::var_os(name),
        dirs::home_dir().as_deref(),
        dirs::data_local_dir().as_deref(),
        dirs::config_dir().as_deref(),
    )
}

/// [`stranded_stores`] over an explicit environment and base directories, so
/// it is testable without touching the process environment.
fn stranded_stores_in(
    env: &dyn Fn(&str) -> Option<OsString>,
    home: Option<&Path>,
    data_local: Option<&Path>,
    config: Option<&Path>,
) -> Vec<StrandedStore> {
    let Some(root) = env("LAMU_TRAIN_DATA_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &(store, variables, under_root, base, rel) in ROOTED_STORES {
        if variables.iter().any(|v| env(v).is_some()) {
            continue; // the store's own variable decides; the root does not apply
        }
        let base = match base {
            Base::Home => home,
            Base::DataLocal => data_local,
            Base::Config => config,
        };
        let Some(old) = base.map(|b| b.join(rel)) else {
            continue;
        };
        let new = root.join(under_root);
        if old != new && holds_data(&old) && !holds_data(&new) {
            out.push(StrandedStore {
                store,
                variable: variables[0],
                old,
            });
        }
    }
    out
}

/// Whether `path` holds user data: a directory with a file somewhere in its
/// first few levels, a SQLite database with a row in any table, or any other
/// non-empty file. A database created by an earlier command but never
/// written to is empty, so the warning does not stop the first time a
/// command opens the new store.
fn holds_data(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if meta.is_dir() {
        return tree_has_file(path, 4);
    }
    if meta.len() == 0 {
        return false;
    }
    if path.extension().is_some_and(|e| e == "db") {
        return sqlite_has_rows(path).unwrap_or(true);
    }
    true
}

fn tree_has_file(dir: &Path, depth: u32) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|e| match e.file_type() {
        Ok(t) if t.is_dir() => depth > 0 && tree_has_file(&e.path(), depth - 1),
        Ok(_) => true,
        Err(_) => false,
    })
}

/// `Some(true)` when any table of the SQLite file at `path` has a row; `None`
/// when it cannot be read as one (the caller then judges by size).
fn sqlite_has_rows(path: &Path) -> Option<bool> {
    use rusqlite::{Connection, OpenFlags};
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .ok()?
        .query_map([], |r| r.get(0))
        .ok()?
        .collect::<rusqlite::Result<_>>()
        .ok()?;
    for table in tables {
        let quoted = table.replace('"', "\"\"");
        let any: Option<i64> = conn
            .query_row(&format!("SELECT 1 FROM \"{quoted}\" LIMIT 1"), [], |r| {
                r.get(0)
            })
            .ok();
        if any.is_some() {
            return Some(true);
        }
    }
    Some(false)
}

/// The warning [`warn_stranded_stores`] prints, or `None` when nothing is
/// stranded.
fn stranded_stores_warning(root: &Path, stranded: &[StrandedStore]) -> Option<String> {
    if stranded.is_empty() {
        return None;
    }
    let mut msg = format!(
        "$LAMU_TRAIN_DATA_DIR ({}) is the root of every BLUT store, and these stores \
         are empty under it while their old default locations hold data. To keep \
         using that data, set:",
        root.display()
    );
    for s in stranded {
        msg.push_str(&format!(
            "\n  {}={}  ({})",
            s.variable,
            s.old.display(),
            s.store
        ));
    }
    msg.push_str("\nor move the data under $LAMU_TRAIN_DATA_DIR.");
    Some(msg)
}

/// Warn, once per process, about stores `$LAMU_TRAIN_DATA_DIR` left behind
/// (see [`stranded_stores`]). `blut::cli::run` calls it at startup; the
/// warning goes to the tracing subscriber (stderr for the CLI).
pub fn warn_stranded_stores() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(root) = state_root() else {
            return;
        };
        if let Some(msg) = stranded_stores_warning(&root, &stranded_stores()) {
            tracing::warn!("{msg}");
        }
    });
}

/// The directory of BLUT's own state files (SLA log, cost ledger, budget,
/// trigger bindings and events): `$LAMU_TRAIN_DATA_DIR` when set, else
/// `$HOME/.blut`.
pub fn blut_home() -> PathBuf {
    state_root().unwrap_or_else(|| {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(".blut")
    })
}

/// Directory holding all per-job state. One subdir per job id.
///
/// Default: `~/.local/share/lamu/train-jobs/`
/// Override: `$LAMU_TRAIN_JOBS_DIR`, else `$LAMU_TRAIN_DATA_DIR/train-jobs/`
pub fn jobs_dir() -> Result<PathBuf> {
    if let Some(p) = utf8_path_env("LAMU_TRAIN_JOBS_DIR")? {
        return Ok(p);
    }
    // Job paths are recorded in JSON, so the root must be UTF-8 as well.
    if let Some(root) = utf8_path_env("LAMU_TRAIN_DATA_DIR")?.filter(|p| !p.as_os_str().is_empty())
    {
        return Ok(root.join("train-jobs"));
    }
    let base = dirs::data_local_dir().ok_or_else(|| {
        TrainError::other("data_local_dir() unavailable; set $LAMU_TRAIN_JOBS_DIR")
    })?;
    Ok(base.join("lamu").join("train-jobs"))
}

/// A directory override whose paths end up inside JSON (artifact handles,
/// sidecars, lineage). A non-UTF-8 value cannot be recorded there, so it is
/// refused up front, naming the variable, rather than ignored (it used to
/// fall back to the default directory without a word) or accepted only for
/// the first stage to fail serializing its output.
fn utf8_path_env(name: &str) -> Result<Option<PathBuf>> {
    match std::env::var(name) {
        Ok(p) => Ok(Some(PathBuf::from(p))),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(raw)) => Err(TrainError::other(format!(
            "${name} ({}) is not valid UTF-8; blut records job paths in JSON, so \
             point it at a UTF-8 path",
            raw.to_string_lossy()
        ))),
    }
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
    if let Some(p) = utf8_path_env("LAMU_TRAIN_DATA_DIR")? {
        return Ok(p);
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
    if let Some(p) = std::env::var_os("LAMU_TRAIN_PYTHON") {
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
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent().and_then(|p| p.parent())
    {
        candidates.push(dir.join("share/lamu/python").join(name));
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
    resolve_trainer_script_with(std::env::var("LAMU_TRAINER_PY").ok())
}

/// [`resolve_trainer_script`] with the override passed in rather than read from
/// the process environment.
///
/// The env read is lifted to the caller so tests can exercise both branches
/// WITHOUT mutating global state. They used to `unsafe { env::set_var(...) }`
/// around a shared mutex, which serializes the two tests against each other but
/// not against the hundreds of others reading env concurrently — that is the
/// data race that made `set_var` unsafe in Rust 2024. It passed on a 32-core
/// dev box and failed both tests on a 4-core CI runner, where the interleaving
/// differs.
fn resolve_trainer_script_with(override_path: Option<String>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(PathBuf::from(p));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("python")
            .join("trainer.py"),
    );
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent().and_then(|p| p.parent())
    {
        // <prefix>/bin/lamu-train → <prefix>/share/lamu/python/trainer.py
        candidates.push(
            dir.join("share")
                .join("lamu")
                .join("python")
                .join("trainer.py"),
        );
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
        crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner())
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
        // Engine carve (v1.0): the engine ships ZERO python — the generic
        // `trainer.py` moved to `blut-backends`. With no override and no
        // checked-in `python/trainer.py`, resolution must fail-closed with a
        // clear error (not silently succeed).
        //
        // No env mutation and no lock: the override is a parameter. The
        // previous form wrote LAMU_TRAINER_PY with `unsafe { set_var }`, which
        // races every concurrent env reader in the suite and failed on CI while
        // passing locally.
        let err = resolve_trainer_script_with(None)
            .expect_err("pure engine ships no trainer.py — resolution must error");
        assert!(
            err.to_string().contains("trainer.py not found"),
            "got: {err}"
        );
    }

    #[test]
    fn resolve_trainer_respects_env() {
        // Same reason: the override is passed in, not written to the process
        // environment.
        assert_eq!(
            resolve_trainer_script_with(Some("/some/custom/trainer.py".to_string())).unwrap(),
            PathBuf::from("/some/custom/trainer.py")
        );
    }

    /// E2E follow-up: an install that already set `$LAMU_TRAIN_DATA_DIR`
    /// saw empty jobs and registries once the variable rooted every store,
    /// with no warning. The stores left behind are named with the variable
    /// that points back at them.
    #[test]
    fn stores_left_behind_by_the_state_root_are_named() {
        let td = tempfile::tempdir().unwrap();
        let (home, data, config, root) = (
            td.path().join("home"),
            td.path().join("data"),
            td.path().join("config"),
            td.path().join("root"),
        );
        // Old data: a job, a registry with a row, a policy file.
        std::fs::create_dir_all(data.join("lamu/train-jobs/job-1")).unwrap();
        std::fs::write(data.join("lamu/train-jobs/job-1/status.jsonl"), "{}\n").unwrap();
        std::fs::create_dir_all(home.join(".blut")).unwrap();
        let reg = rusqlite::Connection::open(home.join(".blut/registry.db")).unwrap();
        reg.execute_batch("CREATE TABLE t (x); INSERT INTO t VALUES (1);")
            .unwrap();
        drop(reg);
        std::fs::create_dir_all(config.join("lamu")).unwrap();
        std::fs::write(config.join("lamu/train-policy.toml"), "x = 1\n").unwrap();
        // The root exists (it held materialized training data) and an earlier
        // command created an empty registry and an empty jobs tree in it.
        std::fs::create_dir_all(root.join("train-jobs/empty")).unwrap();
        let empty = rusqlite::Connection::open(root.join("registry.db")).unwrap();
        empty.execute_batch("CREATE TABLE t (x);").unwrap();
        drop(empty);

        let vars = |pairs: &'static [(&'static str, &'static str)], root: PathBuf| {
            move |name: &str| -> Option<OsString> {
                if name == "LAMU_TRAIN_DATA_DIR" {
                    return Some(root.clone().into_os_string());
                }
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.into())
            }
        };
        let found = |env: &dyn Fn(&str) -> Option<OsString>| {
            stranded_stores_in(env, Some(&home), Some(&data), Some(&config))
        };

        let got = found(&vars(&[], root.clone()));
        let named: Vec<(&str, &Path)> = got.iter().map(|s| (s.variable, s.old.as_path())).collect();
        assert_eq!(
            named,
            vec![
                (
                    "LAMU_TRAIN_JOBS_DIR",
                    data.join("lamu/train-jobs").as_path()
                ),
                ("BLUT_REGISTRY_DB", home.join(".blut/registry.db").as_path()),
                (
                    "LAMU_TRAIN_POLICY",
                    config.join("lamu/train-policy.toml").as_path()
                ),
            ]
        );
        let msg = stranded_stores_warning(&root, &got).unwrap();
        assert!(
            msg.contains(&format!(
                "BLUT_REGISTRY_DB={}",
                home.join(".blut/registry.db").display()
            )),
            "{msg}"
        );

        // A store's own variable decides for it.
        let got = found(&vars(
            &[("LAMU_TRAIN_JOBS_DIR", "/elsewhere")],
            root.clone(),
        ));
        assert!(
            got.iter().all(|s| s.variable != "LAMU_TRAIN_JOBS_DIR"),
            "{got:?}"
        );

        // A store already in use under the root is not stranded.
        std::fs::write(root.join("train-policy.toml"), "y = 2\n").unwrap();
        let reg = rusqlite::Connection::open(root.join("registry.db")).unwrap();
        reg.execute("INSERT INTO t VALUES (2)", []).unwrap();
        drop(reg);
        let got = found(&vars(&[], root.clone()));
        let named: Vec<&str> = got.iter().map(|s| s.variable).collect();
        assert_eq!(named, vec!["LAMU_TRAIN_JOBS_DIR"]);

        // Unset: nothing to say.
        let unset = |_: &str| -> Option<OsString> { None };
        assert!(found(&unset).is_empty());
        assert!(stranded_stores_warning(&root, &[]).is_none());
    }

    /// A non-UTF-8 jobs dir was read with `env::var(..).ok()`, so it was
    /// dropped and jobs silently went to the default
    /// `~/.local/share/lamu/train-jobs`. It cannot be honoured either (job
    /// paths are recorded in JSON), so it is refused, naming the variable.
    #[cfg(unix)]
    #[test]
    fn a_non_utf8_jobs_dir_is_refused_not_ignored() {
        use std::os::unix::ffi::OsStrExt;
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let dir = std::ffi::OsStr::from_bytes(b"/nonexistent/jobs-\xff");
        let prev = std::env::var_os("LAMU_TRAIN_JOBS_DIR");
        // SAFETY: TEST_ENV_LOCK serializes env mutation; restored below.
        unsafe { std::env::set_var("LAMU_TRAIN_JOBS_DIR", dir) };
        let got = jobs_dir();
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
        let err = got.expect_err("must refuse, not fall back").to_string();
        assert!(err.contains("LAMU_TRAIN_JOBS_DIR"), "{err}");
    }
}
