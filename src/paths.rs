//! Resolve runtime paths for the trainer subprocess + per-job state.
//!
//! Resolution policy is keep-it-discoverable: every input has an env
//! var the user can override. Defaults match the user's existing
//! local-llm layout (`~/local-llm/.venv` for python, `<crate>/python/`
//! for the bundled trainer.py during dev, XDG `data_local_dir/lamu/`
//! for everything else).

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
// LamQuant multi-root resolution (RCP-1 / RCP-7 / RCP-9)
//
// The monorepo→submodule split scattered the wrapped python scripts
// across THREE roots, none of which is `blut`'s own location:
//
//   • `ai_models/`  lives in the sibling submodule `LamQuant-Neural/`
//     (e.g. `LamQuant-Neural/ai_models/snn/train_mamba_snn.py`).
//   • `scripts/`    lives at the META-repo root
//     (e.g. `<meta>/scripts/bulk_lml_to_lma.py`).
//   • `pccp/`       lives wherever the gate script + registry sit
//     (currently `LamQuant-Neural/pccp/`).
//
// A single `lamquant_home` cannot satisfy both `<home>/ai_models/`
// and `<home>/scripts/` at once, so the old single-root resolution
// made every LamQuant recipe die at its stage-1 `script.exists()`
// preflight. `LamquantRoots` resolves each root independently, with
// an explicit env override per root and a meta-repo auto-detect.
// ─────────────────────────────────────────────────────────────────

/// Canonical labels NPZ root (RCP-6). The monorepo split left three
/// drifted label paths (`<home>/ai_models/snn/labels`,
/// `/mnt/4tb/LamQuant/ai_models/snn/labels`, …); this is the single
/// source of truth, shared by the convert stage + the wrapped
/// `bulk_lml_to_lma.py` default. Override per-run via the stage's
/// `labels_dir_rel` arg or the packer's `--labels-dir`.
pub const DEFAULT_LABELS_DIR: &str = "/mnt/4tb/data/Training/labels";

/// The three (plus firmware) filesystem roots the LamQuant recipe
/// stages resolve their wrapped scripts against. Each root is the
/// directory that *contains* the named subtree (so `ai_models_root`
/// holds `ai_models/`, not `ai_models/` itself).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LamquantRoots {
    /// Dir containing `ai_models/`. Default `<meta>/LamQuant-Neural`.
    pub ai_models_root: PathBuf,
    /// Dir containing `scripts/`. Default `<meta>` (meta-repo root).
    pub scripts_root: PathBuf,
    /// Dir containing `pccp/` (registry + verification_records).
    /// Default: wherever `pccp/` is found (currently `ai_models_root`).
    pub pccp_root: PathBuf,
    /// Dir containing the Lossless submodule that builds the `lml`
    /// encode/decode binary (`<lossless_root>/target/release/lml`).
    /// Default `<meta>/LamQuant-Lossless`.
    pub lossless_root: PathBuf,
}

/// Path of the `lml` encode/decode binary RELATIVE to `lossless_root`.
/// The Lossless submodule is a cargo crate; `cargo build --release`
/// drops the binary here. RCP-2 resolves the path; it does NOT build
/// the binary (encode is the operator's `cargo build --release` step).
pub const LML_BINARY_REL: &[&str] = &["target", "release", "lml"];

/// Detect the LamQuant meta-repo root: the directory that owns the
/// submodules (`blut/`, `LamQuant-Neural/`, …).
///
/// Detection order:
///   1. `$BLUT_META_ROOT` (explicit override).
///   2. Walk up from `$CARGO_MANIFEST_DIR` (dev/cargo-test) then from
///      `current_exe()` (installed), looking for a dir that both has
///      a `.gitmodules` AND a `LamQuant-Neural/` child — the
///      unambiguous meta-repo signature.
///   3. Sensible default `/mnt/4tb/LamQuant`.
///
/// Returns a clean `Err` only if every candidate is unusable AND the
/// default does not exist, so callers never panic.
pub fn meta_repo_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_META_ROOT") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Ok(p);
        }
        return Err(TrainError::other(format!(
            "$BLUT_META_ROOT={} is not a directory",
            p.display()
        )));
    }

    let mut starts: Vec<PathBuf> = Vec::new();
    // CARGO_MANIFEST_DIR points at `<meta>/blut` during dev/test.
    starts.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            starts.push(dir.to_path_buf());
        }
    }
    for start in &starts {
        if let Some(meta) = walk_up_for_meta(start) {
            return Ok(meta);
        }
    }

    let default = PathBuf::from("/mnt/4tb/LamQuant");
    if default.is_dir() {
        return Ok(default);
    }
    Err(TrainError::other(format!(
        "could not detect the LamQuant meta-repo: no ancestor of {:?} has \
         both .gitmodules and LamQuant-Neural/, and the default {} does not \
         exist. Set $BLUT_META_ROOT (or the per-root $BLUT_AI_MODELS / \
         $BLUT_SCRIPTS / $BLUT_PCCP) to override.",
        starts,
        default.display()
    )))
}

/// Walk up from `start` looking for the meta-repo signature
/// (`.gitmodules` + `LamQuant-Neural/`). Returns the first match.
fn walk_up_for_meta(start: &Path) -> Option<PathBuf> {
    let mut cur: Option<&Path> = Some(start);
    while let Some(dir) = cur {
        let has_gitmodules = dir.join(".gitmodules").is_file();
        let has_neural = dir.join("LamQuant-Neural").is_dir();
        if has_gitmodules && has_neural {
            return Some(dir.to_path_buf());
        }
        cur = dir.parent();
    }
    None
}

impl LamquantRoots {
    /// Resolve all roots from env overrides + meta-repo detection.
    ///
    /// Per-root env overrides (each takes precedence over detection):
    ///   • `$BLUT_AI_MODELS` (or `$LAMQUANT_NEURAL`) → `ai_models_root`
    ///   • `$BLUT_SCRIPTS`                            → `scripts_root`
    ///   • `$BLUT_PCCP`                               → `pccp_root`
    ///
    /// Without overrides:
    ///   • `ai_models_root` = `<meta>/LamQuant-Neural` if it holds
    ///     `ai_models/`, else `<meta>` (monorepo fallback).
    ///   • `scripts_root`   = `<meta>` if it holds `scripts/`, else
    ///     `ai_models_root` if IT holds `scripts/`.
    ///   • `pccp_root`      = first of [`ai_models_root`, `<meta>`]
    ///     that holds `pccp/`.
    ///
    /// Each resolved root is validated to contain its named subtree;
    /// a missing root yields a clean `Err` (never a panic).
    pub fn resolve() -> Result<Self> {
        let meta = meta_repo_root().ok();
        let ai_models_root = resolve_ai_models_root(meta.as_deref())?;
        let scripts_root = resolve_scripts_root(meta.as_deref(), &ai_models_root)?;
        let pccp_root = resolve_pccp_root(meta.as_deref(), &ai_models_root)?;
        let lossless_root = resolve_lossless_root(meta.as_deref())?;
        Ok(Self {
            ai_models_root,
            scripts_root,
            pccp_root,
            lossless_root,
        })
    }

    /// Absolute path to the `lml` encode/decode binary (RCP-2).
    ///
    /// Resolution order:
    ///   1. `$BLUT_LML` — explicit path to the binary itself.
    ///   2. `<lossless_root>/target/release/lml`.
    ///
    /// Unlike `ai_models_script` / `scripts_script`, this does NOT
    /// existence-check: the release binary is produced by an operator
    /// `cargo build --release` in the Lossless submodule and may not be
    /// present in a fresh checkout. Callers (the encode stage) assert
    /// existence at run-time preflight so the path is always resolvable
    /// for wiring + tests even before the binary is built.
    pub fn lml_binary(&self) -> PathBuf {
        if let Ok(p) = std::env::var("BLUT_LML") {
            return PathBuf::from(p);
        }
        let mut p = self.lossless_root.clone();
        for c in LML_BINARY_REL {
            p.push(c);
        }
        p
    }

    /// Build + existence-check the absolute path to a script under
    /// `ai_models/`. `rel` is the path RELATIVE to `ai_models_root`
    /// and MUST start with `ai_models` (kept explicit so call sites
    /// read like the on-disk layout). Clean `Err` if absent.
    pub fn ai_models_script(&self, rel: &[&str]) -> Result<PathBuf> {
        join_existing(&self.ai_models_root, rel, "ai_models", "BLUT_AI_MODELS")
    }

    /// Build + existence-check a script under `scripts/`. `rel` is
    /// relative to `scripts_root` and MUST start with `scripts`.
    pub fn scripts_script(&self, rel: &[&str]) -> Result<PathBuf> {
        join_existing(&self.scripts_root, rel, "scripts", "BLUT_SCRIPTS")
    }
}

/// `$BLUT_AI_MODELS` / `$LAMQUANT_NEURAL` → `<meta>/LamQuant-Neural`
/// (if it holds `ai_models/`) → `<meta>` (monorepo fallback).
fn resolve_ai_models_root(meta: Option<&Path>) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_AI_MODELS").or_else(|_| std::env::var("LAMQUANT_NEURAL")) {
        let p = PathBuf::from(p);
        return validate_holds(p, "ai_models", "$BLUT_AI_MODELS");
    }
    let meta = meta.ok_or_else(|| {
        TrainError::other(
            "ai_models_root: meta-repo not detected; set $BLUT_AI_MODELS to the dir holding ai_models/",
        )
    })?;
    let neural = meta.join("LamQuant-Neural");
    if neural.join("ai_models").is_dir() {
        return Ok(neural);
    }
    if meta.join("ai_models").is_dir() {
        return Ok(meta.to_path_buf());
    }
    Err(TrainError::other(format!(
        "ai_models/ not found under {} or {}; set $BLUT_AI_MODELS",
        neural.display(),
        meta.display()
    )))
}

/// `$BLUT_SCRIPTS` → `<meta>` (if it holds `scripts/`) → `ai_models_root`.
fn resolve_scripts_root(meta: Option<&Path>, ai_models_root: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_SCRIPTS") {
        let p = PathBuf::from(p);
        return validate_holds(p, "scripts", "$BLUT_SCRIPTS");
    }
    if let Some(meta) = meta {
        if meta.join("scripts").is_dir() {
            return Ok(meta.to_path_buf());
        }
    }
    if ai_models_root.join("scripts").is_dir() {
        return Ok(ai_models_root.to_path_buf());
    }
    Err(TrainError::other(format!(
        "scripts/ not found under the meta-repo or {}; set $BLUT_SCRIPTS",
        ai_models_root.display()
    )))
}

/// `$BLUT_PCCP` → first dir holding `pccp/`, searched in order:
/// `ai_models_root`, `<meta>/LamQuant-Neural`, `<meta>`. The Neural
/// submodule is checked explicitly so a `BLUT_AI_MODELS` override (to
/// a stub) doesn't lose the real `LamQuant-Neural/pccp/` that the gate
/// stages read.
fn resolve_pccp_root(meta: Option<&Path>, ai_models_root: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_PCCP") {
        let p = PathBuf::from(p);
        return validate_holds(p, "pccp", "$BLUT_PCCP");
    }
    let mut candidates: Vec<PathBuf> = vec![ai_models_root.to_path_buf()];
    if let Some(meta) = meta {
        candidates.push(meta.join("LamQuant-Neural"));
        candidates.push(meta.to_path_buf());
    }
    for c in &candidates {
        if c.join("pccp").is_dir() {
            return Ok(c.clone());
        }
    }
    Err(TrainError::other(format!(
        "pccp/ not found under {:?}; set $BLUT_PCCP",
        candidates
    )))
}

/// `$BLUT_LOSSLESS` → `<meta>/LamQuant-Lossless` (the sibling submodule
/// that builds the `lml` binary) → `<meta>` (monorepo fallback, where a
/// `target/` would sit at root). Validated to be a directory; the
/// release binary inside it is NOT required to exist at resolve time
/// (RCP-2: encode is the operator's `cargo build --release` step).
fn resolve_lossless_root(meta: Option<&Path>) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_LOSSLESS") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Ok(p);
        }
        return Err(TrainError::other(format!(
            "$BLUT_LOSSLESS={} is not a directory",
            p.display()
        )));
    }
    let meta = meta.ok_or_else(|| {
        TrainError::other(
            "lossless_root: meta-repo not detected; set $BLUT_LOSSLESS to the dir \
             that builds the lml binary (holds target/release/lml)",
        )
    })?;
    let lossless = meta.join("LamQuant-Lossless");
    if lossless.is_dir() {
        return Ok(lossless);
    }
    // Monorepo fallback: the Lossless crate lives at the meta root.
    Ok(meta.to_path_buf())
}

/// Validate that `root` holds the `expects` subtree; clean `Err`
/// naming the env var to set if not.
fn validate_holds(root: PathBuf, expects: &str, env_name: &str) -> Result<PathBuf> {
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
/// final path exists. Clean `Err` naming the override env var.
fn join_existing(root: &Path, rel: &[&str], expect_first: &str, env_name: &str) -> Result<PathBuf> {
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

    /// RCP-2: `$BLUT_LML` overrides the computed binary path.
    #[test]
    fn lml_binary_respects_env_override() {
        let _g = lock();
        let prev = std::env::var("BLUT_LML").ok();
        unsafe {
            std::env::set_var("BLUT_LML", "/custom/path/to/lml");
        }
        let roots = LamquantRoots {
            ai_models_root: PathBuf::from("/x"),
            scripts_root: PathBuf::from("/x"),
            pccp_root: PathBuf::from("/x"),
            lossless_root: PathBuf::from("/x/LamQuant-Lossless"),
        };
        assert_eq!(roots.lml_binary(), PathBuf::from("/custom/path/to/lml"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BLUT_LML", v),
                None => std::env::remove_var("BLUT_LML"),
            }
        }
    }

    /// Without `$BLUT_LML`, the binary path is
    /// `<lossless_root>/target/release/lml` (RCP-2). Resolution does
    /// NOT existence-check — the release binary is the operator's
    /// `cargo build --release` step.
    #[test]
    fn lml_binary_defaults_under_lossless_root() {
        let _g = lock();
        let prev = std::env::var("BLUT_LML").ok();
        unsafe {
            std::env::remove_var("BLUT_LML");
        }
        let roots = LamquantRoots {
            ai_models_root: PathBuf::from("/meta/LamQuant-Neural"),
            scripts_root: PathBuf::from("/meta"),
            pccp_root: PathBuf::from("/meta/LamQuant-Neural"),
            lossless_root: PathBuf::from("/meta/LamQuant-Lossless"),
        };
        assert_eq!(
            roots.lml_binary(),
            PathBuf::from("/meta/LamQuant-Lossless/target/release/lml")
        );
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("BLUT_LML", v);
            }
        }
    }
}
