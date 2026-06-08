//! LamQuant multi-root path resolution (cookbook-local).
//!
//! Carved out of blut-core's `paths.rs` at C2a so blut-core holds NO
//! LamQuant domain path knowledge (which roots, which env vars, the
//! `/mnt/4tb` defaults). The generic primitives this builds on —
//! `meta_repo_root` (meta-repo detection), `validate_holds`,
//! `join_existing` — stay in `blut::paths` and are imported here.
//!
//! ─────────────────────────────────────────────────────────────────
//! The monorepo→submodule split scattered the wrapped python scripts
//! across THREE roots, none of which is `blut`'s own location:
//!
//!   • `ai_models/`  lives in the sibling submodule `LamQuant-Neural/`
//!     (e.g. `LamQuant-Neural/ai_models/snn/train_mamba_snn.py`).
//!   • `scripts/`    lives at the META-repo root
//!     (e.g. `<meta>/scripts/bulk_lml_to_lma.py`).
//!   • `pccp/`       lives wherever the gate script + registry sit
//!     (currently `LamQuant-Neural/pccp/`).
//!
//! A single `lamquant_home` cannot satisfy both `<home>/ai_models/`
//! and `<home>/scripts/` at once, so the old single-root resolution
//! made every LamQuant recipe die at its stage-1 `script.exists()`
//! preflight. `LamquantRoots` resolves each root independently, with
//! an explicit env override per root and a meta-repo auto-detect.
//! ─────────────────────────────────────────────────────────────────

use std::path::{Path, PathBuf};

use blut::error::{Result, TrainError};
use blut::paths::{join_existing, meta_repo_root, validate_holds};

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
    /// Dir containing the BLUT-owned python training tree
    /// (`<blut_python_root>/python/lamquant/...`). After the MOVE-B
    /// boundary migration, ALL training + preprocessing scripts live
    /// here (in the PUBLIC `blut/` submodule), NOT under
    /// `ai_models_root` (the PRIVATE Neural submodule). The two PCCP
    /// gate stages still resolve `ai_models/pccp_gate.py` under
    /// `ai_models_root` (governance stays in Neural). Default
    /// `<meta>/blut` (validated to hold `python/`).
    pub blut_python_root: PathBuf,
}

/// Path of the `lml` encode/decode binary RELATIVE to `lossless_root`.
/// The Lossless submodule is a cargo crate; `cargo build --release`
/// drops the binary here. RCP-2 resolves the path; it does NOT build
/// the binary (encode is the operator's `cargo build --release` step).
pub const LML_BINARY_REL: &[&str] = &["target", "release", "lml"];

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
        let blut_python_root = resolve_blut_python_root(meta.as_deref())?;
        Ok(Self {
            ai_models_root,
            scripts_root,
            pccp_root,
            lossless_root,
            blut_python_root,
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

    /// Build + existence-check a script under the BLUT-owned python
    /// training tree. `rel` is relative to `blut_python_root` and MUST
    /// start with `python` (so call sites read like the on-disk layout
    /// `python/lamquant/<area>/<script>.py`). Clean `Err` if absent.
    ///
    /// This is the MOVE-B analogue of `ai_models_script`: after the
    /// boundary migration, every train/preprocess stage resolves its
    /// wrapped script here instead of under `ai_models_root`.
    pub fn blut_python_script(&self, rel: &[&str]) -> Result<PathBuf> {
        join_existing(&self.blut_python_root, rel, "python", "BLUT_PYTHON")
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

/// `$BLUT_PYTHON` → `<meta>/blut` (the public BLUT submodule that owns
/// the training python tree post MOVE-B) → `<meta>` (monorepo
/// fallback). Validated to hold `python/`; a missing root yields a
/// clean `Err` naming the override knob.
fn resolve_blut_python_root(meta: Option<&Path>) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_PYTHON") {
        let p = PathBuf::from(p);
        return validate_holds(p, "python", "$BLUT_PYTHON");
    }
    let meta = meta.ok_or_else(|| {
        TrainError::other(
            "blut_python_root: meta-repo not detected; set $BLUT_PYTHON to the dir \
             that holds python/lamquant/ (the BLUT-owned training tree)",
        )
    })?;
    let blut = meta.join("blut");
    if blut.join("python").is_dir() {
        return Ok(blut);
    }
    // Monorepo fallback: python/ sits at the meta root.
    if meta.join("python").is_dir() {
        return Ok(meta.to_path_buf());
    }
    Err(TrainError::other(format!(
        "python/ not found under {} or {}; set $BLUT_PYTHON",
        blut.display(),
        meta.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Process-wide lock for tests that mutate environment variables.
    /// This test binary is a separate process from blut's, so a
    /// cookbook-local lock is correct (blut's `TEST_ENV_LOCK` is
    /// `pub(crate)`, invisible cross-crate).
    static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_ENV_LOCK.lock().unwrap()
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
            blut_python_root: PathBuf::from("/x/blut"),
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
            blut_python_root: PathBuf::from("/meta/blut"),
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
