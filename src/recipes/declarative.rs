// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Declarative (`.toml`) recipes (Phase G / C3).
//!
//! A user writes a recipe as a `.toml` file — a NAMED, ordered chain of
//! stages (by name) with per-stage args + a backend tag — and BLUT
//! compiles it to a runnable [`CompiledPlan`] with NO Rust. Stages are
//! resolved by name from the registered cookbooks' `stages_erased()`
//! registries; the chain is RUNTIME-kind-checked by
//! [`CompiledPlan::from_erased_chain`]. The produced plan runs through the
//! SAME executor as a compiled recipe.
//!
//! Format:
//! ```toml
//! name = "my_pipeline"
//! backend = "my_backend"        # informational tag (the stages carry the real backend)
//!
//! [[stages]]
//! stage = "prepare_data"
//! args  = { in_dir = "/data/raw", out_dir = "/data/packed", corpus = "dataset_a" }
//!
//! [[stages]]
//! stage = "train_model"
//! args  = { preset = "fast", tier = 8 }
//! ```
//!
//! Discovery (F4): `~/.config/blut/recipes/*.toml` (override with
//! `$BLUT_USER_RECIPES_DIR`).

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::framework::Registry;
use crate::framework::plan::CompiledPlan;

/// A declarative recipe parsed from a `.toml` file.
#[derive(Debug, Clone, Deserialize)]
pub struct DeclarativeRecipe {
    pub name: String,
    /// Informational backend tag (the resolved stages carry the real,
    /// compiler-checked backend; this is for display + discovery).
    #[serde(default)]
    pub backend: Option<String>,
    pub stages: Vec<DeclStage>,
}

/// One stage in a declarative chain.
#[derive(Debug, Clone, Deserialize)]
pub struct DeclStage {
    /// Stage name — must exist in a registered cookbook's `stages_erased()`.
    pub stage: String,
    /// Per-stage args (validated against the stage's schema at run time).
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, thiserror::Error)]
pub enum DeclarativeError {
    #[error("read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("declarative recipe '{recipe}' has no stages")]
    NoStages { recipe: String },
    #[error(
        "declarative recipe '{recipe}': stage '{stage}' is not in any registered \
         cookbook (see `blut stage list`)"
    )]
    UnknownStage { recipe: String, stage: String },
    /// A plan-build failure (the runtime kind-chain check).
    #[error("declarative recipe '{recipe}': {detail}")]
    Plan { recipe: String, detail: String },
}

impl DeclarativeRecipe {
    /// Parse a `.toml` file (does NOT resolve stages — see [`compile`]).
    pub fn load(path: &Path) -> Result<Self, DeclarativeError> {
        let body = std::fs::read_to_string(path).map_err(|e| DeclarativeError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        Self::parse(&body, &path.display().to_string())
    }

    /// Parse from a TOML string (path is for error messages only).
    pub fn parse(body: &str, path: &str) -> Result<Self, DeclarativeError> {
        toml::from_str(body).map_err(|e| DeclarativeError::Parse {
            path: path.to_string(),
            source: e,
        })
    }

    /// Resolve every stage by name from `reg`, then build a runtime-kind-checked
    /// linear [`CompiledPlan`]. An unknown stage or a kind-chain break is a
    /// clear error naming the offending stage.
    pub fn compile(&self, reg: &Registry) -> Result<CompiledPlan, DeclarativeError> {
        if self.stages.is_empty() {
            return Err(DeclarativeError::NoStages {
                recipe: self.name.clone(),
            });
        }
        let mut chain: Vec<(
            std::sync::Arc<dyn crate::framework::stage::StageDyn>,
            serde_json::Value,
        )> = Vec::with_capacity(self.stages.len());
        for s in &self.stages {
            let ctor =
                reg.find_erased_stage(&s.stage)
                    .ok_or_else(|| DeclarativeError::UnknownStage {
                        recipe: self.name.clone(),
                        stage: s.stage.clone(),
                    })?;
            chain.push((ctor(), s.args.clone()));
        }
        // recipe_args = a per-stage provenance summary (audit only; the
        // executor uses each node's own args).
        let recipe_args = serde_json::json!({
            "declarative": true,
            "backend": self.backend,
            "stages": self.stages.iter()
                .map(|s| serde_json::json!({ "stage": s.stage, "args": s.args }))
                .collect::<Vec<_>>(),
        });
        CompiledPlan::from_erased_chain(self.name.clone(), recipe_args, chain).map_err(|e| {
            DeclarativeError::Plan {
                recipe: self.name.clone(),
                detail: e.to_string(),
            }
        })
    }
}

/// F4: the user declarative-recipes directory — `$BLUT_USER_RECIPES_DIR`
/// else `~/.config/blut/recipes`.
pub fn user_recipes_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("BLUT_USER_RECIPES_DIR") {
        return Some(PathBuf::from(d));
    }
    dirs::config_dir().map(|d| d.join("blut").join("recipes"))
}

/// F4: discover `*.toml` declarative recipes in the user dir, returning
/// `(name, path)` for each that PARSES (a malformed file is skipped with a
/// warning, never fatal). A missing dir yields an empty list. Sorted by
/// name for stable display.
pub fn scan_user_recipes() -> Vec<(String, PathBuf)> {
    match user_recipes_dir() {
        Some(dir) => scan_user_recipes_at(&dir),
        None => Vec::new(),
    }
}

/// Path-injectable [`scan_user_recipes`] (the default scans
/// [`user_recipes_dir`]). Lets tests drive a tempdir without mutating the
/// process environment.
pub fn scan_user_recipes_at(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        match DeclarativeRecipe::load(&path) {
            Ok(r) => out.push((r.name, path)),
            Err(e) => tracing::warn!("skipping declarative recipe {}: {e}", path.display()),
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_recipe() {
        let toml = r#"
            name = "demo"
            backend = "lamquant"
            [[stages]]
            stage = "a"
            args = { x = 1, y = "z" }
            [[stages]]
            stage = "b"
        "#;
        let r = DeclarativeRecipe::parse(toml, "demo.toml").unwrap();
        assert_eq!(r.name, "demo");
        assert_eq!(r.backend.as_deref(), Some("lamquant"));
        assert_eq!(r.stages.len(), 2);
        assert_eq!(r.stages[0].stage, "a");
        assert_eq!(r.stages[0].args["x"], serde_json::json!(1));
        assert_eq!(r.stages[0].args["y"], serde_json::json!("z"));
        // omitted args default to null.
        assert!(r.stages[1].args.is_null());
    }

    #[test]
    fn unknown_stage_is_a_clear_error() {
        // An empty registry resolves no stages → UnknownStage names it.
        let toml = "name = \"x\"\n[[stages]]\nstage = \"no_such_stage\"\n";
        let r = DeclarativeRecipe::parse(toml, "x.toml").unwrap();
        let reg = Registry::new();
        // (CompiledPlan has no Debug, so don't format the Ok arm.)
        match r.compile(&reg) {
            Err(DeclarativeError::UnknownStage { stage, .. }) => assert_eq!(stage, "no_such_stage"),
            Err(e) => panic!("expected UnknownStage, got a different error: {e}"),
            Ok(_) => panic!("expected UnknownStage, got Ok"),
        }
    }

    #[test]
    fn no_stages_is_rejected() {
        // A recipe with an empty stages array fails to compile.
        let toml = "name = \"x\"\nstages = []\n";
        let r = DeclarativeRecipe::parse(toml, "x.toml").unwrap();
        assert!(matches!(
            r.compile(&Registry::new()),
            Err(DeclarativeError::NoStages { .. })
        ));
    }

    #[test]
    fn scan_missing_dir_is_empty() {
        // A missing dir → empty, never an error. Path-injected so the test
        // mutates no process environment (no unsafe set_var).
        let td = tempfile::tempdir().unwrap();
        assert!(scan_user_recipes_at(&td.path().join("does-not-exist")).is_empty());
    }

    #[test]
    fn scan_finds_parseable_toml_and_skips_garbage() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(
            td.path().join("good.toml"),
            "name = \"good\"\n[[stages]]\nstage = \"x\"\n",
        )
        .unwrap();
        std::fs::write(td.path().join("garbage.toml"), "not = [valid toml").unwrap();
        std::fs::write(td.path().join("ignored.txt"), "name = \"nope\"").unwrap();
        let found = scan_user_recipes_at(td.path());
        // Only the parseable .toml surfaces; garbage is skipped, .txt ignored.
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, "good");
    }
}
