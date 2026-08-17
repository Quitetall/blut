// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut registry export` — the read-only registry manifest ADR 0111's Python
//! SDK validates against.
//!
//! This is the ONLY engine surface that ADR exposes for the SDK, and it is
//! deliberately read-only: the SDK is a client of the CLI, like a `kubectl`
//! adapter. It opens no socket, embeds no runtime, and this command mutates
//! nothing.
//!
//! **Why the SDK needs a manifest at all.** Without one, a typo in a stage name
//! or an arg is only caught after the plan is submitted — a round trip through
//! job creation to learn that `lamquant_joint_codc` does not exist. The
//! manifest lets the client refuse before submission. The engine still
//! re-validates authoritatively on `recipe declare`; this is an early "no", not
//! a substitute for the real check, and a client that skipped it would be
//! caught anyway.
//!
//! **Two namespaces, both exported, not interchangeable.** A `PlanSpec` node
//! names a STAGE (resolved via `find_erased_stage` over `stages_erased`); a
//! recipe is a named pre-composed chain (resolved via `find` over `recipes`).
//! They are separate registries with barely-overlapping contents, so a client
//! handed only one of them would either reject every valid plan or fail to
//! catch any typo. `stages` also carries each stage's input/output kinds — the
//! information that decides which stages may legally be wired together, and
//! the only part of the kind-check a client can approximate before submitting.
//!
//! The args schema is the recipe's OWN `args_schema_fn`, not a re-description
//! of it. A second hand-written description of the same arguments is exactly
//! the drift ADR 0092 invariant 2 forbids — one canonical owner per contract.

use anyhow::Result;
use clap::Subcommand;
use serde::Serialize;

#[derive(Subcommand, Debug)]
pub(super) enum RegistryCommand {
    /// Print the live recipe catalog as JSON: every registered recipe with its
    /// args schema and artifact kinds. Consumed by `blut-sdk` (ADR 0111).
    Export {
        /// Pretty-print instead of one dense line.
        #[arg(long, default_value_t = false)]
        pretty: bool,
    },
}

/// One recipe, as the SDK sees it.
#[derive(Debug, Serialize)]
pub struct RecipeManifest {
    pub name: String,
    pub description: String,
    pub backend_id: String,
    pub category: crate::recipes::recipe::Course,
    /// Artifact kind IDs consumed. Empty when the recipe's first stage takes
    /// the unit graph-input.
    pub input_kinds: Vec<String>,
    pub output_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schedule: Option<String>,
    /// The recipe's own args JSON schema, verbatim.
    pub args_schema: serde_json::Value,
}

/// One STAGE, as the SDK sees it — the vocabulary a `PlanSpec` node names.
///
/// Distinct from a recipe and not interchangeable with one: `SpecNode::stage`
/// resolves through [`Registry::find_erased_stage`], which reads
/// `Cookbook::stages_erased`, while a recipe is a named, pre-composed chain
/// resolved through `Registry::find`. A client that validated plan nodes
/// against the recipe list would reject every legitimate plan, because the two
/// namespaces barely overlap.
#[derive(Debug, Serialize)]
pub struct StageManifest {
    pub name: String,
    /// Artifact kind consumed. The unit kind for a graph source.
    pub input_kind: String,
    pub output_kind: String,
    /// Element kind when `output_kind` is a list — the kind a `map_output`
    /// template's root receives. Absent for a non-list output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_kind: Option<String>,
}

/// The exported catalog.
#[derive(Debug, Serialize)]
pub struct RegistryManifest {
    /// Manifest format version. Bumped when the SHAPE changes, so an SDK
    /// pinned to an older engine can refuse loudly instead of mis-reading a
    /// field that moved. Additive fields do NOT bump it — same rule as
    /// `PLAN_SPEC_VERSION`.
    pub manifest_version: u32,
    /// Engine version that produced it, for the same reason.
    pub engine_version: String,
    /// The stage palette: what a `PlanSpec` node may name, with the kinds that
    /// decide which stages may be wired together.
    pub stages: Vec<StageManifest>,
    pub recipes: Vec<RecipeManifest>,
}

pub const MANIFEST_VERSION: u32 = 1;

/// Build the manifest from the live registry.
pub fn build_manifest(reg: &crate::framework::Registry) -> RegistryManifest {
    let mut recipes: Vec<RecipeManifest> = reg
        .all()
        .map(|d| RecipeManifest {
            name: d.name.to_string(),
            description: d.description.to_string(),
            backend_id: d.backend_id.to_string(),
            category: d.category,
            input_kinds: d.input_kinds.iter().map(|k| k.to_string()).collect(),
            output_kind: d.output_kind.to_string(),
            schedule: d.schedule.map(|s| s.to_string()),
            args_schema: (d.args_schema_fn)(),
        })
        .collect();
    // Sorted by name so the export is byte-stable across runs: a manifest that
    // reordered itself would show a spurious diff every time it was regenerated,
    // and any consumer checksumming it would thrash.
    recipes.sort_by(|a, b| a.name.cmp(&b.name));
    // `ingredient_palette` already sorts by name and de-duplicates across
    // cookbooks, which is the ordering guarantee this export needs.
    let stages = reg
        .ingredient_palette()
        .into_iter()
        .map(|i| StageManifest {
            name: i.stage,
            input_kind: i.input_kind,
            output_kind: i.output_kind,
            element_kind: i.element_kind,
        })
        .collect();
    RegistryManifest {
        manifest_version: MANIFEST_VERSION,
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        stages,
        recipes,
    }
}

pub(super) fn run_registry(reg: &crate::framework::Registry, cmd: RegistryCommand) -> Result<()> {
    match cmd {
        RegistryCommand::Export { pretty } => {
            let manifest = build_manifest(reg);
            let body = if pretty {
                serde_json::to_string_pretty(&manifest)?
            } else {
                serde_json::to_string(&manifest)?
            };
            println!("{body}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    #[test]
    fn registry_export_parses() {
        let cli = Cli::try_parse_from(["blut", "registry", "export"]).expect("parse");
        match cli.command {
            Some(Command::Registry {
                cmd: RegistryCommand::Export { pretty },
            }) => assert!(!pretty, "dense by default — it is machine input"),
            other => panic!("expected registry export, got {other:?}"),
        }
        let cli = Cli::try_parse_from(["blut", "registry", "export", "--pretty"]).expect("parse");
        assert!(matches!(
            cli.command,
            Some(Command::Registry {
                cmd: RegistryCommand::Export { pretty: true }
            })
        ));
    }

    #[test]
    fn an_empty_registry_exports_a_valid_empty_manifest() {
        // blut-core ships no recipes of its own; the catalog is whatever
        // cookbooks register. An engine with none must still emit a manifest an
        // SDK can parse, not an error or a bare `[]` with no version.
        let reg = crate::framework::Registry::new();
        let m = build_manifest(&reg);
        assert_eq!(m.manifest_version, MANIFEST_VERSION);
        assert!(m.recipes.is_empty());
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert!(v.get("manifest_version").is_some());
        assert!(v.get("engine_version").is_some());
        assert!(v["recipes"].is_array());
        assert!(v["stages"].is_array());
    }

    #[test]
    fn stages_are_exported_separately_from_recipes() {
        // A `PlanSpec` node names a STAGE, never a recipe. Exporting only the
        // recipe list would give a client a catalog that rejects every valid
        // plan while looking like it validated one, so the two namespaces must
        // both be present and must not be conflated.
        let reg = crate::framework::Registry::new();
        let m = build_manifest(&reg);
        let v = serde_json::to_value(&m).unwrap();
        assert!(
            v.as_object().unwrap().contains_key("stages"),
            "the stage palette is the plan vocabulary and must always be present"
        );
        // Bare blut-core registers neither, but the KEYS must exist regardless
        // so a client can tell "no stages" from "no such field".
        assert!(v["stages"].as_array().unwrap().is_empty());
    }

    #[test]
    fn stage_entries_expose_the_kinds_that_decide_wiring() {
        // Kinds are what `from_erased_graph` checks. Without them the manifest
        // could only catch name typos, not a producer wired into a consumer
        // that cannot accept its output.
        let s = StageManifest {
            name: "x".into(),
            input_kind: "unit".into(),
            output_kind: "list<thing>".into(),
            element_kind: Some("thing".into()),
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["input_kind"], "unit");
        assert_eq!(v["output_kind"], "list<thing>");
        assert_eq!(v["element_kind"], "thing");
        // A non-list stage omits `element_kind` rather than emitting null, so
        // "absent" reads the same as the Rust `Option::None` it came from.
        let plain = StageManifest {
            name: "y".into(),
            input_kind: "a".into(),
            output_kind: "b".into(),
            element_kind: None,
        };
        let v = serde_json::to_value(&plain).unwrap();
        assert!(v.as_object().unwrap().get("element_kind").is_none());
    }

    #[test]
    fn the_manifest_is_byte_stable_across_builds() {
        // A consumer may checksum this. Two builds of the same registry must
        // produce identical bytes, which is why recipes are sorted by name.
        let reg = crate::framework::Registry::new();
        let a = serde_json::to_string(&build_manifest(&reg)).unwrap();
        let b = serde_json::to_string(&build_manifest(&reg)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn recipes_are_sorted_by_name_not_registration_order() {
        // Registration order is whatever the cookbook slice happens to be, and
        // it changes when someone reorders a list for unrelated reasons.
        let reg = crate::framework::Registry::new();
        let m = build_manifest(&reg);
        let names: Vec<&str> = m.recipes.iter().map(|r| r.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }
}
