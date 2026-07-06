// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut errors` — ErrorDomain catalogs + per-job failure breakdown (ADR 0072 A4).
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

use super::*;

#[derive(Subcommand, Debug)]
pub(super) enum ErrorsCommand {
    /// List every registered `ErrorDomain` catalog (ADR 0072 A4). A
    /// cookbook registers one via `register_error_domain!`; none does
    /// today (a later cookbook-side workflow adds the first), so this
    /// prints "no error domains registered" instead of assuming a
    /// catalog exists.
    List {
        /// Emit the catalog as JSON (for scripts/agents).
        #[arg(long)]
        json: bool,
    },
    /// Show a job's terminal failure: the origin/course/recipe/stage/
    /// ingredient breakdown, read from the same status.jsonl `blut
    /// results` reads lineage from. Prints "no failure recorded" for a
    /// job that succeeded or hasn't run/failed yet.
    Show {
        /// Job id (prefix ok).
        job: String,
        /// Emit machine-readable JSON instead of a human summary.
        #[arg(long)]
        json: bool,
    },
}

/// `blut errors list|show` (ADR 0072 A4) dispatch — mirrors `run_recipe`'s
/// shape over `RecipeCommand`.
pub(super) fn run_errors(reg: &crate::framework::Registry, cmd: ErrorsCommand) -> Result<()> {
    match cmd {
        ErrorsCommand::List { json } => run_errors_list(reg, json),
        ErrorsCommand::Show { job, json } => run_errors_show(&job, json),
    }
}

/// `blut errors list [--json]`: every registered `ErrorDomain` catalog,
/// unioned across the cookbooks in `reg` (mirrors `RecipeCommand::List`'s
/// composition over `reg.all()`). Cookbooks registering one is a separate,
/// later workflow — no implementor ships today — so an empty union prints
/// "no error domains registered" rather than assuming one exists.
pub(super) fn run_errors_list(reg: &crate::framework::Registry, json: bool) -> Result<()> {
    let mut domains: Vec<&'static crate::framework::error_domain::ErrorDomainDef> =
        reg.all_error_domains().collect();
    domains.sort_by(|a, b| a.name.cmp(b.name));

    if json {
        let arr: Vec<_> = domains
            .iter()
            .map(|d| {
                serde_json::json!({
                    "name": d.name,
                    "codes": d.codes.iter().map(|(code, description)| {
                        serde_json::json!({ "code": code, "description": description })
                    }).collect::<Vec<_>>(),
                })
            })
            .collect();
        emit_json(&arr)?;
        return Ok(());
    }

    if domains.is_empty() {
        println!("no error domains registered");
        return Ok(());
    }
    for d in domains {
        println!(
            "{}  ({} code{})",
            d.name,
            d.codes.len(),
            if d.codes.len() == 1 { "" } else { "s" }
        );
        for (code, description) in d.codes {
            println!("  {code:<24} {description}");
        }
    }
    Ok(())
}

/// `blut errors show <job> [--json]`: a job's terminal failure — the
/// origin/course/recipe/stage/ingredient breakdown, extracted from
/// `status.jsonl`'s `StageFailed` event (the SAME source `blut results`
/// reads lineage from), plus provenance (recipe, outcome) from the lineage
/// DB `blut results` also opens. "no failure recorded" for a job that
/// succeeded or hasn't run/failed yet — never assumes a failure exists.
/// If any `status.jsonl` line failed to parse during the scan (a torn/
/// truncated write — disk-full, or the orchestrator dying mid-flush —
/// could have clobbered exactly the terminal `StageFailed` line), that is
/// surfaced distinctly instead of a flatly confident "no failure".
pub(super) fn run_errors_show(job: &str, json: bool) -> Result<()> {
    let job_id = crate::jobs::resolve_job_id(job).map_err(|e| anyhow!("{e}"))?;
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("open lineage.db: {e}"))?;
    let run = db.get_run(&job_id).map_err(|e| anyhow!("{e}"))?;
    let lookup = crate::framework::lineage::job_failure(&job_id).map_err(|e| anyhow!("{e}"))?;

    if json {
        let out = serde_json::json!({
            "job": job_id,
            "recipe": run.as_ref().map(|r| r.recipe.clone()),
            "outcome": run.as_ref().and_then(|r| r.outcome.clone()),
            "failure": lookup.failure,
            "parse_errors": lookup.parse_errors,
        });
        emit_json(&out)?;
        return Ok(());
    }

    println!("job     : {job_id}");
    if let Some(r) = &run {
        println!("recipe  : {}", r.recipe);
        println!("outcome : {}", r.outcome.as_deref().unwrap_or("?"));
    }

    let Some(jf) = lookup.failure else {
        if lookup.parse_errors > 0 {
            println!(
                "\nno StageFailed event found, but {} status.jsonl line(s) could not be parsed \
                 — the record may be incomplete (a torn/truncated write?). This is NOT a \
                 confirmed clean success.",
                lookup.parse_errors
            );
        } else {
            println!("\nno failure recorded (job succeeded, or hasn't run/failed yet)");
        }
        return Ok(());
    };

    if lookup.parse_errors > 0 {
        println!(
            "\nnote: {} other status.jsonl line(s) could not be parsed during this scan — \
             earlier lineage detail may be incomplete.",
            lookup.parse_errors
        );
    }
    println!(
        "\nterminal failure @ stage '{}' (node {})",
        jf.stage, jf.node_idx
    );
    match &jf.failure {
        Some(f) => {
            println!("  code       : {}", f.code);
            println!("  domain     : {}", f.domain);
            println!("  severity   : {}", f.severity);
            // `FaultOrigin` has no `Display` impl (it's an A1 type; adding one
            // is out of this command's scope) — `{:?}` on its PascalCase
            // variants (Engine/Cookbook/External) already reads fine.
            println!("  origin     : {:?}", f.origin);
            println!(
                "  course     : {}",
                f.course.as_deref().unwrap_or("(unknown)")
            );
            println!(
                "  recipe     : {}",
                f.recipe.as_deref().unwrap_or("(unknown)")
            );
            println!(
                "  stage      : {}",
                f.stage.as_deref().unwrap_or(jf.stage.as_str())
            );
            println!(
                "  ingredient : {}",
                f.ingredient.as_deref().unwrap_or("(unknown)")
            );
            if !f.context.is_empty() {
                println!("  context    :");
                for (k, v) in &f.context {
                    println!("    {k} = {v}");
                }
            }
            println!("  message    : {}", f.message);
        }
        None => {
            println!("  (no structured StageFailure attached — raw error only)");
            println!("  error      : {}", jf.error);
        }
    }
    Ok(())
}

#[cfg(test)]
mod errors_cli_tests {
    //! ADR 0072 A4: `blut errors list` / `blut errors show <job>` — clap
    //! parsing + the empty-registry "no error domains registered" case
    //! (no cookbook has called `register_error_domain!` yet). The
    //! structured origin/course/recipe/stage/ingredient breakdown itself
    //! is covered at its data source in
    //! `framework::lineage::tests::job_failure_surfaces_full_structured_breakdown`
    //! — `run_errors_show` is a thin formatter over that.
    use super::{Cli, Command, ErrorsCommand};
    use clap::Parser;

    fn errors_of(argv: &[&str]) -> ErrorsCommand {
        match Cli::try_parse_from(argv).expect("parse").command {
            Some(Command::Errors { cmd }) => cmd,
            other => panic!("expected errors, got {other:?}"),
        }
    }

    #[test]
    fn list_parses_default_json_false() {
        match errors_of(&["blut", "errors", "list"]) {
            ErrorsCommand::List { json } => assert!(!json),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn list_json_flag_parses() {
        match errors_of(&["blut", "errors", "list", "--json"]) {
            ErrorsCommand::List { json } => assert!(json),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn show_requires_job_and_parses_json_flag() {
        // No job id ⇒ parse error.
        assert!(Cli::try_parse_from(["blut", "errors", "show"]).is_err());
        match errors_of(&["blut", "errors", "show", "20260702-000000-abcdef", "--json"]) {
            ErrorsCommand::Show { job, json } => {
                assert_eq!(job, "20260702-000000-abcdef");
                assert!(json);
            }
            other => panic!("got {other:?}"),
        }
    }

    /// The chicken-and-egg case this command was explicitly designed for:
    /// zero cookbooks have called `register_error_domain!` yet, so the
    /// composed registry is empty. `run_errors_list` must print "no error
    /// domains registered" and return `Ok(())`, never panic or error —
    /// in both text and `--json` modes.
    #[test]
    fn errors_list_on_empty_registry_prints_gracefully_text() {
        let reg = crate::framework::Registry::new();
        assert!(super::run_errors_list(&reg, false).is_ok());
    }

    #[test]
    fn errors_list_on_empty_registry_prints_gracefully_json() {
        let reg = crate::framework::Registry::new();
        // `--json` on an empty catalog must still serialize (an empty
        // array), not error.
        assert!(super::run_errors_list(&reg, true).is_ok());
    }
}
