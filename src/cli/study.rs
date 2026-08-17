// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut study` — ADR 0109 multi-objective studies.
//!
//! A study IS an HPO run whose deliverable is a Pareto front rather than a
//! leaderboard. The trials execute through exactly the same path — same plan
//! build, same broker admission, same DAG cache, same early-stop — so this
//! module deliberately owns none of that. It translates a declared
//! [`StudySpec`] into the HPO command that already exists, then reads the trial
//! stream back with every declared objective instead of one.
//!
//! Reusing the execution path rather than cloning it is the point: ADR 0092
//! invariant 4 is that CLI, TUI, local and remote resolve operations through
//! ONE application module, and a second launcher here would drift from
//! `blut hpo run` on retries, admission and cancellation semantics — the exact
//! failure that invariant exists to prevent.
//!
//! **The scheduler still needs a scalar.** ASHA and friends rank trials to
//! decide what to early-stop, and there is no total order over a vector. The
//! FIRST declared objective is handed to the scheduler for that purpose only;
//! it never decides the deliverable, which is computed from all objectives at
//! the end. This is stated in the run banner so nobody infers that objective
//! one is "the" objective.

use anyhow::{Context, Result, anyhow};
use clap::Subcommand;

use crate::hpo::pareto::Direction;
use crate::hpo::study::{StudySpec, reconstruct_study};

#[derive(Subcommand, Debug)]
pub(super) enum StudyCommand {
    /// Run a multi-objective study: sample trials, execute them through the
    /// ordinary HPO path, and write the Pareto front to `pareto.json`.
    Run {
        /// Study spec (TOML): objectives, search space, sampler.
        spec: String,
        /// Base args as inline JSON (the fixed part; search dims overlay it).
        #[arg(long, default_value = "{}")]
        args: String,
        /// Trial budget.
        #[arg(long, default_value_t = 24)]
        max_trials: u32,
        /// Tenant whose quota admits each trial.
        #[arg(long, default_value = "default")]
        tenant: String,
        /// Launcher for trial execution (mirrors `blut hpo run`).
        #[arg(long, default_value = "local")]
        launcher: String,
    },
    /// Re-derive `pareto.json` from a finished job without re-running it.
    /// Useful after amending a spec's objectives, and it is how a study that
    /// crashed after its trials completed is salvaged.
    Report {
        /// Job id (defaults to the most recent HPO job).
        job: Option<String>,
        /// The study spec whose objectives are read back.
        #[arg(long)]
        spec: String,
    },
}

/// Map a study's sampler name onto the HPO `--algo` that carries it.
///
/// The multi-objective samplers are not reachable through `--algo` yet, so a
/// study naming one is REFUSED rather than silently downgraded to `random`: a
/// study that quietly searched with the wrong sampler would still produce a
/// plausible front, and nobody would know the model was never used.
fn algo_for(sampler: &str) -> Result<&'static str> {
    match sampler.trim().to_ascii_lowercase().as_str() {
        "random" => Ok("random"),
        "tpe" => Ok("tpe"),
        "asha" => Ok("asha"),
        other => Err(anyhow!(
            "sampler '{other}' is declared but not yet reachable from `blut study run`: \
             the search loop (hpo::study::StudyLedger) and the samplers \
             (hpo::surrogate, hpo::gp) are implemented and tested, but the \
             search-driver node that feeds them per-round results is not wired \
             (ADR 0109). Use sampler = \"random\" | \"tpe\" | \"asha\" until it is, \
             rather than running a study that silently searched with a different \
             algorithm than it declared."
        )),
    }
}

pub(super) async fn run_study(reg: &crate::framework::Registry, cmd: StudyCommand) -> Result<()> {
    match cmd {
        StudyCommand::Report { job, spec } => {
            let spec = load_spec(&spec)?;
            let (job_id, manifest) = super::hpo::resolve_hpo_job(job)?;
            emit_report(&spec, &job_id, &manifest)
        }
        StudyCommand::Run {
            spec,
            args,
            max_trials,
            tenant,
            launcher,
        } => {
            let spec_path = spec;
            let spec = load_spec(&spec_path)?;
            let algo = algo_for(&spec.sampler)?;

            // The scheduler ranks on one metric; the deliverable does not.
            let primary = &spec.objectives[0];
            let mode = match primary.direction {
                Direction::Maximize => "max",
                Direction::Minimize => "min",
            };
            eprintln!(
                "study {}: {} objectives ({}), sampler={} → algo={}, {max_trials} trials",
                spec.recipe,
                spec.objectives.len(),
                spec.objective_names().join(", "),
                spec.sampler,
                algo
            );
            eprintln!(
                "  scheduler ranks on '{}' ({mode}) for early-stop ONLY; the front is \
                 computed from all {} objectives",
                primary.name,
                spec.objectives.len()
            );

            // The search space travels as a temp YAML because that is the
            // interface `hpo run` already takes; re-encoding it here keeps the
            // study spec the single place a space is declared.
            let space_yaml = serde_yaml::to_string(&spec.space)
                .map_err(|e| anyhow!("re-encode search space: {e}"))?;
            let tmp =
                std::env::temp_dir().join(format!("blut-study-space-{}.yaml", std::process::id()));
            std::fs::write(&tmp, space_yaml)
                .with_context(|| format!("write temp search space {}", tmp.display()))?;

            let hpo_cmd = super::hpo::HpoCommand::Run {
                name: spec.recipe.clone(),
                args,
                space: Some(tmp.display().to_string()),
                param: Vec::new(),
                algo: algo.to_string(),
                metric: primary.metric.clone(),
                mode: mode.to_string(),
                max_trials,
                seed: spec.seed,
                metric_budget_key: "epoch".to_string(),
                eta: 3,
                min_budget: 1,
                max_budget: 0,
                grace: 1,
                percentile: 50,
                shared_cache: false,
                launcher,
                sync_io: false,
                tenant,
                experiment: None,
            };

            let outcome = super::hpo::run_hpo(reg, hpo_cmd).await;
            let _ = std::fs::remove_file(&tmp);
            let job_id = outcome?.ok_or_else(|| anyhow!("study run produced no job id"))?;

            let (_, manifest) = super::hpo::resolve_hpo_job(Some(job_id.clone()))?;
            emit_report(&spec, &job_id, &manifest)
        }
    }
}

fn load_spec(path: &str) -> Result<StudySpec> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read study spec {path}"))?;
    StudySpec::from_toml(&text).map_err(|e| anyhow!("{path}: {e}"))
}

/// Read the trial stream back, compute the front, and write `pareto.json` into
/// the job directory beside the other artifacts.
fn emit_report(
    spec: &StudySpec,
    job_id: &str,
    manifest: &crate::hpo::results::HpoManifest,
) -> Result<()> {
    let lines = crate::jobs::read_status_lines(job_id)
        .with_context(|| format!("read status stream for {job_id}"))?;
    let ledger = reconstruct_study(spec, manifest, &lines);
    let report = ledger.report(spec);

    let dir = crate::jobs::job_dir_path(job_id)?;
    let path = dir.join("pareto.json");
    let body = serde_json::to_string_pretty(&report)
        .map_err(|e| anyhow!("serialize pareto report: {e}"))?;
    // Same temp-then-rename the manifest uses: a reader must never observe a
    // half-written front.
    let tmp = dir.join("pareto.json.tmp");
    std::fs::write(&tmp, &body).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("rename into {}", path.display()))?;

    let measured = ledger
        .trials()
        .iter()
        .filter(|t| t.status.is_measured())
        .count();
    eprintln!(
        "study done: {} non-dominated of {measured} measured ({} trials total) — {}",
        report.front.len(),
        ledger.len(),
        path.display()
    );
    if !report.unmeasured_trials.is_empty() {
        // Named, not hidden: a study whose trials mostly failed must not read
        // as a clean small front.
        eprintln!(
            "  {} trial(s) produced no point: {:?}",
            report.unmeasured_trials.len(),
            report.unmeasured_trials
        );
    }
    if report.front.len() < 2 {
        eprintln!(
            "  NOTE: ADR 0109 expects a front of at least 2 points. A smaller front \
             usually means the objectives do not actually compete, or too few trials \
             produced a complete measurement."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    #[test]
    fn study_run_parses_with_the_adr_gate_s_own_argv() {
        // Exactly the invocation ADR 0109's gate names.
        let cli = Cli::try_parse_from([
            "blut",
            "study",
            "run",
            "tests/fixtures/study_multiobj.toml",
            "--max-trials",
            "24",
        ])
        .expect("the gate's argv must parse");
        match cli.command {
            Some(Command::Study {
                cmd: StudyCommand::Run {
                    spec, max_trials, ..
                },
            }) => {
                assert_eq!(spec, "tests/fixtures/study_multiobj.toml");
                assert_eq!(max_trials, 24);
            }
            other => panic!("expected study run, got {other:?}"),
        }
    }

    #[test]
    fn study_report_parses_and_defaults_its_job_to_the_latest() {
        let cli =
            Cli::try_parse_from(["blut", "study", "report", "--spec", "s.toml"]).expect("parse");
        match cli.command {
            Some(Command::Study {
                cmd: StudyCommand::Report { job, spec },
            }) => {
                assert_eq!(spec, "s.toml");
                assert!(job.is_none(), "omitted job means the most recent");
            }
            other => panic!("expected study report, got {other:?}"),
        }
    }

    #[test]
    fn an_unreachable_sampler_is_refused_rather_than_downgraded() {
        // The model-based samplers exist but their driver is not wired. A study
        // declaring one must FAIL, not quietly run random search and hand back
        // a plausible-looking front.
        for s in ["gp", "mvtpe", "grid"] {
            let err = algo_for(s).unwrap_err().to_string();
            assert!(err.contains("not yet reachable"), "{s}: {err}");
            assert!(err.contains("ADR 0109"), "{s}: should cite the ADR");
        }
    }

    #[test]
    fn the_reachable_samplers_map_to_their_hpo_algo() {
        assert_eq!(algo_for("random").unwrap(), "random");
        assert_eq!(algo_for("TPE").unwrap(), "tpe", "case-insensitive");
        assert_eq!(algo_for(" asha ").unwrap(), "asha", "trimmed");
    }

    #[test]
    fn the_gate_fixture_is_refused_today_and_says_why() {
        // The shipped fixture declares mvtpe. Until the driver is wired,
        // `blut study run` on it must refuse with a message that explains the
        // gap rather than producing a front from the wrong search.
        let spec =
            StudySpec::from_toml(include_str!("../../tests/fixtures/study_multiobj.toml")).unwrap();
        assert_eq!(spec.sampler, "mvtpe");
        assert!(algo_for(&spec.sampler).is_err());
    }
}
