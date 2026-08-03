// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut ledger` — read and curate the append-only run ledger (ADR 0154).
//!
//! Every mutating subcommand here APPENDS. Nothing rewrites or deletes a line,
//! because the ledger is the record of what happened, not a view of what is
//! currently true. `promote` appends a promotion; `gc` appends tombstones. If a
//! future subcommand needs to remove a line, it is the wrong subcommand.

use anyhow::{Result, anyhow};
use clap::Subcommand;

use crate::run_ledger::{Arm, Record, RunLedger, Tier, Verdict, compare};
use crate::tenant::Tenant;

#[derive(Subcommand, Debug)]
pub(super) enum LedgerCommand {
    /// List runs and their effective tier.
    List {
        /// Show only this tier (`scratch` | `recorded` | `canonical`).
        #[arg(long)]
        tier: Option<String>,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show every record for one run, in write order.
    Show {
        /// Run uid (or unique prefix).
        uid: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Promote a run to `canonical`. Appends; never rewrites.
    Promote {
        /// Run uid (or unique prefix).
        uid: String,
        /// Why this run is canonical. Required — a promotion without a stated
        /// reason is not a judgement, it is a click.
        #[arg(long)]
        reason: String,
        /// Who is promoting. Defaults to $USER.
        #[arg(long)]
        by: Option<String>,
    },
    /// Collect `scratch` runs, leaving a tombstone per run so citations to a
    /// collected run still resolve. Refuses to touch any other tier.
    Gc {
        /// Actually append the tombstones. Without this, only reports.
        #[arg(long)]
        commit: bool,
    },
    /// Structural checks over the ledger.
    /// Compare two configurations on a metric, and refuse to overclaim.
    ///
    /// Groups runs into ARMS by (experiment, config_fingerprint) and asks
    /// whether a difference between two arms is larger than the spread you get
    /// from rerunning the SAME arm with a different seed. With one run per arm
    /// there is no such spread, and the answer is INDETERMINATE regardless of
    /// how large the gap looks — which is the case this exists for.
    Compare {
        /// Experiment id both arms belong to, e.g. `E1`.
        experiment: String,
        /// Baseline arm: a config_fingerprint (or unique prefix).
        baseline: String,
        /// Candidate arm: a config_fingerprint (or unique prefix).
        candidate: String,
        /// Metric to compare, as recorded in `run_ended.metrics`.
        #[arg(long, default_value = "best_val_r")]
        metric: String,
        /// Two-sided significance threshold.
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    Verify {
        /// Also require every TRUTH_LEDGER §2 citation to resolve to a run.
        #[arg(long)]
        require_ledger_rows_resolve: bool,
        /// Repository root containing docs/TRUTH_LEDGER.md. Defaults to
        /// $LAMQUANT_META_ROOT, then the nearest ancestor of the current
        /// directory containing that file.
        #[arg(long, value_name = "PATH")]
        truth_ledger_root: Option<std::path::PathBuf>,
    },
}

fn open() -> Result<RunLedger> {
    let path = RunLedger::default_path().map_err(|e| anyhow!("{e}"))?;
    Ok(RunLedger::at(path))
}

fn parse_tier(s: &str) -> Result<Tier> {
    match s {
        "scratch" => Ok(Tier::Scratch),
        "recorded" => Ok(Tier::Recorded),
        "canonical" => Ok(Tier::Canonical),
        other => Err(anyhow!(
            "unknown tier {other:?}; expected scratch | recorded | canonical"
        )),
    }
}

fn resolve_uid(ledger: &RunLedger, prefix: &str) -> Result<String> {
    let tiers = ledger.tiers().map_err(|e| anyhow!("{e}"))?;
    let hits: Vec<&String> = tiers.keys().filter(|k| k.starts_with(prefix)).collect();
    match hits.len() {
        0 => Err(anyhow!("no run matches {prefix:?}")),
        1 => Ok(hits[0].clone()),
        // Never guess which run the operator meant — a wrong promotion is a
        // false provenance claim, and those are the expensive kind.
        n => Err(anyhow!(
            "{prefix:?} is ambiguous: {n} runs match ({}…)",
            hits.iter()
                .take(3)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub(super) fn run_ledger_cmd(cmd: LedgerCommand) -> Result<()> {
    let ledger = open()?;
    match cmd {
        LedgerCommand::List { tier, json } => {
            let want = tier.as_deref().map(parse_tier).transpose()?;
            let tiers = ledger.tiers().map_err(|e| anyhow!("{e}"))?;
            let rows: Vec<(&String, &Tier)> = tiers
                .iter()
                .filter(|(_, t)| want.is_none_or(|w| **t == w))
                .collect();
            if json {
                let out: Vec<_> = rows
                    .iter()
                    .map(|(uid, t)| serde_json::json!({ "run_uid": uid, "tier": t }))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else if rows.is_empty() {
                println!("(no runs)");
            } else {
                for (uid, t) in rows {
                    println!("{uid}  {t:?}");
                }
            }
        }
        LedgerCommand::Show { uid, json } => {
            let uid = resolve_uid(&ledger, &uid)?;
            let read = ledger.read().map_err(|e| anyhow!("{e}"))?;
            let recs: Vec<&Record> = read.records.iter().filter(|r| r.run_uid() == uid).collect();
            if json {
                println!("{}", serde_json::to_string_pretty(&recs)?);
            } else {
                for r in recs {
                    println!("{r:?}");
                }
            }
        }
        LedgerCommand::Promote { uid, reason, by } => {
            if reason.trim().is_empty() {
                return Err(anyhow!("--reason must not be blank"));
            }
            let uid = resolve_uid(&ledger, &uid)?;
            let by = by
                .or_else(|| std::env::var("USER").ok())
                .ok_or_else(|| anyhow!("cannot determine promoter; pass --by"))?;
            append_promotion(&ledger, &uid, reason, by)?;
            println!("promoted {uid} -> canonical");
        }
        LedgerCommand::Gc { commit } => {
            let tiers = ledger.tiers().map_err(|e| anyhow!("{e}"))?;
            let read = ledger.read().map_err(|e| anyhow!("{e}"))?;
            // Only Scratch is ever collectable, and a run already tombstoned is
            // not collected twice.
            let already: std::collections::BTreeSet<&str> = read
                .records
                .iter()
                .filter_map(|r| match r {
                    Record::Cleared { run_uid, .. } => Some(run_uid.as_str()),
                    _ => None,
                })
                .collect();
            let summaries = summarize_runs(&read.records);
            let victims: Vec<&String> = tiers
                .iter()
                .filter(|(uid, t)| {
                    t.is_clearable()
                        && !already.contains(uid.as_str())
                        && summaries
                            .get(uid.as_str())
                            .is_some_and(RunSummary::can_clear)
                })
                .map(|(uid, _)| uid)
                .collect();
            if victims.is_empty() {
                println!("nothing to collect");
                return Ok(());
            }
            if !commit {
                println!("would collect {} scratch run(s):", victims.len());
                for uid in &victims {
                    println!("  {uid}");
                }
                println!("re-run with --commit to append tombstones");
                return Ok(());
            }
            for uid in victims {
                let recipe = read
                    .records
                    .iter()
                    .find_map(|r| match r {
                        Record::RunStarted {
                            run_uid, recipe, ..
                        } if run_uid == uid => Some(recipe.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                append_clear(&ledger, uid, recipe)?;
                println!("collected {uid}");
            }
        }
        LedgerCommand::Compare {
            experiment,
            baseline,
            candidate,
            metric,
            alpha,
            json,
        } => {
            let read = ledger.read().map_err(|e| anyhow!("{e}"))?;
            let arm = |fingerprint: &str| -> Arm {
                // Collect the metric from every ended run whose start declared
                // this experiment and whose config fingerprint matches.
                let uids: Vec<&str> = read
                    .records
                    .iter()
                    .filter_map(|r| match r {
                        Record::RunStarted {
                            run_uid,
                            experiment: e,
                            identity,
                            ..
                        } if e.as_deref() == Some(experiment.as_str())
                            && identity
                                .config_fingerprint
                                .as_deref()
                                .is_some_and(|c| c.starts_with(fingerprint)) =>
                        {
                            Some(run_uid.as_str())
                        }
                        _ => None,
                    })
                    .collect();
                let values = read
                    .records
                    .iter()
                    .filter_map(|r| match r {
                        Record::RunEnded {
                            run_uid, metrics, ..
                        } if uids.contains(&run_uid.as_str()) => metrics.get(&metric).copied(),
                        _ => None,
                    })
                    .collect();
                Arm {
                    label: fingerprint.to_string(),
                    values,
                }
            };

            let result = compare(&metric, &arm(&baseline), &arm(&candidate), alpha);
            if json {
                // Hand-rolled: the report is small and adding a Serialize impl
                // to the comparison types would put a presentation concern in
                // the analysis module.
                let verdict = match &result.verdict {
                    Verdict::Indeterminate {
                        reason,
                        seeds_per_arm_needed,
                    } => format!(
                        "{{\"kind\":\"indeterminate\",\"reason\":{},\"seeds_per_arm_needed\":{}}}",
                        serde_json::to_string(reason)?,
                        seeds_per_arm_needed
                    ),
                    Verdict::Decided {
                        p_value,
                        significant,
                    } => format!(
                        "{{\"kind\":\"decided\",\"p_value\":{p_value},\"significant\":{significant}}}"
                    ),
                };
                println!(
                    "{{\"metric\":{},\"baseline_n\":{},\"candidate_n\":{},\"delta\":{},\"within_sd\":{},\"min_achievable_p\":{},\"verdict\":{}}}",
                    serde_json::to_string(&result.metric)?,
                    result.baseline_n,
                    result.candidate_n,
                    result.delta,
                    result
                        .within_sd
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "null".into()),
                    result.min_achievable_p,
                    verdict
                );
            } else {
                println!("experiment {experiment}  metric {}", result.metric);
                println!(
                    "  {:<24} n={}  mean {:.6}",
                    result.baseline,
                    result.baseline_n,
                    arm(&baseline).mean()
                );
                println!(
                    "  {:<24} n={}  mean {:.6}",
                    result.candidate,
                    result.candidate_n,
                    arm(&candidate).mean()
                );
                println!("  delta            {:+.6}", result.delta);
                match result.within_sd {
                    Some(sd) => println!("  within-arm sd    {sd:.6}  (the noise floor)"),
                    None => println!("  within-arm sd    unknown — no arm has a repeat"),
                }
                if let Some(effect) = result.effect_size {
                    println!("  effect size      {effect:.2} sd");
                }
                match &result.verdict {
                    Verdict::Indeterminate {
                        reason,
                        seeds_per_arm_needed,
                    } => {
                        println!("  VERDICT          INDETERMINATE");
                        println!("    {reason}");
                        println!(
                            "    {seeds_per_arm_needed} seeds per arm would make this decidable."
                        );
                    }
                    Verdict::Decided {
                        p_value,
                        significant,
                    } => {
                        println!(
                            "  VERDICT          {} (exact permutation p = {p_value:.4})",
                            if *significant {
                                "SIGNIFICANT"
                            } else {
                                "not significant"
                            }
                        );
                    }
                }
            }
        }
        LedgerCommand::Verify {
            require_ledger_rows_resolve,
            truth_ledger_root,
        } => {
            let read = ledger.read().map_err(|e| anyhow!("{e}"))?;
            let mut problems = Vec::new();
            if read.malformed > 0 {
                problems.push(format!("{} malformed line(s)", read.malformed));
            }
            problems.extend(ledger_structure_problems(&read.records));
            if require_ledger_rows_resolve {
                problems.extend(unresolved_ledger_rows(
                    &read.records,
                    truth_ledger_root.as_deref(),
                )?);
            }
            if problems.is_empty() {
                println!("ledger verify: OK ({} record(s))", read.records.len());
            } else {
                for p in &problems {
                    eprintln!("FAIL: {p}");
                }
                return Err(anyhow!("{} problem(s)", problems.len()));
            }
        }
    }
    Ok(())
}

fn append_promotion(ledger: &RunLedger, uid: &str, reason: String, by: String) -> Result<()> {
    let record = Record::Promoted {
        schema: crate::run_ledger::SCHEMA.into(),
        run_uid: uid.to_string(),
        promoted_unix: now_unix(),
        reason,
        by,
    };
    ledger
        .append_checked(&record, |read| {
            if read.malformed > 0 {
                return Err(crate::error::TrainError::other(format!(
                    "ledger contains {} malformed line(s); refusing promotion",
                    read.malformed
                )));
            }
            let summaries = summarize_runs(&read.records);
            let summary = summaries
                .get(uid)
                .ok_or_else(|| crate::error::TrainError::other(format!("{uid}: no run history")))?;
            if !summary.can_promote() {
                return Err(crate::error::TrainError::other(format!(
                    "{uid}: only one completed, uncurated run may be promoted"
                )));
            }
            Ok(())
        })
        .map_err(|error| anyhow!("{error}"))
}

fn append_clear(ledger: &RunLedger, uid: &str, recipe: String) -> Result<()> {
    let record = Record::Cleared {
        schema: crate::run_ledger::SCHEMA.into(),
        run_uid: uid.to_string(),
        cleared_unix: now_unix(),
        recipe,
    };
    ledger
        .append_checked(&record, |read| {
            if read.malformed > 0 {
                return Err(crate::error::TrainError::other(format!(
                    "ledger contains {} malformed line(s); refusing collection",
                    read.malformed
                )));
            }
            let summaries = summarize_runs(&read.records);
            let summary = summaries
                .get(uid)
                .ok_or_else(|| crate::error::TrainError::other(format!("{uid}: no run history")))?;
            if !summary.can_clear() {
                return Err(crate::error::TrainError::other(format!(
                    "{uid}: only one completed Scratch run may be collected"
                )));
            }
            Ok(())
        })
        .map_err(|error| anyhow!("{error}"))
}

#[derive(Default)]
struct RunSummary {
    start_count: usize,
    end_count: usize,
    promoted_count: usize,
    cleared_count: usize,
    effective_tier: Option<Tier>,
    is_lamquant: bool,
    order_valid: bool,
}

impl RunSummary {
    fn can_promote(&self) -> bool {
        self.start_count == 1
            && self.end_count == 1
            && matches!(self.effective_tier, Some(Tier::Scratch | Tier::Recorded))
            && self.promoted_count == 0
            && self.cleared_count == 0
            && self.order_valid
    }

    fn can_clear(&self) -> bool {
        self.can_promote() && self.effective_tier == Some(Tier::Scratch)
    }
}

fn summarize_runs(records: &[Record]) -> std::collections::BTreeMap<&str, RunSummary> {
    let mut runs = std::collections::BTreeMap::new();
    for record in records {
        let summary = runs.entry(record.run_uid()).or_insert_with(|| RunSummary {
            order_valid: true,
            ..RunSummary::default()
        });
        match record {
            Record::RunStarted { tenant, .. } => {
                if summary.start_count != 0 {
                    summary.order_valid = false;
                }
                summary.start_count += 1;
                summary.is_lamquant =
                    Tenant::parse(tenant).is_some_and(|tenant| tenant.project() == "lamquant");
            }
            Record::RunEnded { tier, .. } => {
                if summary.start_count != 1 || summary.end_count != 0 || *tier == Tier::Canonical {
                    summary.order_valid = false;
                }
                summary.end_count += 1;
                summary.effective_tier = Some(*tier);
            }
            Record::Promoted { .. } => {
                if !summary.can_promote() {
                    summary.order_valid = false;
                }
                summary.promoted_count += 1;
                summary.effective_tier = Some(Tier::Canonical);
            }
            Record::Cleared { .. } => {
                if !summary.can_clear() {
                    summary.order_valid = false;
                }
                summary.cleared_count += 1;
            }
        }
    }
    runs
}

fn ledger_structure_problems(records: &[Record]) -> Vec<String> {
    let mut problems = Vec::new();
    for (run_uid, summary) in summarize_runs(records) {
        if summary.start_count != 1 {
            problems.push(format!(
                "{run_uid}: {} run_started records",
                summary.start_count
            ));
        }
        if summary.end_count > 1 {
            problems.push(format!(
                "{run_uid}: {} run_ended records",
                summary.end_count
            ));
        }
        if !summary.order_valid {
            problems.push(format!("{run_uid}: invalid append order"));
        }
    }
    problems
}

#[derive(Debug, PartialEq, Eq)]
struct TruthLedgerRow {
    id: String,
    source: String,
}

fn source_mentions(source: &str, value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    if source.trim() == value {
        return true;
    }
    if source
        .split('`')
        .enumerate()
        .any(|(index, span)| index % 2 == 1 && span.trim() == value)
    {
        return true;
    }
    source.split_whitespace().any(|token| {
        token.trim_matches(|character: char| {
            matches!(
                character,
                ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
            )
        }) == value
    })
}

fn identity_matches_source(identity: &crate::run_ledger::RunIdentity, source: &str) -> bool {
    source_mentions(source, &identity.blut_job_id)
        || identity
            .trainer_run_id
            .as_deref()
            .is_some_and(|value| source_mentions(source, value))
        || identity
            .git_sha
            .as_deref()
            .is_some_and(|value| source_mentions(source, value))
        || identity
            .checkpoint_sha256
            .iter()
            .any(|value| source_mentions(source, value))
        || identity
            .pccp_change_id
            .as_deref()
            .is_some_and(|value| source_mentions(source, value))
        || identity
            .attestation_ref
            .as_deref()
            .is_some_and(|value| source_mentions(source, value))
}

/// Every TRUTH_LEDGER §2 row id claimed by some run, vs the ids the ledger
/// knows. Reported rather than enforced silently: rows `2.4` and `2.7` are
/// expected to fail today because they cite `/tmp/eval_april.py` and a
/// scratchpad script — provenance that cannot be recovered. Surfacing that is
/// the point (ADR 0154 Consequences).
fn unresolved_ledger_rows(
    records: &[Record],
    truth_ledger_root: Option<&std::path::Path>,
) -> Result<Vec<String>> {
    let lamquant_runs: std::collections::BTreeSet<_> = summarize_runs(records)
        .into_iter()
        .filter_map(|(run_uid, summary)| {
            (summary.start_count == 1
                && summary.end_count == 1
                && summary.is_lamquant
                && summary.order_valid)
                .then_some(run_uid)
        })
        .collect();

    let mut out = Vec::new();
    for row in truth_ledger_rows(truth_ledger_root)? {
        let matching_runs: std::collections::BTreeSet<_> = records
            .iter()
            .filter(|record| lamquant_runs.contains(record.run_uid()))
            .filter_map(|record| match record {
                Record::RunStarted {
                    run_uid, identity, ..
                }
                | Record::RunEnded {
                    run_uid, identity, ..
                } if identity.ledger_rows.contains(&row.id)
                    && identity_matches_source(identity, &row.source) =>
                {
                    Some(run_uid.as_str())
                }
                _ => None,
            })
            .collect();
        if matching_runs.is_empty() {
            out.push(format!(
                "TRUTH_LEDGER §2 row {} source {:?} matches no uniquely owned LamQuant ledger run",
                row.id, row.source
            ));
        } else if matching_runs.len() > 1 {
            out.push(format!(
                "TRUTH_LEDGER §2 row {} source {:?} matches {} LamQuant ledger runs (ambiguous provenance)",
                row.id,
                row.source,
                matching_runs.len()
            ));
        }
    }
    Ok(out)
}

/// §2 row ids scraped from `docs/TRUTH_LEDGER.md`, matching the composer's rule
/// (`\d+\.\d+` in the first cell). The explicit verification flag is
/// fail-closed: absent root or ledger source is missing evidence, not zero rows.
fn truth_ledger_rows(explicit_root: Option<&std::path::Path>) -> Result<Vec<TruthLedgerRow>> {
    if let Some(root) = explicit_root {
        return truth_ledger_rows_at(root);
    }
    if let Some(root) = std::env::var_os("LAMQUANT_META_ROOT") {
        return truth_ledger_rows_at(std::path::Path::new(&root));
    }
    let cwd = std::env::current_dir().map_err(|error| anyhow!("cannot read cwd: {error}"))?;
    for root in cwd.ancestors() {
        if root.join("docs/TRUTH_LEDGER.md").is_file() {
            return truth_ledger_rows_at(root);
        }
    }
    Err(anyhow!(
        "cannot locate docs/TRUTH_LEDGER.md; pass --truth-ledger-root, set LAMQUANT_META_ROOT, or run from the LamQuant tree"
    ))
}

fn truth_ledger_rows_at(root: &std::path::Path) -> Result<Vec<TruthLedgerRow>> {
    if root.as_os_str().is_empty() {
        return Err(anyhow!("truth ledger root must be non-empty"));
    }
    let path = root.join("docs/TRUTH_LEDGER.md");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| anyhow!("cannot read {}: {error}", path.display()))?;
    let mut out = Vec::new();
    let mut in_section = false;
    let mut found_section = false;
    let mut found_header = false;
    let mut found_separator = false;
    let mut header_width = 0;
    let mut value_column = None;
    let mut source_column = None;
    for line in text.lines() {
        let section_suffix = line.strip_prefix("## §2");
        if section_suffix.is_some_and(|suffix| {
            suffix.is_empty() || suffix.chars().next().is_some_and(char::is_whitespace)
        }) {
            in_section = true;
            found_section = true;
            continue;
        }
        if in_section && line.starts_with("## ") {
            break;
        }
        if !in_section {
            continue;
        }
        if !line.trim_start().starts_with('|') {
            if found_header && !out.is_empty() {
                if line.trim().is_empty() {
                    break;
                }
                return Err(anyhow!(
                    "{} §2 has non-table content before the provenance table ends",
                    path.display()
                ));
            }
            continue;
        }
        let cells: Vec<_> = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.replace("**", "").trim().to_string())
            .collect();
        if !found_header && cells.first().is_some_and(|cell| cell == "#") {
            let value_columns: Vec<_> = cells
                .iter()
                .enumerate()
                .filter_map(|(index, cell)| (cell == "Value").then_some(index))
                .collect();
            let source_columns: Vec<_> = cells
                .iter()
                .enumerate()
                .filter_map(|(index, cell)| (cell == "Run id / source").then_some(index))
                .collect();
            let id_columns = cells.iter().filter(|cell| cell.as_str() == "#").count();
            if id_columns != 1 || value_columns.len() != 1 || source_columns.len() != 1 {
                return Err(anyhow!(
                    "{} §2 provenance table requires unique #, Value, and Run id / source columns",
                    path.display()
                ));
            }
            header_width = cells.len();
            value_column = value_columns.first().copied();
            source_column = source_columns.first().copied();
            found_header = true;
            continue;
        }
        if !found_header {
            continue;
        }
        let is_separator = cells.iter().all(|cell| {
            !cell.is_empty() && cell.chars().all(|character| matches!(character, '-' | ':'))
        });
        if is_separator {
            if found_separator || !out.is_empty() || cells.len() != header_width {
                return Err(anyhow!(
                    "{} §2 has a misplaced or width-mismatched table separator",
                    path.display()
                ));
            }
            found_separator = true;
            continue;
        }
        if !found_separator {
            return Err(anyhow!(
                "{} §2 provenance table has no separator",
                path.display()
            ));
        }
        if cells.len() != header_width {
            return Err(anyhow!(
                "{} §2 provenance row width does not match its header",
                path.display()
            ));
        }
        for (name, column) in [("Value", value_column), ("Run id / source", source_column)] {
            if !column
                .and_then(|index| cells.get(index))
                .is_some_and(|cell| !cell.is_empty())
            {
                return Err(anyhow!(
                    "{} §2 provenance row has empty {name}",
                    path.display()
                ));
            }
        }
        let source = source_column
            .and_then(|index| cells.get(index))
            .expect("validated source column")
            .clone();
        let first = cells.first().map(String::as_str).unwrap_or("");
        if !first.split_once('.').is_some_and(|(a, b)| {
            !a.is_empty()
                && !b.is_empty()
                && a.chars().all(|c| c.is_ascii_digit())
                && b.chars().all(|c| c.is_ascii_digit())
        }) {
            return Err(anyhow!(
                "{} §2 contains malformed provenance row id {first:?}",
                path.display()
            ));
        }
        out.push(TruthLedgerRow {
            id: first.to_string(),
            source,
        });
    }
    if !found_section {
        return Err(anyhow!("{} has no exact §2 section", path.display()));
    }
    if !found_header {
        return Err(anyhow!(
            "{} §2 has no provenance table header",
            path.display()
        ));
    }
    if out.is_empty() {
        return Err(anyhow!(
            "{} §2 contains no parseable provenance rows",
            path.display()
        ));
    }
    let unique: std::collections::BTreeSet<_> = out.iter().map(|row| &row.id).collect();
    if unique.len() != out.len() {
        return Err(anyhow!(
            "{} §2 contains duplicate provenance row ids",
            path.display()
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_ledger::{Intent, Outcome, RunIdentity, SCHEMA};
    use std::sync::{Arc, Barrier};

    fn write_truth_ledger(root: &std::path::Path, body: &str) {
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("docs/TRUTH_LEDGER.md"), body).unwrap();
    }

    const HEADER: &str = "| # | Value | Run id / source |\n|---|---|---|\n";

    fn started(uid: &str, tenant: &str, rows: &[&str]) -> Record {
        Record::RunStarted {
            schema: SCHEMA.into(),
            run_uid: uid.into(),
            recipe: "test".into(),
            intent: Intent::Campaign,
            started_unix: 0,
            tenant: tenant.into(),
            experiment: None,
            seed: None,
            identity: RunIdentity {
                trainer_run_id: Some(uid.into()),
                ledger_rows: rows.iter().map(|row| (*row).into()).collect(),
                ..RunIdentity::default()
            },
        }
    }

    fn ended(uid: &str, rows: &[&str]) -> Record {
        ended_at_tier(uid, rows, Tier::Recorded)
    }

    fn ended_at_tier(uid: &str, rows: &[&str], tier: Tier) -> Record {
        Record::RunEnded {
            schema: SCHEMA.into(),
            run_uid: uid.into(),
            ended_unix: 1,
            duration_secs: 1,
            outcome: Outcome::Completed,
            tier,
            identity: RunIdentity {
                trainer_run_id: Some(uid.into()),
                ledger_rows: rows.iter().map(|row| (*row).into()).collect(),
                ..RunIdentity::default()
            },
            metrics: std::collections::BTreeMap::new(),
        }
    }

    fn promoted(uid: &str) -> Record {
        Record::Promoted {
            schema: SCHEMA.into(),
            run_uid: uid.into(),
            promoted_unix: 2,
            reason: "test".into(),
            by: "test".into(),
        }
    }

    fn cleared(uid: &str) -> Record {
        Record::Cleared {
            schema: SCHEMA.into(),
            run_uid: uid.into(),
            cleared_unix: 2,
            recipe: "test".into(),
        }
    }

    #[test]
    fn tier_parsing_rejects_unknown() {
        assert!(parse_tier("scratch").is_ok());
        assert!(parse_tier("canonical").is_ok());
        assert!(parse_tier("promoted").is_err());
    }

    #[test]
    fn missing_truth_ledger_source_fails_closed() {
        let root = tempfile::tempdir().unwrap();
        let error = truth_ledger_rows_at(root.path()).unwrap_err();
        assert!(error.to_string().contains("TRUTH_LEDGER.md"));

        let error = truth_ledger_rows_at(std::path::Path::new("")).unwrap_err();
        assert!(error.to_string().contains("non-empty"));
    }

    #[test]
    fn truth_ledger_parser_requires_exact_nonempty_section_two() {
        for body in [
            "## §20 — wrong section\n| 2.1 | value |\n",
            "## §2 — empty section\n| # | value |\n|---|---|\n",
            "## §2 — missing header\n| 2.1 | value |\n",
            "## §2 — malformed row\n| # | Value | Run id / source |\n|---|---|---|\n| 2.x | a | run |\n",
            "## §2 — short row\n| # | Value | Run id / source |\n|---|---|---|\n| 2.1 |\n",
            "## §2 — empty source\n| # | Value | Run id / source |\n|---|---|---|\n| 2.1 | a | |\n",
            "## §2 — short separator\n| # | Value | Run id / source |\n|---|\n| 2.1 | a | run |\n",
            "## §2 — missing leading pipe\n| # | Value | Run id / source |\n|---|---|---|\n| 2.1 | a | run-a |\n2.2 | b | run-b |\n",
        ] {
            let root = tempfile::tempdir().unwrap();
            write_truth_ledger(root.path(), body);
            assert!(truth_ledger_rows_at(root.path()).is_err());
        }
    }

    #[test]
    fn truth_ledger_parser_reads_rows_only_from_exact_section_two() {
        let root = tempfile::tempdir().unwrap();
        write_truth_ledger(
            root.path(),
            &format!(
                "## §2 — provenance\n{HEADER}| 2.1 | a | run-a |\n| **2.2** | b | run-b |\n## §3 — later\n| 3.1 | c |\n"
            ),
        );
        assert_eq!(
            truth_ledger_rows_at(root.path()).unwrap(),
            vec![
                TruthLedgerRow {
                    id: "2.1".into(),
                    source: "run-a".into(),
                },
                TruthLedgerRow {
                    id: "2.2".into(),
                    source: "run-b".into(),
                },
            ]
        );
    }

    #[test]
    fn foreign_tenant_cannot_resolve_lamquant_provenance() {
        let root = tempfile::tempdir().unwrap();
        write_truth_ledger(
            root.path(),
            &format!("## §2 — provenance\n{HEADER}| 2.1 | a | `mine` / `duplicate` |\n"),
        );
        let records = vec![started("foreign", "tritium", &["2.1"])];
        let problems = unresolved_ledger_rows(&records, Some(root.path())).unwrap();
        assert_eq!(problems.len(), 1);

        let records = vec![
            started("mine", "lamquant/codec", &["2.1"]),
            ended("mine", &[]),
        ];
        assert!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .is_empty()
        );

        let records = vec![started("invalid", "lamquant/../clinical", &["2.1"])];
        assert_eq!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .len(),
            1
        );

        let records = vec![
            started("duplicate", "lamquant/a", &[]),
            started("duplicate", "lamquant/b", &["2.1"]),
        ];
        assert_eq!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .len(),
            1
        );
        assert!(!ledger_structure_problems(&records).is_empty());
    }

    #[test]
    fn truth_ledger_parser_rejects_duplicate_row_ids() {
        let root = tempfile::tempdir().unwrap();
        write_truth_ledger(
            root.path(),
            &format!("## §2 — provenance\n{HEADER}| 2.1 | a | run-a |\n| 2.1 | b | run-b |\n"),
        );
        let error = truth_ledger_rows_at(root.path()).unwrap_err();
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn duplicate_terminal_records_are_ambiguous() {
        let root = tempfile::tempdir().unwrap();
        write_truth_ledger(
            root.path(),
            &format!("## §2 — provenance\n{HEADER}| 2.1 | a | duplicate |\n"),
        );
        let records = vec![
            started("duplicate", "lamquant", &[]),
            ended("duplicate", &[]),
            ended("duplicate", &["2.1"]),
        ];
        assert!(!ledger_structure_problems(&records).is_empty());
        assert_eq!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn ledger_row_claim_must_match_cited_source_identity() {
        let root = tempfile::tempdir().unwrap();
        write_truth_ledger(
            root.path(),
            &format!("## §2 — provenance\n{HEADER}| 2.4 | a | `/tmp/eval_april.py` |\n"),
        );
        let records = vec![started("unrelated-run", "lamquant", &["2.4"])];
        assert_eq!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .len(),
            1,
            "a self-declared row backlink cannot replace cited source identity"
        );

        write_truth_ledger(
            root.path(),
            &format!("## §2 — provenance\n{HEADER}| 2.4 | a | `wanted-run` |\n"),
        );
        let records = vec![started("wanted-run", "lamquant", &["2.4"])];
        assert_eq!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .len(),
            1,
            "an incomplete run cannot resolve a scientific result"
        );
        let records = vec![
            started("wanted-run", "lamquant", &["2.4"]),
            ended("wanted-run", &[]),
        ];
        assert!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn later_start_cannot_repair_invalid_append_order() {
        let root = tempfile::tempdir().unwrap();
        write_truth_ledger(
            root.path(),
            &format!("## §2 — provenance\n{HEADER}| 2.1 | a | out-of-order |\n"),
        );
        let records = vec![
            ended("out-of-order", &["2.1"]),
            started("out-of-order", "lamquant", &[]),
        ];
        assert!(!ledger_structure_problems(&records).is_empty());
        assert_eq!(
            unresolved_ledger_rows(&records, Some(root.path()))
                .unwrap()
                .len(),
            1
        );

        for record in [promoted("late"), cleared("late")] {
            assert!(
                !ledger_structure_problems(&[record, started("late", "lamquant", &[]),]).is_empty()
            );
        }
    }

    #[test]
    fn curation_requires_completed_compatible_tier() {
        for records in [
            vec![started("x", "lamquant", &[]), promoted("x")],
            vec![started("x", "lamquant", &[]), cleared("x")],
            vec![started("x", "lamquant", &[]), ended("x", &[]), cleared("x")],
            vec![
                started("x", "lamquant", &[]),
                ended_at_tier("x", &[], Tier::Scratch),
                promoted("x"),
                cleared("x"),
            ],
            vec![
                started("x", "lamquant", &[]),
                ended_at_tier("x", &[], Tier::Canonical),
            ],
        ] {
            assert!(!ledger_structure_problems(&records).is_empty());
        }

        for records in [
            vec![
                started("x", "lamquant", &[]),
                ended("x", &[]),
                promoted("x"),
            ],
            vec![
                started("x", "lamquant", &[]),
                ended_at_tier("x", &[], Tier::Scratch),
                cleared("x"),
            ],
        ] {
            assert!(ledger_structure_problems(&records).is_empty());
        }
    }

    #[test]
    fn concurrent_promotions_append_exactly_one_record() {
        let root = tempfile::tempdir().unwrap();
        let ledger = RunLedger::at(root.path().join("ledger.jsonl"));
        ledger.append(&started("x", "lamquant", &[])).unwrap();
        ledger.append(&ended("x", &[])).unwrap();

        let barrier = Arc::new(Barrier::new(3));
        let handles: Vec<_> = (0..2)
            .map(|index| {
                let ledger = ledger.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    append_promotion(&ledger, "x", format!("reason-{index}"), format!("u{index}"))
                })
            })
            .collect();
        barrier.wait();
        let successes = handles
            .into_iter()
            .map(|handle| handle.join().unwrap().is_ok())
            .filter(|success| *success)
            .count();
        assert_eq!(successes, 1);

        let read = ledger.read().unwrap();
        assert_eq!(
            read.records
                .iter()
                .filter(|record| matches!(record, Record::Promoted { .. }))
                .count(),
            1
        );
        assert!(ledger_structure_problems(&read.records).is_empty());
    }

    #[test]
    fn curation_rejects_malformed_history_without_appending() {
        let root = tempfile::tempdir().unwrap();
        let ledger = RunLedger::at(root.path().join("ledger.jsonl"));
        ledger.append(&started("x", "lamquant", &[])).unwrap();
        ledger.append(&ended("x", &[])).unwrap();
        {
            use std::io::Write as _;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(ledger.path())
                .unwrap();
            file.write_all(b"{not-json}\n").unwrap();
        }
        let before = std::fs::read(ledger.path()).unwrap();

        assert!(append_promotion(&ledger, "x", "reason".into(), "user".into()).is_err());
        assert!(append_clear(&ledger, "x", "test".into()).is_err());
        assert_eq!(std::fs::read(ledger.path()).unwrap(), before);
    }

    #[test]
    fn curation_rejects_unterminated_history_without_appending() {
        let root = tempfile::tempdir().unwrap();
        let ledger = RunLedger::at(root.path().join("ledger.jsonl"));
        ledger.append(&started("x", "lamquant", &[])).unwrap();
        ledger.append(&ended("x", &[])).unwrap();
        let mut before = std::fs::read(ledger.path()).unwrap();
        assert_eq!(before.pop(), Some(b'\n'));
        std::fs::write(ledger.path(), &before).unwrap();

        assert!(append_promotion(&ledger, "x", "reason".into(), "user".into()).is_err());
        assert_eq!(std::fs::read(ledger.path()).unwrap(), before);
    }

    #[test]
    fn concurrent_clears_and_promote_vs_clear_have_one_legal_winner() {
        for promote_race in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let ledger = RunLedger::at(root.path().join("ledger.jsonl"));
            ledger.append(&started("x", "lamquant", &[])).unwrap();
            ledger
                .append(&ended_at_tier("x", &[], Tier::Scratch))
                .unwrap();

            let barrier = Arc::new(Barrier::new(3));
            let first_ledger = ledger.clone();
            let first_barrier = Arc::clone(&barrier);
            let first = std::thread::spawn(move || {
                first_barrier.wait();
                append_clear(&first_ledger, "x", "test".into())
            });
            let second_ledger = ledger.clone();
            let second_barrier = Arc::clone(&barrier);
            let second = std::thread::spawn(move || {
                second_barrier.wait();
                if promote_race {
                    append_promotion(&second_ledger, "x", "reason".into(), "user".into())
                } else {
                    append_clear(&second_ledger, "x", "test".into())
                }
            });
            barrier.wait();
            let successes = [first.join().unwrap(), second.join().unwrap()]
                .into_iter()
                .filter(Result::is_ok)
                .count();
            assert_eq!(successes, 1);

            let read = ledger.read().unwrap();
            let curations = read
                .records
                .iter()
                .filter(|record| matches!(record, Record::Promoted { .. } | Record::Cleared { .. }))
                .count();
            assert_eq!(curations, 1);
            assert!(ledger_structure_problems(&read.records).is_empty());
        }
    }
}
