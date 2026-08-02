// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut ledger` — read and curate the append-only run ledger (ADR 0152).
//!
//! Every mutating subcommand here APPENDS. Nothing rewrites or deletes a line,
//! because the ledger is the record of what happened, not a view of what is
//! currently true. `promote` appends a promotion; `gc` appends tombstones. If a
//! future subcommand needs to remove a line, it is the wrong subcommand.

use anyhow::{Result, anyhow};
use clap::Subcommand;

use crate::run_ledger::{Record, RunLedger, Tier};

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
    Verify {
        /// Also require every TRUTH_LEDGER §2 citation to resolve to a run.
        #[arg(long)]
        require_ledger_rows_resolve: bool,
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
            ledger
                .append(&Record::Promoted {
                    schema: crate::run_ledger::SCHEMA.into(),
                    run_uid: uid.clone(),
                    promoted_unix: now_unix(),
                    reason,
                    by,
                })
                .map_err(|e| anyhow!("{e}"))?;
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
            let victims: Vec<&String> = tiers
                .iter()
                .filter(|(uid, t)| t.is_clearable() && !already.contains(uid.as_str()))
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
                ledger
                    .append(&Record::Cleared {
                        schema: crate::run_ledger::SCHEMA.into(),
                        run_uid: uid.clone(),
                        cleared_unix: now_unix(),
                        recipe,
                    })
                    .map_err(|e| anyhow!("{e}"))?;
                println!("collected {uid}");
            }
        }
        LedgerCommand::Verify {
            require_ledger_rows_resolve,
        } => {
            let read = ledger.read().map_err(|e| anyhow!("{e}"))?;
            let mut problems = Vec::new();
            if read.malformed > 0 {
                problems.push(format!("{} malformed line(s)", read.malformed));
            }
            // A Promoted/RunEnded for a run that never started means the stream
            // lost its head — worth naming rather than rendering a headless run.
            let started: std::collections::BTreeSet<&str> = read
                .records
                .iter()
                .filter_map(|r| match r {
                    Record::RunStarted { run_uid, .. } => Some(run_uid.as_str()),
                    _ => None,
                })
                .collect();
            for r in &read.records {
                if !matches!(r, Record::RunStarted { .. }) && !started.contains(r.run_uid()) {
                    problems.push(format!("{}: record without a run_started", r.run_uid()));
                }
            }
            if require_ledger_rows_resolve {
                problems.extend(unresolved_ledger_rows(&read.records));
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

/// Every TRUTH_LEDGER §2 row id claimed by some run, vs the ids the ledger
/// knows. Reported rather than enforced silently: rows `2.4` and `2.7` are
/// expected to fail today because they cite `/tmp/eval_april.py` and a
/// scratchpad script — provenance that cannot be recovered. Surfacing that is
/// the point (ADR 0152 Consequences).
fn unresolved_ledger_rows(records: &[Record]) -> Vec<String> {
    let mut claimed = std::collections::BTreeSet::new();
    for r in records {
        let identity = match r {
            Record::RunStarted { identity, .. } | Record::RunEnded { identity, .. } => identity,
            _ => continue,
        };
        for row in &identity.ledger_rows {
            claimed.insert(row.clone());
        }
    }
    let mut out = Vec::new();
    for row in truth_ledger_row_ids() {
        if !claimed.contains(&row) {
            out.push(format!(
                "TRUTH_LEDGER §2 row {row} is cited by no ledger run (unresolvable provenance)"
            ));
        }
    }
    out
}

/// §2 row ids scraped from `docs/TRUTH_LEDGER.md`, matching the composer's rule
/// (`\d+\.\d+` in the first cell). Returns empty when the file is absent so a
/// submodule-only checkout does not fail the check spuriously.
fn truth_ledger_row_ids() -> Vec<String> {
    let Ok(root) = std::env::var("LAMQUANT_META_ROOT") else {
        return Vec::new();
    };
    let path = std::path::Path::new(&root).join("docs/TRUTH_LEDGER.md");
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut in_section = false;
    for line in text.lines() {
        if line.starts_with("## §2") {
            in_section = true;
            continue;
        }
        if in_section && line.starts_with("## ") {
            break;
        }
        if !in_section || !line.starts_with('|') {
            continue;
        }
        let first = line
            .trim_matches('|')
            .split('|')
            .next()
            .unwrap_or("")
            .trim();
        let first = first.trim_matches('*').trim();
        if first.split_once('.').is_some_and(|(a, b)| {
            !a.is_empty()
                && !b.is_empty()
                && a.chars().all(|c| c.is_ascii_digit())
                && b.chars().all(|c| c.is_ascii_digit())
        }) {
            out.push(first.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_parsing_rejects_unknown() {
        assert!(parse_tier("scratch").is_ok());
        assert!(parse_tier("canonical").is_ok());
        assert!(parse_tier("promoted").is_err());
    }

    #[test]
    fn missing_meta_root_yields_no_rows_rather_than_failing() {
        // A submodule-only checkout must not fail the check spuriously.
        unsafe { std::env::remove_var("LAMQUANT_META_ROOT") };
        assert!(truth_ledger_row_ids().is_empty());
    }
}
