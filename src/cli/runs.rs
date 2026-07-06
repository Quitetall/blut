// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut runs`/`jobs`/`log`/`cancel`/`dag`/`compare`/`results` — job
//! inspection and control.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

use super::*;

#[derive(Subcommand, Debug)]
pub(super) enum RunsCommand {
    /// Diff two jobs' recipe/args provenance + show each outcome.
    Diff {
        /// First job id (or unique prefix).
        id1: String,
        /// Second job id (or unique prefix).
        id2: String,
        /// Show identical keys too (default elides them).
        #[arg(long)]
        all: bool,
        /// Emit the diff as JSON (for scripts/agents).
        #[arg(long)]
        json: bool,
    },
}

/// `blut compare <A> <B>` (E4): provenance + a side-by-side FINAL-metric panel
/// (with Δ) + the GPU-saturation summary, all from the queryable metric store.
pub(super) fn run_compare(a: &str, b: &str) -> Result<()> {
    use std::collections::BTreeMap;
    let ja = crate::jobs::resolve_job_id(a).map_err(|e| anyhow!("{e}"))?;
    let jb = crate::jobs::resolve_job_id(b).map_err(|e| anyhow!("{e}"))?;
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("open lineage.db: {e}"))?;

    let show_run = |id: &str| -> String {
        match db.get_run(id) {
            Ok(Some(r)) => format!(
                "{id}  recipe={} outcome={} git={}",
                r.recipe,
                r.outcome.as_deref().unwrap_or("?"),
                r.git_sha
                    .as_deref()
                    .map(|s| &s[..s.len().min(8)])
                    .unwrap_or("?"),
            ),
            _ => format!("{id}  (no provenance row)"),
        }
    };
    println!("A  {}", show_run(&ja));
    println!("B  {}", show_run(&jb));

    let ma: BTreeMap<String, f64> = db
        .final_metrics(&ja)
        .map_err(|e| anyhow!("{e}"))?
        .into_iter()
        .collect();
    let mb: BTreeMap<String, f64> = db
        .final_metrics(&jb)
        .map_err(|e| anyhow!("{e}"))?
        .into_iter()
        .collect();
    let keys: std::collections::BTreeSet<&String> = ma.keys().chain(mb.keys()).collect();
    if keys.is_empty() {
        println!("\n(no metrics recorded for either run)");
    } else {
        println!(
            "\n{:<18} {:>12} {:>12} {:>12}",
            "metric", "A", "B", "Δ(B−A)"
        );
        for k in keys {
            let fmt = |v: Option<&f64>| v.map(|x| format!("{x:.4}")).unwrap_or_else(|| "—".into());
            let delta = match (ma.get(k), mb.get(k)) {
                (Some(x), Some(y)) => format!("{:+.4}", y - x),
                _ => "—".into(),
            };
            println!(
                "{:<18} {:>12} {:>12} {:>12}",
                k,
                fmt(ma.get(k)),
                fmt(mb.get(k)),
                delta
            );
        }
    }

    // GPU saturation (the owner's first-class metric).
    let sat = |id: &str| db.gpu_saturation(id, 50.0).ok().flatten();
    if let (Some(sa), Some(sb)) = (sat(&ja), sat(&jb)) {
        println!(
            "\nGPU saturation   A {:.1}% (wasted {:.0}%)   B {:.1}% (wasted {:.0}%)",
            sa.saturation,
            sa.wasted * 100.0,
            sb.saturation,
            sb.wasted * 100.0
        );
    }
    Ok(())
}

/// `blut results <job> [--json] [--metric M]` — one run's results from the
/// metric store (ADR 0071 A3): best (peak) value + per-step trajectory +
/// per-band PRD + the produced ckpt path. Replaces grepping `BLUT_METRIC` out
/// of raw logs and `ls -t`-hunting the run CSV. The metric store is a derived,
/// rebuildable index (ADR 0071 §3) — an un-flushed/just-started job has no rows
/// yet, so we say "no data yet" rather than erroring.
pub(super) fn run_results(
    job: &str,
    json: bool,
    metric: &str,
    force_maximize: Option<bool>,
) -> Result<()> {
    let job_id = crate::jobs::resolve_job_id(job).map_err(|e| anyhow!("{e}"))?;
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("open lineage.db: {e}"))?;

    // Direction: explicit --maximize/--minimize wins; else the name heuristic —
    // val_r-shaped headlines maximize, a PRD/loss/err-shaped name minimizes.
    // `--maximize`/`--minimize` is the escape hatch for a non-standard name.
    let maximize = force_maximize.unwrap_or_else(|| {
        let lower = metric.to_ascii_lowercase();
        !(lower.contains("prd")
            || lower.contains("loss")
            || lower.contains("err")
            || lower.contains("mae")
            || lower.contains("rmse")
            || lower.contains("nrmse"))
    });

    let best = db
        .best_metric(&job_id, metric, maximize)
        .map_err(|e| anyhow!("{e}"))?;
    let series = db
        .metric_series(&job_id, metric)
        .map_err(|e| anyhow!("{e}"))?;
    let finals = db.final_metrics(&job_id).map_err(|e| anyhow!("{e}"))?;
    // per-band PRD: final metrics whose name carries a band prefix + "prd"
    // (delta_prd / theta_prd / … — emitted by --detail-bands).
    let per_band: Vec<(String, f64)> = finals
        .iter()
        .filter(|(k, _)| {
            let kl = k.to_ascii_lowercase();
            kl.contains("prd") && kl != "prd" && kl != metric.to_ascii_lowercase()
        })
        .cloned()
        .collect();
    let ckpt = db
        .terminal_artifact(&job_id)
        .map_err(|e| anyhow!("{e}"))?
        .and_then(|a| a.sidecar_path);

    let has_data = best.is_some() || !series.is_empty() || !finals.is_empty() || ckpt.is_some();

    if json {
        let traj: Vec<serde_json::Value> = series
            .iter()
            .map(|(s, v)| serde_json::json!({ "step": s, "value": v }))
            .collect();
        let band_obj: serde_json::Map<String, serde_json::Value> = per_band
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(v)))
            .collect();
        let out = serde_json::json!({
            "job": job_id,
            "metric": metric,
            "maximize": maximize,
            "has_data": has_data,
            "best": best.map(|(step, value)| serde_json::json!({ "step": step, "value": value })),
            "trajectory": traj,
            "per_band_prd": band_obj,
            "ckpt_path": ckpt,
        });
        emit_json(&out)?;
        return Ok(());
    }

    println!("job  {job_id}");
    if !has_data {
        println!("(no data yet — the run hasn't flushed any metrics to the store)");
        return Ok(());
    }
    match best {
        Some((step, value)) => println!(
            "best {metric}  {value:.4}  @ step {step}  ({} of {} samples)",
            if maximize { "max" } else { "min" },
            series.len()
        ),
        None => println!("best {metric}  — (no per-step samples recorded)"),
    }
    if let Some(p) = &ckpt {
        println!("ckpt {p}");
    }
    if !per_band.is_empty() {
        println!("\nper-band PRD (final):");
        for (k, v) in &per_band {
            println!("  {k:<16} {v:>8.3}");
        }
    }
    if !series.is_empty() {
        // A compact sparkline-free trajectory tail (last up to 8 points) so the
        // shape is legible without a plotting dep.
        let tail: Vec<&(i64, f64)> = series.iter().rev().take(8).collect();
        print!("\n{metric} trajectory (last {}):", tail.len());
        for (s, v) in tail.into_iter().rev() {
            print!("  {s}:{v:.4}");
        }
        println!();
    }
    Ok(())
}

/// `blut dag <job> [--json]` — render a job's DAG: per-node status + edges,
/// built from the persisted `plan.json` + the live `status.jsonl` (+ HPO trial
/// attribution when present). No daemon; re-run to refresh.
pub(super) fn run_dag(job: Option<String>, json: bool) -> Result<()> {
    let job_id = match job {
        Some(q) => crate::jobs::resolve_job_id(&q).map_err(|e| anyhow!("{e}"))?,
        // `list_jobs` sorts ascending by timestamp-monotonic id; prefer the most
        // recent job that ACTUALLY has a `plan.json` (a pre-v0.20 run has none,
        // so blindly taking the last job would error on a stale job).
        None => crate::jobs::list_jobs()
            .map_err(|e| anyhow!("list jobs: {e}"))?
            .into_iter()
            .rev()
            .map(|s| s.id)
            .find(|id| {
                crate::paths::job_dir(id)
                    .ok()
                    .and_then(|d| crate::framework::graph::PlanGraph::read_from(&d))
                    .is_some()
            })
            .ok_or_else(|| anyhow!("no jobs with a plan.json (run a recipe first)"))?,
    };
    let snap = crate::framework::graph_snapshot(&job_id).map_err(|e| anyhow!("{e}"))?;
    if json {
        emit_json(&snap)?;
        return Ok(());
    }
    // Tally per-status for a one-line header.
    let mut counts: std::collections::BTreeMap<&'static str, u32> =
        std::collections::BTreeMap::new();
    for n in &snap.nodes {
        *counts.entry(n.status.as_str()).or_default() += 1;
    }
    let tally = counts
        .iter()
        .map(|(s, c)| format!("{c} {s}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "dag {} (job {}) — {} nodes, {} edges [{tally}]",
        snap.name,
        snap.job,
        snap.nodes.len(),
        snap.edges.len()
    );
    println!(
        "{:<4} {:<22} {:<8} {:<8} {:<6} preds  detail",
        "idx", "ingredient", "status", "elapsed", "trial"
    );
    for n in &snap.nodes {
        let preds = snap
            .edges
            .iter()
            .filter(|e| e.to == n.idx)
            .map(|e| e.from.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let preds = if preds.is_empty() {
            "─".to_string()
        } else {
            preds
        };
        let elapsed = n
            .elapsed_secs
            .map(|s| format!("{s:.1}s"))
            .unwrap_or_else(|| "─".into());
        let trial = n
            .hpo
            .as_ref()
            .map(|h| format!("t{}", h.trial_id))
            .unwrap_or_else(|| "─".into());
        // For an HPO node the overlay (the diff that defines the trial) is the
        // useful detail; otherwise fall back to the args summary.
        let detail = match &n.hpo {
            Some(h) => h
                .overlay
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" "),
            None => n.args_summary.clone(),
        };
        println!(
            "{:<4} {:<22} {:<8} {:<8} {:<6} {:<6} {}",
            n.idx,
            n.stage_name,
            n.status.as_str(),
            elapsed,
            trial,
            preds,
            detail
        );
    }
    Ok(())
}

pub(super) fn run_jobs(json: bool) -> Result<()> {
    let jobs = jobs::list_jobs()?;
    if json {
        // JobSummary derives Serialize — emit the array verbatim so a
        // script/agent gets the same data the table renders.
        emit_json(&jobs)?;
        return Ok(());
    }
    if jobs.is_empty() {
        println!("no jobs.");
        return Ok(());
    }
    println!(
        "{:<24} {:<10} {:<6} {:<24} last",
        "id", "state", "pid", "output"
    );
    for j in jobs {
        let last = match (j.last_step, j.last_loss, j.final_loss) {
            (_, _, Some(fl)) => format!("final_loss={fl:.4}"),
            (Some(step), Some(loss), _) => format!("step={step} loss={loss:.4}"),
            _ => "-".into(),
        };
        let pid = j.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        let output = j.output_name.unwrap_or_else(|| "-".into());
        println!(
            "{:<24} {:<10} {:<6} {:<24} {}",
            j.id,
            j.state.as_str(),
            pid,
            output,
            last
        );
    }
    Ok(())
}

pub(super) fn run_runs_cmd(cmd: RunsCommand) -> Result<()> {
    match cmd {
        RunsCommand::Diff {
            id1,
            id2,
            all,
            json,
        } => crate::runs::diff(&id1, &id2, all, json).map_err(|e| anyhow!("{e}")),
    }
}

pub(super) async fn run_cancel(id_query: &str, grace: Duration) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    eprintln!("cancelling {id} (grace {grace:?})...");
    jobs::cancel_job(&id, grace).await?;
    eprintln!("cancelled.");
    Ok(())
}

pub(super) fn run_log(id_query: &str, tail: usize, json: bool) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    let updates = jobs::read_status(&id)?;
    if json {
        // Raw status stream as JSON lines (one StatusUpdate per line),
        // tail-trimmed like the rendered view.
        let start = if tail == 0 {
            0
        } else {
            updates.len().saturating_sub(tail)
        };
        for u in &updates[start..] {
            println!(
                "{}",
                serde_json::to_string(u).map_err(|e| anyhow!("serialize status: {e}"))?
            );
        }
        return Ok(());
    }
    let rendered = jobs::render_log(&updates);
    if tail == 0 {
        print!("{rendered}");
    } else {
        let lines: Vec<&str> = rendered.lines().collect();
        let start = lines.len().saturating_sub(tail);
        for l in &lines[start..] {
            println!("{l}");
        }
    }
    Ok(())
}
