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
/// metric store (ADR 0156 A3): best (peak) value + per-step trajectory +
/// per-band PRD + the produced ckpt path. Replaces grepping `BLUT_METRIC` out
/// of raw logs and `ls -t`-hunting the run CSV. The metric store is a derived,
/// rebuildable index (ADR 0156 §3) — an un-flushed/just-started job has no rows
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
        // The final stage's artifact sidecar; `blut artifact inspect` reads it.
        println!("output sidecar {p}");
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

/// What a job ran: its recipe (from the resume marker a registry recipe
/// writes, else the plan graph every plan launch writes, which is also what
/// the lineage index records), else a legacy single-trainer job's output name.
/// The `jobs` column showed only that output name, which no plan launch has,
/// so it read `-` for every job a recipe ran.
fn job_recipe(job: &jobs::JobSummary) -> Option<String> {
    let dir = crate::paths::job_dir(&job.id).ok();
    dir.as_deref()
        .and_then(|dir| RecipeMarker::read_from(dir).ok().map(|marker| marker.name))
        .or_else(|| {
            dir.as_deref()
                .and_then(crate::framework::PlanGraph::read_from)
                .map(|graph| graph.name)
        })
        .or_else(|| job.output_name.clone())
}

pub(super) fn run_jobs(json: bool) -> Result<()> {
    let jobs = jobs::list_jobs()?;
    if json {
        // JobSummary derives Serialize; each object also carries the `recipe`
        // the table renders, so a script/agent gets the same data.
        let rows = jobs
            .iter()
            .map(|job| {
                let mut row = serde_json::to_value(job)?;
                if let Some(object) = row.as_object_mut() {
                    object.insert("recipe".into(), serde_json::json!(job_recipe(job)));
                }
                Ok(row)
            })
            .collect::<serde_json::Result<Vec<_>>>()?;
        emit_json(&rows)?;
        return Ok(());
    }
    if jobs.is_empty() {
        println!("no jobs.");
        return Ok(());
    }
    println!(
        "{:<25} {:<10} {:<8} {:<24} last",
        "id", "state", "pid", "recipe"
    );
    for j in jobs {
        let last = match (j.last_step, j.last_loss, j.final_loss) {
            (_, _, Some(fl)) => format!("final_loss={fl:.4}"),
            (Some(step), Some(loss), _) => format!("step={step} loss={loss:.4}"),
            _ => "-".into(),
        };
        // A job whose stages run in-process has no trainer pid; show the
        // live `blut` process driving it, which is what `blut cancel` signals.
        let pid = j
            .pid
            .or_else(|| {
                (j.state == JobState::Running)
                    .then(|| jobs::live_driver(&j.id))
                    .flatten()
            })
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        let recipe = job_recipe(&j).unwrap_or_else(|| "-".into());
        println!(
            "{:<25} {:<10} {:<8} {:<24} {}",
            j.id,
            j.state.as_str(),
            pid,
            recipe,
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
    let state = jobs::read_state(&id).ok();
    if let Some(state @ (JobState::Done | JobState::Failed | JobState::Cancelled)) = state
        && jobs::live_driver(&id).is_none()
    {
        // A Slurm job or a systemd unit one of its stages started can still
        // run (its BLUT process was killed after recording the state); this
        // stops and reports those, and leaves the state as it is.
        jobs::cancel_job(&id, grace).await?;
        eprintln!(
            "{id} already finished ({}); nothing else to cancel.",
            state.as_str()
        );
        return Ok(());
    }
    eprintln!("cancelling {id} (grace {grace:?})...");
    jobs::cancel_job(&id, grace).await?;
    eprintln!("cancelled.");
    Ok(())
}

/// What `blut log` says when it has nothing to print. It renders trainer
/// progress (`StatusUpdate`) only; a job whose stages run in-process records
/// stage events instead, and the log printed nothing at all with exit 0, as
/// if the job had never run. The hint goes to stderr so `--json` stays clean.
fn empty_log_hint(id: &str) -> Result<String> {
    let n = jobs::read_status_lines(id)?.len();
    Ok(if n == 0 {
        format!("no log entries recorded for {id} yet.")
    } else {
        format!(
            "no trainer progress in {id}'s log; its {n} stage event(s) are shown by \
             `blut dag {id}` (and `blut lineage show {id}`)."
        )
    })
}

/// The trainer progress of a job as `blut log` prints it: [`jobs::render_log`]
/// lines, each prefixed with the stage that forwarded it when more than one
/// stage did (the trials of an HPO run, say), so interleaved progress stays
/// attributable. `notes[i]`, when present, is appended to update `i`'s lines.
fn render_progress(progress: &[jobs::Progress], notes: &[Option<String>]) -> String {
    let sources: std::collections::BTreeSet<_> =
        progress.iter().filter_map(|p| p.source.as_ref()).collect();
    let attributed = sources.len() >= 2;
    let mut out = String::new();
    for (i, p) in progress.iter().enumerate() {
        let prefix = match (&p.source, attributed) {
            (Some((idx, stage)), true) => format!("[{idx} {stage}] "),
            _ => String::new(),
        };
        let note = notes.get(i).and_then(Option::as_deref);
        for line in jobs::render_log(std::slice::from_ref(&p.update)).lines() {
            out.push_str(&prefix);
            out.push_str(line);
            if let Some(note) = note {
                out.push_str("  (");
                out.push_str(note);
                out.push(')');
            }
            out.push('\n');
        }
    }
    out
}

/// Where a path a trainer reported lives now. A stage runs in a private
/// attempt directory, `<job>/stages/.tmp-<idx>-<stage>-<key>`, which is renamed
/// to `<job>/stages/<idx>-<stage>` when the stage commits, so the `saved` and
/// `done` paths a trainer prints name a directory that no longer exists.
#[derive(Debug, PartialEq)]
enum AttemptPath {
    /// The attempt committed: the same file under the committed directory.
    Committed(std::path::PathBuf),
    /// The path is still in an attempt directory (the stage is running, or
    /// that attempt did not commit); `stages/<idx>-<stage>` is where it moves
    /// on commit.
    Uncommitted { committed_dir: String },
    /// Not inside one of this job's attempt directories.
    Other,
}

fn attempt_path(job_id: &str, path: &std::path::Path) -> AttemptPath {
    use std::path::{Component, PathBuf};
    let comps: Vec<Component<'_>> = path.components().collect();
    let found = comps.windows(3).position(|w| {
        w[0].as_os_str() == job_id
            && w[1].as_os_str() == "stages"
            && w[2].as_os_str().to_string_lossy().starts_with(".tmp-")
    });
    let Some(at) = found else {
        return AttemptPath::Other;
    };
    let tmp_name = comps[at + 2].as_os_str().to_string_lossy().into_owned();
    // `.tmp-<idx>-<stage>-<key hex>` → `<idx>-<stage>`.
    let Some((committed_name, key)) = tmp_name[".tmp-".len()..].rsplit_once('-') else {
        return AttemptPath::Other;
    };
    if key.is_empty() || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
        return AttemptPath::Other;
    }
    let stages: PathBuf = comps[..at + 2].iter().collect();
    let tmp_dir = stages.join(&tmp_name);
    let committed_dir = stages.join(committed_name);
    if committed_dir.is_dir() && !tmp_dir.exists() {
        let rest: PathBuf = comps[at + 3..].iter().collect();
        AttemptPath::Committed(committed_dir.join(rest))
    } else {
        AttemptPath::Uncommitted {
            committed_dir: format!("stages/{committed_name}"),
        }
    }
}

/// [`attempt_path`] applied to the paths in a job's progress: a committed
/// attempt's `saved` / `done` path is rewritten to the committed directory;
/// one still in an attempt directory is kept and gets a note saying so.
fn commit_attempt_paths(
    job_id: &str,
    progress: Vec<jobs::Progress>,
) -> (Vec<jobs::Progress>, Vec<Option<String>>) {
    let mut notes = Vec::with_capacity(progress.len());
    let progress = progress
        .into_iter()
        .map(|mut p| {
            let path = match &mut p.update {
                crate::protocol::StatusUpdate::Saved { path } => Some(path),
                crate::protocol::StatusUpdate::Done { checkpoint_dir, .. } => Some(checkpoint_dir),
                _ => None,
            };
            let mut note = None;
            if let Some(path) = path {
                match attempt_path(job_id, path) {
                    AttemptPath::Committed(committed) => *path = committed,
                    AttemptPath::Uncommitted { committed_dir } => {
                        note = Some(format!(
                            "attempt directory; it becomes {committed_dir} when the stage commits"
                        ));
                    }
                    AttemptPath::Other => {}
                }
            }
            notes.push(note);
            p
        })
        .collect();
    (progress, notes)
}

pub(super) fn run_log(id_query: &str, tail: usize, json: bool) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    let progress = jobs::read_progress(&id)?;
    if progress.is_empty() {
        eprintln!("{}", empty_log_hint(&id)?);
        return Ok(());
    }
    let (progress, notes) = commit_attempt_paths(&id, progress);
    if json {
        // Status stream as JSON lines (one StatusUpdate per line), tail-trimmed
        // like the rendered view. An update a stage forwarded also names it
        // (`node_idx`, `stage_name`).
        let start = if tail == 0 {
            0
        } else {
            progress.len().saturating_sub(tail)
        };
        for p in &progress[start..] {
            let mut line =
                serde_json::to_value(&p.update).map_err(|e| anyhow!("serialize status: {e}"))?;
            if let (Some((idx, stage)), Some(object)) = (&p.source, line.as_object_mut()) {
                object.insert("node_idx".into(), serde_json::json!(idx));
                object.insert("stage_name".into(), serde_json::json!(stage));
            }
            println!("{line}");
        }
        return Ok(());
    }
    let rendered = render_progress(&progress, &notes);
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

#[cfg(test)]
mod progress_render_tests {
    use super::{job_recipe, render_progress};
    use crate::jobs::{JobState, JobSummary, Progress};
    use crate::protocol::StatusUpdate;

    fn step(source: Option<(u32, &str)>, step: u32, loss: f32) -> Progress {
        Progress {
            source: source.map(|(idx, stage)| (idx, stage.to_string())),
            update: StatusUpdate::Step {
                step,
                total: 2,
                loss,
                lr: 0.001,
                vram_mb: 0,
            },
        }
    }

    /// One trainer stage renders as before; the interleaved progress of two
    /// (the trials of an HPO run) says which stage each line came from.
    #[test]
    fn interleaved_progress_names_its_stage() {
        let one = render_progress(&[step(Some((0, "train")), 1, 2.0), step(None, 2, 1.0)], &[]);
        assert_eq!(
            one,
            "step 1/2  loss=2.0000  lr=1.00e-3  vram=0MB\nstep 2/2  loss=1.0000  lr=1.00e-3  vram=0MB\n"
        );
        let two = render_progress(
            &[
                step(Some((1, "train")), 1, 2.0),
                step(Some((2, "train")), 1, 3.0),
            ],
            &[],
        );
        assert_eq!(
            two,
            "[1 train] step 1/2  loss=2.0000  lr=1.00e-3  vram=0MB\n\
             [2 train] step 1/2  loss=3.0000  lr=1.00e-3  vram=0MB\n"
        );
    }

    /// E2E follow-up: `blut log` showed the `.tmp-…` attempt directory a
    /// trainer saved into, which is renamed when the stage commits. A
    /// committed attempt's paths name the committed directory; a path still
    /// in an attempt directory says so.
    #[test]
    fn saved_paths_follow_the_committed_stage_directory() {
        use crate::protocol::StatusUpdate;
        use std::path::PathBuf;
        let td = tempfile::tempdir().unwrap();
        let id = "20261030-000000-000000002";
        let stages = td.path().join(id).join("stages");
        let tmp = stages.join(".tmp-1-train-0a1b2c");
        let saved = |p: PathBuf| Progress {
            source: Some((1, "train".into())),
            update: StatusUpdate::Saved { path: p },
        };
        let done = |p: PathBuf| Progress {
            source: Some((1, "train".into())),
            update: StatusUpdate::Done {
                final_loss: 0.5,
                checkpoint_dir: p,
            },
        };
        let elsewhere = PathBuf::from("/data/ckpt");
        let progress = || {
            vec![
                saved(tmp.join("ckpt/step-10")),
                done(tmp.join("ckpt")),
                saved(elsewhere.clone()),
            ]
        };

        // Running: the attempt directory exists; the path is kept and noted.
        std::fs::create_dir_all(tmp.join("ckpt")).unwrap();
        let (running, notes) = super::commit_attempt_paths(id, progress());
        assert_eq!(running, progress());
        let rendered = render_progress(&running, &notes);
        assert!(
            rendered
                .lines()
                .next()
                .unwrap()
                .ends_with("(attempt directory; it becomes stages/1-train when the stage commits)"),
            "{rendered}"
        );
        assert_eq!(notes[2], None, "a path outside the job is left alone");

        // Committed: the attempt directory was renamed.
        std::fs::rename(&tmp, stages.join("1-train")).unwrap();
        let (committed, notes) = super::commit_attempt_paths(id, progress());
        assert_eq!(
            committed,
            vec![
                saved(stages.join("1-train/ckpt/step-10")),
                done(stages.join("1-train/ckpt")),
                saved(elsewhere.clone()),
            ]
        );
        assert!(notes.iter().all(Option::is_none), "{notes:?}");
        assert!(!render_progress(&committed, &notes).contains(".tmp-"));

        // Another job's attempt directory is not this job's.
        assert_eq!(
            super::attempt_path("other-job", &stages.join(".tmp-1-train-0a1b2c/x")),
            super::AttemptPath::Other
        );
    }

    /// The `jobs` column showed only a legacy single-trainer output name, so
    /// it read `-` for every job a recipe ran.
    #[test]
    fn a_jobs_row_names_the_recipe_the_job_ran() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var_os("LAMU_TRAIN_JOBS_DIR");
        // SAFETY: TEST_ENV_LOCK serializes env mutation; restored below.
        unsafe { std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path()) };
        let summary = |id: &str, output_name: Option<&str>| JobSummary {
            id: id.into(),
            state: JobState::Done,
            pid: None,
            output_name: output_name.map(str::to_string),
            last_loss: None,
            last_step: None,
            final_loss: None,
        };
        let marked = td.path().join("20261030-000000-000000001");
        std::fs::create_dir_all(&marked).unwrap();
        super::RecipeMarker {
            name: "registry_recipe".into(),
            args: serde_json::json!({}),
            source_args: None,
            resolved_handles: Default::default(),
            launcher: None,
        }
        .write_to(&marked)
        .unwrap();
        let planned = td.path().join("20261030-000000-000000002");
        std::fs::create_dir_all(&planned).unwrap();
        crate::framework::PlanGraph {
            name: "registry://plan@prod".into(),
            nodes: vec![],
            edges: vec![],
            condition_gates: vec![],
        }
        .write_to(&planned)
        .unwrap();
        let got = [
            job_recipe(&summary("20261030-000000-000000001", None)),
            job_recipe(&summary("20261030-000000-000000002", None)),
            job_recipe(&summary("20261030-000000-000000003", Some("legacy-out"))),
            job_recipe(&summary("20261030-000000-000000004", None)),
        ];
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
        assert_eq!(
            got,
            [
                Some("registry_recipe".to_string()),
                Some("registry://plan@prod".to_string()),
                Some("legacy-out".to_string()),
                None,
            ]
        );
    }
}

#[cfg(test)]
mod empty_log_hint_tests {
    use super::empty_log_hint;

    /// A job with only stage events printed an empty log; the hint must point
    /// at the command that shows them.
    #[test]
    fn a_stage_event_only_log_points_at_dag() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var_os("LAMU_TRAIN_JOBS_DIR");
        // SAFETY: TEST_ENV_LOCK serializes env mutation; restored below.
        unsafe { std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path()) };
        let id = "20261020-000000-000000010";
        let dir = td.path().join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let empty = empty_log_hint(id).unwrap();
        std::fs::write(
            dir.join("status.jsonl"),
            "{\"kind\":\"stage_begin\",\"node_idx\":0,\"stage_name\":\"s\",\"input_hash\":\"x\"}\n",
        )
        .unwrap();
        let events = empty_log_hint(id).unwrap();
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
        assert!(empty.contains("no log entries"), "{empty}");
        assert!(
            events.contains("1 stage event") && events.contains("blut dag"),
            "{events}"
        );
    }
}
