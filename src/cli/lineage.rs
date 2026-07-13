// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut lineage` — provenance show/freshness/trace/reindex.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

use super::*;

#[derive(Subcommand, Debug)]
pub(super) enum LineageCommand {
    /// Show a job's ingredient lineage (input→output hashes, cache hits).
    Show {
        /// Job id (or unique prefix).
        id: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Trace a checkpoint's full UPSTREAM provenance by content-hash prefix:
    /// the git SHA, hardware, and input-hash chain that produced it.
    Trace {
        /// Output content-hash prefix (≥ 6 hex chars recommended).
        hash: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Provenance GRAPH (ADR 0099): the full transitive upstream DAG of an
    /// artifact (every input that produced it, recursively). Emits DOT by
    /// default; `--json` emits typed rows. Clinical/`restricted`-tenant nodes are
    /// fail-closed excluded from the export (ADR 0061).
    Graph {
        /// Output content-hash (full, lowercased — the leaf artifact).
        hash: String,
        /// Emit typed JSON rows instead of Graphviz DOT.
        #[arg(long)]
        json: bool,
    },
    /// Model CARD (ADR 0099): a deterministic, content-addressed card for a
    /// model artifact — its data sources, recipe/config, metrics, and gate
    /// outcome. Rebuilding on the same lineage yields a byte-identical
    /// `card_hash`. Clinical/`restricted` data is fail-closed excluded (export).
    Card {
        /// The model artifact's content hash (full, lowercased).
        hash: String,
        /// Emit the typed JSON card instead of the human summary.
        #[arg(long)]
        json: bool,
    },
    /// Run DIFF (ADR 0099): the symmetric difference of two runs — differing
    /// recipe, config fingerprint, input hashes, args, gate outcome, and
    /// headline metrics.
    Diff {
        /// First run (job id or unique prefix).
        run_a: String,
        /// Second run (job id or unique prefix).
        run_b: String,
        /// Emit the typed JSON diff instead of the human summary.
        #[arg(long)]
        json: bool,
    },
    /// Rebuild the lineage index from the job dirs (the DB is a derived index —
    /// safe to delete + reindex).
    Reindex,
    /// FRESHNESS (G): is a job's output stale? Reports code-drift (the run's
    /// git SHA vs current HEAD — a STALE result means a re-run would
    /// re-execute, since the cache key includes code_sha). With
    /// `--data-version <v>`, also flags any produced artifact whose recorded
    /// `data_version` (in its metadata `extra`) differs from `<v>`.
    Freshness {
        /// Job id (or unique prefix).
        id: String,
        /// Current data/manifest version to check artifact `extra.data_version`
        /// against (optional — code-drift is always reported).
        #[arg(long)]
        data_version: Option<String>,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
}

pub(super) fn run_lineage(id_query: &str, json: bool) -> Result<()> {
    let nodes = crate::framework::lineage::job_lineage(id_query).map_err(|e| anyhow!("{e}"))?;
    if json {
        emit_json(&nodes)?;
        return Ok(());
    }
    if nodes.is_empty() {
        println!("no ingredient lineage (job has no framework status events).");
        return Ok(());
    }
    for n in &nodes {
        let inp = n.input_hash.as_deref().unwrap_or("-");
        let out = n.output_hash.as_deref().unwrap_or("-");
        let short = |h: &str| h.chars().take(12).collect::<String>();
        if n.cached {
            println!(
                "  {:>2} {:<28} [CACHE HIT {}]",
                n.node_idx,
                n.stage,
                short(out)
            );
        } else {
            let took = n
                .elapsed
                .map(|e| format!("{e:?}"))
                .unwrap_or_else(|| "-".into());
            println!(
                "  {:>2} {:<28} in={} → out={}  {}",
                n.node_idx,
                n.stage,
                short(inp),
                short(out),
                took
            );
        }
    }
    Ok(())
}

/// ADR 0099: emit an artifact's transitive UPSTREAM provenance graph — DOT by
/// default (pipe to `dot -Tsvg` or a sidecar), typed JSON with `--json`. This is
/// an EXPORT, so clinical/`restricted`-tenant nodes are fail-closed excluded
/// (ADR 0061) — a graph that is entirely restricted comes back empty.
pub(super) fn run_lineage_graph(hash: &str, json: bool) -> Result<()> {
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let graph = db.graph_upstream(hash, true).map_err(|e| anyhow!("{e}"))?;
    if graph.nodes.is_empty() {
        return Err(anyhow!(
            "no exportable lineage for artifact {hash} (all-restricted graph, or unknown hash)"
        ));
    }
    // An unknown hash yields a single node with no indexed artifact metadata and
    // no edges — surface that as a clear error, not a degenerate one-node graph.
    if graph.nodes.len() == 1 && graph.edges.is_empty() && graph.nodes[0].stage_name.is_none() {
        return Err(anyhow!(
            "unknown artifact {hash} — not in the lineage index (typo, or run `blut lineage reindex`)"
        ));
    }
    if json {
        emit_json(&graph)?;
    } else {
        print!("{}", graph.to_dot());
    }
    Ok(())
}

/// ADR 0099: emit a model artifact's deterministic content-addressed card.
/// An EXPORT — clinical/`restricted` data is fail-closed excluded.
pub(super) fn run_lineage_card(hash: &str, json: bool) -> Result<()> {
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let card = db
        .model_card(hash, true)
        .map_err(|e| anyhow!("{e}"))?
        .ok_or_else(|| {
            anyhow!("no exportable card for {hash} (unknown hash, or a restricted model)")
        })?;
    if json {
        emit_json(&card)?;
    } else {
        print!("{}", card.render());
    }
    Ok(())
}

/// ADR 0099: report the symmetric difference of two runs.
pub(super) fn run_lineage_diff(a_query: &str, b_query: &str, json: bool) -> Result<()> {
    let a = crate::jobs::resolve_job_id(a_query).map_err(|e| anyhow!("{e}"))?;
    let b = crate::jobs::resolve_job_id(b_query).map_err(|e| anyhow!("{e}"))?;
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let diff = db.run_diff(&a, &b).map_err(|e| anyhow!("{e}"))?;
    if json {
        emit_json(&diff)?;
        return Ok(());
    }
    print!("{}", render_lineage_diff(&a, &b, &diff));
    Ok(())
}

fn render_lineage_diff(a: &str, b: &str, diff: &crate::lineage_report::RunDiff) -> String {
    use std::fmt::Write;

    if diff.is_empty() {
        return format!(
            "runs {a} and {b} are identical across recipe/config/inputs/args/gate/metrics.\n"
        );
    }
    let opt = |o: &Option<String>| o.as_deref().unwrap_or("—").to_string();
    let optf = |o: &Option<f64>| o.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
    let opt_json = |o: &Option<serde_json::Value>| {
        o.as_ref()
            .map(|value| serde_json::to_string(value).expect("JSON Value always serializes"))
            .unwrap_or_else(|| "—".into())
    };
    let mut out = String::new();
    writeln!(&mut out, "diff {a} ↔ {b}:").expect("write to String");
    if let Some((x, y)) = &diff.recipe {
        writeln!(&mut out, "  recipe    : {} → {}", opt(x), opt(y)).expect("write to String");
    }
    if let Some((x, y)) = &diff.config_fingerprint {
        writeln!(&mut out, "  config_fp : {} → {}", opt(x), opt(y)).expect("write to String");
    }
    if let Some((x, y)) = &diff.input_hashes {
        writeln!(&mut out, "  inputs    : {} → {}", x.join(","), y.join(","))
            .expect("write to String");
    }
    for delta in &diff.arg_deltas {
        writeln!(
            &mut out,
            "  arg {:<11}: {} → {}",
            delta.path,
            opt_json(&delta.a),
            opt_json(&delta.b)
        )
        .expect("write to String");
    }
    if let Some((x, y)) = &diff.gate_outcome {
        writeln!(&mut out, "  gate      : {} → {}", opt(x), opt(y)).expect("write to String");
    }
    for (m, x, y) in &diff.metric_deltas {
        writeln!(&mut out, "  {m:<16}: {} → {}", optf(x), optf(y)).expect("write to String");
    }
    out
}

pub(super) fn run_lineage_cmd(cmd: LineageCommand) -> Result<()> {
    match cmd {
        LineageCommand::Show { id, json } => run_lineage(&id, json),
        LineageCommand::Trace { hash, json } => run_lineage_trace(&hash, json),
        LineageCommand::Graph { hash, json } => run_lineage_graph(&hash, json),
        LineageCommand::Card { hash, json } => run_lineage_card(&hash, json),
        LineageCommand::Diff { run_a, run_b, json } => run_lineage_diff(&run_a, &run_b, json),
        LineageCommand::Reindex => run_lineage_reindex(),
        LineageCommand::Freshness {
            id,
            data_version,
            json,
        } => run_lineage_freshness(&id, data_version, json),
    }
}

/// FRESHNESS (G): report whether a job's output is stale. Code-drift is
/// always reported (the run's git SHA vs HEAD — STALE ⇒ a re-run
/// re-executes because the cache key includes code_sha). With
/// `--data-version`, also flag artifacts whose recorded `extra.data_version`
/// differs from the supplied current value.
pub(super) fn run_lineage_freshness(
    id_query: &str,
    data_version: Option<String>,
    json: bool,
) -> Result<()> {
    let job_id = crate::jobs::resolve_job_id(id_query).map_err(|e| anyhow!("{e}"))?;
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let code = db.code_freshness(&job_id).map_err(|e| anyhow!("{e}"))?;
    let short = |h: &str| h.get(..12).unwrap_or(h).to_string();

    // Optional data-version freshness: compare each produced artifact's
    // recorded `extra.data_version` against the supplied current value.
    let mut data_stale: Vec<(String, String)> = Vec::new(); // (stage, recorded)
    let mut data_checked = 0usize;
    if let Some(current) = data_version.as_deref() {
        for rec in crate::framework::lineage::scan_artifacts(&job_id).map_err(|e| anyhow!("{e}"))? {
            if let Some(v) = rec.meta.extra.get("data_version").and_then(|v| v.as_str()) {
                data_checked += 1;
                if v != current {
                    data_stale.push((
                        rec.meta.produced_by_stage.clone().unwrap_or_default(),
                        v.to_string(),
                    ));
                }
            }
        }
    }

    if json {
        let v = serde_json::json!({
            "job_id": job_id,
            "code": code,
            "data_version_current": data_version,
            "data_artifacts_checked": data_checked,
            "data_stale": data_stale.iter()
                .map(|(s, r)| serde_json::json!({ "stage": s, "recorded": r }))
                .collect::<Vec<_>>(),
        });
        emit_json(&v)?;
        return Ok(());
    }

    match &code {
        crate::lineage_db::CodeFreshness::Fresh { git_sha } => {
            println!(
                "job {job_id}: code FRESH (built at HEAD {})",
                short(git_sha)
            )
        }
        crate::lineage_db::CodeFreshness::Stale { built_sha, head } => println!(
            "job {job_id}: code STALE (built at {}, HEAD is {} — a re-run re-executes)",
            short(built_sha),
            short(head)
        ),
        crate::lineage_db::CodeFreshness::Unknown => {
            println!("job {job_id}: code UNKNOWN (no recorded git SHA to compare)")
        }
    }
    if let Some(current) = &data_version {
        if data_stale.is_empty() {
            println!("data: FRESH ({data_checked} artifact(s) at data_version {current})");
        } else {
            for (stage, recorded) in &data_stale {
                println!(
                    "data: STALE — {stage} built from data_version {recorded}, current is {current}"
                );
            }
        }
    }
    Ok(())
}

/// Reproducibility query: the full upstream provenance chain that produced a
/// checkpoint, by content-hash prefix, from the LineageDB.
pub(super) fn run_lineage_trace(hash: &str, json: bool) -> Result<()> {
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let matches = db.find_artifacts(hash).map_err(|e| anyhow!("{e}"))?;
    let target = match matches.first() {
        None => {
            return Err(anyhow!(
                "no indexed artifact with content-hash prefix '{hash}' \
                 (older runs predate the index — `blut lineage reindex`)"
            ));
        }
        Some(a) => {
            if matches.len() > 1 {
                eprintln!(
                    "note: {} artifacts match '{hash}'; tracing the most recent",
                    matches.len()
                );
            }
            a.content_hash.clone()
        }
    };
    let chain = db.trace(&target).map_err(|e| anyhow!("{e}"))?;
    if json {
        emit_json(&chain)?;
        return Ok(());
    }
    let short = |h: &str| h.get(..16).unwrap_or(h).to_string();
    println!(
        "provenance trace for {} — {} hop(s), upstream:",
        short(&target),
        chain.len()
    );
    for (i, step) in chain.iter().enumerate() {
        let a = &step.artifact;
        println!(
            "  [{i}] {} :: {} = {}",
            a.stage_name,
            a.kind,
            short(&a.content_hash)
        );
        if let Some(run) = &step.run {
            // `?` for unrecorded hardware — NOT `0` (which reads as "zero RAM").
            let ram = run
                .ram_gib
                .map(|g| format!("{g}G"))
                .unwrap_or_else(|| "?".into());
            let vram = run
                .vram_mib
                .map(|m| format!("{m}M"))
                .unwrap_or_else(|| "?".into());
            println!(
                "      job={} recipe={} git={} ram={ram} vram={vram} outcome={}",
                run.job_id,
                run.recipe,
                run.git_sha.as_deref().unwrap_or("?"),
                run.outcome.as_deref().unwrap_or("?"),
            );
        }
    }
    Ok(())
}

/// Rebuild the lineage index from the job dirs. The DB is a DERIVED index, so
/// this is always safe (idempotent ingest); use it after deleting `lineage.db`
/// or to backfill runs that predate the index.
pub(super) fn run_lineage_reindex() -> Result<()> {
    let jobs_root = crate::paths::jobs_dir()?;
    let (mut indexed, mut skipped) = (0u32, 0u32);
    let rd = std::fs::read_dir(&jobs_root)
        .with_context(|| format!("read jobs dir {}", jobs_root.display()))?;
    {
        for e in rd.flatten() {
            let Some(job_id) = e.file_name().to_str().map(String::from) else {
                continue;
            };
            // Recipe identity comes from the run marker; legacy/bare-spawn jobs
            // have none → skip (nothing to attribute the run to).
            let Ok(marker) = RecipeMarker::read_from(&e.path()) else {
                skipped += 1;
                continue;
            };
            let outcome = crate::jobs::read_state(&job_id)
                .map(|s| format!("{s:?}").to_lowercase())
                .unwrap_or_else(|_| "unknown".into());
            match crate::lineage_db::ingest_job(&job_id, &marker.name, &outcome) {
                Ok(()) => indexed += 1,
                Err(err) => {
                    tracing::warn!("reindex {job_id}: {err}");
                    skipped += 1;
                }
            }
        }
    }
    eprintln!("reindexed {indexed} run(s), skipped {skipped}");
    Ok(())
}

#[cfg(test)]
mod lineage_diff_render_tests {
    use super::render_lineage_diff;

    #[test]
    fn human_diff_renders_input_and_argument_deltas() {
        let diff = crate::lineage_report::RunDiff {
            run_a: "a".into(),
            run_b: "b".into(),
            recipe: None,
            config_fingerprint: None,
            input_hashes: Some((vec!["input-a".into()], vec!["input-b".into()])),
            arg_deltas: vec![crate::lineage_report::ArgDelta {
                path: "$/lr".into(),
                a: Some(serde_json::json!(0.01)),
                b: Some(serde_json::json!(0.02)),
            }],
            gate_outcome: None,
            metric_deltas: Vec::new(),
        };
        let rendered = render_lineage_diff("a", "b", &diff);
        assert!(rendered.contains("input-a → input-b"));
        assert!(rendered.contains("$/lr"));
        assert!(rendered.contains("0.01 → 0.02"));
    }
}
