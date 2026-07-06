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

pub(super) fn run_lineage_cmd(cmd: LineageCommand) -> Result<()> {
    match cmd {
        LineageCommand::Show { id, json } => run_lineage(&id, json),
        LineageCommand::Trace { hash, json } => run_lineage_trace(&hash, json),
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
