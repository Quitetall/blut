// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut partition` — partition-set define/list/status/backfill.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

use super::*;

#[derive(Subcommand, Debug)]
pub(super) enum PartitionCommand {
    /// Declare + persist a partition set: `define <recipe> <name> --dim
    /// corpus=dataset_a,corpus_x --dim fold=0,1,2`.
    Define {
        recipe: String,
        name: String,
        /// One axis per flag: `--dim axis=v1,v2,v3` (repeatable).
        #[arg(long = "dim", value_name = "AXIS=v1,v2")]
        dim: Vec<String>,
    },
    /// List all defined partition sets.
    List,
    /// Per-cell materialization status (done / pending) for a set.
    Status { recipe: String, name: String },
    /// Run the recipe for every NOT-yet-materialized cell (`--force` = all),
    /// recording per-cell status. Cells run sequentially.
    Backfill {
        recipe: String,
        name: String,
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Base recipe args (the fixed part; each cell overlays its axis values).
        #[arg(long, default_value = "{}")]
        args: String,
        #[arg(long, default_value = "local")]
        launcher: String,
    },
}

/// Apply a partition cell's `axis=value` overrides onto base recipe args:
/// `args[axis] = <scalar>` (int / float / bool, else the verbatim string).
pub(super) fn apply_cell_overrides(
    base: &serde_json::Value,
    overrides: &[String],
) -> serde_json::Value {
    let mut obj = base.as_object().cloned().unwrap_or_default();
    for ov in overrides {
        if let Some((k, v)) = ov.split_once('=') {
            let val = if let Ok(i) = v.parse::<i64>() {
                serde_json::json!(i)
            } else if let Ok(f) = v.parse::<f64>() {
                // Require a digit and a finite value, mirroring hydra's
                // parse_override_value: bare `nan`/`inf` (which f64::from_str
                // accepts) must stay strings — `json!(NAN)` silently emits
                // JSON null, corrupting the cell's args.
                if v.bytes().any(|b| b.is_ascii_digit()) && f.is_finite() {
                    serde_json::json!(f)
                } else {
                    serde_json::json!(v)
                }
            } else if let Ok(b) = v.parse::<bool>() {
                serde_json::json!(b)
            } else {
                serde_json::json!(v)
            };
            obj.insert(k.to_string(), val);
        }
    }
    serde_json::Value::Object(obj)
}

pub(super) async fn run_partition(
    reg: &crate::framework::Registry,
    cmd: PartitionCommand,
) -> Result<()> {
    use crate::config::partition::{PartitionDim, PartitionSet, PartitionStatus};
    match cmd {
        PartitionCommand::Define { recipe, name, dim } => {
            if reg.find(&recipe).is_none() {
                return Err(anyhow!("recipe '{recipe}' not in catalog"));
            }
            let dims: Vec<PartitionDim> = dim
                .iter()
                .map(|d| {
                    let (axis, vals) = d
                        .split_once('=')
                        .ok_or_else(|| anyhow!("--dim '{d}' must be axis=v1,v2"))?;
                    let values: Vec<String> = vals
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect();
                    Ok::<_, anyhow::Error>(PartitionDim {
                        axis: axis.to_string(),
                        values,
                    })
                })
                .collect::<Result<_>>()?;
            let set = PartitionSet { name, recipe, dims };
            let cells = set.validate().map_err(|e| anyhow!("{e}"))?;
            let path = set.save().map_err(|e| anyhow!("{e}"))?;
            println!(
                "defined partition '{}/{}' — {cells} cells → {}",
                set.recipe,
                set.name,
                path.display()
            );
        }
        PartitionCommand::List => {
            let sets = PartitionSet::list().map_err(|e| anyhow!("{e}"))?;
            if sets.is_empty() {
                println!("(no partition sets defined)");
            }
            for (recipe, name) in sets {
                println!("{recipe}/{name}");
            }
        }
        PartitionCommand::Status { recipe, name } => {
            let set = PartitionSet::load(&recipe, &name).map_err(|e| anyhow!("{e}"))?;
            let done = set.statuses().map_err(|e| anyhow!("{e}"))?;
            let cells = set.cells();
            let n_done = cells
                .iter()
                .filter(|c| {
                    done.get(&c.key)
                        .is_some_and(PartitionStatus::is_materialized)
                })
                .count();
            println!("{recipe}/{name} — {n_done}/{} materialized", cells.len());
            for c in &cells {
                let st = match done.get(&c.key) {
                    Some(s) if s.is_materialized() => format!("done (job {})", s.job_id),
                    Some(s) => format!("{} (job {})", s.outcome, s.job_id),
                    None => "pending".to_string(),
                };
                println!("  {:<32} {st}", c.key);
            }
        }
        PartitionCommand::Backfill {
            recipe,
            name,
            force,
            args,
            launcher,
        } => {
            let set = PartitionSet::load(&recipe, &name).map_err(|e| anyhow!("{e}"))?;
            let base: serde_json::Value = serde_json::from_str(&args)
                .map_err(|e| anyhow!("--args is not valid JSON: {e}"))?;
            let launch_target: crate::config::launcher::LaunchTarget =
                launcher.parse().map_err(|e| anyhow!("{e}"))?;
            let targets = set.backfill_targets(force).map_err(|e| anyhow!("{e}"))?;
            if targets.is_empty() {
                println!("{recipe}/{name}: nothing to backfill (all cells materialized)");
                return Ok(());
            }
            // Phase-G PARALLEL scheduler: round-robin cells across the
            // launcher's device set, then run each device's cells SEQUENTIALLY
            // (its per-device lock would REJECT a second concurrent cell — so
            // never start two cells on one GPU), with the devices running
            // CONCURRENTLY. Result: at most one cell per GPU at a time, up to
            // `n_dev` cells in flight. capacity=1 → one chain → sequential
            // (byte-identical to the old loop). record_status appends atomically,
            // so concurrent writes from different device-chains are safe.
            //
            // NB the broker RAM-admission gate runs PER cell against LIVE free
            // RAM; two cells launching ~simultaneously could both pass it on the
            // SAME snapshot before either's usage registers. Cross-cell RAM
            // coordination is now DONE (not a future slice): a SHARED box-fit
            // RAM semaphore (`mem_sem` below) bounds the SUM of concurrently
            // SCHEDULED cells to the box, mirroring the `ParallelExecutor`'s
            // per-node memory budget. The per-cell gate stays in force
            // (defense-in-depth: it handles live free RAM incl. non-scheduler
            // consumers). On 1 GPU there's no concurrency, so the semaphore is
            // acquired/released serially — byte-identical to the old loop.
            // Concurrency is bounded by the launcher's CAPACITY: `device_set()`
            // is `0..Launcher::capacity()` by default (see
            // `config::launcher::Launcher::{capacity,device_set}`), with
            // `$BLUT_SCHED_DEVICES` layered on top ONLY as a SUBSET override.
            // So `n_dev == launcher.capacity()` in the common (no-override)
            // case, and a subset otherwise — never more than capacity.
            let launcher = crate::config::launcher::launcher_for(launch_target);
            let devices = launcher.device_set();
            if devices.is_empty() {
                return Err(anyhow!(
                    "launcher for {launch_target:?} reports no devices — cannot backfill"
                ));
            }
            // At most `launcher.capacity()` cells run concurrently; with a
            // `$BLUT_SCHED_DEVICES` subset, fewer. `device_set().len()` IS that
            // bound (it can only narrow `0..capacity()`, never widen it).
            debug_assert!(devices.len() <= launcher.capacity().max(1));
            let n_dev = devices.len();
            eprintln!(
                "backfill {recipe}/{name}: {} cell(s) across {} device(s) {devices:?}",
                targets.len(),
                n_dev
            );
            // Bucket cells per device (round-robin), preserving cell order.
            let mut per_device: Vec<Vec<_>> = (0..n_dev).map(|_| Vec::new()).collect();
            for (i, cell) in targets.into_iter().enumerate() {
                per_device[i % n_dev].push(cell);
            }
            // Cross-cell RAM admission (never-OOM): size the box-fit budget ONCE
            // (MemTotal − floor, same source the executor uses) and share one
            // semaphore across all device-chains. Each cell acquires its
            // (clamped) footprint in GiB permits for its whole run, so the SUM
            // of concurrent cells can't exceed the box. `0` ⇒ probe failed ⇒
            // ungated (per-cell broker gate alone), the pre-slice behaviour.
            let budget_gib = scheduler_box_fit_budget_gib().unwrap_or(0);
            let mem_sem =
                std::sync::Arc::new(tokio::sync::Semaphore::new(budget_gib.max(1) as usize));
            let (set, recipe, base) = (&set, &recipe, &base);
            let chains = per_device.into_iter().enumerate().map(|(d, cells)| {
                let dev = devices[d];
                let mem_sem = mem_sem.clone();
                async move {
                    let (mut ok, mut failed) = (0usize, 0usize);
                    for cell in cells {
                        let cell_args = apply_cell_overrides(base, &cell.overrides);
                        eprintln!("[gpu {dev}] cell {}", cell.key);
                        // Per-cell footprint (RAM GiB) for the cross-cell gate:
                        // the SAME `recipe_footprint` the per-cell broker
                        // admission resolves on, rounded UP to whole GiB (never
                        // under-bill). Clamp+acquire happens in `gated_cell_run`.
                        let footprint = recipe_footprint(recipe, &cell_args);
                        let footprint_gib =
                            footprint.ram_bytes.div_ceil(crate::broker::footprint::GIB) as u32;
                        let cell_run = run_one_recipe(
                            reg,
                            recipe,
                            cell_args,
                            None,
                            false,
                            launch_target,
                            Some(dev),
                            false,
                            // ADR 0096: partition-cell tenant namespacing is a
                            // follow-up; cells run under the default namespace.
                            crate::tenant::Tenant::default(),
                        );
                        let (outcome, job_id) = match gated_cell_run(
                            mem_sem.clone(),
                            footprint_gib,
                            budget_gib,
                            cell_run,
                        )
                        .await
                        {
                            Ok(jid) => {
                                ok += 1;
                                ("done", jid)
                            }
                            Err(e) => {
                                eprintln!("[gpu {dev}] cell {} FAILED: {e}", cell.key);
                                failed += 1;
                                ("failed", String::new())
                            }
                        };
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(0);
                        if let Err(e) = set.record_status(&PartitionStatus {
                            key: cell.key.clone(),
                            job_id,
                            outcome: outcome.to_string(),
                            recorded_at: now,
                        }) {
                            // A lost status write would silently re-run a
                            // completed cell next backfill — warn, don't swallow.
                            eprintln!(
                                "warning: could not record status for cell {}: {e}",
                                cell.key
                            );
                        }
                    }
                    (ok, failed)
                }
            });
            let totals = futures::future::join_all(chains).await;
            let ok: usize = totals.iter().map(|(o, _)| o).sum();
            let failed: usize = totals.iter().map(|(_, f)| f).sum();
            eprintln!("backfill done — {ok} materialized, {failed} failed");
            if failed > 0 {
                return Err(anyhow!(
                    "{failed} cell(s) failed (re-run `partition backfill` to retry only those)"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod cell_override_and_truncate_tests {
    use super::{apply_cell_overrides, truncate_for_col};
    use serde_json::json;

    #[test]
    fn overrides_coerce_scalars() {
        let out = apply_cell_overrides(
            &json!({}),
            &[
                "a=3".into(),
                "b=2.5".into(),
                "c=true".into(),
                "d=hello".into(),
            ],
        );
        assert_eq!(out, json!({"a": 3, "b": 2.5, "c": true, "d": "hello"}));
    }

    #[test]
    fn bare_nan_inf_stay_strings() {
        // f64::from_str accepts these, but json!(NAN) emits null — a silent
        // corruption of the cell's args. They must survive as strings.
        let out = apply_cell_overrides(
            &json!({}),
            &[
                "a=nan".into(),
                "b=inf".into(),
                "c=-inf".into(),
                "d=infinity".into(),
            ],
        );
        assert_eq!(
            out,
            json!({"a": "nan", "b": "inf", "c": "-inf", "d": "infinity"})
        );
    }

    #[test]
    fn overflowing_exponent_stays_string() {
        // "1e999" parses to +inf: has digits but non-finite → string, not null.
        let out = apply_cell_overrides(&json!({}), &["a=1e999".into()]);
        assert_eq!(out, json!({"a": "1e999"}));
    }

    #[test]
    fn truncate_is_char_correct() {
        // 5 chars, 15 bytes — fits a 5-char column and must NOT be cut.
        assert_eq!(truncate_for_col("ααβββ", 5), "ααβββ");
        assert_eq!(truncate_for_col("ααβββ", 4), "ααβ…");
        assert_eq!(truncate_for_col("ascii", 5), "ascii");
        assert_eq!(truncate_for_col("ascii!", 5), "asci…");
    }
}
