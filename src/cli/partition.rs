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
        #[arg(long = "dim", value_name = "AXIS=v1,v2", conflicts_with = "spec")]
        dim: Vec<String>,
        /// Typed PartitionSpec JSON (`time`, `categorical`, or `multi`).
        #[arg(long, value_name = "JSON", conflicts_with = "dim")]
        spec: Option<String>,
        /// Finite inclusive range required when `--spec` contains `time`.
        #[arg(long, value_name = "START:END", requires = "spec")]
        partitions: Option<String>,
    },
    /// List all defined partition sets.
    List,
    /// Per-cell lineage-derived status matrix for a set.
    Status {
        recipe: String,
        name: String,
        /// Base recipe args (the fixed part; each cell overlays its axis values).
        #[arg(long, default_value = "{}")]
        args: String,
        #[arg(long, default_value = "local")]
        launcher: String,
        #[arg(long, default_value = "default")]
        tenant: String,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Run the recipe for every NOT-yet-materialized cell (`--force` = all),
    /// recording per-cell status. Cells run sequentially.
    Backfill {
        recipe: String,
        name: String,
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Run only cells whose lineage row is absent.
        #[arg(long, default_value_t = false, conflicts_with_all = ["force", "stale"])]
        missing: bool,
        /// Run only materialized cells whose resolved upstream identity moved.
        #[arg(long, default_value_t = false, conflicts_with_all = ["force", "missing"])]
        stale: bool,
        /// Explicit cell keys/first-axis values, or an inclusive first-axis range.
        #[arg(long, conflicts_with_all = ["missing", "stale"])]
        partitions: Option<String>,
        /// Base recipe args (the fixed part; each cell overlays its axis values).
        #[arg(long, default_value = "{}")]
        args: String,
        #[arg(long, default_value = "local")]
        launcher: String,
        #[arg(long, default_value = "default")]
        tenant: String,
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

fn partition_matrix(
    reg: &crate::framework::Registry,
    set: &crate::config::partition::PartitionSet,
    base: &serde_json::Value,
    tenant: &crate::tenant::Tenant,
    launch_target: crate::config::launcher::LaunchTarget,
) -> Result<(
    std::collections::BTreeMap<String, crate::config::partition::CellStatus>,
    std::collections::BTreeMap<String, PartitionEvaluation>,
)> {
    use crate::config::partition::{CellStatus, partition_args_are_restricted};
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("lineage index: {e}"))?;
    let tenant_key = tenant.to_string();
    let mut indexed = db
        .partition_statuses(&tenant_key, &set.recipe, &set.name)
        .map_err(|e| anyhow!("partition status index: {e}"))?;

    // Lazy rebuild of v5 rows from the canonical append-only status log and
    // its referenced run. This keeps lineage.db disposable/reindexable.
    for legacy in set
        .statuses_for_tenant(&tenant_key)
        .map_err(|e| anyhow!("{e}"))?
        .into_values()
    {
        let already_current = indexed.get(&legacy.key).is_some_and(|row| {
            row.recorded_unix == legacy.recorded_at
                && row.job_id == legacy.job_id
                && row.outcome == legacy.outcome
                && row.input_fingerprint == legacy.input_fingerprint.as_deref().unwrap_or_default()
        });
        if already_current || legacy.job_id.is_empty() {
            continue;
        }
        let Some(run) = db
            .get_run(&legacy.job_id)
            .map_err(|e| anyhow!("partition lineage {}: {e}", legacy.job_id))?
        else {
            continue;
        };
        let Some(args_json) = run.args_json else {
            continue;
        };
        let row = crate::lineage_db::PartitionStatusRow {
            tenant: run.tenant,
            recipe: set.recipe.clone(),
            partition_set: set.name.clone(),
            partition_key: legacy.key.clone(),
            job_id: legacy.job_id,
            outcome: legacy.outcome,
            resolved_args_json: args_json,
            input_fingerprint: legacy.input_fingerprint.unwrap_or_default(),
            recorded_unix: legacy.recorded_at,
        };
        if row.tenant == tenant_key {
            db.reconcile_partition_status(&row)
                .map_err(|e| anyhow!("reindex partition status: {e}"))?;
            indexed.insert(row.partition_key.clone(), row);
        }
    }

    let legacy = set
        .statuses_for_tenant(&tenant_key)
        .map_err(|e| anyhow!("{e}"))?;
    let mut matrix = std::collections::BTreeMap::new();
    let mut evaluated_by_key = std::collections::BTreeMap::new();
    for cell in set.cells() {
        let source_args = apply_cell_overrides(base, &cell.overrides);
        let restricted = partition_args_are_restricted(&source_args, tenant);
        let resolved =
            crate::registry_args::resolve_recipe_args(source_args.clone(), tenant, launch_target)
                .and_then(|resolved| {
                    let recipe = reg.find(&set.recipe).ok_or_else(|| {
                        crate::error::TrainError::other(format!(
                            "recipe '{}' not in catalog",
                            set.recipe
                        ))
                    })?;
                    (recipe.compile_fn)(resolved)
                        .map(|plan| {
                            let compiled_args = plan.recipe_args().clone();
                            PartitionEvaluation {
                        input_fingerprint:
                            crate::config::partition::partition_input_fingerprint_with_execution(
                                &source_args,
                                &compiled_args,
                                &plan.execution_fingerprint(),
                            ),
                        compiled_args,
                    }
                        })
                        .map_err(|e| {
                            crate::error::TrainError::other(format!("recipe compile failed: {e}"))
                        })
                });
        let canonical = legacy.get(&cell.key);
        let status = if restricted {
            CellStatus::Restricted
        } else if canonical.is_none() {
            // The SQLite row is only an index. Without the canonical JSONL
            // record, a cell cannot be promoted to materialized.
            CellStatus::Missing
        } else if canonical.is_some_and(|status| !status.is_materialized()) {
            CellStatus::Failed
        } else if let Ok(current) = &resolved {
            let canonical = canonical.expect("checked Some above");
            let matching_row = indexed.get(&cell.key).filter(|row| {
                row.job_id == canonical.job_id
                    && row.outcome == canonical.outcome
                    && row.recorded_unix == canonical.recorded_at
                    && row.input_fingerprint
                        == canonical.input_fingerprint.as_deref().unwrap_or_default()
            });
            db.derive_partition_status(
                matching_row,
                &tenant_key,
                &set.recipe,
                &current.compiled_args,
                &current.input_fingerprint,
                false,
            )
            .map_err(|e| anyhow!("derive partition status {}: {e}", cell.key))?
        } else {
            // A previously materialized handle that no longer resolves is
            // stale by definition; the actual backfill surfaces the resolver
            // error if the operator selects it.
            CellStatus::Stale
        };
        if let Ok(evaluated) = resolved {
            evaluated_by_key.insert(cell.key.clone(), evaluated);
        }
        matrix.insert(cell.key, status);
    }
    Ok((matrix, evaluated_by_key))
}

#[derive(Clone, Debug)]
struct PartitionEvaluation {
    compiled_args: serde_json::Value,
    input_fingerprint: String,
}

pub(super) async fn run_partition(
    reg: &crate::framework::Registry,
    cmd: PartitionCommand,
) -> Result<()> {
    use crate::config::partition::{PartitionDim, PartitionSet, PartitionStatus};
    match cmd {
        PartitionCommand::Define {
            recipe,
            name,
            dim,
            spec,
            partitions,
        } => {
            if reg.find(&recipe).is_none() {
                return Err(anyhow!("recipe '{recipe}' not in catalog"));
            }
            let dims: Vec<PartitionDim> = if let Some(spec) = spec {
                let spec: crate::config::partition::PartitionSpec = serde_json::from_str(&spec)
                    .map_err(|e| anyhow!("--spec is not valid PartitionSpec JSON: {e}"))?;
                crate::config::partition::dims_from_spec(&spec, partitions.as_deref())
                    .map_err(|e| anyhow!("{e}"))?
            } else {
                dim.iter()
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
                    .collect::<Result<_>>()?
            };
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
        PartitionCommand::Status {
            recipe,
            name,
            args,
            launcher,
            tenant,
            json,
        } => {
            let set = PartitionSet::load(&recipe, &name).map_err(|e| anyhow!("{e}"))?;
            let base: serde_json::Value = serde_json::from_str(&args)
                .map_err(|e| anyhow!("--args is not valid JSON: {e}"))?;
            let launch_target: crate::config::launcher::LaunchTarget =
                launcher.parse().map_err(|e| anyhow!("{e}"))?;
            let tenant = crate::tenant::Tenant::parse(&tenant)
                .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
            let (matrix, _) = partition_matrix(reg, &set, &base, &tenant, launch_target)?;
            let cells = set.cells();
            let n_done = matrix
                .values()
                .filter(|&&status| status == crate::config::partition::CellStatus::Materialized)
                .count();
            if json {
                let rows: Vec<_> = cells
                    .iter()
                    .map(|cell| {
                        serde_json::json!({
                            "key": cell.key,
                            "status": matrix[&cell.key].as_str(),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "recipe": recipe,
                        "partition_set": name,
                        "tenant": tenant.to_string(),
                        "materialized": n_done,
                        "total": cells.len(),
                        "cells": rows,
                    }))?
                );
                return Ok(());
            }
            println!("{recipe}/{name} — {n_done}/{} materialized", cells.len());
            for c in &cells {
                println!("  {:<32} {}", c.key, matrix[&c.key].as_str());
            }
        }
        PartitionCommand::Backfill {
            recipe,
            name,
            force,
            missing,
            stale,
            partitions,
            args,
            launcher,
            tenant,
        } => {
            let set = PartitionSet::load(&recipe, &name).map_err(|e| anyhow!("{e}"))?;
            let base: serde_json::Value = serde_json::from_str(&args)
                .map_err(|e| anyhow!("--args is not valid JSON: {e}"))?;
            let launch_target: crate::config::launcher::LaunchTarget =
                launcher.parse().map_err(|e| anyhow!("{e}"))?;
            let tenant = crate::tenant::Tenant::parse(&tenant)
                .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
            let (matrix, evaluated_by_key) =
                partition_matrix(reg, &set, &base, &tenant, launch_target)?;
            let selector = if force {
                crate::config::partition::BackfillSelector::Force
            } else if missing {
                crate::config::partition::BackfillSelector::Missing
            } else if stale {
                crate::config::partition::BackfillSelector::Stale
            } else {
                crate::config::partition::BackfillSelector::Default
            };
            let all_cells = set.cells();
            let selected_cells = match partitions.as_deref() {
                Some(selection) => {
                    crate::config::partition::select_partition_cells(&all_cells, selection)
                        .map_err(|e| anyhow!("{e}"))?
                }
                None => all_cells,
            };
            let targets = crate::config::partition::select_backfill_targets(
                &selected_cells,
                &matrix,
                selector,
            );
            // Explicit/forced selection of a Restricted key is a refusal. An
            // unrelated Restricted key outside a safe selector remains visible
            // in the matrix but does not block permitted cells.
            let restriction_scope = if partitions.is_some() || force {
                &selected_cells
            } else {
                &targets
            };
            let restricted: Vec<_> = restriction_scope
                .iter()
                .filter(|cell| {
                    matrix.get(&cell.key) == Some(&crate::config::partition::CellStatus::Restricted)
                })
                .map(|cell| cell.key.as_str())
                .collect();
            if !restricted.is_empty() {
                return Err(anyhow!(
                    "partition backfill refused before launch: Restricted selected cell(s): {}",
                    restricted.join(", ")
                ));
            }
            if targets.is_empty() {
                println!("{recipe}/{name}: nothing selected for backfill");
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
            let (set, recipe, base, evaluated_by_key) = (&set, &recipe, &base, &evaluated_by_key);
            let chains = per_device.into_iter().enumerate().map(|(d, cells)| {
                let dev = devices[d];
                let mem_sem = mem_sem.clone();
                let tenant = tenant.clone();
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
                        let partition = cell.partition_key().map_err(|e| anyhow!("{e}"));
                        let cell_run = async {
                            run_one_partitioned_recipe(
                                reg,
                                recipe,
                                cell_args.clone(),
                                true,
                                launch_target,
                                Some(dev),
                                force || stale,
                                tenant.clone(),
                                partition?,
                            )
                            .await
                        };
                        let (outcome, job_id, evaluation) = match gated_cell_run(
                            mem_sem.clone(),
                            footprint_gib,
                            budget_gib,
                            cell_run,
                        )
                        .await
                        {
                            Ok((jid, compiled_args, input_fingerprint)) => {
                                ok += 1;
                                (
                                    "done",
                                    jid,
                                    PartitionEvaluation {
                                        compiled_args,
                                        input_fingerprint,
                                    },
                                )
                            }
                            Err(e) => {
                                eprintln!("[gpu {dev}] cell {} FAILED: {e}", cell.key);
                                failed += 1;
                                (
                                    "failed",
                                    String::new(),
                                    evaluated_by_key
                                        .get(&cell.key)
                                        .cloned()
                                        .unwrap_or_else(|| PartitionEvaluation {
                                            input_fingerprint:
                                                crate::config::partition::partition_input_fingerprint(
                                                    &cell_args,
                                                    &cell_args,
                                                ),
                                            compiled_args: cell_args.clone(),
                                        }),
                                )
                            }
                        };
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(0);
                        let canonical_status = PartitionStatus {
                            tenant: tenant.to_string(),
                            key: cell.key.clone(),
                            job_id: job_id.clone(),
                            outcome: outcome.to_string(),
                            recorded_at: now,
                            input_fingerprint: Some(evaluation.input_fingerprint.clone()),
                        };
                        if let Err(e) = set.record_status(&canonical_status) {
                            // A lost status write would silently re-run a
                            // completed cell next backfill — warn and do not
                            // create an index row with no canonical record.
                            eprintln!(
                                "warning: could not record status for cell {}: {e}",
                                cell.key
                            );
                            continue;
                        }
                        let row = crate::lineage_db::PartitionStatusRow {
                            tenant: tenant.to_string(),
                            recipe: recipe.to_string(),
                            partition_set: set.name.clone(),
                            partition_key: cell.key.clone(),
                            job_id,
                            outcome: outcome.to_string(),
                            resolved_args_json: serde_json::to_string(&evaluation.compiled_args)
                                .expect("resolved recipe args serialize"),
                            input_fingerprint: evaluation.input_fingerprint,
                            recorded_unix: now,
                        };
                        match crate::lineage_db::LineageDb::open()
                            .and_then(|db| db.record_partition_status(&row))
                        {
                            Ok(()) => {}
                            Err(e) => eprintln!(
                                "warning: could not index partition status for cell {}: {e}",
                                cell.key
                            ),
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
    use super::{Cli, Parser, apply_cell_overrides, truncate_for_col};
    use serde_json::json;

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct ToyPartitionArtifact {
        corpus: String,
    }

    impl crate::framework::Artifact for ToyPartitionArtifact {
        const KIND: &'static str = "test.partition";
        const SCHEMA: u32 = 1;

        fn content_hash(&self) -> crate::framework::ContentHash {
            crate::framework::ContentHash::of_bytes(self.corpus.as_bytes())
        }

        fn primary_path(&self) -> &std::path::Path {
            std::path::Path::new(".")
        }
    }

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    struct ToyPartitionArgs {
        corpus: String,
        #[serde(default)]
        restricted: bool,
    }

    static TOY_PARTITION_RUNS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    struct ToyPartitionStage;

    #[async_trait::async_trait]
    impl crate::framework::Stage for ToyPartitionStage {
        const NAME: &'static str = "toy_partition_stage";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [crate::framework::Resource] = &[crate::framework::Resource::Cpu];
        type Input = ();
        type Output = ToyPartitionArtifact;
        type Args = ToyPartitionArgs;

        async fn run(
            &self,
            _ctx: &crate::framework::StageContext,
            _input: (),
            args: &Self::Args,
        ) -> std::result::Result<Self::Output, crate::framework::StageError> {
            TOY_PARTITION_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ToyPartitionArtifact {
                corpus: args.corpus.clone(),
            })
        }
    }

    impl crate::framework::Compatible<crate::backends::LamuTrainerBackend> for ToyPartitionStage {}

    fn compile_toy_partition(
        raw: serde_json::Value,
    ) -> std::result::Result<crate::framework::plan::CompiledPlan, crate::framework::RecipeError>
    {
        let args: ToyPartitionArgs = serde_json::from_value(raw).map_err(|e| {
            crate::framework::RecipeError::InvalidArgs {
                field: None,
                message: e.to_string(),
            }
        })?;
        // Empty recipe provenance keeps this fixture on the 2-GiB light-job
        // admission path; the stage args still vary and the partition key is
        // what separates otherwise identical content-addressed runs.
        Ok(
            crate::framework::Plan::<(), crate::backends::LamuTrainerBackend>::new(
                "toy_partition",
                json!({}),
            )
            .start(ToyPartitionStage, args)
            .finish()
            .into_compiled(),
        )
    }

    static TOY_PARTITION_DEF: crate::recipes::recipe::RecipeDef =
        crate::recipes::recipe::RecipeDef {
            name: "toy_partition",
            description: "partition CLI fixture",
            backend_id: "lamu_trainer",
            category: crate::recipes::recipe::RecipeCategory::User,
            input_kinds: &[],
            output_kind: "test.partition",
            schedule: None,
            args_schema_fn: || crate::recipes::recipe::schema_of::<ToyPartitionArgs>(),
            compile_fn: compile_toy_partition,
        };

    struct ToyPartitionCookbook;

    impl crate::framework::Cookbook for ToyPartitionCookbook {
        fn name(&self) -> &'static str {
            "toy_partition"
        }

        fn recipes(&self) -> &'static [&'static crate::recipes::recipe::RecipeDef] {
            static RECIPES: &[&crate::recipes::recipe::RecipeDef] = &[&TOY_PARTITION_DEF];
            RECIPES
        }
    }

    struct EnvRestore(Vec<(String, Option<std::ffi::OsString>)>);

    impl EnvRestore {
        fn set(vars: &[(&str, std::path::PathBuf)]) -> Self {
            let old = vars
                .iter()
                .map(|(key, value)| {
                    let previous = std::env::var_os(key);
                    // SAFETY: the caller holds TEST_ENV_LOCK for this guard's
                    // complete lifetime.
                    unsafe { std::env::set_var(key, value) };
                    ((*key).to_string(), previous)
                })
                .collect();
            Self(old)
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (key, value) in self.0.drain(..).rev() {
                // SAFETY: TEST_ENV_LOCK still lives outside this guard.
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(key, value),
                        None => std::env::remove_var(key),
                    }
                }
            }
        }
    }

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

    #[test]
    fn partition_selector_flags_are_mutually_exclusive() {
        assert!(
            Cli::try_parse_from([
                "blut",
                "partition",
                "backfill",
                "recipe",
                "set",
                "--missing",
                "--stale",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "blut",
                "partition",
                "backfill",
                "recipe",
                "set",
                "--partitions",
                "a,b",
                "--stale",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "blut",
                "partition",
                "backfill",
                "recipe",
                "set",
                "--partitions",
                "a,b",
                "--force",
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "blut",
                "partition",
                "backfill",
                "recipe",
                "set",
                "--force",
                "--missing",
            ])
            .is_err()
        );
    }

    #[test]
    fn restricted_classification_is_per_cell_and_tenant_dominates() {
        use crate::config::partition::partition_args_are_restricted;
        let research = crate::tenant::Tenant::parse("research/dev").unwrap();
        let clinical = crate::tenant::Tenant::parse("clinical/prod").unwrap();
        assert!(!partition_args_are_restricted(
            &json!({"classification":"public"}),
            &research
        ));
        assert!(partition_args_are_restricted(
            &json!({"classification":"restricted"}),
            &research
        ));
        assert!(partition_args_are_restricted(
            &json!({"nested":{"restricted":true}}),
            &research
        ));
        assert!(partition_args_are_restricted(&json!({}), &clinical));
    }

    #[tokio::test]
    // The guard intentionally serializes process-global environment mutation
    // for the complete async CLI run; no production task takes this test lock.
    #[allow(clippy::await_holding_lock)]
    async fn partition_backfill_executes_three_cells_indexes_matrix_and_refuses_restricted() {
        use crate::config::partition::{CellStatus, PartitionDim, PartitionSet};
        use crate::framework::Registry;

        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let _env = EnvRestore::set(&[
            ("BLUT_PARTITIONS_DIR", temp.path().join("partitions")),
            ("LAMU_TRAIN_JOBS_DIR", temp.path().join("jobs")),
            ("LAMU_TRAIN_DATA_DIR", temp.path().join("data")),
            ("XDG_DATA_HOME", temp.path().join("xdg-data")),
            ("XDG_CACHE_HOME", temp.path().join("xdg-cache")),
            ("BLUT_SCHED_DEVICES", std::path::PathBuf::from("0")),
            ("CUDA_VISIBLE_DEVICES", std::path::PathBuf::new()),
            ("BLUT_NO_CONTAIN", std::path::PathBuf::from("1")),
        ]);
        TOY_PARTITION_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut reg = Registry::new();
        reg.register(Box::new(ToyPartitionCookbook));
        let set = PartitionSet {
            name: "three".into(),
            recipe: "toy_partition".into(),
            dims: vec![PartitionDim {
                axis: "corpus".into(),
                values: vec!["a".into(), "b".into(), "c".into()],
            }],
        };
        set.save().unwrap();

        super::run_partition(
            &reg,
            super::PartitionCommand::Backfill {
                recipe: set.recipe.clone(),
                name: set.name.clone(),
                force: false,
                missing: false,
                stale: false,
                partitions: None,
                args: "{}".into(),
                launcher: "local".into(),
                tenant: "default".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            TOY_PARTITION_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            3
        );
        assert!(
            set.statuses()
                .unwrap()
                .values()
                .all(|status| status.is_materialized())
        );
        let tenant = crate::tenant::Tenant::default();
        let (matrix, _) = super::partition_matrix(
            &reg,
            &set,
            &json!({}),
            &tenant,
            crate::config::launcher::LaunchTarget::Local,
        )
        .unwrap();
        assert_eq!(matrix.len(), 3);
        assert!(
            matrix
                .values()
                .all(|status| *status == CellStatus::Materialized)
        );

        // A second default backfill sees the same matrix and launches nothing.
        super::run_partition(
            &reg,
            super::PartitionCommand::Backfill {
                recipe: set.recipe.clone(),
                name: set.name.clone(),
                force: false,
                missing: false,
                stale: false,
                partitions: None,
                args: "{}".into(),
                launcher: "local".into(),
                tenant: "default".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            TOY_PARTITION_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            3
        );

        // The SQLite table is only a projection. Even a clock-skewed row that
        // appears newer cannot outrank the exact canonical JSONL record.
        let original_c = set.statuses().unwrap()["corpus=c"].clone();
        let db = crate::lineage_db::LineageDb::open().unwrap();
        db.record_partition_status(&crate::lineage_db::PartitionStatusRow {
            tenant: crate::tenant::DEFAULT_PROJECT.into(),
            recipe: set.recipe.clone(),
            partition_set: set.name.clone(),
            partition_key: "corpus=c".into(),
            job_id: "noncanonical-future-row".into(),
            outcome: "failed".into(),
            resolved_args_json: "{}".into(),
            input_fingerprint: "noncanonical".into(),
            recorded_unix: original_c.recorded_at + 10_000,
        })
        .unwrap();
        let (canonical_matrix, _) = super::partition_matrix(
            &reg,
            &set,
            &json!({}),
            &tenant,
            crate::config::launcher::LaunchTarget::Local,
        )
        .unwrap();
        assert_eq!(canonical_matrix["corpus=c"], CellStatus::Materialized);
        assert_eq!(
            db.partition_statuses("default", &set.recipe, &set.name)
                .unwrap()["corpus=c"]
                .job_id,
            original_c.job_id
        );

        // Canonical JSONL dominates a stale SQLite projection. This models a
        // transient index-write failure after a failed attempt was appended.
        // The retry is an all-cache-hit job: its stage counter stays flat, but
        // the executor must still emit current-job metadata + cache proof so
        // the resulting status is Materialized rather than permanently Stale.
        let original_a = set.statuses().unwrap()["corpus=a"].clone();
        set.record_status(&crate::config::partition::PartitionStatus {
            tenant: crate::tenant::DEFAULT_PROJECT.into(),
            key: "corpus=a".into(),
            job_id: String::new(),
            outcome: "failed".into(),
            recorded_at: original_a.recorded_at + 1,
            input_fingerprint: original_a.input_fingerprint.clone(),
        })
        .unwrap();
        let (failed_matrix, _) = super::partition_matrix(
            &reg,
            &set,
            &json!({}),
            &tenant,
            crate::config::launcher::LaunchTarget::Local,
        )
        .unwrap();
        assert_eq!(failed_matrix["corpus=a"], CellStatus::Failed);
        super::run_partition(
            &reg,
            super::PartitionCommand::Backfill {
                recipe: set.recipe.clone(),
                name: set.name.clone(),
                force: false,
                missing: false,
                stale: false,
                partitions: None,
                args: "{}".into(),
                launcher: "local".into(),
                tenant: "default".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            TOY_PARTITION_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "failed canonical retry was served from the partition cache"
        );
        let (restored_matrix, _) = super::partition_matrix(
            &reg,
            &set,
            &json!({}),
            &tenant,
            crate::config::launcher::LaunchTarget::Local,
        )
        .unwrap();
        assert_eq!(restored_matrix["corpus=a"], CellStatus::Materialized);

        // Invalidate exactly one canonical input identity. `--stale` must
        // select only that cell and bypass its otherwise-warm cache entry.
        let before_stale = set.statuses().unwrap();
        let original_b = before_stale["corpus=b"].clone();
        set.record_status(&crate::config::partition::PartitionStatus {
            tenant: crate::tenant::DEFAULT_PROJECT.into(),
            key: original_b.key.clone(),
            job_id: original_b.job_id.clone(),
            outcome: "done".into(),
            // Same timestamp is deliberate: JSONL append order is canonical,
            // while the disposable SQLite projection permits equal-time
            // replacement for concurrent completions.
            recorded_at: original_b.recorded_at,
            input_fingerprint: Some("invalidated-upstream-for-cell-b".into()),
        })
        .unwrap();
        let (stale_matrix, _) = super::partition_matrix(
            &reg,
            &set,
            &json!({}),
            &tenant,
            crate::config::launcher::LaunchTarget::Local,
        )
        .unwrap();
        assert_eq!(stale_matrix["corpus=b"], CellStatus::Stale);
        assert_eq!(stale_matrix["corpus=a"], CellStatus::Materialized);
        assert_eq!(stale_matrix["corpus=c"], CellStatus::Materialized);
        super::run_partition(
            &reg,
            super::PartitionCommand::Backfill {
                recipe: set.recipe.clone(),
                name: set.name.clone(),
                force: false,
                missing: false,
                stale: true,
                partitions: None,
                args: "{}".into(),
                launcher: "local".into(),
                tenant: "default".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            TOY_PARTITION_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "stale-only selector recomputed exactly one cell"
        );
        let after_stale = set.statuses().unwrap();
        assert_eq!(
            after_stale["corpus=a"].job_id,
            before_stale["corpus=a"].job_id
        );
        assert_ne!(after_stale["corpus=b"].job_id, original_b.job_id);
        assert_eq!(
            after_stale["corpus=c"].job_id,
            before_stale["corpus=c"].job_id
        );
        let (fresh_matrix, _) = super::partition_matrix(
            &reg,
            &set,
            &json!({}),
            &tenant,
            crate::config::launcher::LaunchTarget::Local,
        )
        .unwrap();
        assert!(
            fresh_matrix
                .values()
                .all(|status| *status == CellStatus::Materialized)
        );

        // Per-cell policy: an unrelated Restricted cell does not block a safe
        // `--missing` target, but explicitly selecting it is refused.
        let mixed = PartitionSet {
            name: "mixed".into(),
            recipe: "toy_partition".into(),
            dims: vec![PartitionDim {
                axis: "restricted".into(),
                values: vec!["false".into(), "true".into()],
            }],
        };
        mixed.save().unwrap();
        super::run_partition(
            &reg,
            super::PartitionCommand::Backfill {
                recipe: mixed.recipe.clone(),
                name: mixed.name.clone(),
                force: false,
                missing: true,
                stale: false,
                partitions: None,
                args: r#"{"corpus":"safe"}"#.into(),
                launcher: "local".into(),
                tenant: "default".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            TOY_PARTITION_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            5,
            "safe missing cell ran while unrelated Restricted cell stayed refused"
        );
        let mixed_refusal = super::run_partition(
            &reg,
            super::PartitionCommand::Backfill {
                recipe: mixed.recipe,
                name: mixed.name,
                force: true,
                missing: false,
                stale: false,
                partitions: Some("true".into()),
                args: r#"{"corpus":"safe"}"#.into(),
                launcher: "local".into(),
                tenant: "default".into(),
            },
        )
        .await
        .unwrap_err();
        assert!(
            mixed_refusal
                .to_string()
                .contains("Restricted selected cell")
        );
        assert_eq!(
            TOY_PARTITION_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            5
        );

        let refusal = super::run_partition(
            &reg,
            super::PartitionCommand::Backfill {
                recipe: set.recipe,
                name: set.name,
                force: true,
                missing: false,
                stale: false,
                partitions: None,
                args: r#"{"classification":"restricted"}"#.into(),
                launcher: "local".into(),
                tenant: "default".into(),
            },
        )
        .await
        .unwrap_err();
        assert!(refusal.to_string().contains("Restricted selected cell"));
        assert_eq!(
            TOY_PARTITION_RUNS.load(std::sync::atomic::Ordering::SeqCst),
            5,
            "Restricted preflight refused before launching any cell"
        );
    }
}
