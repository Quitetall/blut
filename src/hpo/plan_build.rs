// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! HPO fan-out plan builder (v0.20 Phase 2).
//!
//! Samples `n_trials` configs from the search space, compiles each as an
//! independent sub-plan over the base recipe args (a sampled overlay → distinct
//! cache key + durable-resume dir for free), and merges them into ONE
//! [`CompiledPlan`] via [`CompiledPlan::from_components`]. The executor then
//! runs all trials in parallel under the existing GPU semaphore + memory-admission
//! admission.

use serde_json::Value;

use crate::framework::Registry;
use crate::framework::plan::{CompiledPlan, NodeId};

use super::sampler::Sampler;
use super::space::{Overlay, SearchSpace, apply_overlay};

/// Per-trial metadata: the trial's id, its sampled overlay, and the first
/// global node id of its sub-graph in the merged plan (the caller maps a
/// node/topo index → trial via these contiguous ranges).
#[derive(Clone, Debug)]
pub struct TrialPlan {
    pub trial_id: u32,
    pub overlay: Overlay,
    pub node_offset: NodeId,
}

/// Build the merged fan-out plan + per-trial metadata. Errors if the recipe is
/// unknown or any trial fails to compile.
pub fn build_hpo_plan(
    reg: &Registry,
    recipe: &str,
    base_args: &Value,
    space: &SearchSpace,
    sampler: &mut dyn Sampler,
    n_trials: u32,
) -> Result<(CompiledPlan, Vec<TrialPlan>), String> {
    let def = reg
        .find(recipe)
        .ok_or_else(|| format!("recipe '{recipe}' not in catalog"))?;
    if n_trials == 0 {
        return Err("n_trials must be >= 1".into());
    }
    let mut components = Vec::with_capacity(n_trials as usize);
    let mut overlays: Vec<Overlay> = Vec::with_capacity(n_trials as usize);
    for t in 0..n_trials {
        let overlay = sampler.ask(space, &[]);
        let mut args = base_args.clone();
        apply_overlay(&mut args, &overlay);
        let comp = (def.compile_fn)(args).map_err(|e| format!("trial {t} compile failed: {e}"))?;
        components.push(comp);
        overlays.push(overlay);
    }
    let (merged, offsets) =
        CompiledPlan::from_components(recipe.to_string(), base_args.clone(), components);
    let trials = offsets
        .into_iter()
        .zip(overlays)
        .enumerate()
        .map(|(i, (node_offset, overlay))| TrialPlan {
            trial_id: i as u32,
            overlay,
            node_offset,
        })
        .collect();
    Ok((merged, trials))
}
