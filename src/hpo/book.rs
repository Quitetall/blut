// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The trial book: which plan nodes belong to which HPO trial, including the
//! trials a policy spawns while the search runs (TPE and surrogate
//! suggestions, PBT clones).
//!
//! The initial population is known when the plan is built and lands in
//! `hpo.json` before execution. A spawned trial only gets plan nodes when the
//! executor injects it, so the policy reserves a label when it asks for the
//! spawn and the executor reports the injected nodes back
//! ([`crate::framework::control::ControlPolicy::on_spawned`]). The book then
//! gives the trial the next id, maps its nodes, and (for a book bound to a job
//! directory) rewrites `hpo.json`, so `blut hpo show`, `blut hpo best` and a
//! study's Pareto front rank spawned trials exactly like initial ones, during
//! and after the run. The policy reads the same map, so it learns from the
//! trials it suggested.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;

use super::results::{HpoManifest, TrialRec};
use super::space::Overlay;

/// The live trial map shared by an HPO policy and the CLI.
pub struct TrialBook {
    state: Mutex<BookState>,
    /// The job directory whose `hpo.json` mirrors the book, if any.
    job_dir: Option<PathBuf>,
}

struct BookState {
    manifest: HpoManifest,
    /// Spawns asked for but not yet injected, by label.
    pending: HashMap<String, Overlay>,
    next_label: u64,
    /// trial id → (topo index, stage name) of the trial's last node.
    terminal: HashMap<u32, (u32, String)>,
}

impl TrialBook {
    /// An in-memory book over a manifest (no `hpo.json` is written).
    pub fn new(manifest: HpoManifest) -> Self {
        Self {
            state: Mutex::new(BookState {
                manifest,
                pending: HashMap::new(),
                next_label: 0,
                terminal: HashMap::new(),
            }),
            job_dir: None,
        }
    }

    /// A book for an initial population given only its topo map and
    /// overlays (the shape policies are constructed with).
    pub fn from_parts(trial_of_topo: Vec<Option<u32>>, overlays: Vec<Overlay>) -> Self {
        let mut n_nodes = vec![0u32; overlays.len()];
        for t in trial_of_topo.iter().flatten() {
            if let Some(n) = n_nodes.get_mut(*t as usize) {
                *n += 1;
            }
        }
        let trials = overlays
            .into_iter()
            .enumerate()
            .map(|(i, overlay)| TrialRec {
                trial_id: i as u32,
                overlay,
                n_nodes: n_nodes[i],
            })
            .collect();
        Self::new(HpoManifest {
            recipe: String::new(),
            algo: String::new(),
            metric: String::new(),
            mode: String::new(),
            budget_key: String::new(),
            trials,
            trial_of_topo,
        })
    }

    /// A book that rewrites `<job_dir>/hpo.json` whenever a trial is added.
    pub fn persisted(manifest: HpoManifest, job_dir: PathBuf) -> Self {
        let mut book = Self::new(manifest);
        book.job_dir = Some(job_dir);
        book
    }

    /// The trial that owns plan node `node_idx` (its topo position).
    pub fn trial_of(&self, node_idx: u32) -> Option<u32> {
        self.state
            .lock()
            .manifest
            .trial_of_topo
            .get(node_idx as usize)
            .copied()
            .flatten()
    }

    /// A trial's overlay (its sampled hyperparameters).
    pub fn overlay(&self, trial: u32) -> Option<Overlay> {
        self.state
            .lock()
            .manifest
            .trials
            .get(trial as usize)
            .map(|t| t.overlay.clone())
    }

    /// Number of trials known so far (initial + bound spawns).
    pub fn len(&self) -> usize {
        self.state.lock().manifest.trials.len()
    }

    /// Whether the book has no trials.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reserve a label for a spawn carrying `overlay`. The trial gets its id
    /// only when [`TrialBook::bind`] sees the injected nodes, so a spawn the
    /// executor drops leaves no gap in the trial ids.
    pub fn reserve(&self, overlay: Overlay) -> String {
        let mut st = self.state.lock();
        let label = format!("hpo-spawn-{}", st.next_label);
        st.next_label += 1;
        st.pending.insert(label.clone(), overlay);
        label
    }

    /// Record an injected spawn: give it the next trial id, map its nodes
    /// (`(topo index, stage name)`, in topo order) and persist. `None` when
    /// the label was not reserved by this book.
    pub fn bind(&self, label: &str, nodes: &[(u32, &str)]) -> Option<u32> {
        let (trial, snapshot) = {
            let mut st = self.state.lock();
            let overlay = st.pending.remove(label)?;
            let trial = st.manifest.trials.len() as u32;
            st.manifest.trials.push(TrialRec {
                trial_id: trial,
                overlay,
                n_nodes: nodes.len() as u32,
            });
            for (idx, _) in nodes {
                let idx = *idx as usize;
                let map = &mut st.manifest.trial_of_topo;
                if map.len() <= idx {
                    map.resize(idx + 1, None);
                }
                map[idx] = Some(trial);
            }
            if let Some((idx, stage)) = nodes.iter().max_by_key(|(idx, _)| *idx) {
                st.terminal.insert(trial, (*idx, (*stage).to_string()));
            }
            (trial, self.job_dir.as_ref().map(|_| st.manifest.clone()))
        };
        if let (Some(dir), Some(manifest)) = (&self.job_dir, snapshot)
            && let Err(e) = manifest.write_to(dir)
        {
            tracing::warn!(
                "hpo: could not record a spawned trial in {}: {e}",
                dir.join("hpo.json").display()
            );
        }
        Some(trial)
    }

    /// The stage directory of a SPAWNED trial's last node
    /// (`<job_dir>/stages/<topo>-<stage>`), where a PBT clone of it resumes
    /// from. `None` for an initial trial or an in-memory book.
    pub fn spawned_stage_dir(&self, trial: u32) -> Option<PathBuf> {
        let dir = self.job_dir.as_deref()?;
        let st = self.state.lock();
        let (idx, stage) = st.terminal.get(&trial)?;
        Some(stage_dir(dir, *idx, stage))
    }

    /// A copy of the manifest as it stands.
    pub fn manifest(&self) -> HpoManifest {
        self.state.lock().manifest.clone()
    }
}

/// `<job_dir>/stages/<topo>-<stage>`, the executor's per-node stage directory.
fn stage_dir(job_dir: &Path, idx: u32, stage: &str) -> PathBuf {
    job_dir.join("stages").join(format!("{idx}-{stage}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn overlay(lr: f64) -> Overlay {
        vec![("lr".to_string(), json!(lr))]
    }

    #[test]
    fn a_bound_spawn_becomes_the_next_trial_and_is_persisted() {
        let td = tempfile::tempdir().unwrap();
        let initial =
            TrialBook::from_parts(vec![Some(0), Some(1)], vec![overlay(0.1), overlay(0.2)]);
        let book = TrialBook::persisted(initial.manifest(), td.path().to_path_buf());

        let dropped = book.reserve(overlay(0.9));
        let label = book.reserve(overlay(0.3));
        assert_eq!(book.bind(&label, &[(2, "train"), (3, "eval")]), Some(2));
        assert_eq!(book.trial_of(3), Some(2));
        assert_eq!(book.overlay(2), Some(overlay(0.3)));
        assert_eq!(book.bind("hpo-spawn-unknown", &[(4, "train")]), None);
        assert_eq!(book.len(), 3, "a dropped spawn leaves no gap: {dropped}");
        assert_eq!(
            book.spawned_stage_dir(2),
            Some(td.path().join("stages").join("3-eval"))
        );

        let on_disk = HpoManifest::read_from(td.path()).expect("hpo.json written");
        assert_eq!(on_disk.trials.len(), 3);
        assert_eq!(on_disk.trials[2].n_nodes, 2);
        assert_eq!(
            on_disk.trial_of_topo,
            vec![Some(0), Some(1), Some(2), Some(2)]
        );
    }
}
