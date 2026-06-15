//! Partitions — a NAMED, persistent partition key-space over a recipe, with
//! per-cell materialization tracking + backfill (BLUT-API Phase G, the
//! Dagster-class "run one plan over a key-set, fill only the missing cells"
//! primitive).
//!
//! ## Model
//! A [`PartitionSet`] declares a recipe + one or more [`PartitionDim`]s (axes).
//! Its **cells** are the cartesian product of the dimensions — each cell is a
//! set of `axis=value` recipe-arg overrides (the SAME shape `--set` uses), with
//! a stable [`PartitionCell::key`] like `corpus=tusz/fold=0`.
//!
//! ## Why not just `--sweep`
//! The sweep engine ([`super::sweep`]) already expands a cartesian product and
//! skips already-complete cells. Partitions add the three things that make it a
//! first-class primitive: (1) the key-space is **declared + persisted** (named,
//! reusable — you don't retype it), (2) per-cell **materialization** is tracked
//! in a status index, and (3) **backfill** runs only the un-materialized cells.
//!
//! ## Persistence (`~/.config/blut/partitions/`)
//! - definition: `<recipe>__<set>.json`
//! - materialization log: `<recipe>__<set>.status.jsonl` (append-only, last-wins)
//!
//! ## Scope (slice 1)
//! Cells map to **scalar** `axis=value` overrides (top-level recipe-arg fields),
//! reusing the executor's existing `--set` application. Path/template mapping
//! and config-fingerprint stale-detection are deliberately out of this slice.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

/// One partition axis: a recipe-arg field and the discrete values it ranges
/// over. `axis` is the dotted arg path the cell renders as a `--set` override
/// (e.g. `corpus`, `tier`, `fold`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartitionDim {
    pub axis: String,
    pub values: Vec<String>,
}

/// A named, persistent partition key-space over a recipe.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartitionSet {
    /// Set name (unique per recipe). `[A-Za-z0-9_-]+`.
    pub name: String,
    /// The recipe this set partitions (must exist in the catalog at run time).
    pub recipe: String,
    /// One or more axes. Cells = cartesian product, in this order.
    pub dims: Vec<PartitionDim>,
}

/// One concrete partition: a stable key + the `axis=value` overrides that
/// parameterize the recipe for this cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionCell {
    /// Stable identity, axes in declared order: `corpus=tusz/fold=0`.
    pub key: String,
    /// `--set`-shaped overrides: `["corpus=tusz", "fold=0"]`.
    pub overrides: Vec<String>,
}

/// One materialization record (append-only; last-wins per `key`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartitionStatus {
    pub key: String,
    pub job_id: String,
    /// `"done"` materializes the cell; anything else (e.g. `"failed"`) does not.
    pub outcome: String,
    /// Unix epoch seconds when recorded.
    pub recorded_at: i64,
}

impl PartitionStatus {
    pub fn is_materialized(&self) -> bool {
        self.outcome == "done"
    }
}

fn name_ok(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl PartitionSet {
    /// Validate the set: name/recipe well-formed, ≥1 dim, axes unique +
    /// well-formed, every dim has ≥1 value. Returns the count of cells on Ok so
    /// a caller can guard against an accidental combinatorial blowup.
    pub fn validate(&self) -> Result<usize> {
        if !name_ok(&self.name) {
            return Err(TrainError::other(format!(
                "partition set name '{}' must be non-empty [A-Za-z0-9_-]",
                self.name
            )));
        }
        if self.recipe.is_empty() {
            return Err(TrainError::other("partition set recipe is empty"));
        }
        if self.dims.is_empty() {
            return Err(TrainError::other("partition set needs ≥1 dimension"));
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut cells = 1usize;
        for d in &self.dims {
            if !name_ok(&d.axis) {
                return Err(TrainError::other(format!(
                    "partition axis '{}' must be [A-Za-z0-9_-]",
                    d.axis
                )));
            }
            if !seen.insert(&d.axis) {
                return Err(TrainError::other(format!("duplicate partition axis '{}'", d.axis)));
            }
            if d.values.is_empty() {
                return Err(TrainError::other(format!("partition axis '{}' has no values", d.axis)));
            }
            if d.values.iter().any(|v| v.contains('/') || v.contains('=') || v.is_empty()) {
                return Err(TrainError::other(format!(
                    "partition axis '{}' values must be non-empty and contain no '/' or '='",
                    d.axis
                )));
            }
            cells = cells.saturating_mul(d.values.len());
        }
        Ok(cells)
    }

    /// Render each dimension as a sweep axis string (`axis=v1,v2,...`) for
    /// [`super::sweep::cartesian`].
    fn sweep_axes(&self) -> Vec<String> {
        self.dims
            .iter()
            .map(|d| format!("{}={}", d.axis, d.values.join(",")))
            .collect()
    }

    /// Expand to one [`PartitionCell`] per cell of the cartesian product.
    /// Reuses the sweep engine so the expansion semantics match `--sweep`.
    pub fn cells(&self) -> Vec<PartitionCell> {
        super::sweep::cartesian(&self.sweep_axes())
            .into_iter()
            .map(|overrides| PartitionCell {
                key: overrides.join("/"),
                overrides,
            })
            .collect()
    }

    // ── persistence ────────────────────────────────────────────────

    /// Directory holding all partition definitions + status logs
    /// (`~/.config/blut/partitions/`). `$BLUT_PARTITIONS_DIR` overrides (tests).
    pub fn dir() -> Result<PathBuf> {
        if let Ok(p) = std::env::var("BLUT_PARTITIONS_DIR") {
            return Ok(PathBuf::from(p));
        }
        let base = dirs::config_dir()
            .ok_or_else(|| TrainError::other("cannot resolve ~/.config (set $BLUT_PARTITIONS_DIR)"))?;
        Ok(base.join("blut").join("partitions"))
    }

    fn def_path(recipe: &str, name: &str) -> Result<PathBuf> {
        Ok(Self::dir()?.join(format!("{recipe}__{name}.json")))
    }

    fn status_path(recipe: &str, name: &str) -> Result<PathBuf> {
        Ok(Self::dir()?.join(format!("{recipe}__{name}.status.jsonl")))
    }

    /// Persist this set's definition. Validates first; refuses to overwrite a
    /// DIFFERENT set silently is the caller's concern (this is a plain write).
    pub fn save(&self) -> Result<PathBuf> {
        self.validate()?;
        let dir = Self::dir()?;
        std::fs::create_dir_all(&dir).map_err(|e| TrainError::other(format!("mkdir {dir:?}: {e}")))?;
        let path = Self::def_path(&self.recipe, &self.name)?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TrainError::other(format!("serialize partition set: {e}")))?;
        std::fs::write(&path, json).map_err(|e| TrainError::other(format!("write {path:?}: {e}")))?;
        Ok(path)
    }

    /// Load a named set for a recipe.
    pub fn load(recipe: &str, name: &str) -> Result<Self> {
        let path = Self::def_path(recipe, name)?;
        let body = std::fs::read_to_string(&path)
            .map_err(|e| TrainError::other(format!("partition set {recipe}/{name} not found ({path:?}): {e}")))?;
        let set: Self = serde_json::from_str(&body)
            .map_err(|e| TrainError::other(format!("parse partition set {path:?}: {e}")))?;
        set.validate()?;
        Ok(set)
    }

    /// List all defined sets as `(recipe, name)`, sorted. Missing dir = empty.
    pub fn list() -> Result<Vec<(String, String)>> {
        let dir = Self::dir()?;
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(TrainError::other(format!("read_dir {dir:?}: {e}"))),
        };
        let mut out = Vec::new();
        for ent in rd.flatten() {
            let fname = ent.file_name();
            let s = fname.to_string_lossy();
            if let Some(stem) = s.strip_suffix(".json") {
                if let Some((recipe, name)) = stem.split_once("__") {
                    out.push((recipe.to_string(), name.to_string()));
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Append a materialization record (append-only; never rewrites history).
    pub fn record_status(&self, status: &PartitionStatus) -> Result<()> {
        let dir = Self::dir()?;
        std::fs::create_dir_all(&dir).map_err(|e| TrainError::other(format!("mkdir {dir:?}: {e}")))?;
        let path = Self::status_path(&self.recipe, &self.name)?;
        let line = serde_json::to_string(status)
            .map_err(|e| TrainError::other(format!("serialize status: {e}")))?;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| TrainError::other(format!("open {path:?}: {e}")))?;
        writeln!(f, "{line}").map_err(|e| TrainError::other(format!("append {path:?}: {e}")))?;
        Ok(())
    }

    /// Latest status per key (last-wins). Missing log = empty map.
    pub fn statuses(&self) -> Result<BTreeMap<String, PartitionStatus>> {
        let path = Self::status_path(&self.recipe, &self.name)?;
        let body = match std::fs::read_to_string(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(e) => return Err(TrainError::other(format!("read {path:?}: {e}"))),
        };
        let mut map = BTreeMap::new();
        for line in body.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Tolerate a corrupt trailing line (partial append) — skip it.
            if let Ok(rec) = serde_json::from_str::<PartitionStatus>(line) {
                map.insert(rec.key.clone(), rec);
            }
        }
        Ok(map)
    }

    /// The cells NOT yet materialized (no `done` status), in cell order. This is
    /// exactly the backfill work-list. `force` returns ALL cells (re-run).
    pub fn backfill_targets(&self, force: bool) -> Result<Vec<PartitionCell>> {
        if force {
            return Ok(self.cells());
        }
        let done = self.statuses()?;
        Ok(self
            .cells()
            .into_iter()
            .filter(|c| !done.get(&c.key).is_some_and(PartitionStatus::is_materialized))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set() -> PartitionSet {
        PartitionSet {
            name: "by_corpus_fold".into(),
            recipe: "lamquant_snn_4state".into(),
            dims: vec![
                PartitionDim { axis: "corpus".into(), values: vec!["tusz".into(), "chbmit".into()] },
                PartitionDim { axis: "fold".into(), values: vec!["0".into(), "1".into(), "2".into()] },
            ],
        }
    }

    // Point the persistence dir at a private tempdir. `BLUT_PARTITIONS_DIR` is
    // process-global, so serialize env-mutating tests behind TEST_ENV_LOCK (the
    // crate's convention, cf. sweep::tests) — else parallel tests clobber each
    // other's dir. The guard holds BOTH the lock and the tempdir for the test.
    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        _td: tempfile::TempDir,
    }
    fn tmp_env() -> EnvGuard {
        let lock = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        // SAFETY: TEST_ENV_LOCK serializes this env mutation across tests.
        unsafe { std::env::set_var("BLUT_PARTITIONS_DIR", td.path()) };
        EnvGuard { _lock: lock, _td: td }
    }

    #[test]
    fn validate_counts_cells_and_rejects_bad() {
        assert_eq!(set().validate().unwrap(), 6);
        let mut s = set();
        s.dims.push(PartitionDim { axis: "corpus".into(), values: vec!["x".into()] });
        assert!(s.validate().is_err(), "duplicate axis rejected");
        let mut s2 = set();
        s2.dims[0].values.clear();
        assert!(s2.validate().is_err(), "empty values rejected");
        let mut s3 = set();
        s3.dims[0].values = vec!["a=b".into()];
        assert!(s3.validate().is_err(), "'=' in value rejected");
    }

    #[test]
    fn cells_are_the_cartesian_product_with_stable_keys() {
        let cells = set().cells();
        assert_eq!(cells.len(), 6);
        let keys: Vec<&str> = cells.iter().map(|c| c.key.as_str()).collect();
        assert!(keys.contains(&"corpus=tusz/fold=0"));
        assert!(keys.contains(&"corpus=chbmit/fold=2"));
        // overrides are --set-shaped
        let c0 = cells.iter().find(|c| c.key == "corpus=tusz/fold=0").unwrap();
        assert_eq!(c0.overrides, vec!["corpus=tusz".to_string(), "fold=0".to_string()]);
    }

    #[test]
    fn save_load_list_roundtrip() {
        let _g = tmp_env();
        let s = set();
        s.save().unwrap();
        let loaded = PartitionSet::load(&s.recipe, &s.name).unwrap();
        assert_eq!(loaded, s);
        let listed = PartitionSet::list().unwrap();
        assert_eq!(listed, vec![(s.recipe.clone(), s.name.clone())]);
    }

    #[test]
    fn backfill_targets_shrink_as_cells_materialize() {
        let _g = tmp_env();
        let s = set();
        s.save().unwrap();
        // all 6 missing initially
        assert_eq!(s.backfill_targets(false).unwrap().len(), 6);
        // materialize two cells
        for key in ["corpus=tusz/fold=0", "corpus=chbmit/fold=1"] {
            s.record_status(&PartitionStatus {
                key: key.into(),
                job_id: "job-1".into(),
                outcome: "done".into(),
                recorded_at: 1,
            })
            .unwrap();
        }
        let remaining = s.backfill_targets(false).unwrap();
        assert_eq!(remaining.len(), 4, "two materialized → four left");
        assert!(!remaining.iter().any(|c| c.key == "corpus=tusz/fold=0"));
        // a FAILED status does NOT materialize
        s.record_status(&PartitionStatus {
            key: "corpus=tusz/fold=1".into(),
            job_id: "job-2".into(),
            outcome: "failed".into(),
            recorded_at: 2,
        })
        .unwrap();
        assert_eq!(s.backfill_targets(false).unwrap().len(), 4, "failed cell still pending");
        // force returns all
        assert_eq!(s.backfill_targets(true).unwrap().len(), 6);
    }

    #[test]
    fn statuses_last_wins_per_key() {
        let _g = tmp_env();
        let s = set();
        let k = "corpus=tusz/fold=0";
        s.record_status(&PartitionStatus { key: k.into(), job_id: "a".into(), outcome: "failed".into(), recorded_at: 1 }).unwrap();
        s.record_status(&PartitionStatus { key: k.into(), job_id: "b".into(), outcome: "done".into(), recorded_at: 2 }).unwrap();
        let st = s.statuses().unwrap();
        assert_eq!(st.get(k).unwrap().job_id, "b");
        assert!(st.get(k).unwrap().is_materialized());
    }
}
