// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Partitions — a NAMED, persistent partition key-space over a recipe, with
//! per-cell materialization tracking + backfill (BLUT-API Phase G, the
//! Dagster-class "run one plan over a key-set, fill only the missing cells"
//! primitive).
//!
//! ## Model
//! A [`PartitionSet`] declares a recipe + one or more [`PartitionDim`]s (axes).
//! Its **cells** are the cartesian product of the dimensions — each cell is a
//! set of `axis=value` recipe-arg overrides (the SAME shape `--set` uses), with
//! a stable [`PartitionCell::key`] like `corpus=dataset_a/fold=0`.
//!
//! ## Why not just `--sweep`
//! The sweep engine ([`super::sweep`]) already expands a cartesian product and
//! skips already-complete cells. Partitions add the three things that make it a
//! first-class primitive: (1) the key-space is **declared + persisted** (named,
//! reusable — you don't retype it), (2) per-cell **materialization** is tracked
//! in a status index, and (3) **backfill** runs only the un-materialized cells.
//!
//! ## Persistence (`~/.config/blut/partitions/`)
//! - definition: `<recipe>~<set>.json`
//! - materialization log:
//!   `tenants/<tenant>/<recipe>~<set>.status.jsonl` (append-only, last-wins)
//! - legacy default-tenant logs at the old root path remain readable
//!
//! ## Cell mapping
//! Cells map to **scalar** `axis=value` overrides (top-level recipe-arg fields),
//! reusing the executor's existing `--set` application. Typed time and
//! categorical specs are expanded to that finite persisted key-space.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

// Device chains in one backfill append concurrently. Hold one process-wide
// guard and issue one complete record write so formatted JSON cannot interleave
// at a line boundary inside this process.
static STATUS_APPEND_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// WASM-safe wire contracts are re-exported through the engine so cookbook
/// crates need only their normal `blut` dependency, not a second direct
/// dependency on the keystone crate.
pub use blut_types::partition::{PartitionKey, PartitionSpec, PartitionValue, TimeGranularity};

/// One partition axis: a recipe-arg field and the discrete values it ranges
/// over. `axis` is the dotted arg path the cell renders as a `--set` override
/// (e.g. `corpus`, `tier`, `fold`).
//
// (The example axes above are illustrative names — any recipe-arg field works.)
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
    /// Stable identity, axes in declared order: `corpus=dataset_a/fold=0`.
    pub key: String,
    /// `--set`-shaped overrides: `["corpus=dataset_a", "fold=0"]`.
    pub overrides: Vec<String>,
}

impl PartitionCell {
    /// Typed wire identity used by the executor/cache seam.
    pub fn partition_key(&self) -> Result<blut_types::partition::PartitionKey> {
        let values = self
            .overrides
            .iter()
            .map(|item| {
                let (dimension, value) = item.split_once('=').ok_or_else(|| {
                    TrainError::other(format!("malformed partition override '{item}'"))
                })?;
                Ok(blut_types::partition::PartitionValue::new(dimension, value))
            })
            .collect::<Result<Vec<_>>>()?;
        blut_types::partition::PartitionKey::new(values)
            .map_err(|e| TrainError::other(format!("invalid partition key '{}': {e}", self.key)))
    }
}

/// One materialization record (append-only; last-wins per `key`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartitionStatus {
    /// Owning tenant. Legacy records deserialize as `default`.
    #[serde(default = "default_partition_tenant")]
    pub tenant: String,
    pub key: String,
    pub job_id: String,
    /// `"done"` materializes the cell; anything else (e.g. `"failed"`) does not.
    pub outcome: String,
    /// Unix epoch seconds when recorded.
    pub recorded_at: i64,
    /// Canonical identity of source handles plus the compiled/resolved recipe
    /// args. Missing on legacy records, which therefore cannot prove freshness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fingerprint: Option<String>,
}

fn default_partition_tenant() -> String {
    crate::tenant::DEFAULT_PROJECT.to_string()
}

impl PartitionStatus {
    pub fn is_materialized(&self) -> bool {
        self.outcome == "done"
    }
}

/// The richer per-cell status matrix (ADR 0101): derived from the recorded
/// outcome + the clinical flag + whether an upstream partition's artifact hash
/// changed. `--missing`/`--stale` backfill selectors target exactly the cells in
/// those states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellStatus {
    /// Done AND its upstreams unchanged — a re-run is a cache hit (free).
    Materialized,
    /// Done, but an upstream partition's artifact hash changed — needs a re-run.
    Stale,
    /// Ran and did not complete (outcome ≠ `done`).
    Failed,
    /// A clinical/PHI (ADR 0061) cell — never surfaced as materialized in a
    /// cloud view, and a backfill refuses it fail-closed.
    Restricted,
    /// Never run.
    Missing,
}

/// Which cells a backfill should expose to admission. Restricted cells are
/// deliberately included by `Force`: the caller must refuse them explicitly,
/// never make them disappear as though they were already materialized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackfillSelector {
    #[default]
    Default,
    Force,
    Missing,
    Stale,
}

/// Select cells from an already-derived status matrix while preserving the
/// partition set's declared cell order.
pub fn select_backfill_targets(
    cells: &[PartitionCell],
    matrix: &BTreeMap<String, CellStatus>,
    selector: BackfillSelector,
) -> Vec<PartitionCell> {
    cells
        .iter()
        .filter(|cell| {
            let status = matrix
                .get(&cell.key)
                .copied()
                .unwrap_or(CellStatus::Missing);
            match selector {
                BackfillSelector::Force => true,
                BackfillSelector::Missing => status == CellStatus::Missing,
                BackfillSelector::Stale => status == CellStatus::Stale,
                BackfillSelector::Default => matches!(
                    status,
                    CellStatus::Missing | CellStatus::Stale | CellStatus::Failed
                ),
            }
        })
        .cloned()
        .collect()
}

/// Fail-closed classification for one cell's source args. Tenant policy
/// dominates; otherwise cookbooks can expose an explicit per-cell marker via a
/// `restricted=true` flag or a classification field. This keeps the policy in
/// ordinary recipe data instead of forcing authors into a BLUT-specific UI.
pub fn partition_args_are_restricted(
    args: &serde_json::Value,
    tenant: &crate::tenant::Tenant,
) -> bool {
    if tenant.is_restricted() {
        return true;
    }
    let mut pending = vec![args];
    while let Some(value) = pending.pop() {
        match value {
            serde_json::Value::Object(object) => {
                for (key, value) in object {
                    let key = key.to_ascii_lowercase();
                    if (key == "restricted" && value.as_bool() == Some(true))
                        || (matches!(
                            key.as_str(),
                            "classification" | "data_class" | "security_class" | "access_class"
                        ) && value.as_str().is_some_and(|class| {
                            matches!(
                                class.to_ascii_lowercase().as_str(),
                                "restricted" | "clinical" | "phi"
                            )
                        }))
                    {
                        return true;
                    }
                    pending.push(value);
                }
            }
            serde_json::Value::Array(values) => pending.extend(values),
            _ => {}
        }
    }
    false
}

/// Stable identity used by the lineage matrix to decide whether the inputs a
/// cell would consume still match its last materialization. Source args retain
/// immutable registry handles (`dataset://`, `model://`, `experiment://`),
/// while compiled args carry their live resolution and cookbook defaults.
pub fn partition_input_fingerprint(
    source_args: &serde_json::Value,
    compiled_args: &serde_json::Value,
) -> String {
    partition_input_fingerprint_with_execution(source_args, compiled_args, "")
}

pub(crate) fn partition_input_fingerprint_with_execution(
    source_args: &serde_json::Value,
    compiled_args: &serde_json::Value,
    execution_fingerprint: &str,
) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"blut.partition.input.v1");
    hasher.update((execution_fingerprint.len() as u64).to_le_bytes());
    hasher.update(execution_fingerprint.as_bytes());
    for value in [source_args, compiled_args] {
        let canonical = crate::framework::CacheHandle::canonical_json_bytes(value);
        hasher.update((canonical.len() as u64).to_le_bytes());
        hasher.update(canonical);
    }
    faster_hex::hex_string(&hasher.finalize())
}

/// Convert a typed wire spec into the persisted finite dimensions used by the
/// backfill scheduler. Time specs require a finite inclusive range.
pub fn dims_from_spec(spec: &PartitionSpec, time_range: Option<&str>) -> Result<Vec<PartitionDim>> {
    fn append(
        spec: &PartitionSpec,
        time_range: Option<&str>,
        out: &mut Vec<PartitionDim>,
    ) -> Result<()> {
        match spec {
            PartitionSpec::Categorical { dimension, values } => out.push(PartitionDim {
                axis: dimension.clone(),
                values: values.clone(),
            }),
            PartitionSpec::Time {
                dimension,
                granularity,
                tz,
            } => {
                if !matches!(tz.as_str(), "UTC" | "Etc/UTC" | "Z") {
                    return Err(TrainError::other(format!(
                        "time partition timezone '{tz}' is unsupported; use UTC"
                    )));
                }
                let range = time_range.ok_or_else(|| {
                    TrainError::other("a time PartitionSpec requires --partitions START:END")
                })?;
                out.push(PartitionDim {
                    axis: dimension.clone(),
                    values: expand_time_range(range, *granularity)?,
                });
            }
            PartitionSpec::Multi { dimensions } => {
                if dimensions.is_empty() {
                    return Err(TrainError::other(
                        "a Multi PartitionSpec needs at least one dimension",
                    ));
                }
                let time_count = dimensions
                    .iter()
                    .filter(|dimension| matches!(dimension, PartitionSpec::Time { .. }))
                    .count();
                if time_count > 1 {
                    return Err(TrainError::other(
                        "a Multi PartitionSpec supports at most one time dimension",
                    ));
                }
                for dimension in dimensions {
                    if matches!(dimension, PartitionSpec::Multi { .. }) {
                        return Err(TrainError::other(
                            "nested Multi PartitionSpec values are not supported",
                        ));
                    }
                    append(dimension, time_range, out)?;
                }
            }
        }
        Ok(())
    }

    let mut dims = Vec::new();
    append(spec, time_range, &mut dims)?;
    Ok(dims)
}

fn expand_time_range(range: &str, granularity: TimeGranularity) -> Result<Vec<String>> {
    let (start, end) = range.split_once(':').ok_or_else(|| {
        TrainError::other("--partitions time range must be START:END (inclusive)")
    })?;
    let mut values = Vec::new();
    match granularity {
        TimeGranularity::Day | TimeGranularity::Week => {
            let mut current = chrono::NaiveDate::parse_from_str(start, "%Y-%m-%d")
                .map_err(|e| TrainError::other(format!("invalid time range start: {e}")))?;
            let end = chrono::NaiveDate::parse_from_str(end, "%Y-%m-%d")
                .map_err(|e| TrainError::other(format!("invalid time range end: {e}")))?;
            let step = if granularity == TimeGranularity::Day {
                1
            } else {
                7
            };
            while current <= end {
                values.push(current.format("%Y-%m-%d").to_string());
                current = current
                    .checked_add_days(chrono::Days::new(step))
                    .ok_or_else(|| TrainError::other("time partition range overflow"))?;
                if values.len() > MAX_CELLS {
                    return Err(TrainError::other(
                        "time partition range exceeds cell ceiling",
                    ));
                }
            }
        }
        TimeGranularity::Hour => {
            let mut current = chrono::NaiveDateTime::parse_from_str(start, "%Y-%m-%dT%H")
                .map_err(|e| TrainError::other(format!("invalid hourly range start: {e}")))?;
            let end = chrono::NaiveDateTime::parse_from_str(end, "%Y-%m-%dT%H")
                .map_err(|e| TrainError::other(format!("invalid hourly range end: {e}")))?;
            while current <= end {
                values.push(current.format("%Y-%m-%dT%H").to_string());
                current = current
                    .checked_add_signed(chrono::Duration::hours(1))
                    .ok_or_else(|| TrainError::other("hourly partition range overflow"))?;
                if values.len() > MAX_CELLS {
                    return Err(TrainError::other(
                        "time partition range exceeds cell ceiling",
                    ));
                }
            }
        }
        TimeGranularity::Month => {
            let parse = |value: &str| -> Result<(i32, u32)> {
                let date = chrono::NaiveDate::parse_from_str(&format!("{value}-01"), "%Y-%m-%d")
                    .map_err(|e| TrainError::other(format!("invalid monthly range: {e}")))?;
                use chrono::Datelike;
                Ok((date.year(), date.month()))
            };
            let (mut year, mut month) = parse(start)?;
            let end = parse(end)?;
            while (year, month) <= end {
                values.push(format!("{year:04}-{month:02}"));
                if month == 12 {
                    year = year
                        .checked_add(1)
                        .ok_or_else(|| TrainError::other("monthly partition range overflow"))?;
                    month = 1;
                } else {
                    month += 1;
                }
                if values.len() > MAX_CELLS {
                    return Err(TrainError::other(
                        "time partition range exceeds cell ceiling",
                    ));
                }
            }
        }
    }
    if values.is_empty() {
        return Err(TrainError::other(
            "--partitions range end must not precede its start",
        ));
    }
    Ok(values)
}

/// Restrict a declared cell space by an explicit comma-separated set or an
/// inclusive first-dimension range. Full stable keys and first-axis values are
/// accepted; unknown tokens fail closed.
pub fn select_partition_cells(
    cells: &[PartitionCell],
    selection: &str,
) -> Result<Vec<PartitionCell>> {
    if let Some((start, end)) = selection.split_once(':') {
        let selected: Vec<_> = cells
            .iter()
            .filter(|cell| {
                cell.overrides
                    .first()
                    .and_then(|item| item.split_once('='))
                    .is_some_and(|(_, value)| value >= start && value <= end)
            })
            .cloned()
            .collect();
        if selected.is_empty() {
            return Err(TrainError::other(format!(
                "--partitions range '{selection}' selects no declared cells"
            )));
        }
        return Ok(selected);
    }
    let requested: std::collections::BTreeSet<_> = selection
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .collect();
    if requested.is_empty() {
        return Err(TrainError::other("--partitions explicit set is empty"));
    }
    let selected: Vec<_> = cells
        .iter()
        .filter(|cell| {
            requested.contains(cell.key.as_str())
                || cell
                    .overrides
                    .first()
                    .and_then(|item| item.split_once('='))
                    .is_some_and(|(_, value)| requested.contains(value))
        })
        .cloned()
        .collect();
    let mut matched = std::collections::BTreeSet::new();
    for cell in &selected {
        if requested.contains(cell.key.as_str()) {
            matched.insert(cell.key.as_str());
        }
        if let Some((_, value)) = cell.overrides.first().and_then(|item| item.split_once('='))
            && requested.contains(value)
        {
            matched.insert(value);
        }
    }
    let unknown: Vec<_> = requested.difference(&matched).copied().collect();
    if !unknown.is_empty() {
        return Err(TrainError::other(format!(
            "--partitions names undeclared cell(s): {}",
            unknown.join(", ")
        )));
    }
    Ok(selected)
}

impl CellStatus {
    /// Derive a cell's status. Clinical dominates (checked first, fail-closed):
    /// a `restricted` cell is `Restricted` regardless of its run state.
    pub fn derive(
        recorded: Option<&PartitionStatus>,
        restricted: bool,
        upstream_stale: bool,
    ) -> CellStatus {
        if restricted {
            return CellStatus::Restricted;
        }
        match recorded {
            None => CellStatus::Missing,
            Some(s) if !s.is_materialized() => CellStatus::Failed,
            Some(_) if upstream_stale => CellStatus::Stale,
            Some(_) => CellStatus::Materialized,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CellStatus::Materialized => "materialized",
            CellStatus::Stale => "stale",
            CellStatus::Failed => "failed",
            CellStatus::Restricted => "restricted",
            CellStatus::Missing => "missing",
        }
    }
}

/// Build the status matrix for a set's cells: each cell → its derived
/// [`CellStatus`]. `restricted_of` flags a clinical cell (ADR 0061); `stale_of`
/// reports whether a materialized cell's upstreams drifted (the ADR-0100/lineage
/// hash comparison — injected so this stays a pure, testable derivation).
pub fn status_matrix(
    cells: &[PartitionCell],
    latest: &std::collections::BTreeMap<String, PartitionStatus>,
    restricted_of: impl Fn(&str) -> bool,
    stale_of: impl Fn(&str) -> bool,
) -> Vec<(String, CellStatus)> {
    cells
        .iter()
        .map(|c| {
            let recorded = latest.get(&c.key);
            // Short-circuit: the (potentially DB/lineage-costly) stale check is
            // only relevant for a materialized cell — a Missing/Failed/Restricted
            // cell never reaches the stale branch of `derive`.
            let stale = recorded.is_some_and(PartitionStatus::is_materialized) && stale_of(&c.key);
            let status = CellStatus::derive(recorded, restricted_of(&c.key), stale);
            (c.key.clone(), status)
        })
        .collect()
}

/// Hard ceiling on a partition's cell count — `validate()` rejects above it so
/// an accidental product (many dims × many values) can't explode a backfill.
const MAX_CELLS: usize = 100_000;

/// A recipe / set / axis name is safe iff non-empty and `[A-Za-z0-9_-]` — no
/// `.` or `/`, so it can never traverse out of the partitions dir. The on-disk
/// file separator is `~` ([`SEP`]), which is NOT in this alphabet, so a
/// `<recipe>~<name>` filename is unambiguous even when a name contains `_`
/// (a `__`-based separator would alias `a_`/`b` with `a`/`_b`).
fn name_ok(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// The `<recipe>SEP<name>` file separator — outside [`name_ok`]'s alphabet so it
/// can't appear in either token, making the filename collision-free.
const SEP: char = '~';

/// A partition VALUE is safe iff non-empty and free of path separators (`/`,
/// `\`), the `axis=value` separator (`=`), and the sweep-grammar
/// metacharacters (`,` choice · `:` range · `[]()` interval/list · `*` glob).
/// Values are never re-parsed by `cells()` (it builds the product directly), but
/// the cell's overrides ARE applied downstream as `--set axis=value`, so a
/// metachar there could still be re-interpreted — reject it at the source.
fn value_ok(v: &str) -> bool {
    !v.is_empty()
        && !v.chars().any(|c| {
            matches!(
                c,
                '/' | '\\' | '=' | ',' | ':' | '[' | ']' | '(' | ')' | '*'
            )
        })
}

impl PartitionSet {
    /// Validate the set: name/recipe well-formed, ≥1 dim, axes unique +
    /// well-formed, every dim has ≥1 metachar-free value, and the cell count is
    /// within `MAX_CELLS`. Returns the (bounded) cell count on Ok.
    pub fn validate(&self) -> Result<usize> {
        if !name_ok(&self.name) {
            return Err(TrainError::other(format!(
                "partition set name '{}' must be non-empty [A-Za-z0-9_-]",
                self.name
            )));
        }
        if !name_ok(&self.recipe) {
            return Err(TrainError::other(format!(
                "partition set recipe '{}' must be non-empty [A-Za-z0-9_-]",
                self.recipe
            )));
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
                return Err(TrainError::other(format!(
                    "duplicate partition axis '{}'",
                    d.axis
                )));
            }
            if d.values.is_empty() {
                return Err(TrainError::other(format!(
                    "partition axis '{}' has no values",
                    d.axis
                )));
            }
            if !d.values.iter().all(|v| value_ok(v)) {
                return Err(TrainError::other(format!(
                    "partition axis '{}' values must be non-empty and free of path / sweep-grammar \
                     metacharacters (/ \\ = , : [ ] ( ) *)",
                    d.axis
                )));
            }
            cells = cells
                .checked_mul(d.values.len())
                .filter(|&c| c <= MAX_CELLS)
                .ok_or_else(|| {
                    TrainError::other(format!(
                        "partition '{}' would expand to more than {MAX_CELLS} cells \
                         (combinatorial blowup)",
                        self.name
                    ))
                })?;
        }
        Ok(cells)
    }

    /// Expand to one [`PartitionCell`] per cell of the cartesian product. Built
    /// DIRECTLY (each value → an `axis=value` override, dims in declared order)
    /// — values are never round-tripped through the sweep grammar, so a value
    /// can't be re-parsed (e.g. a comma split into two) or escape its cell.
    pub fn cells(&self) -> Vec<PartitionCell> {
        let mut acc: Vec<Vec<String>> = vec![Vec::new()];
        for d in &self.dims {
            let mut next = Vec::with_capacity(acc.len() * d.values.len());
            for prefix in &acc {
                for v in &d.values {
                    let mut overrides = prefix.clone();
                    overrides.push(format!("{}={}", d.axis, v));
                    next.push(overrides);
                }
            }
            acc = next;
        }
        acc.into_iter()
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
        let base = dirs::config_dir().ok_or_else(|| {
            TrainError::other("cannot resolve ~/.config (set $BLUT_PARTITIONS_DIR)")
        })?;
        Ok(base.join("blut").join("partitions"))
    }

    /// The chokepoint EVERY path builder passes through, so even `load()` with
    /// an attacker-/typo-supplied `recipe`/`name` can't traverse out of the
    /// partitions dir (`..` and `/` are not in `name_ok`'s alphabet) or alias
    /// another set's file (the `~` separator can't appear in either token).
    fn guard_identity(recipe: &str, name: &str) -> Result<()> {
        if !name_ok(recipe) || !name_ok(name) {
            return Err(TrainError::other(format!(
                "unsafe partition identity '{recipe}/{name}' — recipe + name must be [A-Za-z0-9_-]"
            )));
        }
        Ok(())
    }

    fn def_path(recipe: &str, name: &str) -> Result<PathBuf> {
        Self::guard_identity(recipe, name)?;
        Ok(Self::dir()?.join(format!("{recipe}{SEP}{name}.json")))
    }

    fn status_path(recipe: &str, name: &str) -> Result<PathBuf> {
        Self::guard_identity(recipe, name)?;
        Ok(Self::dir()?.join(format!("{recipe}{SEP}{name}.status.jsonl")))
    }

    fn tenant_status_path(recipe: &str, name: &str, tenant: &str) -> Result<PathBuf> {
        Self::guard_identity(recipe, name)?;
        let tenant = crate::tenant::Tenant::parse(tenant).ok_or_else(|| {
            TrainError::other(format!("invalid partition status tenant '{tenant}'"))
        })?;
        Ok(Self::dir()?
            .join("tenants")
            .join(tenant.as_path())
            .join(format!("{recipe}{SEP}{name}.status.jsonl")))
    }

    /// Persist this set's definition. Validates first. A plain write: it
    /// overwrites any existing file at the same `<recipe>~<name>` path —
    /// guarding against clobbering a DIFFERENT set is the caller's concern.
    pub fn save(&self) -> Result<PathBuf> {
        self.validate()?;
        let dir = Self::dir()?;
        std::fs::create_dir_all(&dir)
            .map_err(|e| TrainError::other(format!("mkdir {dir:?}: {e}")))?;
        let path = Self::def_path(&self.recipe, &self.name)?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TrainError::other(format!("serialize partition set: {e}")))?;
        std::fs::write(&path, json)
            .map_err(|e| TrainError::other(format!("write {path:?}: {e}")))?;
        Ok(path)
    }

    /// Load a named set for a recipe.
    pub fn load(recipe: &str, name: &str) -> Result<Self> {
        let path = Self::def_path(recipe, name)?;
        let body = std::fs::read_to_string(&path).map_err(|e| {
            TrainError::other(format!(
                "partition set {recipe}/{name} not found ({path:?}): {e}"
            ))
        })?;
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
                // `SEP` can't appear in either token, so `split_once` is exact.
                if let Some((recipe, name)) = stem.split_once(SEP) {
                    out.push((recipe.to_string(), name.to_string()));
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Append a materialization record (append-only; never rewrites history).
    pub fn record_status(&self, status: &PartitionStatus) -> Result<()> {
        let _append_guard = STATUS_APPEND_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let path = Self::tenant_status_path(&self.recipe, &self.name, &status.tenant)?;
        let dir = path
            .parent()
            .ok_or_else(|| TrainError::other("partition status path has no parent"))?;
        std::fs::create_dir_all(dir)
            .map_err(|e| TrainError::other(format!("mkdir {dir:?}: {e}")))?;
        let mut line = serde_json::to_vec(status)
            .map_err(|e| TrainError::other(format!("serialize status: {e}")))?;
        line.push(b'\n');
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| TrainError::other(format!("open {path:?}: {e}")))?;
        f.write_all(&line)
            .map_err(|e| TrainError::other(format!("append {path:?}: {e}")))?;
        Ok(())
    }

    /// Latest status per key (last-wins). Missing log = empty map.
    pub fn statuses(&self) -> Result<BTreeMap<String, PartitionStatus>> {
        self.statuses_for_tenant(crate::tenant::DEFAULT_PROJECT)
    }

    /// Tenant-scoped canonical last-wins view. Default-tenant reads merge the
    /// pre-tenancy legacy file first, then the scoped file, so migration is
    /// additive and a later scoped record wins.
    pub fn statuses_for_tenant(&self, tenant: &str) -> Result<BTreeMap<String, PartitionStatus>> {
        let mut map = BTreeMap::new();
        let mut paths = Vec::new();
        if tenant == crate::tenant::DEFAULT_PROJECT {
            paths.push(Self::status_path(&self.recipe, &self.name)?);
        }
        paths.push(Self::tenant_status_path(&self.recipe, &self.name, tenant)?);
        for path in paths {
            let body = match std::fs::read_to_string(&path) {
                Ok(body) => body,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(TrainError::other(format!("read {path:?}: {e}"))),
            };
            for line in body.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                // Tolerate a corrupt trailing line (partial append) — skip it.
                if let Ok(rec) = serde_json::from_str::<PartitionStatus>(line)
                    && rec.tenant == tenant
                {
                    map.insert(rec.key.clone(), rec);
                }
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
            .filter(|c| {
                !done
                    .get(&c.key)
                    .is_some_and(PartitionStatus::is_materialized)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_status_matrix_derives_all_states() {
        use std::collections::BTreeMap;
        let cells: Vec<PartitionCell> = ["a", "b", "c", "d", "phi"]
            .iter()
            .map(|k| PartitionCell {
                key: k.to_string(),
                overrides: vec![],
            })
            .collect();
        let done = |k: &str| PartitionStatus {
            tenant: crate::tenant::DEFAULT_PROJECT.into(),
            key: k.into(),
            job_id: "j".into(),
            outcome: "done".into(),
            recorded_at: 1,
            input_fingerprint: None,
        };
        let mut latest = BTreeMap::new();
        latest.insert("a".to_string(), done("a")); // materialized (fresh)
        latest.insert("b".to_string(), done("b")); // materialized but stale
        latest.insert(
            "c".to_string(),
            PartitionStatus {
                tenant: crate::tenant::DEFAULT_PROJECT.into(),
                key: "c".into(),
                job_id: "j".into(),
                outcome: "failed".into(),
                recorded_at: 1,
                input_fingerprint: None,
            },
        ); // failed
        // "d" has no record → missing. "phi" is restricted → restricted (even done).
        latest.insert("phi".to_string(), done("phi"));

        let restricted = |k: &str| k == "phi";
        let stale = |k: &str| k == "b";
        let m: std::collections::BTreeMap<_, _> = status_matrix(&cells, &latest, restricted, stale)
            .into_iter()
            .collect();
        assert_eq!(m["a"], CellStatus::Materialized);
        assert_eq!(m["b"], CellStatus::Stale);
        assert_eq!(m["c"], CellStatus::Failed);
        assert_eq!(m["d"], CellStatus::Missing);
        // Clinical dominates: restricted even though it recorded `done`.
        assert_eq!(m["phi"], CellStatus::Restricted);
    }

    fn set() -> PartitionSet {
        PartitionSet {
            name: "by_corpus_fold".into(),
            recipe: "lamquant_snn_4state".into(),
            dims: vec![
                PartitionDim {
                    axis: "corpus".into(),
                    values: vec!["tusz".into(), "chbmit".into()],
                },
                PartitionDim {
                    axis: "fold".into(),
                    values: vec!["0".into(), "1".into(), "2".into()],
                },
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
        let lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        // SAFETY: TEST_ENV_LOCK serializes this env mutation across tests.
        unsafe { std::env::set_var("BLUT_PARTITIONS_DIR", td.path()) };
        EnvGuard {
            _lock: lock,
            _td: td,
        }
    }

    #[test]
    fn validate_counts_cells_and_rejects_bad() {
        assert_eq!(set().validate().unwrap(), 6);
        let mut s = set();
        s.dims.push(PartitionDim {
            axis: "corpus".into(),
            values: vec!["x".into()],
        });
        assert!(s.validate().is_err(), "duplicate axis rejected");
        let mut s2 = set();
        s2.dims[0].values.clear();
        assert!(s2.validate().is_err(), "empty values rejected");
        let mut s3 = set();
        s3.dims[0].values = vec!["a=b".into()];
        assert!(s3.validate().is_err(), "'=' in value rejected");
    }

    #[test]
    fn typed_specs_expand_categorical_multi_and_utc_time() {
        let categorical = PartitionSpec::Categorical {
            dimension: "corpus".into(),
            values: vec!["tuh".into(), "chbmit".into()],
        };
        assert_eq!(dims_from_spec(&categorical, None).unwrap().len(), 1);

        let multi = PartitionSpec::Multi {
            dimensions: vec![
                categorical,
                PartitionSpec::Time {
                    dimension: "day".into(),
                    granularity: TimeGranularity::Day,
                    tz: "UTC".into(),
                },
            ],
        };
        let dims = dims_from_spec(&multi, Some("2026-07-01:2026-07-03")).unwrap();
        assert_eq!(dims[0].values, ["tuh", "chbmit"]);
        assert_eq!(dims[1].values, ["2026-07-01", "2026-07-02", "2026-07-03"]);
        assert!(
            dims_from_spec(
                &PartitionSpec::Time {
                    dimension: "day".into(),
                    granularity: TimeGranularity::Day,
                    tz: "America/New_York".into(),
                },
                Some("2026-07-01:2026-07-02")
            )
            .is_err(),
            "unsupported timezone must fail closed"
        );
    }

    #[test]
    fn explicit_and_range_partition_selection_are_exact() {
        let cells = set().cells();
        let explicit = select_partition_cells(&cells, "tusz").unwrap();
        assert_eq!(explicit.len(), 3);
        assert!(
            explicit
                .iter()
                .all(|cell| cell.key.starts_with("corpus=tusz/"))
        );

        let range = select_partition_cells(&cells, "chbmit:tusz").unwrap();
        assert_eq!(range.len(), 6);
        let full = select_partition_cells(&cells, "corpus=tusz/fold=1").unwrap();
        assert_eq!(full.len(), 1);
        assert!(select_partition_cells(&cells, "unknown").is_err());
    }

    #[test]
    fn cells_do_not_reparse_grammar_metachars() {
        // A value containing a comma must yield exactly ONE cell, verbatim — the
        // direct product never feeds values back through the sweep grammar
        // (which would split `a,b` into two cells). validate() also rejects such
        // a value, but cells() must be safe regardless of how a set is built.
        let s = PartitionSet {
            name: "g".into(),
            recipe: "r".into(),
            dims: vec![PartitionDim {
                axis: "x".into(),
                values: vec!["a,b".into()],
            }],
        };
        let cells = s.cells();
        assert_eq!(cells.len(), 1, "comma value is one cell, not two");
        assert_eq!(cells[0].key, "x=a,b");
        assert_eq!(cells[0].overrides, vec!["x=a,b".to_string()]);
    }

    #[test]
    fn validate_rejects_metachars_and_unsafe_recipe() {
        for bad in ["a,b", "1:3", "x[0]", "g*"] {
            let mut s = set();
            s.dims[0].values = vec![bad.into()];
            assert!(
                s.validate().is_err(),
                "grammar/path metachar value {bad:?} rejected"
            );
        }
        let mut traversal = set();
        traversal.recipe = "../evil".into();
        assert!(
            traversal.validate().is_err(),
            "path-traversal recipe rejected"
        );
        let mut tilde = set();
        tilde.name = "a~b".into();
        assert!(
            tilde.validate().is_err(),
            "the file separator '~' rejected in a name"
        );
    }

    #[test]
    fn boundary_underscore_identities_do_not_alias() {
        // `a_`/`b` and `a`/`_b` would BOTH map to `a___b.json` under a `__`
        // separator — the `~` separator keeps them distinct files.
        let _g = tmp_env();
        let a = PartitionSet {
            name: "b".into(),
            recipe: "a_".into(),
            dims: vec![PartitionDim {
                axis: "x".into(),
                values: vec!["0".into()],
            }],
        };
        let b = PartitionSet {
            name: "_b".into(),
            recipe: "a".into(),
            ..a.clone()
        };
        a.save().unwrap();
        b.save().unwrap();
        assert_eq!(PartitionSet::load("a_", "b").unwrap(), a, "a_/b intact");
        assert_eq!(
            PartitionSet::load("a", "_b").unwrap(),
            b,
            "a/_b not clobbered by a_/b"
        );
        assert_eq!(PartitionSet::list().unwrap().len(), 2, "two distinct files");
    }

    #[test]
    fn validate_rejects_combinatorial_blowup() {
        // 20^6 = 64M cells ≫ MAX_CELLS — must be refused, not saturated.
        let big = PartitionSet {
            name: "big".into(),
            recipe: "r".into(),
            dims: (0..6)
                .map(|i| PartitionDim {
                    axis: format!("a{i}"),
                    values: (0..20).map(|j| j.to_string()).collect(),
                })
                .collect(),
        };
        assert!(big.validate().is_err(), "combinatorial blowup rejected");
    }

    #[test]
    fn unsafe_identity_path_is_refused_at_load() {
        let _g = tmp_env();
        // load() builds the path from its args BEFORE reading; an unsafe recipe
        // or name must be refused at the path chokepoint, never read from disk.
        assert!(PartitionSet::load("../../etc/passwd", "x").is_err());
        assert!(
            PartitionSet::load("a~b", "x").is_err(),
            "'~' (separator) in recipe refused"
        );
    }

    #[test]
    fn cells_are_the_cartesian_product_with_stable_keys() {
        let cells = set().cells();
        assert_eq!(cells.len(), 6);
        let keys: Vec<&str> = cells.iter().map(|c| c.key.as_str()).collect();
        assert!(keys.contains(&"corpus=tusz/fold=0"));
        assert!(keys.contains(&"corpus=chbmit/fold=2"));
        // overrides are --set-shaped
        let c0 = cells
            .iter()
            .find(|c| c.key == "corpus=tusz/fold=0")
            .unwrap();
        assert_eq!(
            c0.overrides,
            vec!["corpus=tusz".to_string(), "fold=0".to_string()]
        );
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
                tenant: crate::tenant::DEFAULT_PROJECT.into(),
                key: key.into(),
                job_id: "job-1".into(),
                outcome: "done".into(),
                recorded_at: 1,
                input_fingerprint: None,
            })
            .unwrap();
        }
        let remaining = s.backfill_targets(false).unwrap();
        assert_eq!(remaining.len(), 4, "two materialized → four left");
        assert!(!remaining.iter().any(|c| c.key == "corpus=tusz/fold=0"));
        // a FAILED status does NOT materialize
        s.record_status(&PartitionStatus {
            tenant: crate::tenant::DEFAULT_PROJECT.into(),
            key: "corpus=tusz/fold=1".into(),
            job_id: "job-2".into(),
            outcome: "failed".into(),
            recorded_at: 2,
            input_fingerprint: None,
        })
        .unwrap();
        assert_eq!(
            s.backfill_targets(false).unwrap().len(),
            4,
            "failed cell still pending"
        );
        // force returns all
        assert_eq!(s.backfill_targets(true).unwrap().len(), 6);
    }

    #[test]
    fn statuses_last_wins_per_key() {
        let _g = tmp_env();
        let s = set();
        let k = "corpus=tusz/fold=0";
        s.record_status(&PartitionStatus {
            tenant: crate::tenant::DEFAULT_PROJECT.into(),
            key: k.into(),
            job_id: "a".into(),
            outcome: "failed".into(),
            recorded_at: 1,
            input_fingerprint: None,
        })
        .unwrap();
        s.record_status(&PartitionStatus {
            tenant: crate::tenant::DEFAULT_PROJECT.into(),
            key: k.into(),
            job_id: "b".into(),
            outcome: "done".into(),
            recorded_at: 2,
            input_fingerprint: None,
        })
        .unwrap();
        let st = s.statuses().unwrap();
        assert_eq!(st.get(k).unwrap().job_id, "b");
        assert!(st.get(k).unwrap().is_materialized());
    }

    #[test]
    fn concurrent_status_appends_preserve_every_jsonl_record() {
        let _g = tmp_env();
        let s = set();
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let set = &s;
                scope.spawn(move || {
                    for record in 0..64 {
                        set.record_status(&PartitionStatus {
                            tenant: crate::tenant::DEFAULT_PROJECT.into(),
                            key: format!("worker={worker}/record={record}"),
                            job_id: format!("job-{worker}-{record}"),
                            outcome: "done".into(),
                            recorded_at: record,
                            input_fingerprint: Some(format!("fp-{worker}-{record}")),
                        })
                        .unwrap();
                    }
                });
            }
        });
        assert_eq!(s.statuses().unwrap().len(), 4 * 64);
    }

    #[test]
    fn restricted_scan_is_iterative_for_deep_programmatic_args() {
        let mut args = serde_json::json!({"classification": "phi"});
        for _ in 0..256 {
            args = serde_json::json!({"nested": args});
        }
        assert!(partition_args_are_restricted(
            &args,
            &crate::tenant::Tenant::default()
        ));
    }

    #[test]
    fn canonical_status_history_is_tenant_isolated() {
        let _g = tmp_env();
        let s = set();
        let key = "corpus=tusz/fold=0";
        for (tenant, job_id, outcome) in [
            (crate::tenant::DEFAULT_PROJECT, "default-job", "done"),
            ("research/dev", "research-job", "failed"),
            ("clinical/phi", "clinical-job", "failed"),
        ] {
            s.record_status(&PartitionStatus {
                tenant: tenant.into(),
                key: key.into(),
                job_id: job_id.into(),
                outcome: outcome.into(),
                recorded_at: 1,
                input_fingerprint: Some(format!("fingerprint-{tenant}")),
            })
            .unwrap();
        }

        assert_eq!(s.statuses().unwrap()[key].job_id, "default-job");
        assert_eq!(
            s.statuses_for_tenant("research/dev").unwrap()[key].job_id,
            "research-job"
        );
        assert_eq!(
            s.statuses_for_tenant("clinical/phi").unwrap()[key].job_id,
            "clinical-job"
        );
    }
}
