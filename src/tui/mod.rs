// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut tui` — the interactive training cockpit.
//!
//! A single ratatui surface multiplexed by a `View` enum. The default
//! `Cockpit` view is a single-column overview (header / running jobs /
//! resources / recipe menu); the other views are launchers + analytic /
//! provenance panels, all reading the engine's OWN state — the jobs store,
//! the lineage DB, the content-addressed cache, and the plan graph (the same
//! sources as the `jobs` / `log` / `dag` / `lineage` CLI subcommands). There
//! is no domain knowledge and no separate data model: a panel shows what the
//! engine recorded for a run, so it works for any cookbook.
//!
//! ## Views
//!   * `Jobs`        — full jobs list (all states, color-coded)
//!   * `Log`         — the selected job's `status.jsonl` tail
//!   * `System`      — GPU / MEM / DISK / CPU probe
//!   * `History`     — every job ⋈ its lineage record (recipe / outcome / metric)
//!   * `Leaderboard` — runs ranked by the active metric
//!   * `Compare`     — marked runs side by side (final metrics + GPU sat)
//!   * `Dag`         — the selected run's plan graph + per-node status
//!   * `Lineage`     — stage hashes · cache hits/misses · code freshness
//!   * `Artifacts`   — the run's content-addressed stage outputs
//!   * `Metrics`     — the selected run's final metric values
//!   * `Catalog`     — registered recipes by course + their args schema
//!   * `Reset`       — generic maintenance (prune cache / clear job / forget footprints)
//!
//! ## Keybindings (cockpit view)
//!   ↑ / ↓ / j / k   move selection (jobs/list rows)
//!   Enter           open the selected job's log
//!   r               refresh jobs + system
//!   c               cancel selected job (SIGTERM via `blut cancel`)
//!   R               custom-recipe fuzzy picker
//!   `<recipe hotkey>` launch a recipe (pre-baked defaults for the
//!                   registered cookbook recipes, schema template else)
//!   J/L/Y/H/B/C/G/I/A/M/P/X  switch to Jobs / Log / sYstem / History /
//!                      leaderBoard / Compare / daG / lIneage / Artifacts /
//!                      Metrics / catalog(P) / maintenance(X) views
//!   Esc / b         back to cockpit view (or quit from cockpit)
//!   q / Ctrl-C      quit
//! (cockpit + output + file_browser). T1/T2 of the BLUT cockpit track;
//! U3-U7 + the three-cockpit merge per the training-cockpit directive.

use anyhow::{Context, Result};
use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use crate::jobs::{self, JobState, JobSummary};
// The TUI sources its recipe catalog from the injected Registry
// (App.catalog) — blut-core ships NO recipes. The in-module tests bring
// their own domain-free fixture catalog (`test_fixtures`) so the generic
// catalog / filter / menu logic is exercised without any concrete
// cookbook (those moved to the cookbook crates at C2a / C2b).
#[cfg(test)]
use test_fixtures::{FIXTURE, test_registry};

/// Domain-free recipe fixtures for the TUI tests. blut-core is a generic
/// engine with zero recipes, so its TUI tests run against synthetic
/// `RecipeDef`s wrapped in a test cookbook — exactly the shape a real
/// cookbook crate registers at runtime. Names are neutral (no real recipe
/// names) and avoid the reserved-hotkey first chars.
#[cfg(test)]
mod test_fixtures {
    use crate::framework::{Cookbook, Registry};
    use crate::recipes::recipe::{RecipeCategory, RecipeDef};

    fn empty_object_schema() -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    /// A `schema_of`-shaped schema with real typed fields, so the per-field
    /// form (F2) tests have editable rows + scalar types to round-trip. Two
    /// typed fields (`lr: number` default 0.001, `tag: string` no default).
    fn typed_args_schema() -> serde_json::Value {
        serde_json::json!({
            "$ref": "#/definitions/Args",
            "definitions": {
                "Args": {
                    "type": "object",
                    "properties": {
                        "lr": { "type": "number", "default": 0.001 },
                        "tag": { "type": "string" }
                    }
                }
            }
        })
    }

    pub(super) static TRAIN_ALPHA: RecipeDef = RecipeDef {
        name: "train_alpha",
        description: "fixture training recipe alpha",
        backend_id: "fixture",
        category: RecipeCategory::Train,
        input_kinds: &["dataset.jsonl"],
        output_kind: "checkpoint.hf",
        schedule: None,
        args_schema_fn: empty_object_schema,
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };
    pub(super) static TRAIN_BETA: RecipeDef = RecipeDef {
        name: "train_beta",
        description: "fixture training recipe beta",
        backend_id: "fixture",
        category: RecipeCategory::Train,
        input_kinds: &["dataset.jsonl"],
        output_kind: "checkpoint.hf",
        schedule: None,
        args_schema_fn: empty_object_schema,
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };
    pub(super) static EVAL_GAMMA: RecipeDef = RecipeDef {
        name: "eval_gamma",
        description: "fixture evaluation recipe gamma",
        backend_id: "fixture",
        category: RecipeCategory::Eval,
        input_kinds: &["checkpoint.hf"],
        output_kind: "eval.report",
        schedule: None,
        args_schema_fn: empty_object_schema,
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };

    /// A `dataset.jsonl`-consuming recipe whose schema declares the
    /// `registered_name` input convention — so the F3 picker→inject tests can
    /// assert the picked dataset lands in the right form field.
    fn dataset_input_schema() -> serde_json::Value {
        serde_json::json!({
            "$ref": "#/definitions/Args",
            "definitions": {
                "Args": {
                    "type": "object",
                    "properties": {
                        "registered_name": { "type": "string" },
                        "epochs": { "type": "integer", "default": 1 }
                    }
                }
            }
        })
    }

    pub(super) static TRAIN_EPSILON: RecipeDef = RecipeDef {
        name: "train_epsilon",
        description: "fixture recipe consuming a registered dataset",
        backend_id: "fixture",
        category: RecipeCategory::Train,
        input_kinds: &["dataset.jsonl"],
        output_kind: "checkpoint.hf",
        schedule: None,
        args_schema_fn: dataset_input_schema,
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };

    /// A graph-input recipe (empty `input_kinds` → skips the F3 dataset
    /// picker, opens the form directly) WITH a real typed args schema, so the
    /// F2 per-field form tests have editable rows.
    pub(super) static TRAIN_DELTA: RecipeDef = RecipeDef {
        name: "train_delta",
        description: "fixture training recipe delta (graph-input, typed args)",
        backend_id: "fixture",
        category: RecipeCategory::Train,
        input_kinds: &[],
        output_kind: "checkpoint.hf",
        schedule: None,
        args_schema_fn: typed_args_schema,
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };

    /// The fixture catalog the TUI tests index (stands in for the recipes
    /// a real cookbook crate registers at runtime). Train + Eval
    /// categories exercise the category-grouped cockpit/menu paths.
    pub(super) static FIXTURE: &[&RecipeDef] =
        &[&TRAIN_ALPHA, &TRAIN_BETA, &EVAL_GAMMA, &TRAIN_DELTA];

    struct FixtureCookbook;
    impl Cookbook for FixtureCookbook {
        fn name(&self) -> &'static str {
            "fixture"
        }
        fn recipes(&self) -> &'static [&'static RecipeDef] {
            FIXTURE
        }
    }

    /// A registry holding exactly the fixture cookbook — what the TUI
    /// tests build their `App` from (stands in for a real cookbook crate's
    /// `registry()`).
    pub(super) fn test_registry() -> Registry {
        let mut r = Registry::new();
        r.register(Box::new(FixtureCookbook));
        r
    }
}

/// Point `datasets_db` at an empty per-process temp file so the F3 dataset
/// picker never touches the operator's real `conversations.db`. The fixture
/// recipes declare `input_kinds`, so opening one routes through
/// `open_dataset_picker`; with an empty DB it finds no datasets and falls
/// straight through to the args editor (the path the existing Picker→Editor
/// tests assert). Idempotent + safe to call from every test helper.
#[cfg(test)]
fn isolate_datasets_db_for_tests() {
    use std::sync::OnceLock;
    static DB: OnceLock<std::path::PathBuf> = OnceLock::new();
    let path = DB.get_or_init(|| {
        std::env::temp_dir().join(format!("blut-tui-datasets-{}.db", std::process::id()))
    });
    // SAFETY: `set_var` is `unsafe` because a concurrent `getenv` in another
    // thread is a data race. Here every caller writes the SAME OnceLock-derived
    // path, so the only value ever stored is identical — a benign race (the
    // observed value is the same regardless of ordering). This is `#[cfg(test)]`
    // and the env var is read only by `datasets_db::registry_path()`, which the
    // dataset picker calls synchronously from the (single-threaded) test driving
    // the key handler, never concurrently with this write.
    unsafe {
        std::env::set_var("LAMU_MEMORY_DB", path);
    }
}

mod console;
mod render;
mod system;
mod theme;
mod views;

// The draw_* painters live in render.rs but keep their unqualified names
// here — the event loop and the headless check harness call them as before.
use render::*;

/// Which surface the single ratatui frame is currently showing.
/// `Cockpit` is the default single-column overview; the rest are the
/// detail panels (revived from the formerly-dead `draw_jobs`/`draw_log`
/// /`draw_system`) and the screens ported from the Python cockpit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    /// The BLUT engine console — the default landing surface (redesign).
    Console,
    Cockpit,
    Jobs,
    Log,
    System,
    History,
    Leaderboard,
    Compare,
    Dag,
    Lineage,
    Artifacts,
    Metrics,
    Catalog,
    Reset,
}

impl View {
    fn title(self) -> &'static str {
        match self {
            View::Console => "BLUT Console",
            View::Cockpit => "Training Cockpit",
            View::Jobs => "Jobs",
            View::Log => "Job Log",
            View::System => "System",
            View::History => "Run History",
            View::Leaderboard => "Leaderboard",
            View::Compare => "Compare Runs",
            View::Dag => "Run DAG",
            View::Lineage => "Lineage & Provenance",
            View::Artifacts => "Artifacts",
            View::Metrics => "Run Metrics",
            View::Catalog => "Recipe Catalog",
            View::Reset => "Maintenance",
        }
    }
}

/// One editable row in the per-field args form (F2). Built from a
/// recipe schema's top-level `properties`: a `name`, a `ty` hint
/// (the schema's scalar `type` string, e.g. `"number"` / `"string"`,
/// or `"any"` when the schema declares none / a union), and the live
/// `value` the user edits as text. On submit the rows are reassembled
/// into the args JSON object and run through the SAME
/// `validate_args_against_schema` the freeform buffer used.
struct EditorField {
    name: String,
    ty: String,
    value: String,
}

/// Modal overlay state. `None` = main jobs+log view; `Picker` floats
/// a recipe list over the main view; `DatasetPicker` floats a
/// kind-filtered dataset list (F3/U5) before the editor; `Editor`
/// shows a per-field args form (F2) built from the recipe's schema
/// properties, with an optional raw-JSON fallback toggle.
enum Overlay {
    None,
    Picker {
        query: String,
        cursor: usize,
    },
    /// F3/U5: pick a dataset (filtered to the recipe's `input_kinds`)
    /// to feed the recipe's input before opening the args editor.
    /// `datasets` is the pre-filtered, owned list (name + kind +
    /// source_path captured at open). Esc skips → editor with no
    /// dataset injected.
    DatasetPicker {
        recipe: &'static crate::recipes::RecipeDef,
        datasets: Vec<DatasetChoice>,
        cursor: usize,
    },
    Editor {
        recipe: &'static str,
        /// Per-field form rows (the DEFAULT surface). Empty only for a
        /// truly schema-less recipe (`{}`), in which case the form shows
        /// a hint and the raw fallback is the way to add fields.
        fields: Vec<EditorField>,
        /// Which field row is focused (typing edits this row's value).
        focus: usize,
        /// Raw-JSON fallback toggle (Ctrl+R). `false` = the per-field
        /// form (default); `true` = the freeform JSON buffer in
        /// `raw_buffer`. Submitting from raw mode uses `raw_buffer`
        /// verbatim; from form mode the fields are assembled to JSON.
        raw_mode: bool,
        /// The freeform JSON buffer used while `raw_mode` is on. Seeded
        /// from the fields when the toggle flips on; re-parsed back into
        /// the fields when it flips off (so an edit in either view
        /// survives the toggle).
        raw_buffer: String,
    },
}

/// A dataset row offered by the kind-filtered picker (F3). Owns just the
/// fields the picker needs — name (the `registered_name` we inject),
/// kind (shown), and source_path (injected when a recipe takes a path
/// arg rather than a registered name).
#[derive(Clone)]
struct DatasetChoice {
    name: String,
    kind: String,
    source_path: String,
    n_examples: i64,
}

#[derive(Clone, Copy)]
enum BuiltinAction {
    Refresh,
    Quit,
    Cancel,
    OpenPicker,
}

/// Auto-refresh cadence for jobs list + system probes.
const REFRESH_TICK: Duration = Duration::from_millis(1500);

/// Refresh ticks (≈ `FOCUS_GIVE_UP_TICKS × REFRESH_TICK` ≈ 30 s) a
/// post-spawn auto-focus waits for its job to surface before giving up.
const FOCUS_GIVE_UP_TICKS: u8 = 20;

/// Maximum log lines kept in memory per selected job. Older lines are
/// dropped (caller hits `r` or re-selects to re-tail from disk).
const MAX_LOG_LINES: usize = 2000;

struct App {
    /// The engine-console model (mesh / DAG / broker / ledger / cache / gov).
    console: console::ConsoleModel,
    jobs: Vec<JobSummary>,
    selected: ListState,
    log_lines: Vec<String>,
    log_job_id: Option<String>,
    system: system::SystemSnapshot,
    last_refresh: Instant,
    status_msg: Option<(String, Instant)>,
    overlay: Overlay,
    /// User pressed quit — main loop exits at top of next iteration.
    quit: bool,
    /// Active surface (cockpit / detail panel / migrated screen).
    view: View,
    /// Cached run rows (History / Leaderboard), sourced from the jobs store
    /// ⋈ the lineage DB. Re-read on view-entry + refresh so the panels
    /// aren't I/O-bound per draw.
    runs: Vec<views::RunRow>,
    /// Per-job view caches, loaded for `current_job_id()` on view-entry +
    /// refresh: the selected run's artifacts, plan DAG, lineage/provenance,
    /// and final metrics.
    artifacts: Vec<views::ArtifactRow>,
    dag: Option<crate::framework::GraphSnapshot>,
    lineage: views::LineageView,
    metrics: Vec<(String, f64)>,
    compare: Vec<views::CompareCol>,
    /// Cursor for list-style views (history / leaderboard / artifacts /
    /// catalog). `marked` holds the job ids selected for Compare.
    list_cursor: usize,
    marked: Vec<String>,
    /// Reset view: which destructive action is armed (two-press confirm).
    reset_cursor: usize,
    reset_armed: Option<(usize, Instant)>,
    /// The cookbook registry this session was launched with (the binary
    /// composes it). Source of the recipe catalog + per-recipe default
    /// args — the TUI indexes the composed catalog, not any static slice,
    /// so it is domain-agnostic. [[project_blut_cookbook_split]]
    registry: crate::framework::Registry,
    /// Flat recipe catalog (union of the registry's cookbooks), collected
    /// once at startup. `filter_recipes` / `recipe_menu` index into this.
    catalog: Vec<&'static crate::recipes::RecipeDef>,
    /// F1: after a TUI-launched spawn, auto-focus the new job's Log on the
    /// first refresh that surfaces it. `focus_baseline_top` is the newest
    /// job id at spawn time; jobs are newest-first, so when index 0 differs
    /// from it the new job has appeared and we jump to its live tail.
    pending_focus: bool,
    focus_baseline_top: Option<String>,
    /// Refresh ticks left before a pending auto-focus gives up — so a spawn
    /// whose job never surfaces (child died before writing its job dir)
    /// can't hijack focus onto an unrelated job that appears much later.
    focus_ticks_left: u8,
}

/// Two-press confirm window for the destructive Maintenance actions.
const RESET_WINDOW: Duration = Duration::from_secs(3);

/// Maintenance-view rows: generic, domain-agnostic destructive actions on
/// the engine's own state (cache / job dir / footprint store).
const RESET_ROWS: &[views::ResetAction] = &[
    views::ResetAction::PruneCache,
    views::ResetAction::ClearJob,
    views::ResetAction::ForgetFootprints,
];

impl App {
    fn new(registry: crate::framework::Registry) -> Self {
        let mut selected = ListState::default();
        selected.select(Some(0));
        // Collect the catalog once: the union of the registered cookbooks'
        // recipes. Elements are `&'static`, so the Vec owns no borrow of
        // `registry` and `App` can hold both without a self-referential tie.
        let catalog: Vec<&'static crate::recipes::RecipeDef> = registry.all().collect();
        Self {
            console: console::ConsoleModel::demo(),
            jobs: Vec::new(),
            selected,
            log_lines: Vec::new(),
            log_job_id: None,
            system: system::SystemSnapshot::default(),
            last_refresh: Instant::now() - REFRESH_TICK,
            status_msg: None,
            overlay: Overlay::None,
            quit: false,
            view: View::Console,
            runs: Vec::new(),
            artifacts: Vec::new(),
            dag: None,
            lineage: views::LineageView::default(),
            metrics: Vec::new(),
            compare: Vec::new(),
            list_cursor: 0,
            marked: Vec::new(),
            reset_cursor: 0,
            reset_armed: None,
            registry,
            catalog,
            pending_focus: false,
            focus_baseline_top: None,
            focus_ticks_left: 0,
        }
    }

    /// Switch the active view, lazily (re)loading the data it needs.
    /// Resets the list cursor so a freshly-entered view starts at the
    /// top.
    fn set_view(&mut self, view: View) {
        // Capture the selected run BEFORE the cursor resets — the per-job
        // views (DAG / Lineage / Artifacts / Metrics) key off it.
        let job = self.current_job_id();
        self.view = view;
        self.list_cursor = 0;
        self.load_view_data(view, job.as_deref());
        if view == View::Reset {
            self.reset_cursor = 0;
            self.reset_armed = None;
        }
        self.set_status(format!("view: {}", view.title()));
    }

    /// (Re)load the data the given view renders. Split from `set_view` so the
    /// periodic refresh can re-pull the active view (live DAG / metrics on a
    /// running job). `job` is the run the per-job views target.
    fn load_view_data(&mut self, view: View, job: Option<&str>) {
        match view {
            View::History => self.runs = views::run_history(views::DEFAULT_METRIC),
            View::Leaderboard => self.runs = views::leaderboard(views::DEFAULT_METRIC, false),
            View::Compare => self.compare = views::compare(&self.marked),
            View::Artifacts => self.artifacts = job.map(views::artifacts_for).unwrap_or_default(),
            View::Dag => self.dag = job.and_then(views::dag_for),
            View::Lineage => self.lineage = job.map(views::lineage_for).unwrap_or_default(),
            View::Metrics => self.metrics = job.map(views::metrics_for).unwrap_or_default(),
            _ => {}
        }
        // A refresh may have SHRUNK the active list (a job disappeared, the DB
        // returned fewer rows). Re-clamp the cursor so it can't point past the
        // end — otherwise `current_job_id()` (which drives Compare-mark,
        // Enter→DAG, and the Reset→ClearJob target) would act on a different run
        // than the highlighted row.
        let len = self.active_list_len();
        self.list_cursor = self.list_cursor.min(len.saturating_sub(1));
    }

    /// Number of rows in the cursor-driven list for the active view (0 for views
    /// with no `list_cursor` — including Reset, which uses its own
    /// `reset_cursor`, so its `list_cursor` is harmlessly clamped to 0).
    fn active_list_len(&self) -> usize {
        match self.view {
            View::History | View::Leaderboard => self.runs.len(),
            View::Artifacts => self.artifacts.len(),
            View::Catalog => self.catalog.len(),
            _ => 0,
        }
    }

    /// The job id the per-job views (DAG / Lineage / Artifacts / Metrics)
    /// operate on: the Cockpit/Jobs/Log selection, else the run under the list
    /// cursor (History / Leaderboard), else the newest job.
    fn current_job_id(&self) -> Option<String> {
        // On the cockpit + the Jobs/Log views the live cursor is `self.selected`
        // over `self.jobs`; honour it so jumping straight to a detail view uses
        // the highlighted job, not just the newest.
        if matches!(self.view, View::Cockpit | View::Jobs | View::Log) {
            if let Some(j) = self.selected.selected().and_then(|i| self.jobs.get(i)) {
                return Some(j.id.clone());
            }
        }
        if let Some(r) = self.runs.get(self.list_cursor) {
            return Some(r.job_id.clone());
        }
        self.jobs.first().map(|j| j.id.clone())
    }

    /// Move the cursor in whichever list-style view is active.
    fn move_list(&mut self, delta: isize) {
        if self.view == View::Reset {
            // RESET_ROWS is a fixed, non-empty const slice.
            let len = RESET_ROWS.len() as isize;
            self.reset_cursor = (self.reset_cursor as isize + delta).rem_euclid(len) as usize;
            return;
        }
        let len = self.active_list_len();
        if len == 0 {
            return;
        }
        self.list_cursor = (self.list_cursor as isize + delta).rem_euclid(len as isize) as usize;
    }

    /// Filtered list of recipes against the picker's fuzzy query.
    /// Returns `(idx_in_catalog, score)` pairs sorted by score desc —
    /// indices into the supplied `catalog`. Takes the catalog as a param
    /// (rather than `&self`) so callers can pass `&self.catalog` as a
    /// disjoint-field borrow alongside a `&mut self.overlay` in the
    /// picker handler.
    fn filter_recipes(catalog: &[&'static crate::recipes::RecipeDef], query: &str) -> Vec<usize> {
        use fuzzy_matcher::FuzzyMatcher;
        use fuzzy_matcher::skim::SkimMatcherV2;
        let matcher = SkimMatcherV2::default();
        let mut scored: Vec<(usize, i64)> = catalog
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                if query.is_empty() {
                    Some((i, 0))
                } else {
                    matcher.fuzzy_match(r.name, query).map(|s| (i, s))
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| catalog[a.0].name.cmp(catalog[b.0].name))
        });
        scored.into_iter().map(|(i, _)| i).collect()
    }

    fn open_picker(&mut self) {
        self.overlay = Overlay::Picker {
            query: String::new(),
            cursor: 0,
        };
    }

    fn submit_editor(&mut self) {
        let Overlay::Editor {
            recipe,
            fields,
            raw_mode,
            raw_buffer,
            ..
        } = &self.overlay
        else {
            return;
        };
        let recipe = *recipe;
        // The buffer to launch = the raw JSON when the fallback is on, else
        // the per-field form assembled back into a JSON object.
        let buffer = if *raw_mode {
            raw_buffer.clone()
        } else {
            assemble_fields(fields)
        };
        // U3 (F2): run the SAME schema preflight the CLI does (B/P5) BEFORE
        // launching, so a malformed / wrong-typed / missing arg surfaces a
        // precise message in-TUI and the Editor STAYS OPEN to fix — instead
        // of a detached job that only shows `Failed` minutes later. The
        // catalog holds the recipe's `args_schema_fn`; on any reject we set
        // the status and return without spawning.
        if let Err(msg) = self.validate_editor_args(recipe, &buffer) {
            self.set_status(msg);
            return;
        }
        self.spawn_recipe(recipe, &buffer);
        self.overlay = Overlay::None;
    }

    /// Validate an Editor buffer for `recipe` against its arg schema. `Ok` =
    /// safe to launch; `Err(msg)` is a human-facing reason (bad JSON, or a
    /// schema violation) for the status bar. An unknown recipe / missing
    /// schema is permissive (serde + the CLI compile backstop it).
    fn validate_editor_args(&self, recipe: &str, buffer: &str) -> Result<(), String> {
        let raw: serde_json::Value =
            serde_json::from_str(buffer).map_err(|e| format!("args are not valid JSON: {e}"))?;
        if let Some(def) = self.catalog.iter().find(|r| r.name == recipe) {
            let schema = (def.args_schema_fn)();
            crate::recipes::recipe::validate_args_against_schema(recipe, &schema, &raw)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Toggle the Editor's raw-JSON fallback (Ctrl+R). Form→raw seeds the raw
    /// buffer from the current fields. Raw→form re-parses the raw buffer back
    /// into the fields — but ONLY if it's valid JSON. On a parse error we STAY
    /// in raw mode and set a status, so a malformed raw edit can never silently
    /// flip to a stale form (and then launch stale data) — the user is kept on
    /// the JSON they must fix. No-op unless the Editor overlay is open.
    fn toggle_editor_raw_mode(&mut self) {
        let mut parse_err: Option<String> = None;
        if let Overlay::Editor {
            fields,
            raw_mode,
            raw_buffer,
            ..
        } = &mut self.overlay
        {
            if *raw_mode {
                // raw → form: re-derive field VALUES from the edited JSON,
                // keeping the existing field set + types (a typo'd key doesn't
                // spawn phantom rows); unknown keys are appended as `any`.
                match serde_json::from_str::<serde_json::Value>(raw_buffer) {
                    Ok(serde_json::Value::Object(m)) => {
                        let mut seen = std::collections::HashSet::new();
                        for f in fields.iter_mut() {
                            if let Some(v) = m.get(&f.name) {
                                f.value = value_to_field_text(v);
                            } else {
                                f.value.clear();
                            }
                            seen.insert(f.name.clone());
                        }
                        for (k, v) in &m {
                            if !seen.contains(k) {
                                fields.push(EditorField {
                                    name: k.clone(),
                                    ty: "any".into(),
                                    value: value_to_field_text(v),
                                });
                            }
                        }
                        *raw_mode = false;
                    }
                    Ok(_) => {
                        // Valid JSON but not an object (e.g. a bare array) — the
                        // form can't represent it; keep raw mode + warn.
                        parse_err =
                            Some("raw args must be a JSON object to switch to the form".into());
                    }
                    Err(e) => {
                        // Stay in raw mode so the user fixes the JSON; the form
                        // is never shown with stale values.
                        parse_err = Some(format!("raw JSON invalid ({e}) — staying in raw mode"));
                    }
                }
            } else {
                *raw_buffer = assemble_fields(fields);
                *raw_mode = true;
            }
        }
        if let Some(msg) = parse_err {
            self.set_status(msg);
        }
    }

    /// Open the args Editor overlay for a recipe as a PER-FIELD FORM (F2),
    /// built from the recipe's schema `properties` and seeded with the same
    /// prefill the freeform buffer used: the schema-default template ⊕ the
    /// owning cookbook's domain overlay (E2). The serde defaults are the
    /// single source; the overlay only adds domain paths + curated
    /// non-default starts on top. Keeps blut-core domain-agnostic — no
    /// hardcoded paths.
    fn open_editor(&mut self, recipe: &'static crate::recipes::RecipeDef) {
        // The prefill JSON is the source of the seed VALUES; the schema's
        // `properties` is the source of the field SET + per-field type hints.
        let prefill = self.registry.prefill_args(recipe.name);
        let schema = (recipe.args_schema_fn)();
        let fields = build_fields(&schema, &prefill);
        // raw_buffer mirrors the form so flipping to the raw fallback starts
        // from the same content the form shows.
        let raw_buffer = assemble_fields(&fields);
        self.overlay = Overlay::Editor {
            recipe: recipe.name,
            fields,
            focus: 0,
            raw_mode: false,
            raw_buffer,
        };
    }

    /// Open the kind-filtered dataset picker for a recipe (F3/U5). Queries
    /// `datasets_db` for datasets whose `kind` matches the recipe's
    /// `input_kinds`, then floats the selectable list. A recipe with EMPTY
    /// `input_kinds` (graph-input, no dataset to pick) skips straight to the
    /// args editor — unchanged from before this feature. If the db can't be
    /// opened (e.g. no datasets registered yet) we degrade to the editor with
    /// a status note rather than blocking the launch.
    fn open_dataset_picker(&mut self, recipe: &'static crate::recipes::RecipeDef) {
        if recipe.input_kinds.is_empty() {
            self.open_editor(recipe);
            return;
        }
        let datasets = match crate::datasets_db::open()
            .and_then(|conn| crate::datasets_db::list_by_kinds(&conn, recipe.input_kinds))
        {
            Ok(rows) => rows
                .into_iter()
                .map(|r| DatasetChoice {
                    name: r.name,
                    kind: r.kind,
                    source_path: r.source_path.to_string_lossy().into_owned(),
                    n_examples: r.n_examples,
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                // No registry / no matching datasets is not an error worth
                // blocking on — fall through to the editor so the operator can
                // still type a path by hand.
                self.set_status(format!(
                    "dataset picker: {e}; opening args editor (set the input by hand)"
                ));
                self.open_editor(recipe);
                return;
            }
        };
        if datasets.is_empty() {
            self.set_status(format!(
                "no datasets of kind [{}] registered — opening args editor",
                recipe.input_kinds.join(", ")
            ));
            self.open_editor(recipe);
            return;
        }
        self.overlay = Overlay::DatasetPicker {
            recipe,
            datasets,
            cursor: 0,
        };
    }

    /// A dataset was picked in the DatasetPicker (F3): inject it into the
    /// recipe's input arg, then open the per-field editor. The injection
    /// target is schema-driven — the recipe's args schema decides which field
    /// receives the dataset. A `registered_name` field (the
    /// `materialize_dataset_path` convention) gets the dataset's registered
    /// NAME, so the stage re-resolves + re-hashes it from `datasets_db` at run
    /// time; otherwise a `path` / `input` / `dataset` / `lma_root` field gets
    /// the dataset's `source_path`. If the schema declares none of those we
    /// still open the editor (the operator wires the input by hand) and note it
    /// in the status bar. The chosen value is layered onto the prefill so the
    /// editor opens with the dataset already filled in.
    fn pick_dataset(&mut self, recipe: &'static crate::recipes::RecipeDef, choice: &DatasetChoice) {
        let schema = (recipe.args_schema_fn)();
        let prop_names = schema_prop_names(&schema);
        // Prefer the registered-name convention; fall back to a path arg.
        let (field, value) = if prop_names.iter().any(|n| n == "registered_name") {
            ("registered_name", choice.name.clone())
        } else if let Some(p) = ["path", "input", "dataset", "lma_root"]
            .into_iter()
            .find(|c| prop_names.iter().any(|n| n == c))
        {
            (p, choice.source_path.clone())
        } else {
            // No obvious input arg in the schema — open the editor anyway so
            // the launch isn't blocked, and tell the operator.
            self.set_status(format!(
                "picked '{}' but recipe '{}' has no registered_name/path arg — set the input by hand",
                choice.name, recipe.name
            ));
            self.open_editor(recipe);
            return;
        };
        // Build the editor from prefill, then overwrite the input field's row
        // value with the picked dataset.
        self.open_editor(recipe);
        if let Overlay::Editor {
            fields, raw_buffer, ..
        } = &mut self.overlay
        {
            if let Some(row) = fields.iter_mut().find(|f| f.name == field) {
                row.value = value.clone();
            } else {
                // The schema-default template may have omitted an optional
                // field with no default; add it so the picked dataset is
                // actually carried into the args.
                fields.push(EditorField {
                    name: field.to_string(),
                    ty: schema_field_type(&schema, field),
                    value: value.clone(),
                });
            }
            *raw_buffer = assemble_fields(fields);
            self.set_status(format!(
                "input: {field} = '{value}' (from dataset '{}')",
                choice.name
            ));
        }
    }

    /// Spawn `blut recipe run <name> --args '<json>'` detached. Status
    /// bar reports pid; new job appears on next refresh tick.
    fn spawn_recipe(&mut self, name: &str, args_json: &str) {
        let exe = std::env::current_exe().unwrap_or_else(|_| "blut".into());
        match std::process::Command::new(exe)
            .args(["recipe", "run", name, "--args", args_json])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(child) => {
                self.set_status(format!(
                    "spawned '{name}' (pid {}) — focusing its log when it appears…",
                    child.id()
                ));
                // F1: remember the current newest job so refresh_jobs can
                // detect the new one (which lands at index 0, newest-first)
                // and jump straight to its live status.jsonl tail.
                self.focus_baseline_top = self.jobs.first().map(|j| j.id.clone());
                self.pending_focus = true;
                self.focus_ticks_left = FOCUS_GIVE_UP_TICKS;
            }
            Err(e) => self.set_status(format!("spawn '{name}' failed: {e}")),
        }
    }

    /// Auto-assigned recipe hotkeys. Builds the menu dynamically from
    /// the injected catalog (sorted by category + name), assigning hotkeys from
    /// the pool `1..9, then a..z` skipping the reserved keys (q quit,
    /// r refresh, c cancel, R custom-recipe-picker, j/k vi navigation,
    /// l reserved for U3 log toggle). Recipes beyond the available
    /// hotkeys still show in the menu but require `R` to launch.
    fn recipe_menu(
        catalog: &[&'static crate::recipes::RecipeDef],
    ) -> Vec<(Option<char>, &'static crate::recipes::RecipeDef)> {
        // Menu order = each course's pipeline position (ADR 0051).
        // `Course::order()` is a compiler-forced exhaustive match, so a
        // newly-added course can't silently drop to the bottom here.
        let mut sorted: Vec<&'static crate::recipes::RecipeDef> = catalog.to_vec();
        sorted.sort_by(|a, b| {
            a.category
                .order()
                .cmp(&b.category.order())
                .then_with(|| a.name.cmp(b.name))
        });
        // Reserved keys: the cockpit built-ins (q/r/c/R), vi nav (j/k/l), and
        // back-nav (b) — excluded from the recipe-hotkey pool so a recipe can
        // never shadow a navigation key (lowercase + uppercase both reserved).
        let reserved: &[char] = &['q', 'Q', 'r', 'R', 'c', 'C', 'j', 'k', 'l', 'b', 'B'];
        let mut pool: Vec<char> = ('1'..='9').collect();
        pool.extend('a'..='z');
        pool.retain(|c| !reserved.contains(c));
        let mut menu = Vec::with_capacity(sorted.len());
        let mut pool_iter = pool.into_iter();
        for r in sorted {
            menu.push((pool_iter.next(), r));
        }
        menu
    }

    /// Compact built-in keymap displayed in the status bar.
    fn builtin_keys() -> &'static [(char, BuiltinAction, &'static str)] {
        &[
            ('q', BuiltinAction::Quit, "quit"),
            ('r', BuiltinAction::Refresh, "refresh"),
            ('c', BuiltinAction::Cancel, "cancel job"),
            ('R', BuiltinAction::OpenPicker, "custom recipe"),
        ]
    }

    fn refresh_jobs(&mut self) {
        match jobs::list_jobs() {
            Ok(mut js) => {
                // Newest first (jobs::list_jobs already sorts by id desc but be defensive).
                js.sort_by(|a, b| b.id.cmp(&a.id));
                self.jobs = js;
                if self.jobs.is_empty() {
                    self.selected.select(None);
                } else {
                    let cur = self.selected.selected().unwrap_or(0);
                    self.selected.select(Some(cur.min(self.jobs.len() - 1)));
                }
                // F1: a TUI-launched spawn is pending focus — when its job
                // surfaces (a new newest id at index 0, distinct from the
                // pre-spawn baseline), select it + jump to the Log tail.
                if self.pending_focus {
                    let appeared = self
                        .jobs
                        .first()
                        .is_some_and(|top| Some(&top.id) != self.focus_baseline_top.as_ref());
                    if appeared {
                        self.pending_focus = false;
                        self.selected.select(Some(0));
                        self.set_view(View::Log);
                        self.refresh_log();
                    } else {
                        // Bound the wait: give up after FOCUS_GIVE_UP_TICKS so a
                        // spawn whose job never surfaces can't hijack focus onto
                        // an unrelated job that appears much later.
                        self.focus_ticks_left = self.focus_ticks_left.saturating_sub(1);
                        if self.focus_ticks_left == 0 {
                            self.pending_focus = false;
                        }
                    }
                }
            }
            Err(e) => self.set_status(format!("list_jobs failed: {e}")),
        }
    }

    /// Rebuild the engine-console model. Starts from the representative demo,
    /// then overlays the newest RUNNING job's real DAG / cache / phase from its
    /// `status.jsonl` (the mesh / broker / ε loaders land in the next slice, so
    /// those panels stay representative until then).
    fn refresh_console(&mut self) {
        let mut m = console::ConsoleModel::demo();
        let job = self
            .jobs
            .iter()
            .find(|j| matches!(j.state, JobState::Running))
            .or_else(|| self.jobs.first());
        if let Some(job) = job
            && let Ok(dir) = jobs::job_dir_path(&job.id)
            && m.apply_status_jsonl(&dir.join("status.jsonl"))
        {
            m.run_id = job.id.chars().take(8).collect();
            if let Some(name) = &job.output_name {
                m.plan = name.clone();
            }
        }
        // Overlay the real mesh + ε budgets when the p2p plane is compiled in.
        #[cfg(feature = "p2p")]
        {
            m.apply_mesh();
            m.apply_privacy();
        }
        self.console = m;
    }

    fn refresh_system(&mut self) {
        self.system = system::SystemSnapshot::probe();
    }

    fn refresh_log(&mut self) {
        let Some(idx) = self.selected.selected() else {
            self.log_lines.clear();
            self.log_job_id = None;
            return;
        };
        let Some(job) = self.jobs.get(idx) else {
            return;
        };
        match jobs::read_status(&job.id) {
            Ok(updates) => {
                let rendered = jobs::render_log(&updates);
                let lines: Vec<String> = rendered
                    .lines()
                    .rev()
                    .take(MAX_LOG_LINES)
                    .map(String::from)
                    .collect();
                self.log_lines = lines.into_iter().rev().collect();
                self.log_job_id = Some(job.id.clone());
            }
            Err(e) => {
                self.log_lines = vec![format!("read_status({}): {e}", job.id)];
                self.log_job_id = Some(job.id.clone());
            }
        }
    }

    fn move_selection(&mut self, delta: isize) {
        if self.jobs.is_empty() {
            self.selected.select(None);
            return;
        }
        let n = self.jobs.len() as isize;
        let cur = self.selected.selected().unwrap_or(0) as isize;
        let next = ((cur + delta).rem_euclid(n)) as usize;
        self.selected.select(Some(next));
        self.refresh_log();
    }

    fn set_status(&mut self, msg: impl Into<String>) {
        self.status_msg = Some((msg.into(), Instant::now()));
    }

    fn cancel_selected(&mut self) {
        let Some(idx) = self.selected.selected() else {
            self.set_status("no job selected");
            return;
        };
        let Some(job) = self.jobs.get(idx) else {
            return;
        };
        if !matches!(job.state, JobState::Running) {
            self.set_status(format!(
                "job {} is {} — nothing to cancel",
                job.id,
                job.state.as_str()
            ));
            return;
        }
        let id = job.id.clone();
        self.set_status(format!("SIGTERM {id} (grace 10s)..."));
        // Spawn detached process so we don't block the UI on grace
        // period; user sees state flip on next refresh.
        let _ =
            std::process::Command::new(std::env::current_exe().unwrap_or_else(|_| "blut".into()))
                .args(["cancel", &id])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();
    }

    /// Refresh every live data source: jobs, system probes, log tail,
    /// and — if a diagnostic view is active — its scanned rows.
    fn refresh_all(&mut self) {
        self.refresh_jobs();
        self.refresh_console();
        self.refresh_system();
        self.refresh_log();
        // Re-pull whatever the active view shows (keeps a running job's DAG /
        // metrics live).
        let job = self.current_job_id();
        self.load_view_data(self.view, job.as_deref());
        self.last_refresh = Instant::now();
    }

    /// Toggle the run under the list cursor in/out of the Compare selection
    /// (History / Leaderboard views). Capped at 3 marks. `marked` holds job
    /// ids.
    fn toggle_mark(&mut self) {
        let Some(row) = self.runs.get(self.list_cursor) else {
            return;
        };
        let id = row.job_id.clone();
        if let Some(pos) = self.marked.iter().position(|n| *n == id) {
            self.marked.remove(pos);
            self.set_status(format!("unmarked {id}"));
        } else if self.marked.len() >= 3 {
            self.set_status("compare holds at most 3 runs — unmark one first");
        } else {
            self.marked.push(id.clone());
            self.set_status(format!("marked {id} for compare ({}/3)", self.marked.len()));
        }
    }

    /// Fire the armed Reset action (two-press confirm). First press on a
    /// row arms it + shows a confirm hint; a second Enter within
    /// [`RESET_WINDOW`] runs the destructive action.
    fn fire_reset(&mut self) {
        let idx = self.reset_cursor;
        let Some(action) = RESET_ROWS.get(idx).copied() else {
            return;
        };
        let armed = self
            .reset_armed
            .map(|(i, t)| i == idx && t.elapsed() < RESET_WINDOW)
            .unwrap_or(false);
        if armed {
            self.reset_armed = None;
            let job = self.current_job_id();
            let msg = views::reset(action, job.as_deref());
            self.set_status(msg);
        } else {
            self.reset_armed = Some((idx, Instant::now()));
            self.set_status(format!(
                "{}: press Enter again within {}s to confirm",
                action.label(),
                RESET_WINDOW.as_secs()
            ));
        }
    }
}

// ── F2 per-field form helpers (free fns, schema/JSON only) ──────────────
//
// These bridge the recipe's args JSON schema + the prefill JSON to the
// editable form rows and back. They are pure (no &self) so the render code
// and the key handler can call them with disjoint borrows.

/// Resolve a `schema_of`-shaped schema (`{"$ref":"#/definitions/<Name>",
/// "definitions":{…}}`) to its root args object (the `<Name>` definition,
/// falling back to `"Args"`). Mirrors `recipe::schema_root` (which is
/// private to that module).
fn schema_args_root(
    schema: &serde_json::Value,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    let defs = schema.get("definitions").and_then(|d| d.as_object());
    if let Some(defs) = defs {
        let name = schema
            .get("$ref")
            .and_then(|r| r.as_str())
            .and_then(|r| r.rsplit('/').next())
            .unwrap_or("Args");
        if let Some(obj) = defs
            .get(name)
            .or_else(|| defs.get("Args"))
            .and_then(|d| d.as_object())
        {
            return Some(obj);
        }
    }
    // An inline (non-$ref) object schema — used by fixtures + simple recipes.
    schema.as_object().filter(|o| o.contains_key("properties"))
}

/// Ordered list of the recipe schema's top-level property names. Used by the
/// dataset-picker injection to find the recipe's input arg.
fn schema_prop_names(schema: &serde_json::Value) -> Vec<String> {
    schema_args_root(schema)
        .and_then(|root| root.get("properties"))
        .and_then(|p| p.as_object())
        .map(|props| props.keys().cloned().collect())
        .unwrap_or_default()
}

/// The scalar `type` string a schema declares for `field` (e.g. `"number"`),
/// or `"any"` when it declares none / a union (`Option<T>` → `["T","null"]`).
fn schema_field_type(schema: &serde_json::Value, field: &str) -> String {
    schema_args_root(schema)
        .and_then(|root| root.get("properties"))
        .and_then(|p| p.as_object())
        .and_then(|props| props.get(field))
        .and_then(|d| d.get("type"))
        .and_then(|t| t.as_str())
        .map(String::from)
        .unwrap_or_else(|| "any".into())
}

/// Build the per-field form rows for a recipe from its schema `properties`,
/// seeding each row's value from the prefill JSON (the schema-default
/// template ⊕ cookbook overlay). Field ORDER follows the schema's
/// `properties` order so the form reads like the recipe's `Args` struct.
/// A field present in the prefill but absent from `properties` (e.g. an
/// overlay-only key) is appended after the declared fields so nothing the
/// prefill set is silently dropped.
fn build_fields(schema: &serde_json::Value, prefill: &str) -> Vec<EditorField> {
    let prefill_obj: serde_json::Map<String, serde_json::Value> =
        match serde_json::from_str::<serde_json::Value>(prefill) {
            Ok(serde_json::Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
    let mut rows = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(props) = schema_args_root(schema)
        .and_then(|root| root.get("properties"))
        .and_then(|p| p.as_object())
    {
        for (name, def) in props {
            let ty = def
                .get("type")
                .and_then(|t| t.as_str())
                .map(String::from)
                .unwrap_or_else(|| "any".into());
            // Seed from the prefill value (already the merged default), else
            // leave blank — a blank optional field is omitted on assembly.
            let value = prefill_obj
                .get(name)
                .map(value_to_field_text)
                .unwrap_or_default();
            rows.push(EditorField {
                name: name.clone(),
                ty,
                value,
            });
            seen.insert(name.clone());
        }
    }
    // Overlay-only / extra prefill keys not declared in the schema.
    for (name, v) in &prefill_obj {
        if !seen.contains(name) {
            rows.push(EditorField {
                name: name.clone(),
                ty: "any".into(),
                value: value_to_field_text(v),
            });
        }
    }
    rows
}

/// Render a JSON value as the text a form row shows. Strings drop their
/// surrounding quotes (the operator edits the raw text); everything else is
/// its compact JSON form (so `true` / `8` / `["a","b"]` round-trip).
fn value_to_field_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Parse a form row's text back into a typed JSON value, guided by the
/// field's schema `type`. The point is to NOT make every value a string:
///   * `number`/`integer` → a JSON number (falls back to a string if the
///     text isn't numeric, so the schema preflight reports the precise
///     "expected number, got string" rather than a parse panic);
///   * `boolean` → `true`/`false` (case-insensitive), else string;
///   * `array`/`object` → parsed JSON (else string, so malformed JSON is
///     caught by the preflight, not here);
///   * `string` → the text verbatim;
///   * `any`/union → best-effort `serde_json::from_str`, else a string (so
///     an Option<number> typed `8` becomes `8`, while free text stays text).
fn field_value_json(ty: &str, text: &str) -> serde_json::Value {
    use serde_json::Value;
    let t = text.trim();
    match ty {
        "string" => Value::String(text.to_string()),
        "number" | "integer" => serde_json::from_str::<serde_json::Number>(t)
            .map(Value::Number)
            .unwrap_or_else(|_| Value::String(text.to_string())),
        "boolean" => match t.to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(text.to_string()),
        },
        "array" | "object" => {
            serde_json::from_str::<Value>(t).unwrap_or_else(|_| Value::String(text.to_string()))
        }
        // "any" / unknown / union (Option) — try JSON, fall back to string.
        _ => serde_json::from_str::<Value>(t).unwrap_or_else(|_| Value::String(text.to_string())),
    }
}

/// Assemble the form rows back into a pretty-printed args JSON object. A row
/// left BLANK is omitted (so an untouched optional field doesn't force a
/// `null` / empty value into the args — the recipe's serde default applies).
/// Blank strings are omitted too; a recipe that genuinely needs an empty
/// string is vanishingly rare and the raw-JSON fallback covers it.
fn assemble_fields(fields: &[EditorField]) -> String {
    let mut map = serde_json::Map::new();
    for f in fields {
        if f.value.trim().is_empty() {
            continue;
        }
        map.insert(f.name.clone(), field_value_json(&f.ty, &f.value));
    }
    serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_else(|_| "{}".into())
}

/// Entrypoint registered as `blut tui`. The caller (the cookbook binary)
/// supplies the composed cookbook [`Registry`]; the cockpit's recipe
/// catalog comes from it, not a static slice.
pub async fn run(registry: crate::framework::Registry) -> Result<()> {
    // Detect NO_COLOR / TERM=dumb / locale once before the first draw so
    // every theme getter returns the right style.
    theme::detect("auto", "auto");
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).context("alt screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend).context("terminal")?;

    let result = run_app(&mut term, registry).await;

    // Always restore the terminal, even on error.
    disable_raw_mode().ok();
    execute!(
        term.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .ok();
    term.show_cursor().ok();
    result
}

/// `blut tui --check` (P12 / F6): a REAL self-check in the shipped binary, not a
/// unit-test-only affordance. Builds the App from the live registry and renders
/// every view headless to a `TestBackend`, asserting each produces a non-blank
/// buffer; exits 0 on success. Lets CI / an operator verify the cockpit builds
/// + every view draws without entering raw mode.
pub fn check(registry: crate::framework::Registry) -> Result<()> {
    use ratatui::backend::TestBackend;
    let views = [
        View::Console,
        View::Cockpit,
        View::Jobs,
        View::Log,
        View::System,
        View::History,
        View::Leaderboard,
        View::Compare,
        View::Dag,
        View::Lineage,
        View::Artifacts,
        View::Metrics,
        View::Catalog,
        View::Reset,
    ];
    let mut app = App::new(registry);
    app.refresh_jobs();
    let mut term = Terminal::new(TestBackend::new(120, 40)).context("test terminal")?;
    for view in views {
        app.view = view;
        term.draw(|f| draw(f, &mut app))
            .with_context(|| format!("draw view {view:?}"))?;
        let blank = term
            .backend()
            .buffer()
            .content()
            .iter()
            .all(|c| c.symbol().trim().is_empty());
        if blank {
            anyhow::bail!("tui --check: view {view:?} rendered a completely blank buffer");
        }
    }

    // Also render every modal OVERLAY headless so a future break in the recipe
    // Picker, the F2 per-field args form (+ its raw-JSON fallback), or the F3
    // dataset picker is caught by `tui --check`, not just by unit tests.
    app.view = View::Cockpit;
    let mut overlays_checked = 0usize;
    // Recipe picker.
    app.open_picker();
    overlays_checked += render_overlay_check(&mut term, &mut app, "Picker")?;
    // The per-field form + raw fallback + dataset picker need a recipe; only
    // exercise them when the live catalog has one (blut-core's bare registry
    // may be empty — then the overlay paths are covered by the unit fixtures).
    if let Some(recipe) = app.catalog.first().copied() {
        app.open_editor(recipe);
        overlays_checked += render_overlay_check(&mut term, &mut app, "Editor(form)")?;
        // Flip to the raw-JSON fallback and render that surface too.
        if let Overlay::Editor {
            raw_mode,
            raw_buffer,
            fields,
            ..
        } = &mut app.overlay
        {
            *raw_buffer = assemble_fields(fields);
            *raw_mode = true;
        }
        overlays_checked += render_overlay_check(&mut term, &mut app, "Editor(raw)")?;
        // Dataset picker with a synthetic row (no DB touch) so the draw path
        // is exercised even on a box with no registered datasets.
        app.overlay = Overlay::DatasetPicker {
            recipe,
            datasets: vec![DatasetChoice {
                name: "example".into(),
                kind: recipe
                    .input_kinds
                    .first()
                    .copied()
                    .unwrap_or("dataset")
                    .into(),
                source_path: "/path/to/example.jsonl".into(),
                n_examples: 1,
            }],
            cursor: 0,
        };
        overlays_checked += render_overlay_check(&mut term, &mut app, "DatasetPicker")?;
    }
    app.overlay = Overlay::None;

    println!(
        "blut tui --check: OK ({} views + {overlays_checked} overlays render)",
        views.len()
    );
    Ok(())
}

/// Render the current frame (whatever overlay is set on `app`) to the test
/// backend and assert it isn't completely blank. Returns 1 so callers can sum
/// a count. Helper for [`check`] so the overlay smoke-checks stay DRY.
fn render_overlay_check(
    term: &mut Terminal<ratatui::backend::TestBackend>,
    app: &mut App,
    label: &str,
) -> Result<usize> {
    term.draw(|f| draw(f, app))
        .with_context(|| format!("draw overlay {label}"))?;
    let blank = term
        .backend()
        .buffer()
        .content()
        .iter()
        .all(|c| c.symbol().trim().is_empty());
    if blank {
        anyhow::bail!("tui --check: overlay {label} rendered a completely blank buffer");
    }
    Ok(1)
}

async fn run_app<B: ratatui::backend::Backend>(
    term: &mut Terminal<B>,
    registry: crate::framework::Registry,
) -> Result<()> {
    let mut app = App::new(registry);
    app.refresh_jobs();
    app.refresh_system();
    app.refresh_log();

    while !app.quit {
        // Stale status banner cleared after 3 s.
        if let Some((_, when)) = &app.status_msg {
            if when.elapsed() > Duration::from_secs(3) {
                app.status_msg = None;
            }
        }
        // Periodic auto-refresh (jobs + system) — log tail re-fetched
        // on Enter / selection change to avoid blocking the loop on
        // every tick.
        if app.last_refresh.elapsed() >= REFRESH_TICK {
            // Full refresh: jobs + system + log + the ACTIVE view's data, so a
            // running job's DAG / Metrics / Lineage / Artifacts panels stay live
            // without a manual `r`. (`refresh_all` stamps `last_refresh`.)
            app.refresh_all();
        }

        term.draw(|f| draw(f, &mut app))
            .map_err(|e| anyhow::anyhow!("draw: {e}"))?;

        // Poll with a short deadline so the loop body runs at refresh
        // cadence even when no key is pressed.
        if event::poll(Duration::from_millis(200))
            .map_err(|e| anyhow::anyhow!("event::poll: {e}"))?
        {
            if let Event::Key(k) = event::read().map_err(|e| anyhow::anyhow!("event::read: {e}"))? {
                handle_key(&mut app, k);
            }
        }
    }
    Ok(())
}

fn handle_key(app: &mut App, k: event::KeyEvent) {
    // Ctrl-C always quits, regardless of overlay state.
    if k.modifiers.contains(KeyModifiers::CONTROL) && matches!(k.code, KeyCode::Char('c')) {
        app.quit = true;
        return;
    }
    match &mut app.overlay {
        Overlay::Editor {
            fields,
            focus,
            raw_mode,
            raw_buffer,
            ..
        } => {
            // Ctrl+R toggles the raw-JSON fallback (F2 keeps a raw escape
            // hatch, but the per-field form is the default). Handled in a
            // method so a parse failure on raw→form can set a status message
            // (needs &mut self, not just the overlay borrow).
            if matches!(k.code, KeyCode::Char('r')) && k.modifiers.contains(KeyModifiers::CONTROL) {
                app.toggle_editor_raw_mode();
                return;
            }
            if *raw_mode {
                // Freeform JSON buffer (the fallback). Same keys as the old
                // single-buffer editor: type / Backspace edit, Enter inserts a
                // newline, Ctrl/Shift+Enter submits, Esc cancels.
                match k.code {
                    KeyCode::Esc => app.overlay = Overlay::None,
                    KeyCode::Enter
                        if k.modifiers.contains(KeyModifiers::CONTROL)
                            || k.modifiers.contains(KeyModifiers::SHIFT) =>
                    {
                        app.submit_editor();
                    }
                    KeyCode::Enter | KeyCode::Char('\n') => raw_buffer.push('\n'),
                    KeyCode::Backspace => {
                        raw_buffer.pop();
                    }
                    KeyCode::Char(c) => raw_buffer.push(c),
                    _ => {}
                }
            } else {
                // Per-field form (DEFAULT). ↑↓/Tab move between rows; typing
                // edits the focused row's value; Backspace deletes a char;
                // Ctrl/Shift+Enter (or plain Enter) submits.
                let n = fields.len();
                match k.code {
                    KeyCode::Esc => app.overlay = Overlay::None,
                    KeyCode::Up => {
                        if n > 0 {
                            *focus = (*focus + n - 1) % n;
                        }
                    }
                    KeyCode::Down | KeyCode::Tab => {
                        if n > 0 {
                            *focus = (*focus + 1) % n;
                        }
                    }
                    KeyCode::BackTab => {
                        if n > 0 {
                            *focus = (*focus + n - 1) % n;
                        }
                    }
                    KeyCode::Enter => app.submit_editor(),
                    KeyCode::Backspace => {
                        if let Some(f) = fields.get_mut(*focus) {
                            f.value.pop();
                        }
                    }
                    KeyCode::Char(c) => {
                        if let Some(f) = fields.get_mut(*focus) {
                            f.value.push(c);
                        }
                    }
                    _ => {}
                }
            }
        }
        Overlay::DatasetPicker {
            recipe,
            datasets,
            cursor,
        } => match k.code {
            // Esc SKIPS the picker → open the editor with no dataset injected
            // (recipe input typed by hand). Matches the "Esc to skip" contract.
            KeyCode::Esc => {
                let recipe = *recipe;
                app.open_editor(recipe);
            }
            KeyCode::Up | KeyCode::Char('k') => *cursor = cursor.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                let last = datasets.len().saturating_sub(1);
                *cursor = (*cursor + 1).min(last);
            }
            KeyCode::Enter => {
                let recipe = *recipe;
                if let Some(choice) = datasets.get(*cursor).cloned() {
                    app.pick_dataset(recipe, &choice);
                }
            }
            _ => {}
        },
        Overlay::Picker { query, cursor } => match k.code {
            KeyCode::Esc => app.overlay = Overlay::None,
            KeyCode::Up | KeyCode::Char('k') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                *cursor = cursor.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                // TUI-07: clamp to the last filtered row so Down-past-end
                // stays in range (and Enter never no-ops on a phantom row).
                let last = App::filter_recipes(&app.catalog, query)
                    .len()
                    .saturating_sub(1);
                *cursor = (*cursor + 1).min(last);
            }
            KeyCode::Up => *cursor = cursor.saturating_sub(1),
            KeyCode::Down => {
                // TUI-07: clamp to the last filtered row.
                let last = App::filter_recipes(&app.catalog, query)
                    .len()
                    .saturating_sub(1);
                *cursor = (*cursor + 1).min(last);
            }
            KeyCode::Backspace => {
                query.pop();
                *cursor = 0;
            }
            KeyCode::Char(c) => {
                query.push(c);
                *cursor = 0;
            }
            KeyCode::Enter => {
                let filtered = App::filter_recipes(&app.catalog, query);
                if let Some(idx) = filtered.get(*cursor) {
                    let recipe = app.catalog[*idx];
                    // F3/U5: route through the kind-filtered dataset picker
                    // (which itself skips straight to the editor when the
                    // recipe has no input_kinds).
                    app.open_dataset_picker(recipe);
                }
            }
            _ => {}
        },
        Overlay::None => handle_key_main(app, k),
    }
}

/// Key handling for the non-overlay (main) state. Dispatches first on
/// the active [`View`]: the Cockpit view keeps the original behavior
/// (jobs nav + recipe hotkeys + view switches); the detail views handle
/// their own list nav + actions and route Esc/b back to the cockpit.
fn handle_key_main(app: &mut App, k: event::KeyEvent) {
    // ── View switching (works from any view) ───────────────────────
    // Capital letters jump straight to a detail view. Chosen so they
    // don't collide with the lowercase recipe-hotkey pool (1-9,a-z).
    match k.code {
        KeyCode::Char('E') => return app.set_view(View::Console),
        KeyCode::Char('K') => return app.set_view(View::Cockpit),
        KeyCode::Char('J') => return app.set_view(View::Jobs),
        KeyCode::Char('L') => return app.set_view(View::Log),
        KeyCode::Char('Y') => return app.set_view(View::System),
        KeyCode::Char('H') => return app.set_view(View::History),
        KeyCode::Char('B') => return app.set_view(View::Leaderboard),
        KeyCode::Char('C') => return app.set_view(View::Compare),
        KeyCode::Char('G') => return app.set_view(View::Dag),
        KeyCode::Char('I') => return app.set_view(View::Lineage),
        KeyCode::Char('A') => return app.set_view(View::Artifacts),
        KeyCode::Char('M') => return app.set_view(View::Metrics),
        KeyCode::Char('P') => return app.set_view(View::Catalog),
        KeyCode::Char('X') => return app.set_view(View::Reset),
        // Ctrl-C handled by caller; q always quits.
        KeyCode::Char('q') => {
            app.quit = true;
            return;
        }
        _ => {}
    }

    if app.view == View::Cockpit {
        handle_key_cockpit(app, k);
    } else {
        handle_key_detail(app, k);
    }
}

/// Cockpit-view keys: the original single-column overview behavior.
fn handle_key_cockpit(app: &mut App, k: event::KeyEvent) {
    match k.code {
        // Esc quits from the cockpit (familiar from lml TUI).
        KeyCode::Esc => app.quit = true,
        KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
        KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
        KeyCode::Enter => {
            // Enter on the cockpit opens the Log view for the selected
            // job (revives draw_log) rather than silently refreshing.
            app.refresh_log();
            app.set_view(View::Log);
        }
        KeyCode::Char(c) => {
            // Built-ins (reserved chars per `recipe_menu`).
            let builtin = App::builtin_keys().iter().find(|(k, _, _)| *k == c);
            if let Some((_, action, _)) = builtin {
                match action {
                    BuiltinAction::Refresh => {
                        app.refresh_all();
                        app.set_status("refreshed");
                    }
                    BuiltinAction::Quit => app.quit = true,
                    BuiltinAction::Cancel => app.cancel_selected(),
                    BuiltinAction::OpenPicker => app.open_picker(),
                }
                return;
            }
            // Recipe hotkeys (auto-assigned per category order). Routes
            // through the kind-filtered dataset picker (F3/U5) when the recipe
            // declares input_kinds, else straight to the per-field args editor
            // (F2) prefilled with pre-baked defaults / the schema template.
            let menu = App::recipe_menu(&app.catalog);
            if let Some((_, recipe)) = menu.iter().find(|(k, _)| *k == Some(c)) {
                app.open_dataset_picker(recipe);
            }
        }
        _ => {}
    }
}

/// Detail-view keys: list navigation + per-view actions; Esc/b returns
/// to the cockpit.
fn handle_key_detail(app: &mut App, k: event::KeyEvent) {
    match k.code {
        KeyCode::Esc | KeyCode::Char('b') => app.set_view(View::Console),
        KeyCode::Up | KeyCode::Char('k') => {
            if matches!(app.view, View::Jobs | View::Log) {
                app.move_selection(-1);
            } else {
                app.move_list(-1);
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if matches!(app.view, View::Jobs | View::Log) {
                app.move_selection(1);
            } else {
                app.move_list(1);
            }
        }
        KeyCode::Char('r') => {
            app.refresh_all();
            app.set_status("refreshed");
        }
        KeyCode::Char('c') if matches!(app.view, View::Jobs | View::Log) => {
            app.cancel_selected();
        }
        KeyCode::Enter => match app.view {
            View::Jobs => {
                app.refresh_log();
                app.set_view(View::Log);
            }
            // Drill from a run into its plan DAG.
            View::History | View::Leaderboard => app.set_view(View::Dag),
            View::Reset => app.fire_reset(),
            _ => {}
        },
        // History / Leaderboard: [m]ark a run for Compare; [space] too.
        KeyCode::Char('m') | KeyCode::Char(' ')
            if matches!(app.view, View::History | View::Leaderboard) =>
        {
            app.toggle_mark();
        }
        _ => {}
    }
}

/// Pure state-transition tests for the cockpit (§5.9). These drive the
/// *key handler* + the pure helper functions directly — no terminal, no
/// raw mode, no live process / filesystem probes. They assert the
/// in-memory `App` state after each synthetic key, the fuzzy-filter
/// ordering, the schema→JSON templates, and the recipe-menu hotkey
/// assignment invariants.
#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;

/// V2 validation harness: drive the cockpit headless with POPULATED + adversarial
/// data through every view, every terminal size, and long key sequences, asserting
/// it never panics and never renders a blank frame. The other test modules cover
/// the EMPTY-state drawers; this one fills the per-view caches (runs / artifacts /
/// dag / lineage / metrics / compare) with synthetic engine data — including the
/// nasty cases (NaN/inf metrics, 300-char names, unicode, missing hashes, a 1×1
/// terminal) — so a layout/format/overflow bug surfaces here, not in production.
#[cfg(test)]
#[path = "harness_tests.rs"]
mod harness_tests;
