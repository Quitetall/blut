//! `blut tui` — the single, complete interactive training cockpit.
//!
//! This is the canonical training cockpit for the whole project. It
//! is a superset of the three cockpits it replaced:
//!   * the lamquant-lossless hub `CockpitPanel` (the in-process Rust
//!     probe screen the hub used to render at SCREEN_TRAIN), and
//!   * the retired Python `legacy/python_cockpit/cockpit.py`
//!     (1381 LOC — run history, leaderboard, compare, checkpoints,
//!     presets, hparams, reset, export, live metrics), and
//!   * the original T1/T2 BLUT scaffold (jobs + log + system + recipe
//!     launcher).
//!
//! The hub's "Train a model" tile now execs `blut tui` directly (no
//! more in-process duplicate cockpit).
//!
//! ## Views
//!
//! A `View` enum multiplexes the single ratatui surface across the
//! migrated screens. The default `Cockpit` view is the single-column
//! overview (header / pipeline status / resources / recipe menu). The
//! other views revive the formerly-dead detail panels + port the
//! Python cockpit's diagnostic screens:
//!   * `Jobs`        — full jobs LIST (all states, color-coded)
//!   * `Log`         — per-job status.jsonl tail of the selected job
//!   * `System`      — full GPU/MEM/DISK/CPU probe panel
//!   * `History`     — `training_logs/*.csv` run history
//!   * `Leaderboard` — runs ranked by best validation R
//!   * `Compare`     — side-by-side metric table for marked runs
//!   * `Checkpoints` — `.ckpt` browser grouped by dir
//!   * `Presets`     — preset catalog + hyperparameter reference
//!   * `Metrics`     — live tail of the newest training-log CSV
//!   * `Reset`       — destructive maintenance (tmux/numba/logs)
//!
//! ## Keybindings (cockpit view)
//!   ↑ / ↓ / j / k   move selection (jobs/list rows)
//!   Enter           refresh log / open selected
//!   r               refresh jobs + system
//!   c               cancel selected job (SIGTERM via `blut cancel`)
//!   R               custom-recipe fuzzy picker
//!   `<recipe hotkey>` launch a recipe (pre-baked LamQuant defaults
//!                   for the lamquant_* recipes, schema template else)
//!   J/L/Y/H/B/K/P/M/X  switch to Jobs/Log/sYstem/History/leaderBoard/
//!                      checKpoints/Presets/Metrics/reset views
//!   Esc / b         back to cockpit view (or quit from cockpit)
//!   q / Ctrl-C      quit
//!
//! Lift origin: design borrowed from `lamquant-core/src/tui/panels/`
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

mod system;
mod theme;
mod views;

/// Which surface the single ratatui frame is currently showing.
/// `Cockpit` is the default single-column overview; the rest are the
/// detail panels (revived from the formerly-dead `draw_jobs`/`draw_log`
/// /`draw_system`) and the screens ported from the Python cockpit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Cockpit,
    Jobs,
    Log,
    System,
    History,
    Leaderboard,
    Compare,
    Checkpoints,
    Presets,
    Metrics,
    Reset,
}

impl View {
    fn title(self) -> &'static str {
        match self {
            View::Cockpit => "Training Cockpit",
            View::Jobs => "Jobs",
            View::Log => "Job Log",
            View::System => "System",
            View::History => "Run History",
            View::Leaderboard => "Leaderboard",
            View::Compare => "Compare Runs",
            View::Checkpoints => "Checkpoints",
            View::Presets => "Presets & Hyperparameters",
            View::Metrics => "Live Metrics",
            View::Reset => "Reset Training State",
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
    /// Repo root the diagnostic views scan (training_logs/, runs/,
    /// checkpoints/, weights/). Resolved once at startup.
    repo_root: std::path::PathBuf,
    /// Cached run-history / leaderboard rows + checkpoint rows. Re-read
    /// on view-entry + refresh so the panels aren't I/O-bound per draw.
    runs: Vec<views::RunRow>,
    ckpts: Vec<views::CkptRow>,
    /// Cursor + multi-select for list-style views (history/leaderboard/
    /// checkpoints). `marked` holds run names selected for Compare.
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

/// Two-press confirm window for the destructive Reset actions, matching
/// the lamquant-lossless cockpit's `RESET_WINDOW_SECS`.
const RESET_WINDOW: Duration = Duration::from_secs(3);

/// Reset-view rows: the three destructive maintenance actions plus the
/// non-destructive export action.
const RESET_ROWS: &[views::ResetAction] = &[
    views::ResetAction::KillTmux,
    views::ResetAction::ClearNumba,
    views::ResetAction::ClearLogs,
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
            jobs: Vec::new(),
            selected,
            log_lines: Vec::new(),
            log_job_id: None,
            system: system::SystemSnapshot::default(),
            last_refresh: Instant::now() - REFRESH_TICK,
            status_msg: None,
            overlay: Overlay::None,
            quit: false,
            view: View::Cockpit,
            repo_root: views::repo_root(),
            runs: Vec::new(),
            ckpts: Vec::new(),
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
        self.view = view;
        self.list_cursor = 0;
        match view {
            View::History | View::Leaderboard | View::Compare => {
                self.runs = if view == View::Leaderboard {
                    views::leaderboard(&self.repo_root)
                } else {
                    views::run_history(&self.repo_root)
                };
            }
            View::Checkpoints => {
                self.ckpts = views::checkpoints(&self.repo_root);
            }
            View::Reset => {
                self.reset_cursor = 0;
                self.reset_armed = None;
            }
            _ => {}
        }
        self.set_status(format!("view: {}", view.title()));
    }

    /// Move the cursor in whichever list-style view is active.
    fn move_list(&mut self, delta: isize) {
        let len = match self.view {
            View::History | View::Leaderboard | View::Compare => self.runs.len(),
            View::Checkpoints => self.ckpts.len(),
            View::Reset => RESET_ROWS.len(),
            _ => 0,
        };
        if len == 0 {
            return;
        }
        if self.view == View::Reset {
            let cur = self.reset_cursor as isize;
            self.reset_cursor = (cur + delta).rem_euclid(len as isize) as usize;
            return;
        }
        let cur = self.list_cursor as isize;
        self.list_cursor = (cur + delta).rem_euclid(len as isize) as usize;
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
        let raw: serde_json::Value = serde_json::from_str(buffer)
            .map_err(|e| format!("args are not valid JSON: {e}"))?;
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
    fn pick_dataset(
        &mut self,
        recipe: &'static crate::recipes::RecipeDef,
        choice: &DatasetChoice,
    ) {
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
        // Reserved keys: q, Q, r, R, c, C, j, k, l (lowercase / uppercase
        // map to the same action so we exclude both cases).
        let reserved: &[char] = &['q', 'Q', 'r', 'R', 'c', 'C', 'j', 'k', 'l'];
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
        self.refresh_system();
        self.refresh_log();
        match self.view {
            View::History | View::Compare => self.runs = views::run_history(&self.repo_root),
            View::Leaderboard => self.runs = views::leaderboard(&self.repo_root),
            View::Checkpoints => self.ckpts = views::checkpoints(&self.repo_root),
            _ => {}
        }
        self.last_refresh = Instant::now();
    }

    /// Toggle the run under the list cursor in/out of the Compare
    /// selection (History / Leaderboard views). Capped at 3 marks to
    /// match the Python cockpit's compare-up-to-3 behavior.
    fn toggle_mark(&mut self) {
        let Some(row) = self.runs.get(self.list_cursor) else {
            return;
        };
        let name = row.name.clone();
        if let Some(pos) = self.marked.iter().position(|n| *n == name) {
            self.marked.remove(pos);
            self.set_status(format!("unmarked {name}"));
        } else if self.marked.len() >= 3 {
            self.set_status("compare holds at most 3 runs — unmark one first");
        } else {
            self.marked.push(name.clone());
            self.set_status(format!(
                "marked {name} for compare ({}/3)",
                self.marked.len()
            ));
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
            let msg = views::reset(&self.repo_root, action);
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
fn schema_args_root(schema: &serde_json::Value) -> Option<&serde_json::Map<String, serde_json::Value>> {
    let defs = schema.get("definitions").and_then(|d| d.as_object());
    if let Some(defs) = defs {
        let name = schema
            .get("$ref")
            .and_then(|r| r.as_str())
            .and_then(|r| r.rsplit('/').next())
            .unwrap_or("Args");
        if let Some(obj) = defs.get(name).or_else(|| defs.get("Args")).and_then(|d| d.as_object()) {
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
    // every theme getter returns the right style (matches lamquant).
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
        View::Cockpit,
        View::Jobs,
        View::Log,
        View::System,
        View::History,
        View::Leaderboard,
        View::Compare,
        View::Checkpoints,
        View::Presets,
        View::Metrics,
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
                kind: recipe.input_kinds.first().copied().unwrap_or("dataset").into(),
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
            app.refresh_jobs();
            app.refresh_system();
            // Refresh log too — running jobs grow fast.
            app.refresh_log();
            app.last_refresh = Instant::now();
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
        KeyCode::Char('J') => return app.set_view(View::Jobs),
        KeyCode::Char('L') => return app.set_view(View::Log),
        KeyCode::Char('Y') => return app.set_view(View::System),
        KeyCode::Char('H') => return app.set_view(View::History),
        KeyCode::Char('B') => return app.set_view(View::Leaderboard),
        KeyCode::Char('K') => return app.set_view(View::Checkpoints),
        KeyCode::Char('P') => return app.set_view(View::Presets),
        KeyCode::Char('M') => return app.set_view(View::Metrics),
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
        KeyCode::Esc | KeyCode::Char('b') => app.set_view(View::Cockpit),
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
            View::Reset => app.fire_reset(),
            _ => {}
        },
        // History / Leaderboard: [m]ark a run for Compare; [space] too.
        KeyCode::Char('m') | KeyCode::Char(' ')
            if matches!(app.view, View::History | View::Leaderboard) =>
        {
            app.toggle_mark();
        }
        // Compare-runs entry from history/leaderboard.
        KeyCode::Char('C') if matches!(app.view, View::History | View::Leaderboard) => {
            app.set_view(View::Compare);
        }
        // Export config (Python cockpit [e]) — only on the Reset view.
        KeyCode::Char('e') if app.view == View::Reset => {
            let msg = views::export_presets(&app.repo_root, &app.catalog);
            app.set_status(msg);
        }
        _ => {}
    }
}

/// Top-level frame dispatcher. Splits body + 1-row footer, routes the
/// body to the active view's drawer, then renders the footer + any
/// modal overlay on top.
fn draw(f: &mut Frame<'_>, app: &mut App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),   // body
            Constraint::Length(1), // footer status bar
        ])
        .split(f.area());
    let body = outer[0];
    match app.view {
        View::Cockpit => draw_cockpit_body(f, body, app),
        View::Jobs => draw_jobs(f, body, app),
        View::Log => draw_log(f, body, app),
        View::System => draw_system(f, body, app),
        View::History => draw_history(f, body, app),
        View::Leaderboard => draw_leaderboard(f, body, app),
        View::Compare => draw_compare(f, body, app),
        View::Checkpoints => draw_checkpoints(f, body, app),
        View::Presets => draw_presets(f, body, app),
        View::Metrics => draw_metrics(f, body, app),
        View::Reset => draw_reset(f, body, app),
    }
    draw_status(f, outer[1], app);
    draw_overlay(f, app);
}

/// Section header for the cockpit menu — a dim-indented heading in the
/// project's `theme::highlight` style (mirrors lamquant's `section_header`).
fn section_header(title: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(title.to_string(), theme::highlight()),
    ])
}

/// One `[key] Label   description` menu row (mirrors lamquant's `opt`):
/// the key hint in `theme::key_hint`, the label in `theme::normal`, the
/// description in `theme::dim`. `key` is `None` for items reachable only
/// via the `R` recipe picker (shown as `[-]`).
fn opt_row(key: Option<char>, label: &str, desc: &str) -> Line<'static> {
    let key_str = match key {
        Some(c) => format!("[{c}]"),
        None => "[-]".to_string(),
    };
    Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("{key_str:<4}"), theme::key_hint()),
        Span::raw(" "),
        Span::styled(format!("{label:<28}"), theme::normal()),
        Span::styled(menu_desc(desc), theme::dim()),
    ])
}

/// Collapse + clip a recipe/action description to a single menu row.
/// Recipe `DESCRIPTION` constants are paragraph-length (wrapped across
/// several source lines); rendered verbatim with `Wrap` they each spill
/// to 2-3 rows and push lower menu sections (SYSTEM) off a short
/// terminal. Squashing internal whitespace to single spaces + capping
/// the length keeps every row to exactly one line so all sections stay
/// reachable. The full text is still shown in the recipe picker/editor.
fn menu_desc(desc: &str) -> String {
    const MAX: usize = 76;
    let collapsed = desc.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= MAX {
        collapsed
    } else {
        let mut out: String = collapsed.chars().take(MAX.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

/// Cockpit view — the single-column overview, rendered with a vertical
/// [`Layout`] of proper widgets so the boxes always align and the menu
/// reflows with terminal width:
///   * header strip (title + version) — styled [`Paragraph`]
///   * "Pipeline status" — titled [`Block`] (running jobs)
///   * "Resources" — titled [`Block`] (GPU/CPU/MEM/Disk)
///   * grouped recipe/action menu — [`Paragraph`] in the flexible region
///
/// No manual `┌`/`│`/`└` box-drawing: every border comes from `Block`,
/// so the top/bottom/sides can never drift out of alignment (the bug in
/// the old single-Paragraph implementation).
fn draw_cockpit_body(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    // Running jobs decide the Pipeline-status box height (1 row per job,
    // min 1 for the "no jobs" line) + 2 for the Block borders.
    let running: Vec<&JobSummary> = app
        .jobs
        .iter()
        .filter(|j| matches!(j.state, JobState::Running))
        .collect();
    let pipeline_rows = running.len().max(1) as u16;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),                 // header strip
            Constraint::Length(pipeline_rows + 2), // Pipeline status (+ borders)
            Constraint::Length(4),                 // Resources (+ borders)
            Constraint::Min(0),                    // recipe / action menu
        ])
        .split(area);

    // ── Header strip (title left, version right) ─────────────────────
    let version = format!("blut v{}", env!("CARGO_PKG_VERSION"));
    let header = Paragraph::new(Line::from(vec![
        Span::styled(" BLUT Training Cockpit", theme::title()),
        Span::raw("  "),
        Span::styled(version, theme::key_label()),
    ]));
    f.render_widget(header, chunks[0]);

    // Bullet glyph for running jobs — degrade to ASCII when the terminal
    // can't render Unicode (NO_COLOR / TERM=dumb / non-UTF-8 locale).
    let bullet = if theme::ascii_only() { "*" } else { "●" };

    // ── Pipeline status (titled Block — borders auto-align) ──────────
    let pipe_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme::dim())
        .title(Span::styled(" Pipeline status ", theme::heading()));
    let pipe_inner = pipe_block.inner(chunks[1]);
    f.render_widget(pipe_block, chunks[1]);
    let pipe_lines: Vec<Line> = if running.is_empty() {
        vec![Line::from(Span::styled(
            "  No training jobs running",
            theme::dim(),
        ))]
    } else {
        running
            .iter()
            .map(|j| {
                let pid = j.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
                let last = match (j.last_step, j.last_loss) {
                    (Some(step), Some(loss)) => format!("step={step} loss={loss:.4}"),
                    _ => "-".into(),
                };
                Line::from(Span::styled(
                    format!("  {bullet} {} pid {pid} · {last}", j.id),
                    theme::success(),
                ))
            })
            .collect()
    };
    f.render_widget(Paragraph::new(pipe_lines), pipe_inner);

    // ── Resources (titled Block) ─────────────────────────────────────
    let res_block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme::dim())
        .title(Span::styled(" Resources ", theme::heading()));
    let res_inner = res_block.inner(chunks[2]);
    f.render_widget(res_block, chunks[2]);
    let s = &app.system;
    let res_lines = vec![
        Line::from(vec![
            Span::styled("  GPU  ", theme::key_hint()),
            Span::styled(s.gpu_summary(), theme::normal()),
        ]),
        Line::from(vec![
            Span::styled("  CPU  ", theme::key_hint()),
            Span::styled(
                format!(
                    "load1={:.2}  ·  MEM {:.1}/{:.1} GiB used  ·  Disk {} free ({}% used)",
                    s.load1, s.mem_used_gb, s.mem_total_gb, s.disk_free_human, s.disk_used_pct
                ),
                theme::dim(),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(res_lines), res_inner);

    // ── Grouped recipe / action menu ─────────────────────────────────
    // Mirrors the lamquant hub cockpit's section convention. Sections:
    //   DATA PREPARATION / PIPELINE OPERATIONS (= TRAINING + PIPELINE
    //   recipes) / EVALUATION / EXPORT / DIAGNOSTICS (migrated Views) /
    //   SYSTEM (built-ins). Every BLUT recipe is reachable by its
    //   auto-assigned hotkey (or the [R] picker if it ran out of keys);
    //   every migrated screen is reachable by its capital-letter View key.
    let menu = App::recipe_menu(&app.catalog);
    // Bucket recipes by category, preserving recipe_menu() order within
    // each bucket and first-seen category order across buckets.
    use std::collections::BTreeMap;
    let mut by_cat: BTreeMap<
        &'static str,
        Vec<(Option<char>, &'static crate::recipes::RecipeDef)>,
    > = BTreeMap::new();
    for (k, r) in &menu {
        by_cat.entry(r.category.label()).or_default().push((*k, *r));
    }
    let mut printed_cats: Vec<&'static str> = Vec::new();
    for (_, r) in &menu {
        let cat = r.category.label();
        if !printed_cats.contains(&cat) {
            printed_cats.push(cat);
        }
    }

    let mut lines: Vec<Line> = Vec::new();
    for cat in &printed_cats {
        lines.push(section_header(cat));
        lines.push(Line::from(""));
        for (k, r) in by_cat.get(cat).unwrap() {
            lines.push(opt_row(*k, r.name, r.description));
        }
        lines.push(Line::from(""));
    }

    // DIAGNOSTICS — the migrated detail/diagnostic Views (parity with the
    // hub cockpit's DIAGNOSTICS section + the Python cockpit screens).
    lines.push(section_header("DIAGNOSTICS"));
    lines.push(Line::from(""));
    for (key, label, desc) in [
        ('J', "Jobs", "all jobs, color-coded by state"),
        ('L', "Log", "status.jsonl tail of selected job"),
        ('Y', "System", "full GPU / MEM / DISK / CPU probe"),
        ('H', "Run history", "training_logs/*.csv runs + best R"),
        ('B', "Leaderboard", "runs ranked by best validation R"),
        ('K', "Checkpoints", ".ckpt browser grouped by dir"),
        ('M', "Live metrics", "tail of the newest training CSV"),
    ] {
        lines.push(opt_row(Some(key), label, desc));
    }
    lines.push(Line::from(""));

    // PLANNING — preset / hyperparameter reference (hub PLANNING parity).
    lines.push(section_header("PLANNING"));
    lines.push(Line::from(""));
    lines.push(opt_row(
        Some('P'),
        "Presets & hyperparameters",
        "preset catalog, decoder tiers, hparam groups",
    ));
    lines.push(Line::from(""));

    // SYSTEM — built-in actions + destructive maintenance (hub SYSTEM
    // parity). The reset/export screen plus the cockpit built-ins.
    lines.push(section_header("SYSTEM"));
    lines.push(Line::from(""));
    lines.push(opt_row(
        Some('X'),
        "Reset / export",
        "kill tmux · clear numba · clear logs · export",
    ));
    for (k, _, label) in App::builtin_keys() {
        lines.push(opt_row(Some(*k), label, ""));
    }

    let menu_para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(menu_para, chunks[3]);
}

fn centered_rect(area: Rect, pct_w: u16, pct_h: u16) -> Rect {
    let w = area.width.saturating_mul(pct_w) / 100;
    let h = area.height.saturating_mul(pct_h) / 100;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

fn draw_overlay(f: &mut Frame<'_>, app: &App) {
    match &app.overlay {
        Overlay::None => {}
        Overlay::Picker { query, cursor } => {
            let area = centered_rect(f.area(), 60, 60);
            // Clear by drawing an empty block underneath.
            let bg = Block::default()
                .borders(Borders::ALL)
                .border_style(theme::dim())
                .title(Span::styled(
                    " pick recipe (type to filter, ↑↓ select, Enter open, Esc cancel) ",
                    theme::title(),
                ));
            f.render_widget(bg.clone(), area);

            let inner = Rect {
                x: area.x + 1,
                y: area.y + 1,
                width: area.width.saturating_sub(2),
                height: area.height.saturating_sub(2),
            };
            let query_line = Line::from(vec![
                Span::styled("> ", theme::warning()),
                Span::styled(query.clone(), theme::normal()),
                Span::styled("_", theme::dim()),
            ]);
            let query_widget = Paragraph::new(query_line);
            let query_area = Rect {
                x: inner.x,
                y: inner.y,
                width: inner.width,
                height: 1,
            };
            f.render_widget(query_widget, query_area);

            let list_area = Rect {
                x: inner.x,
                y: inner.y + 2,
                width: inner.width,
                height: inner.height.saturating_sub(2),
            };
            let filtered = App::filter_recipes(&app.catalog, query);
            let items: Vec<ListItem> = filtered
                .iter()
                .enumerate()
                .map(|(i, &idx)| {
                    let r = app.catalog[idx];
                    let name_style = if i == *cursor {
                        theme::selected()
                    } else {
                        theme::heading()
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("{:<32} ", r.name), name_style),
                        Span::styled(format!("[{}] ", r.backend_id), theme::key_hint()),
                        Span::styled(
                            r.description.chars().take(80).collect::<String>(),
                            theme::dim(),
                        ),
                    ]))
                })
                .collect();
            let list = List::new(items);
            f.render_widget(list, list_area);
        }
        Overlay::DatasetPicker {
            recipe,
            datasets,
            cursor,
        } => {
            let area = centered_rect(f.area(), 64, 60);
            let bg = Block::default()
                .borders(Borders::ALL)
                .border_style(theme::dim())
                .title(Span::styled(
                    format!(
                        " pick dataset for {} [{}] — ↑↓ select, Enter pick, Esc skip ",
                        recipe.name,
                        recipe.input_kinds.join(",")
                    ),
                    theme::title(),
                ));
            let inner = bg.inner(area);
            f.render_widget(bg, area);
            let items: Vec<ListItem> = datasets
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let name_style = if i == *cursor {
                        theme::selected()
                    } else {
                        theme::heading()
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("{:<28} ", truncate(&d.name, 28)), name_style),
                        Span::styled(format!("[{}] ", d.kind), theme::key_hint()),
                        Span::styled(format!("{} ex  ", d.n_examples), theme::normal()),
                        Span::styled(truncate(&d.source_path, 40), theme::dim()),
                    ]))
                })
                .collect();
            f.render_widget(List::new(items), inner);
        }
        Overlay::Editor {
            recipe,
            fields,
            focus,
            raw_mode,
            raw_buffer,
        } => {
            let area = centered_rect(f.area(), 72, 72);
            if *raw_mode {
                let block = Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme::dim())
                    .title(Span::styled(
                        format!(
                            " edit args (RAW JSON): {recipe} — type to edit, Ctrl+Enter submit, Ctrl+R form, Esc cancel "
                        ),
                        theme::title(),
                    ));
                let text: Vec<Line> =
                    raw_buffer.lines().map(|l| Line::from(l.to_string())).collect();
                let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
                f.render_widget(para, area);
            } else {
                let block = Block::default()
                    .borders(Borders::ALL)
                    .border_style(theme::dim())
                    .title(Span::styled(
                        format!(
                            " edit args: {recipe} — ↑↓/Tab move, type to edit, Enter submit, Ctrl+R raw, Esc cancel "
                        ),
                        theme::title(),
                    ));
                let inner = block.inner(area);
                f.render_widget(block, area);
                let mut lines: Vec<Line> = Vec::new();
                if fields.is_empty() {
                    lines.push(Line::from(Span::styled(
                        "(this recipe declares no args — press Enter to launch, or Ctrl+R to add raw JSON)",
                        theme::dim(),
                    )));
                } else {
                    for (i, fld) in fields.iter().enumerate() {
                        let focused = i == *focus;
                        let marker = if focused { "▶ " } else { "  " };
                        let name_style = if focused {
                            theme::selected()
                        } else {
                            theme::heading()
                        };
                        // The focused row shows a cursor caret after the value.
                        let val_display = if focused {
                            format!("{}_", fld.value)
                        } else {
                            fld.value.clone()
                        };
                        lines.push(Line::from(vec![
                            Span::styled(marker.to_string(), theme::success()),
                            Span::styled(format!("{:<24}", fld.name), name_style),
                            Span::styled(format!("({:<7}) ", fld.ty), theme::key_hint()),
                            Span::styled(val_display, theme::normal()),
                        ]));
                    }
                }
                let para = Paragraph::new(lines).wrap(Wrap { trim: false });
                f.render_widget(para, inner);
            }
        }
    }
}

fn draw_jobs(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let items: Vec<ListItem> = if app.jobs.is_empty() {
        vec![ListItem::new(Span::styled(
            "no jobs (yet) — start one with `blut recipe run …`",
            theme::dim(),
        ))]
    } else {
        app.jobs
            .iter()
            .map(|j| {
                let state_style = match j.state {
                    JobState::Running => theme::success().add_modifier(Modifier::BOLD),
                    JobState::Done => theme::highlight(),
                    JobState::Failed => theme::error().add_modifier(Modifier::BOLD),
                    JobState::Cancelled => theme::warning().add_modifier(Modifier::BOLD),
                };
                let pid = j.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
                let output = j.output_name.clone().unwrap_or_else(|| "-".into());
                let last = match (j.last_step, j.last_loss, j.final_loss) {
                    (_, _, Some(fl)) => format!("final_loss={fl:.4}"),
                    (Some(step), Some(loss), _) => format!("step={step} loss={loss:.4}"),
                    _ => "-".into(),
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<22} ", j.id), theme::normal()),
                    Span::styled(format!("{:<9} ", j.state.as_str()), state_style),
                    Span::styled(format!("pid={:<6} ", pid), theme::normal()),
                    Span::styled(format!("out={:<18} ", output), theme::normal()),
                    Span::styled(last, theme::dim()),
                ]))
            })
            .collect()
    };
    let block = Block::default()
        .title(Span::styled(
            format!(
                " jobs ({}) — ↑↓ select, c cancel, r refresh ",
                app.jobs.len()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let list = List::new(items)
        .block(block)
        .highlight_style(theme::selected())
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, area, &mut app.selected);
}

fn draw_log(f: &mut Frame<'_>, area: Rect, app: &App) {
    let title = match &app.log_job_id {
        Some(id) => format!(" log: {id}  (Enter to refresh) "),
        None => " log: (no selection) ".into(),
    };
    let block = Block::default()
        .title(Span::styled(title, theme::title()))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let text: Vec<Line> = if app.log_lines.is_empty() {
        vec![Line::from(Span::styled(
            "(no status.jsonl yet — job may still be starting)",
            theme::dim(),
        ))]
    } else {
        // Show the tail that fits in the visible height. Reserve 2
        // rows for the borders.
        let visible = area.height.saturating_sub(2) as usize;
        let start = app.log_lines.len().saturating_sub(visible);
        app.log_lines[start..]
            .iter()
            .map(|s| Line::from(Span::styled(s.clone(), theme::normal())))
            .collect()
    };
    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_system(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(" system ", theme::title()))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let snap = &app.system;
    let lines: Vec<Line> = vec![
        Line::from(Span::styled("GPU", theme::highlight())),
        Line::from(Span::styled(snap.gpu_summary(), theme::normal())),
        Line::from(""),
        Line::from(Span::styled("MEM", theme::highlight())),
        Line::from(Span::styled(
            format!(
                "used {:.1}/{:.1} GB  free {:.1} GB",
                snap.mem_used_gb, snap.mem_total_gb, snap.mem_avail_gb
            ),
            theme::normal(),
        )),
        Line::from(""),
        Line::from(Span::styled("DISK /mnt/4tb", theme::highlight())),
        Line::from(Span::styled(
            format!(
                "free {} ({}% used)",
                snap.disk_free_human, snap.disk_used_pct
            ),
            theme::normal(),
        )),
        Line::from(""),
        Line::from(Span::styled("CPU", theme::highlight())),
        Line::from(Span::styled(
            format!("load1={:.2}", snap.load1),
            theme::normal(),
        )),
    ];
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    f.render_widget(para, area);
}

/// Shared header line for the migrated detail views.
fn view_header(title: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(title.to_string(), theme::title()),
        Span::styled(
            format!("    blut v{}", env!("CARGO_PKG_VERSION")),
            theme::dim(),
        ),
    ])
}

/// Run History view (Python `_screen_history`): training logs + best-R
/// + epoch + date, plus a checkpoint summary footer.
fn draw_history(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · m mark · C compare · b back) ",
                View::History.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Run History"), Line::from("")];
    if app.runs.is_empty() {
        lines.push(Line::from(Span::styled(
            "No training runs found under training_logs/*.csv.",
            theme::dim(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<42} {:<10} {:<10} {}",
                "Name", "Best R", "Epoch", "Date"
            ),
            theme::dim(),
        )));
        for (i, r) in app.runs.iter().enumerate() {
            let marked = app.marked.contains(&r.name);
            let cursor = i == app.list_cursor;
            let prefix = if cursor { "▶ " } else { "  " };
            let mark = if marked { "✓" } else { " " };
            let r_str = if r.best_r > 0.0 {
                format!("{:.4}", r.best_r)
            } else {
                "—".into()
            };
            let ep_str = if r.total_ep > 0 {
                format!("{}/{}", r.best_ep, r.total_ep)
            } else {
                "—".into()
            };
            let style = if cursor {
                theme::selected()
            } else {
                theme::normal()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{prefix}{mark} "), theme::success()),
                Span::styled(
                    format!(
                        "{:<42} {:<10} {:<10} ",
                        truncate(&r.name, 42),
                        r_str,
                        ep_str
                    ),
                    style,
                ),
                Span::styled(r.date.clone(), theme::dim()),
            ]));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Leaderboard view (Python `_screen_leaderboard`): runs ranked by best
/// R descending, gold marker on #1.
fn draw_leaderboard(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · m mark · C compare · b back) ",
                View::Leaderboard.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Model Leaderboard"), Line::from("")];
    if app.runs.is_empty() {
        lines.push(Line::from(Span::styled(
            "No training logs found. Run some experiments first.",
            theme::dim(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<5} {:<40} {:<10} {:<10} {}",
                "Rank", "Name", "Best R", "Epoch", "Date"
            ),
            theme::dim(),
        )));
        for (i, r) in app.runs.iter().enumerate().take(20) {
            let cursor = i == app.list_cursor;
            let marked = app.marked.contains(&r.name);
            let medal = if i == 0 { " ▸" } else { "" };
            let r_str = if r.best_r > 0.0 {
                format!("{:.4}", r.best_r)
            } else {
                "—".into()
            };
            let ep_str = if r.total_ep > 0 {
                format!("{}/{}", r.best_ep, r.total_ep)
            } else {
                "—".into()
            };
            let style = if cursor {
                theme::selected()
            } else if i == 0 {
                theme::success().add_modifier(Modifier::BOLD)
            } else {
                theme::normal()
            };
            let mark = if marked { "✓" } else { " " };
            lines.push(Line::from(vec![
                Span::styled(
                    format!(
                        "{mark} {:<5} {:<40} {:<10} {:<10} ",
                        i + 1,
                        truncate(&r.name, 40),
                        r_str,
                        ep_str
                    ),
                    style,
                ),
                Span::styled(format!("{}{medal}", r.date), theme::dim()),
            ]));
        }
        if app.runs.len() > 20 {
            lines.push(Line::from(Span::styled(
                format!("  ... {} more runs", app.runs.len() - 20),
                theme::dim(),
            )));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Compare view (Python `_screen_compare`): side-by-side metric table
/// of the marked runs; the per-row winner is highlighted green.
fn draw_compare(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (mark runs in History/Leaderboard with m · b back) ",
                View::Compare.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Compare Runs"), Line::from("")];
    let selected: Vec<&views::RunRow> = app
        .marked
        .iter()
        .filter_map(|n| app.runs.iter().find(|r| r.name == *n))
        .collect();
    if selected.len() < 2 {
        lines.push(Line::from(Span::styled(
            "Mark at least 2 runs (press m on a row in History/Leaderboard) to compare.",
            theme::dim(),
        )));
    } else {
        // Header row of run names.
        let mut hdr = vec![Span::styled(format!("  {:<16}", "Metric"), theme::dim())];
        for r in &selected {
            hdr.push(Span::styled(
                format!("{:<22}", truncate(&r.name, 21)),
                theme::heading(),
            ));
        }
        lines.push(Line::from(hdr));
        lines.push(Line::from(""));
        // epochs (higher not necessarily better — no highlight), best_r,
        // final_r (highlight max).
        let epochs: Vec<f64> = selected.iter().map(|r| r.total_ep as f64).collect();
        let best_r: Vec<f64> = selected.iter().map(|r| r.best_r).collect();
        let final_r: Vec<f64> = selected.iter().map(|r| r.final_r).collect();
        lines.push(metric_row("epochs", &epochs, false, 0));
        lines.push(metric_row("best_r", &best_r, true, 4));
        lines.push(metric_row("final_r", &final_r, true, 4));
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// One Compare-table row. `highlight_max` greens the winning column;
/// `decimals` controls float formatting.
fn metric_row(metric: &str, vals: &[f64], highlight_max: bool, decimals: usize) -> Line<'static> {
    let best = vals.iter().cloned().fold(f64::MIN, f64::max);
    let mut spans = vec![Span::styled(format!("  {:<16}", metric), theme::heading())];
    for v in vals {
        let txt = if decimals == 0 {
            format!("{:<22}", *v as i64)
        } else {
            format!("{:<22.*}", decimals, v)
        };
        let style = if highlight_max && *v == best && best > 0.0 {
            theme::success().add_modifier(Modifier::BOLD)
        } else {
            theme::normal()
        };
        spans.push(Span::styled(txt, style));
    }
    Line::from(spans)
}

/// Checkpoints view (Python `_screen_checkpoints`): all `.ckpt` grouped
/// by directory, with per-dir count + GiB and the newest files.
fn draw_checkpoints(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · r refresh · b back) ",
                View::Checkpoints.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Checkpoints"), Line::from("")];
    if app.ckpts.is_empty() {
        lines.push(Line::from(Span::styled(
            "No checkpoints found under checkpoints/ or weights/.",
            theme::dim(),
        )));
    } else {
        let total_gb: f64 = app.ckpts.iter().map(|c| c.size_mb).sum::<f64>() / 1024.0;
        lines.push(Line::from(Span::styled(
            format!(
                "{} checkpoints  ·  {:.1} GiB total",
                app.ckpts.len(),
                total_gb
            ),
            theme::dim(),
        )));
        lines.push(Line::from(""));
        for (i, c) in app.ckpts.iter().enumerate() {
            let cursor = i == app.list_cursor;
            let prefix = if cursor { "▶ " } else { "  " };
            let style = if cursor {
                theme::selected()
            } else {
                theme::normal()
            };
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), theme::success()),
                Span::styled(format!("{:<40} ", truncate(&c.name, 40)), style),
                Span::styled(format!("{:>8.1} MB  ", c.size_mb), theme::key_hint()),
                Span::styled(format!("{}  ", c.date), theme::dim()),
                Span::styled(c.rel_dir.clone(), theme::dim()),
            ]));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Presets & Hyperparameters view (Python `_screen_presets` +
/// `_screen_hparams` + decoder tiers + validated features). Read-only
/// catalog — the live values live in recipe Args JSON (ADR 0017).
fn draw_presets(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(" {} (b back) ", View::Presets.title()),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Presets & Hyperparameters"), Line::from("")];
    lines.push(Line::from(Span::styled("PRESETS", theme::highlight())));
    for (name, ep, wpe, est, use_case) in views::PRESETS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<12}", name), theme::heading()),
            Span::styled(format!("{ep:<12} {wpe:<10} "), theme::normal()),
            Span::styled(format!("{est:<8}  "), theme::warning()),
            Span::styled(use_case.to_string(), theme::dim()),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "DECODER TIERS",
        theme::highlight(),
    )));
    for (tier, params, note) in views::DECODER_TIERS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {tier:<10}"), theme::heading()),
            Span::styled(format!("{params:<8} "), theme::normal()),
            Span::styled(note.to_string(), theme::dim()),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "PRODUCTION-VALIDATED FEATURES",
        theme::highlight(),
    )));
    for feat in views::VALIDATED_FEATURES {
        lines.push(Line::from(Span::styled(
            format!("  • {feat}"),
            theme::normal(),
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "HYPERPARAMETERS (set via recipe Args JSON — ADR 0017)",
        theme::highlight(),
    )));
    for (group, fields) in views::HPARAM_GROUPS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<16}", group), theme::heading()),
            Span::styled(fields.join(", "), theme::dim()),
        ]));
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Live Metrics view (Python `_screen_live_metrics` terminal tail):
/// the tail of the newest training-log CSV, re-read each tick.
fn draw_metrics(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(" {} (r refresh · b back) ", View::Metrics.title()),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let n = area.height.saturating_sub(4) as usize;
    let tail = views::metrics_tail(&app.repo_root, n.max(10));
    let lines: Vec<Line> = std::iter::once(view_header("Live Metrics"))
        .chain(std::iter::once(Line::from("")))
        .chain(tail.into_iter().map(|s| {
            if s.starts_with('#') {
                Line::from(Span::styled(s, theme::key_hint()))
            } else {
                Line::from(Span::styled(s, theme::normal()))
            }
        }))
        .collect();
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Reset view (Python `_screen_reset` + `_screen_export`): the three
/// destructive maintenance actions (two-press Enter confirm) + export.
fn draw_reset(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · Enter confirm · e export · b back) ",
                View::Reset.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Reset Training State"), Line::from("")];
    lines.push(Line::from(Span::styled(
        "Destructive — each action requires a second Enter to confirm.",
        theme::warning(),
    )));
    lines.push(Line::from(""));
    for (i, action) in RESET_ROWS.iter().enumerate() {
        let cursor = i == app.reset_cursor;
        let armed = app
            .reset_armed
            .map(|(ai, t)| ai == i && t.elapsed() < RESET_WINDOW)
            .unwrap_or(false);
        let prefix = if cursor { "▶ " } else { "  " };
        let style = if armed {
            theme::error().add_modifier(Modifier::BOLD)
        } else if cursor {
            theme::selected()
        } else {
            theme::normal()
        };
        let suffix = if armed {
            "   ← press Enter again to confirm"
        } else {
            ""
        };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), theme::success()),
            Span::styled(format!("{}{suffix}", action.label()), style),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  [e] ", theme::key_hint()),
        Span::styled("Export configuration ", theme::normal()),
        Span::styled(
            "(write recipe Args JSON schemas to repo root)",
            theme::dim(),
        ),
    ]));
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Truncate a string to `max` chars with an ellipsis if needed.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

fn draw_status(f: &mut Frame<'_>, area: Rect, app: &App) {
    let base = match app.view {
        View::Cockpit => {
            "q quit • ↑↓ select • Enter log • r refresh • c cancel • R recipe • J/L/Y/H/B/K/P/M/X views"
        }
        View::Jobs => "↑↓ select • Enter log • c cancel • r refresh • b back • q quit",
        View::Log => "↑↓ select job • c cancel • r refresh • b back • q quit",
        View::System => "r refresh • b back • q quit",
        View::History | View::Leaderboard => {
            "↑↓ move • m mark • C compare • r refresh • b back • q quit"
        }
        View::Compare => "mark runs with m in History/Leaderboard • b back • q quit",
        View::Checkpoints => "↑↓ move • r refresh • b back • q quit",
        View::Presets => "b back • q quit",
        View::Metrics => "r refresh • b back • q quit",
        View::Reset => "↑↓ move • Enter confirm (2x) • e export • b back • q quit",
    };
    let para = if let Some((msg, _)) = &app.status_msg {
        // Status message segment in the success-tinted bar style, the
        // key-hint base in the standard status-bar style (matches the
        // lamquant status-bar convention).
        Paragraph::new(Line::from(vec![
            Span::styled(format!(" {msg} "), theme::status_msg()),
            Span::styled(format!(" {base} "), theme::status_bar()),
        ]))
    } else {
        Paragraph::new(Span::styled(format!(" {base} "), theme::status_bar()))
    };
    f.render_widget(para, area);
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;

    /// Build a deterministic App for rendering tests: no live jobs, a
    /// fixed (empty) system snapshot, and a temp repo root so the
    /// diagnostic views don't scan the real filesystem. Pure in-memory —
    /// no `probe()` / `pgrep` / `nvidia-smi` calls.
    fn test_app() -> App {
        // Force unicode + color on so the alignment test sees `┌`/`│`/`└`
        // and the section-heading assertions are charset-stable.
        theme::detect("always", "unicode");
        super::isolate_datasets_db_for_tests();
        let mut app = App::new(test_registry());
        // Point the repo root at an empty temp dir so views::* don't pick
        // up stray training_logs / checkpoints from the dev tree.
        let tmp = std::env::temp_dir().join(format!("blut-tui-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        app.repo_root = tmp;
        app
    }

    /// Render the app's current view to a fresh `TestBackend` of the
    /// given size and return the resulting buffer. This is the headless
    /// render path the tests assert against (no real terminal, no raw
    /// mode) — equivalent to a `tui --check` smoke.
    fn render_to_test_backend(app: &mut App, w: u16, h: u16) -> Buffer {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).expect("test terminal");
        term.draw(|f| draw(f, app)).expect("draw to test backend");
        term.backend().buffer().clone()
    }

    /// Flatten the buffer into one big string (cells joined row by row,
    /// rows separated by `\n`). Used for `contains` content assertions.
    fn buffer_text(buf: &Buffer) -> String {
        let area = buf.area();
        let mut out = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    /// Symbol at a cell as an owned `String`.
    fn sym(buf: &Buffer, x: u16, y: u16) -> String {
        buf[(x, y)].symbol().to_string()
    }

    #[test]
    fn cockpit_renders_aligned_box() {
        let mut app = test_app();
        app.view = View::Cockpit;
        let buf = render_to_test_backend(&mut app, 120, 40);
        let area = *buf.area();

        // 1. Locate the "Pipeline status" Block. Its title sits on the top
        //    border row, so find the row containing the title text and the
        //    `┌` top-left corner on that same row.
        let mut top_row: Option<u16> = None;
        let mut left_col: Option<u16> = None;
        let mut right_col: Option<u16> = None;
        'outer: for y in 0..area.height {
            let mut row = String::new();
            for x in 0..area.width {
                row.push_str(buf[(x, y)].symbol());
            }
            if row.contains("Pipeline status") && row.contains('┌') {
                top_row = Some(y);
                for x in 0..area.width {
                    let s = sym(&buf, x, y);
                    if s == "┌" {
                        left_col = Some(x);
                    }
                    if s == "┐" {
                        right_col = Some(x);
                    }
                }
                break 'outer;
            }
        }
        let top_row = top_row.expect("Pipeline status top border row not found");
        let left_col = left_col.expect("`┌` top-left corner not found");
        let right_col = right_col.expect("`┐` top-right corner not found");
        assert!(
            right_col > left_col,
            "right border must be to the right of the left border"
        );

        // 2. Find the matching bottom border: the next row below top_row
        //    whose left_col cell is `└`.
        let mut bot_row: Option<u16> = None;
        for y in (top_row + 1)..area.height {
            if sym(&buf, left_col, y) == "└" {
                bot_row = Some(y);
                break;
            }
        }
        let bot_row = bot_row.expect("`└` bottom-left corner not found below top border");

        // 3. Bottom corners must sit in the SAME columns as the top
        //    corners — this is the alignment property the old manual
        //    `└{}┘` (different repeat count) violated.
        assert_eq!(
            sym(&buf, left_col, bot_row),
            "└",
            "bottom-left corner must align with top-left corner column"
        );
        assert_eq!(
            sym(&buf, right_col, bot_row),
            "┘",
            "bottom-right corner `┘` must align with top-right corner `┐` column"
        );

        // 4. Every interior row of the box must have a `│` at EXACTLY the
        //    left_col and right_col — the side borders never drift.
        for y in (top_row + 1)..bot_row {
            assert_eq!(
                sym(&buf, left_col, y),
                "│",
                "left side border drifted at row {y} (expected `│` at col {left_col})"
            );
            assert_eq!(
                sym(&buf, right_col, y),
                "│",
                "right side border drifted at row {y} (expected `│` at col {right_col})"
            );
        }
    }

    #[test]
    fn menu_desc_collapses_and_clips_to_one_line() {
        // Multi-line / over-long descriptions squash to a single
        // ≤76-char row so lower menu sections stay on-screen.
        let long = "Full LamQuant SNN pipeline end-to-end: EDF→.lma encode (lml) → \
             patient-level seizure-stratified split manifest → train → gate.";
        let d = menu_desc(long);
        assert!(!d.contains('\n'));
        assert!(d.chars().count() <= 76, "got {} chars", d.chars().count());
        assert!(
            d.ends_with('…'),
            "long desc must be truncated with ellipsis"
        );
        // Short single-line descriptions pass through unchanged.
        assert_eq!(menu_desc("short desc"), "short desc");
        // Internal newlines/runs collapse to single spaces.
        assert_eq!(menu_desc("a\n  b\t c"), "a b c");
    }

    #[test]
    fn cockpit_shows_all_sections() {
        let mut app = test_app();
        app.view = View::Cockpit;
        let buf = render_to_test_backend(&mut app, 120, 60);
        let text = buffer_text(&buf);
        // The always-present (non-recipe-driven) section headings must
        // render regardless of which cookbooks are registered. The
        // recipe-category headings (DATA PREPARATION / PIPELINE / …) are
        // data-driven off the injected catalog — blut-core's default
        // registry is lamu-only (no DataPrep/Pipeline recipes; those live
        // in cookbook-lamquant since C2a), so they are asserted in the
        // cookbook crate's TUI tests, not here.
        for heading in ["DIAGNOSTICS", "PLANNING", "SYSTEM"] {
            assert!(
                text.contains(heading),
                "cockpit menu missing section heading `{heading}`"
            );
        }
        // The lamu catalog's own categories must still render their
        // headings (proves the recipe-category section path works).
        for heading in ["TRAINING", "EVALUATION"] {
            assert!(
                text.contains(heading),
                "cockpit menu missing lamu recipe-category heading `{heading}`"
            );
        }
        // The Pipeline status + Resources boxes are present by title.
        assert!(
            text.contains("Pipeline status"),
            "missing Pipeline status box"
        );
        assert!(text.contains("Resources"), "missing Resources box");
    }

    #[test]
    fn each_view_renders_nonempty() {
        let views = [
            View::Cockpit,
            View::Jobs,
            View::Log,
            View::System,
            View::History,
            View::Leaderboard,
            View::Compare,
            View::Checkpoints,
            View::Presets,
            View::Metrics,
            View::Reset,
        ];
        for view in views {
            let mut app = test_app();
            app.view = view;
            let buf = render_to_test_backend(&mut app, 120, 40);
            let text = buffer_text(&buf);
            // Non-blank: at least one non-space glyph somewhere.
            assert!(
                text.chars().any(|c| !c.is_whitespace()),
                "view {view:?} rendered a completely blank buffer"
            );
            // Each view surfaces a recognizable title/heading. The cockpit
            // uses its header strip; the detail views use the block title
            // and/or the view_header line.
            let needle = match view {
                View::Cockpit => "BLUT Training Cockpit",
                View::Jobs => "jobs",
                View::Log => "log",
                View::System => "system",
                View::History => "Run History",
                View::Leaderboard => "Leaderboard",
                View::Compare => "Compare Runs",
                View::Checkpoints => "Checkpoints",
                View::Presets => "Presets",
                View::Metrics => "Live Metrics",
                View::Reset => "Reset",
            };
            assert!(
                text.contains(needle),
                "view {view:?} buffer missing expected title text `{needle}`\n--- buffer ---\n{text}"
            );
        }
    }

    #[test]
    fn cockpit_lists_every_recipe_hotkey() {
        // Parity guard: every recipe in the catalog must appear by name in
        // the cockpit menu so no launchable recipe is silently dropped.
        let mut app = test_app();
        app.view = View::Cockpit;
        let buf = render_to_test_backend(&mut app, 160, 80);
        let text = buffer_text(&buf);
        for r in FIXTURE {
            assert!(
                text.contains(r.name),
                "cockpit menu missing recipe `{}` — parity regression",
                r.name
            );
        }
    }

    #[test]
    fn check_renders_every_view_and_overlay_and_returns_ok() {
        // `blut tui --check` (the shipped self-check) must build the App from
        // the live registry and render every view AND every modal overlay (the
        // F2 form + raw fallback, the F3 dataset picker, the recipe picker)
        // headless without panicking, returning Ok. The fixture registry has a
        // recipe, so the overlay branches that need one are exercised here.
        theme::detect("always", "unicode");
        super::isolate_datasets_db_for_tests();
        super::check(test_registry()).expect("tui --check must render all views + overlays and exit Ok");
    }
}

/// Pure state-transition tests for the cockpit (§5.9). These drive the
/// *key handler* + the pure helper functions directly — no terminal, no
/// raw mode, no live process / filesystem probes. They assert the
/// in-memory `App` state after each synthetic key, the fuzzy-filter
/// ordering, the schema→JSON templates, and the recipe-menu hotkey
/// assignment invariants.
#[cfg(test)]
mod state_tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    /// A fresh `App` with no overlay, cockpit view, pointed at a temp
    /// repo root so nothing in these tests touches the dev tree.
    fn app() -> App {
        super::isolate_datasets_db_for_tests();
        let mut a = App::new(test_registry());
        let tmp = std::env::temp_dir().join(format!(
            "blut-tui-state-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        a.repo_root = tmp;
        a
    }

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }
    fn code(kc: KeyCode) -> KeyEvent {
        KeyEvent::new(kc, KeyModifiers::NONE)
    }
    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    // ── Overlay transitions: None → Picker → Editor → Esc → Ctrl-C ──

    #[test]
    fn overlay_none_to_picker_open() {
        let mut a = app();
        assert!(matches!(a.overlay, Overlay::None));
        // 'R' opens the recipe picker from the cockpit.
        handle_key(&mut a, key('R'));
        assert!(
            matches!(a.overlay, Overlay::Picker { .. }),
            "R should open the recipe Picker overlay"
        );
    }

    #[test]
    fn overlay_picker_enter_selects_into_editor() {
        let mut a = app();
        handle_key(&mut a, key('R'));
        // Empty query → all recipes; cursor 0 selects the first filtered
        // recipe. Enter opens the args Editor for it.
        let first = FIXTURE[App::filter_recipes(FIXTURE, "")[0]];
        handle_key(&mut a, code(KeyCode::Enter));
        match &a.overlay {
            Overlay::Editor { recipe, .. } => {
                assert_eq!(*recipe, first.name, "Editor should open the cursor recipe");
            }
            _ => panic!("Enter in Picker should transition to Editor, got non-Editor overlay"),
        }
    }

    #[test]
    fn overlay_editor_esc_returns_to_none() {
        let mut a = app();
        handle_key(&mut a, key('R')); // → Picker
        handle_key(&mut a, code(KeyCode::Enter)); // → Editor
        assert!(matches!(a.overlay, Overlay::Editor { .. }));
        handle_key(&mut a, code(KeyCode::Esc)); // Esc closes the Editor
        assert!(
            matches!(a.overlay, Overlay::None),
            "Esc in Editor should return to the None overlay"
        );
    }

    #[test]
    fn overlay_picker_esc_returns_to_none() {
        let mut a = app();
        handle_key(&mut a, key('R'));
        assert!(matches!(a.overlay, Overlay::Picker { .. }));
        handle_key(&mut a, code(KeyCode::Esc));
        assert!(
            matches!(a.overlay, Overlay::None),
            "Esc in Picker should return to the None overlay"
        );
    }

    #[test]
    fn ctrl_c_quits_from_every_overlay() {
        // Ctrl-C is the global quit, regardless of overlay state.
        for setup in 0..3 {
            let mut a = app();
            match setup {
                0 => {}                            // None
                1 => handle_key(&mut a, key('R')), // Picker
                _ => {
                    handle_key(&mut a, key('R'));
                    handle_key(&mut a, code(KeyCode::Enter)); // Editor
                }
            }
            assert!(!a.quit);
            handle_key(&mut a, ctrl('c'));
            assert!(a.quit, "Ctrl-C must set quit from overlay setup {setup}");
        }
    }

    #[test]
    fn full_overlay_round_trip() {
        // None → Picker → Editor → Esc → None → (Ctrl-C) quit.
        let mut a = app();
        handle_key(&mut a, key('R'));
        assert!(matches!(a.overlay, Overlay::Picker { .. }));
        handle_key(&mut a, code(KeyCode::Enter));
        assert!(matches!(a.overlay, Overlay::Editor { .. }));
        handle_key(&mut a, code(KeyCode::Esc));
        assert!(matches!(a.overlay, Overlay::None));
        assert!(!a.quit);
        handle_key(&mut a, ctrl('c'));
        assert!(a.quit);
    }

    #[test]
    fn editor_typing_appends_and_backspaces_focused_field() {
        // F2: typing in the per-field form edits the FOCUSED row's value (not a
        // single freeform buffer). `train_delta` is graph-input + typed, so
        // opening it goes straight to the form with editable rows.
        let mut a = app();
        a.open_editor(&test_fixtures::TRAIN_DELTA);
        let Overlay::Editor { fields, focus, .. } = &a.overlay else {
            panic!("expected Editor");
        };
        assert!(!fields.is_empty(), "typed recipe must yield form rows");
        let f0 = *focus;
        let before = fields[f0].value.clone();
        handle_key(&mut a, key('x'));
        let Overlay::Editor { fields, .. } = &a.overlay else {
            panic!("expected Editor");
        };
        assert_eq!(fields[f0].value, format!("{before}x"));
        handle_key(&mut a, code(KeyCode::Backspace));
        let Overlay::Editor { fields, .. } = &a.overlay else {
            panic!("expected Editor");
        };
        assert_eq!(
            fields[f0].value, before,
            "Backspace should undo the typed char in the focused field"
        );
    }

    #[test]
    fn editor_field_navigation_wraps() {
        // ↑↓/Tab move focus between rows and wrap at the ends.
        let mut a = app();
        a.open_editor(&test_fixtures::TRAIN_DELTA);
        let n = match &a.overlay {
            Overlay::Editor { fields, .. } => fields.len(),
            _ => panic!("expected Editor"),
        };
        assert!(n >= 2, "need ≥2 fields to test navigation");
        // Down advances.
        handle_key(&mut a, code(KeyCode::Down));
        assert!(matches!(&a.overlay, Overlay::Editor { focus, .. } if *focus == 1));
        // Tab also advances; from the last row it wraps to 0.
        for _ in 1..n {
            handle_key(&mut a, code(KeyCode::Tab));
        }
        assert!(matches!(&a.overlay, Overlay::Editor { focus, .. } if *focus == 0));
        // Up from row 0 wraps to the last row.
        handle_key(&mut a, code(KeyCode::Up));
        assert!(matches!(&a.overlay, Overlay::Editor { focus, .. } if *focus == n - 1));
    }

    #[test]
    fn editor_assembles_typed_fields_to_json() {
        // F2: the form round-trips to a typed args JSON object — a number field
        // stays a JSON number, a string stays a string, and a blank optional
        // field is omitted (so serde's default applies).
        let mut a = app();
        a.open_editor(&test_fixtures::TRAIN_DELTA);
        // Set lr (number) + tag (string); the seed already has lr=0.001.
        if let Overlay::Editor { fields, .. } = &mut a.overlay {
            for f in fields.iter_mut() {
                match f.name.as_str() {
                    "lr" => f.value = "0.01".into(),
                    "tag" => f.value = "exp1".into(),
                    _ => {}
                }
            }
        }
        let json = match &a.overlay {
            Overlay::Editor { fields, .. } => super::assemble_fields(fields),
            _ => panic!("expected Editor"),
        };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["lr"], serde_json::json!(0.01), "number stays a JSON number");
        assert_eq!(v["tag"], serde_json::json!("exp1"), "string stays a string");

        // Blank the string field → omitted on assembly.
        if let Overlay::Editor { fields, .. } = &mut a.overlay {
            for f in fields.iter_mut() {
                if f.name == "tag" {
                    f.value.clear();
                }
            }
        }
        let json = match &a.overlay {
            Overlay::Editor { fields, .. } => super::assemble_fields(fields),
            _ => panic!("expected Editor"),
        };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("tag").is_none(), "blank field must be omitted");
    }

    // ── F3/U5: kind-filtered dataset picker → input injection ───────

    #[test]
    fn graph_input_recipe_skips_dataset_picker() {
        // A recipe with empty input_kinds has nothing to pick → straight to
        // the args form (picker is skipped entirely, unchanged behavior).
        let mut a = app();
        a.open_dataset_picker(&test_fixtures::TRAIN_DELTA);
        assert!(
            matches!(a.overlay, Overlay::Editor { .. }),
            "graph-input recipe must skip the dataset picker"
        );
    }

    #[test]
    fn dataset_picker_filters_by_kind_and_injects_registered_name() {
        // Register two datasets of different kinds into the isolated db; the
        // picker for a `dataset.jsonl` recipe must show only the jsonl one, and
        // picking it must inject its NAME into the `registered_name` form field.
        let mut a = app();
        let conn = crate::datasets_db::open().expect("isolated db opens");
        let td = std::env::temp_dir().join(format!("blut-tui-f3-{:?}", std::thread::current().id()));
        let _ = std::fs::create_dir_all(&td);
        let f = td.join("ds.jsonl");
        std::fs::write(&f, "{\"a\":1}\n{\"b\":2}\n").unwrap();
        // Unique names so parallel tests sharing the process db don't collide.
        let jsonl_name = format!("f3jsonl{:?}", std::thread::current().id())
            .replace(['(', ')', ' '], "");
        let split_name = format!("f3split{:?}", std::thread::current().id())
            .replace(['(', ')', ' '], "");
        let r1 = crate::datasets_db::record_from_jsonl(&jsonl_name, &f, "dataset.jsonl", None).unwrap();
        let r2 = crate::datasets_db::record_from_jsonl(&split_name, &f, "dataset.split", None).unwrap();
        crate::datasets_db::add(&conn, &r1).unwrap();
        crate::datasets_db::add(&conn, &r2).unwrap();

        a.open_dataset_picker(&test_fixtures::TRAIN_EPSILON);
        let names: Vec<String> = match &a.overlay {
            Overlay::DatasetPicker { datasets, .. } => {
                datasets.iter().map(|d| d.name.clone()).collect()
            }
            other => panic!("expected DatasetPicker, got {:?}", std::mem::discriminant(other)),
        };
        assert!(
            names.contains(&jsonl_name),
            "picker must list the dataset.jsonl dataset"
        );
        assert!(
            !names.contains(&split_name),
            "picker must FILTER OUT the dataset.split dataset (wrong kind)"
        );

        // Move the cursor onto our jsonl dataset, then Enter to pick it.
        let idx = match &a.overlay {
            Overlay::DatasetPicker { datasets, .. } => {
                datasets.iter().position(|d| d.name == jsonl_name).unwrap()
            }
            _ => unreachable!(),
        };
        for _ in 0..idx {
            handle_key(&mut a, code(KeyCode::Down));
        }
        handle_key(&mut a, code(KeyCode::Enter));
        // → editor with registered_name pre-filled from the picked dataset.
        match &a.overlay {
            Overlay::Editor { fields, .. } => {
                let rn = fields
                    .iter()
                    .find(|f| f.name == "registered_name")
                    .expect("editor must carry the registered_name field");
                assert_eq!(rn.value, jsonl_name, "picked dataset name injected");
            }
            _ => panic!("Enter on a dataset must open the args editor"),
        }
    }

    #[test]
    fn dataset_picker_esc_skips_to_editor_without_injection() {
        // Esc in the dataset picker skips → editor opens, no dataset injected.
        let mut a = app();
        let conn = crate::datasets_db::open().expect("isolated db opens");
        let td = std::env::temp_dir().join(format!("blut-tui-f3esc-{:?}", std::thread::current().id()));
        let _ = std::fs::create_dir_all(&td);
        let f = td.join("ds.jsonl");
        std::fs::write(&f, "{\"a\":1}\n").unwrap();
        let name = format!("f3esc{:?}", std::thread::current().id()).replace(['(', ')', ' '], "");
        let r = crate::datasets_db::record_from_jsonl(&name, &f, "dataset.jsonl", None).unwrap();
        crate::datasets_db::add(&conn, &r).unwrap();
        a.open_dataset_picker(&test_fixtures::TRAIN_EPSILON);
        assert!(matches!(a.overlay, Overlay::DatasetPicker { .. }));
        handle_key(&mut a, code(KeyCode::Esc));
        match &a.overlay {
            Overlay::Editor { fields, .. } => {
                let rn = fields.iter().find(|f| f.name == "registered_name").unwrap();
                assert!(rn.value.is_empty(), "Esc-skip must not inject a dataset");
            }
            _ => panic!("Esc in dataset picker should open the editor"),
        }
    }

    #[test]
    fn editor_raw_toggle_round_trips() {
        // F2 keeps a raw-JSON fallback (Ctrl+R). Flip to raw, confirm it holds
        // the assembled JSON; flip back, confirm the form survives.
        let mut a = app();
        a.open_editor(&test_fixtures::TRAIN_DELTA);
        assert!(matches!(&a.overlay, Overlay::Editor { raw_mode, .. } if !*raw_mode));
        handle_key(&mut a, ctrl('r')); // form → raw
        let raw = match &a.overlay {
            Overlay::Editor { raw_mode, raw_buffer, .. } => {
                assert!(*raw_mode, "Ctrl+R must enable raw mode");
                raw_buffer.clone()
            }
            _ => panic!("expected Editor"),
        };
        assert!(
            serde_json::from_str::<serde_json::Value>(&raw).is_ok(),
            "raw buffer must hold valid JSON"
        );
        handle_key(&mut a, ctrl('r')); // raw → form
        assert!(matches!(&a.overlay, Overlay::Editor { raw_mode, .. } if !*raw_mode));
    }

    #[test]
    fn editor_raw_toggle_malformed_json_stays_in_raw_mode() {
        // Review finding 1: flipping raw→form with MALFORMED JSON must NOT
        // silently drop to a stale form (and risk launching stale data). It
        // stays in raw mode + warns, so the user fixes the JSON first.
        let mut a = app();
        a.open_editor(&test_fixtures::TRAIN_DELTA);
        handle_key(&mut a, ctrl('r')); // form → raw
        // Corrupt the raw buffer.
        if let Overlay::Editor { raw_buffer, .. } = &mut a.overlay {
            *raw_buffer = "{ not valid json".into();
        }
        handle_key(&mut a, ctrl('r')); // raw → form attempt — must be refused
        match &a.overlay {
            Overlay::Editor { raw_mode, .. } => {
                assert!(*raw_mode, "malformed JSON must keep the editor in raw mode");
            }
            _ => panic!("expected Editor still open"),
        }
        assert!(
            a.status_msg
                .as_ref()
                .is_some_and(|(m, _)| m.contains("raw JSON")),
            "a parse-failure status must be shown"
        );
        // A bare (valid) array is also refused — the form needs an object.
        if let Overlay::Editor { raw_buffer, .. } = &mut a.overlay {
            *raw_buffer = "[1,2,3]".into();
        }
        handle_key(&mut a, ctrl('r'));
        assert!(
            matches!(&a.overlay, Overlay::Editor { raw_mode, .. } if *raw_mode),
            "a non-object JSON must keep raw mode"
        );
    }

    // ── filter_recipes fuzzy ordering ───────────────────────────────

    #[test]
    fn filter_recipes_empty_query_returns_all() {
        let all = App::filter_recipes(FIXTURE, "");
        assert_eq!(
            all.len(),
            FIXTURE.len(),
            "empty query must surface every recipe"
        );
        // Every catalog index appears exactly once.
        let mut seen = all.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), FIXTURE.len(), "no duplicate / missing indices");
    }

    #[test]
    fn filter_recipes_subset_query_orders_best_first() {
        // "train" matches every train_* fixture recipe; the result must be
        // a non-empty subset and every returned recipe's name must
        // actually fuzzy-contain the query subsequence.
        let q = "train";
        let res = App::filter_recipes(FIXTURE, q);
        assert!(!res.is_empty(), "`{q}` should match the train_* recipes");
        for &idx in &res {
            let name = FIXTURE[idx].name;
            assert!(
                is_subsequence(q, name),
                "fuzzy match returned `{name}` which does not contain `{q}` as a subsequence"
            );
        }
        // A more specific query is a strict-or-equal subset of a broader
        // prefix query.
        let broad = App::filter_recipes(FIXTURE, "tra");
        assert!(
            res.len() <= broad.len(),
            "narrower query must not return more rows than a broader one"
        );
    }

    #[test]
    fn filter_recipes_exact_name_ranks_that_recipe_first() {
        // Querying a full recipe name should rank that recipe at the top.
        for r in FIXTURE {
            let res = App::filter_recipes(FIXTURE, r.name);
            assert!(!res.is_empty(), "exact name `{}` matched nothing", r.name);
            assert_eq!(
                FIXTURE[res[0]].name, r.name,
                "exact-name query `{}` should rank itself first, got `{}`",
                r.name, FIXTURE[res[0]].name
            );
        }
    }

    #[test]
    fn filter_recipes_no_match_is_empty() {
        assert!(
            App::filter_recipes(FIXTURE, "zzz_definitely_not_a_recipe_zzz").is_empty(),
            "an impossible query must return no rows"
        );
    }

    /// True iff `needle` appears in `hay` as a (not-necessarily-contiguous)
    /// subsequence — the property a fuzzy matcher guarantees.
    fn is_subsequence(needle: &str, hay: &str) -> bool {
        let mut it = hay.chars();
        needle.chars().all(|nc| it.any(|hc| hc == nc))
    }

    // ── args_template: schema → defaults/placeholders object ────────

    #[test]
    fn args_template_is_an_object_for_every_recipe() {
        for r in FIXTURE {
            let tpl = crate::recipes::recipe::args_template(r);
            // The top level must be a JSON object (the args dict) — never a
            // bare scalar / array, even for a schema-less fixture (→ `{}`).
            assert!(
                tpl.is_object(),
                "args_template(`{}`) must be a JSON object, got {tpl}",
                r.name
            );
        }
    }

    #[test]
    fn open_editor_form_assembles_to_valid_json_for_every_recipe() {
        // F2: open_editor builds a per-field FORM. For every fixture recipe the
        // form must assemble back into a valid JSON object (the buffer the
        // submit path validates + launches).
        for r in FIXTURE {
            let mut a = app();
            a.open_editor(r);
            let Overlay::Editor { fields, .. } = &a.overlay else {
                panic!(
                    "open_editor must produce an Editor overlay for `{}`",
                    r.name
                );
            };
            let json = super::assemble_fields(fields);
            let v = serde_json::from_str::<serde_json::Value>(&json)
                .unwrap_or_else(|e| panic!("open_editor(`{}`) form → bad JSON: {e}\n{json}", r.name));
            assert!(v.is_object(), "assembled args must be a JSON object for `{}`", r.name);
        }
    }

    #[test]
    fn editor_args_validation_gates_launch() {
        // F2/U3: the Editor preflights args against the recipe schema before
        // spawning, so a non-AI human gets a precise message in-TUI.
        let a = app();
        let recipe = a.catalog.first().expect("fixture has a recipe").name;
        // The prefill is valid by construction → accepted.
        let good = a.registry.prefill_args(recipe);
        assert!(
            a.validate_editor_args(recipe, &good).is_ok(),
            "the prefilled buffer must validate for `{recipe}`"
        );
        // Malformed JSON → rejected (parse fails before the schema check).
        let bad = a.validate_editor_args(recipe, "{not json");
        assert!(bad.is_err(), "malformed JSON must be rejected");
        assert!(bad.unwrap_err().contains("JSON"), "message names the JSON fault");
        // Unknown recipe + valid JSON → permissive (serde / CLI compile
        // backstop it); the TUI must not block on a name it can't resolve.
        assert!(a.validate_editor_args("no_such_recipe_xyz", "{}").is_ok());
    }

    // ── recipe_menu hotkey assignment ───────────────────────────────

    #[test]
    fn recipe_menu_has_no_duplicate_hotkeys() {
        let menu = App::recipe_menu(FIXTURE);
        let mut seen = std::collections::HashSet::new();
        for (key, r) in &menu {
            if let Some(c) = key {
                assert!(
                    seen.insert(*c),
                    "duplicate hotkey `{c}` assigned (collides on recipe `{}`)",
                    r.name
                );
            }
        }
    }

    #[test]
    fn recipe_menu_never_reuses_reserved_keys() {
        // The built-in / navigation keys must never be handed to a recipe
        // hotkey or the recipe would shadow (or be shadowed by) the
        // built-in. Mirror the reserved set declared in recipe_menu().
        let reserved: &[char] = &['q', 'Q', 'r', 'R', 'c', 'C', 'j', 'k', 'l'];
        for (key, r) in App::recipe_menu(FIXTURE) {
            if let Some(c) = key {
                assert!(
                    !reserved.contains(&c),
                    "recipe `{}` was assigned reserved hotkey `{c}`",
                    r.name
                );
            }
        }
    }

    #[test]
    fn recipe_menu_lists_every_recipe_once() {
        let menu = App::recipe_menu(FIXTURE);
        assert_eq!(
            menu.len(),
            FIXTURE.len(),
            "menu must contain every recipe exactly once"
        );
        let mut names: Vec<&str> = menu.iter().map(|(_, r)| r.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), FIXTURE.len(), "no duplicate recipe rows");
    }

    // ── View navigation ─────────────────────────────────────────────

    #[test]
    fn capital_keys_switch_views() {
        // Each capital View key, pressed from the cockpit, switches to the
        // matching View. Invalid keys leave the view unchanged.
        let cases = [
            ('J', View::Jobs),
            ('L', View::Log),
            ('Y', View::System),
            ('H', View::History),
            ('B', View::Leaderboard),
            ('K', View::Checkpoints),
            ('P', View::Presets),
            ('M', View::Metrics),
            ('X', View::Reset),
        ];
        for (c, want) in cases {
            let mut a = app();
            assert_eq!(a.view, View::Cockpit);
            handle_key(&mut a, key(c));
            assert_eq!(a.view, want, "key `{c}` should switch to {want:?}");
        }
    }

    #[test]
    fn esc_or_b_returns_detail_view_to_cockpit() {
        // From a detail view, both Esc and 'b' go back to the cockpit.
        for back in [code(KeyCode::Esc), key('b')] {
            let mut a = app();
            handle_key(&mut a, key('J')); // → Jobs
            assert_eq!(a.view, View::Jobs);
            handle_key(&mut a, back);
            assert_eq!(a.view, View::Cockpit, "Esc/b should return to Cockpit");
        }
    }

    #[test]
    fn invalid_key_is_a_noop_in_cockpit() {
        // A key that is neither a view-switch, built-in, nor recipe
        // hotkey must not change view, overlay, or quit.
        let mut a = app();
        // '@' is not in the recipe hotkey pool (1-9,a-z minus reserved),
        // not a capital view key, and not a built-in.
        handle_key(&mut a, key('@'));
        assert_eq!(a.view, View::Cockpit);
        assert!(matches!(a.overlay, Overlay::None));
        assert!(!a.quit);
    }

    #[test]
    fn q_quits_from_any_view() {
        // 'q' is a global quit handled in handle_key_main before view
        // dispatch, so it works from the cockpit and from detail views.
        let mut a = app();
        handle_key(&mut a, key('q'));
        assert!(a.quit, "q should quit from the cockpit");

        let mut a = app();
        handle_key(&mut a, key('H')); // → History
        handle_key(&mut a, key('q'));
        assert!(a.quit, "q should quit from a detail view");
    }

    // ── TUI-07: picker cursor clamp (the bug fixed in this change) ───

    #[test]
    fn picker_down_past_end_clamps_in_range() {
        // Open the picker, narrow to a single match, then press Down many
        // times. Before the fix the cursor ran unbounded and Enter
        // no-op'd on a phantom row; after the fix it clamps to the last
        // filtered index so Enter always selects a real recipe.
        let mut a = app();
        handle_key(&mut a, key('R'));
        // Type a query that matches exactly one recipe.
        for ch in "train_alpha".chars() {
            handle_key(&mut a, key(ch));
        }
        let filtered = App::filter_recipes(FIXTURE, "train_alpha");
        let last = filtered.len().saturating_sub(1);
        // Hammer Down well past the end.
        for _ in 0..50 {
            handle_key(&mut a, code(KeyCode::Down));
        }
        let Overlay::Picker { query, cursor } = &a.overlay else {
            panic!("expected Picker overlay still open");
        };
        let live = App::filter_recipes(FIXTURE, query);
        assert!(
            *cursor <= last,
            "cursor {cursor} ran past last filtered index {last} (TUI-07 regressed)"
        );
        assert!(
            live.get(*cursor).is_some(),
            "cursor {cursor} must index a real filtered row (len {})",
            live.len()
        );
        // And Enter on the clamped cursor must actually SELECT a real recipe
        // (not silently no-op as it did before the clamp). For a recipe with
        // input_kinds that resolves a dataset it advances to the DatasetPicker;
        // otherwise straight to the Editor — either proves the clamp worked.
        handle_key(&mut a, code(KeyCode::Enter));
        assert!(
            matches!(a.overlay, Overlay::Editor { .. } | Overlay::DatasetPicker { .. }),
            "Enter at the clamped cursor must select a recipe (Editor or DatasetPicker), not no-op"
        );
    }

    #[test]
    fn picker_down_then_up_stays_in_range_on_empty_query() {
        // With the full list, Down should advance and never exceed the
        // last index; Up should walk back without underflowing.
        let mut a = app();
        handle_key(&mut a, key('R'));
        let last = App::filter_recipes(FIXTURE, "").len().saturating_sub(1);
        for _ in 0..(FIXTURE.len() + 20) {
            handle_key(&mut a, code(KeyCode::Down));
        }
        let Overlay::Picker { cursor, .. } = &a.overlay else {
            panic!("expected Picker");
        };
        assert_eq!(*cursor, last, "Down must saturate at the last index");
        for _ in 0..(FIXTURE.len() + 20) {
            handle_key(&mut a, code(KeyCode::Up));
        }
        let Overlay::Picker { cursor, .. } = &a.overlay else {
            panic!("expected Picker");
        };
        assert_eq!(*cursor, 0, "Up must saturate at 0");
    }
}
