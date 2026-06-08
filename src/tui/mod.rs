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

    pub(super) static TRAIN_ALPHA: RecipeDef = RecipeDef {
        name: "train_alpha",
        description: "fixture training recipe alpha",
        backend_id: "fixture",
        category: RecipeCategory::Train,
        input_kinds: &["dataset.jsonl"],
        output_kind: "checkpoint.hf",
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
        args_schema_fn: empty_object_schema,
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };

    /// The fixture catalog the TUI tests index (stands in for the recipes
    /// a real cookbook crate registers at runtime). Train + Eval
    /// categories exercise the category-grouped cockpit/menu paths.
    pub(super) static FIXTURE: &[&RecipeDef] = &[&TRAIN_ALPHA, &TRAIN_BETA, &EVAL_GAMMA];

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

/// Modal overlay state. `None` = main jobs+log view; `Picker` floats
/// a recipe list over the main view; `Editor` shows a single-line
/// text buffer prefilled with the recipe's args JSON template.
enum Overlay {
    None,
    Picker {
        query: String,
        cursor: usize,
    },
    Editor {
        recipe: &'static str,
        buffer: String,
    },
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

    /// Build a minimal JSON template from a recipe's schemars schema.
    /// Top-level required fields get `"<TODO>"`; optional fields are
    /// omitted (caller can add post-edit). Falls back to `{}` on any
    /// schema-parse error.
    fn template_for(recipe: &'static crate::recipes::RecipeDef) -> String {
        let schema = (recipe.args_schema_fn)();
        let Some(defs) = schema.get("definitions").and_then(|d| d.as_object()) else {
            return "{}".into();
        };
        // Args is referenced via "$ref": "#/definitions/Args".
        let Some(args_schema) = defs.get("Args").and_then(|a| a.as_object()) else {
            return "{}".into();
        };
        let required: Vec<&str> = args_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let props = args_schema.get("properties").and_then(|p| p.as_object());
        let mut out = serde_json::Map::new();
        for field in &required {
            // Try to render a type-aware placeholder so the user
            // doesn't have to guess.
            let placeholder = props
                .and_then(|p| p.get(*field))
                .and_then(|s| s.get("type"))
                .and_then(|t| t.as_str())
                .map(|ty| match ty {
                    "string" => serde_json::Value::String("<TODO>".into()),
                    "number" | "integer" => serde_json::Value::Number(0.into()),
                    "boolean" => serde_json::Value::Bool(false),
                    "array" => serde_json::Value::Array(vec![]),
                    "object" => serde_json::Value::Object(serde_json::Map::new()),
                    _ => serde_json::Value::String("<TODO>".into()),
                })
                .unwrap_or_else(|| serde_json::Value::String("<TODO>".into()));
            out.insert((*field).to_string(), placeholder);
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(out))
            .unwrap_or_else(|_| "{}".into())
    }

    fn open_picker(&mut self) {
        self.overlay = Overlay::Picker {
            query: String::new(),
            cursor: 0,
        };
    }

    fn submit_editor(&mut self) {
        let Overlay::Editor { recipe, buffer } = &self.overlay else {
            return;
        };
        let recipe = *recipe;
        let buffer = buffer.clone();
        self.spawn_recipe(recipe, &buffer);
        self.overlay = Overlay::None;
    }

    /// Open the args Editor overlay for a recipe, prefilled with the
    /// best available starting point: the pre-baked LamQuant corpus-path
    /// defaults for the `lamquant_*` recipes, else the schemars template.
    /// (Previously the hotkey path always used `template_for`, leaving
    /// `lamquant_default_args` dead — this revives it.)
    fn open_editor(&mut self, recipe: &'static crate::recipes::RecipeDef) {
        // Prefill with the owning cookbook's pre-baked default args (domain
        // data, supplied via Cookbook::default_args), else the schemars
        // template. Keeps blut-core domain-agnostic — no hardcoded paths.
        let buffer = self
            .registry
            .default_args(recipe.name)
            .unwrap_or_else(|| Self::template_for(recipe));
        self.overlay = Overlay::Editor {
            recipe: recipe.name,
            buffer,
        };
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
            Ok(child) => self.set_status(format!(
                "spawned '{name}' (pid {}). Check jobs list next tick.",
                child.id()
            )),
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
        use crate::recipes::RecipeCategory;
        let category_order = [
            RecipeCategory::DataPrep,
            RecipeCategory::Train,
            RecipeCategory::Eval,
            RecipeCategory::Export,
            RecipeCategory::Pipeline,
            RecipeCategory::User,
        ];
        let mut sorted: Vec<&'static crate::recipes::RecipeDef> = catalog.to_vec();
        sorted.sort_by(|a, b| {
            let ai = category_order
                .iter()
                .position(|c| *c == a.category)
                .unwrap_or(99);
            let bi = category_order
                .iter()
                .position(|c| *c == b.category)
                .unwrap_or(99);
            ai.cmp(&bi).then_with(|| a.name.cmp(b.name))
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
        Overlay::Editor { buffer, .. } => match k.code {
            KeyCode::Esc => app.overlay = Overlay::None,
            KeyCode::Enter
                if k.modifiers.contains(KeyModifiers::CONTROL)
                    || k.modifiers.contains(KeyModifiers::SHIFT) =>
            {
                app.submit_editor();
            }
            KeyCode::Char('\n') => {
                buffer.push('\n');
            }
            KeyCode::Enter => {
                // Plain Enter inserts a newline so the user can edit
                // multi-line JSON; Ctrl/Shift-Enter submits.
                buffer.push('\n');
            }
            KeyCode::Backspace => {
                buffer.pop();
            }
            KeyCode::Char(c) => buffer.push(c),
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
                    app.open_editor(recipe);
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
            // Recipe hotkeys (auto-assigned per category order). Opens
            // the args editor prefilled with pre-baked LamQuant defaults
            // (or the schema template for non-lamquant recipes).
            let menu = App::recipe_menu(&app.catalog);
            if let Some((_, recipe)) = menu.iter().find(|(k, _)| *k == Some(c)) {
                app.open_editor(recipe);
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
        Overlay::Editor { recipe, buffer } => {
            let area = centered_rect(f.area(), 70, 70);
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(theme::dim())
                .title(Span::styled(
                    format!(
                        " edit args: {recipe} — type to edit, Backspace, Ctrl+Enter submit, Esc cancel "
                    ),
                    theme::title(),
                ));
            let text: Vec<Line> = buffer.lines().map(|l| Line::from(l.to_string())).collect();
            let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
            f.render_widget(para, area);
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
    fn editor_typing_appends_and_backspaces_buffer() {
        let mut a = app();
        handle_key(&mut a, key('R'));
        handle_key(&mut a, code(KeyCode::Enter)); // → Editor with template
        // Capture the prefill, type, then backspace once.
        let Overlay::Editor { buffer, .. } = &a.overlay else {
            panic!("expected Editor");
        };
        let before = buffer.clone();
        handle_key(&mut a, key('x'));
        let Overlay::Editor { buffer, .. } = &a.overlay else {
            panic!("expected Editor");
        };
        assert_eq!(*buffer, format!("{before}x"));
        handle_key(&mut a, code(KeyCode::Backspace));
        let Overlay::Editor { buffer, .. } = &a.overlay else {
            panic!("expected Editor");
        };
        assert_eq!(*buffer, before, "Backspace should undo the typed char");
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

    // ── template_for: schema → JSON (or {} fallback) ────────────────

    #[test]
    fn template_for_every_recipe_parses_as_json() {
        for r in FIXTURE {
            let tpl = App::template_for(r);
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&tpl);
            assert!(
                parsed.is_ok(),
                "template_for(`{}`) produced unparseable JSON:\n{tpl}",
                r.name
            );
            // Whatever it is, the top level must be a JSON object (the
            // args dict) — never a bare scalar / array.
            assert!(
                parsed.unwrap().is_object(),
                "template_for(`{}`) must be a JSON object",
                r.name
            );
        }
    }

    #[test]
    fn open_editor_prefill_parses_for_every_recipe() {
        // The hotkey path prefills via lamquant_default_args() else the
        // schema template. Either way the prefill must be valid JSON so
        // the user starts from a parseable buffer.
        for r in FIXTURE {
            let mut a = app();
            a.open_editor(r);
            let Overlay::Editor { buffer, .. } = &a.overlay else {
                panic!(
                    "open_editor must produce an Editor overlay for `{}`",
                    r.name
                );
            };
            assert!(
                serde_json::from_str::<serde_json::Value>(buffer).is_ok(),
                "open_editor(`{}`) prefilled unparseable JSON:\n{buffer}",
                r.name
            );
        }
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
        // And Enter on the clamped cursor must actually open the Editor
        // (not silently no-op as it did before the clamp).
        handle_key(&mut a, code(KeyCode::Enter));
        assert!(
            matches!(a.overlay, Overlay::Editor { .. }),
            "Enter at the clamped cursor must open the Editor, not no-op"
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
