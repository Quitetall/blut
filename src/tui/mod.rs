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
//! A [`View`] enum multiplexes the single ratatui surface across the
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
//!   <recipe hotkey> launch a recipe (pre-baked LamQuant defaults for
//!                   the lamquant_* recipes, schema template otherwise)
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
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};

use crate::jobs::{self, JobState, JobSummary};
use crate::recipes::RECIPES;

mod system;
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
    Picker { query: String, cursor: usize },
    Editor { recipe: &'static str, buffer: String },
}

/// A single row in the left-hand action menu. Mirrors the section /
/// hotkey / label / action layout of `lamquant-core/src/tui/panels/
/// cockpit.rs` so users moving from the lml cockpit see a familiar
/// shape.
struct MenuItem {
    section: &'static str,
    key: char,
    label: &'static str,
    action: MenuAction,
}

enum MenuAction {
    /// Spawn a built-in recipe with default LamQuant args.
    Recipe(&'static str),
    /// Run a built-in app action (refresh / quit / cancel / overlay).
    Builtin(BuiltinAction),
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
    fn new() -> Self {
        let mut selected = ListState::default();
        selected.select(Some(0));
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
    /// Returns `(idx_in_RECIPES, score)` pairs sorted by score desc.
    fn filter_recipes(query: &str) -> Vec<usize> {
        use fuzzy_matcher::skim::SkimMatcherV2;
        use fuzzy_matcher::FuzzyMatcher;
        let matcher = SkimMatcherV2::default();
        let mut scored: Vec<(usize, i64)> = RECIPES
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
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| RECIPES[a.0].name.cmp(RECIPES[b.0].name)));
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
        let props = args_schema
            .get("properties")
            .and_then(|p| p.as_object());
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
        let buffer = Self::lamquant_default_args(recipe.name)
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
    /// `RECIPES` (sorted by category + name), assigning hotkeys from
    /// the pool `1..9, then a..z` skipping the reserved keys (q quit,
    /// r refresh, c cancel, R custom-recipe-picker, j/k vi navigation,
    /// l reserved for U3 log toggle). Recipes beyond the available
    /// hotkeys still show in the menu but require `R` to launch.
    fn recipe_menu() -> Vec<(Option<char>, &'static crate::recipes::RecipeDef)> {
        use crate::recipes::RecipeCategory;
        let category_order = [
            RecipeCategory::DataPrep,
            RecipeCategory::Train,
            RecipeCategory::Eval,
            RecipeCategory::Export,
            RecipeCategory::Pipeline,
            RecipeCategory::User,
        ];
        let mut sorted: Vec<&'static crate::recipes::RecipeDef> = RECIPES.iter().copied().collect();
        sorted.sort_by(|a, b| {
            let ai = category_order.iter().position(|c| *c == a.category).unwrap_or(99);
            let bi = category_order.iter().position(|c| *c == b.category).unwrap_or(99);
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

    /// Pre-baked args JSON for the four LamQuant training recipes.
    /// Points at the corpus paths the rest of the repo uses by
    /// default. Override via the `R` custom-recipe overlay.
    fn lamquant_default_args(name: &str) -> Option<String> {
        let lma = "/mnt/4tb/data/lma";
        let split = "/mnt/4tb/LamQuant/data/manifests/snn_train_val_split.json";
        let labels = "/mnt/4tb/LamQuant/ai_models/snn/labels";
        let eeg = "/mnt/4tb/data/lml/edf.lml";
        Some(match name {
            "lamquant_data_prep" => format!(
                r#"{{
  "lml_root": "{eeg}",
  "output_dir": "{lma}"
}}"#
            ),
            "lamquant_snn" => format!(
                r#"{{
  "labels_dir": "{labels}",
  "eeg_dir": "{eeg}",
  "preset": "production",
  "subband": true,
  "epochs": 5,
  "lma_output_dir": "{lma}",
  "convert_limit": 1,
  "split_manifest": "{split}"
}}"#
            ),
            "lamquant_encoder" => format!(
                r#"{{
  "lma_output_dir": "{lma}",
  "split_manifest": "{split}"
}}"#
            ),
            "lamquant_combined_decoder" => format!(
                r#"{{
  "lma_output_dir": "{lma}",
  "split_manifest": "{split}"
}}"#
            ),
            "lamquant_oracle" => format!(
                r#"{{
  "lma_output_dir": "{lma}",
  "split_manifest": "{split}"
}}"#
            ),
            _ => return None,
        })
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
        let Some(job) = self.jobs.get(idx) else { return };
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
        let Some(job) = self.jobs.get(idx) else { return };
        if !matches!(job.state, JobState::Running) {
            self.set_status(format!("job {} is {} — nothing to cancel", job.id, job.state.as_str()));
            return;
        }
        let id = job.id.clone();
        self.set_status(format!("SIGTERM {id} (grace 10s)..."));
        // Spawn detached process so we don't block the UI on grace
        // period; user sees state flip on next refresh.
        let _ = std::process::Command::new(std::env::current_exe().unwrap_or_else(|_| "blut".into()))
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
            self.set_status(format!("marked {name} for compare ({}/3)", self.marked.len()));
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

/// Entrypoint registered as `blut tui`.
pub async fn run() -> Result<()> {
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture).context("alt screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend).context("terminal")?;

    let result = run_app(&mut term).await;

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

async fn run_app<B: ratatui::backend::Backend>(term: &mut Terminal<B>) -> Result<()> {
    let mut app = App::new();
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
            KeyCode::Up | KeyCode::Char('k')
                if k.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                *cursor = cursor.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j')
                if k.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                *cursor += 1
            }
            KeyCode::Up => *cursor = cursor.saturating_sub(1),
            KeyCode::Down => *cursor += 1,
            KeyCode::Backspace => {
                query.pop();
                *cursor = 0;
            }
            KeyCode::Char(c) => {
                query.push(c);
                *cursor = 0;
            }
            KeyCode::Enter => {
                let filtered = App::filter_recipes(query);
                if let Some(idx) = filtered.get(*cursor) {
                    let recipe = RECIPES[*idx];
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
            let menu = App::recipe_menu();
            if let Some((_, recipe)) = menu.iter().find(|(k, _)| *k == Some(c)) {
                app.open_editor(*recipe);
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
            let msg = views::export_presets(&app.repo_root);
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

/// Cockpit view — the single-column overview (header strip → Pipeline
/// Status box → Resources block → divider → grouped recipe menu → key
/// list). Renders as one big Paragraph so layout reflows with width.
fn draw_cockpit_body(f: &mut Frame<'_>, area: Rect, app: &mut App) {
    let outer = [area];
    let total_w = outer[0].width as usize;
    let inner_w = total_w.saturating_sub(4);
    let dash: String = "─".repeat(inner_w.max(10));

    let mut lines: Vec<Line> = Vec::new();
    // ── Header ───────────────────────────────────────────────────────
    lines.push(Line::from(Span::styled(
        format!("  {dash}"),
        Style::default().fg(Color::DarkGray),
    )));
    let title_left = "BLUT Training Cockpit";
    let title_right = format!("blut v{}", env!("CARGO_PKG_VERSION"));
    let pad = inner_w.saturating_sub(title_left.len() + title_right.len() + 1);
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            title_left,
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" ".repeat(pad)),
        Span::styled(title_right, Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::from(Span::styled(
        format!("  {dash}"),
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));

    // ── Pipeline Status box ──────────────────────────────────────────
    let inner_inner = inner_w.saturating_sub(4);
    let top = format!("┌─ Pipeline status {}┐", "─".repeat(inner_inner.saturating_sub(17)));
    let bot = format!("└{}┘", "─".repeat(inner_inner.saturating_sub(1) + 1));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(top, Style::default().fg(Color::DarkGray)),
    ]));
    let running: Vec<&JobSummary> = app
        .jobs
        .iter()
        .filter(|j| matches!(j.state, JobState::Running))
        .collect();
    if running.is_empty() {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("│", Style::default().fg(Color::DarkGray)),
            Span::raw("   "),
            Span::styled(
                "No training jobs running",
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" ".repeat(inner_inner.saturating_sub(28))),
            Span::styled("│", Style::default().fg(Color::DarkGray)),
        ]));
    } else {
        for j in &running {
            let pid = j.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
            let last = match (j.last_step, j.last_loss) {
                (Some(step), Some(loss)) => format!("step={step} loss={loss:.4}"),
                _ => "-".into(),
            };
            let line_str = format!(
                "   ● {} pid {} · {}",
                j.id,
                pid,
                last
            );
            let display_len = line_str.chars().count();
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("│", Style::default().fg(Color::DarkGray)),
                Span::styled(line_str, Style::default().fg(Color::Green)),
                Span::raw(" ".repeat(inner_inner.saturating_sub(display_len + 1))),
                Span::styled("│", Style::default().fg(Color::DarkGray)),
            ]));
        }
    }
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(bot, Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::from(""));

    // ── Resources block ──────────────────────────────────────────────
    let s = &app.system;
    let gpu_summary = s.gpu_summary();
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            "Resources",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("         "),
        Span::styled(format!("GPU {}", gpu_summary), Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::from(vec![
        Span::raw("                    "),
        Span::styled(
            format!(
                "CPU load1={:.2}  ·  MEM {:.1}/{:.1} GiB used  ·  Disk {} free",
                s.load1, s.mem_used_gb, s.mem_total_gb, s.disk_free_human
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  {dash}"),
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));

    // ── Recipe menu, grouped by category ────────────────────────────
    let menu = App::recipe_menu();
    use std::collections::BTreeMap;
    let mut by_cat: BTreeMap<&'static str, Vec<(Option<char>, &'static crate::recipes::RecipeDef)>> =
        BTreeMap::new();
    // Preserve the recipe_menu order within each category by using a
    // Vec value, not a Set.
    for (k, r) in &menu {
        by_cat.entry(r.category.label()).or_default().push((*k, *r));
    }
    // Render in fixed order matching the menu's natural sort.
    let mut printed_cats: Vec<&'static str> = Vec::new();
    for (_, r) in &menu {
        let cat = r.category.label();
        if !printed_cats.contains(&cat) {
            printed_cats.push(cat);
        }
    }
    for cat in &printed_cats {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                *cat,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
        for (k, r) in by_cat.get(cat).unwrap() {
            let key_str = match k {
                Some(c) => format!("[{c}]"),
                None => " - ".into(),
            };
            // Truncate description to fit comfortably.
            let max_desc = inner_w.saturating_sub(32 + r.name.len() + 6);
            let desc: String = r
                .description
                .chars()
                .take(max_desc.max(20))
                .collect();
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{key_str:<4}"),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    format!("{:<28}", r.name),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(desc, Style::default().fg(Color::DarkGray)),
            ]));
        }
        lines.push(Line::from(""));
    }
    // Built-in keys footer line in the menu (matches the SYSTEM
    // section of the lamquant-core cockpit).
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            "BUILT-IN",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(""));
    for (k, _, label) in App::builtin_keys() {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("[{}] ", k),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::raw(*label),
        ]));
    }

    // ── Views navigation (migrated screens) ─────────────────────────
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            "VIEWS",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(""));
    for (key, label) in [
        ("J", "Jobs (all states)"),
        ("L", "Log tail (selected job)"),
        ("Y", "System (GPU/MEM/DISK/CPU)"),
        ("H", "Run history"),
        ("B", "Leaderboard (ranked by R)"),
        ("K", "Checkpoints browser"),
        ("P", "Presets & hyperparameters"),
        ("M", "Live metrics tail"),
        ("X", "Reset / export (maintenance)"),
    ] {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("[{key}] "),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::raw(label),
        ]));
    }

    let para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(para, outer[0]);
}

fn centered_rect(area: Rect, pct_w: u16, pct_h: u16) -> Rect {
    let w = area.width.saturating_mul(pct_w) / 100;
    let h = area.height.saturating_mul(pct_h) / 100;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

fn draw_overlay(f: &mut Frame<'_>, app: &App) {
    match &app.overlay {
        Overlay::None => {}
        Overlay::Picker { query, cursor } => {
            let area = centered_rect(f.area(), 60, 60);
            // Clear by drawing an empty block underneath.
            let bg = Block::default()
                .borders(Borders::ALL)
                .title(" pick recipe (type to filter, ↑↓ select, Enter open, Esc cancel) ");
            f.render_widget(bg.clone(), area);

            let inner = Rect {
                x: area.x + 1,
                y: area.y + 1,
                width: area.width.saturating_sub(2),
                height: area.height.saturating_sub(2),
            };
            let query_line = Line::from(vec![
                Span::styled("> ", Style::default().fg(Color::Yellow)),
                Span::raw(query.clone()),
                Span::styled("_", Style::default().fg(Color::DarkGray)),
            ]);
            let query_widget = Paragraph::new(query_line);
            let query_area = Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 };
            f.render_widget(query_widget, query_area);

            let list_area = Rect {
                x: inner.x,
                y: inner.y + 2,
                width: inner.width,
                height: inner.height.saturating_sub(2),
            };
            let filtered = App::filter_recipes(query);
            let items: Vec<ListItem> = filtered
                .iter()
                .enumerate()
                .map(|(i, &idx)| {
                    let r = RECIPES[idx];
                    let style = if i == *cursor {
                        Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("{:<32} ", r.name),
                            style.add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("[{}] ", r.backend_id), Style::default().fg(Color::Cyan)),
                        Span::styled(
                            r.description.chars().take(80).collect::<String>(),
                            Style::default().fg(Color::DarkGray),
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
                .title(format!(
                    " edit args: {recipe} — type to edit, Backspace, Ctrl+Enter submit, Esc cancel "
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
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.jobs
            .iter()
            .map(|j| {
                let state_color = match j.state {
                    JobState::Running => Color::Green,
                    JobState::Done => Color::Cyan,
                    JobState::Failed => Color::Red,
                    JobState::Cancelled => Color::Yellow,
                };
                let pid = j.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
                let output = j.output_name.clone().unwrap_or_else(|| "-".into());
                let last = match (j.last_step, j.last_loss, j.final_loss) {
                    (_, _, Some(fl)) => format!("final_loss={fl:.4}"),
                    (Some(step), Some(loss), _) => format!("step={step} loss={loss:.4}"),
                    _ => "-".into(),
                };
                ListItem::new(Line::from(vec![
                    Span::raw(format!("{:<22} ", j.id)),
                    Span::styled(
                        format!("{:<9} ", j.state.as_str()),
                        Style::default().fg(state_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(format!("pid={:<6} ", pid)),
                    Span::raw(format!("out={:<18} ", output)),
                    Span::styled(last, Style::default().fg(Color::DarkGray)),
                ]))
            })
            .collect()
    };
    let block = Block::default()
        .title(format!(" jobs ({}) — ↑↓ select, c cancel, r refresh ", app.jobs.len()))
        .borders(Borders::ALL);
    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, area, &mut app.selected);
}

fn draw_log(f: &mut Frame<'_>, area: Rect, app: &App) {
    let title = match &app.log_job_id {
        Some(id) => format!(" log: {id}  (Enter to refresh) "),
        None => " log: (no selection) ".into(),
    };
    let block = Block::default().title(title).borders(Borders::ALL);
    let text: Vec<Line> = if app.log_lines.is_empty() {
        vec![Line::from(Span::styled(
            "(no status.jsonl yet — job may still be starting)",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        // Show the tail that fits in the visible height. Reserve 2
        // rows for the borders.
        let visible = area.height.saturating_sub(2) as usize;
        let start = app.log_lines.len().saturating_sub(visible);
        app.log_lines[start..]
            .iter()
            .map(|s| Line::from(Span::raw(s.clone())))
            .collect()
    };
    let para = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_system(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().title(" system ").borders(Borders::ALL);
    let snap = &app.system;
    let lines: Vec<Line> = vec![
        Line::from(Span::styled("GPU", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw(snap.gpu_summary())),
        Line::from(""),
        Line::from(Span::styled("MEM", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw(format!(
            "used {:.1}/{:.1} GB  free {:.1} GB",
            snap.mem_used_gb, snap.mem_total_gb, snap.mem_avail_gb
        ))),
        Line::from(""),
        Line::from(Span::styled("DISK /mnt/4tb", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw(format!(
            "free {} ({}% used)",
            snap.disk_free_human, snap.disk_used_pct
        ))),
        Line::from(""),
        Line::from(Span::styled("CPU", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
        Line::from(Span::raw(format!("load1={:.2}", snap.load1))),
    ];
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: true });
    f.render_widget(para, area);
}

/// Shared header line for the migrated detail views.
fn view_header(title: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            title.to_string(),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("    blut v{}", env!("CARGO_PKG_VERSION")),
            Style::default().fg(Color::DarkGray),
        ),
    ])
}

/// Run History view (Python `_screen_history`): training logs + best-R
/// + epoch + date, plus a checkpoint summary footer.
fn draw_history(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" {} (↑↓ move · m mark · C compare · b back) ", View::History.title()))
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Run History"), Line::from("")];
    if app.runs.is_empty() {
        lines.push(Line::from(Span::styled(
            "No training runs found under training_logs/*.csv.",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!("  {:<42} {:<10} {:<10} {}", "Name", "Best R", "Epoch", "Date"),
            Style::default().fg(Color::DarkGray),
        )));
        for (i, r) in app.runs.iter().enumerate() {
            let marked = app.marked.iter().any(|n| *n == r.name);
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
                Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{prefix}{mark} "), Style::default().fg(Color::Green)),
                Span::styled(
                    format!("{:<42} {:<10} {:<10} ", truncate(&r.name, 42), r_str, ep_str),
                    style,
                ),
                Span::styled(r.date.clone(), Style::default().fg(Color::DarkGray)),
            ]));
        }
    }
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Leaderboard view (Python `_screen_leaderboard`): runs ranked by best
/// R descending, gold marker on #1.
fn draw_leaderboard(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" {} (↑↓ move · m mark · C compare · b back) ", View::Leaderboard.title()))
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Model Leaderboard"), Line::from("")];
    if app.runs.is_empty() {
        lines.push(Line::from(Span::styled(
            "No training logs found. Run some experiments first.",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!("  {:<5} {:<40} {:<10} {:<10} {}", "Rank", "Name", "Best R", "Epoch", "Date"),
            Style::default().fg(Color::DarkGray),
        )));
        for (i, r) in app.runs.iter().enumerate().take(20) {
            let cursor = i == app.list_cursor;
            let marked = app.marked.iter().any(|n| *n == r.name);
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
                Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
            } else if i == 0 {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let mark = if marked { "✓" } else { " " };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{mark} {:<5} {:<40} {:<10} {:<10} ", i + 1, truncate(&r.name, 40), r_str, ep_str),
                    style,
                ),
                Span::styled(format!("{}{medal}", r.date), Style::default().fg(Color::DarkGray)),
            ]));
        }
        if app.runs.len() > 20 {
            lines.push(Line::from(Span::styled(
                format!("  ... {} more runs", app.runs.len() - 20),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Compare view (Python `_screen_compare`): side-by-side metric table
/// of the marked runs; the per-row winner is highlighted green.
fn draw_compare(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" {} (mark runs in History/Leaderboard with m · b back) ", View::Compare.title()))
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
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        // Header row of run names.
        let mut hdr = vec![Span::styled(
            format!("  {:<16}", "Metric"),
            Style::default().fg(Color::DarkGray),
        )];
        for r in &selected {
            hdr.push(Span::styled(
                format!("{:<22}", truncate(&r.name, 21)),
                Style::default().add_modifier(Modifier::BOLD),
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
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// One Compare-table row. `highlight_max` greens the winning column;
/// `decimals` controls float formatting.
fn metric_row(metric: &str, vals: &[f64], highlight_max: bool, decimals: usize) -> Line<'static> {
    let best = vals.iter().cloned().fold(f64::MIN, f64::max);
    let mut spans = vec![Span::styled(
        format!("  {:<16}", metric),
        Style::default().add_modifier(Modifier::BOLD),
    )];
    for v in vals {
        let txt = if decimals == 0 {
            format!("{:<22}", *v as i64)
        } else {
            format!("{:<22.*}", decimals, v)
        };
        let style = if highlight_max && *v == best && best > 0.0 {
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        spans.push(Span::styled(txt, style));
    }
    Line::from(spans)
}

/// Checkpoints view (Python `_screen_checkpoints`): all `.ckpt` grouped
/// by directory, with per-dir count + GiB and the newest files.
fn draw_checkpoints(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" {} (↑↓ move · r refresh · b back) ", View::Checkpoints.title()))
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Checkpoints"), Line::from("")];
    if app.ckpts.is_empty() {
        lines.push(Line::from(Span::styled(
            "No checkpoints found under checkpoints/ or weights/.",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        let total_gb: f64 = app.ckpts.iter().map(|c| c.size_mb).sum::<f64>() / 1024.0;
        lines.push(Line::from(Span::styled(
            format!("{} checkpoints  ·  {:.1} GiB total", app.ckpts.len(), total_gb),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(""));
        for (i, c) in app.ckpts.iter().enumerate() {
            let cursor = i == app.list_cursor;
            let prefix = if cursor { "▶ " } else { "  " };
            let style = if cursor {
                Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), Style::default().fg(Color::Green)),
                Span::styled(format!("{:<40} ", truncate(&c.name, 40)), style),
                Span::styled(
                    format!("{:>8.1} MB  ", c.size_mb),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(format!("{}  ", c.date), Style::default().fg(Color::DarkGray)),
                Span::styled(c.rel_dir.clone(), Style::default().fg(Color::DarkGray)),
            ]));
        }
    }
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Presets & Hyperparameters view (Python `_screen_presets` +
/// `_screen_hparams` + decoder tiers + validated features). Read-only
/// catalog — the live values live in recipe Args JSON (ADR 0017).
fn draw_presets(f: &mut Frame<'_>, area: Rect, _app: &App) {
    let block = Block::default()
        .title(format!(" {} (b back) ", View::Presets.title()))
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Presets & Hyperparameters"), Line::from("")];
    lines.push(Line::from(Span::styled(
        "PRESETS",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    for (name, ep, wpe, est, use_case) in views::PRESETS {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<12}", name),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("{ep:<12} {wpe:<10} ")),
            Span::styled(format!("{est:<8}  "), Style::default().fg(Color::Yellow)),
            Span::styled(use_case.to_string(), Style::default().fg(Color::DarkGray)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "DECODER TIERS",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    for (tier, params, note) in views::DECODER_TIERS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {tier:<10}"), Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{params:<8} ")),
            Span::styled(note.to_string(), Style::default().fg(Color::DarkGray)),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "PRODUCTION-VALIDATED FEATURES",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    for feat in views::VALIDATED_FEATURES {
        lines.push(Line::from(Span::raw(format!("  • {feat}"))));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "HYPERPARAMETERS (set via recipe Args JSON — ADR 0017)",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    for (group, fields) in views::HPARAM_GROUPS {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<16}", group),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(fields.join(", "), Style::default().fg(Color::DarkGray)),
        ]));
    }
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Live Metrics view (Python `_screen_live_metrics` terminal tail):
/// the tail of the newest training-log CSV, re-read each tick.
fn draw_metrics(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" {} (r refresh · b back) ", View::Metrics.title()))
        .borders(Borders::ALL);
    let n = area.height.saturating_sub(4) as usize;
    let tail = views::metrics_tail(&app.repo_root, n.max(10));
    let lines: Vec<Line> = std::iter::once(view_header("Live Metrics"))
        .chain(std::iter::once(Line::from("")))
        .chain(tail.into_iter().map(|s| {
            if s.starts_with('#') {
                Line::from(Span::styled(s, Style::default().fg(Color::Cyan)))
            } else {
                Line::from(Span::raw(s))
            }
        }))
        .collect();
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Reset view (Python `_screen_reset` + `_screen_export`): the three
/// destructive maintenance actions (two-press Enter confirm) + export.
fn draw_reset(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" {} (↑↓ move · Enter confirm · e export · b back) ", View::Reset.title()))
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Reset Training State"), Line::from("")];
    lines.push(Line::from(Span::styled(
        "Destructive — each action requires a second Enter to confirm.",
        Style::default().fg(Color::Yellow),
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
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
        } else if cursor {
            Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let suffix = if armed { "   ← press Enter again to confirm" } else { "" };
        lines.push(Line::from(vec![
            Span::styled(prefix.to_string(), Style::default().fg(Color::Green)),
            Span::styled(format!("{}{suffix}", action.label()), style),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  [e] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::raw("Export configuration "),
        Span::styled(
            "(write recipe Args JSON schemas to repo root)",
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
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
    let line = if let Some((msg, _)) = &app.status_msg {
        format!("{msg}    │    {base}")
    } else {
        base.into()
    };
    let para = Paragraph::new(Span::styled(line, Style::default().fg(Color::DarkGray)));
    f.render_widget(para, area);
}
