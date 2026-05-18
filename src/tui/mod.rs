//! `blut tui` — interactive training cockpit.
//!
//! Single-screen ratatui app that fuses three previously-disjoint
//! surfaces into one:
//!   * `blut jobs`            → top-left jobs list
//!   * `blut log <id> --tail` → bottom log tail of the selected job
//!   * `nvidia-smi` + `free`  → right-hand system probes panel
//!
//! Keybindings:
//!   ↑ / ↓ / j / k   move job selection
//!   Enter           refresh log immediately
//!   r               refresh jobs + system
//!   c               cancel selected job (SIGTERM via `blut cancel`)
//!   q / Esc / Ctrl-C   quit
//!
//! Lift origin: design borrowed from `lamquant-core/src/tui/panels/`
//! (cockpit + output + file_browser). T1 of the BLUT cockpit track
//! per /home/brianklam/.claude/plans/optimized-seeking-bentley.md.

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

/// Modal overlay state. `None` = main jobs+log view; `Picker` floats
/// a recipe list over the main view; `Editor` shows a single-line
/// text buffer prefilled with the recipe's args JSON template.
enum Overlay {
    None,
    Picker { query: String, cursor: usize },
    Editor { recipe: &'static str, buffer: String },
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
}

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
        }
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
        // Spawn `blut recipe run <recipe> --args '<buffer>'` detached.
        // Status reflects in jobs list on next refresh tick.
        let exe = std::env::current_exe().unwrap_or_else(|_| "blut".into());
        match std::process::Command::new(exe)
            .args(["recipe", "run", recipe, "--args", &buffer])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(child) => self.set_status(format!(
                "spawned recipe '{recipe}' (pid {}). Check jobs list next tick.",
                child.id()
            )),
            Err(e) => self.set_status(format!("spawn failed: {e}")),
        }
        self.overlay = Overlay::None;
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
                    let template = App::template_for(recipe);
                    app.overlay = Overlay::Editor {
                        recipe: recipe.name,
                        buffer: template,
                    };
                }
            }
            _ => {}
        },
        Overlay::None => match k.code {
            KeyCode::Char('q') | KeyCode::Esc => app.quit = true,
            KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
            KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
            KeyCode::Enter => app.refresh_log(),
            KeyCode::Char('r') => {
                app.refresh_jobs();
                app.refresh_system();
                app.refresh_log();
                app.last_refresh = Instant::now();
                app.set_status("refreshed");
            }
            KeyCode::Char('c') => app.cancel_selected(),
            KeyCode::Char('R') => app.open_picker(),
            _ => {}
        },
    }
}

fn draw(f: &mut Frame<'_>, app: &mut App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(f.area());

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(outer[0]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(body[0]);

    draw_jobs(f, left[0], app);
    draw_log(f, left[1], app);
    draw_system(f, body[1], app);
    draw_status(f, outer[1], app);
    draw_overlay(f, app);
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

fn draw_status(f: &mut Frame<'_>, area: Rect, app: &App) {
    let base = "q quit  •  ↑↓ select  •  Enter refresh log  •  r refresh  •  c cancel  •  R run recipe  •  tick 1.5s";
    let line = if let Some((msg, _)) = &app.status_msg {
        format!("{msg}    │    {base}")
    } else {
        base.into()
    };
    let para = Paragraph::new(Span::styled(line, Style::default().fg(Color::DarkGray)));
    f.render_widget(para, area);
}
