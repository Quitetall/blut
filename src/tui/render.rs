// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Rendering layer for the TUI cockpit — the `draw` dispatcher, every
//! per-view `draw_*` painter, the overlay painter, and the small pure
//! layout helpers they share (`section_header`, `centered_rect`,
//! `view_header`, `node_status_style`, `truncate`). Pure paint: state
//! lives in [`super::App`]; input handling stays in `tui/mod.rs`.
//!
//! Mounted as a child of the `tui` module (`use super::*`), so painters
//! read the App state and module-level imports exactly as they did when
//! they lived inline; `render_tests` (the headless TestBackend suite)
//! moves with them.

use super::*;

/// Top-level frame dispatcher. Splits body + 1-row footer, routes the
/// body to the active view's drawer, then renders the footer + any
/// modal overlay on top.
pub(super) fn draw(f: &mut Frame<'_>, app: &mut App) {
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
        View::Dag => draw_dag(f, body, app),
        View::Lineage => draw_lineage(f, body, app),
        View::Artifacts => draw_artifacts(f, body, app),
        View::Metrics => draw_metrics(f, body, app),
        View::Catalog => draw_catalog(f, body, app),
        View::Reset => draw_reset(f, body, app),
    }
    draw_status(f, outer[1], app);
    draw_overlay(f, app);
}

/// Section header for the cockpit menu — a dim-indented heading in the
/// a dim-indented heading in the `theme::highlight` style.
pub(super) fn section_header(title: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(title.to_string(), theme::highlight()),
    ])
}

/// One `[key] Label   description` menu row:
/// the key hint in `theme::key_hint`, the label in `theme::normal`, the
/// description in `theme::dim`. `key` is `None` for items reachable only
/// via the `R` recipe picker (shown as `[-]`).
pub(super) fn opt_row(key: Option<char>, label: &str, desc: &str) -> Line<'static> {
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
pub(super) fn menu_desc(desc: &str) -> String {
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
pub(super) fn draw_cockpit_body(f: &mut Frame<'_>, area: Rect, app: &mut App) {
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
    // Sections, top to bottom:
    //   the recipe courses (data-driven off the registered cookbooks'
    //   `Course`s — DataPrep / Train / Eval / Export / …) / RUNS (the
    //   analytic Views) / PROVENANCE (DAG + lineage) /
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

    // RUNS — the jobs store + run-analytic views (all read the engine's own
    // state: the jobs store + the lineage DB).
    lines.push(section_header("RUNS"));
    lines.push(Line::from(""));
    for (key, label, desc) in [
        ('J', "Jobs", "all jobs, color-coded by state"),
        ('L', "Log", "status.jsonl tail of selected job"),
        ('Y', "System", "full GPU / MEM / DISK / CPU probe"),
        ('H', "Run history", "all jobs ⋈ recipe / outcome / metric"),
        ('B', "Leaderboard", "runs ranked by the active metric"),
        ('C', "Compare", "marked runs side by side"),
        ('M', "Metrics", "the selected run's final metrics"),
    ] {
        lines.push(opt_row(Some(key), label, desc));
    }
    lines.push(Line::from(""));

    // PROVENANCE — the plan graph + lineage / artifacts for a selected run.
    lines.push(section_header("PROVENANCE"));
    lines.push(Line::from(""));
    for (key, label, desc) in [
        ('G', "DAG", "the run's plan graph + per-node status"),
        ('I', "Lineage", "ingredient hashes · cache · code freshness"),
        ('A', "Artifacts", "content-addressed outputs of a run"),
        ('P', "Catalog", "registered recipes by course + args"),
    ] {
        lines.push(opt_row(Some(key), label, desc));
    }
    lines.push(Line::from(""));

    // SYSTEM — built-in actions + generic destructive maintenance.
    lines.push(section_header("SYSTEM"));
    lines.push(Line::from(""));
    lines.push(opt_row(
        Some('X'),
        "Maintenance",
        "prune cache · clear job · forget footprints",
    ));
    for (k, _, label) in App::builtin_keys() {
        lines.push(opt_row(Some(*k), label, ""));
    }

    let menu_para = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(menu_para, chunks[3]);
}

pub(super) fn centered_rect(area: Rect, pct_w: u16, pct_h: u16) -> Rect {
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

pub(super) fn draw_overlay(f: &mut Frame<'_>, app: &App) {
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
                let text: Vec<Line> = raw_buffer
                    .lines()
                    .map(|l| Line::from(l.to_string()))
                    .collect();
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

pub(super) fn draw_jobs(f: &mut Frame<'_>, area: Rect, app: &mut App) {
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

pub(super) fn draw_log(f: &mut Frame<'_>, area: Rect, app: &App) {
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

pub(super) fn draw_system(f: &mut Frame<'_>, area: Rect, app: &App) {
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
        Line::from(Span::styled(
            format!("DISK {}", snap.disk_path),
            theme::highlight(),
        )),
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
pub(super) fn view_header(title: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(title.to_string(), theme::title()),
        Span::styled(
            format!("    blut v{}", env!("CARGO_PKG_VERSION")),
            theme::dim(),
        ),
    ])
}

/// Run History view: every job (newest first) with its recipe, outcome, and
/// the active metric — sourced from the jobs store joined with the lineage DB.
pub(super) fn draw_history(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · m mark · Enter DAG · b back) ",
                View::History.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Run History"), Line::from("")];
    if app.runs.is_empty() {
        lines.push(Line::from(Span::styled(
            "No runs yet — launch a recipe (R) to populate the run history.",
            theme::dim(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<24} {:<19} {:<10} {:<10} {}",
                "Recipe",
                "Job",
                "Outcome",
                views::DEFAULT_METRIC,
                "When"
            ),
            theme::dim(),
        )));
        for (i, r) in app.runs.iter().enumerate() {
            let cursor = i == app.list_cursor;
            let marked = app.marked.contains(&r.job_id);
            let prefix = if cursor { "▶" } else { " " };
            let mark = if marked { "✓" } else { " " };
            let metric = r
                .metric
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "—".into());
            let style = if cursor {
                theme::selected()
            } else {
                theme::normal()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{prefix}{mark} "), theme::success()),
                Span::styled(
                    format!(
                        "{:<24} {:<19} {:<10} {:<10} ",
                        truncate(&r.recipe, 24),
                        truncate(&r.job_id, 19),
                        truncate(&r.outcome, 10),
                        metric
                    ),
                    style,
                ),
                Span::styled(r.when.clone(), theme::dim()),
            ]));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Leaderboard view: runs ranked by the active metric (lineage DB
/// `top_runs_by_metric`); rank 1 is highlighted.
pub(super) fn draw_leaderboard(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · m mark · Enter DAG · b back) ",
                View::Leaderboard.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![
        view_header("Leaderboard"),
        Line::from(Span::styled(
            format!("ranked by {} (lower is better)", views::DEFAULT_METRIC),
            theme::dim(),
        )),
        Line::from(""),
    ];
    if app.runs.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(
                "No runs have recorded a `{}` metric yet.",
                views::DEFAULT_METRIC
            ),
            theme::dim(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<5} {:<24} {:<19} {:<10} {}",
                "Rank",
                "Recipe",
                "Job",
                views::DEFAULT_METRIC,
                "When"
            ),
            theme::dim(),
        )));
        for (i, r) in app.runs.iter().enumerate().take(views::LEADERBOARD_LIMIT) {
            let cursor = i == app.list_cursor;
            let marked = app.marked.contains(&r.job_id);
            let medal = if i == 0 { " ▸" } else { "" };
            let metric = r
                .metric
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "—".into());
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
                        "{mark} {:<5} {:<24} {:<19} {:<10} ",
                        i + 1,
                        truncate(&r.recipe, 24),
                        truncate(&r.job_id, 19),
                        metric
                    ),
                    style,
                ),
                Span::styled(format!("{}{medal}", r.when), theme::dim()),
            ]));
        }
        if app.runs.len() > views::LEADERBOARD_LIMIT {
            lines.push(Line::from(Span::styled(
                format!(
                    "  ... {} more runs",
                    app.runs.len() - views::LEADERBOARD_LIMIT
                ),
                theme::dim(),
            )));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Compare view: the marked runs side by side — recipe, each shared final
/// metric, and GPU saturation. Mark runs with `m` in History / Leaderboard.
pub(super) fn draw_compare(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (mark runs with m in History/Leaderboard · b back) ",
                View::Compare.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Compare Runs"), Line::from("")];
    // Cached in `App` (loaded by `load_view_data` on entry / refresh) so the
    // drawer doesn't reopen the lineage DB every frame.
    let cols = &app.compare;
    if cols.len() < 2 {
        lines.push(Line::from(Span::styled(
            "Mark at least 2 runs (press m on a row in History/Leaderboard) to compare.",
            theme::dim(),
        )));
    } else {
        // Header: a metric-name column + one column per marked run.
        let mut hdr = vec![Span::styled(format!("  {:<20}", "Metric"), theme::dim())];
        let mut ids = vec![Span::styled(format!("  {:<20}", ""), theme::dim())];
        for c in cols {
            hdr.push(Span::styled(
                format!("{:<20}", truncate(&c.recipe, 19)),
                theme::heading(),
            ));
            ids.push(Span::styled(
                format!("{:<20}", truncate(&c.job_id, 19)),
                theme::dim(),
            ));
        }
        lines.push(Line::from(hdr));
        lines.push(Line::from(ids));
        lines.push(Line::from(""));
        // The union of metric names across the marked runs, sorted.
        let mut names: Vec<String> = Vec::new();
        for c in cols {
            for (n, _) in &c.metrics {
                if !names.contains(n) {
                    names.push(n.clone());
                }
            }
        }
        names.sort();
        for n in &names {
            let mut row = vec![Span::styled(
                format!("  {:<20}", truncate(n, 19)),
                theme::heading(),
            )];
            for c in cols {
                let v = c.metrics.iter().find(|(m, _)| m == n).map(|(_, v)| *v);
                let txt = v.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into());
                row.push(Span::styled(format!("{txt:<20}"), theme::normal()));
            }
            lines.push(Line::from(row));
        }
        // GPU saturation row.
        let mut g = vec![Span::styled(
            format!("  {:<20}", "gpu_saturation%"),
            theme::heading(),
        )];
        for c in cols {
            let txt = c
                .gpu
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "—".into());
            g.push(Span::styled(format!("{txt:<20}"), theme::normal()));
        }
        lines.push(Line::from(g));
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Style a node-status tag by outcome category.
pub(super) fn node_status_style(s: crate::framework::NodeStatus) -> ratatui::style::Style {
    use crate::framework::NodeStatus as N;
    match s {
        N::Done | N::Skipped => theme::success(),
        N::Running => theme::key_hint(),
        N::Failed | N::Killed | N::Pruned => theme::error(),
        N::Blocked => theme::warning(),
        N::Pending | N::Ready => theme::dim(),
    }
}

/// DAG view: the selected run's plan graph — each node's derived status,
/// elapsed time, and cache state, plus the edge list. Built from the engine's
/// `graph_snapshot` (the same source as `blut dag`).
pub(super) fn draw_dag(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(" {} (r refresh · b back) ", View::Dag.title()),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Run DAG"), Line::from("")];
    match &app.dag {
        None => lines.push(Line::from(Span::styled(
            "No plan graph for the selected run — pick a run in History (Enter), or none has run yet.",
            theme::dim(),
        ))),
        Some(g) => {
            lines.push(Line::from(vec![
                Span::styled(format!("{}  ", g.name), theme::heading()),
                Span::styled(g.job.clone(), theme::dim()),
            ]));
            lines.push(Line::from(Span::styled(
                format!("{} ingredient(s) · {} edge(s)", g.nodes.len(), g.edges.len()),
                theme::dim(),
            )));
            lines.push(Line::from(""));
            for n in &g.nodes {
                let elapsed = n
                    .elapsed_secs
                    .map(|s| format!("{s:.1}s"))
                    .unwrap_or_else(|| "—".into());
                let cache = if n.cache_hit { "  (cache hit)" } else { "" };
                lines.push(Line::from(vec![
                    Span::styled(format!("  [{:>7}] ", n.status.as_str()), node_status_style(n.status)),
                    Span::styled(format!("{:>2} ", n.idx), theme::dim()),
                    Span::styled(
                        format!("{:<28} ", truncate(&n.stage_name, 28)),
                        theme::normal(),
                    ),
                    Span::styled(format!("{elapsed}{cache}"), theme::dim()),
                ]));
            }
            if !g.edges.is_empty() {
                lines.push(Line::from(""));
                let edge_str = g
                    .edges
                    .iter()
                    .map(|e| format!("{}→{}", e.from, e.to))
                    .collect::<Vec<_>>()
                    .join("  ");
                lines.push(Line::from(vec![
                    Span::styled("  edges: ", theme::dim()),
                    Span::styled(edge_str, theme::dim()),
                ]));
            }
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Lineage view: the selected run's per-stage provenance (input → output
/// content hashes, cache state, timing), its cache hit/miss totals, and the
/// code-freshness verdict (did the building code drift from HEAD?).
pub(super) fn draw_lineage(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(" {} (r refresh · b back) ", View::Lineage.title()),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let lv = &app.lineage;
    let mut lines: Vec<Line> = vec![view_header("Lineage & Provenance"), Line::from("")];
    if lv.rows.is_empty() {
        lines.push(Line::from(Span::styled(
            "No lineage for the selected run — pick a run in History (Enter), or none has run yet.",
            theme::dim(),
        )));
    } else {
        let fresh_style = match lv.freshness.as_str() {
            "STALE" => theme::error(),
            "FRESH" => theme::success(),
            _ => theme::dim(),
        };
        lines.push(Line::from(vec![
            Span::styled("code: ", theme::dim()),
            Span::styled(lv.freshness.clone(), fresh_style),
            Span::styled(
                format!(
                    "     cache: {} hit / {} miss",
                    lv.cache_hits, lv.cache_misses
                ),
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<3} {:<24} {:<12} {:<12} {:<7} {}",
                "#", "Ingredient", "Input", "Output", "Cached", "Elapsed"
            ),
            theme::dim(),
        )));
        for r in &lv.rows {
            let cached = if r.cached { "yes" } else { "no" };
            lines.push(Line::from(Span::styled(
                format!(
                    "  {:<3} {:<24} {:<12} {:<12} {:<7} {}",
                    r.node_idx,
                    truncate(&r.stage, 24),
                    r.input,
                    r.output,
                    cached,
                    r.elapsed
                ),
                theme::normal(),
            )));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Artifacts view: the content-addressed outputs the selected run
/// materialized (kind, producing stage, hash, time), read from its stage
/// sidecars.
pub(super) fn draw_artifacts(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · r refresh · b back) ",
                View::Artifacts.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Artifacts"), Line::from("")];
    if app.artifacts.is_empty() {
        lines.push(Line::from(Span::styled(
            "No artifacts for the selected run — pick a run in History (Enter), or none produced output.",
            theme::dim(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "  {:<24} {:<24} {:<14} {}",
                "Kind", "Ingredient", "Hash", "When"
            ),
            theme::dim(),
        )));
        for (i, a) in app.artifacts.iter().enumerate() {
            let cursor = i == app.list_cursor;
            let prefix = if cursor { "▶ " } else { "  " };
            let style = if cursor {
                theme::selected()
            } else {
                theme::normal()
            };
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), theme::success()),
                Span::styled(
                    format!(
                        "{:<24} {:<24} ",
                        truncate(&a.kind, 24),
                        truncate(&a.stage, 24)
                    ),
                    style,
                ),
                Span::styled(format!("{:<14} ", a.hash), theme::key_hint()),
                Span::styled(a.when.clone(), theme::dim()),
            ]));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Recipe Catalog view: every recipe the loaded cookbooks registered (name,
/// course, description) with the selected recipe's backend, I/O kinds,
/// schedule, and args schema expanded below. Driven entirely by the live
/// registry — no domain knowledge, no stale reference tables.
pub(super) fn draw_catalog(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(" {} (↑↓ move · b back) ", View::Catalog.title()),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Recipe Catalog"), Line::from("")];
    if app.catalog.is_empty() {
        lines.push(Line::from(Span::styled(
            "No recipes registered — load a cookbook into the binary's Registry.",
            theme::dim(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!("  {:<22} {:<11} {}", "Recipe", "Course", "Description"),
            theme::dim(),
        )));
        for (i, r) in app.catalog.iter().enumerate() {
            let cursor = i == app.list_cursor;
            let prefix = if cursor { "▶ " } else { "  " };
            let style = if cursor {
                theme::selected()
            } else {
                theme::normal()
            };
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), theme::success()),
                Span::styled(format!("{:<22} ", truncate(r.name, 22)), style),
                Span::styled(format!("{:<11} ", r.category.label()), theme::key_hint()),
                Span::styled(truncate(r.description, 40), theme::dim()),
            ]));
        }
        // Selected-recipe detail card.
        if let Some(r) = app.catalog.get(app.list_cursor) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("▸ {}", r.name),
                theme::title(),
            )));
            let kinds_in = if r.input_kinds.is_empty() {
                "(none — graph input)".to_string()
            } else {
                r.input_kinds.join(", ")
            };
            lines.push(Line::from(vec![
                Span::styled("  backend ", theme::dim()),
                Span::styled(r.backend_id.to_string(), theme::normal()),
                Span::styled("    in ", theme::dim()),
                Span::styled(kinds_in, theme::normal()),
                Span::styled("  →  out ", theme::dim()),
                Span::styled(r.output_kind.to_string(), theme::normal()),
            ]));
            if let Some(cal) = r.schedule {
                lines.push(Line::from(vec![
                    Span::styled("  schedule ", theme::dim()),
                    Span::styled(cal.to_string(), theme::normal()),
                ]));
            }
            let schema = (r.args_schema_fn)();
            let props = schema_prop_names(&schema);
            lines.push(Line::from(Span::styled(
                if props.is_empty() {
                    "  args: (none)".to_string()
                } else {
                    "  args:".to_string()
                },
                theme::dim(),
            )));
            for p in &props {
                let ty = schema_field_type(&schema, p);
                lines.push(Line::from(vec![
                    Span::styled(format!("    {:<24} ", truncate(p, 24)), theme::heading()),
                    Span::styled(ty, theme::dim()),
                ]));
            }
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Run Metrics view: the selected run's final metric values (name → value),
/// read from the lineage DB.
pub(super) fn draw_metrics(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(" {} (r refresh · b back) ", View::Metrics.title()),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Run Metrics"), Line::from("")];
    if app.metrics.is_empty() {
        lines.push(Line::from(Span::styled(
            "No metrics for the selected run — pick a run in History (Enter), or none has logged metrics yet.",
            theme::dim(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!("  {:<32} {}", "Metric", "Value"),
            theme::dim(),
        )));
        for (name, value) in &app.metrics {
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<32} ", truncate(name, 32)), theme::heading()),
                Span::styled(format!("{value:.6}"), theme::normal()),
            ]));
        }
    }
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Maintenance view: generic, domain-agnostic destructive actions on the
/// engine's own state — prune the cache, delete the selected job's dir,
/// forget footprint calibrations. Each needs a second Enter to confirm.
pub(super) fn draw_reset(f: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            format!(
                " {} (↑↓ move · Enter confirm (2×) · b back) ",
                View::Reset.title()
            ),
            theme::title(),
        ))
        .border_style(theme::dim())
        .borders(Borders::ALL);
    let mut lines: Vec<Line> = vec![view_header("Maintenance"), Line::from("")];
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
    // "Delete the selected job's directory" acts on the current run — show it.
    lines.push(Line::from(""));
    let target = app
        .current_job_id()
        .unwrap_or_else(|| "(none selected)".into());
    lines.push(Line::from(vec![
        Span::styled("  selected job (for clear): ", theme::dim()),
        Span::styled(target, theme::normal()),
    ]));
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Truncate a string to `max` chars with an ellipsis if needed.
pub(super) fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

pub(super) fn draw_status(f: &mut Frame<'_>, area: Rect, app: &App) {
    let base = match app.view {
        View::Cockpit => {
            "q quit • ↑↓ select • Enter log • r refresh • c cancel • R recipe • J/L/Y/H/B/C/G/I/A/M/P/X views"
        }
        View::Jobs => "↑↓ select • Enter log • c cancel • r refresh • b back • q quit",
        View::Log => "↑↓ select job • c cancel • r refresh • b back • q quit",
        View::System => "r refresh • b back • q quit",
        View::History | View::Leaderboard => {
            "↑↓ move • m mark • Enter DAG • r refresh • b back • q quit"
        }
        View::Compare => "mark runs with m in History/Leaderboard • b back • q quit",
        View::Dag | View::Lineage | View::Metrics => "r refresh • b back • q quit",
        View::Artifacts => "↑↓ move • r refresh • b back • q quit",
        View::Catalog => "↑↓ move • b back • q quit",
        View::Reset => "↑↓ move • Enter confirm (2×) • b back • q quit",
    };
    let para = if let Some((msg, _)) = &app.status_msg {
        // Status message segment in the success-tinted bar style, the
        // key-hint base in the standard status-bar style.
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
        // The analytic views read the global lineage DB / jobs store lazily
        // (only when `set_view`/refresh runs); the render tests set `app.view`
        // directly, so the per-job caches stay at their empty `new()` defaults
        // and the drawers render their deterministic empty states.
        App::new(test_registry())
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
        let long = "A deliberately long recipe description that runs well past the \
             single-row clip width so the squash-and-ellipsis path is exercised.";
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
        // recipe-course headings are data-driven off the injected catalog,
        // so they depend on the test registry and are asserted in the
        // cookbook crates' TUI tests, not here.
        for heading in ["RUNS", "PROVENANCE", "SYSTEM"] {
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
            View::Dag,
            View::Lineage,
            View::Artifacts,
            View::Metrics,
            View::Catalog,
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
                View::Dag => "Run DAG",
                View::Lineage => "Lineage",
                View::Artifacts => "Artifacts",
                View::Metrics => "Run Metrics",
                View::Catalog => "Recipe Catalog",
                View::Reset => "Maintenance",
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
        super::check(test_registry())
            .expect("tui --check must render all views + overlays and exit Ok");
    }
}
