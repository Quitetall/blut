// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Console DAG builder — compose a plan from a cookbook's ingredients.
//!
//! The BLUT console lets you assemble your own DAG from the detected
//! ingredients (registered stages) instead of running a pre-baked recipe: pick
//! ingredients into a canvas, wire them, and validate. "Validate" is the real
//! payoff — it builds a [`PlanSpec`](crate::framework::plan_spec::PlanSpec) and
//! runs it through `compile`, so every wiring break is a typed kind-check error
//! BEFORE anything executes (ADR 0078). A valid plan can be written out as a
//! `.json` PlanSpec — the same IR the Python SDK / `blut recipe declare` consume.
//!
//! This is pure state + paint + key handling; the registry lives on the `App`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::framework::Ingredient;
use crate::framework::plan_spec::{PLAN_SPEC_VERSION, PlanSpec, SpecNode};
use crate::tui::theme;

/// Which pane the builder's cursor is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Focus {
    #[default]
    Palette,
    Canvas,
}

/// The severity of the builder's one-line status message.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StatusKind {
    Ok,
    Err,
    Info,
}

/// A node placed on the canvas: the chosen ingredient + its kinds (copied so
/// the canvas renders without re-resolving the palette) + its args JSON
/// (prefilled from the stage schema, editable with `i`).
#[derive(Clone, Debug)]
pub struct BuilderNode {
    pub stage: String,
    pub input_kind: String,
    pub output_kind: String,
    pub args: serde_json::Value,
}

/// An in-progress per-node args edit: the target node index + the JSON buffer
/// being typed (compact single-line so `Enter` = save).
#[derive(Clone, Debug)]
pub struct ArgsEdit {
    pub node: usize,
    pub buffer: String,
}

/// The DAG-builder state: the plan being composed + the interaction cursors.
pub struct DagBuilder {
    pub name: String,
    pub nodes: Vec<BuilderNode>,
    /// `(from, to)` producer→consumer edges, indices into `nodes`.
    pub edges: Vec<(usize, usize)>,
    pub focus: Focus,
    pub palette_cursor: usize,
    pub node_cursor: usize,
    /// The pending edge source while wiring (`e` on a node, then `e` on another).
    pub connect_from: Option<usize>,
    /// `Some` while editing a node's args JSON (`i`); captures all keys.
    pub editing: Option<ArgsEdit>,
    pub status: Option<(String, StatusKind)>,
}

impl Default for DagBuilder {
    fn default() -> Self {
        Self {
            name: "custom-dag".into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            focus: Focus::Palette,
            palette_cursor: 0,
            node_cursor: 0,
            connect_from: None,
            editing: None,
            status: None,
        }
    }
}

impl DagBuilder {
    fn set(&mut self, msg: impl Into<String>, kind: StatusKind) {
        self.status = Some((msg.into(), kind));
    }

    /// Append the palette ingredient at `palette_cursor` as a canvas node, with
    /// its args prefilled from the stage's schema template (serde defaults +
    /// `<TODO>` placeholders for required fields) — the same start the recipe
    /// editor uses.
    fn add_from_palette(&mut self, palette: &[Ingredient], reg: &crate::framework::Registry) {
        let Some(ing) = palette.get(self.palette_cursor) else {
            return;
        };
        let args = reg
            .find_erased_stage(&ing.stage)
            .map(|ctor| crate::recipes::recipe::args_template_from_schema(&ctor().args_schema()))
            .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
        self.nodes.push(BuilderNode {
            stage: ing.stage.clone(),
            input_kind: ing.input_kind.clone(),
            output_kind: ing.output_kind.clone(),
            args,
        });
        self.node_cursor = self.nodes.len() - 1;
        self.set(
            format!("added '{}' (i to edit args)", ing.stage),
            StatusKind::Info,
        );
    }

    /// `i` on the canvas: open the args editor for the selected node, seeded with
    /// its current args as compact JSON.
    fn edit_args(&mut self) {
        let Some(n) = self.nodes.get(self.node_cursor) else {
            return;
        };
        let buffer = serde_json::to_string(&n.args).unwrap_or_else(|_| "{}".into());
        self.editing = Some(ArgsEdit {
            node: self.node_cursor,
            buffer,
        });
        self.set(
            "edit args (JSON) · Enter save · Esc cancel",
            StatusKind::Info,
        );
    }

    /// Key handling while the args editor is open: type into the buffer, Enter
    /// parses + saves it onto the node, Esc cancels.
    fn handle_edit_key(&mut self, code: crossterm::event::KeyCode) {
        use crossterm::event::KeyCode;
        let Some(edit) = self.editing.as_mut() else {
            return;
        };
        match code {
            KeyCode::Esc => {
                self.editing = None;
                self.set("edit cancelled", StatusKind::Info);
            }
            KeyCode::Backspace => {
                edit.buffer.pop();
            }
            KeyCode::Char(c) => edit.buffer.push(c),
            KeyCode::Enter => {
                let ArgsEdit { node, buffer } = self.editing.take().unwrap();
                match serde_json::from_str::<serde_json::Value>(&buffer) {
                    Ok(v) if v.is_object() => {
                        if let Some(n) = self.nodes.get_mut(node) {
                            n.args = v;
                        }
                        self.set("args updated", StatusKind::Ok);
                    }
                    Ok(_) => {
                        // Valid JSON but not an object — keep the buffer so the
                        // operator can fix it, same as the parse-error path.
                        self.editing = Some(ArgsEdit { node, buffer });
                        self.set("args must be a JSON object", StatusKind::Err);
                    }
                    Err(e) => {
                        // Re-open so the operator can fix the typo, not lose it.
                        self.editing = Some(ArgsEdit { node, buffer });
                        self.set(format!("invalid JSON: {e}"), StatusKind::Err);
                    }
                }
            }
            _ => {}
        }
    }

    /// `e` on the canvas: first press marks the edge source, second press adds
    /// the edge `source → node_cursor` (a self-loop or duplicate is refused).
    fn wire(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        match self.connect_from {
            None => {
                self.connect_from = Some(self.node_cursor);
                self.set("wire: pick the destination, then `e`", StatusKind::Info);
            }
            Some(from) => {
                let to = self.node_cursor;
                self.connect_from = None;
                if from == to {
                    self.set("wire: source and destination are the same", StatusKind::Err);
                } else if self.edges.contains(&(from, to)) {
                    self.set(format!("wire: {from}→{to} already exists"), StatusKind::Err);
                } else {
                    self.edges.push((from, to));
                    self.set(
                        format!(
                            "wired {from} ({}) → {to} ({}) — validate to kind-check",
                            self.nodes[from].output_kind, self.nodes[to].input_kind
                        ),
                        StatusKind::Info,
                    );
                }
            }
        }
    }

    /// `d` on the canvas: delete the selected node and every incident edge,
    /// re-indexing the survivors so the `PlanSpec` stays dense.
    fn delete_node(&mut self) {
        if self.node_cursor >= self.nodes.len() {
            return;
        }
        let victim = self.node_cursor;
        self.nodes.remove(victim);
        self.edges.retain(|&(a, b)| a != victim && b != victim);
        for (a, b) in self.edges.iter_mut() {
            if *a > victim {
                *a -= 1;
            }
            if *b > victim {
                *b -= 1;
            }
        }
        self.connect_from = None;
        self.node_cursor = self.node_cursor.min(self.nodes.len().saturating_sub(1));
        self.set("deleted node", StatusKind::Info);
    }

    /// Assemble the current canvas into a `PlanSpec` (empty args per node — the
    /// builder validates TOPOLOGY; per-node args are a run-time concern).
    fn to_plan_spec(&self) -> PlanSpec {
        PlanSpec {
            name: self.name.clone(),
            nodes: self
                .nodes
                .iter()
                .map(|n| SpecNode {
                    stage: n.stage.clone(),
                    args: n.args.clone(),
                })
                .collect(),
            edges: self
                .edges
                .iter()
                .map(|&(a, b)| (a as u32, b as u32))
                .collect(),
            expansions: Vec::new(),
            version: PLAN_SPEC_VERSION,
        }
    }

    /// `v`: the real kind-check — build the `PlanSpec` and `compile` it against
    /// the registry. Sets an Ok/Err status with the precise typed error.
    fn validate(&mut self, reg: &crate::framework::Registry) {
        if self.nodes.is_empty() {
            self.set(
                "nothing to validate — add ingredients first",
                StatusKind::Err,
            );
            return;
        }
        match self.to_plan_spec().compile(reg) {
            Ok(_) => self.set(
                format!(
                    "✓ valid — {} nodes, {} edges kind-check",
                    self.nodes.len(),
                    self.edges.len()
                ),
                StatusKind::Ok,
            ),
            Err(e) => self.set(e.to_string(), StatusKind::Err),
        }
    }

    /// `w`: write the plan as a `.json` PlanSpec under `<data_dir>/plans/` — the
    /// same IR `blut recipe declare <file>.json` consumes. Validates first.
    fn write(&mut self, reg: &crate::framework::Registry) {
        // Restrict the name to a safe filename charset — it's the only place the
        // (pub, editable) name touches the filesystem, so this fail-closed guard
        // closes any path-traversal vector (`../`, absolute paths, NUL).
        if self.name.is_empty()
            || !self
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            self.set(
                "plan name must be non-empty, ASCII alphanumeric / - / _ (no path separators)",
                StatusKind::Err,
            );
            return;
        }
        let spec = self.to_plan_spec();
        if spec.compile(reg).is_err() {
            self.set(
                "refusing to write an invalid plan — validate (v) first",
                StatusKind::Err,
            );
            return;
        }
        let dir = match crate::paths::data_dir() {
            Ok(d) => d.join("plans"),
            Err(e) => {
                self.set(format!("no data dir: {e}"), StatusKind::Err);
                return;
            }
        };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.set(format!("mkdir failed: {e}"), StatusKind::Err);
            return;
        }
        let path = dir.join(format!("{}.json", self.name));
        let json = serde_json::to_string_pretty(&spec).unwrap_or_default();
        match std::fs::write(&path, json) {
            Ok(()) => self.set(format!("wrote {}", path.display()), StatusKind::Ok),
            Err(e) => self.set(format!("write failed: {e}"), StatusKind::Err),
        }
    }

    fn move_cursor(&mut self, delta: isize, palette_len: usize) {
        let (cur, len) = match self.focus {
            Focus::Palette => (&mut self.palette_cursor, palette_len),
            Focus::Canvas => (&mut self.node_cursor, self.nodes.len()),
        };
        if len == 0 {
            *cur = 0;
            return;
        }
        let next = (*cur as isize + delta).clamp(0, len as isize - 1);
        *cur = next as usize;
    }

    /// Handle a key on the Build tab. Returns `true` if consumed (so global
    /// console keys — tab digits, `q` — still fire for keys the builder ignores).
    pub fn handle_key(
        &mut self,
        code: crossterm::event::KeyCode,
        palette: &[Ingredient],
        reg: &crate::framework::Registry,
    ) -> bool {
        use crossterm::event::KeyCode;
        // The args editor is a modal: while open it captures every key.
        if self.editing.is_some() {
            self.handle_edit_key(code);
            return true;
        }
        match code {
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Palette => Focus::Canvas,
                    Focus::Canvas => Focus::Palette,
                };
                true
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_cursor(-1, palette.len());
                true
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_cursor(1, palette.len());
                true
            }
            KeyCode::Enter | KeyCode::Char('a') => {
                if self.focus == Focus::Palette {
                    self.add_from_palette(palette, reg);
                } else {
                    self.focus = Focus::Palette;
                }
                true
            }
            KeyCode::Char('e') if self.focus == Focus::Canvas => {
                self.wire();
                true
            }
            KeyCode::Char('i') if self.focus == Focus::Canvas => {
                self.edit_args();
                true
            }
            KeyCode::Char('d') if self.focus == Focus::Canvas => {
                self.delete_node();
                true
            }
            KeyCode::Char('x') => {
                *self = DagBuilder {
                    name: std::mem::take(&mut self.name),
                    ..DagBuilder::default()
                };
                self.set("cleared", StatusKind::Info);
                true
            }
            KeyCode::Char('v') => {
                self.validate(reg);
                true
            }
            KeyCode::Char('w') => {
                self.write(reg);
                true
            }
            _ => false,
        }
    }
}

// ── rendering ─────────────────────────────────────────────────────────────

/// Render the Build surface: ingredient palette (left) + DAG canvas (right)
/// over a status/help line.
pub fn draw_build(f: &mut Frame<'_>, area: Rect, b: &DagBuilder, palette: &[Ingredient]) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(2)])
        .split(area);
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(rows[0]);

    draw_palette(f, cols[0], b, palette);
    draw_canvas(f, cols[1], b);
    draw_status(f, rows[1], b);
}

fn panel(title: String, active: bool) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(if active {
            theme::signal()
        } else {
            theme::panel_border()
        })
        .title(Span::styled(format!(" {title} "), theme::label()))
}

fn draw_palette(f: &mut Frame<'_>, area: Rect, b: &DagBuilder, palette: &[Ingredient]) {
    let active = b.focus == Focus::Palette;
    let arrow = if theme::ascii_only() { "->" } else { "→" };
    let inner_h = area.height.saturating_sub(2) as usize;
    let start = b.palette_cursor.saturating_sub(inner_h.saturating_sub(1));
    let lines: Vec<Line> = palette
        .iter()
        .enumerate()
        .skip(start)
        .take(inner_h)
        .map(|(i, ing)| {
            let sel = active && i == b.palette_cursor;
            let seed = ing.input_kind == "()";
            let name_style = if sel {
                theme::tab_active()
            } else if seed {
                theme::verified()
            } else {
                theme::signal()
            };
            let inb = if seed {
                "seed"
            } else {
                ing.input_kind.as_str()
            };
            Line::from(vec![
                Span::styled(if sel { "▸ " } else { "  " }, theme::signal_bold()),
                Span::styled(format!("{:<24}", ing.stage), name_style),
                Span::styled(
                    format!("{inb} {arrow} {}", ing.output_kind),
                    theme::panel_border(),
                ),
            ])
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(panel(format!("Ingredients · {}", palette.len()), active)),
        area,
    );
}

fn draw_canvas(f: &mut Frame<'_>, area: Rect, b: &DagBuilder) {
    let active = b.focus == Focus::Canvas;
    let arrow = if theme::ascii_only() { "->" } else { "→" };
    let mut lines: Vec<Line> = Vec::new();
    if b.nodes.is_empty() {
        lines.push(Line::from(Span::styled(
            " empty — Tab to the palette, ↑↓ pick, Enter to add an ingredient",
            theme::panel_border(),
        )));
    }
    for (i, n) in b.nodes.iter().enumerate() {
        let sel = active && i == b.node_cursor;
        let is_src = b.connect_from == Some(i);
        let marker = if is_src {
            Span::styled("◆ ", theme::amber())
        } else if sel {
            Span::styled("▸ ", theme::signal_bold())
        } else {
            Span::styled("  ", theme::panel_border())
        };
        // Predecessor indices feeding this node (edge order = tuple order).
        let preds: Vec<String> = b
            .edges
            .iter()
            .filter(|&&(_, to)| to == i)
            .map(|&(from, _)| from.to_string())
            .collect();
        let feed = if preds.is_empty() {
            String::new()
        } else {
            format!("  ← {}", preds.join(","))
        };
        let nargs = n.args.as_object().map(|o| o.len()).unwrap_or(0);
        let args_span = if nargs > 0 {
            Span::styled(format!("  {{{nargs}}}"), theme::verified())
        } else {
            Span::styled("  {}", theme::panel_border())
        };
        lines.push(Line::from(vec![
            marker,
            Span::styled(format!("{i} "), theme::panel_border()),
            Span::styled(
                format!("{:<22}", n.stage),
                if sel {
                    theme::tab_active()
                } else {
                    theme::signal()
                },
            ),
            Span::styled(
                format!("{} {arrow} {}", n.input_kind, n.output_kind),
                theme::panel_border(),
            ),
            args_span,
            Span::styled(feed, theme::amber()),
        ]));
    }
    let title = format!("DAG · {} · {}n {}e", b.name, b.nodes.len(), b.edges.len());
    f.render_widget(Paragraph::new(lines).block(panel(title, active)), area);
}

fn draw_status(f: &mut Frame<'_>, area: Rect, b: &DagBuilder) {
    // While editing a node's args, the status area becomes the JSON input line.
    if let Some(edit) = &b.editing {
        let stage = b
            .nodes
            .get(edit.node)
            .map(|n| n.stage.as_str())
            .unwrap_or("?");
        let prompt = Line::from(vec![
            Span::styled(format!(" args[{stage}] "), theme::signal_bold()),
            Span::styled(edit.buffer.clone(), theme::metric()),
            Span::styled("_", theme::signal()),
        ]);
        let hint = Line::from(Span::styled(
            " type JSON · Enter save · Esc cancel",
            theme::panel_border(),
        ));
        f.render_widget(Paragraph::new(vec![prompt, hint]), area);
        return;
    }
    let help = Line::from(Span::styled(
        " Tab pane · ↑↓ move · Enter/a add · e wire · i args · d delete · v validate · w write · x clear",
        theme::panel_border(),
    ));
    let status = match &b.status {
        None => Line::from(Span::styled(
            " compose a DAG from ingredients, then `v` to kind-check it",
            theme::label(),
        )),
        Some((msg, kind)) => {
            let style = match kind {
                StatusKind::Ok => theme::verified(),
                StatusKind::Err => theme::halt(),
                StatusKind::Info => theme::signal(),
            };
            Line::from(Span::styled(format!(" {msg}"), style))
        }
    };
    f.render_widget(Paragraph::new(vec![status, help]), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyCode;

    fn palette() -> Vec<Ingredient> {
        vec![
            Ingredient {
                stage: "seed".into(),
                input_kind: "()".into(),
                output_kind: "lma".into(),
                element_kind: None,
            },
            Ingredient {
                stage: "train".into(),
                input_kind: "lma".into(),
                output_kind: "ckpt".into(),
                element_kind: None,
            },
        ]
    }

    #[test]
    fn add_wire_delete_reindexes_edges() {
        let reg = crate::framework::Registry::new();
        let p = palette();
        let mut b = DagBuilder::default();
        // add two nodes
        b.handle_key(KeyCode::Enter, &p, &reg); // add seed (cursor 0)
        b.palette_cursor = 1;
        b.handle_key(KeyCode::Enter, &p, &reg); // add train
        assert_eq!(b.nodes.len(), 2);
        // wire 0 → 1 (focus canvas, source cursor 0, dest cursor 1)
        b.focus = Focus::Canvas;
        b.node_cursor = 0;
        b.handle_key(KeyCode::Char('e'), &p, &reg); // mark source 0
        b.node_cursor = 1;
        b.handle_key(KeyCode::Char('e'), &p, &reg); // edge 0→1
        assert_eq!(b.edges, vec![(0, 1)]);
        // delete node 0 → edge dropped, no dangling index
        b.node_cursor = 0;
        b.handle_key(KeyCode::Char('d'), &p, &reg);
        assert_eq!(b.nodes.len(), 1);
        assert!(b.edges.is_empty(), "incident edge removed");
    }

    #[test]
    fn self_loop_is_refused() {
        let reg = crate::framework::Registry::new();
        let p = palette();
        let mut b = DagBuilder::default();
        b.handle_key(KeyCode::Enter, &p, &reg);
        b.focus = Focus::Canvas;
        b.node_cursor = 0;
        b.handle_key(KeyCode::Char('e'), &p, &reg);
        b.handle_key(KeyCode::Char('e'), &p, &reg); // same node
        assert!(b.edges.is_empty());
        assert!(matches!(b.status, Some((_, StatusKind::Err))));
    }

    #[test]
    fn write_refuses_unsafe_names() {
        let reg = crate::framework::Registry::new();
        let mut b = DagBuilder {
            name: "../evil".into(),
            ..DagBuilder::default()
        };
        b.handle_key(KeyCode::Char('w'), &[], &reg);
        let (msg, kind) = b.status.as_ref().unwrap();
        assert_eq!(*kind, StatusKind::Err);
        assert!(
            msg.contains("name must"),
            "expected name-guard error, got: {msg}"
        );
    }

    #[test]
    fn edit_args_saves_valid_json_and_rejects_invalid() {
        let reg = crate::framework::Registry::new();
        let p = palette();
        let mut b = DagBuilder::default();
        b.handle_key(KeyCode::Enter, &p, &reg); // add node 0
        b.focus = Focus::Canvas;
        b.node_cursor = 0;
        b.handle_key(KeyCode::Char('i'), &p, &reg);
        assert!(b.editing.is_some(), "i opens the args editor");
        b.editing.as_mut().unwrap().buffer = "{\"lr\":0.01}".into();
        b.handle_key(KeyCode::Enter, &p, &reg);
        assert!(b.editing.is_none(), "Enter saves + closes");
        assert_eq!(b.nodes[0].args["lr"], serde_json::json!(0.01));
        assert_eq!(
            b.to_plan_spec().nodes[0].args["lr"],
            serde_json::json!(0.01)
        );
        // Invalid JSON keeps the editor open so the typo can be fixed.
        b.handle_key(KeyCode::Char('i'), &p, &reg);
        b.editing.as_mut().unwrap().buffer = "{bad".into();
        b.handle_key(KeyCode::Enter, &p, &reg);
        assert!(b.editing.is_some());
        assert!(matches!(b.status, Some((_, StatusKind::Err))));
        // Valid JSON that isn't an object also keeps the editor open (buffer
        // preserved), not silently discarded.
        b.editing.as_mut().unwrap().buffer = "42".into();
        b.handle_key(KeyCode::Enter, &p, &reg);
        assert!(b.editing.is_some(), "non-object JSON keeps the buffer");
        assert_eq!(b.editing.as_ref().unwrap().buffer, "42");
    }

    #[test]
    fn validate_empty_is_an_error() {
        let reg = crate::framework::Registry::new();
        let mut b = DagBuilder::default();
        b.handle_key(KeyCode::Char('v'), &[], &reg);
        assert!(matches!(b.status, Some((_, StatusKind::Err))));
    }

    #[test]
    fn to_plan_spec_is_dense_and_versioned() {
        let p = palette();
        let reg = crate::framework::Registry::new();
        let mut b = DagBuilder::default();
        b.handle_key(KeyCode::Enter, &p, &reg);
        let spec = b.to_plan_spec();
        assert_eq!(spec.nodes.len(), 1);
        assert_eq!(spec.version, PLAN_SPEC_VERSION);
    }
}
