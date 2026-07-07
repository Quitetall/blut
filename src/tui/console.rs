// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT engine console — the home dashboard.
//!
//! This is the redesigned TUI surface: not a training cockpit (that moved to
//! the cookbook), but mission control for the BLUT ENGINE — the symmetric mesh,
//! the typed DAG executing, the never-OOM broker, the DP privacy ledger, the
//! content-addressed cache, and the fail-closed clinical governance.
//!
//! [`ConsoleModel`] is the data; [`draw_console`] renders the at-a-glance home:
//! a status strip (the six numbers that matter), the plan pipeline, and a row of
//! instruments, over a governance footer. The model is populated by loaders from
//! the artifacts the engine already writes (`status.jsonl`, the peer registry,
//! the privacy ledger, the broker footprint) — [`ConsoleModel::demo`] stands in
//! when there is no live run.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::tui::theme;

/// Coarse plan lifecycle phase. The full vocabulary is matched by the renderer;
/// the demo model only exercises `Running` — live loaders (a completed / failed
/// run) construct the rest.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Pending,
    Running,
    Succeeded,
    Failed,
}

/// A peer's local trust level (ADR 0079).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    Anonymous,
    Registered,
    Trusted,
}

/// A node in the executing typed DAG. `Blocked` (a fail-closed gate refusal) is
/// rendered but not present in the demo — a live gate denial constructs it.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Cached,
    Done,
    Running,
    Queued,
    Blocked,
}

pub struct MeshPeer {
    pub id: String,
    pub trust: Trust,
    pub caps: String,
    pub reputation: f64,
    pub is_self: bool,
}

pub struct DagNode {
    pub name: String,
    pub state: NodeState,
    pub note: String,
}

pub struct BrokerState {
    pub held_gib: u32,
    pub budget_gib: u32,
    pub top_stage: String,
    pub headroom_gib: u32,
}

pub struct CorpusBudget {
    pub name: String,
    pub eps_spent: f64,
    pub eps_budget: f64,
    /// `true` ⇒ Restricted/clinical: local-only, gradients never dispatch.
    pub restricted: bool,
}

pub struct CacheEvent {
    pub hit: bool,
    pub stage: String,
    pub hash: String,
    pub note: String,
}

/// The whole console state.
pub struct ConsoleModel {
    pub run_id: String,
    pub plan: String,
    pub phase: Phase,
    pub elapsed: String,
    pub stages_done: u32,
    pub stages_total: u32,
    pub mesh: Vec<MeshPeer>,
    pub dag: Vec<DagNode>,
    pub broker: BrokerState,
    pub corpora: Vec<CorpusBudget>,
    pub cache_hit_pct: u32,
    pub cache_mibs: u32,
    pub cache: Vec<CacheEvent>,
    pub gov_violations: u32,
}

impl ConsoleModel {
    /// A representative model for when there is no live run yet (or `--demo`).
    /// Uses real BLUT stage names + concepts so the surface reads true.
    pub fn demo() -> Self {
        Self {
            run_id: "7e72347a".into(),
            plan: "train_joint · tier7".into(),
            phase: Phase::Running,
            elapsed: "04:12".into(),
            stages_done: 7,
            stages_total: 11,
            mesh: vec![
                MeshPeer {
                    id: "node·a1b2f9".into(),
                    trust: Trust::Trusted,
                    caps: "24c · 62G · RTX4090 · sched+worker".into(),
                    reputation: 0.98,
                    is_self: true,
                },
                MeshPeer {
                    id: "node·c3d4e7".into(),
                    trust: Trust::Registered,
                    caps: "16c · 32G · A100 · worker".into(),
                    reputation: 0.91,
                    is_self: false,
                },
                MeshPeer {
                    id: "node·e5f6a2".into(),
                    trust: Trust::Anonymous,
                    caps: "8c · 16G · cpu-only".into(),
                    reputation: 0.50,
                    is_self: false,
                },
                MeshPeer {
                    id: "pool·gpu×3".into(),
                    trust: Trust::Registered,
                    caps: "k8s BlutWorkerPool".into(),
                    reputation: 0.88,
                    is_self: false,
                },
            ],
            dag: vec![
                DagNode {
                    name: "codec_ready".into(),
                    state: NodeState::Cached,
                    note: "0cc92f50 · reused".into(),
                },
                DagNode {
                    name: "fed-shard".into(),
                    state: NodeState::Done,
                    note: "ab5c8b3f · 4 parts".into(),
                },
                DagNode {
                    name: "fed-local-train ×4".into(),
                    state: NodeState::Running,
                    note: "map fan-out · 3 done · σ 1.1".into(),
                },
                DagNode {
                    name: "fed-aggregate".into(),
                    state: NodeState::Queued,
                    note: "on initiator · min 2".into(),
                },
                DagNode {
                    name: "eval_codec_pccp".into(),
                    state: NodeState::Queued,
                    note: "gate: R ≥ 0.85".into(),
                },
            ],
            broker: BrokerState {
                held_gib: 35,
                budget_gib: 46,
                top_stage: "train_joint (SOAP)".into(),
                headroom_gib: 11,
            },
            corpora: vec![
                CorpusBudget {
                    name: "tuh-eeg · public".into(),
                    eps_spent: 5.8,
                    eps_budget: 10.0,
                    restricted: false,
                },
                CorpusBudget {
                    name: "internal-decoder".into(),
                    eps_spent: 3.9,
                    eps_budget: 5.0,
                    restricted: false,
                },
                CorpusBudget {
                    name: "clinical-tuab".into(),
                    eps_spent: 0.0,
                    eps_budget: 0.0,
                    restricted: true,
                },
            ],
            cache_hit_pct: 73,
            cache_mibs: 412,
            cache: vec![
                CacheEvent {
                    hit: true,
                    stage: "materialize_windows".into(),
                    hash: "0cc92f50".into(),
                    note: "reused".into(),
                },
                CacheEvent {
                    hit: true,
                    stage: "snn_controller".into(),
                    hash: "ab5c8b3f".into(),
                    note: "reused".into(),
                },
                CacheEvent {
                    hit: false,
                    stage: "train_joint".into(),
                    hash: "7e72347a".into(),
                    note: "ran 4.2s".into(),
                },
                CacheEvent {
                    hit: false,
                    stage: "fed-local-train[2]".into(),
                    hash: "4bf74e0c".into(),
                    note: "412 MiB/s".into(),
                },
            ],
            gov_violations: 0,
        }
    }
}

// ── rendering helpers ─────────────────────────────────────────────────────

fn glyphs() -> (&'static str, &'static str, &'static str) {
    // (filled bar, empty bar, arrow) — ASCII fallback for dumb terminals.
    if theme::ascii_only() {
        ("#", "-", "->")
    } else {
        ("█", "░", "→")
    }
}

/// A colored proportion bar of `width` cells.
fn bar(frac: f64, width: usize, style: ratatui::style::Style) -> Vec<Span<'static>> {
    let (fill, empty, _) = glyphs();
    let f = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    vec![
        Span::styled(fill.repeat(f), style),
        Span::styled(empty.repeat(width.saturating_sub(f)), theme::panel_border()),
    ]
}

fn panel(title: &str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(theme::panel_border())
        .title(Span::styled(format!(" {title} "), theme::label()))
}

fn node_marker(state: NodeState) -> Span<'static> {
    let (g, st) = match state {
        NodeState::Cached => ("✓", theme::verified()),
        NodeState::Done => ("•", theme::label()),
        NodeState::Running => ("⟳", theme::signal_bold()),
        NodeState::Queued => ("○", theme::panel_border()),
        NodeState::Blocked => ("⨯", theme::halt()),
    };
    let g = if theme::ascii_only() {
        match state {
            NodeState::Cached => "x",
            NodeState::Done => "*",
            NodeState::Running => "~",
            NodeState::Queued => "o",
            NodeState::Blocked => "!",
        }
    } else {
        g
    };
    Span::styled(g, st)
}

// ── the home dashboard ────────────────────────────────────────────────────

/// Render the console home: status strip, plan pipeline, instrument row, gov.
pub fn draw_console(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // status strip
            Constraint::Min(9),    // plan pipeline
            Constraint::Min(8),    // instruments
            Constraint::Length(1), // governance footer
        ])
        .split(area);

    draw_strip(f, rows[0], m);
    draw_plan(f, rows[1], m);
    draw_instruments(f, rows[2], m);
    draw_gov(f, rows[3], m);
}

fn draw_strip(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let cells = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 6); 6])
        .split(area);

    let (phase_txt, phase_style) = match m.phase {
        Phase::Running => ("Running", theme::signal_bold()),
        Phase::Succeeded => ("Succeeded", theme::verified()),
        Phase::Failed => ("Failed", theme::halt()),
        Phase::Pending => ("Pending", theme::label()),
    };
    let ready = m
        .mesh
        .iter()
        .filter(|p| p.trust != Trust::Anonymous)
        .count();
    let (eps_spent, eps_budget) = m
        .corpora
        .iter()
        .filter(|c| !c.restricted)
        .fold((0.0, 0.0), |(s, b), c| (s + c.eps_spent, b + c.eps_budget));
    let eps_rem = (eps_budget - eps_spent).max(0.0);

    let cell = |label: &str, value: Vec<Span<'static>>, sub: &str| {
        Paragraph::new(vec![
            Line::from(Span::styled(label.to_string(), theme::label())),
            Line::from(value),
            Line::from(Span::styled(sub.to_string(), theme::panel_border())),
        ])
        .block(
            Block::default()
                .borders(Borders::RIGHT)
                .border_style(theme::panel_border()),
        )
    };

    f.render_widget(
        cell(
            "PHASE",
            vec![Span::styled(phase_txt, phase_style)],
            &format!("{}/{} stages", m.stages_done, m.stages_total),
        ),
        cells[0],
    );
    f.render_widget(
        cell(
            "MESH",
            vec![
                Span::styled(m.mesh.len().to_string(), theme::metric()),
                Span::styled(" nodes", theme::label()),
            ],
            &format!("{ready} ready"),
        ),
        cells[1],
    );
    f.render_widget(
        cell(
            "PRIVACY ε",
            vec![
                Span::styled(format!("{eps_rem:.1}"), theme::metric()),
                Span::styled(format!(" / {eps_budget:.0}"), theme::label()),
            ],
            "remaining",
        ),
        cells[2],
    );
    f.render_widget(
        cell(
            "CACHE",
            vec![
                Span::styled(format!("{}", m.cache_hit_pct), theme::verified()),
                Span::styled("% hit", theme::label()),
            ],
            &format!("{} MiB/s", m.cache_mibs),
        ),
        cells[3],
    );
    f.render_widget(
        cell(
            "BROKER",
            vec![
                Span::styled(format!("{}", m.broker.headroom_gib), theme::metric()),
                Span::styled(" GiB", theme::label()),
            ],
            "never-OOM",
        ),
        cells[4],
    );
    let blocks_style = if m.gov_violations == 0 {
        theme::verified()
    } else {
        theme::halt()
    };
    f.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled("BLOCKS", theme::label())),
            Line::from(Span::styled(m.gov_violations.to_string(), blocks_style)),
            Line::from(Span::styled("fail-closed", theme::panel_border())),
        ]),
        cells[5],
    );
}

fn draw_plan(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let (_, _, arrow) = glyphs();
    // Pipeline chip line: name + state marker, arrow-joined.
    let mut pipeline: Vec<Span<'static>> = Vec::new();
    for (i, n) in m.dag.iter().enumerate() {
        if i > 0 {
            pipeline.push(Span::styled(format!("  {arrow}  "), theme::panel_border()));
        }
        pipeline.push(node_marker(n.state));
        pipeline.push(Span::raw(" "));
        let name_style = match n.state {
            NodeState::Running => theme::signal_bold(),
            NodeState::Cached => theme::verified(),
            NodeState::Blocked => theme::halt(),
            _ => theme::label(),
        };
        pipeline.push(Span::styled(n.name.clone(), name_style));
    }

    let mut lines = vec![Line::from(pipeline), Line::from("")];
    for n in &m.dag {
        let state_txt = match n.state {
            NodeState::Cached => "cache hit",
            NodeState::Done => "done",
            NodeState::Running => "running",
            NodeState::Queued => "queued",
            NodeState::Blocked => "blocked",
        };
        lines.push(Line::from(vec![
            Span::raw(" "),
            node_marker(n.state),
            Span::styled(format!(" {:<22}", n.name), theme::signal()),
            Span::styled(format!("{state_txt:<11}"), theme::label()),
            Span::styled(n.note.clone(), theme::panel_border()),
        ]));
    }

    let title = format!(
        "Plan · typed DAG   [{}]   run {} · {} · ✓ kind-checked before launch",
        m.plan, m.run_id, m.elapsed
    );
    f.render_widget(Paragraph::new(lines).block(panel(&title)), area);
}

fn draw_instruments(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
        ])
        .split(area);
    draw_mesh(f, cols[0], m);
    draw_broker(f, cols[1], m);
    draw_privacy(f, cols[2], m);
    draw_cache(f, cols[3], m);
}

fn draw_mesh(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let mut lines = Vec::new();
    for p in &m.mesh {
        let (t, ts) = match p.trust {
            Trust::Trusted => ("trusted", theme::verified()),
            Trust::Registered => ("registd", theme::signal()),
            Trust::Anonymous => ("anon", theme::label()),
        };
        let mut id_line = vec![Span::styled(format!("{:<12}", p.id), theme::signal())];
        if p.is_self {
            id_line.push(Span::styled(" self", theme::verified()));
        }
        lines.push(Line::from(id_line));
        lines.push(Line::from(vec![
            Span::styled(format!(" {t:<8}"), ts),
            Span::styled(format!("rep {:.2} ", p.reputation), theme::panel_border()),
        ]));
        lines.push(Line::from(vec![Span::styled(
            format!(" {}", p.caps),
            theme::panel_border(),
        )]));
    }
    f.render_widget(Paragraph::new(lines).block(panel("Symmetric mesh")), area);
}

fn draw_broker(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let frac = m.broker.held_gib as f64 / m.broker.budget_gib.max(1) as f64;
    let lines = vec![
        Line::from(vec![
            Span::styled(format!("{}", m.broker.held_gib), theme::signal_bold()),
            Span::styled(
                format!(" / {} GiB box-fit", m.broker.budget_gib),
                theme::label(),
            ),
        ]),
        Line::from(bar(frac, 22, theme::signal())),
        Line::from(""),
        Line::from(vec![
            Span::styled("held  ", theme::label()),
            Span::styled(m.broker.top_stage.clone(), theme::panel_border()),
        ]),
        Line::from(vec![
            Span::styled("admits", theme::label()),
            Span::styled(
                format!("  {} GiB headroom", m.broker.headroom_gib),
                theme::panel_border(),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            if theme::ascii_only() {
                "over-budget refused - never OOMs the box"
            } else {
                "✓ over-budget refused — never OOMs the box"
            },
            theme::verified(),
        )),
    ];
    f.render_widget(
        Paragraph::new(lines).block(panel("Broker · memory admission")),
        area,
    );
}

fn draw_privacy(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let mut lines = Vec::new();
    for c in &m.corpora {
        if c.restricted {
            lines.push(Line::from(Span::styled(c.name.clone(), theme::halt())));
            let x = if theme::ascii_only() { "!" } else { "⨯" };
            lines.push(Line::from(Span::styled(
                format!(" {x} Restricted · local-only"),
                theme::halt(),
            )));
            lines.push(Line::from(Span::styled(
                "   gradients never dispatch",
                theme::panel_border(),
            )));
        } else {
            let rem = (c.eps_budget - c.eps_spent).max(0.0);
            let frac = c.eps_spent / c.eps_budget.max(0.001);
            let bar_style = if rem < c.eps_budget * 0.25 {
                theme::amber()
            } else {
                theme::signal()
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{:<17}", c.name), theme::label()),
                Span::styled(format!("ε {rem:.1}/{:.0}", c.eps_budget), theme::signal()),
            ]));
            lines.push(Line::from(bar(frac, 22, bar_style)));
        }
    }
    f.render_widget(
        Paragraph::new(lines).block(panel("Privacy ledger · ε")),
        area,
    );
}

fn draw_cache(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let mut lines = vec![
        Line::from(vec![
            Span::styled(format!("{}", m.cache_hit_pct), theme::verified()),
            Span::styled("% hit  ", theme::label()),
            Span::styled(format!("{} MiB/s", m.cache_mibs), theme::panel_border()),
        ]),
        Line::from(""),
    ];
    for e in &m.cache {
        let (tag, ts) = if e.hit {
            ("HIT ", theme::verified())
        } else {
            ("MISS", theme::amber())
        };
        lines.push(Line::from(vec![
            Span::styled(tag, ts),
            Span::styled(format!(" {:<16}", e.stage), theme::panel_border()),
        ]));
        lines.push(Line::from(vec![
            Span::styled(format!("     {} ", e.hash), theme::panel_border()),
            Span::styled(e.note.clone(), theme::label()),
        ]));
    }
    f.render_widget(
        Paragraph::new(lines).block(panel("Content-addressed cache")),
        area,
    );
}

fn draw_gov(f: &mut Frame<'_>, area: Rect, m: &ConsoleModel) {
    let shield = if theme::ascii_only() { "[#]" } else { "◈" };
    let line = Line::from(vec![
        Span::styled(
            format!(" {shield} CLINICAL HARD-BLOCK ENFORCED  "),
            theme::verified(),
        ),
        Span::styled(
            "Restricted/PHI local-only · weights + gradients never leave the box  ",
            theme::label(),
        ),
        Span::styled(
            format!("· {} violations ", m.gov_violations),
            if m.gov_violations == 0 {
                theme::verified()
            } else {
                theme::halt()
            },
        ),
        Span::styled("· ADR 0061·0080", theme::panel_border()),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    #[test]
    fn demo_model_is_coherent() {
        let m = ConsoleModel::demo();
        assert_eq!(m.stages_total, 11);
        assert!(m.mesh.iter().any(|p| p.is_self));
        assert!(m.corpora.iter().any(|c| c.restricted), "a clinical corpus");
        assert_eq!(m.gov_violations, 0);
    }

    #[test]
    fn draws_without_panicking_at_several_sizes() {
        theme::detect("always", "unicode");
        let m = ConsoleModel::demo();
        for (w, h) in [(120u16, 40u16), (80, 30), (200, 60), (60, 24)] {
            let backend = TestBackend::new(w, h);
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| draw_console(f, f.area(), &m)).unwrap();
        }
    }

    #[test]
    fn ascii_fallback_draws() {
        theme::detect("never", "ascii");
        let m = ConsoleModel::demo();
        let backend = TestBackend::new(100, 36);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_console(f, f.area(), &m)).unwrap();
        theme::detect("auto", "auto");
    }
}
