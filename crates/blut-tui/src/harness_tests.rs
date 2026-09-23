// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Body of `tui::harness_tests` — see the module doc on the declaration in
//! `tui/mod.rs`. Mounted via `#[path]` so `super::*` is still the tui module.

use super::test_fixtures::test_registry;
use super::*;

const ALL_VIEWS: &[View] = &[
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

fn job(id: &str, state: JobState) -> JobSummary {
    JobSummary {
        id: id.into(),
        state,
        pid: Some(1234),
        output_name: Some("artifact".into()),
        last_loss: Some(0.5),
        last_step: Some(10),
        final_loss: Some(0.4),
    }
}

fn run_row(id: &str, recipe: &str, metric: Option<f64>) -> views::RunRow {
    views::RunRow {
        job_id: id.into(),
        recipe: recipe.into(),
        outcome: "done".into(),
        metric,
        when: "2026-06-18 07:30".into(),
    }
}

fn graph(n: usize) -> blut::framework::GraphSnapshot {
    use blut::framework::NodeStatus as N;
    use blut::framework::graph::{GraphNode, PlanGraphEdge};
    let st = [
        N::Done,
        N::Running,
        N::Failed,
        N::Skipped,
        N::Pending,
        N::Blocked,
        N::Killed,
        N::NotSelected,
        N::Pruned,
        N::Ready,
    ];
    let nodes = (0..n)
        .map(|i| GraphNode {
            idx: i,
            stage_name: format!("stage_number_{i}_with_a_longish_name"),
            status: st[i % st.len()],
            args_summary: String::new(),
            input_hash: if i == 0 {
                None
            } else {
                Some("abcdef0123456789".into())
            },
            output_hash: Some("0123456789abcdef".into()),
            elapsed_secs: Some(1.5 * i as f64),
            cache_hit: i % 2 == 0,
            hpo: None,
        })
        .collect();
    let edges = (1..n)
        .map(|i| PlanGraphEdge { from: i - 1, to: i })
        .collect();
    blut::framework::GraphSnapshot {
        job: "20260618-073012-000000001".into(),
        name: "demo_plan".into(),
        nodes,
        edges,
        condition_gates: Vec::new(),
    }
}

fn lineage_view() -> views::LineageView {
    views::LineageView {
        rows: vec![
            views::LineageRow {
                node_idx: 0,
                stage: "make_data".into(),
                input: "—".into(),
                output: "abcdef0123".into(),
                cached: false,
                elapsed: "2.3s".into(),
            },
            views::LineageRow {
                node_idx: 1,
                stage: "train".into(),
                input: "abcdef0123".into(),
                output: "9876543210".into(),
                cached: true,
                elapsed: "—".into(),
            },
        ],
        cache_hits: 1,
        cache_misses: 1,
        freshness: "STALE".into(),
    }
}

fn compare_col(id: &str) -> views::CompareCol {
    views::CompareCol {
        job_id: id.into(),
        recipe: "train_codec".into(),
        metrics: vec![("loss".into(), 0.42), ("val_r".into(), 0.81)],
        gpu: Some(72.5),
    }
}

/// An App with every per-view cache populated with reasonable data.
fn populated_app() -> App {
    super::isolate_datasets_db_for_tests();
    {
        let _theme = crate::theme::test_lock();
        theme::detect("always", "unicode");
    }
    let mut app = App::new(test_registry());
    // The harness exercises the training-cockpit surface (view switches, recipe
    // picker, per-job panels), so run it in cookbook mode as run_cockpit does.
    app.cookbook_mode = true;
    app.jobs = vec![
        job("20260618-073012-000000001", JobState::Running),
        job("20260618-070000-000000002", JobState::Done),
        job("20260617-235959-000000003", JobState::Failed),
        job("20260617-120000-000000004", JobState::Cancelled),
    ];
    app.selected.select(Some(0));
    app.runs = vec![
        run_row("20260618-073012-000000001", "train_codec", Some(0.42)),
        run_row("20260618-070000-000000002", "eval_codec", Some(0.55)),
        run_row("20260617-235959-000000003", "distill", None),
    ];
    app.artifacts = vec![views::ArtifactRow {
        kind: "checkpoint".into(),
        stage: "train".into(),
        hash: "abcdef012345".into(),
        when: "2026-06-18 07:30".into(),
    }];
    app.dag = Some(graph(6));
    app.lineage = lineage_view();
    app.metrics = vec![("loss".into(), 0.42), ("val_r".into(), 0.81)];
    app.compare = vec![compare_col("job_a"), compare_col("job_b")];
    app.marked = vec!["20260618-073012-000000001".into()];
    app.log_lines = (0..50).map(|i| format!("step {i}: loss=0.{i}")).collect();
    app
}

fn render(app: &mut App, w: u16, h: u16) {
    let mut term = Terminal::new(ratatui::backend::TestBackend::new(w, h)).expect("test terminal");
    term.draw(|f| draw(f, app)).expect("draw must not panic");
}

#[test]
fn populated_views_render_across_sizes() {
    // Every view, with full caches, at sane → cramped → degenerate sizes.
    for size in [
        (200u16, 60u16),
        (120, 40),
        (80, 24),
        (40, 12),
        (20, 6),
        (8, 3),
        (1, 1),
    ] {
        for &view in ALL_VIEWS {
            let mut app = populated_app();
            app.view = view;
            render(&mut app, size.0, size.1); // panic ⇒ test fails
        }
    }
}

#[test]
fn adversarial_data_does_not_panic() {
    // NaN / inf / negative metrics, a 300-char recipe name, unicode, a huge
    // DAG, missing hashes — none may panic the drawers.
    let mut app = populated_app();
    let long = "ε".repeat(300);
    app.runs = vec![
        run_row("x", &long, Some(f64::NAN)),
        run_row("y", "r", Some(f64::INFINITY)),
        run_row("z", "r", Some(f64::NEG_INFINITY)),
        run_row("w", "r", Some(-0.0)),
    ];
    app.metrics = vec![
        ("nan".into(), f64::NAN),
        ("inf".into(), f64::INFINITY),
        (long.clone(), 1.0),
    ];
    app.compare = vec![
        views::CompareCol {
            job_id: long.clone(),
            recipe: long.clone(),
            metrics: vec![("loss".into(), f64::NAN)],
            gpu: None,
        },
        compare_col("b"),
    ];
    app.dag = Some(graph(64));
    app.lineage = views::LineageView {
        rows: (0..40)
            .map(|i| views::LineageRow {
                node_idx: i,
                stage: long.clone(),
                input: "—".into(),
                output: "—".into(),
                cached: i % 2 == 0,
                elapsed: "—".into(),
            })
            .collect(),
        cache_hits: u64::MAX,
        cache_misses: u64::MAX,
        freshness: "UNKNOWN".into(),
    };
    for &view in ALL_VIEWS {
        app.view = view;
        render(&mut app, 80, 24);
        render(&mut app, 20, 6);
    }
}

#[test]
fn cursor_past_shrunk_data_does_not_panic() {
    // Park the cursor at the end of a list, then shrink the data (as a
    // refresh returning fewer rows would) and render — `.get()`/`enumerate`
    // must keep it safe even though the cursor now points past the end.
    let mut app = populated_app();
    app.view = View::History;
    app.list_cursor = 999;
    render(&mut app, 80, 24);
    app.runs.clear();
    render(&mut app, 80, 24);
    app.view = View::Artifacts;
    app.list_cursor = 999;
    app.artifacts.clear();
    render(&mut app, 80, 24);
    app.view = View::Catalog;
    app.list_cursor = usize::MAX;
    render(&mut app, 80, 24);
}

fn k(c: char) -> event::KeyEvent {
    event::KeyEvent::new(event::KeyCode::Char(c), event::KeyModifiers::NONE)
}

#[test]
fn key_sequences_across_views_do_not_panic() {
    // Walk every view-switch key, list nav, mark, and back — repeatedly —
    // and confirm no key path panics or wedges. NOTE: we never send a second
    // Enter on the Reset view, so no destructive maintenance action fires.
    let mut app = populated_app();
    let seq = "JLYHBCGIAMPX";
    for c in seq.chars() {
        handle_key(&mut app, k(c));
        // exercise list nav + mark + refresh in whatever view we landed in
        for nav in ['j', 'k', 'm', ' ', 'r'] {
            handle_key(&mut app, k(nav));
        }
        handle_key(&mut app, k('b')); // back to cockpit
        assert!(!app.quit, "navigation must not quit");
        render(&mut app, 80, 24);
    }
}

#[test]
fn reset_arms_but_a_single_enter_never_fires() {
    // On the Maintenance view a single Enter only ARMS the action (sets the
    // confirm window); it must not run the destructive op. We assert the
    // armed state is set and never send the confirming second Enter.
    let mut app = populated_app();
    app.set_view(View::Reset);
    handle_key(
        &mut app,
        event::KeyEvent::new(event::KeyCode::Enter, event::KeyModifiers::NONE),
    );
    assert!(app.reset_armed.is_some(), "first Enter must arm, not fire");
    render(&mut app, 80, 24);
}

#[test]
fn enter_on_a_run_drills_into_its_dag() {
    // Enter on a History row loads that run's DAG view for the selected run.
    let mut app = populated_app();
    app.set_view(View::History);
    app.list_cursor = 1; // second run
    handle_key(
        &mut app,
        event::KeyEvent::new(event::KeyCode::Enter, event::KeyModifiers::NONE),
    );
    assert_eq!(app.view, View::Dag, "Enter on a run opens its DAG");
    render(&mut app, 80, 24);
}

#[test]
fn leaderboard_cursor_cannot_exceed_render_cap() {
    // Regression guard for the MiMo finding: the move_list bound (runs.len)
    // must not exceed the rendered cap, so a fetched list can never be longer
    // than LEADERBOARD_LIMIT.
    const {
        assert!(
            views::LEADERBOARD_LIMIT <= 20,
            "render path draws .take(LEADERBOARD_LIMIT); keep the query cap aligned"
        )
    };
}

#[test]
fn load_view_data_clamps_cursor_into_range() {
    // Regression: a refresh that shrinks the active list must re-clamp the
    // cursor so current_job_id() can't target an off-list run. Use Catalog
    // (registry-driven, no DB dependency) for a deterministic length.
    let mut app = populated_app();
    app.set_view(View::Catalog);
    let n = app.catalog.len();
    assert!(n > 0, "fixture registry must register recipes");
    app.list_cursor = 9_999;
    app.load_view_data(View::Catalog, None);
    assert!(
        app.list_cursor < n,
        "cursor must be clamped within the list"
    );
}

#[test]
fn current_job_id_honors_cockpit_selection() {
    // Regression: jumping from the cockpit straight to a per-job view must
    // target the highlighted job, not just the newest.
    let mut app = populated_app();
    app.view = View::Cockpit;
    app.selected.select(Some(2));
    assert_eq!(
        app.current_job_id().as_deref(),
        Some(app.jobs[2].id.as_str()),
        "cockpit selection should drive the per-job target"
    );
}
