// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Body of `tui::state_tests` — see the module doc on the declaration in
//! `tui/mod.rs`. Mounted via `#[path]` so `super::*` is still the tui module.

use super::*;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// A fresh `App` with no overlay, cockpit view, pointed at a temp
/// repo root so nothing in these tests touches the dev tree. Starts on the
/// (legacy) Cockpit view — these exercise the cockpit recipe-picker / editor
/// overlays, which the new default Console view doesn't host.
fn app() -> App {
    super::isolate_datasets_db_for_tests();
    let mut a = App::new(test_registry());
    a.view = super::View::Cockpit;
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
    assert_eq!(
        v["lr"],
        serde_json::json!(0.01),
        "number stays a JSON number"
    );
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
    let jsonl_name =
        format!("f3jsonl{:?}", std::thread::current().id()).replace(['(', ')', ' '], "");
    let split_name =
        format!("f3split{:?}", std::thread::current().id()).replace(['(', ')', ' '], "");
    let r1 = crate::datasets_db::record_from_jsonl(&jsonl_name, &f, "dataset.jsonl", None).unwrap();
    let r2 = crate::datasets_db::record_from_jsonl(&split_name, &f, "dataset.split", None).unwrap();
    crate::datasets_db::add(&conn, &r1).unwrap();
    crate::datasets_db::add(&conn, &r2).unwrap();

    a.open_dataset_picker(&test_fixtures::TRAIN_EPSILON);
    let names: Vec<String> = match &a.overlay {
        Overlay::DatasetPicker { datasets, .. } => {
            datasets.iter().map(|d| d.name.clone()).collect()
        }
        other => panic!(
            "expected DatasetPicker, got {:?}",
            std::mem::discriminant(other)
        ),
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
        Overlay::Editor {
            raw_mode,
            raw_buffer,
            ..
        } => {
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
        assert!(
            v.is_object(),
            "assembled args must be a JSON object for `{}`",
            r.name
        );
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
    assert!(
        bad.unwrap_err().contains("JSON"),
        "message names the JSON fault"
    );
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
        ('C', View::Compare),
        ('G', View::Dag),
        ('I', View::Lineage),
        ('A', View::Artifacts),
        ('M', View::Metrics),
        ('P', View::Catalog),
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
    // From a detail view, both Esc and 'b' go back to the Console home.
    for back in [code(KeyCode::Esc), key('b')] {
        let mut a = app();
        handle_key(&mut a, key('J')); // → Jobs
        assert_eq!(a.view, View::Jobs);
        handle_key(&mut a, back);
        assert_eq!(
            a.view,
            super::View::Console,
            "Esc/b should return to the Console home"
        );
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
        matches!(
            a.overlay,
            Overlay::Editor { .. } | Overlay::DatasetPicker { .. }
        ),
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
