---
name: blut-cockpit-tui
description: Architecture + invariants of the `blut tui` cockpit (src/tui/) — state machine, keymap, cursor-clamp invariant, gatherer/draw split. For future TUI reviews.
metadata:
  type: project
---

# BLUT cockpit TUI (`src/tui/`)

Ratatui single-surface cockpit, shipping in 1.0. Four files:
- `mod.rs` (~4200 lines incl. tests) — `App` state, `Overlay` enum, `View` enum, key handlers, all `draw_*` fns, F2 form helpers.
- `views.rs` — pure engine-data gatherers (`run_history` / `leaderboard` / `compare` / `metrics_for` / `artifacts_for` / `lineage_for` / `dag_for` / `reset`). Each reads BLUT's own state (jobs store, lineage DB, cache, plan graph); best-effort, never panics (all `.ok()/.unwrap_or_default()`).
- `theme.rs` — atomics-backed style getters, NO_COLOR/TERM=dumb/locale detection.
- `system.rs` — once-per-tick `/proc` + `nvidia-smi` + `df` probe, all best-effort → "n/a".

## Key invariants (don't break)
- **No hot path.** Everything is per-keystroke or per-1.5s-tick. Gatherers do bounded I/O; render is O(rows). No numeric loops. Performance is a non-issue here.
- **Cursor-clamp invariant** (`load_view_data` tail): `list_cursor = min(list_cursor, active_list_len-1)`. Only zeroes when the list is empty/shrinks below the cursor; preserves an in-range cursor. `current_job_id()` depends on this to target the highlighted row.
- **`current_job_id()` capture-before-reset**: `set_view` captures the job id BEFORE flipping `self.view` and zeroing `list_cursor`, so drilling History→Enter→DAG targets the highlighted run (still on History's cursor at capture time).
- **Leaderboard cap alignment**: `leaderboard()` over-fetches `LEADERBOARD_LIMIT*3`, dedups by job_id, `.take(LEADERBOARD_LIMIT)`. `active_list_len` = `runs.len() ≤ 20`, and `draw_leaderboard` `.take(LEADERBOARD_LIMIT)`. Keep these three aligned (there's a `const{}` assert guarding it).
- **Reserved hotkey pool** (`recipe_menu`): `q Q r R c C j k l b B` excluded from the 1-9,a-z recipe pool. Capital view keys (J L Y H B C G I A M P X) are matched in `handle_key_main` BEFORE the cockpit/detail dispatch, so they can't be shadowed by recipe hotkeys (which are lowercase/digits only).
- **Two-press reset confirm** (`fire_reset` + `RESET_WINDOW=3s`): first Enter arms `(idx, Instant)`; second Enter within window + same idx fires. Un-bypassable; `key_sequences` harness test never sends the 2nd Enter.

## Reviewed-clean fixes (2026-06 round, all verified correct)
refresh tick → `refresh_all()`; `list_cursor` re-clamp in `load_view_data`; `current_job_id` honors Cockpit/Jobs/Log selection; `b`/`B` reserved; `leaderboard` over-fetch+dedup+cap; `ForgetFootprints` reports write errors. No regressions found.

## Minor notes (not bugs)
- `ForgetFootprints` calls `store.forget(k)` per key → `save()` (full-store rewrite) once PER entry = O(N) disk writes. Fine for the tiny footprint store; would matter if it grew large. A batch `forget_all`+single save would be cleaner.
- `pending_focus` give-up budget (`FOCUS_GIVE_UP_TICKS=20`) is decremented in `refresh_jobs`, which runs on manual `r` too — hammering `r` burns the ~30s budget faster. Bounded + benign.
