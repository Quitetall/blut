// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Fuzz target: `blut::recipes::declarative::DeclarativeRecipe::parse`.
//!
//! Declarative recipes are user-authored `.toml` files discovered from
//! `~/.config/blut/recipes/*.toml` (or `$BLUT_USER_RECIPES_DIR`) -- untrusted
//! text a user can hand-edit or accidentally corrupt. `parse` is already a
//! pure, pub function taking `&str`, so this target just needs a lossy UTF-8
//! decode of the fuzzer's raw bytes before calling it.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let s = String::from_utf8_lossy(data);
    let _ = blut::recipes::declarative::DeclarativeRecipe::parse(&s, "fuzz");
});
