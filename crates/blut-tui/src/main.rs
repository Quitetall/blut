// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-tui` binary — reached as `blut tui` via the engine's external
//! subcommand dispatch (ADR 0083). This standalone build carries an EMPTY
//! registry, so it serves the engine-generic views (jobs / log / system /
//! history / leaderboard / …) with no cookbook recipes; a cookbook binary
//! (e.g. `lqt`) links the `blut-tui` LIB instead and opens the same console
//! over its live registry.

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let check = std::env::args().skip(1).any(|a| a == "--check");
    let reg = blut::framework::Registry::new();
    if check {
        blut_tui::check(reg)
    } else {
        blut_tui::run_console_loop(std::sync::Arc::new(reg)).await
    }
}
