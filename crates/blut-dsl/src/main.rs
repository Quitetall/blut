// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-dsl` — evaluate a `.star` recipe to a `PlanSpec` JSON on stdout.
//!
//! The engine shells out to this binary for `.star` recipes (ADR 0078) so
//! that `starlark` — and its `serde_json/arbitrary_precision` feature, which
//! would break the engine's internally-tagged enums under Cargo feature
//! unification — never links into the engine process. The emitted JSON is
//! consumed via the engine's plain `.json` PlanSpec path.
//!
//! Usage:
//!   blut-dsl <script.star> [--args '<json>']
//! Prints the PlanSpec as JSON to stdout; a parse/eval error goes to stderr
//! with a non-zero exit.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;

/// Evaluate a Starlark BLUT recipe to a PlanSpec JSON.
#[derive(Parser, Debug)]
#[command(name = "blut-dsl", version)]
struct Cli {
    /// Path to the `.star` script (must define `build(args)`).
    script: PathBuf,
    /// Args JSON passed to the script's `build(args)`.
    #[arg(long, default_value = "{}")]
    args: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("blut-dsl: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<String> {
    let cli = Cli::parse();
    let args: serde_json::Value =
        serde_json::from_str(&cli.args).context("parse --args as JSON")?;
    let src = std::fs::read_to_string(&cli.script)
        .with_context(|| format!("read script {}", cli.script.display()))?;
    let label = cli.script.display().to_string();
    let spec =
        blut_dsl::evaluate_script(&src, &label, &args).map_err(|e| anyhow::anyhow!("{e}"))?;
    serde_json::to_string_pretty(&spec).context("serialize PlanSpec")
}
