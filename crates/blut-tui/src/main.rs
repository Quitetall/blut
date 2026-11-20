// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-tui` binary — reached as `blut tui` via the engine's external
//! subcommand dispatch (ADR 0083). This standalone build carries an EMPTY
//! registry, so it serves the engine-generic views (jobs / log / system /
//! history / leaderboard / …) with no cookbook recipes; a cookbook binary
//! (e.g. `lqt`) links the `blut-tui` LIB instead and opens the same console
//! over its live registry.

use std::process::ExitCode;

const USAGE: &str = "\
blut-tui — the BLUT terminal cockpit (reached as `blut tui`)

Usage: blut-tui [--check]

Options:
  --check        Headless self-check: render every view to a test backend and
                 exit 0 if all draw non-blank. Needs no terminal.
  -h, --help     Print this help
  -V, --version  Print the version

With no option, opens the interactive cockpit (needs a terminal).";

/// What the command line asks for. Anything unrecognised is refused: this
/// binary used to ignore its arguments, so `blut-tui --help` (or a typo such
/// as `--chek` in a CI script) opened the raw-mode cockpit instead.
#[derive(Debug, PartialEq, Eq)]
enum Mode {
    Console,
    Check,
    Help,
    Version,
}

fn parse_mode<I: IntoIterator<Item = String>>(args: I) -> Result<Mode, String> {
    let mut mode = Mode::Console;
    for arg in args {
        mode = match arg.as_str() {
            "--check" => Mode::Check,
            "-h" | "--help" => return Ok(Mode::Help),
            "-V" | "--version" => return Ok(Mode::Version),
            other => return Err(format!("unexpected argument '{other}'")),
        };
    }
    Ok(mode)
}

#[tokio::main]
async fn main() -> ExitCode {
    let mode = match parse_mode(std::env::args().skip(1)) {
        Ok(mode) => mode,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = match mode {
        Mode::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Mode::Version => {
            println!("blut-tui {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Mode::Check => blut_tui::check(blut::framework::Registry::new()),
        Mode::Console => {
            blut_tui::run_console_loop(std::sync::Arc::new(blut::framework::Registry::new())).await
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, parse_mode};

    fn parse(args: &[&str]) -> Result<Mode, String> {
        parse_mode(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn help_version_and_check_are_recognised_and_typos_refused() {
        assert_eq!(parse(&[]), Ok(Mode::Console));
        assert_eq!(parse(&["--check"]), Ok(Mode::Check));
        assert_eq!(parse(&["--help"]), Ok(Mode::Help));
        assert_eq!(parse(&["-h"]), Ok(Mode::Help));
        assert_eq!(parse(&["--version"]), Ok(Mode::Version));
        assert_eq!(parse(&["--check", "--help"]), Ok(Mode::Help));
        assert!(parse(&["--chek"]).unwrap_err().contains("--chek"));
        assert!(parse(&["jobs"]).is_err());
    }
}
