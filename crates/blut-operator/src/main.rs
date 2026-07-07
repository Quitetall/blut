// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-operator` — emit CRDs and (later) run the reconcile loop.

use anyhow::Result;
use clap::{Parser, Subcommand};
use kube::CustomResourceExt;

use blut_operator::{BlutPlan, BlutWorkerPool};

#[derive(Parser, Debug)]
#[command(
    name = "blut-operator",
    about = "BLUT Kubernetes operator (ADR 0037 adapter)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print the CRD YAML for `kubectl apply -f -` (BlutPlan + BlutWorkerPool).
    Crds,
    /// Run the reconcile loop (watches BlutPlan / BlutWorkerPool). [not yet wired]
    Run,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Command::Crds => {
            // Two YAML documents, `---`-separated, ready for `kubectl apply`.
            print!("{}", serde_yaml::to_string(&BlutPlan::crd())?);
            println!("---");
            println!(
                "{}",
                serde_yaml::to_string(&BlutWorkerPool::crd())?.trim_end()
            );
            Ok(())
        }
        Command::Run => {
            anyhow::bail!("reconcile loop not yet wired (B2 reconcilers land next)")
        }
    }
}
