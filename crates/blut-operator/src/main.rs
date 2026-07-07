// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-operator` — emit CRDs and (later) run the reconcile loop.

use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use kube::runtime::Controller;
use kube::runtime::watcher::Config;
use kube::{Api, Client, CustomResourceExt};

use blut_operator::reconcile::{Context, error_policy, reconcile_plan, reconcile_pool};
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
        Command::Run => tokio::runtime::Runtime::new()?.block_on(run_operator()),
    }
}

/// Watch both CRDs and reconcile them until shutdown. Needs an in-cluster or
/// kubeconfig-provided client (infra-gated; the kind smoke exercises it).
async fn run_operator() -> Result<()> {
    let client = Client::try_default().await?;
    let ctx = Arc::new(Context {
        client: client.clone(),
    });

    let plans: Api<BlutPlan> = Api::all(client.clone());
    let pools: Api<BlutWorkerPool> = Api::all(client.clone());

    let plan_ctrl = Controller::new(plans, Config::default())
        .run(reconcile_plan, error_policy, ctx.clone())
        .for_each(|res| async move {
            if let Err(e) = res {
                tracing::warn!("plan controller: {e}");
            }
        });

    let pool_ctrl = Controller::new(pools, Config::default())
        .run(reconcile_pool, error_policy, ctx)
        .for_each(|res| async move {
            if let Err(e) = res {
                tracing::warn!("pool controller: {e}");
            }
        });

    tracing::info!("blut-operator watching BlutPlan + BlutWorkerPool");
    tokio::join!(plan_ctrl, pool_ctrl);
    Ok(())
}
