// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-operator` — print the CRDs, or run the reconcile loop.

use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use k8s_openapi::api::apps::v1::StatefulSet;
use k8s_openapi::api::batch::v1::Job;
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::Controller;
use kube::runtime::reflector::ObjectRef;
use kube::runtime::watcher::Config;
use kube::{Api, Client};

use blut_operator::reconcile::{Context, PLAN_LABEL, error_policy, reconcile_plan, reconcile_pool};
use blut_operator::{BlutPlan, BlutWorkerPool, crds_yaml};

#[derive(Parser, Debug)]
#[command(
    name = "blut-operator",
    version,
    about = "BLUT Kubernetes operator: runs BlutPlan resources as Jobs and keeps BlutWorkerPool StatefulSets alive"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print the CRD YAML (BlutPlan + BlutWorkerPool) for `kubectl apply -f -`.
    Crds,
    /// Run the reconcile loop until SIGTERM or ctrl-c.
    Run {
        /// Watch only this namespace (the RBAC can then be a Role instead of a
        /// ClusterRole). Default: every namespace.
        #[arg(long, env = "BLUT_OPERATOR_NAMESPACE")]
        namespace: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Crds => {
            print!("{}", crds_yaml()?);
            Ok(())
        }
        Command::Run { namespace } => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| "info".into()),
                )
                .init();
            tokio::runtime::Runtime::new()?.block_on(run_operator(namespace))
        }
    }
}

fn scoped<K>(client: &Client, namespace: Option<&str>) -> Api<K>
where
    K: kube::Resource<Scope = k8s_openapi::NamespaceResourceScope>,
    <K as kube::Resource>::DynamicType: Default,
{
    match namespace {
        Some(ns) => Api::namespaced(client.clone(), ns),
        None => Api::all(client.clone()),
    }
}

/// Log a reconcile failure. A child's event can arrive after its owner was
/// deleted; that is routine, so it is logged at debug, not as a warning.
fn log_outcome<T, E: std::error::Error, Q: std::error::Error>(
    which: &str,
    res: Result<T, kube::runtime::controller::Error<E, Q>>,
) {
    match res {
        Ok(_) => {}
        Err(kube::runtime::controller::Error::ObjectNotFound(obj)) => {
            tracing::debug!("{which} controller: {obj} is gone");
        }
        Err(e) => tracing::warn!("{which} controller: {e}"),
    }
}

/// Watch both CRDs (and the Jobs and StatefulSets they own) and reconcile them
/// until shutdown.
async fn run_operator(namespace: Option<String>) -> Result<()> {
    let client = Client::try_default().await?;
    let ns = namespace.as_deref();
    let ctx = Arc::new(Context {
        client: client.clone(),
    });

    let owned = Config::default().labels("app.kubernetes.io/managed-by=blut-operator");
    let plan_ctrl = Controller::new(scoped::<BlutPlan>(&client, ns), Config::default())
        .owns(scoped::<Job>(&client, ns), owned.clone())
        // Pods too: a deleted or evicted pod is gone soon after it stops, and
        // the plan's status should say what happened to it.
        .watches(scoped::<Pod>(&client, ns), owned.clone(), |pod: Pod| {
            let ns = pod.metadata.namespace.clone();
            pod.metadata
                .labels
                .as_ref()
                .and_then(|l| l.get(PLAN_LABEL))
                .map(|plan| {
                    let r = ObjectRef::<BlutPlan>::new(plan);
                    match ns {
                        Some(ns) => r.within(&ns),
                        None => r,
                    }
                })
        })
        .shutdown_on_signal()
        .run(reconcile_plan, error_policy, ctx.clone())
        .for_each(|res| async move { log_outcome("plan", res) });

    let pool_ctrl = Controller::new(scoped::<BlutWorkerPool>(&client, ns), Config::default())
        .owns(scoped::<StatefulSet>(&client, ns), owned)
        .shutdown_on_signal()
        .run(reconcile_pool, error_policy, ctx)
        .for_each(|res| async move { log_outcome("pool", res) });

    tracing::info!(
        namespace = ns.unwrap_or("<all>"),
        "blut-operator watching BlutPlan + BlutWorkerPool"
    );
    tokio::join!(plan_ctrl, pool_ctrl);
    tracing::info!("blut-operator stopped");
    Ok(())
}
