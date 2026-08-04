// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-metrics` binary — reached as `blut metrics` via the ADR 0083
//! external-subcommand dispatch. Tails engine telemetry records and serves an
//! OpenMetrics scrape endpoint; optionally forwards spans to an OTLP collector.
//!
//! This process owns ALL network I/O for telemetry. The engine only writes
//! records to `status.jsonl` (ADR 0034), which is why a slow or dead collector
//! can never back-pressure stage execution.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use blut_metrics::MetricStore;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "blut-metrics", about = "BLUT telemetry sidecar (ADR 0097)")]
struct Args {
    /// status.jsonl file(s) to tail. Repeatable.
    #[arg(long = "status", required = true)]
    status: Vec<std::path::PathBuf>,
    /// Scrape bind address. Loopback by default — a metrics endpoint is an
    /// export surface and should be opted onto a network deliberately.
    #[arg(long, default_value = "127.0.0.1:9464")]
    bind: std::net::SocketAddr,
    /// Re-read cadence, milliseconds.
    #[arg(long, default_value_t = 1000)]
    interval_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let store = Arc::new(Mutex::new(MetricStore::new()));

    // Re-read from the top each pass: the store is idempotent for gauges and
    // max-merges counters, so a re-read cannot corrupt a series. Simple and
    // correct beats an offset cursor for a scrape-driven exporter.
    let tail = {
        let store = store.clone();
        let paths = args.status.clone();
        let interval = std::time::Duration::from_millis(args.interval_ms.max(50));
        tokio::spawn(async move {
            loop {
                let mut fresh = MetricStore::new();
                for path in &paths {
                    if let Ok(text) = std::fs::read_to_string(path) {
                        for line in text.lines() {
                            fresh.ingest_line(line);
                        }
                    }
                }
                if let Ok(mut guard) = store.lock() {
                    *guard = fresh;
                }
                tokio::time::sleep(interval).await;
            }
        })
    };

    let app = blut_metrics::router(store);
    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("bind {}", args.bind))?;
    tracing::info!("blut-metrics serving OpenMetrics on http://{}", args.bind);
    axum::serve(listener, app).await.context("serve")?;
    tail.abort();
    Ok(())
}
