// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut-web` binary — reached as `blut web` via the engine's external
//! subcommand dispatch (ADR 0083). Custody: binding beyond loopback REQUIRES
//! the ADR-0095 token store; without one the server refuses (fail-closed),
//! never an open box on a network interface.

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "blut-web", about = "BLUT web sidecar (read-only views)")]
struct Args {
    /// Bind address. Non-loopback requires --tokens (fail-closed).
    #[arg(long, default_value = "127.0.0.1:7838")]
    bind: std::net::SocketAddr,
    /// Path to the ADR-0095 token store (`web-tokens.toml`). Defaults to
    /// `~/.blut/web-tokens.toml` when present.
    #[arg(long)]
    tokens: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();

    let token_path = args.tokens.clone().or_else(|| {
        dirs_path()
            .map(|d| d.join("web-tokens.toml"))
            .filter(|p| p.exists())
    });
    let tokens = match token_path {
        Some(p) => {
            let text = std::fs::read_to_string(&p)
                .with_context(|| format!("read token store {}", p.display()))?;
            let store = blut::rbac::TokenStore::parse(&text)
                .map_err(|e| anyhow::anyhow!("token store {}: {e}", p.display()))?;
            tracing::info!("token auth ENABLED ({} )", p.display());
            Some(std::sync::Arc::new(store))
        }
        None => {
            if !args.bind.ip().is_loopback() {
                bail!(
                    "refusing to bind {} without a token store — non-loopback \
                     requires --tokens (ADR 0095 fail-closed custody)",
                    args.bind
                );
            }
            tracing::warn!(
                "no token store — serving UNAUTHENTICATED on loopback only \
                 (create ~/.blut/web-tokens.toml to enable auth)"
            );
            None
        }
    };

    let app = blut_web::build_router(blut_web::AppState {
        tokens,
        lineage_path: None,
    });
    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("bind {}", args.bind))?;
    tracing::info!("blut-web serving read-only views on http://{}", args.bind);
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}

fn dirs_path() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".blut"))
}
