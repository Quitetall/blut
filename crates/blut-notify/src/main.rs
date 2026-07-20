// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam

use anyhow::{Result, anyhow};
use blut_notify::{NotificationEnvelope, NotifySink, SinkBoundary, deliver};
use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(name = "blut-notify", about = "BLUT notification custody boundary")]
struct Cli {
    /// Notification rules + sink definitions. When present, tail status/SLA
    /// sources as a daemon; without it, retain the one-envelope stdin mode.
    #[arg(long)]
    config: Option<std::path::PathBuf>,
    /// Process available lines once and exit (daemon/config mode).
    #[arg(long, default_value_t = false)]
    once: bool,
    /// Poll interval in seconds (daemon/config mode).
    #[arg(long, default_value_t = 5)]
    interval: u64,
    #[arg(long)]
    jobs_dir: Option<std::path::PathBuf>,
    #[arg(long)]
    sla: Option<std::path::PathBuf>,
    #[arg(long)]
    cursor: Option<std::path::PathBuf>,
    /// Custody declaration for one-envelope stdin mode.
    #[arg(long, value_enum, default_value_t = BoundaryArg::OffBox)]
    boundary: BoundaryArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BoundaryArg {
    Local,
    OffBox,
}

impl From<BoundaryArg> for SinkBoundary {
    fn from(value: BoundaryArg) -> Self {
        match value {
            BoundaryArg::Local => Self::Local,
            BoundaryArg::OffBox => Self::OffBox,
        }
    }
}

struct StdoutSink(SinkBoundary);

impl NotifySink for StdoutSink {
    fn boundary(&self) -> SinkBoundary {
        self.0
    }

    fn send(&mut self, envelope: &NotificationEnvelope) -> Result<(), String> {
        let body = serde_json::to_string(envelope).map_err(|error| error.to_string())?;
        println!("{body}");
        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(config_path) = cli.config {
        let config = blut_notify::config::NotifyConfig::load(&config_path)
            .map_err(|error| anyhow!("load {}: {error}", config_path.display()))?;
        let jobs_dir = cli.jobs_dir.unwrap_or_else(default_jobs_dir);
        let sla = cli.sla.unwrap_or_else(|| dot_blut().join("sla.jsonl"));
        let cursor = cli
            .cursor
            .unwrap_or_else(|| dot_blut().join("notify-cursors.json"));
        let mut notifier =
            blut_notify::tailer::Notifier::open(config, cursor).map_err(|error| anyhow!(error))?;
        loop {
            let report = notifier
                .run_once(&jobs_dir, &sla)
                .map_err(|error| anyhow!(error))?;
            if report.lines_read > 0 || cli.once {
                eprintln!(
                    "blut-notify: lines={} matched={} delivered={} custody_refused={} malformed={}",
                    report.lines_read,
                    report.matched,
                    report.delivered,
                    report.custody_refused,
                    report.malformed
                );
            }
            if cli.once {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_secs(cli.interval.max(1)));
        }
    }

    // Payloads arrive on stdin, never argv, so PHI cannot leak through the
    // process list. Tenant's custom Deserialize validates the wire identity.
    let envelope: NotificationEnvelope = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|_| anyhow!("invalid notification envelope"))?;
    let mut sink = StdoutSink(cli.boundary.into());
    deliver(&mut sink, &envelope).map_err(|error| anyhow!(error))
}

fn dot_blut() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
        .join(".blut")
}

fn default_jobs_dir() -> std::path::PathBuf {
    std::env::var_os("LAMU_TRAIN_JOBS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_default()
                .join(".local/share/lamu/train-jobs")
        })
}
