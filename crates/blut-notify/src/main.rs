// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam

use anyhow::{Result, anyhow};
use blut_notify::{NotificationEnvelope, NotifySink, SinkBoundary, deliver};
use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(name = "blut-notify", about = "BLUT notification custody boundary")]
struct Cli {
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
    // Payloads arrive on stdin, never argv, so PHI cannot leak through the
    // process list. Tenant's custom Deserialize validates the wire identity.
    let envelope: NotificationEnvelope = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|_| anyhow!("invalid notification envelope"))?;
    let mut sink = StdoutSink(cli.boundary.into());
    deliver(&mut sink, &envelope).map_err(|error| anyhow!(error))
}
