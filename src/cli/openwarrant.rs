// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Provider-owned, read-only process transport. Callers select trust and basis.
use crate::framework::Registry;
use crate::framework::openwarrant_receipt::{
    ReceiptError, VerificationFiles, read_verification_key, verify_files,
};
use clap::Subcommand;
use std::{io::Write, path::PathBuf};

#[derive(Debug, Subcommand)]
pub(super) enum ReceiptCommand {
    /// Emit blut/openwarrant-verification/v1 JSON. A validated receipt is not
    /// OpenWarrant assurance. Unavailable/rejected inputs exit unsuccessfully
    /// with distinct status values. All paths and trust come from the caller.
    Verify {
        #[arg(long)]
        receipt: PathBuf,
        /// Trusted Ed25519 PUBLIC key: exactly 32 raw bytes, not a private key.
        #[arg(long)]
        public_key: PathBuf,
        /// Expected checked DispatchBinding JSON, not extracted from receipt.
        #[arg(long)]
        binding: PathBuf,
        /// Expected PlanSpec JSON, not extracted from receipt.
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        job: PathBuf,
        /// Retained exact producer binary; no fallback to the current binary.
        #[arg(long)]
        producer_executable: PathBuf,
    },
}

pub(super) fn run(reg: &Registry, cmd: ReceiptCommand) -> anyhow::Result<()> {
    let ReceiptCommand::Verify {
        receipt,
        public_key,
        binding,
        plan,
        job,
        producer_executable,
    } = cmd;
    let result = read_verification_key(&public_key).and_then(|key| {
        verify_files(
            &VerificationFiles {
                receipt,
                binding,
                plan,
                job,
                producer_executable,
            },
            &key,
            reg,
        )
    });
    let report = match &result {
        Ok(native) => serde_json::json!({
            "schema": "blut/openwarrant-verification/v1",
            "status": "validated", "assurance": "not-established", "native": native,
        }),
        Err(error) => serde_json::json!({
            "schema": "blut/openwarrant-verification/v1",
            "status": match error {ReceiptError::Unavailable(_) => "unavailable", ReceiptError::Rejected(_) => "rejected"},
            "assurance": "not-established", "reason": error.to_string(),
        }),
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &report)?;
    writeln!(stdout)?;
    result.map(|_| ()).map_err(Into::into)
}
