// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Provider-owned, read-only process transport. Callers select trust and basis.
use crate::framework::Registry;
use crate::framework::openwarrant_receipt::{
    ReceiptError, SealedVerificationFiles, VerificationFiles, read_verification_key, verify_files,
    verify_sealed_files,
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
        #[arg(
            long,
            required_unless_present = "sealed_inputs",
            conflicts_with = "sealed_inputs"
        )]
        public_key: Option<PathBuf>,
        /// Expected checked DispatchBinding JSON, not extracted from receipt.
        #[arg(long)]
        binding: PathBuf,
        /// Expected PlanSpec JSON, not extracted from receipt.
        #[arg(
            long,
            required_unless_present = "sealed_inputs",
            conflicts_with = "sealed_inputs"
        )]
        plan: Option<PathBuf>,
        #[arg(
            long,
            required_unless_present = "sealed_inputs",
            conflicts_with = "sealed_inputs"
        )]
        job: Option<PathBuf>,
        /// Retained exact producer binary; no fallback to the current binary.
        #[arg(
            long,
            required_unless_present = "sealed_inputs",
            conflicts_with = "sealed_inputs"
        )]
        producer_executable: Option<PathBuf>,
        /// Explicit v2 manifest of independently captured kernel-sealed inputs.
        /// Ordinary file transport keeps v1 and cannot be mixed with this mode.
        #[arg(long)]
        sealed_inputs: Option<PathBuf>,
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
        sealed_inputs,
    } = cmd;
    let schema = if sealed_inputs.is_some() {
        "blut/openwarrant-verification/v2"
    } else {
        "blut/openwarrant-verification/v1"
    };
    let result = if let Some(inputs) = sealed_inputs {
        verify_sealed_files(
            &SealedVerificationFiles {
                receipt,
                binding,
                inputs,
            },
            reg,
        )
    } else {
        // Clap requires every ordinary input, but preserve a typed refusal for
        // programmatic construction rather than panicking on absent paths.
        match (public_key, plan, job, producer_executable) {
            (Some(public_key), Some(plan), Some(job), Some(producer_executable)) => {
                read_verification_key(&public_key).and_then(|key| {
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
                })
            }
            _ => Err(ReceiptError::Rejected(
                "ordinary native inputs incomplete".into(),
            )),
        }
    };
    let report = match &result {
        Ok(native) => serde_json::json!({
            "schema": schema,
            "status": "validated", "assurance": "not-established", "native": native,
        }),
        Err(error) => serde_json::json!({
            "schema": schema,
            "status": match error {ReceiptError::Unavailable(_) => "unavailable", ReceiptError::Rejected(_) => "rejected"},
            "assurance": "not-established", "reason": error.to_string(),
        }),
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &report)?;
    writeln!(stdout)?;
    result.map(|_| ()).map_err(Into::into)
}
