// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut cloud` — the cloud compute queue (ADR 0082 / 0067 T3.1).
//!
//! Mounted as a child of the `cli` module and glob-imported back (the
//! `cli/p2p.rs` pattern), so `super::*` resolves the shared imports.

use super::*;

/// `blut cloud` subcommands. Behind the `cloud` feature.
#[cfg(feature = "cloud")]
#[derive(Subcommand, Debug)]
pub(super) enum CloudCommand {
    /// Turnkey end-to-end smoke: submit the built-in `p2p-echo` stage through the
    /// cloud queue over an object store, run one worker, and print the
    /// content-verified output. Proves the whole data plane with NO cloud account.
    /// Point `--store` at a durable local or network-mounted directory.
    Smoke {
        /// Object-store root directory (local filesystem; created if absent).
        #[arg(long, default_value = "/tmp/blut-cloud-smoke")]
        store: std::path::PathBuf,
        /// Text payload — `p2p-echo` returns it uppercased.
        #[arg(long, default_value = "hello cloud")]
        input: String,
    },
}

/// `blut cloud` — the cloud compute queue (ADR 0082 / 0067 T3.1).
#[cfg(feature = "cloud")]
pub(super) async fn run_cloud_cmd(
    mut reg: crate::framework::Registry,
    cmd: CloudCommand,
) -> Result<()> {
    use crate::cloud::queue::MemQueue;
    use crate::cloud::store::ObjStore;
    use crate::cloud::submitter::{CloudPoll, CloudSubmitSpec, CloudSubmitter};
    use crate::cloud::worker::run_one;
    use crate::framework::artifact::{Artifact, ContentHash};
    use crate::framework::stage::ErasedArtifact;
    use crate::p2p::dispatch::DefaultDispatchPolicy;
    use crate::p2p::smoke::{self, SMOKE_STAGE, SmokeText};
    use crate::p2p::task::ResourceRequest;
    use crate::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};
    use std::sync::Arc;

    // Ship the built-in p2p-echo smoke stage so both submit + worker resolve it.
    smoke::register(&mut reg);
    let reg = Arc::new(reg);

    match cmd {
        CloudCommand::Smoke { store, input } => {
            std::fs::create_dir_all(&store)
                .with_context(|| format!("create object-store dir {}", store.display()))?;
            let blob_store = Arc::new(ObjStore::local(&store).context("open object store")?);
            let queue = Arc::new(MemQueue::new());

            // Produce the input artifact on disk (the submitter's src_root).
            let src_root = tempfile::tempdir()?;
            let in_path = src_root.path().join("in.txt");
            std::fs::write(&in_path, input.as_bytes())?;
            let art = SmokeText {
                content_hash: ContentHash::hash_file(&in_path).context("hash input")?,
                path: in_path,
            };
            let input_hash = art.content_hash();
            let erased = ErasedArtifact::from_typed(&art).context("erase input")?;
            let expected = smoke::expected_echo_hash(&input);

            let submitter = CloudSubmitter::new(blob_store.clone(), queue.clone(), reg.clone());
            eprintln!(
                "submitting p2p-echo to the cloud queue (store: {})…",
                store.display()
            );
            let handle = submitter
                .submit(CloudSubmitSpec {
                    // Fixed id: re-running against the same --store is idempotent
                    // (blobs are content-addressed; the MemQueue is fresh each run).
                    job_id: "cloud-smoke-1".into(),
                    stage_name: SMOKE_STAGE.into(),
                    input: erased,
                    src_root: src_root.path().to_path_buf(),
                    args: serde_json::json!({}),
                    input_hash,
                    expected_output_hash: expected,
                    data_class: DataClass::Public,
                    resources: ResourceRequest::default(),
                    priority: 0,
                    timeout_secs: 30,
                })
                .await
                .context("submit cloud job")?;

            // Run one worker pass (Registered trust — the clinical hard-block posture).
            // One matrix instance, shared by the policy and the worker, so a future
            // customization can't let them diverge.
            let matrix = DispatchMatrix::default();
            let policy = DefaultDispatchPolicy::new(matrix.clone());
            let work_root = tempfile::tempdir()?;
            eprintln!("running a cloud worker…");
            run_one(
                blob_store.as_ref(),
                queue.as_ref(),
                reg.as_ref(),
                &policy,
                &matrix,
                "cloud-smoke-worker",
                TrustLevel::Registered,
                30,
                work_root.path(),
                None,
            )
            .await
            .context("cloud worker")?;

            let out_dir = tempfile::tempdir()?;
            match handle
                .poll(out_dir.path())
                .await
                .context("poll cloud job")?
            {
                CloudPoll::Succeeded(output) => {
                    let out: SmokeText = output.into_typed().context("decode output")?;
                    let body = std::fs::read_to_string(&out.path)?;
                    println!(
                        "✔ cloud round-trip verified over {}; output:",
                        store.display()
                    );
                    println!("{body}");
                    Ok(())
                }
                CloudPoll::Failed(m) => Err(anyhow!("cloud job failed: {m}")),
                CloudPoll::Pending => Err(anyhow!("job still pending after the worker ran")),
                CloudPoll::Cancelled => Err(anyhow!("job cancelled")),
                CloudPoll::Unknown => Err(anyhow!("job unknown")),
            }
        }
    }
}

#[cfg(all(test, feature = "cloud"))]
mod cloud_cli_tests {
    use super::{Cli, CloudCommand, Command, run_cloud_cmd};
    use clap::Parser;

    #[test]
    fn cloud_smoke_parses_with_defaults() {
        match Cli::try_parse_from(["blut", "cloud", "smoke"])
            .expect("parse")
            .command
        {
            Some(Command::Cloud {
                cmd: CloudCommand::Smoke { input, .. },
            }) => {
                assert_eq!(input, "hello cloud");
            }
            other => panic!("expected cloud smoke, got {other:?}"),
        }
    }

    #[test]
    fn cloud_smoke_parses_overrides() {
        match Cli::try_parse_from([
            "blut", "cloud", "smoke", "--input", "hi", "--store", "/tmp/x",
        ])
        .expect("parse")
        .command
        {
            Some(Command::Cloud {
                cmd: CloudCommand::Smoke { input, store },
            }) => {
                assert_eq!(input, "hi");
                assert_eq!(store, std::path::PathBuf::from("/tmp/x"));
            }
            other => panic!("got {other:?}"),
        }
    }

    // Drives the REAL `blut cloud smoke` handler end-to-end (submit → worker →
    // poll → content-verify) over a local-fs object store — the CLI proof.
    #[tokio::test]
    async fn cloud_smoke_handler_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let reg = crate::framework::Registry::new();
        run_cloud_cmd(
            reg,
            CloudCommand::Smoke {
                store: dir.path().to_path_buf(),
                input: "hello cli".to_string(),
            },
        )
        .await
        .expect("cloud smoke handler must round-trip");
    }
}
