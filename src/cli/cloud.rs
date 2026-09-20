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
        /// Object store: a local directory (created if absent), or a URL.
        ///
        /// `s3://bucket/prefix` reaches S3 and any S3-compatible endpoint
        /// (MinIO, R2, GCS via its S3 gateway) when built with `--features s3`.
        /// Credentials and endpoint come from the environment -- the same
        /// variables the AWS CLI reads -- never from a flag, because an
        /// argument lands in shell history, the process table, and any log that
        /// records argv.
        #[arg(long, default_value = "/tmp/blut-cloud-smoke")]
        store: String,
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
    use crate::cloud::submitter::{CloudPoll, CloudSubmitSpec, CloudSubmitter};
    use crate::cloud::worker::run_one;
    use crate::framework::artifact::ContentHash;
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
            let blob_store = open_smoke_store(&store)?;
            let queue = Arc::new(MemQueue::new());

            // Produce the input artifact on disk (the submitter's src_root).
            let src_root = tempfile::tempdir()?;
            let in_path = src_root.path().join("in.txt");
            std::fs::write(&in_path, input.as_bytes())?;
            let art = SmokeText {
                content_hash: ContentHash::hash_file(&in_path).context("hash input")?,
                path: in_path,
            };
            let erased = ErasedArtifact::from_typed(&art).context("erase input")?;

            let submitter = CloudSubmitter::new(blob_store.clone(), queue.clone(), reg.clone());
            eprintln!("submitting p2p-echo to the cloud queue (store: {store})…");
            let handle = submitter
                .submit(CloudSubmitSpec {
                    // Fixed id: re-running against the same --store is idempotent
                    // (blobs are content-addressed; the MemQueue is fresh each run).
                    job_id: "cloud-smoke-1".into(),
                    tenant: crate::tenant::Tenant::default(),
                    stage_name: SMOKE_STAGE.into(),
                    invocation_key: crate::framework::InvocationKey::from_digest(
                        ContentHash::of_bytes(b"cloud-smoke-1"),
                    ),
                    input: erased,
                    src_root: src_root.path().to_path_buf(),
                    args: serde_json::json!({}),
                    expected_content_id: None,
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
                &blob_store,
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
                    println!("✔ cloud round-trip verified over {store}; output:");
                    println!("{body}");
                    Ok(())
                }
                CloudPoll::Failed(m) => Err(anyhow!("cloud job failed: {m}")),
                CloudPoll::Pending => Err(anyhow!("job still pending after the worker ran")),
                CloudPoll::Cancelled => Err(anyhow!("job cancelled")),
                CloudPoll::TimedOut => Err(anyhow!("job timed out")),
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
                assert_eq!(store, "/tmp/x");
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
                store: dir.path().display().to_string(),
                input: "hello cli".to_string(),
            },
        )
        .await
        .expect("cloud smoke handler must round-trip");
    }
}

/// Open the smoke store from `--store`: a URL when it has a scheme, otherwise a
/// local directory (created if absent, so the default path keeps working).
///
/// A bare Windows-style path like `C:\\store` is NOT a URL; the scheme check
/// requires more than one character before the colon so a drive letter cannot
/// be mistaken for one.
#[cfg(feature = "cloud")]
fn open_smoke_store(store: &str) -> Result<crate::framework::object_store::ObjectStore> {
    use crate::framework::object_store::ObjectStore;
    if looks_like_url(store) {
        #[cfg(feature = "s3")]
        {
            return ObjectStore::from_url(store, "cloud")
                .with_context(|| format!("open object store {store}"));
        }
        #[cfg(not(feature = "s3"))]
        {
            return Err(anyhow!(
                "{store} is a URL, but this binary was built without the `s3` feature; \
                 rebuild with --features s3, or pass a local directory"
            ));
        }
    }
    let path = std::path::Path::new(store);
    std::fs::create_dir_all(path)
        .with_context(|| format!("create object-store dir {}", path.display()))?;
    ObjectStore::local_provider(path, "cloud").context("open object store")
}

/// Does this look like a URL rather than a path? Requires a scheme of at least
/// two characters, so `C:\\store` stays a path.
#[cfg(feature = "cloud")]
fn looks_like_url(s: &str) -> bool {
    match s.split_once("://") {
        Some((scheme, _)) => {
            scheme.len() > 1
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
        }
        None => false,
    }
}

#[cfg(all(test, feature = "cloud"))]
mod cloud_store_arg_tests {
    use super::looks_like_url;

    #[test]
    fn urls_are_urls_and_paths_are_paths() {
        assert!(looks_like_url("s3://bucket/prefix"));
        assert!(looks_like_url("file:///tmp/store"));
        assert!(!looks_like_url("/tmp/blut-cloud-smoke"));
        assert!(!looks_like_url("relative/dir"));
    }

    #[test]
    fn a_windows_drive_letter_is_not_a_scheme() {
        // One-character "scheme" is a drive letter, not a URL.
        assert!(!looks_like_url("C://store"));
    }
}
