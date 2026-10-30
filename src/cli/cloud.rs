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
        /// Default: a fresh private temporary directory, removed when the
        /// smoke ends (it was the fixed, shared `/tmp/blut-cloud-smoke`).
        ///
        /// `s3://bucket/prefix` reaches AWS S3 or an S3-compatible endpoint
        /// when built with `--features s3`. The endpoint must support
        /// conditional writes (`If-None-Match: *`): Cloudflare R2 and MinIO do
        /// (both checked), Backblaze B2 does not.
        /// Credentials and endpoint come from the environment -- the same
        /// variables the AWS CLI reads -- never from a flag, because an
        /// argument lands in shell history, the process table, and any log that
        /// records argv.
        #[arg(long)]
        store: Option<String>,
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
            // Held until the smoke returns: a default store is removed then.
            let (store, _scratch_store) = smoke_store_location(store)?;
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
                .map_err(|e| flat_error("submit cloud job", e))?;

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
            .map_err(|e| flat_error("cloud worker", e))?;

            let out_dir = tempfile::tempdir()?;
            match handle
                .poll(out_dir.path())
                .await
                .map_err(|e| flat_error("poll cloud job", e))?
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
                cmd: CloudCommand::Smoke { input, store },
            }) => {
                assert_eq!(input, "hello cloud");
                assert_eq!(store, None, "no fixed shared default path");
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
                assert_eq!(store.as_deref(), Some("/tmp/x"));
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
                store: Some(dir.path().display().to_string()),
                input: "hello cli".to_string(),
            },
        )
        .await
        .expect("cloud smoke handler must round-trip");
    }

    /// E2E follow-up: the default store was the fixed shared path
    /// `/tmp/blut-cloud-smoke`. It is a fresh private directory per run,
    /// removed when the smoke ends, and the handler round-trips over it.
    #[tokio::test]
    async fn the_default_smoke_store_is_private_and_removed() {
        let (a, guard_a) = super::smoke_store_location(None).unwrap();
        let (b, guard_b) = super::smoke_store_location(None).unwrap();
        assert_ne!(a, b, "each run gets its own store");
        assert_ne!(a, "/tmp/blut-cloud-smoke");
        assert!(std::path::Path::new(&a).is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&a).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "private to the user: {mode:o}");
        }
        drop((guard_a, guard_b));
        assert!(!std::path::Path::new(&a).exists(), "removed afterwards");
        assert!(!std::path::Path::new(&b).exists());
        // An explicit store is used as given and never removed.
        let (given, guard) = super::smoke_store_location(Some("/x/y".into())).unwrap();
        assert_eq!((given.as_str(), guard.is_none()), ("/x/y", true));

        run_cloud_cmd(
            crate::framework::Registry::new(),
            CloudCommand::Smoke {
                store: None,
                input: "hello default".to_string(),
            },
        )
        .await
        .expect("cloud smoke over the default store");
    }
}

/// One line per failure. The cloud and object-store errors each embed their
/// source's message in their own, so printing the error chain repeated the
/// provider's response (an S3 XML body appeared five times).
#[cfg(feature = "cloud")]
fn flat_error(what: &str, error: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{what}: {error}")
}

/// Where `cloud smoke` keeps its store: `--store` when given, else a fresh
/// private temporary directory (`blut-cloud-smoke-*`, mode 0700 on Unix),
/// whose guard the caller holds until the smoke ends and which is removed
/// then. The default used to be the fixed `/tmp/blut-cloud-smoke`, shared by
/// every user and run on the machine.
#[cfg(feature = "cloud")]
fn smoke_store_location(store: Option<String>) -> Result<(String, Option<tempfile::TempDir>)> {
    if let Some(store) = store {
        return Ok((store, None));
    }
    let mut builder = tempfile::Builder::new();
    builder.prefix("blut-cloud-smoke-");
    // tempfile creates a directory with the process umask (0755 here); the
    // store holds the job's input and output, so other users get no access.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    let dir = builder
        .tempdir()
        .context("create a temporary smoke store")?;
    eprintln!(
        "store: {} (temporary; removed when the smoke ends; pass --store to keep one)",
        dir.path().display()
    );
    Ok((dir.path().display().to_string(), Some(dir)))
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
            // `--store` promises a directory "created if absent"; a `file://`
            // URL to a missing directory failed with "Unable to canonicalize
            // filesystem root" while the same path given bare worked.
            if let Some(dir) = store.strip_prefix("file://").filter(|d| d.starts_with('/')) {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("create object-store dir {dir}"))?;
            }
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

#[cfg(all(test, feature = "s3"))]
mod cloud_file_url_tests {
    #[test]
    fn a_file_url_store_is_created_like_a_bare_path() {
        // E2E regression: `blut cloud smoke --store file:///new/dir` failed
        // with "Unable to canonicalize filesystem root".
        let td = tempfile::tempdir().unwrap();
        let dir = td.path().join("fresh");
        let url = format!("file://{}", dir.display());
        super::open_smoke_store(&url).expect("file:// store opens");
        assert!(dir.is_dir());
    }
}

#[cfg(all(test, feature = "cloud"))]
mod flat_error_tests {
    #[test]
    fn a_wrapped_error_prints_its_message_once() {
        #[derive(Debug)]
        struct Leaf;
        impl std::fmt::Display for Leaf {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("<Error>NoSuchBucket</Error>")
            }
        }
        impl std::error::Error for Leaf {}
        #[derive(Debug)]
        struct Wrapper(Leaf);
        impl std::fmt::Display for Wrapper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "put failed: {}", self.0)
            }
        }
        impl std::error::Error for Wrapper {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let shown = format!("{:?}", super::flat_error("submit cloud job", Wrapper(Leaf)));
        assert_eq!(shown.matches("NoSuchBucket").count(), 1, "{shown}");
    }
}
