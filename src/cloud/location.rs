// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Where a cloud queue lives.
//!
//! One string names it for every process that uses it: the submitter reads it
//! from `BLUT_CLOUD_QUEUE`, each worker from `blut cloud worker --queue`. It is
//! `s3://bucket[/prefix]` (a build with the `s3` feature; credentials and the
//! endpoint from the `AWS_*` environment only), `file:///abs/dir`, or a plain
//! directory. Input and output bundles go to `<prefix>/cloud/v1/dispatch-bundles/`
//! and the queue to `<prefix>/cloud/queue/v1/`, so both ends address the same
//! objects without further configuration.

use std::sync::Arc;

use super::CloudError;
use super::object_queue::{ObjectQueue, ObjectQueueOptions};
use crate::framework::object_store::ObjectStore;

/// The prefix under the location's own path where cloud objects go. The same
/// one `blut cloud smoke` uses.
const CLOUD_PREFIX: &str = "cloud";

/// An opened queue location: the bundle store and the job queue on it.
#[derive(Clone, Debug)]
pub struct CloudLocation {
    store: ObjectStore,
    queue: Arc<ObjectQueue>,
    spec: String,
}

impl CloudLocation {
    /// Open `spec`. A directory (bare or `file://`) is created if absent and
    /// made absolute, so a relative path means the same place to every
    /// process that is handed [`spec`](Self::spec) afterwards.
    pub fn open(spec: &str, options: ObjectQueueOptions) -> Result<Self, CloudError> {
        let spec = spec.trim();
        if spec.is_empty() {
            return Err(CloudError::Dispatch(
                "the cloud queue location is empty".into(),
            ));
        }
        if looks_like_url(spec) {
            return open_url(spec, options);
        }
        let dir = std::path::absolute(spec).map_err(|error| {
            CloudError::Dispatch(format!("cloud queue directory {spec}: {error}"))
        })?;
        std::fs::create_dir_all(&dir).map_err(|error| {
            CloudError::Dispatch(format!(
                "create cloud queue directory {}: {error}",
                dir.display()
            ))
        })?;
        let provider =
            object_store::local::LocalFileSystem::new_with_prefix(&dir).map_err(|error| {
                CloudError::Dispatch(format!("open cloud queue directory: {error}"))
            })?;
        let provider: Arc<dyn object_store::ObjectStore> = Arc::new(provider);
        Ok(Self {
            store: ObjectStore::provider(provider.clone(), CLOUD_PREFIX)?,
            queue: Arc::new(ObjectQueue::new(provider, CLOUD_PREFIX, options)),
            spec: dir.display().to_string(),
        })
    }

    /// Where input and output bundles are stored.
    pub fn store(&self) -> &ObjectStore {
        &self.store
    }

    /// The job queue.
    pub fn queue(&self) -> &Arc<ObjectQueue> {
        &self.queue
    }

    /// The location as opened: an absolute directory or the URL. It carries no
    /// credentials (an `s3://` URL with userinfo is refused), so it can be
    /// recorded with a job for `blut cancel` to reopen.
    pub fn spec(&self) -> &str {
        &self.spec
    }

    /// The location for logs: scheme and bucket of a URL, a directory as is.
    pub fn display(&self) -> String {
        redact(&self.spec)
    }

    /// One cheap request, so a wrong endpoint or rejected credentials fail a
    /// launch before any stage is submitted.
    pub async fn probe(&self) -> Result<(), CloudError> {
        self.queue.probe().await
    }
}

/// Scheme and bucket only. A queue location is printed in job logs, so the
/// path after the bucket stays out of them.
pub fn redact(spec: &str) -> String {
    match spec.split_once("://") {
        Some((scheme, rest)) if scheme != "file" => {
            let host = rest.split('/').next().unwrap_or("");
            let host = host.rsplit('@').next().unwrap_or(host);
            format!("{scheme}://{host}/…")
        }
        _ => spec.to_string(),
    }
}

/// A URL rather than a path: a scheme of at least two characters, so a
/// Windows drive letter (`C://x`) stays a path.
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

#[cfg(feature = "s3")]
fn open_url(spec: &str, options: ObjectQueueOptions) -> Result<CloudLocation, CloudError> {
    if let Some(dir) = spec.strip_prefix("file://").filter(|d| d.starts_with('/')) {
        std::fs::create_dir_all(dir).map_err(|error| {
            CloudError::Dispatch(format!("create cloud queue directory {dir}: {error}"))
        })?;
    }
    let (provider, url_prefix) = crate::framework::object_store::provider_from_url(spec)?;
    let prefix = crate::framework::object_store::join_prefix(&url_prefix, CLOUD_PREFIX);
    Ok(CloudLocation {
        store: ObjectStore::provider(provider.clone(), &prefix)?,
        queue: Arc::new(ObjectQueue::new(provider, &prefix, options)),
        spec: spec.to_string(),
    })
}

#[cfg(not(feature = "s3"))]
fn open_url(spec: &str, _options: ObjectQueueOptions) -> Result<CloudLocation, CloudError> {
    Err(CloudError::Dispatch(format!(
        "{} is a URL, but this binary was built without the `s3` feature; rebuild with \
         it, or name a directory",
        redact(spec)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_directory_is_recorded_absolute() {
        let td = tempfile::tempdir().unwrap();
        let location = CloudLocation::open(
            &td.path().join("q").display().to_string(),
            ObjectQueueOptions::default(),
        )
        .unwrap();
        assert!(std::path::Path::new(location.spec()).is_absolute());
        assert!(td.path().join("q").is_dir());
    }

    #[test]
    fn redaction_keeps_scheme_and_bucket_only() {
        assert_eq!(redact("s3://bucket/secret/path"), "s3://bucket/…");
        assert_eq!(redact("/tmp/q"), "/tmp/q");
        assert!(looks_like_url("s3://b"));
        assert!(!looks_like_url("C://x"));
    }

    #[cfg(not(feature = "s3"))]
    #[test]
    fn a_url_needs_the_s3_feature() {
        let error = CloudLocation::open("s3://bucket/x", ObjectQueueOptions::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("`s3` feature"), "{error}");
    }
}
