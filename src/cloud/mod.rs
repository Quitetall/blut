//! Cloud compute queue (ADR 0067 · T3.1) — "submit a job, get the result back,
//! billed on compute."
//!
//! Architecture: this is **the P2P data plane with an object store in place of
//! QUIC**. The artifact bundle (`crate::p2p::bundle`), the four fail-closed verify
//! gates, the dispatch seam (`DispatchSubmitter`/`DispatchHandle`), and the trust
//! matrix (`crate::p2p::trust`) are all transport-agnostic and reused verbatim;
//! only the blob transport changes — `crate::p2p::transport::{send_blob,recv_blob}`
//! over QUIC becomes [`store::BlobStore::put_blob`]/`get_blob` over an object store.
//!
//! v1 builds on the `object_store` crate. The `aws` feature compiles the local
//! filesystem + any S3-compatible store (AWS S3, Cloudflare R2, MinIO) behind one
//! trait, so the whole queue is dev-testable on the local filesystem with no cloud
//! account; a real provider is a config swap (GCS/Azure are one more cargo feature). Clinical/PHI EEG is hard-blocked
//! from cloud in v1: cloud workers are capped at `Registered` trust, so the reused
//! `DispatchMatrix` refuses `DataClass::Restricted` to any cloud worker.

pub mod store;

/// Errors from the cloud queue layer. Grows as the sibling modules (queue,
/// submitter, worker) land; v1 starts with the blob transport.
#[derive(Debug)]
pub enum CloudError {
    /// Object-store transport failure (put/get/head).
    Store(String),
}

impl std::fmt::Display for CloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudError::Store(m) => write!(f, "cloud object store: {m}"),
        }
    }
}

impl std::error::Error for CloudError {}
