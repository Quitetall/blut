// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Content-addressed chunk store for blob dedup + resume. The data plane is
//! specified by ADR 0079 (§ "What stays / data plane"); this module is the
//! ADR 0067 **T4.1** slice that implements it.
//!
//! Today `transport::send_blob` re-sends every byte of a bundle on every
//! transfer. A bundle is already split into `CHUNK_MAX` pieces on the wire —
//! this module makes those pieces **content-addressed** so a receiver can:
//!
//!   - **dedup** — skip any chunk it already holds (identical bytes ⇒ identical
//!     hash ⇒ already on disk), and
//!   - **resume** — after a dropped connection, refetch only the chunks it is
//!     still missing.
//!
//! The protocol (wired into `transport` in a follow-up): the sender computes a
//! [`ChunkIndex`] (ordered chunk hashes + total length) and sends it first; the
//! receiver replies with the subset it is *missing* (via [`ChunkStore::missing`]);
//! the sender streams only those chunks; the receiver [`store_chunk`]s each
//! (verifying the hash) and [`reassemble`]s the blob once complete.
//!
//! Integrity is not assumed: `store_chunk` REJECTS bytes that don't hash to the
//! claimed key, so a peer cannot inject wrong content under a good hash.
//!
//! [`store_chunk`]: ChunkStore::store_chunk
//! [`reassemble`]: ChunkStore::reassemble

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::framework::artifact::ContentHash;
use crate::framework::object_store::{BlobStore, FsBlobStore};
use crate::p2p::transport::{CHUNK_MAX, MAX_BLOB_SIZE};

/// An ordered manifest of a blob's content-addressed chunks. Serialized and
/// sent before the chunks themselves so the receiver knows what to ask for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkIndex {
    /// Total plaintext length of the blob, in bytes. Checked on reassembly.
    pub total_len: u64,
    /// Hash of each `CHUNK_MAX`-sized piece, in order. The last may be shorter.
    pub chunk_hashes: Vec<ContentHash>,
}

impl ChunkIndex {
    /// Split `blob` into `CHUNK_MAX` pieces and hash each. Deterministic: the
    /// same bytes always produce the same index (so two senders of identical
    /// content dedup against each other on the receiver).
    pub fn of(blob: &[u8]) -> Self {
        let chunk_hashes = if blob.is_empty() {
            Vec::new()
        } else {
            blob.chunks(CHUNK_MAX).map(ContentHash::of_bytes).collect()
        };
        Self {
            total_len: blob.len() as u64,
            chunk_hashes,
        }
    }

    /// Number of chunks (0 for an empty blob).
    pub fn len(&self) -> usize {
        self.chunk_hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunk_hashes.is_empty()
    }

    /// Reject an implausible index before allocating for it: a `total_len`
    /// above the transfer cap, or a chunk count inconsistent with `total_len`.
    /// Bounds are inclusive of the ragged final chunk.
    pub fn validate(&self) -> Result<(), ChunkError> {
        if self.total_len > MAX_BLOB_SIZE {
            return Err(ChunkError::TooLarge {
                total_len: self.total_len,
            });
        }
        // Expected chunk count = ceil(total_len / CHUNK_MAX); empty ⇒ 0.
        let expected = if self.total_len == 0 {
            0
        } else {
            self.total_len.div_ceil(CHUNK_MAX as u64) as usize
        };
        if self.chunk_hashes.len() != expected {
            return Err(ChunkError::ArityMismatch {
                declared_len: self.total_len,
                got_chunks: self.chunk_hashes.len(),
                expected_chunks: expected,
            });
        }
        Ok(())
    }
}

/// A content-addressed store of blob chunks on the local filesystem. Backed by
/// [`FsBlobStore`], so chunks survive across transfers and jobs — that is what
/// makes dedup work: a chunk seen in any earlier bundle is already present.
#[derive(Debug, Clone)]
pub struct ChunkStore {
    inner: FsBlobStore,
}

impl ChunkStore {
    /// Open (or lazily create) a chunk store rooted at `dir`.
    pub fn new(dir: PathBuf) -> Self {
        Self {
            inner: FsBlobStore::new(dir),
        }
    }

    /// Store every chunk of `blob` and return its index. Idempotent — chunks
    /// already present are re-written with identical bytes (content-addressed).
    /// Used by the SENDER to publish a blob into its own store before offering
    /// the index.
    pub fn put_blob(&self, blob: &[u8]) -> Result<ChunkIndex, ChunkError> {
        let index = ChunkIndex::of(blob);
        // Fail fast on the SENDER: don't publish an over-cap blob that no
        // conforming receiver would accept.
        index.validate()?;
        for (hash, chunk) in index.chunk_hashes.iter().zip(blob.chunks(CHUNK_MAX)) {
            self.inner.put(*hash, chunk).map_err(ChunkError::Io)?;
        }
        Ok(index)
    }

    /// The positions in `index` whose chunks are NOT yet in this store — the
    /// exact set the receiver must request. Empty ⇒ the blob is fully local
    /// (a pure cache hit; nothing to transfer).
    pub fn missing(&self, index: &ChunkIndex) -> Result<Vec<usize>, ChunkError> {
        let mut out = Vec::new();
        for (i, hash) in index.chunk_hashes.iter().enumerate() {
            if !self.inner.head(*hash).map_err(ChunkError::Io)? {
                out.push(i);
            }
        }
        Ok(out)
    }

    /// Store one received chunk, VERIFYING it hashes to `claimed`. A mismatch
    /// is rejected (a peer cannot smuggle wrong bytes under a valid hash) and
    /// nothing is written.
    pub fn store_chunk(&self, claimed: ContentHash, bytes: &[u8]) -> Result<(), ChunkError> {
        if bytes.len() > CHUNK_MAX {
            return Err(ChunkError::ChunkTooLarge { got: bytes.len() });
        }
        let actual = ContentHash::of_bytes(bytes);
        if actual != claimed {
            return Err(ChunkError::HashMismatch { claimed, actual });
        }
        self.inner.put(claimed, bytes).map_err(ChunkError::Io)
    }

    /// Whether a chunk is present.
    pub fn has(&self, hash: ContentHash) -> Result<bool, ChunkError> {
        self.inner.head(hash).map_err(ChunkError::Io)
    }

    /// Reassemble the full blob from locally-stored chunks. Fails if any chunk
    /// is missing (call [`missing`] first) or the reconstructed length doesn't
    /// match `index.total_len`.
    ///
    /// [`missing`]: ChunkStore::missing
    pub fn reassemble(&self, index: &ChunkIndex) -> Result<Vec<u8>, ChunkError> {
        index.validate()?;
        // Grow to the ACTUAL bytes fetched — never speculatively allocate
        // `total_len`. A peer can craft an arity-valid index (e.g. total_len =
        // MAX_BLOB_SIZE with the matching count of *bogus* hashes) that passes
        // `validate()` but names chunks we don't hold; pre-allocating its
        // declared length would OOM before the first `MissingChunk` fires.
        // Reserving per fetched chunk bounds the allocation to real data.
        let mut out: Vec<u8> = Vec::new();
        for hash in &index.chunk_hashes {
            let chunk = self
                .inner
                .get(*hash)
                .map_err(ChunkError::Io)?
                .ok_or(ChunkError::MissingChunk { hash: *hash })?;
            out.reserve(chunk.len());
            out.extend_from_slice(&chunk);
        }
        if out.len() as u64 != index.total_len {
            return Err(ChunkError::LengthMismatch {
                declared: index.total_len,
                actual: out.len() as u64,
            });
        }
        Ok(out)
    }
}

/// Errors from chunk-store operations.
#[derive(Debug)]
pub enum ChunkError {
    /// Underlying blob-store I/O failed.
    Io(std::io::Error),
    /// Received bytes did not hash to the claimed key (integrity violation).
    HashMismatch {
        claimed: ContentHash,
        actual: ContentHash,
    },
    /// A chunk exceeded `CHUNK_MAX`.
    ChunkTooLarge { got: usize },
    /// A declared index exceeded the transfer cap.
    TooLarge { total_len: u64 },
    /// The chunk count is inconsistent with the declared total length.
    ArityMismatch {
        declared_len: u64,
        got_chunks: usize,
        expected_chunks: usize,
    },
    /// A chunk named by the index is absent locally on reassembly.
    MissingChunk { hash: ContentHash },
    /// Reassembled bytes didn't match the declared total length.
    LengthMismatch { declared: u64, actual: u64 },
}

impl std::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkError::Io(e) => write!(f, "chunk store I/O: {e}"),
            ChunkError::HashMismatch { claimed, actual } => write!(
                f,
                "chunk hash mismatch: claimed {}, got {}",
                claimed.to_hex(),
                actual.to_hex()
            ),
            ChunkError::ChunkTooLarge { got } => {
                write!(f, "chunk too large: {got} > {CHUNK_MAX}")
            }
            ChunkError::TooLarge { total_len } => {
                write!(f, "blob index too large: {total_len} > {MAX_BLOB_SIZE}")
            }
            ChunkError::ArityMismatch {
                declared_len,
                got_chunks,
                expected_chunks,
            } => write!(
                f,
                "chunk arity mismatch: {declared_len} bytes ⇒ expected {expected_chunks} chunks, got {got_chunks}"
            ),
            ChunkError::MissingChunk { hash } => {
                write!(f, "missing chunk {} on reassembly", hash.to_hex())
            }
            ChunkError::LengthMismatch { declared, actual } => {
                write!(f, "reassembled length {actual} ≠ declared {declared}")
            }
        }
    }
}

impl std::error::Error for ChunkError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, ChunkStore) {
        let td = tempfile::tempdir().unwrap();
        let s = ChunkStore::new(td.path().to_path_buf());
        (td, s)
    }

    #[test]
    fn round_trip_multi_chunk_blob() {
        let (_td, s) = store();
        // 2.5 chunks so the last is ragged.
        let blob: Vec<u8> = (0..(CHUNK_MAX * 2 + 123))
            .map(|i| (i % 251) as u8)
            .collect();
        let index = s.put_blob(&blob).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(index.total_len, blob.len() as u64);
        assert!(
            s.missing(&index).unwrap().is_empty(),
            "sender has all chunks"
        );
        assert_eq!(s.reassemble(&index).unwrap(), blob);
    }

    #[test]
    fn empty_blob_has_no_chunks() {
        let (_td, s) = store();
        let index = s.put_blob(b"").unwrap();
        assert!(index.is_empty());
        assert_eq!(index.total_len, 0);
        assert_eq!(s.reassemble(&index).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn dedup_only_missing_chunks_transfer() {
        // Receiver already holds chunk 0 (shared prefix); only 1 & 2 are new.
        let (_td, recv) = store();
        let (_td2, send) = store();
        let shared: Vec<u8> = vec![7u8; CHUNK_MAX];
        let blob = {
            let mut b = shared.clone();
            b.extend(std::iter::repeat_n(9u8, CHUNK_MAX));
            b.extend_from_slice(b"tail");
            b
        };
        let index = send.put_blob(&blob).unwrap();
        // Receiver seeds chunk 0 out-of-band (e.g. a prior bundle).
        recv.store_chunk(index.chunk_hashes[0], &shared).unwrap();

        let missing = recv.missing(&index).unwrap();
        assert_eq!(missing, vec![1, 2], "chunk 0 deduped, only 1+2 needed");

        // Transfer exactly the missing chunks (verified on store).
        for &i in &missing {
            let bytes = send.reassemble_one(&index, i);
            recv.store_chunk(index.chunk_hashes[i], &bytes).unwrap();
        }
        assert!(recv.missing(&index).unwrap().is_empty());
        assert_eq!(recv.reassemble(&index).unwrap(), blob);
    }

    #[test]
    fn resume_after_partial_transfer() {
        let (_td, recv) = store();
        let (_td2, send) = store();
        let blob: Vec<u8> = (0..(CHUNK_MAX * 3)).map(|i| (i % 200) as u8).collect();
        let index = send.put_blob(&blob).unwrap();

        // Simulate a drop after only chunk 0 arrived.
        recv.store_chunk(index.chunk_hashes[0], &send.reassemble_one(&index, 0))
            .unwrap();
        // Reconnect: only 1 & 2 remain.
        let remaining = recv.missing(&index).unwrap();
        assert_eq!(remaining, vec![1, 2]);
        for &i in &remaining {
            recv.store_chunk(index.chunk_hashes[i], &send.reassemble_one(&index, i))
                .unwrap();
        }
        assert_eq!(recv.reassemble(&index).unwrap(), blob);
    }

    #[test]
    fn corrupt_chunk_is_rejected() {
        let (_td, s) = store();
        let good = vec![1u8; 4096];
        let hash = ContentHash::of_bytes(&good);
        // Wrong bytes under a real hash → rejected, nothing stored.
        let err = s.store_chunk(hash, b"not the bytes").unwrap_err();
        assert!(matches!(err, ChunkError::HashMismatch { .. }));
        assert!(!s.has(hash).unwrap());
    }

    #[test]
    fn reassemble_missing_chunk_errors() {
        let (_td, s) = store();
        let blob: Vec<u8> = vec![3u8; CHUNK_MAX + 10];
        let index = ChunkIndex::of(&blob); // computed but NOT stored
        let err = s.reassemble(&index).unwrap_err();
        assert!(matches!(err, ChunkError::MissingChunk { .. }));
    }

    #[test]
    fn validate_rejects_arity_and_size() {
        let mut idx = ChunkIndex::of(&vec![0u8; CHUNK_MAX + 1]); // 2 chunks
        assert!(idx.validate().is_ok());
        idx.chunk_hashes.pop(); // now 1 chunk, inconsistent
        assert!(matches!(
            idx.validate().unwrap_err(),
            ChunkError::ArityMismatch { .. }
        ));
        let huge = ChunkIndex {
            total_len: MAX_BLOB_SIZE + 1,
            chunk_hashes: Vec::new(),
        };
        assert!(matches!(
            huge.validate().unwrap_err(),
            ChunkError::TooLarge { .. }
        ));
    }

    #[test]
    fn reassemble_does_not_preallocate_declared_length() {
        // An arity-VALID index naming a huge total_len whose chunks we don't
        // hold must fail with MissingChunk, NOT OOM on a speculative alloc of
        // total_len. Claims ~1 GiB (86 bogus hashes) but nothing is stored.
        let (_td, s) = store();
        let total_len = (CHUNK_MAX as u64) * 86; // ~1 GiB, arity-valid
        let bogus = ChunkIndex {
            total_len,
            chunk_hashes: (0..86u64)
                .map(|i| ContentHash::of_bytes(&i.to_le_bytes()))
                .collect(),
        };
        bogus.validate().expect("arity is internally consistent");
        assert!(matches!(
            s.reassemble(&bogus).unwrap_err(),
            ChunkError::MissingChunk { .. }
        ));
    }

    // Test helper: pull one chunk's bytes back out of a sender store.
    impl ChunkStore {
        fn reassemble_one(&self, index: &ChunkIndex, i: usize) -> Vec<u8> {
            self.inner.get(index.chunk_hashes[i]).unwrap().unwrap()
        }
    }
}
