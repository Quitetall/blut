// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Typed artifacts — the boundary between stages.
//!
//! An artifact is a Rust struct that *references* on-disk bytes,
//! plus a stable content hash, plus a kind tag and schema version.
//! Three properties at once:
//!
//!   - **Type safety:** the Rust type system enforces that stage
//!     inputs/outputs match. A `convert_gguf` stage takes
//!     `HfCheckpoint`; trying to feed it `DatasetJsonl` is a
//!     compile error.
//!   - **Reproducibility:** identical bytes → identical
//!     `ContentHash` → identical cache key downstream. Same plan,
//!     same args, same source data ⇒ skip re-running stages.
//!   - **Audit lineage:** each materialized artifact has a sidecar
//!     `metadata.json` (`ArtifactMetadata`) recording kind, schema,
//!     hash, producing stage, timestamp. A trained model can be
//!     traced back to the data + recipe + stage chain that produced
//!     it just by reading sidecar files in the job dir.
//!
//! The `Artifact` trait is intentionally narrow — `KIND`, `SCHEMA`,
//! `content_hash`, and `primary_path`. Concrete artifacts
//! (`DatasetJsonl`, `HfCheckpoint`, `GgufModel`, `EvalReport`) live
//! in `artifacts/` and pick their own field shape.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// 32-byte SHA-256 content hash. Newtype so accidental use of a raw
/// `[u8; 32]` (which could be anything) is a type error.
///
/// `Display` and `Serialize` emit lowercase hex (the `cache.rs` cache
/// directory uses the hex string as a path component). `FromStr` /
/// `Deserialize` accept hex back. Constant-time comparison via
/// `Eq`/`PartialEq` is unnecessary here — these aren't secrets, just
/// content addresses.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// Hash a contiguous byte slice. Used by `hash_file` after
    /// streaming-read into a single buffer (small files) and in
    /// tests.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(bytes);
        let arr: [u8; 32] = h.finalize().into();
        Self(arr)
    }

    /// Compute SHA-256 over a file's bytes. Two strategies:
    ///
    ///   - Files smaller than `MMAP_THRESHOLD` (16 MiB) use a
    ///     buffered 64 KiB stream-read. Tiny files don't benefit
    ///     from mmap and read() has lower latency below the
    ///     threshold.
    ///   - Files at or above the threshold use mmap. The kernel
    ///     handles paging, the hasher sees the bytes as a single
    ///     contiguous slice, and SHA-256 throughput closes in on
    ///     CPU-bound peak (~2 GB/s with hardware SHA-NI).
    ///
    /// Both paths return identical bytes for identical content.
    pub fn hash_file(path: &Path) -> std::io::Result<Self> {
        const MMAP_THRESHOLD: u64 = 16 * 1024 * 1024;
        let meta = std::fs::metadata(path)?;
        if meta.len() >= MMAP_THRESHOLD && meta.is_file() {
            return Self::hash_file_mmap(path);
        }
        Self::hash_file_streaming(path)
    }

    /// Stream-read fallback. Used for small files + when mmap
    /// fails (some filesystems don't support it).
    fn hash_file_streaming(path: &Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let arr: [u8; 32] = hasher.finalize().into();
        Ok(Self(arr))
    }

    /// mmap path. Falls back to streaming if mmap fails (which can
    /// happen on tmpfs in some kernel configs or on remote FS).
    /// Mmap is fundamentally unsafe (other processes can truncate
    /// the file out from under us, producing SIGBUS). Acceptable
    /// here: BLUT artifacts are content-addressed and live in
    /// directories we own; an external truncate would be a bug
    /// the user wants to know about, not silently hide.
    #[allow(unsafe_code)]
    fn hash_file_mmap(path: &Path) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        // SAFETY: mmap is unsafe because the file's contents can
        // change underneath the borrow. See doc comment above for
        // why we accept the risk in this context.
        let mmap = match unsafe { memmap2::Mmap::map(&f) } {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(
                    "hash_file: mmap failed for {} ({e}); falling back to streaming",
                    path.display()
                );
                return Self::hash_file_streaming(path);
            }
        };
        let mut hasher = Sha256::new();
        hasher.update(&mmap[..]);
        let arr: [u8; 32] = hasher.finalize().into();
        Ok(Self(arr))
    }

    /// Merkle-style hash over a directory: hash each entry's path +
    /// hash, concatenated in sorted order. "Sorted" matters — same
    /// directory contents ⇒ same hash regardless of filesystem
    /// enumeration order. Symlinks are followed (we want the
    /// content-addressed result, not the link). Subdirs recurse.
    ///
    /// Errors propagate from `read_dir` / file open. Malformed
    /// non-UTF-8 paths are hashed via their lossy form — same
    /// fallback `Path::display` uses, deterministic per platform.
    ///
    /// Performance: file hashing is parallelized via rayon —
    /// embarrassingly parallel and the dominant cost on a multi-
    /// file checkpoint dir. The walk itself stays single-threaded
    /// (cheap; deterministic). Output is identical to the serial
    /// `hash_dir_serial` variant.
    pub fn hash_dir(path: &Path) -> std::io::Result<Self> {
        use rayon::prelude::*;
        // Phase 1: cheap serial walk to gather (rel, abs) pairs.
        let mut pairs: Vec<(String, std::path::PathBuf)> = Vec::new();
        Self::collect_files(path, path, &mut pairs)?;
        // Phase 2: parallel hash. Each file is independent; CPU
        // and disk both benefit from multi-thread issue.
        let mut entries: Vec<(String, ContentHash)> = pairs
            .into_par_iter()
            .map(|(rel, abs)| Self::hash_file(&abs).map(|h| (rel, h)))
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut hasher = Sha256::new();
        for (rel, child) in &entries {
            hasher.update(rel.as_bytes());
            hasher.update([0u8]); // separator (NUL — can't appear in path)
            hasher.update(child.0);
        }
        let arr: [u8; 32] = hasher.finalize().into();
        Ok(Self(arr))
    }

    /// Single-threaded variant. Same output as `hash_dir`. Kept
    /// for tests + environments where the rayon thread pool isn't
    /// a fit (single-core, embedded).
    pub fn hash_dir_serial(path: &Path) -> std::io::Result<Self> {
        let mut entries: Vec<(String, ContentHash)> = Vec::new();
        Self::walk_dir(path, path, &mut entries)?;
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut hasher = Sha256::new();
        for (rel, child) in &entries {
            hasher.update(rel.as_bytes());
            hasher.update([0u8]);
            hasher.update(child.0);
        }
        let arr: [u8; 32] = hasher.finalize().into();
        Ok(Self(arr))
    }

    fn collect_files(
        root: &Path,
        cur: &Path,
        out: &mut Vec<(String, std::path::PathBuf)>,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(cur)? {
            let entry = entry?;
            let p = entry.path();
            let meta = entry.metadata()?;
            if meta.is_dir() {
                Self::collect_files(root, &p, out)?;
            } else {
                let rel = p
                    .strip_prefix(root)
                    .map(|r| r.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| p.to_string_lossy().into_owned());
                out.push((rel, p));
            }
        }
        Ok(())
    }

    fn walk_dir(
        root: &Path,
        cur: &Path,
        out: &mut Vec<(String, ContentHash)>,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(cur)? {
            let entry = entry?;
            let p = entry.path();
            let rel = p
                .strip_prefix(root)
                .map(|r| r.to_string_lossy().into_owned())
                .unwrap_or_else(|_| p.to_string_lossy().into_owned());
            let meta = entry.metadata()?;
            if meta.is_dir() {
                Self::walk_dir(root, &p, out)?;
            } else {
                let h = Self::hash_file(&p)?;
                out.push((rel, h));
            }
        }
        Ok(())
    }

    /// Lowercase-hex string. Inverse of `from_hex`. Byte-stable
    /// across platforms; safe to use in path components on every
    /// filesystem we target (POSIX + tmpfs + APFS + NTFS).
    ///
    /// Performance: SIMD-accelerated via faster-hex (SSE 4.1 on
    /// x86, NEON on ARM; scalar lookup on other targets). The
    /// crate auto-detects at runtime; no nightly required.
    pub fn to_hex(self) -> String {
        faster_hex::hex_string(&self.0)
    }

    /// Parse a 64-character hex string. Errors with a clear message
    /// on wrong length or non-hex characters; we don't want a
    /// silent panic in cache lookup paths.
    pub fn from_hex(s: &str) -> Result<Self, ContentHashError> {
        if s.len() != 64 {
            return Err(ContentHashError::WrongLength(s.len()));
        }
        // Index into raw bytes, not `&str`, so a multi-byte UTF-8 character
        // landing on one of the (even) 2-byte chunk boundaries can't panic
        // with "byte index is not a char boundary" — `s.len() == 64` is a
        // BYTE count, so a 64-byte string can still contain non-ASCII chars
        // (e.g. one 3-byte char + 61 ASCII bytes). `u8 as char` is always a
        // total, panic-free cast (every byte is a valid Latin-1 scalar), and
        // `to_digit(16)` rejects anything that isn't an ASCII hex digit —
        // found via cargo-fuzz (ADR 0072 A6) on `fuzz_erased_artifact`
        // within the first smoke run, deserializing a `DatasetJsonl` whose
        // `content_hash` field carried a hostile 64-byte string.
        let bytes = s.as_bytes();
        let mut out = [0u8; 32];
        for (i, byte_str) in (0..64).step_by(2).enumerate() {
            let hi = (bytes[byte_str] as char).to_digit(16);
            let lo = (bytes[byte_str + 1] as char).to_digit(16);
            match (hi, lo) {
                (Some(h), Some(l)) => out[i] = ((h << 4) | l) as u8,
                _ => return Err(ContentHashError::NotHex(byte_str)),
            }
        }
        Ok(Self(out))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ContentHashError {
    #[error("expected 64 hex chars, got {0}")]
    WrongLength(usize),
    #[error("non-hex byte at offset {0}")]
    NotHex(usize),
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Truncated for human-readable use (commit-hash convention).
        // Use to_hex() when the full value is needed.
        let hex = self.to_hex();
        write!(f, "{}", &hex[..12])
    }
}

impl std::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContentHash({})", self.to_hex())
    }
}

impl Serialize for ContentHash {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let s = String::deserialize(d)?;
        Self::from_hex(&s).map_err(D::Error::custom)
    }
}

/// The framework's typed-artifact contract.
///
/// Implementors are concrete data types like `DatasetJsonl`,
/// `HfCheckpoint`, `GgufModel`. The trait is what the `Stage`'s
/// `Input` / `Output` associated types must satisfy. Tuple impls
/// (below) handle multi-input merge stages (e.g. `distill_train`
/// takes `(HfCheckpoint, DatasetJsonl)`).
pub trait Artifact: Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static {
    /// Stable kind tag (e.g. `"dataset.jsonl"`). Must be unique
    /// across the catalog; used in cache keys + sidecar metadata
    /// + stage compatibility checks. Bumping is a breaking change.
    const KIND: &'static str;

    /// Schema version. Bump ONLY when the artifact's struct fields
    /// change in a way that affects downstream semantics — adding
    /// a new field, renaming, repurposing. Do NOT bump for purely
    /// internal optimizations (faster impl, better validation,
    /// etc.) — that just invalidates cache without reason. The
    /// trade-off is correctness vs cache reuse; bias toward cache
    /// reuse when the change is internal.
    const SCHEMA: u32;

    /// For a LIST artifact (`ListOf<E>`), the `KIND` of its elements;
    /// `None` for every scalar/tuple artifact. Reported up the erased
    /// layer as `StageDyn::output_element_kind()` so a typed runtime
    /// `map_output` fan-out (ADR 0078) can kind-check the template's root
    /// against the element type before it runs.
    const ELEMENT_KIND: Option<&'static str> = None;

    /// Whether the artifact's `content_hash()` should walk on-disk
    /// bytes (true) or use a cheap fingerprint of path + size +
    /// mtime (false). Default true matches small artifacts where
    /// the bytes ARE the artifact (dataset.jsonl, eval reports).
    ///
    /// Override `false` for large bulk artifacts where content
    /// equality across reruns is rare and content hashing costs
    /// minutes (large model checkpoints, multi-GB GGUF files,
    /// large memmaps). Concrete artifacts that pick `false`
    /// must implement `content_hash()` using a stat-based
    /// fingerprint instead of bytes.
    const HASH_CONTENTS: bool = true;

    /// Stable content hash. For file-backed artifacts this is the
    /// SHA-256 of the canonical bytes; for composite artifacts a
    /// merkle of children. Idempotent — same content ⇒ same hash
    /// ⇒ same cache key downstream.
    fn content_hash(&self) -> ContentHash;

    /// Read-only path the user can `ls`. Always inside a stable
    /// location (job dir or content-addressed cache); never a
    /// tmpfile that might disappear.
    fn primary_path(&self) -> &Path;

    /// Re-derive the content address from ON-DISK bytes — NOT the cached
    /// `content_hash()` field. The P2P import path (`p2p::bundle::unbundle`)
    /// calls this to verify a transferred artifact against the
    /// coordinator-SIGNED `input_hash`: the cached accessor is a
    /// self-attestation (`DatasetJsonl`/`HfCheckpoint` just
    /// `return self.content_hash`) that proves nothing about the bytes that
    /// actually crossed the wire.
    ///
    /// The default re-walks `primary_path()` (dir → `hash_dir`, file →
    /// `hash_file`). A COMPOSITE artifact whose `content_hash()` is a merkle
    /// over members MUST override — the default sees only `primary_path()` and
    /// can never reproduce the merkle (e.g. `DatasetSplit`).
    fn recompute_content_hash(&self) -> std::io::Result<ContentHash> {
        let p = self.primary_path();
        if p.is_dir() {
            ContentHash::hash_dir(p)
        } else {
            ContentHash::hash_file(p)
        }
    }

    /// Encode to the erased wire form for transit across the `StageDyn`
    /// boundary (and as a merge-tuple member). Default: bincode-of-self
    /// tagged with `KIND`/`SCHEMA`. The tuple impls override this to a
    /// per-child envelope so each member is independently validated on
    /// decode.
    fn encode_erased(
        &self,
    ) -> Result<crate::framework::stage::ErasedArtifact, crate::framework::stage::ErasedEncodeError>
    {
        use crate::framework::stage::{ErasedArtifact, ErasedEncodeError};
        Ok(ErasedArtifact {
            kind: Self::KIND.to_string(),
            schema: Self::SCHEMA,
            payload: bincode::serialize(self).map_err(ErasedEncodeError::Serialize)?,
        })
    }

    /// Decode from the erased wire form, validating kind + schema.
    /// Default: bincode-of-self. The tuple impls override to unpack the
    /// per-child envelope and recursively decode each member — so a
    /// wrong child kind surfaces the ACTUAL child kind, not an opaque
    /// bincode error blamed on the merge stage.
    fn decode_erased(
        e: crate::framework::stage::ErasedArtifact,
    ) -> Result<Self, crate::framework::stage::ErasedDecodeError>
    where
        Self: Sized,
    {
        use crate::framework::stage::ErasedDecodeError;
        if e.kind != Self::KIND {
            return Err(ErasedDecodeError::Kind {
                expected: Self::KIND,
                got: e.kind,
            });
        }
        if e.schema != Self::SCHEMA {
            return Err(ErasedDecodeError::Schema {
                expected: Self::SCHEMA,
                got: e.schema,
            });
        }
        bincode::deserialize(&e.payload).map_err(ErasedDecodeError::Deserialize)
    }
}

/// Sidecar metadata.json next to every materialized artifact.
/// Captures the full audit lineage: which stage produced this, when,
/// what kind it is, what its content hash was at that moment.
///
/// Lives at `<artifact_primary_path>.metadata.json` for files, and
/// at `<artifact_primary_path>/.lamu-meta.json` for directory
/// artifacts (the leading dot keeps it out of recursive content
/// hashing of sibling files — we don't want metadata changing the
/// hash of the artifact it describes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub kind: String,
    pub schema: u32,
    pub content_hash: ContentHash,
    /// The stage that produced this artifact, e.g.
    /// `"materialize_conversations"`. None for graph inputs.
    pub produced_by_stage: Option<String>,
    /// UNIX seconds at production time. For human readability via
    /// `lamu-train log <job>` and for cache LRU pruning.
    pub produced_at_unix_secs: u64,
    /// Optional free-form provenance bag. Recipe args, dataset row
    /// counts, training step counts — whatever the producing stage
    /// cares to record.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ArtifactMetadata {
    pub fn new(kind: impl Into<String>, schema: u32, content_hash: ContentHash) -> Self {
        Self {
            kind: kind.into(),
            schema,
            content_hash,
            produced_by_stage: None,
            produced_at_unix_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            extra: serde_json::Map::new(),
        }
    }

    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.produced_by_stage = Some(stage.into());
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Where to write the sidecar for an artifact whose primary
    /// path is `primary`. Files get `<primary>.metadata.json`;
    /// directories get `<primary>/.lamu-meta.json` (leading dot
    /// keeps it out of recursive content-hash walks).
    ///
    /// Convention: callers must write the primary artifact BEFORE
    /// calling `write_alongside`. The dir/file branch is decided by
    /// querying the primary path on disk; calling early would
    /// classify a not-yet-existing dir as a file. This matches the
    /// natural lifecycle (stage produces output → writes sidecar
    /// last) so it's rarely a footgun in practice.
    pub fn sidecar_path_for(primary: &Path) -> PathBuf {
        if primary.is_dir() {
            primary.join(".lamu-meta.json")
        } else {
            // Append `.metadata.json` to the raw OsString so paths
            // with no extension and paths with multiple dots both
            // round-trip correctly. `with_extension` would replace
            // an existing one, which is wrong for `data.jsonl` →
            // `data.metadata.json` (we want `.jsonl.metadata.json`).
            let mut s = primary.as_os_str().to_os_string();
            s.push(".metadata.json");
            PathBuf::from(s)
        }
    }

    pub fn write_to(&self, sidecar_path: &Path) -> std::io::Result<()> {
        if let Some(parent) = sidecar_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        // Compact JSON (opt-5): ~2× smaller on disk than pretty,
        // faster to serialize, and sidecars are read by tools not
        // humans 99% of the time. Use `jq .` if a human needs to
        // pretty-print one.
        let body = serde_json::to_vec(self).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("serialize sidecar: {e}"),
            )
        })?;
        std::fs::write(sidecar_path, body)
    }

    pub fn read_from(sidecar_path: &Path) -> std::io::Result<Self> {
        let body = std::fs::read(sidecar_path)?;
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("parse sidecar: {e}"),
            )
        })
    }

    /// Convenience: write the sidecar to the canonical location
    /// next to `primary`. Returns the sidecar path on success.
    pub fn write_alongside(&self, primary: &Path) -> std::io::Result<PathBuf> {
        let p = Self::sidecar_path_for(primary);
        self.write_to(&p)?;
        Ok(p)
    }
}

// ── Tuple Artifact impls for multi-input merge stages ────────────
//
// A stage like `distill_train` takes `(HfCheckpoint, DatasetJsonl)`.
// The Plan builder's `merge` API requires that the merged-in tuple
// type itself satisfies `Artifact`. We provide blanket impls for
// 2-tuple and 3-tuple. Higher arities can be added later if a real
// stage needs them; we deliberately don't pre-enable infinite
// arities since the macro for that hides the constraint each impl
// places on its members.
//
// Tuple `KIND` is a compile-time-fixed string of the form
// `"tuple<A,B>"`. The `content_hash` is the merkle of children's
// hashes — order-sensitive (a `(A, B)` differs from `(B, A)`).
// `primary_path` returns the FIRST element's path; this is a
// convention used by tuple-consuming stages, which know to look
// at both children via `Artifact::content_hash` of each side.
//
// Why not just use a single struct for each tuple? Because the type
// system is the point: `Plan::merge<S>` enforces
// `S: Stage<Input = (A, B)>`, and that requires
// `(A, B): Artifact`. Generic blanket impls give us exactly that.

/// `()` is the canonical "no upstream input" artifact. Used as
/// `Stage::Input = ()` for graph-input stages
/// (`materialize_conversations`, `materialize_dataset_path`, etc.)
/// that take their data from outside the plan rather than from a
/// predecessor stage. The hash is the SHA-256 of the empty byte
/// string so cache keys are stable; primary_path is empty.
impl Artifact for () {
    const KIND: &'static str = "()";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&[])
    }
    fn primary_path(&self) -> &Path {
        Path::new("")
    }
}

// Tuple hashes are domain-separated by arity: every tuple's hash
// starts with `b"tuple"` and the arity as a u8. Without this, a
// 2-tuple and a 3-tuple whose concatenated child-hashes happen to
// align could collide — vanishingly unlikely under SHA-256, but
// cheap insurance and makes the hash space self-documenting.
const TUPLE_DOMAIN: &[u8] = b"tuple";

impl<A: Artifact, B: Artifact> Artifact for (A, B) {
    const KIND: &'static str = "tuple<2>";
    const SCHEMA: u32 = crate::framework::stage::TUPLE_ENVELOPE_SCHEMA;

    fn content_hash(&self) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(TUPLE_DOMAIN);
        hasher.update([2u8]);
        hasher.update(self.0.content_hash().0);
        hasher.update(self.1.content_hash().0);
        let arr: [u8; 32] = hasher.finalize().into();
        ContentHash(arr)
    }

    fn primary_path(&self) -> &Path {
        // Convention: first child's path. Tuple consumers know to
        // address members individually via destructuring.
        self.0.primary_path()
    }

    fn encode_erased(
        &self,
    ) -> Result<crate::framework::stage::ErasedArtifact, crate::framework::stage::ErasedEncodeError>
    {
        use crate::framework::stage::{ErasedArtifact, ErasedEncodeError};
        let children = vec![self.0.encode_erased()?, self.1.encode_erased()?];
        Ok(ErasedArtifact {
            kind: Self::KIND.to_string(),
            schema: Self::SCHEMA,
            payload: bincode::serialize(&children).map_err(ErasedEncodeError::Serialize)?,
        })
    }

    fn decode_erased(
        e: crate::framework::stage::ErasedArtifact,
    ) -> Result<Self, crate::framework::stage::ErasedDecodeError> {
        decode_tuple_children::<2>(e, Self::KIND, Self::SCHEMA).and_then(|mut c| {
            // SAFETY: decode_tuple_children::<2> guarantees exactly 2 children.
            let b = B::decode_erased(c.pop().unwrap())?;
            let a = A::decode_erased(c.pop().unwrap())?;
            Ok((a, b))
        })
    }
}

impl<A: Artifact, B: Artifact, C: Artifact> Artifact for (A, B, C) {
    const KIND: &'static str = "tuple<3>";
    const SCHEMA: u32 = crate::framework::stage::TUPLE_ENVELOPE_SCHEMA;

    fn content_hash(&self) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(TUPLE_DOMAIN);
        hasher.update([3u8]);
        hasher.update(self.0.content_hash().0);
        hasher.update(self.1.content_hash().0);
        hasher.update(self.2.content_hash().0);
        let arr: [u8; 32] = hasher.finalize().into();
        ContentHash(arr)
    }

    fn primary_path(&self) -> &Path {
        self.0.primary_path()
    }

    fn encode_erased(
        &self,
    ) -> Result<crate::framework::stage::ErasedArtifact, crate::framework::stage::ErasedEncodeError>
    {
        use crate::framework::stage::{ErasedArtifact, ErasedEncodeError};
        let children = vec![
            self.0.encode_erased()?,
            self.1.encode_erased()?,
            self.2.encode_erased()?,
        ];
        Ok(ErasedArtifact {
            kind: Self::KIND.to_string(),
            schema: Self::SCHEMA,
            payload: bincode::serialize(&children).map_err(ErasedEncodeError::Serialize)?,
        })
    }

    fn decode_erased(
        e: crate::framework::stage::ErasedArtifact,
    ) -> Result<Self, crate::framework::stage::ErasedDecodeError> {
        decode_tuple_children::<3>(e, Self::KIND, Self::SCHEMA).and_then(|mut c| {
            // SAFETY: decode_tuple_children::<3> guarantees exactly 3 children.
            let cc = C::decode_erased(c.pop().unwrap())?;
            let b = B::decode_erased(c.pop().unwrap())?;
            let a = A::decode_erased(c.pop().unwrap())?;
            Ok((a, b, cc))
        })
    }
}

/// Validate a `tuple<N>` envelope's kind, schema, and arity, returning
/// the `N` child `ErasedArtifact`s for per-member recursive decode.
fn decode_tuple_children<const N: usize>(
    e: crate::framework::stage::ErasedArtifact,
    kind: &'static str,
    schema: u32,
) -> Result<Vec<crate::framework::stage::ErasedArtifact>, crate::framework::stage::ErasedDecodeError>
{
    use crate::framework::stage::ErasedDecodeError;
    if e.kind != kind {
        return Err(ErasedDecodeError::Kind {
            expected: kind,
            got: e.kind,
        });
    }
    if e.schema != schema {
        return Err(ErasedDecodeError::Schema {
            expected: schema,
            got: e.schema,
        });
    }
    let children: Vec<crate::framework::stage::ErasedArtifact> =
        bincode::deserialize(&e.payload).map_err(ErasedDecodeError::Deserialize)?;
    if children.len() != N {
        return Err(ErasedDecodeError::Arity {
            expected: N,
            got: children.len(),
        });
    }
    Ok(children)
}

// ---------------------------------------------------------------------------
// ListOf<E> — a homogeneous LIST artifact (ADR 0078 typed `map_output`).
// ---------------------------------------------------------------------------
//
// A stage whose `Output = ListOf<Item>` emits a variable-width list; a
// runtime `map_output` fan-out spawns one template instance per element. The
// erased envelope mirrors the tuple envelope (a bincode `Vec<ErasedArtifact>`,
// one framed element each) EXCEPT the arity is data, not part of the kind:
// `KIND` is the constant `"list"` and the element type is carried separately
// via `ELEMENT_KIND` (so the executor can kind-check the map template without
// knowing the concrete `Item`). The `content_hash` is a merkle over element
// hashes, domain-separated by `b"list"` + a `u32` count so lists of different
// lengths can't collide and an empty list has a stable, distinct hash.

const LIST_DOMAIN: &[u8] = b"list";

/// A homogeneous list of artifacts. `KIND = "list"`; the element kind is
/// reported via [`Artifact::ELEMENT_KIND`]. Produced by a stage that fans a
/// runtime-sized collection out to a `map_output` template. The `E: Artifact`
/// bound lives on the impls, not the struct, so the serde derive generates
/// clean `E: Serialize`/`E: Deserialize` bounds (an `E: Artifact` bound here
/// would give the derive two ambiguous routes to those traits).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ListOf<E>(pub Vec<E>);

impl<E> ListOf<E> {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<E: Artifact> Artifact for ListOf<E> {
    const KIND: &'static str = "list";
    const SCHEMA: u32 = crate::framework::stage::LIST_ENVELOPE_SCHEMA;
    const ELEMENT_KIND: Option<&'static str> = Some(E::KIND);

    fn content_hash(&self) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(LIST_DOMAIN);
        hasher.update((self.0.len() as u32).to_le_bytes());
        for e in &self.0 {
            hasher.update(e.content_hash().0);
        }
        let arr: [u8; 32] = hasher.finalize().into();
        ContentHash(arr)
    }

    fn primary_path(&self) -> &Path {
        // Convention: the first element's path (empty list → empty path).
        // Consumers of a list address elements individually.
        self.0
            .first()
            .map(|e| e.primary_path())
            .unwrap_or(Path::new(""))
    }

    fn recompute_content_hash(&self) -> std::io::Result<ContentHash> {
        // Composite: recompute the merkle from each element's ON-DISK bytes
        // (the default would see only `primary_path()` and miss the rest).
        let mut hasher = Sha256::new();
        hasher.update(LIST_DOMAIN);
        hasher.update((self.0.len() as u32).to_le_bytes());
        for e in &self.0 {
            hasher.update(e.recompute_content_hash()?.0);
        }
        let arr: [u8; 32] = hasher.finalize().into();
        Ok(ContentHash(arr))
    }

    fn encode_erased(
        &self,
    ) -> Result<crate::framework::stage::ErasedArtifact, crate::framework::stage::ErasedEncodeError>
    {
        use crate::framework::stage::{ErasedArtifact, ErasedEncodeError};
        let children = self
            .0
            .iter()
            .map(|e| e.encode_erased())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ErasedArtifact {
            kind: Self::KIND.to_string(),
            schema: Self::SCHEMA,
            payload: bincode::serialize(&children).map_err(ErasedEncodeError::Serialize)?,
        })
    }

    fn decode_erased(
        e: crate::framework::stage::ErasedArtifact,
    ) -> Result<Self, crate::framework::stage::ErasedDecodeError> {
        let children = decode_list_children(e)?;
        let items = children
            .into_iter()
            .map(E::decode_erased)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ListOf(items))
    }
}

/// Validate a `list` envelope's kind + schema and return its element
/// `ErasedArtifact`s (arity is data, so any count is valid). This is the
/// erased-level unpack the executor's runtime fan-out uses — it must split a
/// list into elements WITHOUT knowing the concrete element type.
pub(crate) fn decode_list_children(
    e: crate::framework::stage::ErasedArtifact,
) -> Result<Vec<crate::framework::stage::ErasedArtifact>, crate::framework::stage::ErasedDecodeError>
{
    use crate::framework::stage::ErasedDecodeError;
    let expected_kind = <ListOf<()> as Artifact>::KIND;
    if e.kind != expected_kind {
        return Err(ErasedDecodeError::Kind {
            expected: expected_kind,
            got: e.kind,
        });
    }
    if e.schema != crate::framework::stage::LIST_ENVELOPE_SCHEMA {
        return Err(ErasedDecodeError::Schema {
            expected: crate::framework::stage::LIST_ENVELOPE_SCHEMA,
            got: e.schema,
        });
    }
    bincode::deserialize(&e.payload).map_err(ErasedDecodeError::Deserialize)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --------- ContentHash ---------------------------------------

    #[test]
    fn content_hash_of_bytes_known_value() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let h = ContentHash::of_bytes(b"hello");
        assert_eq!(
            h.to_hex(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn content_hash_hex_round_trip() {
        let h = ContentHash::of_bytes(b"round trip");
        let hex = h.to_hex();
        let back = ContentHash::from_hex(&hex).unwrap();
        assert_eq!(h, back);
    }

    #[test]
    fn content_hash_from_hex_rejects_wrong_length() {
        assert!(matches!(
            ContentHash::from_hex("abcd"),
            Err(ContentHashError::WrongLength(4))
        ));
    }

    #[test]
    fn content_hash_from_hex_rejects_non_hex() {
        let bad = format!("z{}", "a".repeat(63));
        assert!(matches!(
            ContentHash::from_hex(&bad),
            Err(ContentHashError::NotHex(0))
        ));
    }

    /// Regression for a cargo-fuzz find (ADR 0072 A6, `fuzz_erased_artifact`,
    /// hit within the first 15s smoke run): a 64-BYTE string containing a
    /// multi-byte UTF-8 character can land a `step_by(2)` chunk boundary
    /// mid-character. The old impl sliced `&str` directly and panicked with
    /// "byte index is not a char boundary" instead of returning `NotHex`.
    /// 39 ASCII bytes + one 3-byte char ('➝', U+279D) + 22 ASCII bytes = 64
    /// bytes, with the char starting at odd offset 39 so the chunk starting
    /// at byte_str=38 straddles it.
    #[test]
    fn content_hash_from_hex_rejects_non_char_boundary_multibyte() {
        let bad = format!("{}➝{}", "a".repeat(39), "a".repeat(22));
        assert_eq!(bad.len(), 64);
        assert!(matches!(
            ContentHash::from_hex(&bad),
            Err(ContentHashError::NotHex(38))
        ));
    }

    #[test]
    fn content_hash_display_truncates() {
        let h = ContentHash::of_bytes(b"x");
        let disp = format!("{h}");
        assert_eq!(disp.len(), 12);
        assert!(h.to_hex().starts_with(&disp));
    }

    #[test]
    fn content_hash_serde_round_trip() {
        let h = ContentHash::of_bytes(b"serde");
        let json = serde_json::to_string(&h).unwrap();
        // Body is a JSON string literal of the hex.
        assert!(json.starts_with('"') && json.ends_with('"'));
        let back: ContentHash = serde_json::from_str(&json).unwrap();
        assert_eq!(h, back);
    }

    // --------- hash_file / hash_dir ------------------------------

    #[test]
    fn hash_file_matches_of_bytes() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("a.bin");
        std::fs::write(&p, b"contents").unwrap();
        let from_file = ContentHash::hash_file(&p).unwrap();
        let from_bytes = ContentHash::of_bytes(b"contents");
        assert_eq!(from_file, from_bytes);
    }

    #[test]
    fn hash_dir_is_deterministic_across_orders() {
        // Build the same dir twice with files created in different
        // orders. Hash must match.
        fn build(td: &Path, order: &[&str]) -> ContentHash {
            for name in order {
                std::fs::write(td.join(name), name.as_bytes()).unwrap();
            }
            ContentHash::hash_dir(td).unwrap()
        }
        let td1 = tempfile::tempdir().unwrap();
        let td2 = tempfile::tempdir().unwrap();
        let h1 = build(td1.path(), &["a", "b", "c"]);
        let h2 = build(td2.path(), &["c", "a", "b"]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_dir_changes_when_content_changes() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("a"), b"v1").unwrap();
        let h1 = ContentHash::hash_dir(td.path()).unwrap();
        std::fs::write(td.path().join("a"), b"v2").unwrap();
        let h2 = ContentHash::hash_dir(td.path()).unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_dir_recurses_into_subdirs() {
        let td = tempfile::tempdir().unwrap();
        let sub = td.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("nested"), b"deep").unwrap();
        let h = ContentHash::hash_dir(td.path()).unwrap();
        // Smoke: with no other files, swapping the nested file
        // changes the hash.
        std::fs::write(sub.join("nested"), b"different").unwrap();
        let h2 = ContentHash::hash_dir(td.path()).unwrap();
        assert_ne!(h, h2);
    }

    // --------- ArtifactMetadata ----------------------------------

    #[test]
    fn metadata_round_trip() {
        let md = ArtifactMetadata::new("dataset.jsonl", 1, ContentHash::of_bytes(b"x"))
            .with_stage("materialize_conversations")
            .with_extra("n_examples", serde_json::json!(42));
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("artifact.bin");
        std::fs::write(&p, b"payload").unwrap();
        let sidecar = md.write_alongside(&p).unwrap();
        assert!(sidecar.exists());
        let back = ArtifactMetadata::read_from(&sidecar).unwrap();
        assert_eq!(back.kind, "dataset.jsonl");
        assert_eq!(back.schema, 1);
        assert_eq!(back.content_hash, md.content_hash);
        assert_eq!(
            back.produced_by_stage.as_deref(),
            Some("materialize_conversations")
        );
        assert_eq!(back.extra.get("n_examples"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn metadata_sidecar_path_for_file() {
        let p = Path::new("/tmp/foo/data.jsonl");
        let s = ArtifactMetadata::sidecar_path_for(p);
        assert_eq!(s, PathBuf::from("/tmp/foo/data.jsonl.metadata.json"));
    }

    #[test]
    fn metadata_sidecar_path_for_dir() {
        let td = tempfile::tempdir().unwrap();
        // sidecar_path_for branches on `is_dir()`; needs a real dir.
        let s = ArtifactMetadata::sidecar_path_for(td.path());
        assert_eq!(s, td.path().join(".lamu-meta.json"));
    }

    // --------- Tuple Artifact impls ------------------------------

    /// Minimal Artifact impl for tuple-tests. Wraps a u8 + path
    /// pair; content_hash is the byte; primary_path is the path.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestArt {
        byte: u8,
        path: PathBuf,
    }

    impl Artifact for TestArt {
        const KIND: &'static str = "test.art";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(&[self.byte])
        }
        fn primary_path(&self) -> &Path {
            &self.path
        }
    }

    /// A second artifact type with a DIFFERENT kind, for the
    /// wrong-child-kind envelope test.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct OtherArt {
        word: String,
    }
    impl Artifact for OtherArt {
        const KIND: &'static str = "test.other";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(self.word.as_bytes())
        }
        fn primary_path(&self) -> &Path {
            Path::new("/other")
        }
    }

    #[test]
    fn tuple_envelope_round_trips() {
        use crate::framework::stage::TUPLE_ENVELOPE_SCHEMA;
        let a = TestArt {
            byte: 7,
            path: PathBuf::from("/a"),
        };
        let b = OtherArt { word: "hi".into() };
        let env = (a.clone(), b.clone()).encode_erased().unwrap();
        assert_eq!(env.kind, "tuple<2>");
        assert_eq!(env.schema, TUPLE_ENVELOPE_SCHEMA);
        let (ra, rb): (TestArt, OtherArt) = <(TestArt, OtherArt)>::decode_erased(env).unwrap();
        assert_eq!(ra.byte, 7);
        assert_eq!(rb.word, "hi");
    }

    #[test]
    fn tuple_envelope_wrong_child_kind_names_actual_kind() {
        use crate::framework::stage::{ErasedArtifact, ErasedDecodeError, TUPLE_ENVELOPE_SCHEMA};
        // Build an envelope whose SECOND child is the wrong kind
        // (OtherArt) where the consumer expects (TestArt, TestArt).
        let good = ErasedArtifact::from_typed(&TestArt {
            byte: 1,
            path: PathBuf::from("/a"),
        })
        .unwrap();
        let wrong = ErasedArtifact::from_typed(&OtherArt { word: "x".into() }).unwrap();
        let children = vec![good, wrong];
        let env = ErasedArtifact {
            kind: "tuple<2>".into(),
            schema: TUPLE_ENVELOPE_SCHEMA,
            payload: bincode::serialize(&children).unwrap(),
        };
        match <(TestArt, TestArt)>::decode_erased(env) {
            Err(ErasedDecodeError::Kind { expected, got }) => {
                assert_eq!(expected, "test.art");
                assert_eq!(got, "test.other", "must name the ACTUAL child kind");
            }
            other => panic!("expected a per-child Kind error, got {other:?}"),
        }
    }

    #[test]
    fn tuple_envelope_arity_mismatch_detected() {
        use crate::framework::stage::{ErasedArtifact, ErasedDecodeError, TUPLE_ENVELOPE_SCHEMA};
        // A 1-child envelope decoded as a 2-tuple → Arity error.
        let one = vec![
            ErasedArtifact::from_typed(&TestArt {
                byte: 1,
                path: PathBuf::from("/a"),
            })
            .unwrap(),
        ];
        let env = ErasedArtifact {
            kind: "tuple<2>".into(),
            schema: TUPLE_ENVELOPE_SCHEMA,
            payload: bincode::serialize(&one).unwrap(),
        };
        assert!(matches!(
            <(TestArt, TestArt)>::decode_erased(env),
            Err(ErasedDecodeError::Arity {
                expected: 2,
                got: 1
            })
        ));
    }

    // --------- ListOf<E> envelope (ADR 0078) ----------------------

    fn art(byte: u8) -> TestArt {
        TestArt {
            byte,
            path: PathBuf::from(format!("/{byte}")),
        }
    }

    #[test]
    fn list_envelope_round_trips_and_reports_element_kind() {
        use crate::framework::stage::LIST_ENVELOPE_SCHEMA;
        let list = ListOf(vec![art(1), art(2), art(3)]);
        assert_eq!(<ListOf<TestArt> as Artifact>::KIND, "list");
        assert_eq!(
            <ListOf<TestArt> as Artifact>::ELEMENT_KIND,
            Some("test.art")
        );
        let env = list.encode_erased().unwrap();
        assert_eq!(env.kind, "list");
        assert_eq!(env.schema, LIST_ENVELOPE_SCHEMA);
        let back: ListOf<TestArt> = ListOf::decode_erased(env).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back.0[1].byte, 2);
    }

    #[test]
    fn empty_list_round_trips_with_a_distinct_stable_hash() {
        let empty = ListOf::<TestArt>(vec![]);
        assert!(empty.is_empty());
        let env = empty.encode_erased().unwrap();
        let back: ListOf<TestArt> = ListOf::decode_erased(env).unwrap();
        assert_eq!(back.len(), 0);
        // Empty list's hash is stable and differs from a 1-element list.
        assert_eq!(
            empty.content_hash(),
            ListOf::<TestArt>(vec![]).content_hash()
        );
        assert_ne!(empty.content_hash(), ListOf(vec![art(1)]).content_hash());
    }

    #[test]
    fn list_content_hash_is_order_and_length_sensitive() {
        let ab = ListOf(vec![art(1), art(2)]).content_hash();
        let ba = ListOf(vec![art(2), art(1)]).content_hash();
        let a = ListOf(vec![art(1)]).content_hash();
        assert_ne!(ab, ba, "order-sensitive");
        assert_ne!(ab, a, "length-sensitive");
    }

    #[test]
    fn decode_list_children_splits_without_the_element_type() {
        // The erased-level unpack the executor's fan-out uses.
        let list = ListOf(vec![art(5), art(6)]);
        let env = list.encode_erased().unwrap();
        let children = super::decode_list_children(env).unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].kind, "test.art");
    }

    #[test]
    fn decode_list_children_rejects_a_non_list_envelope() {
        use crate::framework::stage::{ErasedArtifact, ErasedDecodeError};
        let tup = (art(1), art(2)).encode_erased().unwrap();
        assert!(matches!(
            super::decode_list_children(tup),
            Err(ErasedDecodeError::Kind { got, .. }) if got == "tuple<2>"
        ));
        // Wrong schema on a list-kinded envelope is also caught.
        let bad = ErasedArtifact {
            kind: "list".into(),
            schema: 999,
            payload: bincode::serialize(&Vec::<ErasedArtifact>::new()).unwrap(),
        };
        assert!(matches!(
            super::decode_list_children(bad),
            Err(ErasedDecodeError::Schema { .. })
        ));
    }

    #[test]
    fn tuple2_content_hash_is_deterministic() {
        let a = TestArt {
            byte: 1,
            path: PathBuf::from("/a"),
        };
        let b = TestArt {
            byte: 2,
            path: PathBuf::from("/b"),
        };
        let pair = (a.clone(), b.clone());
        let h1 = pair.content_hash();
        let h2 = (a.clone(), b.clone()).content_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn tuple2_content_hash_is_order_sensitive() {
        let a = TestArt {
            byte: 1,
            path: PathBuf::from("/a"),
        };
        let b = TestArt {
            byte: 2,
            path: PathBuf::from("/b"),
        };
        let h_ab = (a.clone(), b.clone()).content_hash();
        let h_ba = (b, a).content_hash();
        assert_ne!(h_ab, h_ba);
    }

    #[test]
    fn tuple2_primary_path_returns_first() {
        let a = TestArt {
            byte: 1,
            path: PathBuf::from("/first"),
        };
        let b = TestArt {
            byte: 2,
            path: PathBuf::from("/second"),
        };
        let pair = (a, b);
        assert_eq!(pair.primary_path(), Path::new("/first"));
    }

    #[test]
    fn tuple3_content_hash_includes_all_children() {
        let a = TestArt {
            byte: 1,
            path: PathBuf::from("/a"),
        };
        let b = TestArt {
            byte: 2,
            path: PathBuf::from("/b"),
        };
        let c = TestArt {
            byte: 3,
            path: PathBuf::from("/c"),
        };
        let h_full = (a.clone(), b.clone(), c.clone()).content_hash();
        // Replacing the third element should change the hash.
        let c2 = TestArt {
            byte: 99,
            path: PathBuf::from("/c2"),
        };
        let h_diff = (a, b, c2).content_hash();
        assert_ne!(h_full, h_diff);
    }

    #[test]
    fn tuple_kinds_are_distinct() {
        // Compile-time check that the impls disambiguate by arity.
        assert_eq!(<(TestArt, TestArt) as Artifact>::KIND, "tuple<2>");
        assert_eq!(<(TestArt, TestArt, TestArt) as Artifact>::KIND, "tuple<3>");
    }

    #[test]
    fn unit_artifact_round_trips() {
        let h: ContentHash = ().content_hash();
        assert_eq!(h, ContentHash::of_bytes(&[]));
        assert_eq!(<() as Artifact>::KIND, "()");
        // serde round trip
        let json = serde_json::to_value(()).unwrap();
        let _: () = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn tuple_arity_domain_separation() {
        // 2-tuple of (a, a) and 3-tuple of (a, a, a) must produce
        // distinct hashes even when child hashes are identical —
        // the arity byte in the domain prefix prevents collisions.
        let a = TestArt {
            byte: 7,
            path: PathBuf::from("/a"),
        };
        let h2 = (a.clone(), a.clone()).content_hash();
        let h3 = (a.clone(), a.clone(), a.clone()).content_hash();
        assert_ne!(h2, h3, "tuple arity must affect hash");
    }
}
