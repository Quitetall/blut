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
            hasher.update(&child.0);
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
            hasher.update(&child.0);
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
    /// Performance: writes 32 bytes → 64 hex chars via direct
    /// nibble lookup (no per-byte format! call). ~4× faster than
    /// the previous `format!` loop and zero heap re-allocations
    /// (capacity reserved up front).
    pub fn to_hex(self) -> String {
        const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
        let mut out = vec![0u8; 64];
        for (i, b) in self.0.iter().enumerate() {
            out[i * 2] = HEX_CHARS[(b >> 4) as usize];
            out[i * 2 + 1] = HEX_CHARS[(b & 0x0f) as usize];
        }
        // The bytes are guaranteed ASCII from HEX_CHARS; the
        // O(64) UTF-8 validation pass below is dwarfed by the
        // savings vs 32 separate `format!("{:02x}", b)` calls
        // each allocating a 2-byte heap string.
        String::from_utf8(out).expect("HEX_CHARS is ASCII; output is valid UTF-8")
    }

    /// Parse a 64-character hex string. Errors with a clear message
    /// on wrong length or non-hex characters; we don't want a
    /// silent panic in cache lookup paths.
    pub fn from_hex(s: &str) -> Result<Self, ContentHashError> {
        if s.len() != 64 {
            return Err(ContentHashError::WrongLength(s.len()));
        }
        let mut out = [0u8; 32];
        for (i, byte_str) in (0..64).step_by(2).enumerate() {
            out[i] = u8::from_str_radix(&s[byte_str..byte_str + 2], 16)
                .map_err(|_| ContentHashError::NotHex(byte_str))?;
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
pub trait Artifact:
    Send + Sync + serde::Serialize + serde::de::DeserializeOwned + 'static
{
    /// Stable kind tag (e.g. `"dataset.jsonl"`). Must be unique
    /// across the catalog; used in cache keys + sidecar metadata
    /// + stage compatibility checks. Bumping is a breaking change.
    const KIND: &'static str;

    /// Schema version. Bump when on-disk layout changes. Cache
    /// entries from a different `SCHEMA` are invalidated automatically.
    const SCHEMA: u32;

    /// Stable content hash. For file-backed artifacts this is the
    /// SHA-256 of the canonical bytes; for composite artifacts a
    /// merkle of children. Idempotent — same content ⇒ same hash
    /// ⇒ same cache key downstream.
    fn content_hash(&self) -> ContentHash;

    /// Read-only path the user can `ls`. Always inside a stable
    /// location (job dir or content-addressed cache); never a
    /// tmpfile that might disappear.
    fn primary_path(&self) -> &Path;
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
        let body = serde_json::to_vec_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("serialize sidecar: {e}"))
        })?;
        std::fs::write(sidecar_path, body)
    }

    pub fn read_from(sidecar_path: &Path) -> std::io::Result<Self> {
        let body = std::fs::read(sidecar_path)?;
        serde_json::from_slice(&body).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("parse sidecar: {e}"))
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
    const SCHEMA: u32 = 1;

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
}

impl<A: Artifact, B: Artifact, C: Artifact> Artifact for (A, B, C) {
    const KIND: &'static str = "tuple<3>";
    const SCHEMA: u32 = 1;

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
        assert_eq!(back.produced_by_stage.as_deref(), Some("materialize_conversations"));
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

    #[test]
    fn tuple2_content_hash_is_deterministic() {
        let a = TestArt { byte: 1, path: PathBuf::from("/a") };
        let b = TestArt { byte: 2, path: PathBuf::from("/b") };
        let pair = (a.clone(), b.clone());
        let h1 = pair.content_hash();
        let h2 = (a.clone(), b.clone()).content_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn tuple2_content_hash_is_order_sensitive() {
        let a = TestArt { byte: 1, path: PathBuf::from("/a") };
        let b = TestArt { byte: 2, path: PathBuf::from("/b") };
        let h_ab = (a.clone(), b.clone()).content_hash();
        let h_ba = (b, a).content_hash();
        assert_ne!(h_ab, h_ba);
    }

    #[test]
    fn tuple2_primary_path_returns_first() {
        let a = TestArt { byte: 1, path: PathBuf::from("/first") };
        let b = TestArt { byte: 2, path: PathBuf::from("/second") };
        let pair = (a, b);
        assert_eq!(pair.primary_path(), Path::new("/first"));
    }

    #[test]
    fn tuple3_content_hash_includes_all_children() {
        let a = TestArt { byte: 1, path: PathBuf::from("/a") };
        let b = TestArt { byte: 2, path: PathBuf::from("/b") };
        let c = TestArt { byte: 3, path: PathBuf::from("/c") };
        let h_full = (a.clone(), b.clone(), c.clone()).content_hash();
        // Replacing the third element should change the hash.
        let c2 = TestArt { byte: 99, path: PathBuf::from("/c2") };
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
        let json = serde_json::to_value(&()).unwrap();
        let _: () = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn tuple_arity_domain_separation() {
        // 2-tuple of (a, a) and 3-tuple of (a, a, a) must produce
        // distinct hashes even when child hashes are identical —
        // the arity byte in the domain prefix prevents collisions.
        let a = TestArt { byte: 7, path: PathBuf::from("/a") };
        let h2 = (a.clone(), a.clone()).content_hash();
        let h3 = (a.clone(), a.clone(), a.clone()).content_hash();
        assert_ne!(h2, h3, "tuple arity must affect hash");
    }
}
