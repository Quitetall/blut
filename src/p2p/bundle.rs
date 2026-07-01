// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! P2P artifact-bundle transfer — the data plane.
//!
//! Every BLUT artifact is a FILE-BACKED HANDLE: its `ErasedArtifact` bincode
//! payload carries a `PathBuf` + a self-attested `content_hash`, NOT the bytes
//! at `Artifact::primary_path()`. Shipping that handle alone hands a remote
//! peer a path that doesn't exist on its disk. This module bundles the handle
//! TOGETHER with the backing file/dir bytes, so the peer can materialize the
//! input locally, re-point the handle's paths, and run the stage.
//!
//! ## Shape
//!
//! - [`BundleManifest`] is small (KBs): the erased handle + a per-file table
//!   (rel path + per-file hash) + the whole-pack hash. It rides the EXISTING
//!   `TaskManifest.encrypted_input` / `TaskResult.encrypted_output` slots
//!   (bincode → `crypto::encrypt`).
//! - The bulk **pack** (the actual file bytes) is produced by [`bundle`] and
//!   consumed by [`unbundle`] as a plain `Vec<u8>`. The transport layer
//!   (`transport::send_blob`/`recv_blob`) streams it in <64 MiB chunks (the
//!   QUIC `recv_message` cap) on a side channel and is framing-only — it does
//!   NOT encrypt (see its own doc comment). The caller (`peer_exec.rs`'s
//!   `seal_blob`/`open_blob`) seals the WHOLE plaintext pack with the same
//!   AES-256-GCM primitive used for the small manifest (`crypto::encrypt`)
//!   before handing bytes to `send_blob`, and opens it after `recv_blob`
//!   reassembles them. This module itself stays pack/unpack/VERIFY only —
//!   it never sees ciphertext, only the plaintext pack.
//!
//! ## Pack layout (deterministic, reproducible `blob_sha256`)
//!
//! ```text
//! for each BundleFile (sorted by rel):
//!     [u32 rel_len LE][rel utf8][u64 body_len LE][body bytes]
//! ```
//!
//! ## Fail-closed verification (all BEFORE the stage runs, in execution order)
//!
//! 1. `blob_sha256` over the received pack (fail-fast).
//! 2. Per-file `ContentHash` for EVERY shipped file + an exact file-COUNT check
//!    (catches a partial transfer, a dropped secondary path — e.g.
//!    `DatasetSplit.eval` — or an EXTRA unlisted file).
//! 3. Hard-error rebase: a coordinator/peer struct skew makes decode fail →
//!    reject, never a silent no-op leaving dead producer-absolute paths.
//! 4. Whole-artifact `recompute_content_hash()` on the REBASED handle (so it
//!    must run AFTER the rebase), asserted `== manifest.content_hash == signed
//!    input_hash`. Re-derives the address from disk the artifact's OWN way
//!    (`hash_dir` / merkle), independent of the self-attested per-file table —
//!    catches a never-shipped path AND a malicious peer that fixes per-file
//!    hashes but tampers structure.
//!
//! Any failure deletes the import dir and the stage is NEVER run.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::ContentHash;
use crate::framework::stage::{ErasedArtifact, StageDyn};

/// Bundle wire-format version. A mismatch is a hard reject (no silent skew).
pub const BUNDLE_VERSION: u16 = 1;

/// Which leg a blob belongs to (routes the transport side-stream).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlobDir {
    Input,
    Output,
}

/// One shipped file, relative to the bundle's `src_root`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleFile {
    /// Path relative to `src_root`, '/'-normalized. NUL-free, no `..`, not abs.
    pub rel: String,
    /// Unix permission bits (preserve +x for e.g. `export_firmware` outputs).
    pub mode: u32,
    /// Per-file SHA-256 — checked in-transit AND after rebase (total coverage).
    pub hash: ContentHash,
}

/// The small manifest that rides `encrypted_input`/`encrypted_output`.
/// Carries the handle + the per-file table; NEVER the bulk bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleManifest {
    /// `== BUNDLE_VERSION`; mismatch ⇒ hard reject.
    pub bundle_version: u16,
    /// Producer-local typed handle (paths still SENDER-absolute; rebased on
    /// arrival).
    pub erased: ErasedArtifact,
    /// `== erased.kind`; asserted against the dispatched stage's input kind.
    pub kind: String,
    /// `== erased.schema`.
    pub schema: u32,
    /// Whole-artifact content address. INVARIANT: `== TaskManifest.input_hash`
    /// (path-independent — safe because P2P only dispatches DETERMINISTIC
    /// stages, so the address never depends on a machine-specific path).
    pub content_hash: ContentHash,
    /// The producing stage_dir on the sender — the rebase-FROM prefix. Every
    /// shipped path is `src_root` or a descendant (stages write only under
    /// their own stage_dir).
    pub src_root: PathBuf,
    /// Every shipped FILE, rel-to-`src_root`, sorted by `rel`.
    pub files: Vec<BundleFile>,
    /// Total PLAINTEXT pack length.
    pub blob_len: u64,
    /// SHA-256 over the whole plaintext pack (fail-fast before unpack).
    pub blob_sha256: ContentHash,
}

/// Bundle / unbundle failures. Every variant is fail-closed: the peer never
/// runs a stage whose input didn't fully verify.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("bundle_version {got} unsupported (want {want})")]
    Version { want: u16, got: u16 },
    #[error("kind/schema skew: manifest {m_kind} v{m_schema}, stage wants {s_kind} v{s_schema}")]
    KindMismatch {
        m_kind: String,
        m_schema: u32,
        s_kind: String,
        s_schema: u32,
    },
    #[error("hash binding: bundle.content_hash {bundle} != signed hash {signed}")]
    HashBinding { bundle: String, signed: String },
    #[error("blob sha mismatch: got {got}, manifest {want}")]
    BlobHash { want: String, got: String },
    #[error("file {rel}: hash mismatch or missing after transfer")]
    FileHash { rel: String },
    #[error("rebase failed (decode/encode of the stage input) — refusing to run")]
    Rebase,
    #[error("whole-artifact recompute {got} != content_hash {want}")]
    ContentMismatch { want: String, got: String },
    #[error("unsafe rel path: {rel}")]
    UnsafePath { rel: String },
    #[error("artifact handle did not decode as the stage's input")]
    Undecodable,
    #[error("nothing to ship: no backing files under src_root {0}")]
    Empty(String),
    #[error("pack truncated or malformed at offset {0}")]
    MalformedPack(usize),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Collect the regular files under `p` (recursing dirs), as `(abs_path, rel)`
/// where `rel` is relative to `root`, '/'-normalized. Sidecars are EXCLUDED so
/// the reconstructed dir reproduces the producer's `hash_dir` (which ran before
/// the sidecar was written). Files are returned in sorted-`rel` order.
fn walk_backing(
    abs: &Path,
    root: &Path,
    out: &mut Vec<(PathBuf, String)>,
) -> std::io::Result<()> {
    if abs.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(abs)?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for child in entries {
            walk_backing(&child, root, out)?;
        }
    } else if abs.is_file() {
        if is_sidecar(abs) {
            return Ok(());
        }
        let rel = abs
            .strip_prefix(root)
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| abs.to_string_lossy().into_owned());
        out.push((abs.to_path_buf(), rel));
    }
    Ok(())
}

/// Artifact metadata sidecars are written AFTER the producer hashed the dir, so
/// they must be excluded from the bundle to reproduce `content_hash`. Matches
/// `ArtifactMetadata::sidecar_path_for`: `.lamu-meta.json` inside a dir, or a
/// `*.metadata.json` file beside a file artifact.
fn is_sidecar(p: &Path) -> bool {
    match p.file_name().and_then(|n| n.to_str()) {
        Some(".lamu-meta.json") => true,
        Some(name) => name.ends_with(".metadata.json"),
        None => false,
    }
}

#[cfg(unix)]
fn file_mode(p: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).map(|m| m.permissions().mode()).unwrap_or(0o644)
}
#[cfg(not(unix))]
fn file_mode(_p: &Path) -> u32 {
    0o644
}

/// Build a bundle for `erased` (this stage's input or output), shipping every
/// backing file under `src_root`. Returns the small manifest + the plaintext
/// pack the transport layer will chunk + encrypt.
///
/// `dir` selects which side of the stage (`Input` ⇒ decode as `S::Input`).
pub fn bundle(
    stage: &dyn StageDyn,
    erased: ErasedArtifact,
    src_root: &Path,
    dir: BlobDir,
    expected_hash: &ContentHash,
) -> Result<(BundleManifest, Vec<u8>), BundleError> {
    // 1. Discover backing paths via the type-aware StageDyn method.
    let backings = match dir {
        BlobDir::Input => stage.input_backing_under(&erased, src_root),
        BlobDir::Output => stage.output_backing_under(&erased, src_root),
    }
    .ok_or(BundleError::Undecodable)?;
    if backings.is_empty() {
        return Err(BundleError::Empty(src_root.display().to_string()));
    }

    // 2. Expand to a flat, sorted per-file table (recursing dirs).
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    for b in &backings {
        walk_backing(b, src_root, &mut found)?;
    }
    found.sort_by(|a, c| a.1.cmp(&c.1));
    found.dedup_by(|a, c| a.1 == c.1);
    if found.is_empty() {
        return Err(BundleError::Empty(src_root.display().to_string()));
    }

    let mut files = Vec::with_capacity(found.len());
    let mut pack: Vec<u8> = Vec::new();
    for (abs, rel) in &found {
        reject_unsafe_rel(rel)?;
        let body = std::fs::read(abs)?;
        let hash = ContentHash::of_bytes(&body);
        // [u32 rel_len][rel][u64 body_len][body]
        pack.extend_from_slice(&(rel.len() as u32).to_le_bytes());
        pack.extend_from_slice(rel.as_bytes());
        pack.extend_from_slice(&(body.len() as u64).to_le_bytes());
        pack.extend_from_slice(&body);
        files.push(BundleFile {
            rel: rel.clone(),
            mode: file_mode(abs),
            hash,
        });
    }

    // 3. The whole-artifact address is the coordinator-SIGNED `expected_hash`
    //    (the content-addressed cache key the dispatch was built from). We do
    //    NOT re-walk `primary_path()` on the sender: a metadata SIDECAR written
    //    into a dir artifact AFTER the producer hashed it would make a naive
    //    re-walk disagree with the canonical hash. The peer reproduces this hash
    //    from its sidecar-clean import dir (gate 3 in `unbundle`), which is the
    //    load-bearing byte-level check. Binding to `expected_hash` here keeps
    //    the manifest's address == the signed `input_hash` by construction.
    let content_hash = *expected_hash;

    let blob_sha256 = ContentHash::of_bytes(&pack);
    let manifest = BundleManifest {
        bundle_version: BUNDLE_VERSION,
        kind: erased.kind.clone(),
        schema: erased.schema,
        content_hash,
        src_root: src_root.to_path_buf(),
        files,
        blob_len: pack.len() as u64,
        blob_sha256,
        erased,
    };
    Ok((manifest, pack))
}

/// Materialize a received bundle into `into_stage_dir`, returning the rebased
/// erased handle (paths now peer-local, files verified) ready for `run_erased`.
///
/// `signed_hash` is the coordinator-signed `input_hash` (or `expected_output_hash`
/// on the output leg) the bundle is bound to. On ANY failure the import dir is
/// removed and the stage is never run.
pub fn unbundle(
    stage: &dyn StageDyn,
    manifest: &BundleManifest,
    pack: &[u8],
    into_stage_dir: &Path,
    signed_hash: &ContentHash,
    dir: BlobDir,
) -> Result<ErasedArtifact, BundleError> {
    // ── Pre-checks (no I/O) ───────────────────────────────────────────────
    if manifest.bundle_version != BUNDLE_VERSION {
        return Err(BundleError::Version {
            want: BUNDLE_VERSION,
            got: manifest.bundle_version,
        });
    }
    let (s_kind, s_schema) = (stage.input_kind(), stage.schema());
    // The stage's input kind/schema must match the manifest (fail-closed before
    // any I/O — never unpack bytes for a stage that can't consume them).
    if dir == BlobDir::Input && (manifest.kind != s_kind) {
        return Err(BundleError::KindMismatch {
            m_kind: manifest.kind.clone(),
            m_schema: manifest.schema,
            s_kind: s_kind.to_string(),
            s_schema,
        });
    }
    if manifest.content_hash != *signed_hash {
        return Err(BundleError::HashBinding {
            bundle: manifest.content_hash.to_hex(),
            signed: signed_hash.to_hex(),
        });
    }

    // Content-addressed, idempotent import dir.
    let import_root = into_stage_dir
        .join(".p2p-import")
        .join(manifest.content_hash.to_hex());

    // Wrap the body so any failure cleans up the import dir.
    match unbundle_inner(stage, manifest, pack, &import_root) {
        Ok(rebased) => Ok(rebased),
        Err(e) => {
            let _ = std::fs::remove_dir_all(&import_root);
            Err(e)
        }
    }
}

fn unbundle_inner(
    stage: &dyn StageDyn,
    manifest: &BundleManifest,
    pack: &[u8],
    import_root: &Path,
) -> Result<ErasedArtifact, BundleError> {
    // Gate 1: whole-pack hash (fail-fast before touching disk).
    let got = ContentHash::of_bytes(pack);
    if got != manifest.blob_sha256 {
        return Err(BundleError::BlobHash {
            want: manifest.blob_sha256.to_hex(),
            got: got.to_hex(),
        });
    }
    std::fs::create_dir_all(import_root)?;

    // Unpack the framed pack: [u32 rel_len][rel][u64 body_len][body]*
    // `len_usize` fails closed if a length exceeds usize (a malicious pack on a
    // 32-bit target — e.g. the riscv32 firmware build — claiming a >4 GiB body).
    let mut off = 0usize;
    let mut written: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
    while off < pack.len() {
        let at = off;
        let rel_len = len_usize(read_u32(pack, &mut off)? as u64, at)?;
        let rel = read_bytes(pack, &mut off, rel_len)?;
        let rel = String::from_utf8(rel).map_err(|_| BundleError::MalformedPack(off))?;
        let body_len = len_usize(read_u64(pack, &mut off)?, at)?;
        let body = read_bytes(pack, &mut off, body_len)?;
        reject_unsafe_rel(&rel)?;
        let dest = safe_join(import_root, &rel)?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Defense-in-depth: never write THROUGH a symlink. The import dir is
        // freshly created and we only ever write regular files, but a crafted
        // pack could ship a symlink-shaped entry earlier in the stream; refuse
        // to follow one.
        if dest.is_symlink() || dest.parent().map(|p| has_symlink_ancestor(p, import_root)).unwrap_or(false) {
            return Err(BundleError::UnsafePath { rel });
        }
        std::fs::write(&dest, &body)?;
        set_mode(&dest, manifest, &rel);
        written.insert(rel, body);
    }

    // Gate 2: per-file hash — total coverage (catches partial transfer / drop).
    // Plus an exact-count check so a pack with EXTRA files (written to disk but
    // absent from the manifest table) is rejected — closing the gap where gate 3
    // would be the only thing catching unlisted bytes.
    if written.len() != manifest.files.len() {
        return Err(BundleError::FileHash {
            rel: format!(
                "file count {} != manifest {} (extra or duplicate entries)",
                written.len(),
                manifest.files.len()
            ),
        });
    }
    for f in &manifest.files {
        match written.get(&f.rel) {
            Some(body) if ContentHash::of_bytes(body) == f.hash => {}
            _ => return Err(BundleError::FileHash { rel: f.rel.clone() }),
        }
    }

    // Gate 3: path rewrite (hard error on failure — never run with stale paths).
    let rebased = stage
        .rebase_input_paths(manifest.erased.clone(), &manifest.src_root, import_root)
        .ok_or(BundleError::Rebase)?;

    // Gate 4: whole-artifact recompute on the REBASED handle (re-walks
    // import_root the artifact's OWN way — hash_dir / merkle), asserted equal to
    // the bound content_hash. Independent of the self-attested per-file table.
    let recomputed = stage
        .recompute_input_hash(&rebased)
        .ok_or(BundleError::Undecodable)??;
    if recomputed != manifest.content_hash {
        return Err(BundleError::ContentMismatch {
            want: manifest.content_hash.to_hex(),
            got: recomputed.to_hex(),
        });
    }

    Ok(rebased)
}

#[cfg(unix)]
fn set_mode(dest: &Path, manifest: &BundleManifest, rel: &str) {
    use std::os::unix::fs::PermissionsExt;
    if let Some(f) = manifest.files.iter().find(|f| f.rel == rel) {
        // Apply ONLY the 0o777 permission bits from the sender; mask off
        // setuid/setgid/sticky (0o7000) so a malicious peer can't ship a
        // setuid binary into the import dir.
        let safe = f.mode & 0o777;
        let _ = std::fs::set_permissions(dest, std::fs::Permissions::from_mode(safe));
    }
}
#[cfg(not(unix))]
fn set_mode(_dest: &Path, _manifest: &BundleManifest, _rel: &str) {}

/// Reject a relative path that is empty, absolute, or contains a `..` / NUL
/// component (traversal hardening before it touches the filesystem).
fn reject_unsafe_rel(rel: &str) -> Result<(), BundleError> {
    if rel.is_empty()
        || rel.contains('\0')
        || rel.starts_with('/')
        || Path::new(rel)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::RootDir))
    {
        return Err(BundleError::UnsafePath { rel: rel.to_string() });
    }
    Ok(())
}

/// Join `rel` under `base`, then verify the result is still under `base` (defends
/// against symlink/`..` escape even past the component check).
fn safe_join(base: &Path, rel: &str) -> Result<PathBuf, BundleError> {
    reject_unsafe_rel(rel)?;
    let joined = base.join(rel);
    // Lexical containment is sufficient here: rel has no `..`/abs components, so
    // base.join(rel) cannot lexically escape base. (Symlinks inside the freshly
    // created import_root cannot exist — we only ever write regular files.)
    if !joined.starts_with(base) {
        return Err(BundleError::UnsafePath { rel: rel.to_string() });
    }
    Ok(joined)
}

/// Convert a wire length to `usize`, failing closed if it doesn't fit (a
/// malicious >usize length on a 32-bit target, e.g. the riscv32 firmware build).
fn len_usize(n: u64, at: usize) -> Result<usize, BundleError> {
    usize::try_from(n).map_err(|_| BundleError::MalformedPack(at))
}

/// True if any path component between `from` (exclusive of `stop`) is a symlink.
/// Walks ancestors up to (not including) `stop`; used to refuse writing through
/// a symlink an earlier pack entry might have planted.
fn has_symlink_ancestor(from: &Path, stop: &Path) -> bool {
    let mut cur = Some(from);
    while let Some(p) = cur {
        if p == stop {
            break;
        }
        if p.is_symlink() {
            return true;
        }
        cur = p.parent();
    }
    false
}

fn read_u32(buf: &[u8], off: &mut usize) -> Result<u32, BundleError> {
    let b = read_bytes(buf, off, 4)?;
    Ok(u32::from_le_bytes(b.try_into().unwrap()))
}
fn read_u64(buf: &[u8], off: &mut usize) -> Result<u64, BundleError> {
    let b = read_bytes(buf, off, 8)?;
    Ok(u64::from_le_bytes(b.try_into().unwrap()))
}
fn read_bytes(buf: &[u8], off: &mut usize, n: usize) -> Result<Vec<u8>, BundleError> {
    let end = off.checked_add(n).ok_or(BundleError::MalformedPack(*off))?;
    if end > buf.len() {
        return Err(BundleError::MalformedPack(*off));
    }
    let out = buf[*off..end].to_vec();
    *off = end;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::Artifact;
    use crate::framework::error::StageError;
    use crate::framework::resource::Resource;
    use crate::framework::stage::{Stage, StageContext};
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};

    // A file-backed artifact: a directory (so recompute uses hash_dir and the
    // bundle walks multiple files). The path is the dir root.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct DirArt {
        path: PathBuf,
        content_hash: ContentHash,
    }
    impl Artifact for DirArt {
        const KIND: &'static str = "test.dirart";
        const SCHEMA: u32 = 2;
        fn content_hash(&self) -> ContentHash {
            self.content_hash
        }
        fn primary_path(&self) -> &Path {
            &self.path
        }
    }

    struct DirStage;
    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct NoArgs;
    #[async_trait]
    impl Stage for DirStage {
        const NAME: &'static str = "dir_stage";
        const SCHEMA: u32 = 2;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = DirArt;
        type Output = DirArt;
        type Args = NoArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            input: Self::Input,
            _args: &Self::Args,
        ) -> Result<Self::Output, StageError> {
            Ok(input)
        }
    }

    /// Build a producer stage_dir with a multi-file artifact dir inside it,
    /// returning (src_root, the erased handle, its content hash).
    fn make_dir_artifact() -> (tempfile::TempDir, ErasedArtifact, ContentHash) {
        let src = tempfile::tempdir().unwrap();
        let art_dir = src.path().join("ckpt");
        std::fs::create_dir_all(art_dir.join("nested")).unwrap();
        std::fs::write(art_dir.join("config.json"), b"{\"x\":1}").unwrap();
        std::fs::write(art_dir.join("weights.bin"), vec![7u8; 4096]).unwrap();
        std::fs::write(art_dir.join("nested/tok.json"), b"tokens").unwrap();
        let hash = ContentHash::hash_dir(&art_dir).unwrap();
        let art = DirArt {
            path: art_dir,
            content_hash: hash,
        };
        let erased = ErasedArtifact::from_typed(&art).unwrap();
        (src, erased, hash)
    }

    #[test]
    fn round_trip_directory_artifact() {
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        assert_eq!(manifest.files.len(), 3, "all 3 files shipped");
        assert!(manifest.files.iter().all(|f| f.rel.starts_with("ckpt/")));

        let dest = tempfile::tempdir().unwrap();
        let rebased =
            unbundle(&stage, &manifest, &pack, dest.path(), &hash, BlobDir::Input).unwrap();
        let typed: DirArt = rebased.into_typed().unwrap();
        // Path re-rooted under the peer's import dir; files materialized; the
        // recomputed dir hash equals the producer's.
        assert!(typed.path.starts_with(dest.path()));
        assert!(typed.path.join("weights.bin").exists());
        assert_eq!(ContentHash::hash_dir(&typed.path).unwrap(), hash);
    }

    #[test]
    fn sidecar_is_excluded() {
        let stage = DirStage;
        let (src, _e, _h) = make_dir_artifact();
        let art_dir = src.path().join("ckpt");
        // Inject the sidecar AFTER the producer hashed the dir.
        std::fs::write(art_dir.join(".lamu-meta.json"), b"meta").unwrap();
        let hash = ContentHash::hash_dir(&art_dir).unwrap(); // hash_dir includes dotfiles
        // Re-derive the artifact + hash WITHOUT the sidecar (what the producer saw).
        std::fs::remove_file(art_dir.join(".lamu-meta.json")).unwrap();
        let clean_hash = ContentHash::hash_dir(&art_dir).unwrap();
        std::fs::write(art_dir.join(".lamu-meta.json"), b"meta").unwrap();
        let art = DirArt { path: art_dir, content_hash: clean_hash };
        let erased = ErasedArtifact::from_typed(&art).unwrap();
        let (manifest, _pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &clean_hash).unwrap();
        assert!(
            !manifest.files.iter().any(|f| f.rel.ends_with(".lamu-meta.json")),
            "sidecar must be excluded"
        );
        assert_ne!(hash, clean_hash, "sidecar would change hash_dir if included");
    }

    #[test]
    fn rejects_tampered_blob() {
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (manifest, mut pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        let mid = pack.len() / 2;
        pack[mid] ^= 0xFF; // flip a byte
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(&stage, &manifest, &pack, dest.path(), &hash, BlobDir::Input);
        assert!(matches!(r, Err(BundleError::BlobHash { .. })));
        // Import dir cleaned up on failure.
        assert!(!dest.path().join(".p2p-import").join(hash.to_hex()).exists());
    }

    #[test]
    fn rejects_malicious_table_with_tampered_bytes() {
        // The adversary tampers a file's BYTES but "fixes" its per-file hash in
        // the table. Per-file gate passes; the whole-artifact recompute (gate 3)
        // re-walks disk the artifact's OWN way and catches it.
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (mut manifest, _pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        // Re-pack with a tampered config.json + a matching per-file hash.
        let tampered_body = b"{\"x\":666}".to_vec();
        let tampered_hash = ContentHash::of_bytes(&tampered_body);
        let mut new_pack = Vec::new();
        for f in &mut manifest.files {
            let body = if f.rel.ends_with("config.json") {
                f.hash = tampered_hash; // "fix" the table to match the tamper
                tampered_body.clone()
            } else {
                // re-read original bytes from src
                std::fs::read(src.path().join(&f.rel)).unwrap()
            };
            new_pack.extend_from_slice(&(f.rel.len() as u32).to_le_bytes());
            new_pack.extend_from_slice(f.rel.as_bytes());
            new_pack.extend_from_slice(&(body.len() as u64).to_le_bytes());
            new_pack.extend_from_slice(&body);
        }
        manifest.blob_sha256 = ContentHash::of_bytes(&new_pack); // fix gate 1 too
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(&stage, &manifest, &new_pack, dest.path(), &hash, BlobDir::Input);
        assert!(
            matches!(r, Err(BundleError::ContentMismatch { .. })),
            "whole-artifact recompute must catch structural tamper, got {r:?}"
        );
    }

    #[test]
    fn rejects_hash_binding_mismatch() {
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        let dest = tempfile::tempdir().unwrap();
        let wrong = ContentHash([9u8; 32]);
        let r = unbundle(&stage, &manifest, &pack, dest.path(), &wrong, BlobDir::Input);
        assert!(matches!(r, Err(BundleError::HashBinding { .. })));
    }

    #[test]
    fn rejects_version_skew() {
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (mut manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        manifest.bundle_version = 999;
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(&stage, &manifest, &pack, dest.path(), &hash, BlobDir::Input);
        assert!(matches!(r, Err(BundleError::Version { .. })));
    }

    #[test]
    fn rejects_unsafe_rel_paths() {
        assert!(reject_unsafe_rel("../etc/passwd").is_err());
        assert!(reject_unsafe_rel("/abs").is_err());
        assert!(reject_unsafe_rel("").is_err());
        assert!(reject_unsafe_rel("a\0b").is_err());
        assert!(reject_unsafe_rel("ok/nested.json").is_ok());
    }

    #[test]
    fn rejects_extra_unlisted_file_in_pack() {
        // A malicious peer appends a file NOT in manifest.files. Gate 2's count
        // check rejects it before the stage runs.
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (mut manifest, mut pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        // Append an extra "ckpt/evil.sh" frame the manifest doesn't list.
        let rel = "ckpt/evil.sh";
        let body = b"#!/bin/sh\nrm -rf /\n";
        pack.extend_from_slice(&(rel.len() as u32).to_le_bytes());
        pack.extend_from_slice(rel.as_bytes());
        pack.extend_from_slice(&(body.len() as u64).to_le_bytes());
        pack.extend_from_slice(body);
        manifest.blob_sha256 = ContentHash::of_bytes(&pack); // re-fix gate 1
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(&stage, &manifest, &pack, dest.path(), &hash, BlobDir::Input);
        assert!(
            matches!(r, Err(BundleError::FileHash { .. })),
            "extra unlisted file must be rejected, got {r:?}"
        );
        assert!(!dest.path().join(".p2p-import").join(hash.to_hex()).exists());
    }

    #[cfg(unix)]
    #[test]
    fn setuid_bits_are_stripped() {
        use std::os::unix::fs::PermissionsExt;
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (mut manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        // Force a setuid+exec mode on one file.
        for f in &mut manifest.files {
            f.mode = 0o4755; // setuid rwxr-xr-x
        }
        let dest = tempfile::tempdir().unwrap();
        let rebased =
            unbundle(&stage, &manifest, &pack, dest.path(), &hash, BlobDir::Input).unwrap();
        let typed: DirArt = rebased.into_typed().unwrap();
        let mode = std::fs::metadata(typed.path.join("config.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o7000, 0, "setuid/setgid/sticky must be stripped");
        assert_eq!(mode & 0o777, 0o755, "permission bits preserved");
    }

    #[test]
    fn idempotent_unbundle() {
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, &hash).unwrap();
        let dest = tempfile::tempdir().unwrap();
        let a = unbundle(&stage, &manifest, &pack, dest.path(), &hash, BlobDir::Input).unwrap();
        let b = unbundle(&stage, &manifest, &pack, dest.path(), &hash, BlobDir::Input).unwrap();
        let pa: DirArt = a.into_typed().unwrap();
        let pb: DirArt = b.into_typed().unwrap();
        assert_eq!(pa.path, pb.path, "same content_hash → same import dir");
        assert_eq!(ContentHash::hash_dir(&pb.path).unwrap(), hash);
    }
}
