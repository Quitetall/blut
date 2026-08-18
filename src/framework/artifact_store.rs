// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Canonical content-addressed artifact persistence.
//!
//! Every BLUT artifact is a file-backed handle: its [`ErasedArtifact`] payload
//! carries producer-local paths and a typed content hash, not the backing bytes.
//! Persisting that handle alone leaves cache and remote consumers with stale
//! paths. This module captures the handle together with its backing file or
//! directory bytes, validates their typed contract, and rehydrates the handle
//! beneath a consumer-owned directory.
//!
//! [`ArtifactManifest`] contains the handle, a complete per-file integrity table,
//! and a typed [`ArtifactContentId`]. [`StoredArtifact`] combines it with the deterministic
//! payload pack for cache or object-store persistence. Streaming transports use
//! [`bundle`] and [`unbundle`] to carry the same representation as two pieces;
//! encryption and framing remain adapters and do not own artifact semantics.
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
//! 4. Store-owned [`ArtifactContentId`] recomputation over the canonical kind, schema,
//!    relative paths, modes, and bytes. This identity never depends on a producer
//!    path, mtime, or invocation key. Artifacts whose logical hash is byte-based
//!    additionally rerun their typed `recompute_content_hash()` law.
//!
//! Any failure deletes the import dir and the stage is NEVER run.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::{ArtifactContentHasher, ArtifactContentId, ContentHash};
use crate::framework::object_store::MAX_OBJECT_SIZE;
use crate::framework::stage::{ErasedArtifact, StageDyn};

/// Artifact persistence format. Version 3 binds persisted artifacts to ABIR's
/// canonical training-artifact `ArtifactContentId` domain.
pub const ARTIFACT_FORMAT_VERSION: u16 = 3;
// Keep this explicit: a future format version must not silently reinterpret
// its identity field using version 2's SHA-256 derivation.
const LEGACY_ARTIFACT_FORMAT_VERSION: u16 = 2;
/// Compatibility name retained for P2P callers during the 7.8 bridge.
pub const BUNDLE_VERSION: u16 = ARTIFACT_FORMAT_VERSION;
const PORTABLE_ROOT: &str = "__blut_artifact_root_v2__";
const LEGACY_CONTENT_ID_DOMAIN: &[u8] = b"blut.artifact.content.v1";
const UNPERSISTED_ID_DOMAIN: &[u8] = b"blut.artifact.unpersisted.v1";

/// Which typed side of a stage the persisted artifact represents.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArtifactRole {
    Input,
    Output,
}
/// Compatibility name retained for transport callers.
pub type BlobDir = ArtifactRole;

/// One persisted file, relative to the artifact's canonical common root.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactFile {
    /// Path relative to `src_root`, '/'-normalized. NUL-free, no `..`, not abs.
    pub rel: String,
    /// Unix permission bits (preserve +x for e.g. `export_firmware` outputs).
    pub mode: u32,
    /// Per-file SHA-256 — checked in-transit AND after rebase (total coverage).
    pub hash: ContentHash,
}
/// Compatibility name retained for transport callers.
pub type BundleFile = ArtifactFile;

/// Path-bearing handle metadata plus the complete per-file integrity table.
/// Callers restore this manifest before observing its handle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactManifest {
    /// `== ARTIFACT_FORMAT_VERSION`; mismatch is a hard reject.
    pub format_version: u16,
    /// Producer-local typed handle (paths still SENDER-absolute; rebased on
    /// arrival).
    pub erased: ErasedArtifact,
    /// `== erased.kind`; asserted against the dispatched stage's input kind.
    pub kind: String,
    /// `== erased.schema`.
    pub schema: u32,
    /// Whole-artifact identity derived from the typed artifact, never from an
    /// invocation key or serialized path-bearing handle.
    pub content_id: ArtifactContentId,
    /// Existing artifact-level hash used by invocation-key derivation. This may
    /// intentionally be a producer-local stat fingerprint when
    /// `Artifact::HASH_CONTENTS` is false; it is not the portable object key.
    pub logical_hash: ContentHash,
    /// Synthetic path root embedded in `erased`. It contains no producer-local
    /// location and is replaced with the consumer import root during restore.
    pub handle_root: PathBuf,
    /// Every persisted file, relative to the producer root, sorted by `rel`.
    pub files: Vec<ArtifactFile>,
    /// Total PLAINTEXT pack length.
    pub blob_len: u64,
    /// SHA-256 over the whole plaintext pack (fail-fast before unpack).
    pub blob_sha256: ContentHash,
}

impl ArtifactManifest {
    /// Canonical content identity carried by version 3 manifests. Version 2
    /// retained a SHA-256-derived legacy object key in the same wire slot;
    /// unknown future versions must be interpreted only by their own reader.
    pub const fn semantic_content_id(&self) -> Option<ArtifactContentId> {
        if self.format_version == ARTIFACT_FORMAT_VERSION {
            Some(self.content_id)
        } else {
            None
        }
    }
}
/// Compatibility name retained for transport callers.
pub type BundleManifest = ArtifactManifest;

/// Canonical persisted value stored under [`ArtifactManifest::content_id`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredArtifact {
    pub manifest: ArtifactManifest,
    pub pack: Vec<u8>,
}

/// Capture / restore failures. No handle is returned until the typed artifact
/// and every persisted byte have verified.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactStoreError {
    #[error("artifact format version {got} unsupported (want {want})")]
    Version { want: u16, got: u16 },
    #[error("kind/schema skew: manifest {m_kind} v{m_schema}, stage wants {s_kind} v{s_schema}")]
    KindMismatch {
        m_kind: String,
        m_schema: u32,
        s_kind: String,
        s_schema: u32,
    },
    #[error("identity binding: artifact {actual} != expected {expected}")]
    IdentityBinding { actual: String, expected: String },
    #[error("blob sha mismatch: got {got}, manifest {want}")]
    BlobHash { want: String, got: String },
    #[error("blob length {got} != manifest {want}")]
    BlobLength { want: u64, got: u64 },
    #[error("file {rel}: hash mismatch or missing after transfer")]
    FileHash { rel: String },
    #[error("rebase failed (decode/encode of the typed artifact) — refusing to return a handle")]
    Rebase,
    #[error("whole-artifact recompute {got} != content identity {want}")]
    ContentMismatch { want: String, got: String },
    #[error("unsafe rel path: {rel}")]
    UnsafePath { rel: String },
    #[error("artifact handle did not decode as the selected stage role")]
    Undecodable,
    #[error("artifact is not portable: {0}")]
    NonPortable(String),
    #[error("nothing to ship: no backing files under src_root {0}")]
    Empty(String),
    #[error("pack truncated or malformed at offset {0}")]
    MalformedPack(usize),
    #[error("artifact pack is {size} bytes, above its {max}-byte bound")]
    TooLarge { size: u64, max: u64 },
    #[error("cannot reserve {size} bytes for artifact pack: {reason}")]
    Allocation { size: u64, reason: String },
    #[error("artifact restore failed ({primary}); cleanup also failed: {cleanup}")]
    Cleanup {
        primary: String,
        #[source]
        cleanup: std::io::Error,
    },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
/// Compatibility name retained for transport callers.
pub type BundleError = ArtifactStoreError;

/// Collect the regular files under `p` (recursing dirs), as `(abs_path, rel)`
/// where `rel` is relative to `root`, '/'-normalized. Sidecars are EXCLUDED so
/// the reconstructed dir reproduces the producer's `hash_dir` (which ran before
/// the sidecar was written). Files are returned in sorted-`rel` order.
fn walk_backing(abs: &Path, root: &Path, out: &mut Vec<(PathBuf, String)>) -> std::io::Result<()> {
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
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "artifact backing {} is outside producer root {}",
                        abs.display(),
                        root.display()
                    ),
                )
            })?;
        out.push((abs.to_path_buf(), rel));
    }
    Ok(())
}

/// Pick a relocation-stable root for the artifact rather than hashing the
/// caller's stage-directory wrappers. A single directory roots at itself; files
/// root at their parent; multiple backings use their deepest common ancestor.
/// Thus `stage/result.txt` and
/// `stage/.artifact-import/<id>/result.txt` both canonicalize to `result.txt`.
fn canonical_artifact_root(backings: &[PathBuf]) -> std::io::Result<PathBuf> {
    let mut bases = backings.iter().map(|path| {
        if path.is_dir() {
            path.as_path()
        } else {
            path.parent().unwrap_or(path.as_path())
        }
    });
    let mut root = bases
        .next()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "artifact has no backing paths",
            )
        })?
        .to_path_buf();
    for base in bases {
        while !base.starts_with(&root) {
            if !root.pop() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "artifact backing paths have no common root",
                ));
            }
        }
    }
    Ok(root)
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
    std::fs::metadata(p)
        .map(|m| m.permissions().mode())
        .unwrap_or(0o644)
}
#[cfg(not(unix))]
fn file_mode(_p: &Path) -> u32 {
    0o644
}

/// Capture a typed artifact and all of its backing bytes. The optional expected
/// identity is only for values that are analytically known; it is never used to
/// manufacture the returned identity.
pub fn capture(
    stage: &dyn StageDyn,
    erased: ErasedArtifact,
    src_root: &Path,
    role: ArtifactRole,
    expected_content_id: Option<ArtifactContentId>,
) -> Result<StoredArtifact, ArtifactStoreError> {
    let (manifest, pack) = bundle(stage, erased, src_root, role, expected_content_id)?;
    Ok(StoredArtifact { manifest, pack })
}

/// Restore a stored artifact beneath a consumer-owned directory. The returned
/// handle contains only consumer-local paths and has been independently rehashed.
pub fn restore(
    stage: &dyn StageDyn,
    stored: &StoredArtifact,
    into_stage_dir: &Path,
    role: ArtifactRole,
    expected_content_id: Option<ArtifactContentId>,
) -> Result<ErasedArtifact, ArtifactStoreError> {
    unbundle(
        stage,
        &stored.manifest,
        &stored.pack,
        into_stage_dir,
        expected_content_id,
        role,
    )
}

/// Typed lineage identity for an output that could not be captured by the
/// portable store (typically because it contains an externally managed corpus
/// locator outside the stage root). This value is deliberately domain-separated
/// from persisted object IDs and is never published as an object-store key or
/// direct invocation-to-object mapping. It may contribute to a downstream
/// [`InvocationKey`](crate::framework::artifact::InvocationKey), which safely
/// reuses derived work when the external artifact's logical hash is unchanged.
/// Artifacts using stat fingerprints intentionally produce host-local identities
/// and therefore conservatively miss shared downstream caches across machines.
/// `UNPERSISTED_ID_DOMAIN` is an input tag inside ABIR's canonical
/// training-artifact domain, not a competing outer hash domain. Changing this
/// subtype tag or canonical feed safely invalidates downstream cache reuse.
pub fn unpersisted_content_id(
    stage: &dyn StageDyn,
    erased: &ErasedArtifact,
    role: ArtifactRole,
    logical_hash: ContentHash,
) -> ArtifactContentId {
    let identity = match role {
        ArtifactRole::Input => stage.input_portable_identity(erased),
        ArtifactRole::Output => stage.output_portable_identity(erased),
    };
    let mut hasher = ArtifactContentHasher::new();
    hasher.update(UNPERSISTED_ID_DOMAIN);
    hasher.update(&(erased.kind.len() as u64).to_le_bytes());
    hasher.update(erased.kind.as_bytes());
    hasher.update(&erased.schema.to_le_bytes());
    hasher.update(&logical_hash.0);
    if let Some(identity) = identity {
        hasher.update(&(identity.len() as u64).to_le_bytes());
        hasher.update(&identity);
    }
    hasher.finalize()
}

/// Streaming form of [`capture`]: return the manifest and plaintext pack as
/// separate values so a transport can frame or encrypt them independently.
pub fn bundle(
    stage: &dyn StageDyn,
    erased: ErasedArtifact,
    src_root: &Path,
    role: ArtifactRole,
    expected_content_id: Option<ArtifactContentId>,
) -> Result<(ArtifactManifest, Vec<u8>), ArtifactStoreError> {
    let (
        expected_kind,
        expected_schema,
        logical_hash,
        backings,
        contains_absolute_paths,
        primary_path,
        inline,
        allow_external_paths,
    ) = match role {
        ArtifactRole::Input => (
            stage.input_kind(),
            stage.input_schema(),
            stage.input_content_hash(&erased),
            stage.input_backing_under(&erased, src_root),
            stage.input_contains_absolute_paths(&erased),
            stage.input_primary_path(&erased),
            stage.input_inline(),
            stage.input_allows_external_paths(),
        ),
        ArtifactRole::Output => (
            stage.output_kind(),
            stage.output_schema(),
            stage.output_content_hash(&erased),
            stage.output_backing_under(&erased, src_root),
            stage.output_contains_absolute_paths(&erased),
            stage.output_primary_path(&erased),
            stage.output_inline(),
            stage.output_allows_external_paths(),
        ),
    };
    if erased.kind != expected_kind || erased.schema != expected_schema {
        return Err(ArtifactStoreError::KindMismatch {
            m_kind: erased.kind,
            m_schema: erased.schema,
            s_kind: expected_kind.to_string(),
            s_schema: expected_schema,
        });
    }
    let logical_hash = logical_hash.ok_or(ArtifactStoreError::Undecodable)?;

    let backings = backings.ok_or(ArtifactStoreError::Undecodable)?;
    let contains_absolute_paths = contains_absolute_paths.ok_or(ArtifactStoreError::Undecodable)?;
    let primary_path = primary_path.ok_or(ArtifactStoreError::Undecodable)?;
    let unresolved_relative_primary = primary_path.is_relative() && primary_path != Path::new(".");
    if backings.is_empty() && !inline && (contains_absolute_paths || unresolved_relative_primary) {
        return if contains_absolute_paths && allow_external_paths {
            Err(ArtifactStoreError::NonPortable(format!(
                "absolute locator is outside owned root {}",
                src_root.display()
            )))
        } else {
            Err(ArtifactStoreError::Empty(src_root.display().to_string()))
        };
    }

    let artifact_root = if backings.is_empty() {
        src_root.to_path_buf()
    } else {
        canonical_artifact_root(&backings)?
    };
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    for backing in &backings {
        walk_backing(backing, &artifact_root, &mut found)?;
    }
    found.sort_by(|a, b| a.1.cmp(&b.1));
    found.dedup_by(|a, b| a.1 == b.1);
    reject_case_collisions(found.iter().map(|(_, rel)| rel.as_str()))?;
    if found.is_empty() && !backings.is_empty() {
        return Err(ArtifactStoreError::Empty(src_root.display().to_string()));
    }

    let mut planned_pack_len = 0_u64;
    let mut planned = Vec::with_capacity(found.len());
    for (abs, rel) in &found {
        reject_unsafe_rel(rel)?;
        let rel_len = u32::try_from(rel.len()).map_err(|_| ArtifactStoreError::TooLarge {
            size: rel.len() as u64,
            max: u32::MAX as u64,
        })?;
        let body_len = std::fs::metadata(abs)?.len();
        let frame_len = 4_u64
            .checked_add(rel.len() as u64)
            .and_then(|size| size.checked_add(8))
            .and_then(|size| size.checked_add(body_len))
            .ok_or(ArtifactStoreError::TooLarge {
                size: u64::MAX,
                max: MAX_OBJECT_SIZE,
            })?;
        planned_pack_len =
            planned_pack_len
                .checked_add(frame_len)
                .ok_or(ArtifactStoreError::TooLarge {
                    size: u64::MAX,
                    max: MAX_OBJECT_SIZE,
                })?;
        if planned_pack_len > MAX_OBJECT_SIZE {
            return Err(ArtifactStoreError::TooLarge {
                size: planned_pack_len,
                max: MAX_OBJECT_SIZE,
            });
        }
        planned.push((abs, rel, rel_len, body_len));
    }

    let pack_capacity =
        usize::try_from(planned_pack_len).map_err(|_| ArtifactStoreError::TooLarge {
            size: planned_pack_len,
            max: usize::MAX as u64,
        })?;
    let mut pack = Vec::new();
    pack.try_reserve_exact(pack_capacity)
        .map_err(|error| ArtifactStoreError::Allocation {
            size: planned_pack_len,
            reason: error.to_string(),
        })?;
    let mut files = Vec::with_capacity(planned.len());
    for (abs, rel, rel_len, expected_body_len) in planned {
        use std::io::Read;

        pack.extend_from_slice(&rel_len.to_le_bytes());
        pack.extend_from_slice(rel.as_bytes());
        pack.extend_from_slice(&expected_body_len.to_le_bytes());
        let body_start = pack.len();
        std::fs::File::open(abs)?
            .take(expected_body_len.saturating_add(1))
            .read_to_end(&mut pack)?;
        let actual_body_len = (pack.len() - body_start) as u64;
        if actual_body_len != expected_body_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "artifact backing {} changed during capture: expected {expected_body_len} bytes, read {actual_body_len}",
                    abs.display()
                ),
            )
            .into());
        }
        let body = &pack[body_start..];
        files.push(ArtifactFile {
            rel: rel.clone(),
            mode: file_mode(abs),
            hash: ContentHash::of_bytes(body),
        });
    }

    let handle_root = PathBuf::from(PORTABLE_ROOT);
    let portable_erased = match role {
        ArtifactRole::Input => stage.rebase_input_paths(erased, &artifact_root, &handle_root),
        ArtifactRole::Output => {
            stage.rebase_output_paths_checked(erased, &artifact_root, &handle_root)
        }
    }
    .ok_or(ArtifactStoreError::Rebase)?;
    let portable_contains_absolute_paths = match role {
        ArtifactRole::Input => stage.input_contains_absolute_paths(&portable_erased),
        ArtifactRole::Output => stage.output_contains_absolute_paths(&portable_erased),
    }
    .ok_or(ArtifactStoreError::Undecodable)?;
    if portable_contains_absolute_paths && !inline {
        return if allow_external_paths {
            Err(ArtifactStoreError::NonPortable(format!(
                "one or more absolute locators remain outside owned root {}",
                src_root.display()
            )))
        } else {
            Err(ArtifactStoreError::Rebase)
        };
    }

    let content_id = derive_content_id(
        stage,
        role,
        &portable_erased.kind,
        portable_erased.schema,
        &files,
        &pack,
        &portable_erased,
    )?;
    if let Some(expected) = expected_content_id
        && expected != content_id
    {
        return Err(ArtifactStoreError::IdentityBinding {
            actual: content_id.to_hex(),
            expected: expected.to_hex(),
        });
    }

    let manifest = ArtifactManifest {
        format_version: ARTIFACT_FORMAT_VERSION,
        kind: portable_erased.kind.clone(),
        schema: portable_erased.schema,
        content_id,
        logical_hash,
        handle_root,
        files,
        blob_len: pack.len() as u64,
        blob_sha256: ContentHash::of_bytes(&pack),
        erased: portable_erased,
    };
    Ok((manifest, pack))
}

/// Streaming form of [`restore`]. Any failure removes the consumer import dir.
pub fn unbundle(
    stage: &dyn StageDyn,
    manifest: &ArtifactManifest,
    pack: &[u8],
    into_stage_dir: &Path,
    expected_content_id: Option<ArtifactContentId>,
    role: ArtifactRole,
) -> Result<ErasedArtifact, ArtifactStoreError> {
    if !matches!(
        manifest.format_version,
        LEGACY_ARTIFACT_FORMAT_VERSION | ARTIFACT_FORMAT_VERSION
    ) {
        return Err(ArtifactStoreError::Version {
            want: ARTIFACT_FORMAT_VERSION,
            got: manifest.format_version,
        });
    }
    if manifest.handle_root != Path::new(PORTABLE_ROOT) {
        return Err(ArtifactStoreError::Rebase);
    }
    let (expected_kind, expected_schema, inline) = match role {
        ArtifactRole::Input => (
            stage.input_kind(),
            stage.input_schema(),
            stage.input_inline(),
        ),
        ArtifactRole::Output => (
            stage.output_kind(),
            stage.output_schema(),
            stage.output_inline(),
        ),
    };
    if manifest.kind != expected_kind
        || manifest.schema != expected_schema
        || manifest.erased.kind != manifest.kind
        || manifest.erased.schema != manifest.schema
    {
        return Err(ArtifactStoreError::KindMismatch {
            m_kind: manifest.kind.clone(),
            m_schema: manifest.schema,
            s_kind: expected_kind.to_string(),
            s_schema: expected_schema,
        });
    }
    let manifest_contains_absolute_paths = match role {
        ArtifactRole::Input => stage.input_contains_absolute_paths(&manifest.erased),
        ArtifactRole::Output => stage.output_contains_absolute_paths(&manifest.erased),
    }
    .ok_or(ArtifactStoreError::Undecodable)?;
    if manifest_contains_absolute_paths && !inline {
        return Err(ArtifactStoreError::Rebase);
    }
    reject_case_collisions(manifest.files.iter().map(|file| file.rel.as_str()))?;
    if let Some(expected) = expected_content_id
        && expected != manifest.content_id
    {
        return Err(ArtifactStoreError::IdentityBinding {
            actual: manifest.content_id.to_hex(),
            expected: expected.to_hex(),
        });
    }

    let import_root = into_stage_dir
        .join(".artifact-import")
        .join(manifest.content_id.to_hex());
    match unbundle_inner(stage, manifest, pack, &import_root, role) {
        Ok(rebased) => Ok(rebased),
        Err(primary) => match std::fs::remove_dir_all(&import_root) {
            Ok(()) => Err(primary),
            Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => Err(primary),
            Err(cleanup) => Err(ArtifactStoreError::Cleanup {
                primary: primary.to_string(),
                cleanup,
            }),
        },
    }
}

fn unbundle_inner(
    stage: &dyn StageDyn,
    manifest: &ArtifactManifest,
    pack: &[u8],
    import_root: &Path,
    role: ArtifactRole,
) -> Result<ErasedArtifact, ArtifactStoreError> {
    // Gate 1: whole-pack hash (fail-fast before touching disk).
    let got = ContentHash::of_bytes(pack);
    if got != manifest.blob_sha256 {
        return Err(ArtifactStoreError::BlobHash {
            want: manifest.blob_sha256.to_hex(),
            got: got.to_hex(),
        });
    }
    if pack.len() as u64 != manifest.blob_len {
        return Err(ArtifactStoreError::BlobLength {
            want: manifest.blob_len,
            got: pack.len() as u64,
        });
    }

    // The object key is recomputed from the representation the receiver
    // actually received, not trusted from the manifest or typed handle.
    let derived_content_id = if manifest.format_version == LEGACY_ARTIFACT_FORMAT_VERSION {
        derive_legacy_content_id(
            stage,
            role,
            &manifest.kind,
            manifest.schema,
            &manifest.files,
            pack,
            &manifest.erased,
        )?
    } else {
        derive_content_id(
            stage,
            role,
            &manifest.kind,
            manifest.schema,
            &manifest.files,
            pack,
            &manifest.erased,
        )?
    };
    if derived_content_id != manifest.content_id {
        return Err(ArtifactStoreError::ContentMismatch {
            want: manifest.content_id.to_hex(),
            got: derived_content_id.to_hex(),
        });
    }
    match std::fs::remove_dir_all(import_root) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    std::fs::create_dir_all(import_root)?;

    // Unpack the framed pack: [u32 rel_len][rel][u64 body_len][body]*
    // `len_usize` fails closed if a length exceeds usize (a malicious pack on a
    // 32-bit target — e.g. the riscv32 firmware build — claiming a >4 GiB body).
    let mut off = 0usize;
    let mut written: std::collections::HashMap<String, ContentHash> =
        std::collections::HashMap::new();
    let mut written_casefold = std::collections::HashSet::new();
    while off < pack.len() {
        let at = off;
        let rel_len = len_usize(read_u32(pack, &mut off)? as u64, at)?;
        let rel = read_bytes(pack, &mut off, rel_len)?;
        let rel = String::from_utf8(rel).map_err(|_| ArtifactStoreError::MalformedPack(off))?;
        let body_len = len_usize(read_u64(pack, &mut off)?, at)?;
        let body = read_bytes(pack, &mut off, body_len)?;
        reject_unsafe_rel(&rel)?;
        if !written_casefold.insert(rel.to_lowercase()) {
            return Err(ArtifactStoreError::UnsafePath {
                rel: format!("case-colliding path: {rel}"),
            });
        }
        let dest = safe_join(import_root, &rel)?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Defense-in-depth: never write THROUGH a symlink. The import dir is
        // freshly created and we only ever write regular files, but a crafted
        // pack could ship a symlink-shaped entry earlier in the stream; refuse
        // to follow one.
        if dest.is_symlink()
            || dest
                .parent()
                .map(|p| has_symlink_ancestor(p, import_root))
                .unwrap_or(false)
        {
            return Err(ArtifactStoreError::UnsafePath { rel });
        }
        let body_hash = ContentHash::of_bytes(&body);
        std::fs::write(&dest, &body)?;
        set_mode(&dest, manifest, &rel)?;
        written.insert(rel, body_hash);
    }

    // Gate 2: per-file hash — total coverage (catches partial transfer / drop).
    // Plus an exact-count check so a pack with EXTRA files (written to disk but
    // absent from the manifest table) is rejected — closing the gap where gate 3
    // would be the only thing catching unlisted bytes.
    if written.len() != manifest.files.len() {
        return Err(ArtifactStoreError::FileHash {
            rel: format!(
                "file count {} != manifest {} (extra or duplicate entries)",
                written.len(),
                manifest.files.len()
            ),
        });
    }
    for f in &manifest.files {
        match written.get(&f.rel) {
            Some(hash) if *hash == f.hash => {}
            _ => return Err(ArtifactStoreError::FileHash { rel: f.rel.clone() }),
        }
    }

    // Gate 3: directional path rewrite. Both roles fail closed; cache restore
    // must never return an output handle that still points at the producer.
    let rebased = match role {
        ArtifactRole::Input => {
            stage.rebase_input_paths(manifest.erased.clone(), &manifest.handle_root, import_root)
        }
        ArtifactRole::Output => stage.rebase_output_paths_checked(
            manifest.erased.clone(),
            &manifest.handle_root,
            import_root,
        ),
    }
    .ok_or(ArtifactStoreError::Rebase)?;

    // Gate 4a: the rebased typed handle must preserve its documented logical
    // hash. For stat-fingerprint artifacts this validates the carried logical
    // identity without pretending it is a portable object address.
    let logical_hash = match role {
        ArtifactRole::Input => stage.input_content_hash(&rebased),
        ArtifactRole::Output => stage.output_content_hash(&rebased),
    }
    .ok_or(ArtifactStoreError::Undecodable)?;
    if logical_hash != manifest.logical_hash {
        return Err(ArtifactStoreError::ContentMismatch {
            want: manifest.logical_hash.to_hex(),
            got: logical_hash.to_hex(),
        });
    }

    // Gate 4b: when the artifact declares a byte-derived logical hash, rerun its
    // typed law over the consumer-local files. `HASH_CONTENTS = false` artifacts
    // deliberately retain a path/stat logical hash; their complete bytes were
    // already bound independently by ArtifactContentId + the per-file table above.
    let hashes_contents = match role {
        ArtifactRole::Input => stage.input_hashes_contents(),
        ArtifactRole::Output => stage.output_hashes_contents(),
    };
    if !manifest.files.is_empty() && hashes_contents {
        let recomputed = match role {
            ArtifactRole::Input => stage.recompute_input_hash(&rebased),
            ArtifactRole::Output => stage.recompute_output_hash(&rebased),
        }
        .ok_or(ArtifactStoreError::Undecodable)??;
        if recomputed != manifest.logical_hash {
            return Err(ArtifactStoreError::ContentMismatch {
                want: manifest.logical_hash.to_hex(),
                got: recomputed.to_hex(),
            });
        }
    }

    Ok(rebased)
}

/// Derive the portable object identity from the canonical persisted
/// representation. The stage supplies a canonical typed metadata projection
/// with absolute paths normalized and the legacy `content_hash` field removed;
/// file bytes and relative layout are then added independently.
fn derive_content_id(
    stage: &dyn StageDyn,
    role: ArtifactRole,
    kind: &str,
    schema: u32,
    files: &[ArtifactFile],
    pack: &[u8],
    erased: &ErasedArtifact,
) -> Result<ArtifactContentId, ArtifactStoreError> {
    let identity = portable_identity(stage, role, erased)?;
    let mut hasher = ArtifactContentHasher::new();
    feed_artifact_identity(
        |bytes| hasher.update(bytes),
        kind,
        schema,
        &identity,
        files,
        pack,
    );
    Ok(hasher.finalize())
}

fn derive_legacy_content_id(
    stage: &dyn StageDyn,
    role: ArtifactRole,
    kind: &str,
    schema: u32,
    files: &[ArtifactFile],
    pack: &[u8],
    erased: &ErasedArtifact,
) -> Result<ArtifactContentId, ArtifactStoreError> {
    use sha2::{Digest, Sha256};
    let identity = portable_identity(stage, role, erased)?;
    let mut hasher = Sha256::new();
    hasher.update(LEGACY_CONTENT_ID_DOMAIN);
    feed_artifact_identity(
        |bytes| hasher.update(bytes),
        kind,
        schema,
        &identity,
        files,
        pack,
    );
    Ok(ArtifactContentId::from_digest(ContentHash(
        hasher.finalize().into(),
    )))
}

fn portable_identity(
    stage: &dyn StageDyn,
    role: ArtifactRole,
    erased: &ErasedArtifact,
) -> Result<Vec<u8>, ArtifactStoreError> {
    match role {
        ArtifactRole::Input => stage.input_portable_identity(erased),
        ArtifactRole::Output => stage.output_portable_identity(erased),
    }
    .ok_or(ArtifactStoreError::Undecodable)
}

fn feed_artifact_identity(
    mut update: impl FnMut(&[u8]),
    kind: &str,
    schema: u32,
    identity: &[u8],
    files: &[ArtifactFile],
    pack: &[u8],
) {
    update(&(kind.len() as u64).to_le_bytes());
    update(kind.as_bytes());
    update(&schema.to_le_bytes());
    update(&(identity.len() as u64).to_le_bytes());
    update(identity);
    if files.is_empty() {
        update(&[0]);
    } else {
        update(&[1]);
        update(&(files.len() as u64).to_le_bytes());
        for file in files {
            update(&(file.rel.len() as u64).to_le_bytes());
            update(file.rel.as_bytes());
            update(&(file.mode & 0o777).to_le_bytes());
        }
        update(&(pack.len() as u64).to_le_bytes());
        update(pack);
    }
}

#[cfg(unix)]
fn set_mode(dest: &Path, manifest: &BundleManifest, rel: &str) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    if let Some(f) = manifest.files.iter().find(|f| f.rel == rel) {
        // Apply ONLY the 0o777 permission bits from the sender; mask off
        // setuid/setgid/sticky (0o7000) so a malicious peer can't ship a
        // setuid binary into the import dir.
        let safe = f.mode & 0o777;
        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(safe))?;
    }
    Ok(())
}
#[cfg(not(unix))]
fn set_mode(_dest: &Path, _manifest: &BundleManifest, _rel: &str) -> std::io::Result<()> {
    Ok(())
}

/// Reject a relative path that is empty, absolute, or contains a `..` / NUL
/// component (traversal hardening before it touches the filesystem).
fn reject_unsafe_rel(rel: &str) -> Result<(), BundleError> {
    if rel.is_empty()
        || rel.contains('\0')
        || rel.starts_with('/')
        || Path::new(rel).components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        })
    {
        return Err(BundleError::UnsafePath {
            rel: rel.to_string(),
        });
    }
    Ok(())
}

/// Reject layouts that cannot round-trip from a case-sensitive producer to a
/// case-insensitive consumer without one file overwriting another.
fn reject_case_collisions<'a>(rels: impl IntoIterator<Item = &'a str>) -> Result<(), BundleError> {
    let mut folded = std::collections::HashSet::new();
    for rel in rels {
        if !folded.insert(rel.to_lowercase()) {
            return Err(BundleError::UnsafePath {
                rel: format!("case-colliding path: {rel}"),
            });
        }
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
        return Err(BundleError::UnsafePath {
            rel: rel.to_string(),
        });
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

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct StatPairArt {
        first: PathBuf,
        second: PathBuf,
        semantic_tag: String,
        content_hash: ContentHash,
    }
    impl Artifact for StatPairArt {
        const KIND: &'static str = "test.stat-pair";
        const SCHEMA: u32 = 1;
        const HASH_CONTENTS: bool = false;
        fn content_hash(&self) -> ContentHash {
            self.content_hash
        }
        fn primary_path(&self) -> &Path {
            &self.first
        }
    }

    struct StatPairStage;
    #[async_trait]
    impl Stage for StatPairStage {
        const NAME: &'static str = "stat_pair_stage";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = StatPairArt;
        type Output = StatPairArt;
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

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct DeclaredMixedArt(StatPairArt);
    impl Artifact for DeclaredMixedArt {
        const KIND: &'static str = "test.declared-mixed";
        const SCHEMA: u32 = 1;
        const ALLOW_EXTERNAL_PATHS: bool = true;
        fn content_hash(&self) -> ContentHash {
            self.0.content_hash
        }
        fn primary_path(&self) -> &Path {
            &self.0.first
        }
    }

    struct DeclaredMixedStage;
    #[async_trait]
    impl Stage for DeclaredMixedStage {
        const NAME: &'static str = "declared_mixed_stage";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = DeclaredMixedArt;
        type Output = DeclaredMixedArt;
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

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct InlineArt {
        value: String,
        content_hash: ContentHash,
        sentinel: PathBuf,
    }
    impl Artifact for InlineArt {
        const KIND: &'static str = "test.inline";
        const SCHEMA: u32 = 1;
        const INLINE: bool = true;
        fn content_hash(&self) -> ContentHash {
            self.content_hash
        }
        fn primary_path(&self) -> &Path {
            &self.sentinel
        }
    }

    struct InlineStage;
    #[async_trait]
    impl Stage for InlineStage {
        const NAME: &'static str = "inline_stage";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = InlineArt;
        type Output = InlineArt;
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
    fn v3_writes_abir_identity_and_v2_remains_readable() {
        let stage = DirStage;
        let (src, erased, _) = make_dir_artifact();
        let (mut manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();

        assert_eq!(manifest.format_version, ARTIFACT_FORMAT_VERSION);
        assert_eq!(manifest.semantic_content_id(), Some(manifest.content_id));
        assert_eq!(
            manifest.content_id.to_hex(),
            "ed3fbf83cf55e070e75f32c9a95d0e701fe70f028715c27f23f2f2fca1f69aab"
        );

        manifest.format_version = ARTIFACT_FORMAT_VERSION + 1;
        assert_eq!(manifest.semantic_content_id(), None);
        manifest.format_version = ARTIFACT_FORMAT_VERSION;

        let legacy = derive_legacy_content_id(
            &stage,
            BlobDir::Input,
            &manifest.kind,
            manifest.schema,
            &manifest.files,
            &pack,
            &manifest.erased,
        )
        .unwrap();
        assert_ne!(legacy, manifest.content_id);

        manifest.format_version = LEGACY_ARTIFACT_FORMAT_VERSION;
        manifest.content_id = legacy;
        assert_eq!(manifest.semantic_content_id(), None);
        let dest = tempfile::tempdir().unwrap();
        let restored = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(legacy),
            BlobDir::Input,
        )
        .unwrap()
        .into_typed::<DirArt>()
        .unwrap();
        assert!(restored.path.starts_with(dest.path()));
    }

    #[test]
    fn round_trip_directory_artifact() {
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (manifest, pack) = bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        assert_eq!(manifest.files.len(), 3, "all 3 files shipped");
        assert!(manifest.files.iter().all(|f| !f.rel.starts_with("ckpt/")));
        assert_ne!(manifest.content_id.digest(), hash);
        assert_eq!(manifest.logical_hash, hash);

        let dest = tempfile::tempdir().unwrap();
        let rebased = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        )
        .unwrap();
        let typed: DirArt = rebased.into_typed().unwrap();
        // Path re-rooted under the peer's import dir; files materialized; the
        // recomputed dir hash equals the producer's.
        assert!(typed.path.starts_with(dest.path()));
        assert!(typed.path.join("weights.bin").exists());
        assert_eq!(ContentHash::hash_dir(&typed.path).unwrap(), hash);
    }

    #[test]
    fn round_trip_output_uses_output_contract() {
        let stage = DirStage;
        let (src, erased, hash) = make_dir_artifact();
        let (manifest, pack) =
            bundle(&stage, erased, src.path(), ArtifactRole::Output, None).unwrap();
        assert_ne!(manifest.content_id, ArtifactContentId::from_digest(hash));
        assert_eq!(manifest.logical_hash, hash);
        let portable: DirArt = manifest.erased.clone().into_typed().unwrap();
        assert!(!portable.path.is_absolute());

        let dest = tempfile::tempdir().unwrap();
        let restored = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            ArtifactRole::Output,
        )
        .unwrap();
        let restored: DirArt = restored.into_typed().unwrap();
        assert!(restored.path.starts_with(dest.path()));
        assert_eq!(ContentHash::hash_dir(&restored.path).unwrap(), hash);
    }

    #[test]
    fn undeclared_external_locator_remains_a_hard_capture_error() {
        let producer = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        std::fs::write(external.path().join("payload.bin"), b"external").unwrap();
        let art = DirArt {
            path: external.path().to_path_buf(),
            content_hash: ContentHash::hash_dir(external.path()).unwrap(),
        };

        let error = bundle(
            &DirStage,
            ErasedArtifact::from_typed(&art).unwrap(),
            producer.path(),
            ArtifactRole::Output,
            None,
        )
        .unwrap_err();

        assert!(matches!(error, ArtifactStoreError::Empty(_)));
    }

    #[test]
    fn undeclared_mixed_locator_remains_a_hard_rebase_error() {
        let producer = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        let owned_path = producer.path().join("owned.bin");
        let external_path = external.path().join("external.bin");
        std::fs::write(&owned_path, b"owned").unwrap();
        std::fs::write(&external_path, b"external").unwrap();
        let art = StatPairArt {
            first: owned_path,
            second: external_path,
            semantic_tag: "mixed".into(),
            content_hash: ContentHash::of_bytes(b"mixed"),
        };

        let error = bundle(
            &StatPairStage,
            ErasedArtifact::from_typed(&art).unwrap(),
            producer.path(),
            ArtifactRole::Output,
            None,
        )
        .unwrap_err();

        assert!(matches!(error, ArtifactStoreError::Rebase));
    }

    #[test]
    fn declared_mixed_locator_is_explicitly_nonportable() {
        let producer = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        let owned_path = producer.path().join("owned.bin");
        let external_path = external.path().join("external.bin");
        std::fs::write(&owned_path, b"owned").unwrap();
        std::fs::write(&external_path, b"external").unwrap();
        let art = DeclaredMixedArt(StatPairArt {
            first: owned_path,
            second: external_path,
            semantic_tag: "mixed".into(),
            content_hash: ContentHash::of_bytes(b"mixed"),
        });

        let error = bundle(
            &DeclaredMixedStage,
            ErasedArtifact::from_typed(&art).unwrap(),
            producer.path(),
            ArtifactRole::Output,
            None,
        )
        .unwrap_err();

        assert!(matches!(error, ArtifactStoreError::NonPortable(_)));
    }

    #[test]
    fn stat_fingerprint_is_logical_only_and_content_id_is_portable() {
        fn make(root: &Path, logical: &[u8], semantic_tag: &str) -> ErasedArtifact {
            std::fs::write(root.join("first.bin"), b"first").unwrap();
            std::fs::write(root.join("second.bin"), b"second").unwrap();
            ErasedArtifact::from_typed(&StatPairArt {
                first: root.join("first.bin"),
                second: root.join("second.bin"),
                semantic_tag: semantic_tag.to_string(),
                content_hash: ContentHash::of_bytes(logical),
            })
            .unwrap()
        }

        let stage = StatPairStage;
        let host_a = tempfile::tempdir().unwrap();
        let host_b = tempfile::tempdir().unwrap();
        let a = make(
            host_a.path(),
            host_a.path().as_os_str().as_encoded_bytes(),
            "same",
        );
        let b = make(
            host_b.path(),
            host_b.path().as_os_str().as_encoded_bytes(),
            "same",
        );
        let (a_manifest, a_pack) =
            bundle(&stage, a, host_a.path(), ArtifactRole::Output, None).unwrap();
        let (b_manifest, _) = bundle(&stage, b, host_b.path(), ArtifactRole::Output, None).unwrap();

        assert_ne!(a_manifest.logical_hash, b_manifest.logical_hash);
        assert_eq!(
            a_manifest.content_id, b_manifest.content_id,
            "producer path/stat identity must not enter ArtifactContentId"
        );
        assert_eq!(a_manifest.files.len(), 2, "both path fields are persisted");

        let host_c = tempfile::tempdir().unwrap();
        let c = make(host_c.path(), b"another local fingerprint", "different");
        let (c_manifest, _) = bundle(&stage, c, host_c.path(), ArtifactRole::Output, None).unwrap();
        assert_ne!(
            a_manifest.content_id, c_manifest.content_id,
            "semantic handle metadata participates in ArtifactContentId"
        );

        let consumer = tempfile::tempdir().unwrap();
        let restored = unbundle(
            &stage,
            &a_manifest,
            &a_pack,
            consumer.path(),
            Some(a_manifest.content_id),
            ArtifactRole::Output,
        )
        .unwrap()
        .into_typed::<StatPairArt>()
        .unwrap();
        assert!(restored.first.starts_with(consumer.path()));
        assert!(restored.second.starts_with(consumer.path()));
        assert_eq!(restored.content_hash, a_manifest.logical_hash);
    }

    #[test]
    fn inline_artifact_hashes_erased_payload_without_shipping_sentinel() {
        let stage = InlineStage;
        let src = tempfile::tempdir().unwrap();
        let logical_hash = ContentHash::of_bytes(b"inline value");
        let erased = ErasedArtifact::from_typed(&InlineArt {
            value: "inline value".into(),
            content_hash: logical_hash,
            sentinel: PathBuf::from("/dev/null"),
        })
        .unwrap();
        let (manifest, pack) =
            bundle(&stage, erased, src.path(), ArtifactRole::Output, None).unwrap();
        assert!(manifest.files.is_empty());
        assert!(pack.is_empty());
        assert_ne!(manifest.content_id.digest(), logical_hash);

        let consumer = tempfile::tempdir().unwrap();
        let restored = unbundle(
            &stage,
            &manifest,
            &pack,
            consumer.path(),
            Some(manifest.content_id),
            ArtifactRole::Output,
        )
        .unwrap()
        .into_typed::<InlineArt>()
        .unwrap();
        assert_eq!(restored.value, "inline value");
        assert_eq!(restored.sentinel, PathBuf::from("/dev/null"));
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
        let art = DirArt {
            path: art_dir,
            content_hash: clean_hash,
        };
        let erased = ErasedArtifact::from_typed(&art).unwrap();
        let (manifest, _pack) = bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        assert!(
            !manifest
                .files
                .iter()
                .any(|f| f.rel.ends_with(".lamu-meta.json")),
            "sidecar must be excluded"
        );
        assert_ne!(
            hash, clean_hash,
            "sidecar would change hash_dir if included"
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_oversized_sparse_backing_before_reading_it() {
        let src = tempfile::tempdir().unwrap();
        let art_dir = src.path().join("oversized");
        std::fs::create_dir_all(&art_dir).unwrap();
        std::fs::File::create(art_dir.join("sparse.bin"))
            .unwrap()
            .set_len(MAX_OBJECT_SIZE + 1)
            .unwrap();
        let erased = ErasedArtifact::from_typed(&DirArt {
            path: art_dir,
            content_hash: ContentHash::of_bytes(b"oversized fixture"),
        })
        .unwrap();

        let error = bundle(&DirStage, erased, src.path(), ArtifactRole::Output, None).unwrap_err();
        assert!(matches!(
            error,
            ArtifactStoreError::TooLarge {
                max: MAX_OBJECT_SIZE,
                ..
            }
        ));
    }

    #[test]
    fn rejects_relative_non_inline_backing_that_resolves_to_no_owned_file() {
        let src = tempfile::tempdir().unwrap();
        let erased = ErasedArtifact::from_typed(&DirArt {
            path: PathBuf::from("relative-artifact"),
            content_hash: ContentHash::of_bytes(b"unresolved relative fixture"),
        })
        .unwrap();

        let error = bundle(&DirStage, erased, src.path(), ArtifactRole::Output, None).unwrap_err();

        assert!(matches!(error, ArtifactStoreError::Empty(_)));
    }

    #[test]
    fn rejects_tampered_blob() {
        let stage = DirStage;
        let (src, erased, _hash) = make_dir_artifact();
        let (manifest, mut pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        let mid = pack.len() / 2;
        pack[mid] ^= 0xFF; // flip a byte
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        );
        assert!(matches!(r, Err(BundleError::BlobHash { .. })));
        // Import dir cleaned up on failure.
        assert!(
            !dest
                .path()
                .join(".artifact-import")
                .join(manifest.content_id.to_hex())
                .exists()
        );
    }

    #[test]
    fn rejects_malicious_table_with_tampered_bytes() {
        // The adversary tampers a file's BYTES but "fixes" its per-file hash in
        // the table. Per-file gate passes; the whole-artifact recompute (gate 3)
        // re-walks disk the artifact's OWN way and catches it.
        let stage = DirStage;
        let (src, erased, _hash) = make_dir_artifact();
        let (mut manifest, _pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
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
                std::fs::read(src.path().join("ckpt").join(&f.rel)).unwrap()
            };
            new_pack.extend_from_slice(&(f.rel.len() as u32).to_le_bytes());
            new_pack.extend_from_slice(f.rel.as_bytes());
            new_pack.extend_from_slice(&(body.len() as u64).to_le_bytes());
            new_pack.extend_from_slice(&body);
        }
        manifest.blob_sha256 = ContentHash::of_bytes(&new_pack); // fix gate 1 too
        manifest.blob_len = new_pack.len() as u64;
        manifest.content_id = derive_content_id(
            &stage,
            BlobDir::Input,
            &manifest.kind,
            manifest.schema,
            &manifest.files,
            &new_pack,
            &manifest.erased,
        )
        .unwrap();
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(
            &stage,
            &manifest,
            &new_pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        );
        assert!(
            matches!(r, Err(BundleError::ContentMismatch { .. })),
            "typed logical-hash recompute must catch structural tamper, got {r:?}"
        );
    }

    #[test]
    fn rejects_hash_binding_mismatch() {
        let stage = DirStage;
        let (src, erased, _hash) = make_dir_artifact();
        let (manifest, pack) = bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        let dest = tempfile::tempdir().unwrap();
        let wrong = ContentHash([9u8; 32]);
        let r = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(ArtifactContentId::from_digest(wrong)),
            BlobDir::Input,
        );
        assert!(matches!(r, Err(BundleError::IdentityBinding { .. })));
    }

    #[test]
    fn rejects_version_skew() {
        let stage = DirStage;
        let (src, erased, _hash) = make_dir_artifact();
        let (mut manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        manifest.format_version = 999;
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        );
        assert!(matches!(r, Err(BundleError::Version { .. })));
    }

    #[test]
    fn rejects_unsafe_rel_paths() {
        assert!(reject_unsafe_rel("../etc/passwd").is_err());
        assert!(reject_unsafe_rel("/abs").is_err());
        assert!(reject_unsafe_rel("").is_err());
        assert!(reject_unsafe_rel("a\0b").is_err());
        assert!(reject_unsafe_rel("ok/nested.json").is_ok());
        assert!(reject_case_collisions(["Foo.bin", "foo.bin"]).is_err());
        assert!(reject_case_collisions(["a/Foo.bin", "b/foo.bin"]).is_ok());
    }

    #[test]
    fn rejects_extra_unlisted_file_in_pack() {
        // A malicious peer appends a file NOT in manifest.files. Gate 2's count
        // check rejects it before the stage runs.
        let stage = DirStage;
        let (src, erased, _hash) = make_dir_artifact();
        let (mut manifest, mut pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        // Append an extra "ckpt/evil.sh" frame the manifest doesn't list.
        let rel = "ckpt/evil.sh";
        let body = b"#!/bin/sh\nrm -rf /\n";
        pack.extend_from_slice(&(rel.len() as u32).to_le_bytes());
        pack.extend_from_slice(rel.as_bytes());
        pack.extend_from_slice(&(body.len() as u64).to_le_bytes());
        pack.extend_from_slice(body);
        manifest.blob_sha256 = ContentHash::of_bytes(&pack); // re-fix gate 1
        manifest.blob_len = pack.len() as u64;
        manifest.content_id = derive_content_id(
            &stage,
            BlobDir::Input,
            &manifest.kind,
            manifest.schema,
            &manifest.files,
            &pack,
            &manifest.erased,
        )
        .unwrap();
        let dest = tempfile::tempdir().unwrap();
        let r = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        );
        assert!(
            matches!(r, Err(BundleError::FileHash { .. })),
            "extra unlisted file must be rejected, got {r:?}"
        );
        assert!(
            !dest
                .path()
                .join(".artifact-import")
                .join(manifest.content_id.to_hex())
                .exists()
        );
    }

    #[cfg(unix)]
    #[test]
    fn setuid_bits_are_stripped() {
        use std::os::unix::fs::PermissionsExt;
        let stage = DirStage;
        let (src, erased, _hash) = make_dir_artifact();
        let (mut manifest, pack) =
            bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        // Force a setuid+exec mode on one file.
        for f in &mut manifest.files {
            f.mode = 0o4755; // setuid rwxr-xr-x
        }
        manifest.content_id = derive_content_id(
            &stage,
            BlobDir::Input,
            &manifest.kind,
            manifest.schema,
            &manifest.files,
            &pack,
            &manifest.erased,
        )
        .unwrap();
        let dest = tempfile::tempdir().unwrap();
        let rebased = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        )
        .unwrap();
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
        let (manifest, pack) = bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        let dest = tempfile::tempdir().unwrap();
        let a = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        )
        .unwrap();
        let b = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        )
        .unwrap();
        let pa: DirArt = a.into_typed().unwrap();
        let pb: DirArt = b.into_typed().unwrap();
        assert_eq!(
            pa.path, pb.path,
            "same ArtifactContentId -> same import dir"
        );
        assert_eq!(ContentHash::hash_dir(&pb.path).unwrap(), hash);
    }

    #[cfg(unix)]
    #[test]
    fn import_cleanup_failure_aborts_restore() {
        use std::os::unix::fs::PermissionsExt;

        let stage = DirStage;
        let (src, erased, _) = make_dir_artifact();
        let (manifest, pack) = bundle(&stage, erased, src.path(), BlobDir::Input, None).unwrap();
        let dest = tempfile::tempdir().unwrap();
        unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        )
        .unwrap();
        let import_parent = dest.path().join(".artifact-import");
        std::fs::set_permissions(&import_parent, std::fs::Permissions::from_mode(0o555)).unwrap();

        let result = unbundle(
            &stage,
            &manifest,
            &pack,
            dest.path(),
            Some(manifest.content_id),
            BlobDir::Input,
        );
        std::fs::set_permissions(&import_parent, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(matches!(result, Err(ArtifactStoreError::Cleanup { .. })));
    }
}
