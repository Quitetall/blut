//! LamQuant artifact types.
//!
//! Concrete artifacts produced + consumed by the LamQuant stage
//! catalog (`build_manifest`, `precompute_*`, `train_*`,
//! `harden_artifacts`, `pccp_gate`). Each maps to one on-disk
//! file or directory in `$LAMQUANT_HOME` or `/mnt/4tb/LamQuant`.
//!
//! Hashing strategy:
//!
//!   - Small JSON / text artifacts (Manifest) hash bytes — equality
//!     of fields matters and the file is tiny.
//!   - Large numpy memmaps + checkpoint dirs (FullbandMemmap,
//!     JointCkpt, ...) set `HASH_CONTENTS = false` and implement
//!     `content_hash()` via a stat-based fingerprint (path + size
//!     + mtime). Walking ~10 GB of bytes per cache lookup is not
//!       defensible; the producing stage's identity + the cache key
//!       already guarantee what we need.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::framework::artifact::{Artifact, ContentHash};

// ── Manifest ────────────────────────────────────────────────────

/// `manifest_v3.json` — patient holdout assignments + per-window
/// metadata produced by `ai_models/dataset_sim/build_manifest.py`.
/// Small file (<10 MB), so we hash bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub n_windows: i64,
    pub val_fraction: f32,
    pub seed: u64,
}

impl Artifact for Manifest {
    const KIND: &'static str = "lamquant.manifest";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

// ── SplitManifest ───────────────────────────────────────────────

/// `split_manifest.json` — patient-level, seizure-stratified
/// train/val split produced by
/// `ai_models/dataset_sim/build_seizure_split_manifest.py` (RCP-3).
/// Schema: `{ "subjects": {sid: "train"|"val"}, "stems_by_subject":
/// {...}, "meta": {...} }`, consumed by `train_mamba_snn.py
/// --split-manifest`. Small JSON (<a few MB), so we hash bytes —
/// the split assignment is the cache-relevant content. Distinct
/// from `Manifest` (the legacy `manifest_v3.json` window manifest):
/// this one is the subject-grouped split LMA-direct training
/// requires.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitManifest {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    /// Number of subjects assigned to the train split.
    pub n_train_subjects: i64,
    /// Number of subjects assigned to the val split.
    pub n_val_subjects: i64,
}

impl Artifact for SplitManifest {
    const KIND: &'static str = "lamquant.split_manifest";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

// ── FullbandMemmap ──────────────────────────────────────────────

/// `fullband_train.npy` + `fullband_val.npy` pair — bulky memmaps
/// (~10-50 GB total). `HASH_CONTENTS = false`; the fingerprint is
/// (train_path, train_size, train_mtime) + (val_path, val_size,
/// val_mtime). Cache cares about provenance, not byte equality.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullbandMemmap {
    pub train_path: PathBuf,
    pub val_path: PathBuf,
    pub n_windows: i64,
    pub content_hash: ContentHash,
}

impl Artifact for FullbandMemmap {
    const KIND: &'static str = "lamquant.fullband_memmap";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.train_path
    }
}

// ── L3Cache ─────────────────────────────────────────────────────

/// L3 approximation cache directory (~17 GB of precomputed
/// subband coefficients). `HASH_CONTENTS = false`; stat-based
/// fingerprint over the directory's top-level files only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct L3Cache {
    pub dir: PathBuf,
    pub n_windows: i64,
    pub content_hash: ContentHash,
}

impl Artifact for L3Cache {
    const KIND: &'static str = "lamquant.l3_cache";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.dir
    }
}

// ── Checkpoint family ───────────────────────────────────────────

/// Self-supervised MAE-pretrained encoder. Seeds `train_joint`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaeCkpt {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub base_arch: String,
    pub final_loss: f32,
}

impl Artifact for MaeCkpt {
    const KIND: &'static str = "lamquant.ckpt.mae";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

/// Teacher checkpoint (Gen 6 / L3 variant).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherCkpt {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub gen_tag: String,
    pub final_loss: f32,
}

impl Artifact for TeacherCkpt {
    const KIND: &'static str = "lamquant.ckpt.teacher";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

/// Encoder + decoder pair produced by `train_joint`. Split so the
/// firmware export target (encoder) and base-station target
/// (decoder) carry independent identities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointCkpt {
    pub encoder_path: PathBuf,
    pub decoder_path: PathBuf,
    pub content_hash: ContentHash,
    pub final_loss: f32,
    pub tier: u32,
    pub preset: String,
}

impl Artifact for JointCkpt {
    const KIND: &'static str = "lamquant.ckpt.joint";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.encoder_path
    }
}

/// Post-`harden_artifacts` output: encoder ckpt with latents
/// realigned against a strided teacher for Route B deployment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardenedCkpt {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub route_b: bool,
}

impl Artifact for HardenedCkpt {
    const KIND: &'static str = "lamquant.ckpt.hardened";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

// ── LMA corpus / labels / firmware bundle ───────────────────────

/// Per-recording LMA archive directory. Each `<stem>.lma` packs the
/// LML container + annotation sidecars + label NPZ + provenance
/// meta.json (subject_id, content_sha256_lml). The training data
/// path post-2026-05-16 LMA pivot — every train_*.py kernel reads
/// these via `lamquant_codec.training.LmaL3Dataset` /
/// `LmaSignalDataset`. See ADR 0017.
///
/// `HASH_CONTENTS = false`: archive bodies can run to hundreds of
/// GB. Content hash is a stat-fingerprint over the corpus root
/// (file count + total size + per-file mtime sample). Cache cares
/// about provenance, not byte equality.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LmaCorpus {
    pub root: PathBuf,
    pub n_archives: i64,
    pub content_hash: ContentHash,
}

impl Artifact for LmaCorpus {
    const KIND: &'static str = "lamquant.lma_corpus";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.root
    }
}

/// Per-stem activity-label NPZ directory consumed by
/// `lamquant_train_mamba_snn`. Produced by
/// `lamquant_generate_snn_labels` for each EDF dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnnLabels {
    pub dir: PathBuf,
    pub dataset_id: String,
    pub n_stems: i64,
    pub content_hash: ContentHash,
}

impl Artifact for SnnLabels {
    const KIND: &'static str = "lamquant.snn_labels";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.dir
    }
}

/// Firmware export output: C headers (encoder + decoder + snn) +
/// flash-ready `.bin`. Produced by `lamquant_export_firmware`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FirmwareBundle {
    pub bundle_dir: PathBuf,
    pub bin_path: PathBuf,
    pub target: String,
    pub content_hash: ContentHash,
}

impl Artifact for FirmwareBundle {
    const KIND: &'static str = "lamquant.firmware_bundle";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.bundle_dir
    }
}

/// Mamba SNN checkpoint for seizure / activity detection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnnCkpt {
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub head_size_kb: f32,
    pub final_loss: f32,
}

impl Artifact for SnnCkpt {
    const KIND: &'static str = "lamquant.ckpt.snn";
    const SCHEMA: u32 = 1;
    const HASH_CONTENTS: bool = false;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

// ── PCCP gate verdict ───────────────────────────────────────────

/// Output of `pccp_gate` stage. Always produced — never short-
/// circuits the plan. Recipe authors decide downstream behavior
/// based on `passed`. The record_id links back to the
/// `pccp/verification_records/<id>.gate.json` file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PccpVerdict {
    pub gate_json_path: PathBuf,
    pub record_id: String,
    pub passed: bool,
    pub candidate_path: PathBuf,
    pub model_name: String,
    pub content_hash: ContentHash,
}

impl Artifact for PccpVerdict {
    const KIND: &'static str = "lamquant.pccp_verdict";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.gate_json_path
    }
}

// ── Helpers shared across stages ────────────────────────────────

fn mtime_secs(meta: &std::fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Stat-based fingerprint over one path: hash(path_bytes ‖
/// size ‖ mtime_unix_secs). Caller supplies a domain tag so two
/// artifact types over the same file don't collide.
///
/// Convention: the producing STAGE calls this helper and stores
/// the result in its artifact's `content_hash` field. The
/// `Artifact::content_hash()` impl just returns that stored value
/// — it is NOT a fresh stat on every read (would defeat the
/// "skip walking bytes" point of `HASH_CONTENTS = false`).
pub fn stat_fingerprint(domain: &[u8], path: &Path) -> std::io::Result<ContentHash> {
    let meta = std::fs::metadata(path)?;
    let mut h = Sha256::new();
    h.update(domain);
    h.update([0u8]);
    h.update(path.as_os_str().to_string_lossy().as_bytes());
    h.update([0u8]);
    h.update(meta.len().to_le_bytes());
    h.update(mtime_secs(&meta).to_le_bytes());
    let arr: [u8; 32] = h.finalize().into();
    Ok(ContentHash(arr))
}

/// Stat-based fingerprint over a directory: top-level file
/// (name, size, mtime) tuples concatenated in sorted order, then
/// SHA-256. Caveats:
///
///   - Does NOT recurse. Subdirectories are SKIPPED (not even
///     counted). If a producing stage writes a subdir inside the
///     cache dir, the fingerprint won't catch its changes.
///   - Symlinks are skipped too (the `is_file` check follows
///     metadata semantics: a dangling symlink fails the check and
///     is silently dropped). Stages that emit symlinked
///     checkpoints should resolve them before fingerprinting.
///   - Same caveats hold even when nominally "all files at top
///     level" — be explicit when picking this for an artifact.
pub fn stat_fingerprint_dir(domain: &[u8], dir: &Path) -> std::io::Result<ContentHash> {
    let mut entries: Vec<(String, u64, u64)> = Vec::new();
    for ent in std::fs::read_dir(dir)? {
        let ent = ent?;
        let meta = ent.metadata()?;
        if !meta.is_file() {
            continue;
        }
        entries.push((
            ent.file_name().to_string_lossy().into_owned(),
            meta.len(),
            mtime_secs(&meta),
        ));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut h = Sha256::new();
    h.update(domain);
    h.update([0u8]);
    h.update(dir.as_os_str().to_string_lossy().as_bytes());
    h.update([0u8]);
    for (name, size, mtime) in &entries {
        h.update(name.as_bytes());
        h.update([0u8]);
        h.update(size.to_le_bytes());
        h.update(mtime.to_le_bytes());
    }
    let arr: [u8; 32] = h.finalize().into();
    Ok(ContentHash(arr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips() {
        let m = Manifest {
            path: PathBuf::from("/tmp/manifest.json"),
            content_hash: ContentHash::of_bytes(b"x"),
            n_windows: 1_234_567,
            val_fraction: 0.05,
            seed: 42,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.n_windows, 1_234_567);
        assert_eq!(Manifest::KIND, "lamquant.manifest");
    }

    // intentional: these assert the compile-time HASH_CONTENTS contract
    // (CONST-C3) — each `assert!` on a const associated value documents
    // and locks the per-artifact hashing policy. A const-block rewrite
    // would lose the failing-artifact name on a future flip.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn fullband_memmap_skips_content_hashing() {
        assert!(!FullbandMemmap::HASH_CONTENTS);
        assert!(!JointCkpt::HASH_CONTENTS);
        assert!(!MaeCkpt::HASH_CONTENTS);
        assert!(!SnnCkpt::HASH_CONTENTS);
        assert!(!TeacherCkpt::HASH_CONTENTS);
        assert!(!HardenedCkpt::HASH_CONTENTS);
        assert!(!L3Cache::HASH_CONTENTS);
        // Manifest + PccpVerdict still hash bytes — they're small.
        assert!(Manifest::HASH_CONTENTS);
        assert!(PccpVerdict::HASH_CONTENTS);
    }

    #[test]
    fn stat_fingerprint_stable_for_unchanged_file() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("f");
        std::fs::write(&p, b"hello").unwrap();
        let h1 = stat_fingerprint(b"test", &p).unwrap();
        let h2 = stat_fingerprint(b"test", &p).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn stat_fingerprint_changes_on_size_change() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("f");
        std::fs::write(&p, b"hello").unwrap();
        let h1 = stat_fingerprint(b"test", &p).unwrap();
        std::fs::write(&p, b"hello-bigger").unwrap();
        let h2 = stat_fingerprint(b"test", &p).unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn stat_fingerprint_dir_sorted_and_stable() {
        let td1 = tempfile::tempdir().unwrap();
        let td2 = tempfile::tempdir().unwrap();
        // Create the same files in different orders; fingerprint
        // must match if dir layouts + sizes + mtimes do.
        for name in ["a", "b", "c"] {
            std::fs::write(td1.path().join(name), name.as_bytes()).unwrap();
        }
        for name in ["c", "a", "b"] {
            std::fs::write(td2.path().join(name), name.as_bytes()).unwrap();
        }
        // Different tempdir paths + mtimes => cross-dir hashes
        // differ; we only assert that the SAME dir produces a
        // stable hash. (Sort-stability is implicit: any
        // re-iteration of the same dir hits the same sort order.)
        let h1 = stat_fingerprint_dir(b"test", td1.path()).unwrap();
        let h2 = stat_fingerprint_dir(b"test", td2.path()).unwrap();
        let h1b = stat_fingerprint_dir(b"test", td1.path()).unwrap();
        assert_eq!(h1, h1b, "stable on same dir");
        assert_ne!(h1, h2, "different dir paths produce different hashes");
    }
}
