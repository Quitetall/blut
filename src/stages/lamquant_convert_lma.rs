//! Stage — `lamquant_convert_lma`. () → LmaCorpus.
//!
//! Wraps `scripts/bulk_lml_to_lma.py` to pack per-stem `.lml` +
//! annotation sidecars + label NPZ into per-recording `.lma`
//! archives under `output_dir`. Idempotent: re-runs skip stems
//! whose `<stem>.lma` already exists. Produces the canonical
//! LMA corpus consumed by every train_*.py kernel post-LMA pivot
//! (ADR 0017).
//!
//! Deterministic per-stem (same source bytes + same corpus
//! precedence → same LMA bytes), so `DETERMINISTIC = true`.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::LmaCorpus;
use crate::artifacts::lamquant::stat_fingerprint;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{
    blut_env, progress_forwarder, push_opt_u32, python_for, resolve_home, resolve_roots, safe_join,
    scripts_script,
};

pub struct LamquantConvertLma;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty → `$LAMQUANT_HOME` or default.
    #[serde(default)]
    pub lamquant_home: String,
    /// LML source root (per-corpus tree). Empty = builder default.
    #[serde(default)]
    pub lml_root: String,
    /// Labels NPZ directory (relative to lamquant_home). Empty =
    /// the unified canonical labels root
    /// (`crate::paths::DEFAULT_LABELS_DIR`, RCP-6).
    #[serde(default)]
    pub labels_dir_rel: String,
    /// Output LMA corpus directory. Required.
    pub output_dir: PathBuf,
    /// Worker process count. None = builder default (cpu_count / 3).
    #[serde(default)]
    pub workers: Option<u32>,
    /// Cap iteration to first N stems (for smoke runs). None = full.
    #[serde(default)]
    pub limit: Option<u32>,
    /// Pass `--keep-sources` so loose source files survive the pack.
    #[serde(default)]
    pub keep_sources: bool,
    /// Pass `--dry-run`: plan + report only, no LMA writes, no deletes.
    #[serde(default)]
    pub dry_run: bool,
}

#[async_trait]
impl Stage for LamquantConvertLma {
    const NAME: &'static str = "lamquant_convert_lma";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk, Resource::Cpu];
    // The packer is content-deterministic per-stem (LMA bytes are a
    // function of source LML + sidecars + meta JSON + precedence rule).
    // Multi-worker scheduling is the only nondeterminism source and it
    // doesn't affect the output bytes.
    const DETERMINISTIC: bool = true;
    type Input = ();
    type Output = LmaCorpus;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &Args,
    ) -> Result<LmaCorpus, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let python = python_for(&home);
        // RCP-1/RCP-7: `scripts/` lives at the meta-repo root, not
        // under `ai_models_root`. When the caller pins an explicit
        // `lamquant_home`, honor it as the scripts root (hermetic
        // tests lay `scripts/bulk_lml_to_lma.py` under it); otherwise
        // resolve via the detected multi-root layout.
        let script = if args.lamquant_home.is_empty() {
            scripts_script(&resolve_roots()?, &["scripts", "bulk_lml_to_lma.py"])?
        } else {
            let s = home.join("scripts").join("bulk_lml_to_lma.py");
            if !s.exists() {
                return Err(StageError::BadInput(format!(
                    "bulk_lml_to_lma.py not found: {}",
                    s.display()
                )));
            }
            s
        };
        if args.output_dir.as_os_str().is_empty() {
            return Err(StageError::BadInput(
                "output_dir is required for lamquant_convert_lma".into(),
            ));
        }
        std::fs::create_dir_all(&args.output_dir).map_err(|source| StageError::Io {
            path: args.output_dir.clone(),
            source,
        })?;

        // Idempotent skip: when output_dir already holds at least one
        // .lma archive (flat or one level deep under <source>/), treat
        // it as already-converted and skip the EDF→LML→LMA subprocess.
        // This matches the post-Phase-M corpus shape: per-dataset LMAs
        // live at Archive/lma/<source>/<corpus>.lma; the LML mirror tree
        // is no longer kept on disk (regen via Archive/edf/<source>/install.sh).
        let mut found_lma = false;
        if let Ok(rd) = std::fs::read_dir(&args.output_dir) {
            for e in rd.flatten() {
                let path = e.path();
                if path.extension().and_then(|s| s.to_str()) == Some("lma") {
                    found_lma = true;
                    break;
                }
                if path.is_dir() {
                    if let Ok(rd2) = std::fs::read_dir(&path) {
                        for e2 in rd2.flatten() {
                            if e2.path().extension().and_then(|s| s.to_str()) == Some("lma") {
                                found_lma = true;
                                break;
                            }
                        }
                    }
                }
                if found_lma {
                    break;
                }
            }
        }
        if found_lma {
            let n_archives = count_lma_archives(&args.output_dir);
            let content_hash =
                stat_fingerprint(b"lamquant.lma_corpus", &args.output_dir).map_err(|source| {
                    StageError::Io {
                        path: args.output_dir.clone(),
                        source,
                    }
                })?;
            return Ok(LmaCorpus {
                root: args.output_dir.clone(),
                n_archives,
                content_hash,
            });
        }

        // RCP-6: unify the labels-dir default to the canonical
        // `/mnt/4tb/data/Training/labels` root (was the dead
        // monorepo path `<home>/ai_models/snn/labels`). An explicit
        // relative override still resolves under `home`.
        let labels_dir = if args.labels_dir_rel.is_empty() {
            PathBuf::from(crate::paths::DEFAULT_LABELS_DIR)
        } else {
            safe_join(&home, &args.labels_dir_rel)?
        };

        let mut cmd_args: Vec<String> = vec![
            "--output-dir".into(),
            args.output_dir.display().to_string(),
            "--labels-dir".into(),
            labels_dir.display().to_string(),
        ];
        if !args.lml_root.is_empty() {
            cmd_args.push("--lml-root".into());
            cmd_args.push(args.lml_root.clone());
        }
        push_opt_u32(&mut cmd_args, "--workers", args.workers);
        push_opt_u32(&mut cmd_args, "--limit", args.limit);
        if args.keep_sources {
            cmd_args.push("--keep-sources".into());
        }
        if args.dry_run {
            cmd_args.push("--dry-run".into());
        }

        let inv = LamquantInvocation {
            python,
            script,
            cwd: home,
            args: cmd_args,
            env: blut_env(&ctx.job_dir, Self::NAME),
            expected_outputs: vec![args.output_dir.clone()],
            run_manifest_path: None,
        };
        let mut backend = LamquantBackend::new();
        backend
            .run(
                inv,
                Some(progress_forwarder(Self::NAME, ctx.status_tx.clone())),
            )
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        // Count .lma archives under output_dir; build provenance hash.
        let n_archives = count_lma_archives(&args.output_dir);
        let content_hash =
            stat_fingerprint(b"lamquant.lma_corpus", &args.output_dir).map_err(|source| {
                StageError::Io {
                    path: args.output_dir.clone(),
                    source,
                }
            })?;
        Ok(LmaCorpus {
            root: args.output_dir.clone(),
            n_archives,
            content_hash,
        })
    }
}

/// Count `.lma` files under `root`, scanning both root itself and one
/// level deep (matches Archive/lma/<source>/<corpus>.lma).
fn count_lma_archives(root: &std::path::Path) -> i64 {
    let mut n: i64 = 0;
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.flatten() {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) == Some("lma") {
                n += 1;
                continue;
            }
            if path.is_dir() {
                if let Ok(rd2) = std::fs::read_dir(&path) {
                    for e2 in rd2.flatten() {
                        if e2.path().extension().and_then(|s| s.to_str()) == Some("lma") {
                            n += 1;
                        }
                    }
                }
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn rejects_missing_script() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantConvertLma
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    lml_root: String::new(),
                    labels_dir_rel: String::new(),
                    output_dir: td.path().join("out"),
                    workers: None,
                    limit: None,
                    keep_sources: false,
                    dry_run: true,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_empty_output_dir() {
        let td = tempfile::tempdir().unwrap();
        // Lay down the script so we get past that gate.
        let scripts = td.path().join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        std::fs::write(scripts.join("bulk_lml_to_lma.py"), "# stub\n").unwrap();
        let r = LamquantConvertLma
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    lamquant_home: td.path().display().to_string(),
                    lml_root: String::new(),
                    labels_dir_rel: String::new(),
                    output_dir: PathBuf::new(),
                    workers: None,
                    limit: None,
                    keep_sources: false,
                    dry_run: true,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn content_hash_kind_consistent() {
        let lc = LmaCorpus {
            root: PathBuf::from("/tmp/lma"),
            n_archives: 0,
            content_hash: ContentHash::of_bytes(b""),
        };
        use crate::framework::artifact::Artifact;
        assert_eq!(LmaCorpus::KIND, "lamquant.lma_corpus");
        assert_eq!(lc.root.display().to_string(), "/tmp/lma");
    }
}
