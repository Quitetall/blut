//! Stage — `lamquant_encode_lma` (RCP-2). () → LmaCorpus.
//!
//! Wraps the `lml encode <edf_dir> -o <out_dir>` Rust binary
//! (EDF/BDF/BrainVision/… dir → per-recording `.lma`). This is the
//! REAL EDF→.lma encode path the operator runs by hand via
//! `/mnt/4tb/data/Training/encode_corpora.sh`; BLUT previously only
//! knew the dead `bulk_lml_to_lma.py` LML→LMA path
//! (`lamquant_convert_lma`). Each `.lma` packs the compressed `.lml`
//! signal + the original source bytes + every sibling annotation file
//! (TUH `.tse` / `.csv_bi` / `.lbl_bi` / `_summary.txt`) — no byte is
//! dropped.
//!
//! The `lml` binary is resolved via `LamquantRoots::lml_binary()`
//! (`$BLUT_LML` override, else `<meta>/LamQuant-Lossless/target/
//! release/lml`). The path is computed even when the release binary
//! isn't built yet; existence is asserted at run-time preflight so a
//! missing binary surfaces as a clear `BadInput` ("cargo build
//! --release in the Lossless submodule") rather than a spawn error.
//!
//! `DETERMINISTIC = false`: encode is byte-deterministic per
//! recording, but the output is a directory the framework can't
//! byte-equal-hash cheaply (`HASH_CONTENTS = false` on `LmaCorpus`),
//! and multi-input scheduling order does not affect bytes — so this
//! mirrors `lamquant_convert_lma`'s artifact contract: stat-fingerprint
//! the corpus root for provenance, not byte equality.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::LmaCorpus;
use crate::artifacts::lamquant::stat_fingerprint;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{blut_env, progress_forwarder, resolve_roots};

pub struct LamquantEncodeLma;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Input EDF/BDF corpus directory (raw recordings). Required.
    /// Passed verbatim to `lml encode <edf_dir>` — may be a
    /// symlink-farm (`_farm/<corpus>`) or a direct source dir, exactly
    /// as `encode_corpora.sh` does.
    pub edf_dir: PathBuf,
    /// Output LMA corpus directory. Required. `lml encode -o <out_dir>`
    /// writes one `<stem>.lma` per recording here.
    pub out_dir: PathBuf,
    /// Optional human label for the corpus (e.g. "tusz_v2.0.6").
    /// Recorded in status events; does not affect the command line.
    #[serde(default)]
    pub corpus: String,
    /// Pass `-q` (quiet) to suppress `lml`'s log output. Default
    /// true to match `encode_corpora.sh`; status/progress still flow
    /// via the tqdm parser on the remaining stderr lines.
    #[serde(default = "default_quiet")]
    pub quiet: bool,
    /// Pass `--verify`: decode each `.lma` back and assert a clean
    /// roundtrip before declaring success. Off by default (doubles
    /// the work); recipes that want belt-and-suspenders set it true.
    #[serde(default)]
    pub verify: bool,
}

fn default_quiet() -> bool {
    true
}

#[async_trait]
impl Stage for LamquantEncodeLma {
    const NAME: &'static str = "lamquant_encode_lma";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu, Resource::Disk];
    // See module docs: byte-deterministic per recording, but the
    // artifact is a stat-fingerprinted dir; mirror convert_lma.
    const DETERMINISTIC: bool = false;
    type Input = ();
    type Output = LmaCorpus;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &Args,
    ) -> Result<LmaCorpus, StageError> {
        if args.edf_dir.as_os_str().is_empty() {
            return Err(StageError::BadInput(
                "edf_dir is required for lamquant_encode_lma".into(),
            ));
        }
        if args.out_dir.as_os_str().is_empty() {
            return Err(StageError::BadInput(
                "out_dir is required for lamquant_encode_lma".into(),
            ));
        }
        if !args.edf_dir.is_dir() {
            return Err(StageError::BadInput(format!(
                "edf_dir not found or not a directory: {}",
                args.edf_dir.display()
            )));
        }

        // RCP-2: resolve the `lml` binary via the multi-root resolver.
        // `lml_binary()` does not existence-check (the release binary
        // is the operator's `cargo build --release` step); we assert it
        // here so a missing binary is a clear preflight error.
        let roots = resolve_roots()?;
        let lml = roots.lml_binary();
        if !lml.exists() {
            return Err(StageError::BadInput(format!(
                "lml binary not found: {}. Build it with `cargo build --release` \
                 in the Lossless submodule, or set $BLUT_LML to the binary path.",
                lml.display()
            )));
        }

        std::fs::create_dir_all(&args.out_dir).map_err(|source| StageError::Io {
            path: args.out_dir.clone(),
            source,
        })?;

        // Build: `lml encode <edf_dir> -o <out_dir> [-q] [--verify]`.
        // The backend runs `Command::new(python).arg(script).args(..)`;
        // here `python` = the lml binary and `script` = the `encode`
        // subcommand, yielding the exact `encode_corpora.sh` command.
        let mut cmd_args: Vec<String> = vec![
            args.edf_dir.display().to_string(),
            "-o".into(),
            args.out_dir.display().to_string(),
        ];
        if args.quiet {
            cmd_args.push("-q".into());
        }
        if args.verify {
            cmd_args.push("--verify".into());
        }

        let inv = LamquantInvocation {
            python: lml,
            script: PathBuf::from("encode"),
            cwd: args.out_dir.clone(),
            args: cmd_args,
            env: blut_env(&ctx.job_dir, Self::NAME),
            expected_outputs: vec![args.out_dir.clone()],
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

        let n_archives = count_lma_archives(&args.out_dir);
        let content_hash =
            stat_fingerprint(b"lamquant.lma_corpus", &args.out_dir).map_err(|source| {
                StageError::Io {
                    path: args.out_dir.clone(),
                    source,
                }
            })?;
        Ok(LmaCorpus {
            root: args.out_dir.clone(),
            n_archives,
            content_hash,
        })
    }
}

/// Count `.lma` files under `root`, scanning both `root` itself and
/// one level deep (matches `lml encode`'s per-corpus subdir layout:
/// `<out_dir>/<corpus>/<stem>.lma`).
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
    use crate::framework::artifact::Artifact;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn rejects_empty_edf_dir() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantEncodeLma
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    edf_dir: PathBuf::new(),
                    out_dir: td.path().join("out"),
                    corpus: String::new(),
                    quiet: true,
                    verify: false,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_empty_out_dir() {
        let td = tempfile::tempdir().unwrap();
        let edf = td.path().join("edf");
        std::fs::create_dir_all(&edf).unwrap();
        let r = LamquantEncodeLma
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    edf_dir: edf,
                    out_dir: PathBuf::new(),
                    corpus: String::new(),
                    quiet: true,
                    verify: false,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_nonexistent_edf_dir() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantEncodeLma
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    edf_dir: td.path().join("no-such-corpus"),
                    out_dir: td.path().join("out"),
                    corpus: String::new(),
                    quiet: true,
                    verify: false,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn stage_metadata_present() {
        assert_eq!(LamquantEncodeLma::NAME, "lamquant_encode_lma");
        assert_eq!(LamquantEncodeLma::SCHEMA, 1);
        // Output is the reused LmaCorpus artifact (NOT a new type).
        assert_eq!(
            <<LamquantEncodeLma as Stage>::Output as Artifact>::KIND,
            "lamquant.lma_corpus"
        );
    }

    #[test]
    fn deterministic_flag_is_false() {
        // Encode is byte-deterministic per recording, but the artifact
        // is a stat-fingerprinted dir — mirror convert_lma.
        const { assert!(!<LamquantEncodeLma as Stage>::DETERMINISTIC) };
    }

    #[test]
    fn quiet_defaults_to_true() {
        let a: Args = serde_json::from_str(r#"{"edf_dir":"/e","out_dir":"/o"}"#).unwrap();
        assert!(a.quiet);
        assert!(!a.verify);
    }

    #[test]
    fn count_lma_archives_scans_flat_and_nested() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("a.lma"), b"x").unwrap();
        let sub = td.path().join("corpus");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("b.lma"), b"y").unwrap();
        std::fs::write(sub.join("c.lma"), b"z").unwrap();
        std::fs::write(sub.join("notlma.txt"), b"-").unwrap();
        assert_eq!(count_lma_archives(td.path()), 3);
    }
}
