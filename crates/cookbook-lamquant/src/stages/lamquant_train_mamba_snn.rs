//! Stage — `lamquant_train_mamba_snn`.
//!
//! Wraps `ai_models/snn/train_mamba_snn.py`. LmaCorpus → SnnCkpt.
//!
//! Per ADR 0017 (BLUT-canonical + LMA-direct), Input is the LMA
//! corpus; the kernel reads it via `--lma-root <input.root>` +
//! `--split-manifest <args.split_manifest>`. The pre-ADR
//! `--manifest` legacy random-split path is no longer emitted by
//! this stage — direct `python train_mamba_snn.py --manifest …`
//! callers continue to work, but BLUT-driven execution always
//! uses LMA-direct.
//!
//! Nondeterministic. Same args + same data produce slightly
//! different ckpt bytes due to GPU non-determinism + dataloader
//! shuffle + dropout. `DETERMINISTIC = false` so a re-trained
//! upstream doesn't cascade re-runs through downstream cache
//! lookups (logical hash decoupling lands in executor).

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::lamquant::stat_fingerprint;
use crate::artifacts::{LmaCorpus, SnnCkpt};
use crate::backends::lamquant::runner::{
    LamquantBackend, LamquantInvocation, resolve_lamquant_python,
};
use crate::stages::lamquant_helpers::{blut_python_script, blut_pythonpath, resolve_home};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct LamquantTrainMambaSnn;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty → `$LAMQUANT_HOME` or
    /// `~/Desktop/LamQuant`.
    #[serde(default)]
    pub lamquant_home: String,

    /// `--data <dir>` — directory containing `_labels.npz` files.
    /// Required; no preset default.
    pub labels_dir: PathBuf,

    /// `--eeg-dir <dir>` — directory containing raw EEG sources.
    pub eeg_dir: PathBuf,

    /// `--config` preset name. Maps to `SNN_CONFIGS` (fast /
    /// standard / production).
    #[serde(default = "default_preset")]
    pub preset: String,

    /// Enable `--subband` mode (L3 subband features).
    #[serde(default)]
    pub subband: bool,

    /// `--infinite-lr` for continual-training mode.
    #[serde(default)]
    pub infinite_lr: bool,

    /// Override `--epochs`. None = preset default.
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lambda_spike: Option<f32>,
    #[serde(default)]
    pub d_model: Option<u32>,
    #[serde(default)]
    pub d_state: Option<u32>,
    #[serde(default)]
    pub n_layers: Option<u32>,
    #[serde(default)]
    pub max_windows_per_file: Option<u32>,

    /// Output ckpt path. Empty = LamQuant default
    /// `weights/snn/mamba_snn_best.pt` relative to lamquant_home.
    #[serde(default)]
    pub checkpoint_rel: String,

    /// `--export` C-header path (optional). Empty = skip.
    #[serde(default)]
    pub export_rel: String,

    /// `--lma-root <path>` override. Empty = use `input.root`
    /// (the LmaCorpus produced by an upstream `convert_lma` stage).
    /// Non-empty values let the recipe override the corpus location
    /// without rebuilding the chain.
    #[serde(default)]
    pub lma_root: String,
    /// `--split-manifest <path>` — subject-grouped JSON split.
    /// REQUIRED in BLUT-driven execution; the stage rejects empty
    /// strings at compile time.
    pub split_manifest: String,
}

fn default_preset() -> String {
    "production".into()
}

#[async_trait]
impl Stage for LamquantTrainMambaSnn {
    const NAME: &'static str = "lamquant_train_mamba_snn";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = LmaCorpus;
    type Output = SnnCkpt;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: LmaCorpus,
        args: &Args,
    ) -> Result<SnnCkpt, StageError> {
        // RCP-1/RCP-7: an empty `lamquant_home` resolves to the
        // detected `ai_models_root` (the `LamQuant-Neural` submodule),
        // which holds both `ai_models/snn/train_mamba_snn.py` and the
        // `weights/` checkpoint tree this stage writes. `resolve_home`
        // canonicalizes + surfaces a missing root as a clear
        // `BadInput` preflight error instead of a low-level Io error.
        // MOVE-B (2026-05-29): the SNN trainer now lives in the PUBLIC
        // BLUT submodule at `<blut>/python/lamquant/snn/train_mamba_snn.py`,
        // resolved via `blut_python_root` ($BLUT_PYTHON). `lamquant_home`
        // is still resolved for checkpoint output (Neural `weights/`) +
        // data dirs.
        let lamquant_home = resolve_home(&args.lamquant_home)?;
        let python = resolve_lamquant_python(&lamquant_home);
        let (script, python_dir) =
            blut_python_script(&["python", "lamquant", "snn", "train_mamba_snn.py"])?;
        // Resolve labels_dir / eeg_dir against lamquant_home if
        // they're relative; existence-check after resolution.
        let labels_dir = resolve_relative(&lamquant_home, &args.labels_dir);
        let eeg_dir = resolve_relative(&lamquant_home, &args.eeg_dir);
        if !labels_dir.exists() {
            return Err(StageError::BadInput(format!(
                "labels_dir not found: {}",
                labels_dir.display()
            )));
        }
        if !eeg_dir.exists() {
            return Err(StageError::BadInput(format!(
                "eeg_dir not found: {}",
                eeg_dir.display()
            )));
        }

        // FW-2 caveat — writes OUTSIDE `ctx.stage_dir`.
        //
        // This stage's checkpoint lands at `lamquant_home/weights/
        // snn/*.pt` (or `args.checkpoint_rel` under lamquant_home),
        // NOT under `ctx.stage_dir`. The executor's framework-level
        // atomicity (run in `.tmp-<key>`, promote-on-Ok, remove-on-
        // Err) therefore does NOT cover this checkpoint: a mid-train
        // crash can leave a partial `.pt` behind, and the executor
        // cannot clean it because it never sees the path.
        //
        // This is tolerated rather than fixed here because (a) the
        // stage is `DETERMINISTIC = false` — its downstream logical
        // hash is synthesized, not content-derived, so a stale ckpt
        // can't poison a downstream cache key; (b) the checkpoint is
        // a stable user-facing weights artifact the operator wants
        // persisted at a known path across runs, not a throwaway in a
        // job-scoped stage dir; and (c) the trainer itself writes the
        // best checkpoint atomically (best-so-far swap) so a partial
        // `.pt` from a crash is overwritten on the next successful
        // epoch. The resume oracle (the cache entry) is still only
        // written by the executor AFTER the stage returns Ok, so a
        // crashed run never looks complete regardless of the orphan
        // `.pt`. Documented per FW-2 ("at minimum detect + document").
        let checkpoint_path = if args.checkpoint_rel.is_empty() {
            lamquant_home
                .join("weights")
                .join("snn")
                .join("mamba_snn_best.pt")
        } else {
            safe_join(&lamquant_home, &args.checkpoint_rel)?
        };
        debug_assert!(
            !checkpoint_path.starts_with(&ctx.stage_dir),
            "FW-2 note assumes ckpt is outside stage_dir; if it moved inside, \
             drop this caveat and let the framework promote it"
        );
        if let Some(parent) = checkpoint_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        // LMA-direct path: `--lma-root` from input.root (default) or
        // args.lma_root (override); `--split-manifest` from args.
        // Pre-ADR-0017 `--manifest` legacy flag is not emitted by
        // this stage; direct python invocations still support it.
        if args.split_manifest.is_empty() {
            return Err(StageError::BadInput(
                "split_manifest is required (LMA-direct training cannot \
                 operate without a subject-grouped split)"
                    .into(),
            ));
        }
        let lma_root = if args.lma_root.is_empty() {
            input.root.display().to_string()
        } else {
            args.lma_root.clone()
        };
        let mut cmd_args: Vec<String> = vec![
            "--data".into(),
            args.labels_dir.display().to_string(),
            "--eeg-dir".into(),
            args.eeg_dir.display().to_string(),
            "--lma-root".into(),
            lma_root,
            "--split-manifest".into(),
            args.split_manifest.clone(),
            "--config".into(),
            args.preset.clone(),
            "--checkpoint".into(),
            checkpoint_path.display().to_string(),
        ];
        if args.subband {
            cmd_args.push("--subband".into());
        }
        if args.infinite_lr {
            cmd_args.push("--infinite-lr".into());
        }
        push_opt_u32(&mut cmd_args, "--epochs", args.epochs);
        push_opt_f32(&mut cmd_args, "--lr", args.lr);
        push_opt_u32(&mut cmd_args, "--batch-size", args.batch_size);
        push_opt_f32(&mut cmd_args, "--lambda-spike", args.lambda_spike);
        push_opt_u32(&mut cmd_args, "--d-model", args.d_model);
        push_opt_u32(&mut cmd_args, "--d-state", args.d_state);
        push_opt_u32(&mut cmd_args, "--n-layers", args.n_layers);
        push_opt_u32(
            &mut cmd_args,
            "--max-windows-per-file",
            args.max_windows_per_file,
        );

        if !args.export_rel.is_empty() {
            let export_path = safe_join(&lamquant_home, &args.export_rel)?;
            cmd_args.push("--export".into());
            cmd_args.push(export_path.display().to_string());
        }
        // (--lma-root and --split-manifest emitted up-front in the
        // initial vec[] above; the conditional pushes that lived
        // here pre-ADR-0017 are folded into the new LMA-direct flow.)

        // BLUT identity for the RunManifest pre-hook to read.
        // MOVE-B PYTHONPATH: the trainer now imports `lamquant.*`
        // (BLUT-owned package under blut/python), plus the pip-installed
        // `lamquant_neural` (model defs) + `lamquant_core`/`lamquant_codec`
        // (lossless wheels). Only the blut/python package root needs
        // injecting; layer it onto any caller-supplied PYTHONPATH.
        let env = vec![
            ("BLUT_JOB_DIR".into(), ctx.job_dir.display().to_string()),
            ("BLUT_STAGE_NAME".into(), Self::NAME.to_string()),
            ("PYTHONPATH".into(), blut_pythonpath(&python_dir)),
        ];

        let inv = LamquantInvocation {
            python,
            script,
            cwd: python_dir.clone(),
            args: cmd_args,
            env,
            expected_outputs: vec![checkpoint_path.clone()],
            run_manifest_path: None,
        };

        // Progress fan-out: tqdm lines from the trainer surface as
        // StageStep events on the executor's status broadcast.
        let stage_name_owned = Self::NAME.to_string();
        let status_tx = ctx.status_tx.clone();
        let progress_cb = Box::new(move |p: crate::backends::lamquant::runner::Progress| {
            let _ = status_tx.send(blut::framework::status::StageEvent::StageStep {
                node_idx: 0,
                stage_name: stage_name_owned.clone(),
                update: serde_json::json!({
                    "kind": "tqdm",
                    "current": p.current,
                    "total": p.total,
                }),
            });
        })
            as Box<dyn Fn(crate::backends::lamquant::runner::Progress) + Send + Sync>;

        let mut backend = LamquantBackend::new();
        backend
            .run(inv, Some(progress_cb))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        // Stat-based fingerprint for the ckpt (HASH_CONTENTS=false
        // on SnnCkpt — multi-MB to GB file; bytes are stochastic).
        let content_hash =
            stat_fingerprint(b"lamquant.ckpt.snn", &checkpoint_path).map_err(|source| {
                StageError::Io {
                    path: checkpoint_path.clone(),
                    source,
                }
            })?;

        // head_size_kb + final_loss read from a sidecar JSON if the
        // trainer wrote one alongside the ckpt; both fields are
        // best-effort (firmware export pipeline reads them).
        let (head_size_kb, final_loss) = read_snn_sidecar(&checkpoint_path);

        Ok(SnnCkpt {
            path: checkpoint_path,
            content_hash,
            head_size_kb,
            final_loss,
        })
    }
}

/// Join a user-supplied relative path onto `base`, rejecting any
/// component that would escape (`..`) or that supplies an absolute
/// path. Defense against path traversal in stage args. Returns
/// `BadInput` rather than silently relocating writes outside the
/// LamQuant repo root.
fn safe_join(base: &std::path::Path, rel: &str) -> Result<PathBuf, StageError> {
    let p = std::path::Path::new(rel);
    if p.is_absolute() {
        return Err(StageError::BadInput(format!(
            "path '{rel}' must be relative to lamquant_home (no absolute paths)"
        )));
    }
    for component in p.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(StageError::BadInput(format!(
                "path '{rel}' contains '..' — refusing traversal outside lamquant_home"
            )));
        }
    }
    Ok(base.join(p))
}

/// Resolve a possibly-relative path against `base`. Unlike
/// `safe_join`, this allows absolute paths (the caller is
/// supplying a fully-qualified data directory, not relocating an
/// output). Used for read-side args (`labels_dir`, `eeg_dir`).
fn resolve_relative(base: &std::path::Path, p: &std::path::Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

fn push_opt_u32(out: &mut Vec<String>, flag: &str, v: Option<u32>) {
    if let Some(v) = v {
        out.push(flag.to_string());
        out.push(v.to_string());
    }
}
fn push_opt_f32(out: &mut Vec<String>, flag: &str, v: Option<f32>) {
    if let Some(v) = v {
        out.push(flag.to_string());
        out.push(v.to_string());
    }
}

/// Optional sidecar JSON read; trainer writes
/// `<ckpt>.meta.json` with `{"head_size_kb": f, "final_loss": f}`.
/// Missing / malformed sidecar is non-fatal; both fields default to
/// 0.0 so the artifact still records the produced ckpt's path.
fn read_snn_sidecar(ckpt: &std::path::Path) -> (f32, f32) {
    let sidecar = ckpt.with_extension("meta.json");
    let body = match std::fs::read_to_string(&sidecar) {
        Ok(s) => s,
        Err(_) => return (0.0, 0.0),
    };
    let parsed: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return (0.0, 0.0),
    };
    let head = parsed
        .get("head_size_kb")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let loss = parsed
        .get("final_loss")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    (head, loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::artifact::ContentHash;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn corpus(p: &std::path::Path) -> LmaCorpus {
        LmaCorpus {
            root: p.to_path_buf(),
            n_archives: 1,
            content_hash: ContentHash::of_bytes(b"c"),
        }
    }

    #[tokio::test]
    async fn rejects_missing_lamquant_home() {
        let td = tempfile::tempdir().unwrap();
        let split_path = td.path().join("split.json");
        std::fs::write(&split_path, "{}").unwrap();
        let r = LamquantTrainMambaSnn
            .run(
                &ctx(td.path()),
                corpus(td.path()),
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    labels_dir: td.path().to_path_buf(),
                    eeg_dir: td.path().to_path_buf(),
                    preset: "production".into(),
                    subband: false,
                    infinite_lr: false,
                    epochs: None,
                    lr: None,
                    batch_size: None,
                    lambda_spike: None,
                    d_model: None,
                    d_state: None,
                    n_layers: None,
                    max_windows_per_file: None,
                    checkpoint_rel: String::new(),
                    export_rel: String::new(),
                    lma_root: String::new(),
                    split_manifest: split_path.display().to_string(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_missing_labels_dir() {
        let td = tempfile::tempdir().unwrap();
        let split_path = td.path().join("split.json");
        std::fs::write(&split_path, "{}").unwrap();
        // Lay down a fake lamquant_home with the trainer script so
        // we get past the script-exists check + reach labels_dir.
        let home = td.path().join("home");
        std::fs::create_dir_all(home.join("ai_models").join("snn")).unwrap();
        std::fs::write(
            home.join("ai_models")
                .join("snn")
                .join("train_mamba_snn.py"),
            "# stub\n",
        )
        .unwrap();
        let r = LamquantTrainMambaSnn
            .run(
                &ctx(td.path()),
                corpus(td.path()),
                &Args {
                    lamquant_home: home.display().to_string(),
                    labels_dir: td.path().join("no-labels"),
                    eeg_dir: td.path().to_path_buf(),
                    preset: "production".into(),
                    subband: false,
                    infinite_lr: false,
                    epochs: None,
                    lr: None,
                    batch_size: None,
                    lambda_spike: None,
                    d_model: None,
                    d_state: None,
                    n_layers: None,
                    max_windows_per_file: None,
                    checkpoint_rel: String::new(),
                    export_rel: String::new(),
                    lma_root: String::new(),
                    split_manifest: split_path.display().to_string(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_empty_split_manifest() {
        // BLUT-driven execution requires split_manifest; the stage
        // refuses to run without it (ADR 0017 LMA-direct contract).
        let td = tempfile::tempdir().unwrap();
        let home = td.path().join("home");
        std::fs::create_dir_all(home.join("ai_models").join("snn")).unwrap();
        std::fs::write(
            home.join("ai_models")
                .join("snn")
                .join("train_mamba_snn.py"),
            "# stub\n",
        )
        .unwrap();
        let labels = home.join("labels");
        std::fs::create_dir_all(&labels).unwrap();
        let eeg = home.join("eeg");
        std::fs::create_dir_all(&eeg).unwrap();
        let r = LamquantTrainMambaSnn
            .run(
                &ctx(td.path()),
                corpus(td.path()),
                &Args {
                    lamquant_home: home.display().to_string(),
                    labels_dir: labels,
                    eeg_dir: eeg,
                    preset: "production".into(),
                    subband: false,
                    infinite_lr: false,
                    epochs: None,
                    lr: None,
                    batch_size: None,
                    lambda_spike: None,
                    d_model: None,
                    d_state: None,
                    n_layers: None,
                    max_windows_per_file: None,
                    checkpoint_rel: String::new(),
                    export_rel: String::new(),
                    lma_root: String::new(),
                    split_manifest: String::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn safe_join_rejects_parent_dir() {
        let base = std::path::Path::new("/tmp/lamquant");
        let r = safe_join(base, "../../etc/cron.d/evil");
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn safe_join_rejects_absolute_path() {
        let base = std::path::Path::new("/tmp/lamquant");
        let r = safe_join(base, "/etc/cron.d/evil");
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn safe_join_accepts_normal_relative() {
        let base = std::path::Path::new("/tmp/lamquant");
        let r = safe_join(base, "weights/snn/m.pt").unwrap();
        assert_eq!(r, PathBuf::from("/tmp/lamquant/weights/snn/m.pt"));
    }

    #[test]
    fn resolve_relative_handles_absolute_paths() {
        let base = std::path::Path::new("/tmp/lamquant");
        let abs = std::path::Path::new("/mnt/4tb/data");
        let rel = std::path::Path::new("subdir/data");
        assert_eq!(resolve_relative(base, abs), PathBuf::from("/mnt/4tb/data"));
        assert_eq!(
            resolve_relative(base, rel),
            PathBuf::from("/tmp/lamquant/subdir/data")
        );
    }

    #[test]
    fn push_opt_appends_when_set() {
        let mut v: Vec<String> = Vec::new();
        push_opt_u32(&mut v, "--epochs", Some(5));
        push_opt_f32(&mut v, "--lr", Some(0.001));
        push_opt_u32(&mut v, "--batch", None);
        assert_eq!(v, vec!["--epochs", "5", "--lr", "0.001"]);
    }

    #[test]
    fn deterministic_flag_is_false_for_training() {
        const { assert!(!<LamquantTrainMambaSnn as Stage>::DETERMINISTIC) };
    }

    #[test]
    fn sidecar_returns_defaults_for_missing_file() {
        let td = tempfile::tempdir().unwrap();
        let ckpt = td.path().join("nope.pt");
        let (h, l) = read_snn_sidecar(&ckpt);
        assert_eq!(h, 0.0);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn sidecar_parses_present_file() {
        let td = tempfile::tempdir().unwrap();
        let ckpt = td.path().join("m.pt");
        let sidecar = ckpt.with_extension("meta.json");
        std::fs::write(&sidecar, r#"{"head_size_kb": 4.2, "final_loss": 0.123}"#).unwrap();
        let (h, l) = read_snn_sidecar(&ckpt);
        assert!((h - 4.2).abs() < 1e-6);
        assert!((l - 0.123).abs() < 1e-6);
    }
}
