//! Recipe — `lamquant_full_pipeline`.
//!
//! The end-to-end LamQuant SNN pipeline the bar asks for (STATE_REVIEW
//! §4.1): EDF→.lma encode → split-manifest → train → PCCP gate, all in
//! one recipe so the pipeline is self-contained + serializable. Per
//! ADR 0017 (BLUT-canonical + LMA-direct) the typed chain is:
//!
//! ```text
//! lamquant_encode_lma              () → LmaCorpus            (RCP-2)
//!   → lamquant_build_split_manifest  LmaCorpus → SplitManifest (RCP-3)
//!   → _corpus_rebind_from_split      SplitManifest → LmaCorpus (bridge)
//!   → lamquant_train_mamba_snn       LmaCorpus → SnnCkpt
//!   → lamquant_pccp_gate_snn         SnnCkpt → PccpVerdict
//! ```
//!
//! Why the bridge? `build_split_manifest` MUST run before training
//! (train reads the split manifest it writes), so it sits on the
//! critical path — but `train_mamba_snn` consumes a `LmaCorpus`, not a
//! `SplitManifest`. `_corpus_rebind_from_split` is a tiny deterministic
//! typed bridge that re-emits the encoded corpus so the linear
//! `Plan::then()` chain stays typed end-to-end. Same pattern as
//! `lamquant_encoder`'s `_corpus_rebind_from_mae`. Pure data plumbing.
//!
//! Why NOT `lamquant_generate_snn_labels` in the chain? That stage's
//! typed input is `(LmaCorpus, Manifest)` — it needs a `Manifest`
//! artifact this pipeline doesn't produce, and labels are the
//! pre-generated canonical NPZs at `paths::DEFAULT_LABELS_DIR` (RCP-6)
//! that both `build_split_manifest` (`--labels`) and `train_mamba_snn`
//! (`--data`) read by path. Generating labels is a separate data-prep
//! concern (`lamquant_generate_snn_labels` runs standalone); folding it
//! into this chain would require fabricating a Manifest + a second
//! bridge for no data-flow benefit. Documented gap, not fudged.
//!
//! Safe-by-default: PCCP gate runs dry-run + no-promote unless the
//! caller explicitly opts into real promotion.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use blut::framework::error::RecipeError;
use blut::framework::plan::Plan;
use blut::recipes::recipe::{Recipe, RecipeDef};

pub struct LamquantFullPipeline;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty = detected `ai_models_root`.
    #[serde(default)]
    pub lamquant_home: String,

    // ── encode_lma (RCP-2) ────────────────────────────────────
    /// Input EDF/BDF corpus dir → `lml encode`. Required.
    pub edf_dir: PathBuf,
    /// Output LMA corpus dir (`lml encode -o`). Required. Also the
    /// `--lma-root` for split-manifest + the `--lma-root` for train.
    pub lma_output_dir: PathBuf,
    /// Optional human label for the corpus.
    #[serde(default)]
    pub corpus: String,
    /// `lml encode --verify` (decode-back roundtrip check).
    #[serde(default)]
    pub encode_verify: bool,

    // ── build_split_manifest (RCP-3) ──────────────────────────
    /// `--labels <dir>` for split-manifest + `--data <dir>` for train.
    /// Empty = the unified canonical labels root (RCP-6).
    #[serde(default)]
    pub labels_dir: String,
    /// Where the split manifest JSON is written. Required.
    pub split_manifest: PathBuf,
    /// `--val-fraction` for the patient-level split.
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,

    // ── train_mamba_snn ───────────────────────────────────────
    /// `--eeg-dir <dir>` raw EEG sources for training.
    pub eeg_dir: PathBuf,
    /// SNN_CONFIGS preset: fast / standard / production.
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default)]
    pub subband: bool,
    #[serde(default)]
    pub infinite_lr: bool,
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
    #[serde(default)]
    pub checkpoint_rel: String,
    #[serde(default)]
    pub export_rel: String,

    // ── PCCP gate (safe-by-default) ───────────────────────────
    #[serde(default = "default_change_id")]
    pub pccp_change_id: String,
    #[serde(default = "default_description")]
    pub pccp_description: String,
    #[serde(default = "default_author")]
    pub pccp_author: String,
    #[serde(default = "default_change_class")]
    pub pccp_change_class: String,
    #[serde(default = "default_true")]
    pub pccp_dry_run: bool,
    #[serde(default = "default_true")]
    pub pccp_no_promote: bool,
}

fn default_val_fraction() -> f32 {
    0.10
}
fn default_preset() -> String {
    "production".into()
}
fn default_change_id() -> String {
    "PCCP-CHG-DRYRUN".into()
}
fn default_description() -> String {
    "(blut/lamquant_full_pipeline recipe run)".into()
}
fn default_author() -> String {
    "BLUT".into()
}
fn default_change_class() -> String {
    "A.1".into()
}
fn default_true() -> bool {
    true
}

impl Recipe for LamquantFullPipeline {
    type Backend = crate::backends::LamquantBackend;
    const NAME: &'static str = "lamquant_full_pipeline";
    const DESCRIPTION: &'static str = "Full LamQuant SNN pipeline end-to-end: EDF→.lma encode (lml) → \
         patient-level seizure-stratified split manifest → Mamba SNN \
         train (Gpu) → PCCP gate. LMA-direct per ADR 0017. \
         Safe-by-default PCCP gate (dry-run + no-promote).";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        // R23: arg-range validation at the recipe boundary.
        if args.edf_dir.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "edf_dir is required (EDF/BDF corpus dir for lml encode)".into(),
            ));
        }
        if args.lma_output_dir.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "lma_output_dir is required (writable dir for the encoded LMA corpus)".into(),
            ));
        }
        if args.split_manifest.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "split_manifest is required (path for the generated split_manifest.json)".into(),
            ));
        }
        if args.eeg_dir.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "eeg_dir is required (raw EEG sources for training)".into(),
            ));
        }
        if !matches!(args.preset.as_str(), "fast" | "standard" | "production") {
            return Err(RecipeError::InvalidArgs(format!(
                "preset '{}' must be fast|standard|production",
                args.preset
            )));
        }
        if !(args.val_fraction > 0.0 && args.val_fraction < 1.0 && args.val_fraction.is_finite()) {
            return Err(RecipeError::InvalidArgs(format!(
                "val_fraction must be in (0,1); got {}",
                args.val_fraction
            )));
        }
        if let Some(e) = args.epochs {
            if e == 0 {
                return Err(RecipeError::InvalidArgs("epochs must be > 0".into()));
            }
        }
        if let Some(lr) = args.lr {
            if !(lr > 0.0 && lr.is_finite()) {
                return Err(RecipeError::InvalidArgs(format!(
                    "lr must be positive finite; got {lr}"
                )));
            }
        }
        if let Some(b) = args.batch_size {
            if b == 0 {
                return Err(RecipeError::InvalidArgs("batch_size must be > 0".into()));
            }
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let split_manifest_str = args.split_manifest.display().to_string();

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                crate::stages::LamquantEncodeLma,
                crate::stages::lamquant_encode_lma::Args {
                    edf_dir: args.edf_dir.clone(),
                    out_dir: args.lma_output_dir.clone(),
                    corpus: args.corpus.clone(),
                    quiet: true,
                    verify: args.encode_verify,
                },
            )
            .then(
                crate::stages::LamquantBuildSplitManifest,
                crate::stages::lamquant_build_split_manifest::Args {
                    lamquant_home: args.lamquant_home.clone(),
                    // Pin the corpus root explicitly so the split-manifest
                    // stage doesn't depend on the runtime input fallback.
                    lma_root: args.lma_output_dir.display().to_string(),
                    labels_dir: args.labels_dir.clone(),
                    out: args.split_manifest.clone(),
                    val_fraction: args.val_fraction,
                },
            )
            .then(
                CorpusRebindFromSplit,
                CorpusRebindArgs {
                    lma_output_dir: args.lma_output_dir.clone(),
                },
            )
            .then(
                crate::stages::LamquantTrainMambaSnn,
                crate::stages::lamquant_train_mamba_snn::Args {
                    lamquant_home: args.lamquant_home.clone(),
                    // Train reads labels via `--data`; default to the
                    // canonical labels root when unset (RCP-6).
                    labels_dir: if args.labels_dir.is_empty() {
                        PathBuf::from(crate::paths::DEFAULT_LABELS_DIR)
                    } else {
                        PathBuf::from(&args.labels_dir)
                    },
                    eeg_dir: args.eeg_dir.clone(),
                    preset: args.preset.clone(),
                    subband: args.subband,
                    infinite_lr: args.infinite_lr,
                    epochs: args.epochs,
                    lr: args.lr,
                    batch_size: args.batch_size,
                    lambda_spike: args.lambda_spike,
                    d_model: args.d_model,
                    d_state: args.d_state,
                    n_layers: args.n_layers,
                    max_windows_per_file: args.max_windows_per_file,
                    checkpoint_rel: args.checkpoint_rel.clone(),
                    export_rel: args.export_rel.clone(),
                    lma_root: args.lma_output_dir.display().to_string(),
                    split_manifest: split_manifest_str,
                },
            )
            .then(
                crate::stages::LamquantPccpGateSnn,
                crate::stages::lamquant_pccp_gate_snn::Args {
                    lamquant_home: args.lamquant_home.clone(),
                    change_id: args.pccp_change_id.clone(),
                    description: args.pccp_description.clone(),
                    author: args.pccp_author.clone(),
                    change_class: args.pccp_change_class.clone(),
                    dry_run: args.pccp_dry_run,
                    no_promote: args.pccp_no_promote,
                },
            )
            .finish();
        Ok(plan)
    }
}

// ── Typed bridge: SplitManifest → LmaCorpus ─────────────────────
//
// `build_split_manifest` must run before `train_mamba_snn` (train
// reads the manifest it writes), so it sits on the critical path —
// but train consumes a `LmaCorpus`, not a `SplitManifest`. This tiny
// deterministic stage re-emits the encoded LMA corpus so the linear
// `Plan::then()` chain stays typed end-to-end. Pure data plumbing;
// mirrors `lamquant_encoder`'s `_corpus_rebind_from_mae`.

use async_trait::async_trait;

use crate::artifacts::lamquant::stat_fingerprint_dir;
use crate::artifacts::{LmaCorpus, SplitManifest};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

struct CorpusRebindFromSplit;

impl blut::framework::Compatible<crate::backends::LamquantBackend> for CorpusRebindFromSplit {}

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
struct CorpusRebindArgs {
    /// LMA corpus root the upstream `encode_lma` wrote to. Threaded
    /// through so the bridge doesn't have to re-derive it.
    lma_output_dir: PathBuf,
}

#[async_trait]
impl Stage for CorpusRebindFromSplit {
    const NAME: &'static str = "_corpus_rebind_from_split";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    const DETERMINISTIC: bool = true;
    type Input = SplitManifest;
    type Output = LmaCorpus;
    type Args = CorpusRebindArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: SplitManifest,
        args: &CorpusRebindArgs,
    ) -> Result<LmaCorpus, StageError> {
        let root = &args.lma_output_dir;
        if !root.exists() {
            return Err(StageError::BadInput(format!(
                "LMA corpus dir not found: {}",
                root.display()
            )));
        }
        let content_hash =
            stat_fingerprint_dir(b"lamquant.lma_corpus", root).map_err(|source| {
                StageError::Io {
                    path: root.clone(),
                    source,
                }
            })?;
        let n_archives = std::fs::read_dir(root)
            .map_err(|source| StageError::Io {
                path: root.clone(),
                source,
            })?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.eq_ignore_ascii_case("lma"))
                    .unwrap_or(false)
            })
            .count() as i64;
        Ok(LmaCorpus {
            root: root.clone(),
            n_archives,
            content_hash,
        })
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: LamquantFullPipeline::NAME,
    description: LamquantFullPipeline::DESCRIPTION,
    backend_id: <crate::backends::LamquantBackend as blut::backends::TrainingBackend>::ID,
    category: blut::recipes::recipe::RecipeCategory::Pipeline,
    input_kinds: &[],
    output_kind: "lamquant.pccp_verdict",
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args =
            serde_json::from_value(raw).map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        LamquantFullPipeline
            .compile(args)
            .map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            lamquant_home: String::new(),
            edf_dir: PathBuf::from("/tmp/edf"),
            lma_output_dir: PathBuf::from("/tmp/lma"),
            corpus: String::new(),
            encode_verify: false,
            labels_dir: String::new(),
            split_manifest: PathBuf::from("/tmp/split.json"),
            val_fraction: default_val_fraction(),
            eeg_dir: PathBuf::from("/tmp/eeg"),
            preset: default_preset(),
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
            pccp_change_id: default_change_id(),
            pccp_description: default_description(),
            pccp_author: default_author(),
            pccp_change_class: default_change_class(),
            pccp_dry_run: true,
            pccp_no_promote: true,
        }
    }

    #[test]
    fn compiles_to_5_node_plan() {
        // encode_lma → build_split_manifest → _corpus_rebind_from_split
        //   → train_mamba_snn → pccp_gate_snn = 5 nodes, 4 edges.
        let plan = LamquantFullPipeline
            .compile(args())
            .unwrap()
            .into_compiled();
        assert_eq!(plan.n_nodes(), 5);
        assert_eq!(plan.n_edges(), 4);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn rejects_empty_edf_dir() {
        let mut a = args();
        a.edf_dir = PathBuf::new();
        assert!(matches!(
            LamquantFullPipeline.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_empty_lma_output_dir() {
        let mut a = args();
        a.lma_output_dir = PathBuf::new();
        assert!(matches!(
            LamquantFullPipeline.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_empty_split_manifest() {
        let mut a = args();
        a.split_manifest = PathBuf::new();
        assert!(matches!(
            LamquantFullPipeline.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_empty_eeg_dir() {
        let mut a = args();
        a.eeg_dir = PathBuf::new();
        assert!(matches!(
            LamquantFullPipeline.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_invalid_preset() {
        let mut a = args();
        a.preset = "nonsense".into();
        assert!(matches!(
            LamquantFullPipeline.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_out_of_range_val_fraction() {
        let mut a = args();
        a.val_fraction = 0.0;
        assert!(matches!(
            LamquantFullPipeline.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn safe_pccp_defaults_when_omitted() {
        let raw = serde_json::json!({
            "edf_dir": "/tmp/edf",
            "lma_output_dir": "/tmp/lma",
            "split_manifest": "/tmp/split.json",
            "eeg_dir": "/tmp/eeg",
        });
        let a: Args = serde_json::from_value(raw).unwrap();
        assert!(a.pccp_dry_run);
        assert!(a.pccp_no_promote);
        assert!((a.val_fraction - 0.10).abs() < 1e-6);
        assert_eq!(a.preset, "production");
    }

    #[test]
    fn arg_schema_generates() {
        let schema = (DEF.args_schema_fn)();
        assert!(schema != serde_json::Value::Null);
    }

    // ── CorpusRebindFromSplit bridge tests ────────────────────

    use blut::framework::artifact::ContentHash;

    fn split(td: &std::path::Path) -> SplitManifest {
        let p = td.join("split.json");
        std::fs::write(&p, b"{}").unwrap();
        SplitManifest {
            path: p,
            content_hash: ContentHash::of_bytes(b"{}"),
            n_train_subjects: 9,
            n_val_subjects: 1,
        }
    }

    fn bridge_ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn bridge_rejects_missing_corpus_dir() {
        let td = tempfile::tempdir().unwrap();
        let r = CorpusRebindFromSplit
            .run(
                &bridge_ctx(td.path()),
                split(td.path()),
                &CorpusRebindArgs {
                    lma_output_dir: td.path().join("__not_a_dir__"),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn bridge_reemits_corpus_with_archive_count() {
        let td = tempfile::tempdir().unwrap();
        let corpus = td.path().join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        std::fs::write(corpus.join("a.lma"), b"a").unwrap();
        std::fs::write(corpus.join("b.lma"), b"b").unwrap();
        std::fs::write(corpus.join("notes.txt"), b"x").unwrap();
        let out = CorpusRebindFromSplit
            .run(
                &bridge_ctx(td.path()),
                split(td.path()),
                &CorpusRebindArgs {
                    lma_output_dir: corpus.clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(out.root, corpus);
        assert_eq!(out.n_archives, 2);
        assert_ne!(out.content_hash, ContentHash::of_bytes(b""));
    }
}
