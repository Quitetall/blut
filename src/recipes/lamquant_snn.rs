//! Recipe — `lamquant_snn`.
//!
//! End-to-end Mamba SNN training pipeline with PCCP promotion gate.
//! Per ADR 0017 (BLUT-canonical + LMA-direct), the chain is:
//!
//!   lamquant_convert_lma           () → LmaCorpus
//!     → lamquant_train_mamba_snn   LmaCorpus → SnnCkpt
//!         → lamquant_pccp_gate_snn SnnCkpt → PccpVerdict
//!
//! Replaces the pre-ADR `lamquant_build_manifest → train_mamba_snn`
//! chain. The build_manifest stage stays registered + callable as a
//! standalone helper for tooling that still wants a Manifest
//! artifact, but the snn pipeline produces and consumes LmaCorpus
//! end-to-end now.
//!
//! PCCP defaults are dry-run + no-promote (safe). Recipes that want
//! to actually promote on PASS set `pccp_dry_run: false,
//! pccp_no_promote: false` + supply a real `change_id`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::lamquant_convert_lma::{Args as ConvertArgs, LamquantConvertLma};
use crate::stages::lamquant_pccp_gate_snn::{Args as GateArgs, LamquantPccpGateSnn};
use crate::stages::lamquant_train_mamba_snn::{Args as SnnArgs, LamquantTrainMambaSnn};

pub struct LamquantSnn;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty = env / default.
    #[serde(default)]
    pub lamquant_home: String,
    pub labels_dir: PathBuf,
    pub eeg_dir: PathBuf,
    /// SNN_CONFIGS preset: fast / standard / production.
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default)]
    pub subband: bool,
    #[serde(default)]
    pub infinite_lr: bool,

    // Optional overrides (None = preset default).
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

    // ── lamquant_convert_lma knobs ────────────────────────────
    /// LML source root. Empty = packer default.
    #[serde(default)]
    pub lml_root: String,
    /// Labels NPZ dir relative to lamquant_home. Empty = packer default.
    #[serde(default)]
    pub labels_dir_rel: String,
    /// Output LMA corpus dir. Required.
    pub lma_output_dir: PathBuf,
    /// Packer worker count. None = packer default (cpu_count / 3).
    #[serde(default)]
    pub convert_workers: Option<u32>,
    /// Cap conversion to first N stems (smoke runs).
    #[serde(default)]
    pub convert_limit: Option<u32>,

    /// Subject-grouped split manifest JSON path. Required — BLUT-driven
    /// SNN training cannot operate without a deterministic split.
    pub split_manifest: String,

    // PCCP gate knobs (safe-by-default).
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

fn default_preset() -> String {
    "production".into()
}
fn default_change_id() -> String {
    "PCCP-CHG-DRYRUN".into()
}
fn default_description() -> String {
    "(blut/lamquant_snn recipe run)".into()
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

impl Recipe for LamquantSnn {
    type Backend = crate::backends::LamquantBackend;
    const NAME: &'static str = "lamquant_snn";
    const DESCRIPTION: &'static str = "Mamba SNN seizure / activity detector end-to-end: convert LML \
         to LMA, train (Gpu), promote via PCCP gate. LMA-direct per \
         ADR 0017. Safe-by-default (dry-run + no-promote) — recipes \
         must explicitly opt into real promotion.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        // R23: arg-range validation at the recipe boundary.
        if !matches!(args.preset.as_str(), "fast" | "standard" | "production") {
            return Err(RecipeError::InvalidArgs(format!(
                "preset '{}' must be fast|standard|production",
                args.preset
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
        if args.lma_output_dir.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "lma_output_dir is required (writable directory for \
                 the converted LMA corpus)"
                    .into(),
            ));
        }
        if args.split_manifest.is_empty() {
            return Err(RecipeError::InvalidArgs(
                "split_manifest is required (LMA-direct SNN training \
                 cannot operate without a subject-grouped split)"
                    .into(),
            ));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                LamquantConvertLma,
                ConvertArgs {
                    lamquant_home: args.lamquant_home.clone(),
                    lml_root: args.lml_root.clone(),
                    labels_dir_rel: args.labels_dir_rel.clone(),
                    output_dir: args.lma_output_dir.clone(),
                    workers: args.convert_workers,
                    limit: args.convert_limit,
                    keep_sources: false,
                    dry_run: false,
                },
            )
            .then(
                LamquantTrainMambaSnn,
                SnnArgs {
                    lamquant_home: args.lamquant_home.clone(),
                    labels_dir: args.labels_dir.clone(),
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
                    // Explicit lma_root mirrors lamquant_encoder's
                    // recipe pattern — passes the resolved output
                    // dir at compile time rather than relying on
                    // the stage's runtime fallback to `input.root`.
                    // Keeps cross-recipe behaviour uniform.
                    lma_root: args.lma_output_dir.display().to_string(),
                    split_manifest: args.split_manifest.clone(),
                },
            )
            .then(
                LamquantPccpGateSnn,
                GateArgs {
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

pub static DEF: RecipeDef = RecipeDef {
    name: LamquantSnn::NAME,
    description: LamquantSnn::DESCRIPTION,
    backend_id: <crate::backends::LamquantBackend as crate::backends::TrainingBackend>::ID,
    category: crate::recipes::recipe::RecipeCategory::Pipeline,
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
        LamquantSnn.compile(args).map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            lamquant_home: String::new(),
            labels_dir: PathBuf::from("/tmp/labels"),
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
            lml_root: String::new(),
            labels_dir_rel: String::new(),
            lma_output_dir: PathBuf::from("/tmp/lma"),
            convert_workers: None,
            convert_limit: None,
            split_manifest: "/tmp/split.json".into(),
            pccp_change_id: default_change_id(),
            pccp_description: default_description(),
            pccp_author: default_author(),
            pccp_change_class: default_change_class(),
            pccp_dry_run: true,
            pccp_no_promote: true,
        }
    }

    #[test]
    fn compiles_to_3_node_plan() {
        let plan = LamquantSnn.compile(args()).unwrap().into_compiled();
        // 3 nodes: convert_lma → train_mamba_snn → pccp_gate_snn
        // 2 edges. (build_manifest dropped per ADR 0017.)
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn rejects_invalid_preset() {
        let mut a = args();
        a.preset = "nonsense".into();
        let r = LamquantSnn.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_empty_output_dir() {
        let mut a = args();
        a.lma_output_dir = PathBuf::new();
        let r = LamquantSnn.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_empty_split_manifest() {
        let mut a = args();
        a.split_manifest = String::new();
        let r = LamquantSnn.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn safe_defaults_when_pccp_fields_omitted() {
        // Bare minimum args; PCCP fields should default to safe
        // (dry-run + no-promote).
        let raw = serde_json::json!({
            "labels_dir": "/tmp/labels",
            "eeg_dir": "/tmp/eeg",
            "lma_output_dir": "/tmp/lma",
            "split_manifest": "/tmp/split.json",
        });
        let a: Args = serde_json::from_value(raw).unwrap();
        assert!(a.pccp_dry_run);
        assert!(a.pccp_no_promote);
    }
}
