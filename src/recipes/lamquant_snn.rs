//! Recipe — `lamquant_snn`.
//!
//! End-to-end Mamba SNN training pipeline with PCCP promotion gate.
//! First runnable LamQuant pipeline through the BLUT framework:
//!
//!   lamquant_build_manifest
//!     → lamquant_train_mamba_snn
//!         → lamquant_pccp_gate_snn
//!
//! PCCP defaults are dry-run + no-promote (safe). Recipes that want
//! to actually promote on PASS set `pccp_dry_run: false,
//! pccp_no_promote: false` + supply a real `change_id`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::lamquant_build_manifest::{Args as MfArgs, LamquantBuildManifest};
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

    // Manifest builder knobs. Empty string = builder default
    // (q31_dir: ai_models/dataset_sim/q31_events; output_rel:
    // ai_models/dataset_sim/manifest_v3.json; v2_path: alongside
    // q31_dir). Recipes typically leave these blank.
    #[serde(default)]
    pub manifest_q31_dir: String,
    #[serde(default)]
    pub manifest_output_rel: String,
    #[serde(default)]
    pub manifest_v2_path: String,
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,
    #[serde(default = "default_seed")]
    pub manifest_seed: u64,

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
fn default_val_fraction() -> f32 {
    0.05
}
fn default_seed() -> u64 {
    42
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
    const NAME: &'static str = "lamquant_snn";
    const DESCRIPTION: &'static str =
        "Mamba SNN seizure / activity detector end-to-end: build manifest, \
         train (Gpu), promote via PCCP gate. Safe-by-default (dry-run + \
         no-promote) — recipes must explicitly opt into real promotion.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError> {
        // R23: arg-range validation at the recipe boundary.
        if !matches!(args.preset.as_str(), "fast" | "standard" | "production") {
            return Err(RecipeError::InvalidArgs(format!(
                "preset '{}' must be fast|standard|production",
                args.preset
            )));
        }
        if !(args.val_fraction > 0.0 && args.val_fraction < 1.0) {
            return Err(RecipeError::InvalidArgs(format!(
                "val_fraction must be in (0, 1); got {}",
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

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                LamquantBuildManifest,
                MfArgs {
                    lamquant_home: args.lamquant_home.clone(),
                    q31_dir: args.manifest_q31_dir.clone(),
                    output_rel: args.manifest_output_rel.clone(),
                    v2_path: args.manifest_v2_path.clone(),
                    val_fraction: args.val_fraction,
                    seed: args.manifest_seed,
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
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        LamquantSnn.compile(args)
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
            manifest_q31_dir: String::new(),
            manifest_output_rel: String::new(),
            manifest_v2_path: String::new(),
            val_fraction: 0.05,
            manifest_seed: 42,
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
        let plan = LamquantSnn.compile(args()).unwrap();
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
    fn safe_defaults_when_pccp_fields_omitted() {
        // Bare minimum args; PCCP fields should default to safe
        // (dry-run + no-promote).
        let raw = serde_json::json!({
            "labels_dir": "/tmp/labels",
            "eeg_dir": "/tmp/eeg",
        });
        let a: Args = serde_json::from_value(raw).unwrap();
        assert!(a.pccp_dry_run);
        assert!(a.pccp_no_promote);
    }
}
