//! Recipe — `lamquant_oracle`.
//!
//! Teacher-only pipeline (LMA-direct per ADR 0017):
//!
//!   lamquant_convert_lma             () → LmaCorpus
//!     → lamquant_train_l3_teacher    LmaCorpus → TeacherCkpt
//!     → _teacher_to_joint_adapter    TeacherCkpt → JointCkpt   [bridge]
//!     → lamquant_pccp_gate_encoder   JointCkpt → PccpVerdict
//!
//! Replaces the pre-ADR `build_manifest → precompute_fullband →
//! train_teacher → ...` chain. `train_teacher` stays registered as a
//! standalone helper for legacy callers that still need the
//! fullband-memmap path, but the canonical oracle pipeline runs
//! through `train_l3_teacher` which honors the `--lma-root` +
//! `--split-manifest` flags.
//!
//! Uses `LamquantPccpGateEncoder` over the teacher's path-wrapped
//! JointCkpt-shape (encoder_path = teacher_path) so the existing
//! encoder-class gate logic handles it. PCCP gate model string is
//! still "encoder" — the gate's `--model` flag is coupled to the
//! gate.py evaluator table; a "teacher" evaluator class will land
//! when the gate script grows one.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};

pub struct LamquantOracle;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,

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

    /// Subject-grouped split manifest JSON path. Required.
    pub split_manifest: String,

    // ── lamquant_train_l3_teacher knobs ───────────────────────
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub lr_min: Option<f32>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub windows_per_epoch: Option<u32>,
    #[serde(default)]
    pub max_windows: Option<u32>,
    /// `--device` (default "auto"). Empty = pass nothing.
    #[serde(default)]
    pub device: String,
    #[serde(default)]
    pub resume: bool,

    // PCCP (safe-by-default).
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

fn default_change_id() -> String {
    "PCCP-CHG-DRYRUN".into()
}
fn default_description() -> String {
    "(blut/lamquant_oracle recipe run)".into()
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

impl Recipe for LamquantOracle {
    type Backend = crate::backends::LamquantBackend;
    const NAME: &'static str = "lamquant_oracle";
    const DESCRIPTION: &'static str = "LamQuant teacher pipeline: convert_lma → train_l3_teacher \
         (Gpu, nondet) → pccp_gate_encoder. LMA-direct per ADR 0017. \
         Safe-by-default PCCP gate.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        if let Some(e) = args.epochs {
            if e == 0 {
                return Err(RecipeError::InvalidArgs("epochs must be > 0".into()));
            }
        }
        if let Some(b) = args.batch_size {
            if b == 0 {
                return Err(RecipeError::InvalidArgs("batch_size must be > 0".into()));
            }
        }
        if args.lma_output_dir.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "lma_output_dir is required".into(),
            ));
        }
        if args.split_manifest.is_empty() {
            return Err(RecipeError::InvalidArgs(
                "split_manifest is required (LMA-direct teacher training \
                 cannot operate without a subject-grouped split)"
                    .into(),
            ));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let convert_args = crate::stages::lamquant_convert_lma::Args {
            lamquant_home: args.lamquant_home.clone(),
            lml_root: args.lml_root.clone(),
            labels_dir_rel: args.labels_dir_rel.clone(),
            output_dir: args.lma_output_dir.clone(),
            workers: args.convert_workers,
            limit: args.convert_limit,
            keep_sources: false,
            dry_run: false,
        };
        let teacher_args = crate::stages::lamquant_train_l3_teacher::Args {
            lamquant_home: args.lamquant_home.clone(),
            epochs: args.epochs,
            batch_size: args.batch_size,
            lr: args.lr,
            lr_min: args.lr_min,
            width: args.width,
            windows_per_epoch: args.windows_per_epoch,
            max_windows: args.max_windows,
            device: args.device.clone(),
            resume: args.resume,
            lma_root: args.lma_output_dir.display().to_string(),
            split_manifest: args.split_manifest.clone(),
        };
        let gate_args = crate::stages::lamquant_pccp_gate_encoder::Args {
            lamquant_home: args.lamquant_home.clone(),
            change_id: args.pccp_change_id.clone(),
            description: args.pccp_description.clone(),
            author: args.pccp_author.clone(),
            change_class: args.pccp_change_class.clone(),
            dry_run: args.pccp_dry_run,
            no_promote: args.pccp_no_promote,
        };

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(crate::stages::LamquantConvertLma, convert_args)
            .then(crate::stages::LamquantTrainL3Teacher, teacher_args)
            .then(TeacherToJointAdapter, TeacherToJointArgs::default())
            .then(crate::stages::LamquantPccpGateEncoder, gate_args)
            .finish();

        Ok(plan)
    }
}

// ── Adapter: TeacherCkpt → JointCkpt (encoder_path = teacher_path) ──
//
// pccp_gate_encoder takes JointCkpt and runs the gate against
// `input.encoder_path`. By wrapping a TeacherCkpt with the teacher
// path stuffed into encoder_path, the encoder-class gate evaluates
// the teacher unchanged. Decoder path is a sentinel.

use async_trait::async_trait;

use crate::artifacts::{JointCkpt, TeacherCkpt};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

struct TeacherToJointAdapter;

impl crate::framework::Compatible<crate::backends::LamquantBackend> for TeacherToJointAdapter {}

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
struct TeacherToJointArgs {}

#[async_trait]
impl Stage for TeacherToJointAdapter {
    const NAME: &'static str = "_teacher_to_joint_adapter";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = TeacherCkpt;
    type Output = JointCkpt;
    type Args = TeacherToJointArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: TeacherCkpt,
        _args: &TeacherToJointArgs,
    ) -> Result<JointCkpt, StageError> {
        Ok(JointCkpt {
            encoder_path: input.path.clone(),
            decoder_path: std::path::PathBuf::from("(teacher-only)"),
            content_hash: input.content_hash,
            final_loss: input.final_loss,
            tier: 0,
            preset: format!("teacher-{}", input.gen_tag),
        })
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: LamquantOracle::NAME,
    description: LamquantOracle::DESCRIPTION,
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
        LamquantOracle.compile(args).map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            lamquant_home: String::new(),
            lml_root: String::new(),
            labels_dir_rel: String::new(),
            lma_output_dir: PathBuf::from("/tmp/lma"),
            convert_workers: None,
            convert_limit: None,
            split_manifest: "/tmp/split.json".into(),
            epochs: None,
            batch_size: None,
            lr: None,
            lr_min: None,
            width: None,
            windows_per_epoch: None,
            max_windows: None,
            device: String::new(),
            resume: false,
            pccp_change_id: default_change_id(),
            pccp_description: default_description(),
            pccp_author: default_author(),
            pccp_change_class: default_change_class(),
            pccp_dry_run: true,
            pccp_no_promote: true,
        }
    }

    #[test]
    fn compiles_to_4_node_plan() {
        // convert_lma → train_l3_teacher → adapter → gate.
        let plan = LamquantOracle.compile(args()).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 4);
        assert_eq!(plan.n_edges(), 3);
    }

    #[test]
    fn rejects_empty_output_dir() {
        let mut a = args();
        a.lma_output_dir = PathBuf::new();
        assert!(matches!(
            LamquantOracle.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_empty_split_manifest() {
        let mut a = args();
        a.split_manifest = String::new();
        assert!(matches!(
            LamquantOracle.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn safe_pccp_defaults() {
        let raw = serde_json::json!({
            "lma_output_dir": "/tmp/lma",
            "split_manifest": "/tmp/split.json",
        });
        let a: Args = serde_json::from_value(raw).unwrap();
        assert!(a.pccp_dry_run);
        assert!(a.pccp_no_promote);
    }
}
