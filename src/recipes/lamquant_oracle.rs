//! Recipe — `lamquant_oracle`.
//!
//! Teacher-only pipeline:
//!
//!   build_manifest → precompute_fullband → train_teacher → pccp_gate_encoder
//!
//! Uses `LamquantPccpGateEncoder` over the teacher's path-wrapped
//! JointCkpt-shape (teacher_path stuffed into encoder_path) so the
//! existing encoder-class gate logic handles it. PCCP gate model
//! string is still "encoder" here — the gate's `--model` flag is
//! coupled to the gate.py evaluator table; teacher class will land
//! when the gate script grows a "teacher" evaluator.

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};

pub struct LamquantOracle;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default = "default_true")]
    pub headless: bool,
    #[serde(default)]
    pub force_batch_size: Option<u32>,
    #[serde(default)]
    pub freq_weighted_loss: bool,
    #[serde(default)]
    pub resume: bool,
    /// Optional `--logger wandb|mlflow`. Empty = skip.
    #[serde(default)]
    pub logger: String,
    // Manifest knobs.
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,
    #[serde(default = "default_seed64")]
    pub manifest_seed: u64,
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

fn default_seed() -> u32 {
    42
}
fn default_seed64() -> u64 {
    42
}
fn default_val_fraction() -> f32 {
    0.05
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
    const NAME: &'static str = "lamquant_oracle";
    const DESCRIPTION: &'static str =
        "LamQuant teacher pipeline: build_manifest → precompute_fullband → \
         train_teacher (Gpu, nondet). PCCP gate over the trained teacher \
         ckpt. Safe-by-default.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError> {
        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        // After train_teacher, the leading edge is TeacherCkpt. Need
        // to feed `pccp_gate` which currently has variants typed on
        // JointCkpt/SnnCkpt. For oracle, we add a tiny passthrough
        // stage that boxes the teacher ckpt into a JointCkpt shape
        // (encoder_path = teacher_path) so the existing encoder
        // gate can score it. Bridges land alongside their recipe.
        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                crate::stages::LamquantBuildManifest,
                crate::stages::lamquant_build_manifest::Args {
                    lamquant_home: args.lamquant_home.clone(),
                    q31_dir: String::new(),
                    output_rel: String::new(),
                    v2_path: String::new(),
                    val_fraction: args.val_fraction,
                    seed: args.manifest_seed,
                },
            )
            .then(
                crate::stages::LamquantPrecomputeFullband,
                crate::stages::lamquant_precompute_fullband::Args {
                    lamquant_home: args.lamquant_home.clone(),
                    out_dir_rel: String::new(),
                    splits: vec!["train".into(), "val".into()],
                },
            )
            .then(
                crate::stages::LamquantTrainTeacher,
                crate::stages::lamquant_train_teacher::Args {
                    lamquant_home: args.lamquant_home.clone(),
                    headless: args.headless,
                    force_batch_size: args.force_batch_size,
                    seed: args.seed,
                    resume: args.resume,
                    logger: args.logger.clone(),
                    freq_weighted_loss: args.freq_weighted_loss,
                },
            )
            .then(TeacherToJointAdapter, TeacherToJointArgs {})
            .then(
                crate::stages::LamquantPccpGateEncoder,
                crate::stages::lamquant_pccp_gate_encoder::Args {
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
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).unwrap_or(serde_json::Value::Null)
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        LamquantOracle.compile(args)
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            lamquant_home: String::new(),
            seed: 42,
            headless: true,
            force_batch_size: None,
            freq_weighted_loss: false,
            resume: false,
            logger: String::new(),
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
    fn compiles_to_5_node_plan() {
        // build_manifest → fullband → train_teacher → adapter → gate.
        let plan = LamquantOracle.compile(args()).unwrap();
        assert_eq!(plan.n_nodes(), 5);
        assert_eq!(plan.n_edges(), 4);
    }

    #[test]
    fn safe_pccp_defaults() {
        let a: Args = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(a.pccp_dry_run);
        assert!(a.pccp_no_promote);
    }
}
