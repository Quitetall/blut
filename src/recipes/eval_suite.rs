//! Recipe — `eval_suite`.
//!
//! Branch-merge demo + the first user-visible parallel pipeline:
//!
//!   materialize_for_eval
//!     → fork3(eval_loss, eval_lm_harness, eval_judge)
//!     → merge3(merge_reports)
//!
//! All three eval stages consume `(HfCheckpoint, DatasetJsonl)`
//! and produce `EvalReport`. The typed Plan builder's `fork3`
//! enforces the shared-input + tuple-output shape at compile time;
//! `merge3` enforces 3-tuple input shape on `merge_reports`.

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::{
    eval_judge::{Args as JudgeArgs, EvalJudge},
    eval_lm_harness::{Args as HarnessArgs, EvalLmHarness},
    eval_loss::{Args as LossArgs, EvalLoss},
    materialize_for_eval::{Args as MatArgs, MaterializeForEval},
    merge_reports::{Args as MergeArgs, MergeReports},
};

pub struct EvalSuite;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub model_path: std::path::PathBuf,
    #[serde(default)]
    pub base_model: String,
    pub dataset_path: std::path::PathBuf,
    /// Tasks fed to lm-eval-harness. Required (no default — the
    /// recipe is opinionated about wanting >0 evals).
    pub lm_harness_tasks: Vec<String>,
    #[serde(default = "default_num_fewshot")]
    pub lm_harness_fewshot: u32,
    #[serde(default = "default_judge_model")]
    pub judge_model: String,
    #[serde(default = "default_judge_samples")]
    pub judge_samples: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_max_seq")]
    pub max_seq: u32,
}
fn default_num_fewshot() -> u32 {
    0
}
fn default_judge_model() -> String {
    "claude-opus-4-7".into()
}
fn default_judge_samples() -> u32 {
    20
}
fn default_batch_size() -> u32 {
    1
}
fn default_max_seq() -> u32 {
    4096
}

impl Recipe for EvalSuite {
    type Backend = crate::backends::LamuTrainerBackend;
    const NAME: &'static str = "eval_suite";
    const DESCRIPTION: &'static str =
        "Three-way evaluation: cross-entropy + lm-eval-harness benchmarks \
         + judge-model scoring. Produces a combined EvalReport with all \
         three sub-reports + a flat summary.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        if args.lm_harness_tasks.is_empty() {
            return Err(RecipeError::InvalidArgs(
                "lm_harness_tasks must be non-empty".into(),
            ));
        }
        if args.judge_model.is_empty() {
            return Err(RecipeError::InvalidArgs("judge_model must be non-empty".into()));
        }
        // R23: numeric arg ranges.
        if args.judge_samples == 0 {
            return Err(RecipeError::InvalidArgs("judge_samples must be > 0".into()));
        }
        if args.batch_size == 0 || args.max_seq == 0 {
            return Err(RecipeError::InvalidArgs(
                "batch_size + max_seq must be > 0".into(),
            ));
        }
        // R30: mandatory paths must be non-empty AND traversal-free.
        for (label, p) in [
            ("model_path", &args.model_path),
            ("dataset_path", &args.dataset_path),
        ] {
            if p.as_os_str().is_empty() {
                return Err(RecipeError::InvalidArgs(format!(
                    "{label} must be non-empty"
                )));
            }
            if p.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
                return Err(RecipeError::InvalidArgs(format!(
                    "{label} '{}' contains '..' — refusing",
                    p.display()
                )));
            }
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                MaterializeForEval,
                MatArgs {
                    model_path: args.model_path.clone(),
                    base_model: args.base_model.clone(),
                    dataset_path: args.dataset_path.clone(),
                },
            )
            .fork3(
                EvalLoss,
                LossArgs {
                    batch_size: args.batch_size,
                    max_seq: args.max_seq,
                },
                EvalLmHarness,
                HarnessArgs {
                    tasks: args.lm_harness_tasks.clone(),
                    num_fewshot: args.lm_harness_fewshot,
                },
                EvalJudge,
                JudgeArgs {
                    judge_model: args.judge_model.clone(),
                    prompts: vec![],
                    n_samples: args.judge_samples,
                },
            )
            .merge3(MergeReports, MergeArgs::default())
            .finish();

        Ok(plan)
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: EvalSuite::NAME,
    description: EvalSuite::DESCRIPTION,
    backend_id: <crate::backends::LamuTrainerBackend as crate::backends::TrainingBackend>::ID,
    category: crate::recipes::recipe::RecipeCategory::Eval,
    input_kinds: &["checkpoint.hf"],
    output_kind: "eval.report",
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        EvalSuite.compile(args).map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            model_path: "/tmp/m".into(),
            base_model: "Qwen/Qwen3-7B".into(),
            dataset_path: "/tmp/e.jsonl".into(),
            lm_harness_tasks: vec!["hellaswag".into(), "arc_easy".into()],
            lm_harness_fewshot: 0,
            judge_model: default_judge_model(),
            judge_samples: 5,
            batch_size: 1,
            max_seq: 4096,
        }
    }

    #[test]
    fn compiles_to_branch_plan() {
        let plan = EvalSuite.compile(args()).unwrap().into_compiled();
        // 1 materializer + 3 forked evaluators + 1 merge = 5 nodes.
        assert_eq!(plan.n_nodes(), 5);
        // Edges: materializer→3 forks (3), 3 forks→merge (3) = 6.
        assert_eq!(plan.n_edges(), 6);
    }

    #[test]
    fn rejects_empty_lm_harness_tasks() {
        let mut a = args();
        a.lm_harness_tasks.clear();
        let r = EvalSuite.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_empty_judge_model() {
        let mut a = args();
        a.judge_model.clear();
        let r = EvalSuite.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }
}
