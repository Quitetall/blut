//! Recipe — `lamquant_data_prep`.
//!
//! Single-stage recipe: pack the per-corpus LML tree into per-recording
//! `.lma` archives (`scripts/bulk_lml_to_lma.py` via the
//! `lamquant_convert_lma` stage). Output: `LmaCorpus`.
//!
//! Followup pipeline steps (build_manifest, generate_snn_labels)
//! currently run as separate `blut stage run` invocations because
//! they take graph-input or tuple-input shapes the linear `.then`
//! chain can't express without typed bridge stages (see the
//! `L3RebindFromMae` pattern in `lamquant_encoder`). Future iterations
//! add those bridges + chain through.
//!
//! This recipe ships day-one to give BLUT a one-call data-prep
//! entrypoint that matches cockpit's `[d]` "full setup" button.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};

pub struct LamquantDataPrep;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty → `$LAMQUANT_HOME` or default.
    #[serde(default)]
    pub lamquant_home: String,
    /// LML source root. Empty = packer default (built-in /mnt/4tb path).
    #[serde(default)]
    pub lml_root: String,
    /// Labels NPZ dir relative to lamquant_home. Empty = packer default
    /// (`ai_models/snn/labels`).
    #[serde(default)]
    pub labels_dir_rel: String,
    /// Output LMA corpus directory (must be writable). Required.
    pub output_dir: PathBuf,
    /// Worker process count. None = packer default (cpu_count / 3).
    #[serde(default)]
    pub workers: Option<u32>,
    /// Cap iteration to first N stems (smoke runs).
    #[serde(default)]
    pub limit: Option<u32>,
    /// Pass `--keep-sources` so loose source files survive the pack.
    #[serde(default)]
    pub keep_sources: bool,
    /// Pass `--dry-run`: plan + report only, no LMA writes, no deletes.
    #[serde(default)]
    pub dry_run: bool,
}

impl Recipe for LamquantDataPrep {
    type Backend = crate::backends::LamquantBackend;
    const NAME: &'static str = "lamquant_data_prep";
    const DESCRIPTION: &'static str = "Pack the LML tree into per-recording .lma archives via \
         bulk_lml_to_lma.py. One-shot data-prep entrypoint for the \
         LMA-direct training pipeline (ADR 0017).";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        if args.output_dir.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "output_dir is required (writable directory for .lma archives)".into(),
            ));
        }
        if let Some(w) = args.workers {
            if w == 0 {
                return Err(RecipeError::InvalidArgs("workers must be > 0".into()));
            }
        }
        if let Some(l) = args.limit {
            if l == 0 {
                return Err(RecipeError::InvalidArgs(
                    "limit must be > 0 (use None to process all stems)".into(),
                ));
            }
        }
        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;
        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                crate::stages::LamquantConvertLma,
                crate::stages::lamquant_convert_lma::Args {
                    lamquant_home: args.lamquant_home,
                    lml_root: args.lml_root,
                    labels_dir_rel: args.labels_dir_rel,
                    output_dir: args.output_dir,
                    workers: args.workers,
                    limit: args.limit,
                    keep_sources: args.keep_sources,
                    dry_run: args.dry_run,
                },
            )
            .finish();
        Ok(plan)
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: LamquantDataPrep::NAME,
    description: LamquantDataPrep::DESCRIPTION,
    backend_id: <crate::backends::LamquantBackend as crate::backends::TrainingBackend>::ID,
    category: crate::recipes::recipe::RecipeCategory::DataPrep,
    input_kinds: &[],
    output_kind: "lamquant.lma_corpus",
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw_args| {
        let args: Args = serde_json::from_value(raw_args)
            .map_err(|e| RecipeError::InvalidArgs(format!("data_prep args: {e}")))?;
        LamquantDataPrep.compile(args).map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args(out: PathBuf) -> Args {
        Args {
            lamquant_home: String::new(),
            lml_root: String::new(),
            labels_dir_rel: String::new(),
            output_dir: out,
            workers: None,
            limit: None,
            keep_sources: false,
            dry_run: true,
        }
    }

    #[test]
    fn compiles_to_single_node_plan() {
        let plan = LamquantDataPrep
            .compile(args(PathBuf::from("/tmp/lma_test")))
            .unwrap()
            .into_compiled();
        assert_eq!(plan.n_nodes(), 1);
        assert_eq!(plan.n_edges(), 0);
    }

    #[test]
    fn rejects_empty_output_dir() {
        let r = LamquantDataPrep.compile(args(PathBuf::new()));
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_zero_workers() {
        let mut a = args(PathBuf::from("/tmp/lma_test"));
        a.workers = Some(0);
        let r = LamquantDataPrep.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_zero_limit() {
        let mut a = args(PathBuf::from("/tmp/lma_test"));
        a.limit = Some(0);
        let r = LamquantDataPrep.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }
}
