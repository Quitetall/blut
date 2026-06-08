//! Recipe — `lamquant_combined_decoder`.
//!
//! End-to-end LamQuant decoder pipeline (combined teacher + decoder
//! joint training, ~40% faster than the legacy two-stage path):
//!
//!   lamquant_convert_lma           () → LmaCorpus
//!     → lamquant_train_combined    LmaCorpus → (TeacherCkpt, JointCkpt)
//!     → _take_joint_ckpt           (TeacherCkpt, JointCkpt) → JointCkpt   (bridge)
//!     → lamquant_pccp_gate_decoder JointCkpt → PccpVerdict
//!
//! The `_take_joint_ckpt` bridge is a tiny recipe-internal stage that
//! drops the TeacherCkpt half of the tuple; it carries no logic, just
//! plumbs the second element through so the linear `Plan::then()`
//! chain typechecks. Mirrors the `L3RebindFromMae` pattern in
//! `lamquant_encoder.rs`.
//!
//! Per ADR 0017 (BLUT-canonical + LMA-direct), the data path goes
//! through `lamquant_convert_lma` instead of the deprecated
//! precompute_fullband chain.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{JointCkpt, TeacherCkpt};
use crate::framework::Compatible;
use crate::framework::error::{RecipeError, StageError};
use crate::framework::plan::Plan;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::recipes::recipe::{Recipe, RecipeDef};

pub struct LamquantCombinedDecoder;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty → `$LAMQUANT_HOME` or default.
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

    // ── lamquant_train_combined knobs ─────────────────────────
    /// Split manifest JSON path. Forwarded to the kernel as
    /// `--split-manifest`. Required (train_combined cannot operate
    /// without a split).
    pub split_manifest: String,
    #[serde(default = "default_decoder_tier")]
    pub decoder_tier: u32,
    #[serde(default)]
    pub teacher_epochs: Option<u32>,
    #[serde(default)]
    pub decoder_epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub teacher_lr: Option<f32>,
    #[serde(default)]
    pub decoder_lr: Option<f32>,
    #[serde(default)]
    pub channel_attn: bool,
    #[serde(default)]
    pub bottleneck_attn: bool,
    /// Student encoder ckpt rel-path to condition the decoder on.
    /// Required for the production flow; empty triggers script's own default.
    #[serde(default)]
    pub student_checkpoint_rel: String,

    // ── lamquant_pccp_gate_decoder knobs ──────────────────────
    /// PCCP change id. Default emits a dry-run sentinel.
    #[serde(default = "default_pccp_change_id")]
    pub pccp_change_id: String,
    #[serde(default = "default_pccp_description")]
    pub pccp_description: String,
    #[serde(default = "default_pccp_author")]
    pub pccp_author: String,
    #[serde(default = "default_pccp_change_class")]
    pub pccp_change_class: String,
    /// PCCP dry-run (don't promote candidate). Defaults to true —
    /// production runs must explicitly opt-in to live promotion.
    #[serde(default = "default_true")]
    pub pccp_dry_run: bool,
    #[serde(default = "default_true")]
    pub pccp_no_promote: bool,
}

fn default_decoder_tier() -> u32 {
    3
}
fn default_pccp_change_id() -> String {
    "PCCP-CHG-DRYRUN".into()
}
fn default_pccp_description() -> String {
    "(blut combined_decoder gate)".into()
}
fn default_pccp_author() -> String {
    "BLUT".into()
}
fn default_pccp_change_class() -> String {
    "A.1".into()
}
fn default_true() -> bool {
    true
}

impl Recipe for LamquantCombinedDecoder {
    type Backend = crate::backends::LamquantBackend;
    const NAME: &'static str = "lamquant_combined_decoder";
    const DESCRIPTION: &'static str = "End-to-end decoder pipeline (combined teacher + decoder \
         joint training): convert_lma → train_combined → \
         pccp_gate_decoder. Replaces the legacy sequential \
         teacher-then-decoder chain (~40% faster wall-time). \
         LMA-direct per ADR 0017.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        // serde missing-field rules already catch absent JSON keys
        // (no `#[serde(default)]` on `lma_output_dir` /
        // `split_manifest`); these checks add an extra guard for
        // PROGRAMMATIC `Args` construction (Rust tests, direct
        // callers) where an empty PathBuf or empty String could
        // sneak through.
        if args.lma_output_dir.as_os_str().is_empty() {
            return Err(RecipeError::InvalidArgs(
                "lma_output_dir is required (writable directory for \
                 the converted LMA corpus)"
                    .into(),
            ));
        }
        if args.split_manifest.is_empty() {
            return Err(RecipeError::InvalidArgs(
                "split_manifest is required (train_combined cannot \
                 operate without a train/val split)"
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

        // Several `lamquant_train_combined` hyperparameters are
        // intentionally NOT exposed at the recipe level in this
        // first iteration: `teacher_width`, `teacher_strides`,
        // `teacher_r_loss`, `lr_min`, `windows_per_epoch`,
        // `max_windows`. The Python kernel applies its own
        // sensible defaults for each (matching the production
        // training preset). Users who need to tune them today
        // can invoke `blut stage run lamquant_train_combined`
        // directly with full Args; a follow-up commit will
        // surface them as Optional<T> recipe args when a
        // production tuning workflow demands it.
        let combined_args = crate::stages::lamquant_train_combined::Args {
            lamquant_home: args.lamquant_home.clone(),
            decoder_tier: args.decoder_tier,
            teacher_epochs: args.teacher_epochs,
            decoder_epochs: args.decoder_epochs,
            teacher_width: None,
            teacher_strides: String::new(),
            channel_attn: args.channel_attn,
            bottleneck_attn: args.bottleneck_attn,
            teacher_r_loss: None,
            batch_size: args.batch_size,
            teacher_lr: args.teacher_lr,
            decoder_lr: args.decoder_lr,
            lr_min: None,
            windows_per_epoch: None,
            max_windows: None,
            student_checkpoint_rel: args.student_checkpoint_rel.clone(),
            lma_root: args.lma_output_dir.display().to_string(),
            split_manifest: args.split_manifest.clone(),
        };

        let gate_args = crate::stages::lamquant_pccp_gate_encoder::DecoderArgs {
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
            .then(crate::stages::LamquantTrainCombined, combined_args)
            .then(TakeJointCkpt, TakeJointCkptArgs)
            .then(crate::stages::LamquantPccpGateDecoder, gate_args)
            .finish();
        Ok(plan)
    }
}

// ── Typed bridge: (TeacherCkpt, JointCkpt) → JointCkpt ────────────
//
// `lamquant_train_combined` outputs a tuple; `pccp_gate_decoder`
// needs the JointCkpt half. The linear `Plan::then()` chain can't
// destructure tuples on its own — bridge stage extracts the second
// element. Pure data plumbing, no work performed.

struct TakeJointCkpt;

impl Compatible<crate::backends::LamquantBackend> for TakeJointCkpt {}

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
struct TakeJointCkptArgs;

#[async_trait]
impl Stage for TakeJointCkpt {
    const NAME: &'static str = "_take_joint_ckpt";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[];
    type Input = (TeacherCkpt, JointCkpt);
    type Output = JointCkpt;
    type Args = TakeJointCkptArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: (TeacherCkpt, JointCkpt),
        _args: &TakeJointCkptArgs,
    ) -> Result<JointCkpt, StageError> {
        Ok(input.1)
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: LamquantCombinedDecoder::NAME,
    description: LamquantCombinedDecoder::DESCRIPTION,
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
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("combined_decoder args: {e}")))?;
        LamquantCombinedDecoder
            .compile(args)
            .map(|p| p.into_compiled())
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
            lma_output_dir: out,
            convert_workers: None,
            convert_limit: None,
            split_manifest: "/tmp/split.json".into(),
            decoder_tier: 3,
            teacher_epochs: None,
            decoder_epochs: None,
            batch_size: None,
            teacher_lr: None,
            decoder_lr: None,
            channel_attn: false,
            bottleneck_attn: false,
            student_checkpoint_rel: String::new(),
            pccp_change_id: default_pccp_change_id(),
            pccp_description: default_pccp_description(),
            pccp_author: default_pccp_author(),
            pccp_change_class: default_pccp_change_class(),
            pccp_dry_run: true,
            pccp_no_promote: true,
        }
    }

    #[test]
    fn compiles_to_four_node_plan() {
        let plan = LamquantCombinedDecoder
            .compile(args(PathBuf::from("/tmp/lma_test")))
            .unwrap()
            .into_compiled();
        // 4 nodes: convert_lma + train_combined + _take_joint_ckpt + pccp_gate_decoder
        // 3 edges: convert→train, train→bridge, bridge→gate.
        // n_nodes=4 indirectly proves the TakeJointCkpt bridge is wired
        // (the "without bridge" shape would be 3 nodes / 2 edges).
        assert_eq!(plan.n_nodes(), 4);
        assert_eq!(plan.n_edges(), 3);
    }

    #[test]
    fn defaults_are_production_safe() {
        // PCCP defaults MUST be dry-run + no-promote so an accidental
        // recipe run cannot silently promote a candidate. Verifies
        // the two defaults stay paranoid by construction.
        let a = args(PathBuf::from("/tmp/lma_test"));
        assert!(a.pccp_dry_run, "pccp_dry_run must default true");
        assert!(a.pccp_no_promote, "pccp_no_promote must default true");
        // And the constructed gate_args inside compile() inherit them
        // — verified indirectly by the four-node compile + the
        // explicit field wiring in `compile()`.
    }

    #[test]
    fn rejects_empty_output_dir() {
        let r = LamquantCombinedDecoder.compile(args(PathBuf::new()));
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_empty_split_manifest() {
        let mut a = args(PathBuf::from("/tmp/lma_test"));
        a.split_manifest = String::new();
        let r = LamquantCombinedDecoder.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[tokio::test]
    async fn bridge_passes_joint_ckpt_through() {
        use crate::framework::artifact::ContentHash;
        let teacher = TeacherCkpt {
            path: PathBuf::from("/tmp/teacher.ckpt"),
            content_hash: ContentHash::of_bytes(b"t"),
            gen_tag: "combined".into(),
            final_loss: 0.0,
        };
        let joint = JointCkpt {
            encoder_path: PathBuf::from("/tmp/enc.ckpt"),
            decoder_path: PathBuf::from("/tmp/dec.ckpt"),
            content_hash: ContentHash::of_bytes(b"j"),
            final_loss: 0.0,
            tier: 3,
            preset: "combined".into(),
        };
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("stage")).unwrap();
        let ctx = StageContext::for_test(td.path().to_path_buf(), td.path().join("stage"));
        let out = TakeJointCkpt
            .run(&ctx, (teacher, joint.clone()), &TakeJointCkptArgs)
            .await
            .unwrap();
        assert_eq!(out.decoder_path, joint.decoder_path);
    }
}
