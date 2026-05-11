//! Recipe — `lamquant_encoder`.
//!
//! End-to-end LamQuant encoder pipeline:
//!
//!   build_manifest
//!     → precompute_fullband
//!     → precompute_l3
//!     → (optional) pretrain_mae   [if mae_pretrain=true]
//!     → train_joint
//!     → pccp_gate_encoder
//!
//! train_joint's typed input is `L3Cache`; Manifest + FullbandMemmap
//! are reached by the underlying Python kernel via path
//! conventions in `$lamquant_home`. The L3Cache logical hash
//! cascades cache invalidation through; explicit args
//! (manifest_seed, val_fraction) are also folded into train_joint's
//! Args so upstream config drift invalidates the joint.

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};

pub struct LamquantEncoder;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default = "default_preset")]
    pub preset: String,
    #[serde(default = "default_tier")]
    pub tier: u32,
    #[serde(default = "default_seed")]
    pub seed: u32,
    /// Wire pretrain_mae upstream + auto-set encoder_init for joint.
    #[serde(default)]
    pub mae_pretrain: bool,
    /// MAE-init path rel to lamquant_home. Empty + mae_pretrain
    /// = defaults to `ai_models/student/pretrained_mae.ckpt`.
    #[serde(default)]
    pub encoder_init_rel: String,
    // Manifest knobs.
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,
    #[serde(default = "default_seed64")]
    pub manifest_seed: u64,
    // Joint overrides.
    #[serde(default)]
    pub epochs: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub gan: Option<bool>,
    #[serde(default)]
    pub seizure_head: Option<bool>,
    #[serde(default)]
    pub infinite_lr: bool,
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

fn default_preset() -> String {
    "production".into()
}
fn default_tier() -> u32 {
    3
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
    "(blut/lamquant_encoder recipe run)".into()
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

impl Recipe for LamquantEncoder {
    const NAME: &'static str = "lamquant_encoder";
    const DESCRIPTION: &'static str =
        "Full LamQuant encoder pipeline: build_manifest → precompute_fullband \
         → precompute_l3 → (optional) pretrain_mae → train_joint → pccp_gate_encoder. \
         Safe-by-default PCCP gate.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError> {
        if !matches!(args.preset.as_str(), "fast" | "medium" | "production") {
            return Err(RecipeError::InvalidArgs(format!(
                "preset '{}' must be fast|medium|production",
                args.preset
            )));
        }
        if !(1..=4).contains(&args.tier) {
            return Err(RecipeError::InvalidArgs(format!(
                "tier {} must be in 1..=4",
                args.tier
            )));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let encoder_init_rel = if args.mae_pretrain {
            if args.encoder_init_rel.is_empty() {
                "ai_models/student/pretrained_mae.ckpt".to_string()
            } else {
                args.encoder_init_rel.clone()
            }
        } else {
            String::new()
        };

        let joint_args = crate::stages::lamquant_train_joint::Args {
            lamquant_home: args.lamquant_home.clone(),
            preset: args.preset.clone(),
            tier: args.tier,
            seed: args.seed,
            encoder_init_rel: encoder_init_rel.clone(),
            epochs: args.epochs,
            batch_size: args.batch_size,
            lr: args.lr,
            gan: args.gan,
            seizure_head: args.seizure_head,
            infinite_lr: args.infinite_lr,
            resume: String::new(),
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

        // Data-prep chain (always).
        let after_l3 = Plan::new(Self::NAME, recipe_args_json)
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
                crate::stages::LamquantPrecomputeL3,
                crate::stages::lamquant_precompute_l3::Args {
                    lamquant_home: args.lamquant_home.clone(),
                    input_dir: String::new(),
                    workers: 8,
                },
            );

        let plan = if args.mae_pretrain {
            after_l3
                .then(
                    crate::stages::LamquantPretrainMae,
                    crate::stages::lamquant_pretrain_mae::Args {
                        lamquant_home: args.lamquant_home.clone(),
                        output_rel: encoder_init_rel,
                        epochs: None,
                        lr: None,
                        batch_size: None,
                        mask_ratio: None,
                        patch_size: None,
                        windows_per_epoch: None,
                        max_windows: None,
                        seed: Some(args.seed),
                    },
                )
                // MaeCkpt → train_joint: the kernel reads MAE init
                // from the path in Args.encoder_init_rel, but the
                // typed edge requires L3Cache as input. Re-thread
                // L3Cache by passing it through an identity stage.
                // Simpler approach: chain a small typed bridge.
                // For now, accept that the MAE path = side channel
                // and train_joint's typed Input is L3Cache; the MAE
                // pretrain side-effect is captured by encoder_init_rel
                // flowing through Args canonical (cache key).
                // The Plan compiles linearly: L3Cache → MaeCkpt →
                // can't .then(train_joint) because input mismatch.
                // Insert a passthrough.
                .then(
                    L3RebindFromMae,
                    L3RebindArgs {
                        lamquant_home: args.lamquant_home.clone(),
                    },
                )
                .then(crate::stages::LamquantTrainJoint, joint_args)
                .then(crate::stages::LamquantPccpGateEncoder, gate_args)
                .finish()
        } else {
            after_l3
                .then(crate::stages::LamquantTrainJoint, joint_args)
                .then(crate::stages::LamquantPccpGateEncoder, gate_args)
                .finish()
        };

        Ok(plan)
    }
}

// ── Typed bridge: MaeCkpt → L3Cache (re-read from lamquant_home) ──
//
// train_joint takes L3Cache; pretrain_mae outputs MaeCkpt. To keep
// the linear chain typed, we stitch via a tiny deterministic stage
// that re-reads the L3 cache directory under lamquant_home and
// emits a fresh L3Cache artifact. Pure data-plumbing — no real
// work. Lives next to the recipe since it's recipe-internal.

use async_trait::async_trait;

use crate::artifacts::lamquant::stat_fingerprint_dir;
use crate::artifacts::{L3Cache, MaeCkpt};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::stages::lamquant_helpers::resolve_home;

struct L3RebindFromMae;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
struct L3RebindArgs {
    /// Recipe passes lamquant_home through here so the bridge doesn't
    /// have to derive it from the MAE ckpt path (which was brittle
    /// against layout changes).
    lamquant_home: String,
}

#[async_trait]
impl Stage for L3RebindFromMae {
    const NAME: &'static str = "_l3_rebind_from_mae";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = MaeCkpt;
    type Output = L3Cache;
    type Args = L3RebindArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: MaeCkpt,
        args: &L3RebindArgs,
    ) -> Result<L3Cache, StageError> {
        let home = resolve_home(&args.lamquant_home)?;
        let l3_dir = home.join("ai_models").join("dataset_sim").join("q31_events");
        if !l3_dir.exists() {
            return Err(StageError::BadInput(format!(
                "L3 cache dir not found: {}",
                l3_dir.display()
            )));
        }
        let content_hash = stat_fingerprint_dir(b"lamquant.l3_cache", &l3_dir).map_err(
            |source| StageError::Io {
                path: l3_dir.clone(),
                source,
            },
        )?;
        Ok(L3Cache {
            dir: l3_dir,
            n_windows: 0,
            content_hash,
        })
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: LamquantEncoder::NAME,
    description: LamquantEncoder::DESCRIPTION,
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        LamquantEncoder.compile(args)
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            lamquant_home: String::new(),
            preset: default_preset(),
            tier: 3,
            seed: 42,
            mae_pretrain: false,
            encoder_init_rel: String::new(),
            val_fraction: 0.05,
            manifest_seed: 42,
            epochs: None,
            batch_size: None,
            lr: None,
            gan: None,
            seizure_head: None,
            infinite_lr: false,
            pccp_change_id: default_change_id(),
            pccp_description: default_description(),
            pccp_author: default_author(),
            pccp_change_class: default_change_class(),
            pccp_dry_run: true,
            pccp_no_promote: true,
        }
    }

    #[test]
    fn compiles_without_mae() {
        // build_manifest → precompute_fullband → precompute_l3 →
        // train_joint → pccp_gate_encoder = 5 nodes.
        let plan = LamquantEncoder.compile(args()).unwrap();
        assert_eq!(plan.n_nodes(), 5);
        assert_eq!(plan.n_edges(), 4);
    }

    #[test]
    fn compiles_with_mae() {
        let mut a = args();
        a.mae_pretrain = true;
        // build_manifest → fullband → l3 → pretrain_mae →
        // _l3_rebind_from_mae → train_joint → gate = 7 nodes.
        let plan = LamquantEncoder.compile(a).unwrap();
        assert_eq!(plan.n_nodes(), 7);
        assert_eq!(plan.n_edges(), 6);
    }

    #[test]
    fn rejects_invalid_tier() {
        let mut a = args();
        a.tier = 99;
        assert!(matches!(LamquantEncoder.compile(a), Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_invalid_preset() {
        let mut a = args();
        a.preset = "garbage".into();
        assert!(matches!(LamquantEncoder.compile(a), Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn safe_pccp_defaults() {
        let a: Args = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(a.pccp_dry_run);
        assert!(a.pccp_no_promote);
    }
}

