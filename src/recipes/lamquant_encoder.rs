//! Recipe — `lamquant_encoder`.
//!
//! End-to-end LamQuant encoder pipeline. Per ADR 0017 (BLUT-canonical
//! + LMA-direct), the chain is:
//!
//! ```text
//! lamquant_convert_lma              () → LmaCorpus
//!   → (optional) pretrain_mae       LmaCorpus → MaeCkpt
//!   → (optional) _corpus_rebind_from_mae  MaeCkpt → LmaCorpus  (bridge)
//!   → lamquant_train_joint          LmaCorpus → JointCkpt
//!   → lamquant_pccp_gate_encoder    JointCkpt → PccpVerdict
//! ```
//!
//! Replaces the pre-ADR `build_manifest → precompute_fullband →
//! precompute_l3 → ...` chain. The precompute stages stay registered
//! and callable as standalone helpers for legacy tooling, but the
//! encoder pipeline produces and consumes LmaCorpus end-to-end now.
//!
//! `_corpus_rebind_from_mae` is the typed bridge that lets the linear
//! `Plan::then()` chain stitch `pretrain_mae`'s `MaeCkpt` output back
//! into a fresh `LmaCorpus` so `train_joint` can consume it. Pure
//! data-plumbing — the bridge re-emits the corpus that
//! `lamquant_convert_lma` produced, identified by content hash so
//! BLUT's cache key cascade still works.

use std::path::PathBuf;

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
    /// encoder training cannot operate without a deterministic split.
    pub split_manifest: String,

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
    type Backend = crate::backends::LamquantBackend;
    const NAME: &'static str = "lamquant_encoder";
    const DESCRIPTION: &'static str = "LamQuant encoder pipeline: convert_lma → (optional pretrain_mae \
         → bridge) → train_joint → pccp_gate_encoder. LMA-direct per \
         ADR 0017. Safe-by-default PCCP gate.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
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
        if let Some(lr) = args.lr {
            if !(lr > 0.0 && lr.is_finite()) {
                return Err(RecipeError::InvalidArgs(format!(
                    "lr must be positive finite; got {lr}"
                )));
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
                "split_manifest is required (LMA-direct encoder training \
                 cannot operate without a subject-grouped split)"
                    .into(),
            ));
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

        let after_convert = Plan::new(Self::NAME, recipe_args_json)
            .start(crate::stages::LamquantConvertLma, convert_args);

        let plan = if args.mae_pretrain {
            after_convert
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
                        lma_root: args.lma_output_dir.display().to_string(),
                        split_manifest: args.split_manifest.clone(),
                    },
                )
                .then(
                    CorpusRebindFromMae,
                    CorpusRebindArgs {
                        lma_output_dir: args.lma_output_dir.clone(),
                    },
                )
                .then(crate::stages::LamquantTrainJoint, joint_args)
                .then(crate::stages::LamquantPccpGateEncoder, gate_args)
                .finish()
        } else {
            after_convert
                .then(crate::stages::LamquantTrainJoint, joint_args)
                .then(crate::stages::LamquantPccpGateEncoder, gate_args)
                .finish()
        };

        Ok(plan)
    }
}

// ── Typed bridge: MaeCkpt → LmaCorpus (re-emit the converted corpus) ─
//
// `train_joint` takes `LmaCorpus`; `pretrain_mae` outputs `MaeCkpt`.
// To keep the linear chain typed, this tiny deterministic stage
// re-emits the `LmaCorpus` that `lamquant_convert_lma` produced.
// Pure data plumbing — no real work. Lives next to the recipe since
// it's recipe-internal.

use async_trait::async_trait;

use crate::artifacts::lamquant::stat_fingerprint_dir;
use crate::artifacts::{LmaCorpus, MaeCkpt};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

struct CorpusRebindFromMae;

impl crate::framework::Compatible<crate::backends::LamquantBackend> for CorpusRebindFromMae {}

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
struct CorpusRebindArgs {
    /// LMA corpus root the upstream `convert_lma` wrote to. Passed
    /// through here so the bridge doesn't have to re-derive it.
    lma_output_dir: PathBuf,
}

#[async_trait]
impl Stage for CorpusRebindFromMae {
    const NAME: &'static str = "_corpus_rebind_from_mae";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = MaeCkpt;
    type Output = LmaCorpus;
    type Args = CorpusRebindArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: MaeCkpt,
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
        // Count `.lma` entries directly under root. Cheap (one
        // shallow read_dir) and keeps `n_archives` honest for any
        // downstream stage that may key on it.
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
    name: LamquantEncoder::NAME,
    description: LamquantEncoder::DESCRIPTION,
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
        LamquantEncoder.compile(args).map(|p| p.into_compiled())
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
            lml_root: String::new(),
            labels_dir_rel: String::new(),
            lma_output_dir: PathBuf::from("/tmp/lma"),
            convert_workers: None,
            convert_limit: None,
            split_manifest: "/tmp/split.json".into(),
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
        // convert_lma → train_joint → pccp_gate_encoder = 3 nodes.
        let plan = LamquantEncoder.compile(args()).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
    }

    #[test]
    fn compiles_with_mae() {
        let mut a = args();
        a.mae_pretrain = true;
        // convert_lma → pretrain_mae → _corpus_rebind_from_mae →
        // train_joint → gate = 5 nodes.
        let plan = LamquantEncoder.compile(a).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 5);
        assert_eq!(plan.n_edges(), 4);
    }

    #[test]
    fn rejects_invalid_tier() {
        let mut a = args();
        a.tier = 99;
        assert!(matches!(
            LamquantEncoder.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_invalid_preset() {
        let mut a = args();
        a.preset = "garbage".into();
        assert!(matches!(
            LamquantEncoder.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_empty_output_dir() {
        let mut a = args();
        a.lma_output_dir = PathBuf::new();
        assert!(matches!(
            LamquantEncoder.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_empty_split_manifest() {
        let mut a = args();
        a.split_manifest = String::new();
        assert!(matches!(
            LamquantEncoder.compile(a),
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

    // ── CorpusRebindFromMae bridge tests ──────────────────────
    //
    // The bridge is the only point where the typed encoder chain
    // re-emits an LmaCorpus after the (optional) MAE pretrain. A
    // silent cache-key bug here would propagate downstream into
    // train_joint without an obvious failure mode, so the bridge
    // earns explicit coverage.

    use crate::framework::artifact::ContentHash;

    fn dummy_mae(td: &std::path::Path) -> MaeCkpt {
        let ckpt = td.join("mae.ckpt");
        std::fs::write(&ckpt, b"mae").unwrap();
        MaeCkpt {
            path: ckpt,
            content_hash: ContentHash::of_bytes(b"mae"),
            base_arch: "ternary_mobilenet_v5_subband".into(),
            final_loss: 0.0,
        }
    }

    fn bridge_ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn bridge_rejects_missing_corpus_dir() {
        let td = tempfile::tempdir().unwrap();
        let r = CorpusRebindFromMae
            .run(
                &bridge_ctx(td.path()),
                dummy_mae(td.path()),
                &CorpusRebindArgs {
                    lma_output_dir: td.path().join("__not_a_dir__"),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn bridge_counts_lma_archives() {
        let td = tempfile::tempdir().unwrap();
        let corpus = td.path().join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        // Two .lma + one .txt — only .lma should count.
        std::fs::write(corpus.join("a.lma"), b"a").unwrap();
        std::fs::write(corpus.join("b.lma"), b"b").unwrap();
        std::fs::write(corpus.join("notes.txt"), b"x").unwrap();
        let out = CorpusRebindFromMae
            .run(
                &bridge_ctx(td.path()),
                dummy_mae(td.path()),
                &CorpusRebindArgs {
                    lma_output_dir: corpus.clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(out.root, corpus);
        assert_eq!(out.n_archives, 2);
        // content_hash must be derivable, not zero/sentinel.
        assert_ne!(out.content_hash, ContentHash::of_bytes(b""));
    }
}
