//! Recipe — `lamquant_joint_codec`.
//!
//! Standalone `LmaCorpus → JointCkpt → PccpVerdict`. The INNER
//! training loop: train the joint codec (encoder + decoder) on an
//! ALREADY-CONVERTED LMA corpus, with no re-pack step.
//!
//! Use [`lamquant_encoder`](crate::recipes::lamquant_encoder) for the
//! full `convert → (optional MAE) → train → gate` pipeline; use THIS
//! recipe to sweep preset / tier / seed / resume / encoder_init
//! against a FIXED corpus without re-running `convert_lma` each time.
//!
//! Constraint that forces the shape: `Plan::start<S>` requires
//! `S::Input = ()` (plan.rs), but `LamquantTrainJoint::Input =
//! LmaCorpus` (lamquant_train_joint.rs) — so a plan cannot *start*
//! with `train_joint`. The fix mirrors the in-file
//! `CorpusRebindFromMae` precedent in `lamquant_encoder`: a
//! recipe-internal graph-input source stage `LmaCorpusSource`
//! (`Input = ()`, `Output = LmaCorpus`, deterministic) that validates
//! the corpus dir + fingerprints it, kept IN THIS FILE so `src/stages/`
//! stays untouched.
//!
//! Chain: `LmaCorpusSource → LamquantTrainJoint →
//! LamquantPccpGateEncoder`. PCCP is safe-by-default (dry-run +
//! no-promote). `output_kind = "lamquant.pccp_verdict"` +
//! `input_kinds = []` make this a clean swap-candidate of
//! `lamquant_encoder`.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::LmaCorpus;
use crate::artifacts::lamquant::stat_fingerprint_dir;
use crate::framework::Compatible;
use crate::framework::error::{RecipeError, StageError};
use crate::framework::plan::Plan;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::recipes::recipe::{Recipe, RecipeCategory, RecipeDef};
use crate::stages::{
    LamquantPccpGateEncoder, LamquantTrainJoint, lamquant_pccp_gate_encoder, lamquant_train_joint,
};

pub struct LamquantJointCodec;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// LamQuant repo root. Empty = env / default.
    #[serde(default)]
    pub lamquant_home: String,
    /// REQUIRED: root dir of an existing LMA corpus (output of
    /// `lamquant_convert_lma` / `lml archive`).
    pub lma_root: String,
    /// REQUIRED: subject-grouped split manifest JSON path.
    pub split_manifest: String,

    /// `--config` preset (fast | medium | production).
    #[serde(default = "default_preset")]
    pub preset: String,
    /// Decoder tier (1..=4).
    #[serde(default = "default_tier")]
    pub tier: u32,
    #[serde(default = "default_seed")]
    pub seed: u32,
    /// Encoder init (MAE / SFT) path relative to lamquant_home. Empty
    /// = train the encoder from scratch.
    #[serde(default)]
    pub encoder_init_rel: String,
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
    /// `--resume <path>` (or "auto"). Empty = fresh.
    #[serde(default)]
    pub resume: String,

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
    "(blut/lamquant_joint_codec recipe run)".into()
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

impl Recipe for LamquantJointCodec {
    type Backend = crate::backends::LamquantBackend;
    const NAME: &'static str = "lamquant_joint_codec";
    const DESCRIPTION: &'static str = "Train the joint EEG codec (encoder + decoder) on an EXISTING \
         LMA corpus → JointCkpt → PCCP gate. Standalone inner loop (no \
         convert step) for preset/tier/seed/resume sweeps. \
         Safe-by-default PCCP.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        // Arg-range validation at the recipe boundary (mirrors
        // lamquant_encoder).
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
        if args.lma_root.is_empty() {
            return Err(RecipeError::InvalidArgs(
                "lma_root is required (path to an existing LMA corpus)".into(),
            ));
        }
        if args.split_manifest.is_empty() {
            return Err(RecipeError::InvalidArgs(
                "split_manifest is required (LMA-direct joint training \
                 cannot operate without a subject-grouped split)"
                    .into(),
            ));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let joint_args = lamquant_train_joint::Args {
            lamquant_home: args.lamquant_home.clone(),
            preset: args.preset.clone(),
            tier: args.tier,
            seed: args.seed,
            encoder_init_rel: args.encoder_init_rel.clone(),
            epochs: args.epochs,
            batch_size: args.batch_size,
            lr: args.lr,
            gan: args.gan,
            seizure_head: args.seizure_head,
            infinite_lr: args.infinite_lr,
            resume: args.resume.clone(),
            lma_root: args.lma_root.clone(),
            split_manifest: args.split_manifest.clone(),
        };
        let gate_args = lamquant_pccp_gate_encoder::Args {
            lamquant_home: args.lamquant_home.clone(),
            change_id: args.pccp_change_id.clone(),
            description: args.pccp_description.clone(),
            author: args.pccp_author.clone(),
            change_class: args.pccp_change_class.clone(),
            dry_run: args.pccp_dry_run,
            no_promote: args.pccp_no_promote,
        };

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                LmaCorpusSource,
                LmaCorpusSourceArgs {
                    lma_root: PathBuf::from(&args.lma_root),
                },
            )
            .then(LamquantTrainJoint, joint_args)
            .then(LamquantPccpGateEncoder, gate_args)
            .finish();
        Ok(plan)
    }
}

// ── Recipe-internal graph-input source stage: () → LmaCorpus ────────
//
// `train_joint` consumes `LmaCorpus` but `Plan::start` requires the
// first stage's `Input = ()`. This deterministic source stage adapts
// an existing on-disk corpus into the typed `LmaCorpus` artifact so
// the chain can start. Lives in-file (like `CorpusRebindFromMae` in
// `lamquant_encoder`) to keep `src/stages/` out of scope; it does no
// real work beyond validating + fingerprinting the directory.

struct LmaCorpusSource;

impl Compatible<crate::backends::LamquantBackend> for LmaCorpusSource {}

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
struct LmaCorpusSourceArgs {
    /// Root of the existing LMA corpus to feed into `train_joint`.
    lma_root: PathBuf,
}

#[async_trait]
impl Stage for LmaCorpusSource {
    const NAME: &'static str = "_lma_corpus_source";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    const DETERMINISTIC: bool = true;
    type Input = ();
    type Output = LmaCorpus;
    type Args = LmaCorpusSourceArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &LmaCorpusSourceArgs,
    ) -> Result<LmaCorpus, StageError> {
        let root = &args.lma_root;
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
        // Count `.lma` entries directly under root. Cheap (one shallow
        // read_dir) and keeps `n_archives` honest for any downstream
        // stage that may key on it.
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
    name: LamquantJointCodec::NAME,
    description: LamquantJointCodec::DESCRIPTION,
    backend_id: <crate::backends::LamquantBackend as crate::backends::TrainingBackend>::ID,
    // Single-paradigm train recipe (vs lamquant_encoder's Pipeline
    // category) — but identical input_kinds/output_kind so it remains
    // a swap_candidate of the full pipeline.
    category: RecipeCategory::Train,
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
        LamquantJointCodec.compile(args).map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;

    fn args() -> Args {
        Args {
            lamquant_home: String::new(),
            lma_root: "/tmp/lma".into(),
            split_manifest: "/tmp/split.json".into(),
            preset: default_preset(),
            tier: 3,
            seed: 42,
            encoder_init_rel: String::new(),
            epochs: None,
            batch_size: None,
            lr: None,
            gan: None,
            seizure_head: None,
            infinite_lr: false,
            resume: String::new(),
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
        // _lma_corpus_source → train_joint → pccp_gate_encoder.
        let plan = LamquantJointCodec.compile(args()).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn rejects_empty_lma_root() {
        let mut a = args();
        a.lma_root = String::new();
        assert!(matches!(
            LamquantJointCodec.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_empty_split_manifest() {
        let mut a = args();
        a.split_manifest = String::new();
        assert!(matches!(
            LamquantJointCodec.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_invalid_tier() {
        let mut a = args();
        a.tier = 99;
        assert!(matches!(
            LamquantJointCodec.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn rejects_invalid_preset() {
        let mut a = args();
        a.preset = "garbage".into();
        assert!(matches!(
            LamquantJointCodec.compile(a),
            Err(RecipeError::InvalidArgs(_))
        ));
    }

    #[test]
    fn safe_pccp_defaults() {
        let raw = serde_json::json!({
            "lma_root": "/tmp/lma",
            "split_manifest": "/tmp/split.json",
        });
        let a: Args = serde_json::from_value(raw).unwrap();
        assert!(a.pccp_dry_run);
        assert!(a.pccp_no_promote);
    }

    // ── LmaCorpusSource stage tests ───────────────────────────
    fn source_ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn source_rejects_missing_dir() {
        let td = tempfile::tempdir().unwrap();
        let r = LmaCorpusSource
            .run(
                &source_ctx(td.path()),
                (),
                &LmaCorpusSourceArgs {
                    lma_root: td.path().join("__not_a_dir__"),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn source_counts_lma_archives() {
        let td = tempfile::tempdir().unwrap();
        let corpus = td.path().join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        // Two .lma + one .txt — only .lma should count.
        std::fs::write(corpus.join("a.lma"), b"a").unwrap();
        std::fs::write(corpus.join("b.lma"), b"b").unwrap();
        std::fs::write(corpus.join("notes.txt"), b"x").unwrap();
        let out = LmaCorpusSource
            .run(
                &source_ctx(td.path()),
                (),
                &LmaCorpusSourceArgs {
                    lma_root: corpus.clone(),
                },
            )
            .await
            .unwrap();
        assert_eq!(out.root, corpus);
        assert_eq!(out.n_archives, 2);
        // content_hash must be derivable, not zero/sentinel.
        assert_ne!(out.content_hash, ContentHash::of_bytes(b""));
    }

    #[test]
    fn is_swap_candidate_of_encoder() {
        // Same input_kinds=[] + output_kind=lamquant.pccp_verdict
        // means the full encoder pipeline can swap to this inner loop.
        assert!(
            crate::recipes::recipe::swap_candidates(&crate::recipes::lamquant_encoder::DEF)
                .any(|r| r.name == "lamquant_joint_codec")
        );
    }
}
