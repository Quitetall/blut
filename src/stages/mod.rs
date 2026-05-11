//! Concrete stage catalog for BLUT.
//!
//! Each `pub mod` here implements `framework::Stage` for one
//! atomic unit of work. Recipes (in `recipes/`) compose these
//! into typed Plans.
//!
//! v2 commit 4 ships the full SFT-from-conversations pipeline:
//! materialize_conversations → filter_dataset → split_train_eval →
//! register_dataset → sft_train → merge_lora → convert_gguf →
//! register_model. eval_* and the parallel-executor branch stages
//! ship in later commits.

pub mod convert_gguf;
pub mod distill_train;
pub mod dpo_train;
pub mod eval_judge;
pub mod eval_lm_harness;
pub mod eval_loss;
pub mod filter_dataset;
pub mod lamquant_build_manifest;
pub(crate) mod lamquant_helpers;
pub mod lamquant_pccp_gate_encoder;
pub mod lamquant_pccp_gate_snn;
pub mod lamquant_precompute_fullband;
pub mod lamquant_precompute_l3;
pub mod lamquant_pretrain_mae;
pub mod lamquant_train_combined;
pub mod lamquant_train_joint;
pub mod lamquant_train_l3_teacher;
pub mod lamquant_train_mamba_snn;
pub mod lamquant_train_teacher;
pub mod lamquant_train_vocos_decoder;
pub mod materialize_conversations;
pub mod materialize_dataset_path;
pub mod materialize_for_eval;
pub mod merge_lora;
pub mod merge_reports;
pub mod register_dataset;
pub mod register_model;
pub mod sft_train;
pub mod split_train_eval;
pub mod take_train;
pub(crate) mod util;

pub mod catalog;

pub use convert_gguf::ConvertGguf;
pub use distill_train::DistillTrain;
pub use dpo_train::DpoTrain;
pub use eval_judge::EvalJudge;
pub use eval_lm_harness::EvalLmHarness;
pub use eval_loss::EvalLoss;
pub use filter_dataset::FilterDataset;
pub use lamquant_build_manifest::LamquantBuildManifest;
pub use lamquant_pccp_gate_encoder::{LamquantPccpGateDecoder, LamquantPccpGateEncoder};
pub use lamquant_pccp_gate_snn::LamquantPccpGateSnn;
pub use lamquant_precompute_fullband::LamquantPrecomputeFullband;
pub use lamquant_precompute_l3::LamquantPrecomputeL3;
pub use lamquant_pretrain_mae::LamquantPretrainMae;
pub use lamquant_train_combined::LamquantTrainCombined;
pub use lamquant_train_joint::LamquantTrainJoint;
pub use lamquant_train_l3_teacher::LamquantTrainL3Teacher;
pub use lamquant_train_mamba_snn::LamquantTrainMambaSnn;
pub use lamquant_train_teacher::LamquantTrainTeacher;
pub use lamquant_train_vocos_decoder::LamquantTrainVocosDecoder;
pub use materialize_conversations::MaterializeConversations;
pub use materialize_dataset_path::MaterializeDatasetPath;
pub use materialize_for_eval::MaterializeForEval;
pub use merge_lora::MergeLora;
pub use merge_reports::MergeReports;
pub use register_dataset::RegisterDataset;
pub use register_model::RegisterModel;
pub use sft_train::SftTrain;
pub use split_train_eval::SplitTrainEval;
pub use take_train::TakeTrain;
