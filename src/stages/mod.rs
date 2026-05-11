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
pub mod materialize_conversations;
pub mod materialize_for_eval;
pub mod merge_lora;
pub mod merge_reports;
pub mod register_dataset;
pub mod register_model;
pub mod sft_train;
pub mod split_train_eval;
pub mod take_train;

pub use convert_gguf::ConvertGguf;
pub use distill_train::DistillTrain;
pub use dpo_train::DpoTrain;
pub use eval_judge::EvalJudge;
pub use eval_lm_harness::EvalLmHarness;
pub use eval_loss::EvalLoss;
pub use filter_dataset::FilterDataset;
pub use materialize_conversations::MaterializeConversations;
pub use materialize_for_eval::MaterializeForEval;
pub use merge_lora::MergeLora;
pub use merge_reports::MergeReports;
pub use register_dataset::RegisterDataset;
pub use register_model::RegisterModel;
pub use sft_train::SftTrain;
pub use split_train_eval::SplitTrainEval;
pub use take_train::TakeTrain;
