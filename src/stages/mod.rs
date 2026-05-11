//! Concrete stage catalog for BLUT.
//!
//! Each `pub mod` here implements `framework::Stage` for one
//! atomic unit of work. Recipes (in `recipes/`) compose these
//! into typed Plans.
//!
//! v2 commit 4 ships the SFT-from-conversations critical path:
//! materialize_conversations, sft_train, convert_gguf,
//! register_model. Other stages (filter_dataset, split_train_eval,
//! eval_*, dpo_train, distill_train, merge_lora) ship in later
//! commits.

pub mod convert_gguf;
pub mod distill_train;
pub mod dpo_train;
pub mod materialize_conversations;
pub mod register_model;
pub mod sft_train;

pub use convert_gguf::ConvertGguf;
pub use distill_train::DistillTrain;
pub use dpo_train::DpoTrain;
pub use materialize_conversations::MaterializeConversations;
pub use register_model::RegisterModel;
pub use sft_train::SftTrain;
