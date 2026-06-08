//! lamu recipes — generic-LLM training (SFT / DPO / distillation /
//! eval) over the `lamu` + `hf_trainer` backends.
//!
//! Each recipe takes typed args and compiles to a `Plan<()>` against a
//! backend that lives in `blut::backends`. The recipe MACHINERY
//! (`Recipe` trait, `RecipeDef`, `RecipeCategory`) stays in
//! `blut::recipes::recipe`; this crate owns only the concrete recipe
//! DEFs. Moved out of blut-core at C2b (`[[project_blut_cookbook_split]]`).

pub mod distill_from_teacher;
pub mod dpo_from_preferences;
pub mod eval_suite;
pub mod finetune_from_conversations;
pub mod finetune_from_dataset;
pub mod hf_finetune_from_dataset;

pub use distill_from_teacher::DistillFromTeacher;
pub use dpo_from_preferences::DpoFromPreferences;
pub use eval_suite::EvalSuite;
pub use finetune_from_conversations::FinetuneFromConversations;
pub use finetune_from_dataset::FinetuneFromDataset;
pub use hf_finetune_from_dataset::HfFinetuneFromDataset;
