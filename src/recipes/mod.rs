//! Recipes — saved compositions of stages.
//!
//! Each recipe takes typed args and compiles to a `Plan<()>`. The
//! catalog is a `static RECIPES: &[RecipeDef]` mirroring
//! lamu-mcp's `TOOLS` pattern: registering a new recipe is one
//! block of code with a `name`, `description`, and `compile_fn`.

pub mod distill_from_teacher;
pub mod dpo_from_preferences;
pub mod eval_suite;
pub mod finetune_from_conversations;
pub mod finetune_from_dataset;
pub mod lamquant_encoder;
pub mod lamquant_oracle;
pub mod lamquant_snn;
pub mod recipe;

pub use distill_from_teacher::DistillFromTeacher;
pub use dpo_from_preferences::DpoFromPreferences;
pub use eval_suite::EvalSuite;
pub use finetune_from_conversations::FinetuneFromConversations;
pub use finetune_from_dataset::FinetuneFromDataset;
pub use lamquant_encoder::LamquantEncoder;
pub use lamquant_oracle::LamquantOracle;
pub use lamquant_snn::LamquantSnn;
pub use recipe::{Recipe, RecipeDef, RECIPES};
