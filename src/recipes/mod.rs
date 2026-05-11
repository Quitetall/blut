//! Recipes — saved compositions of stages.
//!
//! Each recipe takes typed args and compiles to a `Plan<()>`. The
//! catalog is a `static RECIPES: &[RecipeDef]` mirroring
//! lamu-mcp's `TOOLS` pattern: registering a new recipe is one
//! block of code with a `name`, `description`, and `compile_fn`.

pub mod dpo_from_preferences;
pub mod finetune_from_conversations;
pub mod recipe;

pub use dpo_from_preferences::DpoFromPreferences;
pub use finetune_from_conversations::FinetuneFromConversations;
pub use recipe::{Recipe, RecipeDef, RECIPES};
