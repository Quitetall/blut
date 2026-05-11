//! HF Trainer backend stages.
//!
//! Each stage here drives `HfTrainerRunner` (the subprocess +
//! venv runner) with a task-specific job spec. They mirror the
//! lamu trainer.py stages (`sft_train`, `dpo_train`,
//! `distill_train`) in their typed I/O so recipes can swap
//! backends with minimal change beyond the recipe-level
//! `type Backend` tag.

pub mod hf_dpo_train;
pub mod hf_sft_train;

pub use hf_dpo_train::HfDpoTrain;
pub use hf_sft_train::HfSftTrain;
