//! HuggingFace Trainer backend.
//!
//! Subprocesses a Python wrapper that drives
//! `transformers.Trainer` (SFT, distillation) and
//! `trl.DPOTrainer` (preference fine-tuning). Manages its own
//! venv at `~/.local/share/blut/hf-venv/` — auto-provisioned on
//! first use so users don't fight Python environment drift.
//!
//! Land status (BB-1): stub. Real runner + stages + recipes land
//! in BB-4 + BB-5. This module exists now so the typed `Plan`
//! machinery can name `HfTrainerBackend` at compile time
//! everywhere it needs to without forward-declaring.

// Future:
//   pub mod runner;        // subprocess + JSON wire
//   pub mod venv;          // auto-managed venv at ~/.local/share/blut/hf-venv/
//   pub mod stages;        // hf_sft_train, hf_dpo_train, hf_distill_train, ...
//   pub mod python;        // hf_trainer_runner.py + pyproject.toml metadata

// Nothing public yet — kept module-empty so the backends/mod.rs
// `pub mod hf_trainer;` compiles cleanly.
