//! LAMU trainer backend.
//!
//! Wraps the original `python/trainer.py` wire (TrainSpec JSON
//! argv, StatusUpdate JSON lines on stdout). Kept for back-compat
//! with existing finetune_from_* recipes. New recipes should
//! prefer `backends::hf_trainer` once it lands.

pub mod python_backend;

pub use python_backend::PythonTrainBackend;
