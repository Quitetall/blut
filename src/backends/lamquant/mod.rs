//! LamQuant kernel backend.
//!
//! Bespoke per-kernel argparse runners — each LamQuant training
//! script (`train_joint.py`, `train_mamba_snn.py`,
//! `train_teacher.py`, etc.) shells through `LamquantBackend`
//! with its own arg shape. The runner streams stdout/stderr to
//! tracing and parses tqdm progress for `StageEvent::StageStep`
//! fan-out.

pub mod runner;

pub use runner::{
    default_lamquant_home, resolve_lamquant_python, BackendError, LamquantBackend,
    LamquantInvocation, LamquantRunArtifact, Progress,
};
