//! Backend abstraction.
//!
//! BLUT is an orchestrator. Concrete training engines (HuggingFace
//! Trainer, the LAMU `trainer.py` wire, the LamQuant kernel
//! catalog) implement `TrainingBackend` and live in submodules
//! under `backends/`. Each backend brings its own subprocess
//! runner, its own typed stages (under `backends/<id>/stages/`),
//! and — most importantly — its own identity.
//!
//! Identity matters because the typed `Plan<Out, B>` is
//! parameterized over a backend `B`. A recipe declares
//! `type Backend = HfTrainerBackend`; the compiler then refuses
//! to wire a `LamuTrainerBackend`-tagged stage into its plan. A
//! valid plan can only contain stages compatible with its
//! backend (or backend-agnostic stages — `AgnosticStage`).
//!
//! Why a marker trait instead of a uniform `run(...)` method?
//! Because backends don't share a single execution shape:
//! HF Trainer takes a `TrainingArguments` dict + dataset; LAMU's
//! trainer.py takes a `TrainSpec` JSON; LamQuant kernels are
//! argparse scripts. Forcing a uniform method would either lose
//! all type information at the wire (back to a generic JSON-blob
//! protocol) or constrain every backend to one shape. Instead,
//! each backend defines its own typed kernels and BLUT's job is
//! to compose them via the Plan DAG.
//!
//! Stage compatibility is enforced at compile time by the
//! `BackendStage<B>` trait (lands in BB-2). Backend-agnostic
//! stages (data prep, hashing, materializers) implement
//! `AgnosticStage` and slot into any plan via a blanket impl.

pub mod hf_trainer;
pub mod lamquant;
pub mod lamu;

/// Marker trait for training backend identity. Every concrete
/// backend is a unit-struct that implements this. The `ID` const
/// is the same string used in recipe metadata + CLI hints.
pub trait TrainingBackend: Send + Sync + 'static {
    /// Stable identifier. Used in recipe metadata, CLI output,
    /// status events, and provenance JSON. Treat as a SCHEMA bump:
    /// changing the ID invalidates every cache + audit reference
    /// that points at this backend.
    const ID: &'static str;

    /// Human-readable description for `blut recipe list` / docs.
    const DESCRIPTION: &'static str;
}

/// HuggingFace Trainer backend. Subprocesses a Python wrapper
/// that drives `transformers.Trainer` (and `trl.DPOTrainer` for
/// preference-pair fine-tuning). Manages its own venv at
/// `~/.local/share/blut/hf-venv/` on first use.
///
/// This is BLUT's blessed default for LLM-style training. New
/// `hf_*` recipes ship under it.
pub struct HfTrainerBackend;
impl TrainingBackend for HfTrainerBackend {
    const ID: &'static str = "hf_trainer";
    const DESCRIPTION: &'static str =
        "HuggingFace Trainer (transformers + trl). Auto-managed venv. Default for SFT/DPO/distillation.";
}

/// LAMU's `trainer.py` wire. The original BLUT backend — emits
/// `StatusUpdate` JSON lines, expects a `TrainSpec` blob on argv.
/// Kept for back-compat with the existing finetune_from_*
/// recipes; new recipes should prefer `HfTrainerBackend`.
pub struct LamuTrainerBackend;
impl TrainingBackend for LamuTrainerBackend {
    const ID: &'static str = "lamu";
    const DESCRIPTION: &'static str =
        "LAMU trainer.py (TrainSpec JSON / StatusUpdate wire). Original backend.";
}

/// LamQuant kernel catalog. Each kernel is a bespoke argparse
/// Python script (train_joint, train_mamba_snn, train_teacher,
/// etc). The runner streams stdout + parses tqdm progress.
/// Stages are NOT interchangeable with HF / LAMU — LamQuant
/// kernels are domain-specific (seizure detection, EEG encoder
/// quantization). Recipes under this backend stay bespoke.
pub struct LamquantBackend;
impl TrainingBackend for LamquantBackend {
    const ID: &'static str = "lamquant";
    const DESCRIPTION: &'static str =
        "LamQuant argparse kernels (EEG encoder / decoder / SNN / teacher). Bespoke per-paradigm.";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_ids_are_unique() {
        let ids = [
            HfTrainerBackend::ID,
            LamuTrainerBackend::ID,
            LamquantBackend::ID,
        ];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "backend IDs must be unique");
    }

    #[test]
    fn ids_are_stable_strings() {
        // Lock the wire identifiers — bumping these invalidates
        // every cache + audit reference. Catch unintentional
        // renames here.
        assert_eq!(HfTrainerBackend::ID, "hf_trainer");
        assert_eq!(LamuTrainerBackend::ID, "lamu");
        assert_eq!(LamquantBackend::ID, "lamquant");
    }
}
