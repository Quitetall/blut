//! Backend abstraction — the public 1.0 backend-identity seam.
//!
//! BLUT is an orchestrator. Concrete training engines (HuggingFace
//! Trainer, the LAMU `trainer.py` wire, plus domain backends supplied
//! by cookbook crates) implement [`TrainingBackend`] and live OUTSIDE
//! the engine: at the engine-only v1.0 carve the concrete identity
//! structs (`HfTrainerBackend`, `LamuTrainerBackend`) + their runners /
//! stages moved to the `blut-backends` crate. The engine keeps only the
//! abstract trait here so a domain cookbook can tag its own backend
//! identity without depending on the generic-LLM layer.
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
//! `Compatible<B>` trait. Backend-agnostic stages (data prep,
//! hashing, materializers) implement `Compatible<B>` for all `B`
//! and slot into any plan.

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

// The concrete `HfTrainerBackend` / `LamuTrainerBackend` identities +
// their subprocess runners + typed stages live in the `blut-backends`
// crate (engine carve, v1.0). The LamQuant kernel backend lives in the
// `cookbook-lamquant` crate. The engine ships ZERO concrete backends.

/// Test-only concrete backend identity. The engine ships no concrete
/// backend, but the framework's own unit tests (`plan`, `executor`)
/// need a `TrainingBackend` to parameterize `Plan<Out, B>` and the
/// `Compatible<B>` impls of their toy stages. This in-crate fixture
/// keeps those tests independent of the `blut-backends` carve. It
/// re-uses the `"lamu"` identifier purely as a stable test string.
#[cfg(test)]
pub struct LamuTrainerBackend;

#[cfg(test)]
impl TrainingBackend for LamuTrainerBackend {
    const ID: &'static str = "lamu";
    const DESCRIPTION: &'static str = "Test fixture backend (engine framework tests only).";
}
