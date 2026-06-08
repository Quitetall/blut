//! `Compatible<B>` impls for every Stage in the catalog.
//!
//! Centralized here (vs sprinkled in each stage file) so an
//! auditor can see all backend-compatibility decisions in one
//! place. New stages added without a `Compatible<B>` impl will
//! fail to wire into any `Plan<O, B>` at compile time — a useful
//! "did you forget to declare backend compatibility?" guard.
//!
//! Categorization:
//!
//!   - **Backend-agnostic** (blanket `impl<B>`): pure data
//!     transforms with no backend coupling. Filter, split,
//!     projection. JSONL-in, JSONL-out, no subprocess.
//!
//!   - **LAMU-coupled**: stages that drive `lamu`'s trainer.py
//!     wire OR write to lamu's local registries (datasets_db,
//!     model registry, conversations.db, llama.cpp tools).
//!     Today's `sft_train`, `dpo_train`, `distill_train`,
//!     `convert_gguf`, `register_model`, `register_dataset`,
//!     `materialize_conversations`, `materialize_dataset_path`
//!     all touch lamu surface area; tagged LAMU.
//!
//! Domain-backend stages (e.g. the LamQuant kernel stages) carry their
//! own `Compatible<...>` impls in their owning cookbook crate (orphan
//! rule: the stage types + backend are local there), so they are NOT
//! listed here.

use crate::backends::{HfTrainerBackend, LamuTrainerBackend, TrainingBackend};
use crate::framework::Compatible;

use crate::stages::*;

// ── Backend-agnostic stages ────────────────────────────────────
//
// Pure data transforms. Slot into any plan regardless of backend.

impl<B: TrainingBackend> Compatible<B> for FilterDataset {}
impl<B: TrainingBackend> Compatible<B> for SplitTrainEval {}
impl<B: TrainingBackend> Compatible<B> for TakeTrain {}
impl<B: TrainingBackend> Compatible<B> for MaterializeForEval {}
impl<B: TrainingBackend> Compatible<B> for EvalLoss {}
impl<B: TrainingBackend> Compatible<B> for EvalLmHarness {}
impl<B: TrainingBackend> Compatible<B> for EvalJudge {}
impl<B: TrainingBackend> Compatible<B> for MergeReports {}
impl<B: TrainingBackend> Compatible<B> for MergeLora {}

// ── LAMU-coupled stages ────────────────────────────────────────
//
// Drive lamu's trainer.py wire OR write to lamu-local registries.

impl Compatible<LamuTrainerBackend> for MaterializeConversations {}
impl Compatible<LamuTrainerBackend> for MaterializeDatasetPath {}
impl Compatible<LamuTrainerBackend> for RegisterDataset {}
impl Compatible<LamuTrainerBackend> for SftTrain {}
impl Compatible<LamuTrainerBackend> for DpoTrain {}
impl Compatible<LamuTrainerBackend> for DistillTrain {}
impl Compatible<LamuTrainerBackend> for ConvertGguf {}
impl Compatible<LamuTrainerBackend> for RegisterModel {}

// Multi-backend (lamu + hf_trainer). HF recipes reuse these
// stages — internals aren't lamu-specific (registries are
// BLUT-shared, llama.cpp's convert/quantize tools work on any
// HF-format ckpt). MergeLora is already in the agnostic block
// above.
impl Compatible<HfTrainerBackend> for MaterializeDatasetPath {}
impl Compatible<HfTrainerBackend> for MaterializeConversations {}
impl Compatible<HfTrainerBackend> for RegisterDataset {}
impl Compatible<HfTrainerBackend> for ConvertGguf {}
impl Compatible<HfTrainerBackend> for RegisterModel {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::Stage;

    // Compile-time witness: each named stage typechecks against
    // its declared backend(s). If a future stage refactor breaks
    // the bound, these `fn _f<...>()` lines fail to compile —
    // exactly the auditing property we want.

    fn _agnostic_witness<
        S: Stage + Compatible<HfTrainerBackend> + Compatible<LamuTrainerBackend>,
    >() {
    }
    fn _lamu_witness<S: Stage + Compatible<LamuTrainerBackend>>() {}

    #[test]
    fn agnostic_compose_witnesses() {
        _agnostic_witness::<FilterDataset>();
        _agnostic_witness::<SplitTrainEval>();
        _agnostic_witness::<TakeTrain>();
    }

    #[test]
    fn lamu_witnesses() {
        _lamu_witness::<SftTrain>();
        _lamu_witness::<DpoTrain>();
        _lamu_witness::<DistillTrain>();
        _lamu_witness::<ConvertGguf>();
        _lamu_witness::<RegisterModel>();
    }
}
