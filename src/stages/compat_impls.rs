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
//!   - **LAMQUANT-coupled**: every `lamquant_*` stage shells out
//!     to a LamQuant kernel script under `$LAMQUANT_HOME`.
//!     Always backend = LamquantBackend.

use crate::backends::{HfTrainerBackend, LamquantBackend, LamuTrainerBackend, TrainingBackend};
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

// ── LAMQUANT-coupled stages ────────────────────────────────────
//
// Every `lamquant_*` stage shells out to a LamQuant kernel.

impl Compatible<LamquantBackend> for LamquantBuildManifest {}
impl Compatible<LamquantBackend> for LamquantBuildSplitManifest {}
impl Compatible<LamquantBackend> for LamquantConvertLma {}
impl Compatible<LamquantBackend> for LamquantEncodeLma {}
impl Compatible<LamquantBackend> for LamquantExportFirmware {}
impl Compatible<LamquantBackend> for LamquantGenerateSnnLabels {}
impl Compatible<LamquantBackend> for LamquantHardenArtifacts {}
impl Compatible<LamquantBackend> for LamquantPrecomputeFullband {}
impl Compatible<LamquantBackend> for LamquantPrecomputeL3 {}
impl Compatible<LamquantBackend> for LamquantPretrainMae {}
impl Compatible<LamquantBackend> for LamquantTrainTeacher {}
impl Compatible<LamquantBackend> for LamquantTrainL3Teacher {}
impl Compatible<LamquantBackend> for LamquantTrainJoint {}
impl Compatible<LamquantBackend> for LamquantTrainStudent {}
impl Compatible<LamquantBackend> for LamquantTrainVocosDecoder {}
impl Compatible<LamquantBackend> for LamquantTrainCombined {}
impl Compatible<LamquantBackend> for LamquantTrainMambaSnn {}
impl Compatible<LamquantBackend> for LamquantPccpGateSnn {}
impl Compatible<LamquantBackend> for LamquantPccpGateEncoder {}
impl Compatible<LamquantBackend> for LamquantPccpGateDecoder {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::Stage;

    // Compile-time witness: each named stage typechecks against
    // its declared backend(s). If a future stage refactor breaks
    // the bound, these `fn _f<...>()` lines fail to compile —
    // exactly the auditing property we want.

    fn _agnostic_witness<
        S: Stage
            + Compatible<HfTrainerBackend>
            + Compatible<LamuTrainerBackend>
            + Compatible<LamquantBackend>,
    >() {
    }
    fn _lamu_witness<S: Stage + Compatible<LamuTrainerBackend>>() {}
    fn _lamquant_witness<S: Stage + Compatible<LamquantBackend>>() {}

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

    #[test]
    fn lamquant_witnesses() {
        _lamquant_witness::<LamquantBuildManifest>();
        _lamquant_witness::<LamquantTrainMambaSnn>();
        _lamquant_witness::<LamquantTrainJoint>();
        _lamquant_witness::<LamquantPccpGateSnn>();
    }
}
