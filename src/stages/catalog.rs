//! Static stage catalog.
//!
//! Maps stage name → `Box<dyn StageDyn>` constructor so the
//! `blut stage <name>` Unix-style invocation can locate the
//! stage at runtime. Mirrors the `RECIPES` slice's shape.

use crate::framework::stage::StageDyn;

/// Construct a fresh erased stage handle by name. Returns `None`
/// for unknown names so the CLI can surface a clear error.
pub fn make_stage(name: &str) -> Option<Box<dyn StageDyn>> {
    use super::*;
    Some(match name {
        "lamquant_build_manifest" => Box::new(LamquantBuildManifest),
        "lamquant_convert_lma" => Box::new(LamquantConvertLma),
        "lamquant_export_firmware" => Box::new(LamquantExportFirmware),
        "lamquant_generate_snn_labels" => Box::new(LamquantGenerateSnnLabels),
        "lamquant_harden_artifacts" => Box::new(LamquantHardenArtifacts),
        "lamquant_train_student" => Box::new(LamquantTrainStudent),
        "lamquant_pccp_gate_decoder" => Box::new(LamquantPccpGateDecoder),
        "lamquant_pccp_gate_encoder" => Box::new(LamquantPccpGateEncoder),
        "lamquant_pccp_gate_snn" => Box::new(LamquantPccpGateSnn),
        "lamquant_precompute_fullband" => Box::new(LamquantPrecomputeFullband),
        "lamquant_precompute_l3" => Box::new(LamquantPrecomputeL3),
        "lamquant_pretrain_mae" => Box::new(LamquantPretrainMae),
        "lamquant_train_combined" => Box::new(LamquantTrainCombined),
        "lamquant_train_joint" => Box::new(LamquantTrainJoint),
        "lamquant_train_l3_teacher" => Box::new(LamquantTrainL3Teacher),
        "lamquant_train_mamba_snn" => Box::new(LamquantTrainMambaSnn),
        "lamquant_train_teacher" => Box::new(LamquantTrainTeacher),
        "lamquant_train_vocos_decoder" => Box::new(LamquantTrainVocosDecoder),
        "materialize_conversations" => Box::new(MaterializeConversations),
        "materialize_dataset_path" => Box::new(MaterializeDatasetPath),
        "materialize_for_eval" => Box::new(MaterializeForEval),
        "filter_dataset" => Box::new(FilterDataset),
        "split_train_eval" => Box::new(SplitTrainEval),
        "register_dataset" => Box::new(RegisterDataset),
        "take_train" => Box::new(TakeTrain),
        "sft_train" => Box::new(SftTrain),
        "dpo_train" => Box::new(DpoTrain),
        "distill_train" => Box::new(DistillTrain),
        "merge_lora" => Box::new(MergeLora),
        "convert_gguf" => Box::new(ConvertGguf),
        "register_model" => Box::new(RegisterModel),
        "eval_loss" => Box::new(EvalLoss),
        "eval_lm_harness" => Box::new(EvalLmHarness),
        "eval_judge" => Box::new(EvalJudge),
        "merge_reports" => Box::new(MergeReports),
        _ => return None,
    })
}

/// All catalog names — for `blut stage list` and shell completions.
pub fn names() -> &'static [&'static str] {
    &[
        "lamquant_build_manifest",
        "lamquant_convert_lma",
        "lamquant_export_firmware",
        "lamquant_generate_snn_labels",
        "lamquant_harden_artifacts",
        "lamquant_train_student",
        "lamquant_pccp_gate_decoder",
        "lamquant_pccp_gate_encoder",
        "lamquant_pccp_gate_snn",
        "lamquant_precompute_fullband",
        "lamquant_precompute_l3",
        "lamquant_pretrain_mae",
        "lamquant_train_combined",
        "lamquant_train_joint",
        "lamquant_train_l3_teacher",
        "lamquant_train_mamba_snn",
        "lamquant_train_teacher",
        "lamquant_train_vocos_decoder",
        "materialize_conversations",
        "materialize_dataset_path",
        "materialize_for_eval",
        "filter_dataset",
        "split_train_eval",
        "register_dataset",
        "take_train",
        "sft_train",
        "dpo_train",
        "distill_train",
        "merge_lora",
        "convert_gguf",
        "register_model",
        "eval_loss",
        "eval_lm_harness",
        "eval_judge",
        "merge_reports",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_name_constructs() {
        for n in names() {
            let s = make_stage(n).unwrap_or_else(|| panic!("missing catalog entry: {n}"));
            assert_eq!(s.name(), *n, "catalog name vs stage name mismatch for {n}");
        }
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(make_stage("definitely-not-a-stage").is_none());
    }
}
