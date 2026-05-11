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
        "materialize_conversations" => Box::new(MaterializeConversations),
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
        "materialize_conversations",
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
