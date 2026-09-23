// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Eval reports.
//!
//! Paradigm-agnostic metric bag emitted by `eval_loss`, `eval_lm_harness`,
//! `eval_judge`, and any future eval stages. Combined by
//! `merge_reports` into a single artifact for the `eval_suite` recipe's
//! terminal node.
//!
//! Metric values are JSON to keep the schema open (numbers,
//! per-task subreports, optional CIs). Consumers (the report
//! renderer, downstream `continual_finetune` recipes) parse the
//! values they care about and ignore the rest.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalReport {
    /// Where the JSON form of this report lives on disk (so it
    /// can be inspected without going through the cache).
    pub path: PathBuf,
    /// Logical name of the evaluator that produced this report
    /// (`"eval_loss"`, `"eval_lm_harness"`, `"eval_judge"`,
    /// `"merge_reports"` for the combined output).
    pub evaluator: String,
    /// Free-form metric bag. Convention: top-level numeric metrics
    /// at the root, per-task breakdowns nested under
    /// `tasks.<task_name>`. Consumers may ignore unknown keys.
    #[serde(with = "metrics_codec")]
    pub metrics: serde_json::Value,
    pub content_hash: ContentHash,
}

/// Serde for the free-form `metrics` bag, readable by every format the engine
/// uses.
///
/// A `serde_json::Value` deserializes through `deserialize_any`, which a
/// non-self-describing format cannot provide. Erased artifacts cross the
/// `StageDyn` boundary as bincode, so an `EvalReport` encoded fine and then
/// never decoded: every stage that produced one failed at finalization with
/// "artifact handle did not decode as the selected stage role", after its
/// evaluation had already run.
///
/// Human-readable formats (JSON: reports on disk, plan outputs, the SDK) keep
/// the metrics as a nested object, byte-for-byte as before. Binary formats
/// carry them as a JSON string, which every format can round-trip.
mod metrics_codec {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &serde_json::Value, ser: S) -> Result<S::Ok, S::Error> {
        if ser.is_human_readable() {
            value.serialize(ser)
        } else {
            let text = serde_json::to_string(value).map_err(serde::ser::Error::custom)?;
            ser.serialize_str(&text)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<serde_json::Value, D::Error> {
        if de.is_human_readable() {
            serde_json::Value::deserialize(de)
        } else {
            let text = String::deserialize(de)?;
            serde_json::from_str(&text).map_err(serde::de::Error::custom)
        }
    }
}

impl Artifact for EvalReport {
    const KIND: &'static str = "eval.report";
    // 2: metrics travel as a JSON string in binary encodings. A v1 binary
    // payload could never be decoded, so nothing valid is invalidated.
    const SCHEMA: u32 = 2;
    const ALLOW_EXTERNAL_PATHS: bool = true;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_report_round_trips_via_serde() {
        let r = EvalReport {
            path: PathBuf::from("/tmp/eval.json"),
            evaluator: "eval_loss".into(),
            metrics: serde_json::json!({"loss": 0.42, "ppl": 1.52}),
            content_hash: ContentHash::of_bytes(b"x"),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: EvalReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.evaluator, "eval_loss");
        assert_eq!(back.metrics["loss"], serde_json::json!(0.42));
    }

    #[test]
    fn artifact_kind_is_stable() {
        assert_eq!(EvalReport::KIND, "eval.report");
        assert_eq!(EvalReport::SCHEMA, 2);
    }

    fn report() -> EvalReport {
        EvalReport {
            path: PathBuf::from("/tmp/eval.json"),
            evaluator: "evaluate_held_out".into(),
            metrics: serde_json::json!({
                "loss": 2.8475,
                "n_rows": 200,
                "tasks": {"wikitext": {"ppl": 17.2}},
            }),
            content_hash: ContentHash::of_bytes(b"x"),
        }
    }

    /// The engine hands artifacts between stages as bincode. Before the
    /// metrics codec this failed with "Bincode does not support the
    /// serde::Deserializer::deserialize_any method", so no stage that produced
    /// an `EvalReport` could ever finish.
    #[test]
    fn eval_report_survives_the_erased_stage_boundary() {
        let erased = crate::framework::stage::ErasedArtifact::from_typed(&report()).unwrap();
        let back: EvalReport = erased.into_typed().unwrap();
        assert_eq!(back.metrics, report().metrics);
        assert_eq!(back.evaluator, "evaluate_held_out");
    }

    #[test]
    fn eval_report_round_trips_through_bincode() {
        let bytes = bincode::serialize(&report()).unwrap();
        let back: EvalReport = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.metrics, report().metrics);
    }

    /// JSON consumers see the metrics as an object, exactly as before.
    #[test]
    fn json_keeps_metrics_as_a_nested_object() {
        let json = serde_json::to_value(report()).unwrap();
        assert_eq!(json["metrics"]["loss"], serde_json::json!(2.8475));
        assert_eq!(
            json["metrics"]["tasks"]["wikitext"]["ppl"],
            serde_json::json!(17.2)
        );
    }
}
