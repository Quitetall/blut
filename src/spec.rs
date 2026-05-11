//! TrainSpec — the validated, immutable description of one training run.
//!
//! This type is the single source of truth for a job. The CLI deserializes
//! it from flags, the MCP tool deserializes it from JSON-RPC arguments,
//! the on-disk `spec.json` records it, and the Python trainer reads it
//! verbatim. Adding a knob means adding a field here; nothing reads
//! arguments out-of-band.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

/// The fine-tuning method. Determines memory profile + which adapters
/// land in the saved checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Method {
    /// 4-bit quantized base + LoRA adapters. Fits 7B on 24GB.
    QLora { rank: u32, alpha: u32 },
    /// 16-bit base + LoRA adapters. ~2× the VRAM of QLoRA.
    Lora { rank: u32, alpha: u32 },
    /// All weights trainable. Realistic only for <2B on a single 4090.
    Full,
}

impl Method {
    pub fn default_qlora() -> Self {
        Self::QLora { rank: 16, alpha: 32 }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::QLora { rank, alpha } | Self::Lora { rank, alpha } => {
                if *rank == 0 {
                    return Err(TrainError::invalid_spec("LoRA rank must be > 0"));
                }
                if *alpha == 0 {
                    return Err(TrainError::invalid_spec("LoRA alpha must be > 0"));
                }
                if *rank > 256 {
                    return Err(TrainError::invalid_spec("LoRA rank > 256 is unsupported"));
                }
            }
            Self::Full => {}
        }
        Ok(())
    }
}

/// Source of training examples. Resolved to a JSONL path before the
/// Python trainer runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetSource {
    /// A JSONL file already on disk.
    JsonlPath { path: PathBuf },
    /// Pull conversations from `lamu-mcp` SQLite memory; everything at
    /// or after `since_ts` (UNIX seconds) is included. The dump is
    /// materialized to a JSONL path before training.
    Conversations { since_ts: i64 },
    /// A dataset previously registered in the datasets table by name.
    Registered { name: String },
}

/// Optimizer choice. Mirrors `transformers` `optim=` strings where one
/// exists. APOLLO variants ship via either the upstream PR or a vendored
/// fallback (see step 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Optim {
    AdamW,
    AdamW8bit,
    /// APOLLO rank-1: SGD-level optimizer state, AdamW-comparable loss.
    ApolloMini,
    /// APOLLO rank-4: better-conditioned than rank-1, ~5% more memory.
    ApolloRank4,
}

impl Optim {
    /// Map to the `optim=` argument string used by `transformers`'
    /// `TrainingArguments`. APOLLO strings are valid only after the
    /// upstream PR merges; until then the trainer uses the vendored
    /// fallback path which inspects this enum directly, not the string.
    pub fn transformers_optim_str(&self) -> &'static str {
        match self {
            Self::AdamW => "adamw_torch",
            Self::AdamW8bit => "adamw_8bit",
            Self::ApolloMini => "apollo_mini",
            Self::ApolloRank4 => "apollo",
        }
    }
}

/// Validated, ready-to-execute training description.
///
/// Build via `TrainSpec::builder` (future) or by direct construction +
/// `validate`. Never construct one without calling `validate` — the
/// trainer subprocess assumes every numeric field is in range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainSpec {
    /// HuggingFace repo id (`org/name`) of the base model.
    pub base_model: String,

    /// Registry name to assign to the trained model.
    pub output_name: String,

    /// Where the HF checkpoint will land. Parent dir must exist; trainer
    /// creates the leaf.
    pub output_dir: PathBuf,

    pub method: Method,
    pub dataset: DatasetSource,
    pub optimizer: Optim,

    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,

    /// Random seed for reproducibility. Same seed + same dataset hash
    /// + same spec → same final loss to within numerical noise.
    pub seed: u64,

    /// Final GGUF quant. `f16` skips quantization. Use `Q4_K_M` for
    /// the 4090-friendly default.
    pub quant: String,

    /// If true, skip GGUF convert + registry register. Used by callers
    /// who only want the HF checkpoint (e.g. uploading elsewhere).
    #[serde(default)]
    pub skip_convert: bool,
}

impl TrainSpec {
    /// Sensible defaults for a 4090 + 7B base + QLoRA. Fields you almost
    /// always want to override: `base_model`, `output_name`, `output_dir`,
    /// `dataset`. Everything else has a justifiable default.
    pub fn defaults_qlora_7b(
        base_model: impl Into<String>,
        output_name: impl Into<String>,
        output_dir: PathBuf,
        dataset: DatasetSource,
    ) -> Self {
        Self {
            base_model: base_model.into(),
            output_name: output_name.into(),
            output_dir,
            method: Method::default_qlora(),
            dataset,
            optimizer: Optim::ApolloMini,
            lr: 2e-4,
            epochs: 3,
            batch_size: 1,
            grad_accum: 8,
            seq_len: 4096,
            seed: 42,
            quant: "Q4_K_M".into(),
            skip_convert: false,
        }
    }

    /// Validate every field. Aggressive — fail loud with a specific
    /// message. The trainer subprocess assumes a validated spec; never
    /// hand it an unvalidated one.
    pub fn validate(&self) -> Result<()> {
        if self.base_model.trim().is_empty() {
            return Err(TrainError::invalid_spec("base_model is empty"));
        }
        if !is_safe_hf_repo_id(&self.base_model) {
            return Err(TrainError::invalid_spec(format!(
                "base_model '{}' is not a safe HuggingFace repo id \
                 (expected 'org/name', no path traversal, no leading slash)",
                self.base_model
            )));
        }
        if self.output_name.trim().is_empty() {
            return Err(TrainError::invalid_spec("output_name is empty"));
        }
        if !is_safe_registry_name(&self.output_name) {
            return Err(TrainError::invalid_spec(format!(
                "output_name '{}' must match [A-Za-z0-9_.-]+ \
                 (no leading dot/dash, no '..')",
                self.output_name
            )));
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err(TrainError::invalid_spec("output_dir is empty"));
        }
        if !self.output_dir.is_absolute() {
            return Err(TrainError::invalid_spec(format!(
                "output_dir '{}' must be an absolute path",
                self.output_dir.display()
            )));
        }
        if path_has_parent_dir(&self.output_dir) {
            return Err(TrainError::invalid_spec(format!(
                "output_dir '{}' must not contain '..' segments",
                self.output_dir.display()
            )));
        }
        self.method.validate()?;
        match &self.dataset {
            DatasetSource::JsonlPath { path } => {
                if path.as_os_str().is_empty() {
                    return Err(TrainError::invalid_spec("dataset.path is empty"));
                }
                if !path.is_absolute() {
                    return Err(TrainError::invalid_spec(format!(
                        "dataset.path '{}' must be an absolute path",
                        path.display()
                    )));
                }
                if path_has_parent_dir(path) {
                    return Err(TrainError::invalid_spec(format!(
                        "dataset.path '{}' must not contain '..' segments",
                        path.display()
                    )));
                }
            }
            DatasetSource::Conversations { since_ts } => {
                if *since_ts < 0 {
                    return Err(TrainError::invalid_spec(
                        "dataset.since_ts must be >= 0",
                    ));
                }
            }
            DatasetSource::Registered { name } => {
                if name.trim().is_empty() {
                    return Err(TrainError::invalid_spec(
                        "dataset.name is empty",
                    ));
                }
            }
        }
        if !self.lr.is_finite() || self.lr <= 0.0 || self.lr > 1.0 {
            return Err(TrainError::invalid_spec(format!(
                "lr {} out of (0.0, 1.0]",
                self.lr
            )));
        }
        if self.epochs == 0 {
            return Err(TrainError::invalid_spec("epochs must be >= 1"));
        }
        if self.batch_size == 0 {
            return Err(TrainError::invalid_spec("batch_size must be >= 1"));
        }
        if self.grad_accum == 0 {
            return Err(TrainError::invalid_spec("grad_accum must be >= 1"));
        }
        if self.seq_len < 128 {
            return Err(TrainError::invalid_spec("seq_len < 128 is unsupported"));
        }
        if !is_supported_quant(&self.quant) {
            return Err(TrainError::invalid_spec(format!(
                "quant '{}' is not supported (try Q4_K_M, Q5_K_M, Q8_0, f16)",
                self.quant
            )));
        }
        Ok(())
    }
}

fn is_safe_registry_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
        && !name.starts_with('.')
        && !name.starts_with('-')
        && !name.contains("..")
}

/// Accept the standard HuggingFace `org/name` shape only. Rejects:
///   - empty / whitespace
///   - missing `/`
///   - leading `/` (absolute path coercion attempts)
///   - `..` segment (path traversal)
///   - other path separators (`\`, NUL)
///   - more than one `/` (sub-paths inside a repo aren't valid repo ids)
fn is_safe_hf_repo_id(repo: &str) -> bool {
    let trimmed = repo.trim();
    if trimmed.is_empty() || trimmed.starts_with('/') {
        return false;
    }
    if trimmed.contains('\\') || trimmed.contains('\0') {
        return false;
    }
    let parts: Vec<&str> = trimmed.split('/').collect();
    if parts.len() != 2 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && *p != "."
            && !p.contains("..")
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    })
}

/// True iff `p` contains a `..` component. Catches both literal `..`
/// segments and `..` embedded in a longer name (per Path's component
/// semantics, only the literal `..` segment counts as a parent ref —
/// but we reject either way for defence in depth).
fn path_has_parent_dir(p: &std::path::Path) -> bool {
    p.components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
        || p.to_string_lossy().contains("/..")
        || p.to_string_lossy().starts_with("..")
}

fn is_supported_quant(q: &str) -> bool {
    matches!(
        q,
        "Q2_K"
            | "Q3_K_S"
            | "Q3_K_M"
            | "Q3_K_L"
            | "Q4_K_S"
            | "Q4_K_M"
            | "Q5_K_S"
            | "Q5_K_M"
            | "Q6_K"
            | "Q8_0"
            | "f16"
            | "bf16"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_spec() -> TrainSpec {
        TrainSpec::defaults_qlora_7b(
            "Qwen/Qwen3-7B",
            "test-out",
            PathBuf::from("/tmp/lamu-train-test"),
            DatasetSource::JsonlPath {
                path: PathBuf::from("/tmp/x.jsonl"),
            },
        )
    }

    #[test]
    fn defaults_validate() {
        let s = good_spec();
        s.validate().expect("defaults must validate");
    }

    #[test]
    fn rejects_empty_base_model() {
        let mut s = good_spec();
        s.base_model = String::new();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_base_model_without_slash() {
        let mut s = good_spec();
        s.base_model = "no-slash".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_unsafe_output_name() {
        let mut s = good_spec();
        s.output_name = "../escape".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_unsafe_output_name_dot_prefix() {
        let mut s = good_spec();
        s.output_name = ".hidden".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_output_name_double_dot() {
        let mut s = good_spec();
        s.output_name = "a..b".into();
        assert!(s.validate().is_err(), "registry name with '..' must reject");
        s.output_name = "..".into();
        assert!(s.validate().is_err(), "registry name '..' must reject");
    }

    #[test]
    fn rejects_relative_output_dir() {
        let mut s = good_spec();
        s.output_dir = PathBuf::from("relative/path");
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_output_dir_with_parent_segment() {
        let mut s = good_spec();
        s.output_dir = PathBuf::from("/tmp/../etc/cron.d");
        assert!(
            s.validate().is_err(),
            "absolute path with .. segment must reject"
        );
        s.output_dir = PathBuf::from("/tmp/..");
        assert!(
            s.validate().is_err(),
            "absolute path with trailing .. must reject"
        );
    }

    #[test]
    fn rejects_relative_dataset_path() {
        let mut s = good_spec();
        s.dataset = DatasetSource::JsonlPath {
            path: PathBuf::from("relative/data.jsonl"),
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_dataset_path_with_parent_segment() {
        let mut s = good_spec();
        s.dataset = DatasetSource::JsonlPath {
            path: PathBuf::from("/tmp/../etc/passwd"),
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_base_model_with_dotdot_in_name() {
        let mut s = good_spec();
        s.base_model = "org/...".into();
        assert!(
            s.validate().is_err(),
            "three-dot segment must reject (contains '..')"
        );
        s.base_model = "or..g/name".into();
        assert!(
            s.validate().is_err(),
            "embedded '..' in org segment must reject"
        );
    }

    #[test]
    fn rejects_base_model_with_dotdot() {
        let mut s = good_spec();
        s.base_model = "../etc/passwd".into();
        assert!(s.validate().is_err());
        s.base_model = "org/..".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_base_model_with_leading_slash() {
        let mut s = good_spec();
        s.base_model = "/abs/path".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_base_model_with_too_many_slashes() {
        let mut s = good_spec();
        s.base_model = "org/name/sub".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_base_model_with_backslash() {
        let mut s = good_spec();
        s.base_model = "org\\name".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_unsupported_quant() {
        let mut s = good_spec();
        s.quant = "Q9_GIGA".into();
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_zero_epochs() {
        let mut s = good_spec();
        s.epochs = 0;
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_lr_out_of_range() {
        let mut s = good_spec();
        s.lr = 0.0;
        assert!(s.validate().is_err());
        s.lr = 2.0;
        assert!(s.validate().is_err());
        s.lr = f32::NAN;
        assert!(s.validate().is_err());
    }

    #[test]
    fn rejects_short_seq_len() {
        let mut s = good_spec();
        s.seq_len = 64;
        assert!(s.validate().is_err());
    }

    #[test]
    fn method_validates_lora_rank() {
        let m = Method::QLora { rank: 0, alpha: 32 };
        assert!(m.validate().is_err());
        let m = Method::Lora {
            rank: 1024,
            alpha: 32,
        };
        assert!(m.validate().is_err());
        let m = Method::QLora { rank: 16, alpha: 0 };
        assert!(m.validate().is_err());
    }

    #[test]
    fn json_round_trip_preserves_method_tag() {
        let s = good_spec();
        let json = serde_json::to_string(&s).unwrap();
        let back: TrainSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
        // Tag must be lowercased + snake_cased so Python can match it.
        assert!(json.contains("\"kind\":\"q_lora\""));
        assert!(json.contains("\"kind\":\"jsonl_path\""));
    }

    #[test]
    fn json_round_trip_all_optim_variants() {
        for o in [
            Optim::AdamW,
            Optim::AdamW8bit,
            Optim::ApolloMini,
            Optim::ApolloRank4,
        ] {
            let s = serde_json::to_string(&o).unwrap();
            let back: Optim = serde_json::from_str(&s).unwrap();
            assert_eq!(o, back);
        }
    }

    #[test]
    fn optim_str_mapping_is_stable() {
        assert_eq!(Optim::AdamW.transformers_optim_str(), "adamw_torch");
        assert_eq!(Optim::AdamW8bit.transformers_optim_str(), "adamw_8bit");
        assert_eq!(Optim::ApolloMini.transformers_optim_str(), "apollo_mini");
        assert_eq!(Optim::ApolloRank4.transformers_optim_str(), "apollo");
    }

    #[test]
    fn dataset_conversations_negative_ts_rejected() {
        let mut s = good_spec();
        s.dataset = DatasetSource::Conversations { since_ts: -1 };
        assert!(s.validate().is_err());
    }

    #[test]
    fn dataset_registered_empty_name_rejected() {
        let mut s = good_spec();
        s.dataset = DatasetSource::Registered { name: "".into() };
        assert!(s.validate().is_err());
    }
}
