//! Stage — `lamquant_pccp_gate_encoder`. JointCkpt → PccpVerdict.
//! Passes the joint's encoder_path as the candidate to
//! `ai_models/pccp_gate.py --model encoder`.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{JointCkpt, PccpVerdict};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{LamquantBackend, LamquantInvocation};
use crate::stages::lamquant_helpers::{python_for, resolve_home, script_path};

pub struct LamquantPccpGateEncoder;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default = "default_change_id")]
    pub change_id: String,
    #[serde(default = "default_description")]
    pub description: String,
    #[serde(default = "default_author")]
    pub author: String,
    #[serde(default = "default_change_class")]
    pub change_class: String,
    #[serde(default = "default_true")]
    pub dry_run: bool,
    #[serde(default = "default_true")]
    pub no_promote: bool,
}
fn default_change_id() -> String {
    "PCCP-CHG-DRYRUN".into()
}
fn default_description() -> String {
    "(blut encoder gate)".into()
}
fn default_author() -> String {
    "BLUT".into()
}
fn default_change_class() -> String {
    "A.1".into()
}
fn default_true() -> bool {
    true
}

#[async_trait]
impl Stage for LamquantPccpGateEncoder {
    const NAME: &'static str = "lamquant_pccp_gate_encoder";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = JointCkpt;
    type Output = PccpVerdict;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: JointCkpt,
        args: &Args,
    ) -> Result<PccpVerdict, StageError> {
        run_pccp_gate(
            &args.lamquant_home,
            "encoder",
            &input.encoder_path,
            &args.change_id,
            &args.description,
            &args.author,
            &args.change_class,
            args.dry_run,
            args.no_promote,
        )
        .await
    }
}

pub struct LamquantPccpGateDecoder;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DecoderArgs {
    #[serde(default)]
    pub lamquant_home: String,
    #[serde(default = "default_change_id")]
    pub change_id: String,
    #[serde(default = "default_description_dec")]
    pub description: String,
    #[serde(default = "default_author")]
    pub author: String,
    #[serde(default = "default_change_class")]
    pub change_class: String,
    #[serde(default = "default_true")]
    pub dry_run: bool,
    #[serde(default = "default_true")]
    pub no_promote: bool,
}
fn default_description_dec() -> String {
    "(blut decoder gate)".into()
}

#[async_trait]
impl Stage for LamquantPccpGateDecoder {
    const NAME: &'static str = "lamquant_pccp_gate_decoder";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = JointCkpt;
    type Output = PccpVerdict;
    type Args = DecoderArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: JointCkpt,
        args: &DecoderArgs,
    ) -> Result<PccpVerdict, StageError> {
        run_pccp_gate(
            &args.lamquant_home,
            "decoder",
            &input.decoder_path,
            &args.change_id,
            &args.description,
            &args.author,
            &args.change_class,
            args.dry_run,
            args.no_promote,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pccp_gate(
    lamquant_home: &str,
    model: &str,
    candidate: &std::path::Path,
    change_id: &str,
    description: &str,
    author: &str,
    change_class: &str,
    dry_run: bool,
    no_promote: bool,
) -> Result<PccpVerdict, StageError> {
    let home = resolve_home(lamquant_home)?;
    let python = python_for(&home);
    let script = script_path(&home, &["ai_models", "pccp_gate.py"])?;
    if !candidate.exists() {
        return Err(StageError::BadInput(format!(
            "candidate ckpt not found: {}",
            candidate.display()
        )));
    }
    if change_id.is_empty()
        || change_id.contains('/')
        || change_id.contains('\\')
        || change_id.contains("..")
    {
        return Err(StageError::BadInput(format!(
            "change_id '{change_id}' must not contain path separators or '..'"
        )));
    }
    // R30: cap + scrub free-form text.
    for (label, v) in [
        ("change_id", change_id),
        ("description", description),
        ("author", author),
        ("change_class", change_class),
    ] {
        if v.len() > 512 {
            return Err(StageError::BadInput(format!(
                "{label} length {} > 512 byte cap",
                v.len()
            )));
        }
        if v.chars().any(|c| c.is_control() && c != '\t') {
            return Err(StageError::BadInput(format!(
                "{label} contains control characters (only tab allowed)"
            )));
        }
    }
    let mut cmd_args = vec![
        "--candidate".into(),
        candidate.display().to_string(),
        "--model".into(),
        model.to_string(),
        "--change-id".into(),
        change_id.to_string(),
        "--description".into(),
        description.to_string(),
        "--author".into(),
        author.to_string(),
        "--change-class".into(),
        change_class.to_string(),
    ];
    if dry_run {
        cmd_args.push("--dry-run".into());
    }
    if no_promote {
        cmd_args.push("--no-promote".into());
    }
    let inv = LamquantInvocation {
        python,
        script,
        cwd: home.clone(),
        args: cmd_args,
        env: vec![],
        expected_outputs: vec![],
        run_manifest_path: None,
    };
    let mut backend = LamquantBackend::new();
    backend
        .run(inv, None)
        .await
        .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
    let records_dir = home.join("pccp").join("verification_records");
    let preferred = records_dir.join(format!("{change_id}.json"));
    let mut verdict_path: Option<PathBuf> = None;
    for _ in 0..5 {
        if preferred.exists() {
            verdict_path = Some(preferred.clone());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let verdict_path = match verdict_path {
        Some(p) => p,
        None => {
            // Fallback to most recent json.
            let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
            if let Ok(read) = std::fs::read_dir(&records_dir) {
                for ent in read.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(meta) = ent.metadata() {
                            let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                            if best.as_ref().map_or(true, |(t, _)| mtime > *t) {
                                best = Some((mtime, p));
                            }
                        }
                    }
                }
            }
            best.map(|(_, p)| p).ok_or_else(|| {
                StageError::Backend(anyhow::anyhow!(
                    "gate ran but no verdict file appeared under {}",
                    records_dir.display()
                ))
            })?
        }
    };
    let body = std::fs::read_to_string(&verdict_path).map_err(|source| StageError::Io {
        path: verdict_path.clone(),
        source,
    })?;
    let parsed: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| StageError::Backend(anyhow::anyhow!("parse verdict: {e}")))?;
    let passed = parsed
        .get("passed")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| StageError::Backend(anyhow::anyhow!("verdict missing passed")))?;
    let record_id = parsed
        .get("change_id")
        .and_then(|v| v.as_str())
        .unwrap_or(change_id)
        .to_string();
    let content_hash = ContentHash::hash_file(&verdict_path).map_err(|source| StageError::Io {
        path: verdict_path.clone(),
        source,
    })?;
    Ok(PccpVerdict {
        gate_json_path: verdict_path,
        record_id,
        passed,
        candidate_path: candidate.to_path_buf(),
        model_name: model.to_string(),
        content_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn joint(td: &std::path::Path) -> JointCkpt {
        let enc = td.join("enc.ckpt");
        let dec = td.join("dec.ckpt");
        std::fs::write(&enc, b"e").unwrap();
        std::fs::write(&dec, b"d").unwrap();
        JointCkpt {
            encoder_path: enc,
            decoder_path: dec,
            content_hash: ContentHash::of_bytes(b""),
            final_loss: 0.0,
            tier: 3,
            preset: "test".into(),
        }
    }

    #[tokio::test]
    async fn encoder_rejects_missing_home() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantPccpGateEncoder
            .run(
                &ctx(td.path()),
                joint(td.path()),
                &Args {
                    lamquant_home: td.path().join("nope").display().to_string(),
                    change_id: default_change_id(),
                    description: default_description(),
                    author: default_author(),
                    change_class: default_change_class(),
                    dry_run: true,
                    no_promote: true,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn decoder_rejects_change_id_with_traversal() {
        let td = tempfile::tempdir().unwrap();
        let home = td.path().join("home");
        std::fs::create_dir_all(home.join("ai_models")).unwrap();
        std::fs::write(home.join("ai_models").join("pccp_gate.py"), "# stub").unwrap();
        let r = LamquantPccpGateDecoder
            .run(
                &ctx(td.path()),
                joint(td.path()),
                &DecoderArgs {
                    lamquant_home: home.display().to_string(),
                    change_id: "../etc/passwd".into(),
                    description: default_description_dec(),
                    author: default_author(),
                    change_class: default_change_class(),
                    dry_run: true,
                    no_promote: true,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
