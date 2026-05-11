//! Stage — `lamquant_pccp_gate_snn`.
//!
//! Post-train PCCP audit-trail closer for the SNN paradigm. Takes
//! the produced `SnnCkpt`, shells out to
//! `ai_models/pccp_gate.py --candidate <ckpt> --model snn ...`,
//! reads the resulting verification record, returns a typed
//! `PccpVerdict`.
//!
//! Passthrough semantics: the verdict is always produced — never
//! short-circuits the plan. Recipe authors decide what to do on
//! `passed = false` (typically: emit a notification stage, leave
//! the ckpt unpromoted, halt downstream stages). This stage's
//! contract is "tell the truth"; it does NOT enforce.
//!
//! Deterministic: same ckpt bytes + same gate config → same
//! verdict. The gate's own scoring is content-addressed.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{PccpVerdict, SnnCkpt};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::lamquant_backend::{
    default_lamquant_home, resolve_lamquant_python, LamquantBackend, LamquantInvocation,
};

pub struct LamquantPccpGateSnn;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default)]
    pub lamquant_home: String,
    /// PCCP change identifier (e.g. "PCCP-CHG-2026-05-11-001").
    /// Defaults to a dry-run-style sentinel if unset.
    #[serde(default = "default_change_id")]
    pub change_id: String,
    #[serde(default = "default_description")]
    pub description: String,
    #[serde(default = "default_author")]
    pub author: String,
    /// Modification class per 01-modifications.md (e.g. "A.1").
    #[serde(default = "default_change_class")]
    pub change_class: String,
    /// `--dry-run` — evaluate without writing verdict log.
    #[serde(default)]
    pub dry_run: bool,
    /// `--no-promote` — on PASS, skip registry + CHANGELOG writes.
    #[serde(default)]
    pub no_promote: bool,
}

fn default_change_id() -> String {
    "PCCP-CHG-DRYRUN".into()
}
fn default_description() -> String {
    "(no description provided)".into()
}
fn default_author() -> String {
    "BLUT".into()
}
fn default_change_class() -> String {
    "A.1".into()
}

#[async_trait]
impl Stage for LamquantPccpGateSnn {
    const NAME: &'static str = "lamquant_pccp_gate_snn";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = SnnCkpt;
    type Output = PccpVerdict;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: SnnCkpt,
        args: &Args,
    ) -> Result<PccpVerdict, StageError> {
        let lamquant_home_raw = if args.lamquant_home.is_empty() {
            default_lamquant_home()
        } else {
            PathBuf::from(&args.lamquant_home)
        };
        let lamquant_home = std::fs::canonicalize(&lamquant_home_raw).map_err(|e| {
            StageError::BadInput(format!(
                "lamquant_home not found: {} ({e})",
                lamquant_home_raw.display()
            ))
        })?;
        let python = resolve_lamquant_python(&lamquant_home);
        let script = lamquant_home.join("ai_models").join("pccp_gate.py");
        if !script.exists() {
            return Err(StageError::BadInput(format!(
                "pccp_gate.py not found: {}",
                script.display()
            )));
        }
        if !input.path.exists() {
            return Err(StageError::BadInput(format!(
                "candidate ckpt not found: {}",
                input.path.display()
            )));
        }

        let mut cmd_args: Vec<String> = vec![
            "--candidate".into(),
            input.path.display().to_string(),
            "--model".into(),
            "snn".into(),
            "--change-id".into(),
            args.change_id.clone(),
            "--description".into(),
            args.description.clone(),
            "--author".into(),
            args.author.clone(),
            "--change-class".into(),
            args.change_class.clone(),
        ];
        if args.dry_run {
            cmd_args.push("--dry-run".into());
        }
        if args.no_promote {
            cmd_args.push("--no-promote".into());
        }

        let inv = LamquantInvocation {
            python,
            script,
            cwd: lamquant_home.clone(),
            args: cmd_args,
            env: vec![],
            // Don't pre-assert the verdict file path; the gate
            // script names it after `change_id` and we read the
            // resulting `pccp/verification_records/` dir to find it.
            // PASS / FAIL verdicts are tracked via the resulting
            // JSON file's `passed` field, not via exit code (the
            // gate returns 0 on FAIL too — we want the verdict).
            expected_outputs: vec![],
            run_manifest_path: None,
        };

        let mut backend = LamquantBackend::new();
        let _run = backend
            .run(inv, None)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        // Resolve the verdict file. The gate writes:
        //   pccp/verification_records/<change_id>.json
        // If the change_id has unsafe characters we punt to the
        // most-recently-modified file in the dir (defensive).
        let records_dir = lamquant_home.join("pccp").join("verification_records");
        let preferred = records_dir.join(format!("{}.json", args.change_id));
        let verdict_path = if preferred.exists() {
            preferred
        } else {
            most_recent_json(&records_dir)?.ok_or_else(|| {
                StageError::Backend(anyhow::anyhow!(
                    "gate ran successfully but no verdict file appeared under {}",
                    records_dir.display()
                ))
            })?
        };

        let body = std::fs::read_to_string(&verdict_path).map_err(|source| StageError::Io {
            path: verdict_path.clone(),
            source,
        })?;
        let parsed: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
            StageError::Backend(anyhow::anyhow!(
                "parse verdict JSON at {}: {e}",
                verdict_path.display()
            ))
        })?;
        let passed = parsed
            .get("passed")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| {
                StageError::Backend(anyhow::anyhow!(
                    "verdict JSON at {} missing `passed` bool",
                    verdict_path.display()
                ))
            })?;
        let record_id = parsed
            .get("change_id")
            .and_then(|v| v.as_str())
            .unwrap_or(&args.change_id)
            .to_string();

        let content_hash = ContentHash::hash_file(&verdict_path).map_err(|source| {
            StageError::Io {
                path: verdict_path.clone(),
                source,
            }
        })?;

        Ok(PccpVerdict {
            gate_json_path: verdict_path,
            record_id,
            passed,
            candidate_path: input.path,
            model_name: "snn".into(),
            content_hash,
        })
    }
}

/// Pick the newest `*.json` file in `dir` by mtime. Used as a
/// fallback when the change_id-named file isn't where we expect.
fn most_recent_json(dir: &std::path::Path) -> Result<Option<PathBuf>, StageError> {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(StageError::Io {
                path: dir.to_path_buf(),
                source,
            })
        }
    };
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for ent in read {
        let ent = ent.map_err(|source| StageError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let meta = ent.metadata().map_err(|source| StageError::Io {
            path: p.clone(),
            source,
        })?;
        let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        if best.as_ref().map_or(true, |(t, _)| mtime > *t) {
            best = Some((mtime, p));
        }
    }
    Ok(best.map(|(_, p)| p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::lamquant::stat_fingerprint;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn snn(td: &std::path::Path) -> SnnCkpt {
        let p = td.join("snn.pt");
        std::fs::write(&p, b"fake-weights").unwrap();
        SnnCkpt {
            path: p.clone(),
            content_hash: stat_fingerprint(b"lamquant.ckpt.snn", &p).unwrap(),
            head_size_kb: 4.0,
            final_loss: 0.1,
        }
    }

    #[tokio::test]
    async fn rejects_missing_lamquant_home() {
        let td = tempfile::tempdir().unwrap();
        let r = LamquantPccpGateSnn
            .run(
                &ctx(td.path()),
                snn(td.path()),
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
    async fn rejects_missing_candidate() {
        let td = tempfile::tempdir().unwrap();
        let home = td.path().join("home");
        std::fs::create_dir_all(home.join("ai_models")).unwrap();
        std::fs::write(home.join("ai_models").join("pccp_gate.py"), "# stub").unwrap();
        let bogus_snn = SnnCkpt {
            path: td.path().join("ghost.pt"),
            content_hash: ContentHash::of_bytes(b""),
            head_size_kb: 0.0,
            final_loss: 0.0,
        };
        let r = LamquantPccpGateSnn
            .run(
                &ctx(td.path()),
                bogus_snn,
                &Args {
                    lamquant_home: home.display().to_string(),
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

    #[test]
    fn most_recent_json_returns_none_for_missing_dir() {
        let td = tempfile::tempdir().unwrap();
        let r = most_recent_json(&td.path().join("nope")).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn most_recent_json_picks_newest() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("a.json"), "{}").unwrap();
        // Tiny sleep so mtime sorts deterministically across files.
        std::thread::sleep(std::time::Duration::from_millis(10));
        std::fs::write(td.path().join("b.json"), "{}").unwrap();
        let r = most_recent_json(td.path()).unwrap().unwrap();
        assert!(r.ends_with("b.json"));
    }
}
