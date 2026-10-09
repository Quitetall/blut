// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Candidate provider protocol (BLUT-ADR-0001, proposed), not OpenWarrant assurance.
//! The caller owns collector key custody and obtains the expected Dispatch from
//! its checked records. Receipt validation establishes neither permission nor
//! confinement, metered spend, independent review or human acceptance.
use super::{CacheHandle, ExecCtx, PlanError, PlanResult, PlanSpec, Registry, execute_plan};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Read,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const DOMAIN: &[u8] = b"blut-openwarrant-receipt-v1";
const MAX_PAYLOAD: usize = 2 * 1024 * 1024;
const MAX_FILES: usize = 4096;
const MAX_FILE: u64 = 64 * 1024 * 1024;
const MAX_TOTAL: u64 = 256 * 1024 * 1024;
const MAX_BINARY: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchBinding {
    pub warrant_ref: String,
    pub contract_digest: String,
    pub dispatch_digest: String,
    pub stage_id: String,
    pub attempt_id: String,
}
impl DispatchBinding {
    fn validate(&self) -> Result<(), ReceiptError> {
        for v in [
            &self.warrant_ref,
            &self.contract_digest,
            &self.dispatch_digest,
            &self.stage_id,
            &self.attempt_id,
        ] {
            if v.trim().is_empty() || v.len() > 4096 || v.contains('\0') {
                return Err(rejected("incomplete Dispatch binding"));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    CompletedWithWarnings,
    Failed,
    Cancelled,
    DeadlineExceeded,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}
/// The actual executor result and the signed observation of that same attempt.
/// An error may follow execution (for example, unavailable native evidence).
/// Do not automatically retry execution because receipt capture failed.
pub struct SealedRun {
    pub envelope: Envelope,
    pub result: Result<PlanResult, PlanError>,
}
/// Native signature/source validation only, never OpenWarrant Verified status.
pub struct VerifiedRun {
    pub binding: DispatchBinding,
    pub outcome: Outcome,
    pub run_id: String,
    pub receipt_digest: String,
    pub registry_digest: String,
    pub lineage_reference: String,
}
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("native-receipt-unavailable: {0}")]
    Unavailable(String),
    #[error("native-receipt-rejected: {0}")]
    Rejected(String),
}
fn rejected(v: impl Into<String>) -> ReceiptError {
    ReceiptError::Rejected(v.into())
}
fn unavailable(v: impl std::fmt::Display) -> ReceiptError {
    ReceiptError::Unavailable(v.to_string())
}
fn digest(b: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(b).to_hex())
}
fn message(payload: &[u8]) -> Result<Vec<u8>, ReceiptError> {
    if payload.is_empty() || payload.len() > MAX_PAYLOAD {
        return Err(rejected("payload byte budget"));
    }
    let mut m = Vec::with_capacity(DOMAIN.len() + 8 + payload.len());
    m.extend_from_slice(DOMAIN);
    m.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    m.extend_from_slice(payload);
    Ok(m)
}
fn canonical<T: Serialize>(v: &T) -> Result<Vec<u8>, ReceiptError> {
    serde_json::to_value(v)
        .map(|v| CacheHandle::canonical_json_bytes(&v))
        .map_err(unavailable)
}
fn registry_bytes(reg: &Registry) -> Result<Vec<u8>, ReceiptError> {
    canonical(&crate::cli::registry_export::build_manifest(reg))
}
fn now() -> Result<u64, ReceiptError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(unavailable)?
        .as_millis()
        .try_into()
        .map_err(unavailable)
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeFile {
    path: String,
    bytes: u64,
    digest: String,
    mode: u32,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    version: u16,
    binding: DispatchBinding,
    run_id: String,
    plan_digest: String,
    registry_digest: String,
    binary_digest: String,
    started_unix_ms: u64,
    ended_unix_ms: u64,
    outcome: Outcome,
    warnings: usize,
    stages: usize,
    native_files: Vec<NativeFile>,
}
fn hash_file(path: &Path, max: u64) -> Result<(u64, String, u32), ReceiptError> {
    let meta = fs::symlink_metadata(path).map_err(unavailable)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(rejected("native reference is not a regular file"));
    }
    if meta.len() > max {
        return Err(rejected("native file byte budget"));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(nix::libc::O_NOFOLLOW);
    }
    let mut f = options.open(path).map_err(unavailable)?;
    let mut hash = blake3::Hasher::new();
    let mut bytes = 0u64;
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf).map_err(unavailable)?;
        if n == 0 {
            break;
        }
        bytes = bytes
            .checked_add(n as u64)
            .ok_or_else(|| rejected("native byte overflow"))?;
        if bytes > max {
            return Err(rejected("native file byte budget"));
        }
        hash.update(&buf[..n]);
    }
    if bytes != meta.len() || f.metadata().map_err(unavailable)?.len() != bytes {
        return Err(rejected("native file changed during capture"));
    }
    #[cfg(unix)]
    let mode = {
        use std::os::unix::fs::PermissionsExt;
        f.metadata().map_err(unavailable)?.permissions().mode()
    };
    #[cfg(not(unix))]
    let mode = 0;
    Ok((bytes, format!("blake3:{}", hash.finalize().to_hex()), mode))
}
fn files(root: &Path) -> Result<Vec<NativeFile>, ReceiptError> {
    fn walk(
        root: &Path,
        dir: &Path,
        depth: usize,
        out: &mut Vec<NativeFile>,
        total: &mut u64,
    ) -> Result<(), ReceiptError> {
        if depth > 32 {
            return Err(rejected("native directory depth"));
        }
        let meta = fs::symlink_metadata(dir).map_err(unavailable)?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(rejected("native directory is not plain"));
        }
        for entry in fs::read_dir(dir).map_err(unavailable)? {
            let path = entry.map_err(unavailable)?.path();
            let meta = fs::symlink_metadata(&path).map_err(unavailable)?;
            if meta.is_dir() && !meta.file_type().is_symlink() {
                walk(root, &path, depth + 1, out, total)?;
                continue;
            }
            if out.len() >= MAX_FILES {
                return Err(rejected("native file count budget"));
            }
            let (bytes, hash, mode) = hash_file(&path, MAX_FILE)?;
            *total = total
                .checked_add(bytes)
                .ok_or_else(|| rejected("native byte overflow"))?;
            if *total > MAX_TOTAL {
                return Err(rejected("native total byte budget"));
            }
            let relative = path.strip_prefix(root).map_err(unavailable)?;
            let mut components = Vec::new();
            for component in relative.components() {
                let std::path::Component::Normal(name) = component else {
                    return Err(rejected("ambiguous native path"));
                };
                let name = name
                    .to_str()
                    .ok_or_else(|| rejected("non-UTF8 native path"))?;
                if name.contains('\\') || name.chars().any(char::is_control) {
                    return Err(rejected("ambiguous native path"));
                }
                components.push(name);
            }
            let rel = components.join("/");
            out.push(NativeFile {
                path: rel,
                bytes,
                digest: hash,
                mode,
            });
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, root, 0, &mut out, &mut 0)?;
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}
/// Compile and execute once in a fresh native job directory, then sign observed
/// native bytes. No key persistence, network request or authority act is added.
pub async fn execute_with_receipt(
    spec: &PlanSpec,
    reg: &Registry,
    ctx: ExecCtx,
    binding: DispatchBinding,
    signer: &SigningKey,
) -> Result<SealedRun, ReceiptError> {
    binding.validate()?;
    let plan = spec.compile(reg).map_err(|e| rejected(e.to_string()))?;
    let plan_bytes = spec.canonical_bytes();
    let registry_bytes = registry_bytes(reg)?;
    let executable = std::env::current_exe().map_err(unavailable)?;
    let (_, binary_digest, _) = hash_file(&executable, MAX_BINARY)?;
    let root = ctx.job_dir.clone();
    if let Some(parent) = root.parent() {
        fs::create_dir_all(parent).map_err(unavailable)?;
    }
    match fs::create_dir(&root) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(rejected(
                "native job directory already exists; execution not started",
            ));
        }
        Err(e) => return Err(unavailable(e)),
    }
    let inputs = root.join("_openwarrant_inputs");
    fs::create_dir(&inputs).map_err(unavailable)?;
    let run_id = uuid::Uuid::new_v4().to_string();
    fs::write(inputs.join("plan.json"), &plan_bytes).map_err(unavailable)?;
    fs::write(inputs.join("registry.json"), &registry_bytes).map_err(unavailable)?;
    fs::write(
        inputs.join("binding.json"),
        canonical(&serde_json::json!({"run_id":run_id,"dispatch":binding}))?,
    )
    .map_err(unavailable)?;
    let started = now()?;
    let result = execute_plan(plan, ctx).await;
    let ended = now()?;
    if ended < started {
        return Err(unavailable("wall clock moved backwards after execution"));
    }
    let (outcome, warnings, stages) = match &result {
        Ok(r) => (
            if r.warnings.is_empty() {
                Outcome::Completed
            } else {
                Outcome::CompletedWithWarnings
            },
            r.warnings.len(),
            r.n_stages,
        ),
        Err(PlanError::Cancelled) => (Outcome::Cancelled, 0, 0),
        Err(PlanError::DeadlineExceeded { .. }) => (Outcome::DeadlineExceeded, 0, 0),
        Err(_) => (Outcome::Failed, 0, 0),
    };
    let payload = Payload {
        version: 1,
        binding,
        run_id,
        plan_digest: digest(&plan_bytes),
        registry_digest: digest(&registry_bytes),
        binary_digest,
        started_unix_ms: started,
        ended_unix_ms: ended,
        outcome,
        warnings,
        stages,
        native_files: files(&root)?,
    };
    let payload = serde_json::to_vec(&payload).map_err(unavailable)?;
    let signature = signer.sign(&message(&payload)?).to_bytes().to_vec();
    Ok(SealedRun {
        envelope: Envelope { payload, signature },
        result,
    })
}
/// Verify exact original signed bytes against a separately trusted key and
/// caller-established execution basis. `producer_executable` is the retained
/// exact producer binary, not a path or digest borrowed from the receipt.
/// Native job files must remain accessible. Unavailable evidence is UNKNOWN.
pub fn verify(
    envelope: &Envelope,
    trusted_key: &VerifyingKey,
    expected: &DispatchBinding,
    spec: &PlanSpec,
    reg: &Registry,
    job: &Path,
    producer_executable: &Path,
) -> Result<VerifiedRun, ReceiptError> {
    expected.validate()?;
    let signature =
        Signature::from_slice(&envelope.signature).map_err(|_| rejected("signature shape"))?;
    let signed_message = message(&envelope.payload)?;
    trusted_key
        .verify_strict(&signed_message, &signature)
        .map_err(|_| rejected("untrusted key or altered signature/payload"))?;
    let payload: Payload = serde_json::from_slice(&envelope.payload)
        .map_err(|e| rejected(format!("typed payload: {e}")))?;
    if payload.version != 1
        || payload.binding != *expected
        || uuid::Uuid::parse_str(&payload.run_id).is_err()
        || payload.ended_unix_ms < payload.started_unix_ms
    {
        return Err(rejected(
            "receipt version, Dispatch binding or run identity",
        ));
    }
    match payload.outcome {
        Outcome::Completed if payload.warnings != 0 || payload.stages == 0 => {
            return Err(rejected("contradictory completed result"));
        }
        Outcome::CompletedWithWarnings if payload.warnings == 0 || payload.stages == 0 => {
            return Err(rejected("contradictory warning result"));
        }
        Outcome::Failed | Outcome::Cancelled | Outcome::DeadlineExceeded
            if payload.stages != 0 || payload.warnings != 0 =>
        {
            return Err(rejected("contradictory unsuccessful result"));
        }
        _ => {}
    }
    if payload.plan_digest != digest(&spec.canonical_bytes())
        || payload.registry_digest != digest(&registry_bytes(reg)?)
        || payload.binary_digest != hash_file(producer_executable, MAX_BINARY)?.1
    {
        return Err(rejected("plan, registry or producer binary changed"));
    }
    if payload.native_files != files(job)? {
        return Err(rejected("native job files changed or mixed"));
    }
    let expected_binding =
        canonical(&serde_json::json!({"run_id":payload.run_id,"dispatch":expected}))?;
    if fs::read(job.join("_openwarrant_inputs/binding.json")).map_err(unavailable)?
        != expected_binding
        || fs::read(job.join("_openwarrant_inputs/plan.json")).map_err(unavailable)?
            != spec.canonical_bytes()
        || fs::read(job.join("_openwarrant_inputs/registry.json")).map_err(unavailable)?
            != registry_bytes(reg)?
    {
        return Err(rejected("native input records changed or mixed"));
    }
    if matches!(
        payload.outcome,
        Outcome::Completed | Outcome::CompletedWithWarnings
    ) && !payload
        .native_files
        .iter()
        .any(|f| f.path == "status.jsonl" && f.bytes > 0)
    {
        return Err(rejected("completed receipt has no native lifecycle stream"));
    }
    let mut receipt_identity = signed_message;
    receipt_identity.extend_from_slice(&envelope.signature);
    Ok(VerifiedRun {
        binding: payload.binding,
        outcome: payload.outcome,
        run_id: payload.run_id,
        receipt_digest: digest(&receipt_identity),
        registry_digest: payload.registry_digest,
        lineage_reference: job.join("status.jsonl").display().to_string(),
    })
}
