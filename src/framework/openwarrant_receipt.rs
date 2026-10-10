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
    path::{Path, PathBuf},
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
#[derive(Debug, Serialize)]
pub struct VerifiedRun {
    pub binding: DispatchBinding,
    pub outcome: Outcome,
    pub run_id: String,
    pub receipt_digest: String,
    pub registry_digest: String,
    /// Verified native files, relative to the caller-established job directory.
    pub native_files: Vec<NativeFileReference>,
    /// The retained nonempty lifecycle stream, when available. No missing path
    /// is invented for attempts that fail before a lifecycle writer starts.
    pub lineage_reference: Option<NativeFileReference>,
}
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("native-receipt-unavailable: {0}")]
    Unavailable(String),
    #[error("native-receipt-rejected: {0}")]
    Rejected(String),
}
/// Paths established by the caller, never selected by receipt contents.
/// Transport limits: 8 MiB + 1 KiB envelope JSON, 32 KiB binding JSON,
/// 2 MiB plan JSON. These limits apply before deserialization/allocation.
#[derive(Debug)]
pub struct VerificationFiles {
    pub receipt: PathBuf,
    pub binding: PathBuf,
    pub plan: PathBuf,
    pub job: PathBuf,
    pub producer_executable: PathBuf,
}

fn read_transport(path: &Path, max: u64) -> Result<Vec<u8>, ReceiptError> {
    let meta = fs::symlink_metadata(path).map_err(unavailable)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(rejected("transport input is not a regular file"));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Refuse final-component links; do not block on a swapped FIFO.
        options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(unavailable)?;
    let opened = file.metadata().map_err(unavailable)?;
    if !opened.is_file() || opened.len() > max {
        return Err(rejected("transport input type or byte budget"));
    }
    let mut bytes = Vec::new();
    file.take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(unavailable)?;
    if bytes.len() as u64 > max || bytes.len() as u64 != opened.len() {
        return Err(rejected("transport input byte budget or changed length"));
    }
    Ok(bytes)
}

/// Read-only file transport for the native verifier. Missing evidence remains
/// unavailable; malformed or oversized input is rejected. No plan is executed.
pub fn verify_files(
    request: &VerificationFiles,
    trusted_key: &VerifyingKey,
    reg: &Registry,
) -> Result<VerifiedRun, ReceiptError> {
    fn decode<T: serde::de::DeserializeOwned>(path: &Path, max: u64) -> Result<T, ReceiptError> {
        serde_json::from_slice(&read_transport(path, max)?)
            .map_err(|e| rejected(format!("transport JSON: {e}")))
    }
    let envelope: Envelope = decode(&request.receipt, 8 * 1024 * 1024 + 1024)?;
    let binding: DispatchBinding = decode(&request.binding, 32 * 1024)?;
    let spec: PlanSpec = decode(&request.plan, MAX_PAYLOAD as u64)?;
    verify(
        &envelope,
        trusted_key,
        &binding,
        &spec,
        reg,
        &request.job,
        &request.producer_executable,
    )
}

/// Load exactly 32 raw bytes of a caller-trusted Ed25519 PUBLIC key.
/// This function never reads a private key or selects trust from a receipt.
pub fn read_verification_key(path: &Path) -> Result<VerifyingKey, ReceiptError> {
    let bytes: [u8; 32] = read_transport(path, 32)?
        .try_into()
        .map_err(|_| rejected("public key must contain exactly 32 raw bytes"))?;
    VerifyingKey::from_bytes(&bytes).map_err(|_| rejected("invalid public key"))
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
/// Resolve the native catalog identity from a caller-trusted registry, without
/// borrowing identity from a receipt or a schema/version pin. Uses exactly the
/// manifest bytes and BLAKE3 domain used by the native producer and verifier.
pub fn registry_digest(reg: &Registry) -> Result<String, ReceiptError> {
    registry_bytes(reg).map(|bytes| digest(&bytes))
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
pub struct NativeFileReference {
    pub path: String,
    pub bytes: u64,
    pub digest: String,
    pub mode: u32,
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
    native_files: Vec<NativeFileReference>,
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
fn files(root: &Path) -> Result<Vec<NativeFileReference>, ReceiptError> {
    fn walk(
        root: &Path,
        dir: &Path,
        depth: usize,
        out: &mut Vec<NativeFileReference>,
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
            out.push(NativeFileReference {
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
    let (payload, signed_message) = verify_basis(envelope, trusted_key, expected, spec, reg)?;
    if payload.binary_digest != hash_file(producer_executable, MAX_BINARY)?.1 {
        return Err(rejected("plan, registry or producer binary changed"));
    }
    let observed = files(job)?;
    let mut records = std::collections::BTreeMap::new();
    for name in ["binding", "plan", "registry"] {
        let path = format!("_openwarrant_inputs/{name}.json");
        records.insert(path.clone(), read_transport(&job.join(path), MAX_FILE)?);
    }
    finish_verification(
        envelope,
        payload,
        signed_message,
        expected,
        spec,
        reg,
        observed,
        records,
    )
}

fn verify_basis(
    envelope: &Envelope,
    trusted_key: &VerifyingKey,
    expected: &DispatchBinding,
    spec: &PlanSpec,
    reg: &Registry,
) -> Result<(Payload, Vec<u8>), ReceiptError> {
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
    {
        return Err(rejected("plan, registry or producer binary changed"));
    }
    Ok((payload, signed_message))
}
#[allow(clippy::too_many_arguments)]
fn finish_verification(
    envelope: &Envelope,
    payload: Payload,
    signed_message: Vec<u8>,
    expected: &DispatchBinding,
    spec: &PlanSpec,
    reg: &Registry,
    observed: Vec<NativeFileReference>,
    records: std::collections::BTreeMap<String, Vec<u8>>,
) -> Result<VerifiedRun, ReceiptError> {
    if payload.native_files != observed {
        return Err(rejected("native job files changed or mixed"));
    }
    let expected_binding =
        canonical(&serde_json::json!({"run_id":payload.run_id,"dispatch":expected}))?;
    if records
        .get("_openwarrant_inputs/binding.json")
        .ok_or_else(|| rejected("missing native binding record"))?
        != &expected_binding
        || records
            .get("_openwarrant_inputs/plan.json")
            .ok_or_else(|| rejected("missing native plan record"))?
            != &spec.canonical_bytes()
        || records
            .get("_openwarrant_inputs/registry.json")
            .ok_or_else(|| rejected("missing native registry record"))?
            != &registry_bytes(reg)?
    {
        return Err(rejected("native input records changed or mixed"));
    }
    let lineage_reference = payload
        .native_files
        .iter()
        .find(|f| f.path == "status.jsonl" && f.bytes > 0)
        .cloned();
    if matches!(
        payload.outcome,
        Outcome::Completed | Outcome::CompletedWithWarnings
    ) && lineage_reference.is_none()
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
        native_files: payload.native_files,
        lineage_reference,
    })
}

/// Explicit sealed-input transport. The caller supplies independent trust and
/// complete captured job metadata; no producer is executed or key discovered.
#[derive(Debug)]
pub struct SealedVerificationFiles {
    pub receipt: PathBuf,
    pub binding: PathBuf,
    pub inputs: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedInputs {
    schema: String,
    public_key: PathBuf,
    plan: PathBuf,
    producer_executable: PathBuf,
    files: Vec<SealedNativeFile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SealedNativeFile {
    path: String,
    mode: u32,
    descriptor: PathBuf,
}
/// Original signature, expected plan/catalog and frozen native bytes must all
/// agree. The source modes are separately captured caller metadata, not the
/// descriptor's execution mode. Transport validation never grants assurance.
pub fn verify_sealed_files(
    request: &SealedVerificationFiles,
    reg: &Registry,
) -> Result<VerifiedRun, ReceiptError> {
    let inputs: SealedInputs =
        serde_json::from_slice(&read_transport(&request.inputs, 8 * 1024 * 1024)?)
            .map_err(|e| rejected(format!("sealed inputs JSON: {e}")))?;
    if inputs.schema != "blut/openwarrant-sealed-inputs/v1"
        || inputs.files.is_empty()
        || inputs.files.len() > MAX_FILES
    {
        return Err(rejected("sealed input schema or native file-count budget"));
    }
    let key_bytes: [u8; 32] = read_sealed(&inputs.public_key, 32)?
        .try_into()
        .map_err(|_| rejected("public key must contain exactly 32 raw bytes"))?;
    let key = VerifyingKey::from_bytes(&key_bytes).map_err(|_| rejected("invalid public key"))?;
    let spec: PlanSpec = serde_json::from_slice(&read_sealed(&inputs.plan, MAX_PAYLOAD as u64)?)
        .map_err(|e| rejected(format!("sealed plan JSON: {e}")))?;
    let expected: DispatchBinding =
        serde_json::from_slice(&read_transport(&request.binding, 32 * 1024)?)
            .map_err(|e| rejected(format!("binding JSON: {e}")))?;
    let envelope: Envelope =
        serde_json::from_slice(&read_transport(&request.receipt, 8 * 1024 * 1024 + 1024)?)
            .map_err(|e| rejected(format!("receipt JSON: {e}")))?;
    let (payload, signed_message) = verify_basis(&envelope, &key, &expected, &spec, reg)?;
    if payload.binary_digest != digest(&read_sealed(&inputs.producer_executable, MAX_BINARY)?) {
        return Err(rejected("plan, registry or producer binary changed"));
    }
    let mut paths = std::collections::BTreeSet::new();
    let mut observed = Vec::with_capacity(inputs.files.len());
    let mut records = std::collections::BTreeMap::new();
    let mut total = 0u64;
    for file in inputs.files {
        if file.path.is_empty()
            || file.path.len() > 4096
            || file.path.contains('\\')
            || file.path.chars().any(char::is_control)
            || file.path.starts_with('/')
            || file
                .path
                .split('/')
                .any(|c| c.is_empty() || c == "." || c == "..")
            || !paths.insert(file.path.clone())
            || file.mode & 0o170000 != 0o100000
            || file.mode & 0o022 != 0
        {
            return Err(rejected(
                "invalid, duplicate or writable native source metadata",
            ));
        }
        let bytes = read_sealed(&file.descriptor, MAX_FILE)?;
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| rejected("native byte overflow"))?;
        if total > MAX_TOTAL {
            return Err(rejected("native total byte budget"));
        }
        observed.push(NativeFileReference {
            path: file.path.clone(),
            bytes: bytes.len() as u64,
            digest: digest(&bytes),
            mode: file.mode,
        });
        if [
            "_openwarrant_inputs/binding.json",
            "_openwarrant_inputs/plan.json",
            "_openwarrant_inputs/registry.json",
        ]
        .contains(&file.path.as_str())
        {
            records.insert(file.path, bytes);
        }
    }
    observed.sort_by(|a, b| a.path.cmp(&b.path));
    finish_verification(
        &envelope,
        payload,
        signed_message,
        &expected,
        &spec,
        reg,
        observed,
        records,
    )
}
#[cfg(target_os = "linux")]
fn read_sealed(path: &Path, max: u64) -> Result<Vec<u8>, ReceiptError> {
    use rustix::fs::{SealFlags, fcntl_get_seals};
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let text = path
        .to_str()
        .ok_or_else(|| rejected("descriptor is not UTF-8"))?;
    let pieces: Vec<_> = text.split('/').collect();
    let canonical = |s: &str| s.parse::<u32>().ok().filter(|v| v.to_string() == s);
    let parent = rustix::process::getppid()
        .map(|p| p.as_raw_pid() as u32)
        .ok_or_else(|| unavailable("native parent process unavailable"))?;
    if pieces.len() != 5
        || !pieces[0].is_empty()
        || pieces[1] != "proc"
        || pieces[3] != "fd"
        || canonical(pieces[2]) != Some(parent)
        || canonical(pieces[4]).is_none()
    {
        return Err(rejected("canonical immediate-parent descriptor required"));
    }
    // This is the explicit procfs transport. Ordinary file readers retain
    // O_NOFOLLOW; arbitrary links cannot substitute for a sealed kernel object.
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NONBLOCK)
        .open(path)
        .map_err(unavailable)?;
    let m = f.metadata().map_err(unavailable)?;
    if !m.is_file() || m.mode() & 0o111 != 0 || m.len() > max {
        return Err(rejected(
            "sealed regular non-executable input within byte budget required",
        ));
    }
    let required =
        SealFlags::WRITE | SealFlags::GROW | SealFlags::SHRINK | SealFlags::EXEC | SealFlags::SEAL;
    if !fcntl_get_seals(&f)
        .map_err(|_| rejected("native descriptor is not sealable"))?
        .contains(required)
    {
        return Err(rejected("native descriptor lacks required kernel seals"));
    }
    let mut bytes = Vec::new();
    f.take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(unavailable)?;
    if bytes.len() as u64 != m.len() || bytes.len() as u64 > max {
        return Err(rejected("sealed input byte budget"));
    }
    Ok(bytes)
}
#[cfg(not(target_os = "linux"))]
fn read_sealed(_: &Path, _: u64) -> Result<Vec<u8>, ReceiptError> {
    Err(unavailable(
        "sealed native input transport unsupported on this platform",
    ))
}
