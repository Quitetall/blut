// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use serde::{Deserialize, Serialize};

use crate::model::Capability;

pub const PLUGIN_PROTOCOL_VERSION: u32 = 2;
const CONTROL_MAGIC: &[u8; 4] = b"BPC2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutableDigestAlgorithm {
    /// BLAKE3 derive-key mode with context `blut.plugin-executable.v1`.
    Blake3DomainV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SignatureAlgorithm {
    Ed25519,
}

pub fn executable_digest(
    algorithm: ExecutableDigestAlgorithm,
    executable_bytes: &[u8],
) -> [u8; 32] {
    match algorithm {
        ExecutableDigestAlgorithm::Blake3DomainV1 => {
            let mut hasher = blake3::Hasher::new_derive_key("blut.plugin-executable.v1");
            hasher.update(executable_bytes);
            *hasher.finalize().as_bytes()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeardownPolicy {
    GracefulThenKill,
    ImmediateKill,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessContract {
    pub startup_deadline_millis: u64,
    pub request_deadline_millis: u64,
    pub heartbeat_interval_millis: u64,
    pub heartbeat_grace_millis: u64,
    pub teardown_deadline_millis: u64,
    pub kill_grace_millis: u64,
    pub max_inflight: u32,
    pub max_frame_bytes: u32,
    pub teardown: TeardownPolicy,
}

impl ProcessContract {
    pub fn is_valid(&self) -> bool {
        self.startup_deadline_millis > 0
            && self.request_deadline_millis > 0
            && self.heartbeat_interval_millis > 0
            && self.heartbeat_grace_millis >= self.heartbeat_interval_millis
            && self.teardown_deadline_millis > 0
            && self.kill_grace_millis <= self.teardown_deadline_millis
            && self.max_inflight > 0
            && self.max_frame_bytes >= 64
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginManifest {
    pub protocol_version: u32,
    pub plugin_id: String,
    pub executable_digest_algorithm: ExecutableDigestAlgorithm,
    pub executable_digest: [u8; 32],
    pub capabilities: Vec<Capability>,
    pub process: ProcessContract,
    pub signature_algorithm: SignatureAlgorithm,
    /// Stable verifier key identifier; key resolution is a supervisor policy.
    pub signing_key_id: String,
    pub signature: Vec<u8>,
}

#[derive(Serialize)]
struct UnsignedPluginManifest<'a> {
    protocol_version: u32,
    plugin_id: &'a str,
    executable_digest_algorithm: ExecutableDigestAlgorithm,
    executable_digest: [u8; 32],
    capabilities: &'a [Capability],
    process: &'a ProcessContract,
    signature_algorithm: SignatureAlgorithm,
    signing_key_id: &'a str,
}

impl PluginManifest {
    pub fn normalize(&mut self) -> Result<(), PluginError> {
        self.capabilities.sort_unstable();
        self.capabilities.dedup();
        if self.protocol_version != PLUGIN_PROTOCOL_VERSION {
            return Err(PluginError::ProtocolVersion(self.protocol_version));
        }
        if self.plugin_id.is_empty()
            || self.signing_key_id.is_empty()
            || self.signature.len() != 64
            || self.capabilities.is_empty()
            || self.capabilities.iter().any(|item| item.0.is_empty())
            || !self.process.is_valid()
        {
            return Err(PluginError::InvalidManifest);
        }
        Ok(())
    }

    /// Canonical signature preimage. The signature itself is excluded to avoid
    /// circular signing; capabilities are normalized before serialization.
    pub fn unsigned_signing_bytes(&self) -> Result<Vec<u8>, PluginError> {
        let mut normalized = self.clone();
        normalized.capabilities.sort_unstable();
        normalized.capabilities.dedup();
        if normalized.protocol_version != PLUGIN_PROTOCOL_VERSION
            || normalized.plugin_id.is_empty()
            || normalized.signing_key_id.is_empty()
            || normalized.capabilities.is_empty()
            || normalized.capabilities.iter().any(|item| item.0.is_empty())
            || !normalized.process.is_valid()
        {
            return Err(PluginError::InvalidManifest);
        }
        postcard::to_allocvec(&UnsignedPluginManifest {
            protocol_version: normalized.protocol_version,
            plugin_id: &normalized.plugin_id,
            executable_digest_algorithm: normalized.executable_digest_algorithm,
            executable_digest: normalized.executable_digest,
            capabilities: &normalized.capabilities,
            process: &normalized.process,
            signature_algorithm: normalized.signature_algorithm,
            signing_key_id: &normalized.signing_key_id,
        })
        .map_err(|_| PluginError::MalformedFrame)
    }

    pub fn signing_digest(&self) -> Result<[u8; 32], PluginError> {
        let bytes = self.unsigned_signing_bytes()?;
        let mut hasher = blake3::Hasher::new_derive_key("blut.plugin-manifest.v1");
        hasher.update(&bytes);
        Ok(*hasher.finalize().as_bytes())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginRequest {
    pub request_id: u64,
    pub invocation_id: [u8; 32],
    pub capability: Capability,
    pub payload_content_id: [u8; 32],
    pub shared_memory_lease: Option<String>,
    /// Absolute supervisor-clock deadline. Zero is never interpreted as infinite.
    pub deadline_millis: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginFailure {
    pub domain: String,
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginResponse {
    pub request_id: u64,
    pub output_content_id: Option<[u8; 32]>,
    pub receipt: Vec<u8>,
    pub failure: Option<PluginFailure>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginLifecycle {
    Spawned,
    Handshaking,
    Ready,
    Draining,
    Terminated,
}

impl PluginLifecycle {
    pub const fn permits(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Spawned, Self::Handshaking | Self::Terminated)
                | (
                    Self::Handshaking,
                    Self::Ready | Self::Draining | Self::Terminated
                )
                | (Self::Ready, Self::Draining | Self::Terminated)
                | (Self::Draining, Self::Terminated)
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginControlFrame {
    Hello {
        manifest: PluginManifest,
        supervisor_nonce: [u8; 32],
        deadline_millis: u64,
    },
    Ready {
        supervisor_nonce: [u8; 32],
        process_id: u64,
    },
    Invoke(PluginRequest),
    Complete(PluginResponse),
    Heartbeat {
        sequence: u64,
        monotonic_millis: u64,
    },
    Cancel {
        request_id: u64,
        deadline_millis: u64,
    },
    Shutdown {
        reason: String,
        deadline_millis: u64,
    },
    Ack {
        request_id: Option<u64>,
        lifecycle: PluginLifecycle,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PluginControlLimits {
    pub max_frame_bytes: usize,
    pub max_receipt_bytes: usize,
    pub max_signature_bytes: usize,
}

impl Default for PluginControlLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 1024 * 1024,
            max_receipt_bytes: 256 * 1024,
            max_signature_bytes: 16 * 1024,
        }
    }
}

impl PluginControlLimits {
    pub fn for_process(process: &ProcessContract) -> Self {
        let defaults = Self::default();
        let max_frame_bytes = process.max_frame_bytes as usize;
        Self {
            max_frame_bytes,
            max_receipt_bytes: defaults.max_receipt_bytes.min(max_frame_bytes),
            max_signature_bytes: defaults.max_signature_bytes.min(max_frame_bytes),
        }
    }
}

impl PluginControlFrame {
    pub fn to_control_bytes(&self) -> Result<Vec<u8>, PluginError> {
        self.to_control_bytes_with_limits(PluginControlLimits::default())
    }

    pub fn to_control_bytes_with_limits(
        &self,
        limits: PluginControlLimits,
    ) -> Result<Vec<u8>, PluginError> {
        validate_frame(self, limits)?;
        let mut bytes = Vec::from(CONTROL_MAGIC.as_slice());
        bytes.extend(postcard::to_allocvec(self).map_err(|_| PluginError::MalformedFrame)?);
        if bytes.len() > limits.max_frame_bytes {
            return Err(PluginError::FrameTooLarge);
        }
        Ok(bytes)
    }

    pub fn from_control_bytes(
        bytes: &[u8],
        limits: PluginControlLimits,
    ) -> Result<Self, PluginError> {
        if bytes.len() > limits.max_frame_bytes {
            return Err(PluginError::FrameTooLarge);
        }
        let body = bytes
            .strip_prefix(CONTROL_MAGIC)
            .ok_or(PluginError::BadControlMagic)?;
        let (frame, remainder): (Self, &[u8]) =
            postcard::take_from_bytes(body).map_err(|_| PluginError::MalformedFrame)?;
        if !remainder.is_empty() {
            return Err(PluginError::MalformedFrame);
        }
        validate_frame(&frame, limits)?;
        Ok(frame)
    }
}

fn validate_frame(
    frame: &PluginControlFrame,
    limits: PluginControlLimits,
) -> Result<(), PluginError> {
    match frame {
        PluginControlFrame::Hello {
            manifest,
            deadline_millis,
            ..
        } => {
            let mut normalized = manifest.clone();
            normalized.normalize()?;
            if normalized != *manifest || manifest.signature.len() > limits.max_signature_bytes {
                return Err(PluginError::InvalidManifest);
            }
            if manifest.process.max_frame_bytes as usize > limits.max_frame_bytes {
                return Err(PluginError::FrameTooLarge);
            }
            if *deadline_millis == 0 {
                return Err(PluginError::InvalidDeadline);
            }
        }
        PluginControlFrame::Ready { process_id, .. } if *process_id == 0 => {
            return Err(PluginError::InvalidLifecycle);
        }
        PluginControlFrame::Invoke(request) => validate_request(request)?,
        PluginControlFrame::Complete(response) => {
            validate_response(response, limits.max_receipt_bytes)?;
        }
        PluginControlFrame::Heartbeat {
            monotonic_millis, ..
        } if *monotonic_millis == 0 => return Err(PluginError::InvalidDeadline),
        PluginControlFrame::Cancel {
            deadline_millis, ..
        }
        | PluginControlFrame::Shutdown {
            deadline_millis, ..
        } if *deadline_millis == 0 => return Err(PluginError::InvalidDeadline),
        PluginControlFrame::Shutdown { reason, .. } if reason.is_empty() => {
            return Err(PluginError::MalformedFrame);
        }
        PluginControlFrame::Ack {
            request_id: Some(_),
            lifecycle: PluginLifecycle::Terminated,
        } => return Err(PluginError::InvalidLifecycle),
        _ => {}
    }
    Ok(())
}

fn validate_request(request: &PluginRequest) -> Result<(), PluginError> {
    if request.capability.0.is_empty() || request.deadline_millis == 0 {
        return Err(PluginError::InvalidDeadline);
    }
    Ok(())
}

fn validate_response(
    response: &PluginResponse,
    max_receipt_bytes: usize,
) -> Result<(), PluginError> {
    if response.receipt.len() > max_receipt_bytes
        || response.failure.as_ref().is_some_and(|failure| {
            failure.domain.is_empty()
                || failure.code.is_empty()
                || response.output_content_id.is_some()
        })
    {
        return Err(PluginError::MalformedFrame);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PluginError {
    ProtocolVersion(u32),
    InvalidManifest,
    ManifestUntrusted,
    CapabilityUndeclared(String),
    InvalidDeadline,
    InvalidLifecycle,
    BadControlMagic,
    FrameTooLarge,
    MalformedFrame,
    Transport(String),
    MismatchedResponse,
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PluginError {}

pub trait PluginHost {
    fn verify_manifest(&self, manifest: &PluginManifest) -> Result<(), PluginError>;

    fn exchange(
        &mut self,
        manifest: &PluginManifest,
        request: &PluginRequest,
    ) -> Result<PluginResponse, PluginError>;

    fn call(
        &mut self,
        manifest: &PluginManifest,
        request: &PluginRequest,
    ) -> Result<PluginResponse, PluginError> {
        let mut normalized = manifest.clone();
        normalized.normalize()?;
        if normalized != *manifest {
            return Err(PluginError::InvalidManifest);
        }
        validate_request(request)?;
        let process_limits = PluginControlLimits::for_process(&manifest.process);
        PluginControlFrame::Invoke(request.clone()).to_control_bytes_with_limits(process_limits)?;
        self.verify_manifest(manifest)?;
        if !manifest.capabilities.contains(&request.capability) {
            return Err(PluginError::CapabilityUndeclared(
                request.capability.0.clone(),
            ));
        }
        let response = self.exchange(manifest, request)?;
        PluginControlFrame::Complete(response.clone())
            .to_control_bytes_with_limits(process_limits)?;
        if response.request_id != request.request_id {
            return Err(PluginError::MismatchedResponse);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;

    struct Host;

    impl PluginHost for Host {
        fn verify_manifest(&self, _manifest: &PluginManifest) -> Result<(), PluginError> {
            Ok(())
        }

        fn exchange(
            &mut self,
            _manifest: &PluginManifest,
            request: &PluginRequest,
        ) -> Result<PluginResponse, PluginError> {
            Ok(PluginResponse {
                request_id: request.request_id,
                output_content_id: None,
                receipt: vec![],
                failure: None,
            })
        }
    }

    struct OversizedHost;

    impl PluginHost for OversizedHost {
        fn verify_manifest(&self, _manifest: &PluginManifest) -> Result<(), PluginError> {
            Ok(())
        }

        fn exchange(
            &mut self,
            manifest: &PluginManifest,
            request: &PluginRequest,
        ) -> Result<PluginResponse, PluginError> {
            Ok(PluginResponse {
                request_id: request.request_id,
                output_content_id: None,
                receipt: vec![0; manifest.process.max_frame_bytes as usize],
                failure: None,
            })
        }
    }

    fn manifest() -> PluginManifest {
        PluginManifest {
            protocol_version: PLUGIN_PROTOCOL_VERSION,
            plugin_id: "adapter".to_string(),
            executable_digest_algorithm: ExecutableDigestAlgorithm::Blake3DomainV1,
            executable_digest: executable_digest(
                ExecutableDigestAlgorithm::Blake3DomainV1,
                b"executable",
            ),
            capabilities: vec![Capability("export".to_string())],
            process: ProcessContract {
                startup_deadline_millis: 1_000,
                request_deadline_millis: 5_000,
                heartbeat_interval_millis: 500,
                heartbeat_grace_millis: 1_500,
                teardown_deadline_millis: 1_000,
                kill_grace_millis: 100,
                max_inflight: 4,
                max_frame_bytes: 65_536,
                teardown: TeardownPolicy::GracefulThenKill,
            },
            signature_algorithm: SignatureAlgorithm::Ed25519,
            signing_key_id: "test-key".to_string(),
            signature: vec![1; 64],
        }
    }

    fn request(capability: &str) -> PluginRequest {
        PluginRequest {
            request_id: 1,
            invocation_id: [7; 32],
            capability: Capability(capability.to_string()),
            payload_content_id: [0; 32],
            shared_memory_lease: None,
            deadline_millis: 123,
        }
    }

    #[test]
    fn undeclared_capability_never_reaches_transport() {
        assert!(matches!(
            Host.call(&manifest(), &request("import")),
            Err(PluginError::CapabilityUndeclared(_))
        ));
    }

    #[test]
    fn executable_digest_is_domain_separated_and_stable() {
        assert_eq!(
            executable_digest(ExecutableDigestAlgorithm::Blake3DomainV1, b"abc"),
            [
                0xed, 0x25, 0x1c, 0x0f, 0x1e, 0xb9, 0xc7, 0x7c, 0x6b, 0x0e, 0x39, 0x1d, 0x4c, 0x62,
                0xba, 0x66, 0x29, 0xc1, 0x91, 0x56, 0x9c, 0x3e, 0x5f, 0xa6, 0xce, 0xee, 0x22, 0x59,
                0x6a, 0x82, 0x18, 0xa4,
            ]
        );
    }

    #[test]
    fn manifest_signature_preimage_and_bpc2_wire_are_literal_goldens() {
        assert_eq!(
            manifest().signing_digest().unwrap(),
            [
                35, 152, 182, 151, 168, 246, 64, 146, 153, 175, 245, 69, 198, 185, 247, 4, 17, 128,
                1, 174, 112, 76, 208, 232, 80, 232, 42, 245, 207, 34, 39, 128,
            ]
        );
        let mut invoke_golden = vec![66, 80, 67, 50, 2, 1];
        invoke_golden.extend([7; 32]);
        invoke_golden.extend([6, b'e', b'x', b'p', b'o', b'r', b't']);
        invoke_golden.extend([0; 32]);
        invoke_golden.extend([0, 123]);
        assert_eq!(
            PluginControlFrame::Invoke(request("export"))
                .to_control_bytes()
                .unwrap(),
            invoke_golden
        );
    }

    #[test]
    fn ed25519_manifest_rejects_noncanonical_signature_lengths() {
        for length in [0, 63, 65] {
            let mut invalid = manifest();
            invalid.signature = vec![0; length];
            assert_eq!(invalid.normalize(), Err(PluginError::InvalidManifest));
        }
    }

    #[test]
    fn control_wire_rejects_trailing_bytes_and_zero_deadlines() {
        let frame = PluginControlFrame::Invoke(request("export"));
        let mut bytes = frame.to_control_bytes().unwrap();
        assert_eq!(
            PluginControlFrame::from_control_bytes(&bytes, PluginControlLimits::default()),
            Ok(frame)
        );
        bytes.push(0);
        assert_eq!(
            PluginControlFrame::from_control_bytes(&bytes, PluginControlLimits::default()),
            Err(PluginError::MalformedFrame)
        );

        let mut invalid = request("export");
        invalid.deadline_millis = 0;
        assert_eq!(
            PluginControlFrame::Invoke(invalid).to_control_bytes(),
            Err(PluginError::InvalidDeadline)
        );
    }

    #[test]
    fn lifecycle_never_reenters_ready_after_draining_or_termination() {
        assert!(PluginLifecycle::Spawned.permits(PluginLifecycle::Handshaking));
        assert!(PluginLifecycle::Handshaking.permits(PluginLifecycle::Ready));
        assert!(PluginLifecycle::Ready.permits(PluginLifecycle::Draining));
        assert!(PluginLifecycle::Draining.permits(PluginLifecycle::Terminated));
        assert!(!PluginLifecycle::Draining.permits(PluginLifecycle::Ready));
        assert!(!PluginLifecycle::Terminated.permits(PluginLifecycle::Spawned));
    }

    #[test]
    fn response_cannot_claim_an_output_and_failure_together() {
        let response = PluginResponse {
            request_id: 1,
            output_content_id: Some([1; 32]),
            receipt: vec![],
            failure: Some(PluginFailure {
                domain: "plugin.test".into(),
                code: "failed".into(),
                message: "failed".into(),
                retryable: false,
            }),
        };
        assert_eq!(
            PluginControlFrame::Complete(response).to_control_bytes(),
            Err(PluginError::MalformedFrame)
        );
    }

    #[test]
    fn control_encoder_enforces_the_same_frame_bound_as_the_decoder() {
        let frame = PluginControlFrame::Shutdown {
            reason: "x".repeat(PluginControlLimits::default().max_frame_bytes),
            deadline_millis: 1,
        };
        assert_eq!(frame.to_control_bytes(), Err(PluginError::FrameTooLarge));
        assert_eq!(
            OversizedHost.call(&manifest(), &request("export")),
            Err(PluginError::FrameTooLarge)
        );
    }
}
