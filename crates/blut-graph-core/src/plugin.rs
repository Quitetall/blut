// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::model::Capability;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginManifest {
    pub protocol_version: u32,
    pub plugin_id: String,
    pub executable_digest: [u8; 32],
    pub capabilities: Vec<Capability>,
    pub signer: String,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginRequest {
    pub request_id: u64,
    pub capability: Capability,
    pub payload_content_id: [u8; 32],
    pub shared_memory_lease: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginResponse {
    pub request_id: u64,
    pub output_content_id: Option<[u8; 32]>,
    pub receipt: Vec<u8>,
    pub failure: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PluginError {
    ProtocolVersion(u32),
    ManifestUntrusted,
    CapabilityUndeclared(String),
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
        if manifest.protocol_version != 1 {
            return Err(PluginError::ProtocolVersion(manifest.protocol_version));
        }
        self.verify_manifest(manifest)?;
        if !manifest.capabilities.contains(&request.capability) {
            return Err(PluginError::CapabilityUndeclared(
                request.capability.0.clone(),
            ));
        }
        let response = self.exchange(manifest, request)?;
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

    #[test]
    fn undeclared_capability_never_reaches_transport() {
        let manifest = PluginManifest {
            protocol_version: 1,
            plugin_id: "adapter".to_string(),
            executable_digest: [0; 32],
            capabilities: vec![],
            signer: "test".to_string(),
            signature: vec![],
        };
        let request = PluginRequest {
            request_id: 1,
            capability: Capability("import".to_string()),
            payload_content_id: [0; 32],
            shared_memory_lease: None,
        };
        assert!(matches!(
            Host.call(&manifest, &request),
            Err(PluginError::CapabilityUndeclared(_))
        ));
    }
}
