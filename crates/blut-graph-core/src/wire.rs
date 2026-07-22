// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::vec::Vec;
use core::fmt;

use crate::compile::hash_plan;
use crate::model::{CompiledPlan, PlanId};

const MAGIC: &[u8; 4] = b"BGP1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanLimits {
    pub max_bytes: usize,
    pub max_nodes: usize,
    pub max_buffers: usize,
    pub max_contract_entries: usize,
}

impl Default for PlanLimits {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024,
            max_nodes: 65_536,
            max_buffers: 262_144,
            max_contract_entries: 65_536,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanDecodeError {
    TooLarge,
    BadMagic,
    Malformed,
    UnsupportedSchema(u32),
    LimitExceeded,
    IdentityMismatch,
    InvalidBuffer,
}

impl fmt::Display for PlanDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PlanDecodeError {}

impl CompiledPlan {
    /// Deterministic AOT bytes. Ordered collections are fixed by compilation;
    /// postcard encodes the schema without host layout or pointer dependence.
    pub fn to_aot_bytes(&self) -> Result<Vec<u8>, PlanDecodeError> {
        let mut bytes = Vec::from(MAGIC.as_slice());
        bytes.extend(postcard::to_allocvec(self).map_err(|_| PlanDecodeError::Malformed)?);
        Ok(bytes)
    }

    /// Decode untrusted AOT bytes under structural limits and re-derive the
    /// physical plan identity before returning an executable plan. `max_bytes`
    /// is the allocation bound during postcard decode; the count limits are
    /// post-decode semantic bounds within that already-bounded envelope.
    pub fn from_aot_bytes(bytes: &[u8], limits: PlanLimits) -> Result<Self, PlanDecodeError> {
        if bytes.len() > limits.max_bytes {
            return Err(PlanDecodeError::TooLarge);
        }
        let body = bytes.strip_prefix(MAGIC).ok_or(PlanDecodeError::BadMagic)?;
        let (plan, remainder): (Self, &[u8]) =
            postcard::take_from_bytes(body).map_err(|_| PlanDecodeError::Malformed)?;
        if !remainder.is_empty() {
            return Err(PlanDecodeError::Malformed);
        }
        if plan.schema_version != 1 {
            return Err(PlanDecodeError::UnsupportedSchema(plan.schema_version));
        }
        if plan.nodes.len() > limits.max_nodes
            || plan.order.len() > limits.max_nodes
            || plan.buffers.len() > limits.max_buffers
            || plan.propagated_proofs.len() > limits.max_contract_entries
            || plan.propagated_policy.len() > limits.max_contract_entries
        {
            return Err(PlanDecodeError::LimitExceeded);
        }
        for (index, buffer) in plan.buffers.iter().enumerate() {
            // Compiler output uses dense, ID-ordered buffers so executor lookup
            // remains O(1); hand-built sparse plans are not valid AOT inputs.
            if buffer.id.0 as usize != index
                || buffer.capacity_bytes == 0
                || buffer.consumers.is_empty()
                || !buffer.consumers.contains(&buffer.last_consumer)
            {
                return Err(PlanDecodeError::InvalidBuffer);
            }
        }
        let expected = PlanId(hash_plan(&plan));
        if plan.plan_id != expected {
            return Err(PlanDecodeError::IdentityMismatch);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::model::{ExecutionRealm, GraphId};

    #[test]
    fn empty_plan_round_trip_and_tamper_rejection() {
        let mut plan = CompiledPlan {
            schema_version: 1,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::McuAot,
            order: vec![],
            nodes: vec![],
            buffers: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 0,
        };
        plan.plan_id = PlanId(hash_plan(&plan));
        let bytes = plan.to_aot_bytes().unwrap();
        assert_eq!(
            CompiledPlan::from_aot_bytes(&bytes, PlanLimits::default()).unwrap(),
            plan
        );
        let mut tampered = bytes;
        *tampered.last_mut().unwrap() ^= 1;
        assert!(CompiledPlan::from_aot_bytes(&tampered, PlanLimits::default()).is_err());
    }

    #[test]
    fn oversized_input_fails_before_decode() {
        let bytes = vec![0; 9];
        assert_eq!(
            CompiledPlan::from_aot_bytes(
                &bytes,
                PlanLimits {
                    max_bytes: 8,
                    ..PlanLimits::default()
                }
            ),
            Err(PlanDecodeError::TooLarge)
        );
    }
}
