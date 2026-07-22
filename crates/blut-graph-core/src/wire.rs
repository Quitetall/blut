// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use crate::compile::hash_plan;
use crate::model::{CompiledPlan, Effect, ExecutionRealm, InputBinding, OutputBinding, PlanId};

const MAGIC: &[u8; 4] = b"BGP2";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanLimits {
    pub max_bytes: usize,
    pub max_nodes: usize,
    pub max_buffers: usize,
    pub max_contract_entries: usize,
    pub max_peak_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanAuthorization {
    pub expected_realm: ExecutionRealm,
    pub expected_plan_id: PlanId,
}

impl Default for PlanLimits {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024,
            max_nodes: 65_536,
            max_buffers: 262_144,
            max_contract_entries: 65_536,
            max_peak_bytes: 64 * 1024 * 1024,
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
    ResourceLimitExceeded,
    IdentityMismatch,
    InvalidBuffer,
    InvalidPlan,
    RealmMismatch,
    UnauthorizedPlan,
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
        if plan.schema_version != 2 {
            return Err(PlanDecodeError::UnsupportedSchema(plan.schema_version));
        }
        if plan.nodes.len() > limits.max_nodes
            || plan.order.len() > limits.max_nodes
            || plan.buffers.len() > limits.max_buffers
            || plan.invocation_ports.len() > limits.max_contract_entries
            || plan.propagated_proofs.len() > limits.max_contract_entries
            || plan.propagated_policy.len() > limits.max_contract_entries
        {
            return Err(PlanDecodeError::LimitExceeded);
        }
        if plan
            .invocation_ports
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || plan
                .propagated_proofs
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || plan
                .propagated_policy
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(PlanDecodeError::InvalidPlan);
        }
        if plan.peak_bytes > limits.max_peak_bytes {
            return Err(PlanDecodeError::ResourceLimitExceeded);
        }
        if plan.order.is_empty() || plan.nodes.is_empty() {
            return Err(PlanDecodeError::InvalidPlan);
        }
        let semantic_positions: BTreeMap<_, _> = plan
            .order
            .iter()
            .enumerate()
            .map(|(index, node)| (*node, index))
            .collect();
        if semantic_positions.len() != plan.order.len()
            || plan
                .nodes
                .iter()
                .flat_map(|node| node.semantic_nodes.iter())
                .copied()
                .ne(plan.order.iter().copied())
        {
            return Err(PlanDecodeError::InvalidPlan);
        }
        let mut buffer_bytes = 0u64;
        for (index, buffer) in plan.buffers.iter().enumerate() {
            // Compiler output uses dense, ID-ordered buffers so executor lookup
            // remains O(1); hand-built sparse plans are not valid AOT inputs.
            if buffer.id.0 as usize != index
                || buffer.capacity_bytes == 0
                || buffer.consumers.is_empty()
                || !buffer.consumers.contains(&buffer.last_consumer)
                || buffer.producer.0 as usize >= plan.nodes.len()
                || buffer
                    .consumers
                    .iter()
                    .any(|consumer| consumer.0 as usize >= plan.nodes.len())
                || buffer.consumers.iter().collect::<BTreeSet<_>>().len() != buffer.consumers.len()
                || buffer.consumers.windows(2).any(|pair| pair[0] >= pair[1])
                || buffer.consumers.iter().max() != Some(&buffer.last_consumer)
                || buffer
                    .consumers
                    .iter()
                    .any(|consumer| *consumer <= buffer.producer)
                || !plan.nodes[buffer.producer.0 as usize]
                    .output_bindings
                    .contains(&OutputBinding::Buffer(buffer.id))
                || buffer.consumers.iter().any(|consumer| {
                    !plan.nodes[consumer.0 as usize]
                        .input_bindings
                        .contains(&InputBinding::Buffer(buffer.id))
                })
                || buffer.aliases.is_some_and(|alias| {
                    alias.0 >= buffer.id.0
                        || plan.buffers[alias.0 as usize].aliases.is_some()
                        || plan.buffers[alias.0 as usize].layout != buffer.layout
                        || plan.buffers[alias.0 as usize].capacity_bytes < buffer.capacity_bytes
                })
            {
                return Err(PlanDecodeError::InvalidBuffer);
            }
            if buffer.aliases.is_none() {
                buffer_bytes = buffer_bytes
                    .checked_add(buffer.capacity_bytes)
                    .ok_or(PlanDecodeError::ResourceLimitExceeded)?;
            }
        }
        let mut alias_intervals: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for buffer in &plan.buffers {
            let root = buffer.aliases.unwrap_or(buffer.id);
            alias_intervals
                .entry(root)
                .or_default()
                .push((buffer.producer, buffer.last_consumer));
        }
        for intervals in alias_intervals.values_mut() {
            intervals.sort_unstable();
            if intervals.windows(2).any(|pair| pair[0].1 >= pair[1].0) {
                return Err(PlanDecodeError::InvalidBuffer);
            }
        }
        let mut workspace_bytes = 0u64;
        let mut failure_entries = 0usize;
        let mut used_invocations = BTreeSet::new();
        for (index, node) in plan.nodes.iter().enumerate() {
            failure_entries = failure_entries
                .checked_add(node.failure.domains.len())
                .ok_or(PlanDecodeError::LimitExceeded)?;
            let conversion = node.conversion.is_some();
            for invocation in node
                .input_bindings
                .iter()
                .filter_map(|binding| match binding {
                    InputBinding::Invocation(invocation) => Some(*invocation),
                    _ => None,
                })
            {
                if !used_invocations.insert(invocation) {
                    return Err(PlanDecodeError::InvalidPlan);
                }
            }
            if node.id.0 as usize != index
                || conversion != node.semantic_nodes.is_empty()
                || node.semantic_nodes.len() != node.semantic_types.len()
                || node.semantic_nodes.len() != node.semantic_configs.len()
                || node.input_ports.len() != node.input_bindings.len()
                || node.output_ports.len() != node.output_bindings.len()
                || node.input_ports.iter().any(|port| port.is_empty())
                || node.output_ports.iter().any(|port| port.is_empty())
                || node.input_ports.iter().collect::<BTreeSet<_>>().len()
                    != node.input_ports.len()
                || node.output_ports.iter().collect::<BTreeSet<_>>().len()
                    != node.output_ports.len()
                || node.input_bindings.iter().any(|binding| {
                    matches!(binding, InputBinding::Buffer(buffer) if buffer.0 as usize >= plan.buffers.len())
                })
                || node.input_bindings.iter().any(|binding| {
                    matches!(binding, InputBinding::Invocation(invocation) if *invocation as usize >= plan.invocation_ports.len())
                })
                || node.output_bindings.iter().any(|binding| {
                    matches!(binding, OutputBinding::Buffer(buffer) if buffer.0 as usize >= plan.buffers.len())
                })
                || node.resources.threads == 0
                || node.failure.domains.iter().any(|domain| domain.is_empty())
                || node.failure.domains.windows(2).any(|pair| pair[0] >= pair[1])
                || (node.partiality == crate::Partiality::ExplicitGaps
                    && node.failure.domains.is_empty())
                || (node.effect == Effect::AtMostOnce && node.retry_limit > 0)
                || node.input_bindings.iter().any(|binding| {
                    matches!(binding, InputBinding::Buffer(buffer) if !plan.buffers[buffer.0 as usize].consumers.contains(&node.id))
                })
                || node.output_bindings.iter().any(|binding| {
                    matches!(binding, OutputBinding::Buffer(buffer) if plan.buffers[buffer.0 as usize].producer != node.id)
                })
                || (conversion
                    && (node.input_bindings.len() != 1
                        || node.output_bindings.len() != 1
                        || node.input_ports.as_slice() != ["input"]
                        || node.output_ports.as_slice() != ["output"]
                        || node.input_bindings.contains(&InputBinding::Absent)
                        || node.output_bindings.contains(&OutputBinding::Terminal)
                        || node.partiality != crate::Partiality::Atomic
                        || !node.failure.domains.is_empty()
                        || node.effect != Effect::Pure
                        || node.retry_limit != 0
                        || node.checkpointable))
            {
                return Err(PlanDecodeError::InvalidPlan);
            }
            if let Some(conversion) = &node.conversion {
                let (InputBinding::Buffer(input), OutputBinding::Buffer(output)) =
                    (node.input_bindings[0], node.output_bindings[0])
                else {
                    return Err(PlanDecodeError::InvalidPlan);
                };
                let input = &plan.buffers[input.0 as usize];
                let output = &plan.buffers[output.0 as usize];
                if input.layout != conversion.from
                    || output.layout != conversion.to
                    || input.capacity_bytes > conversion.max_input_bytes
                    || output.capacity_bytes > conversion.max_output_bytes
                {
                    return Err(PlanDecodeError::InvalidPlan);
                }
            }
            let workspace = node
                .resources
                .peak_bytes
                .checked_add(node.resources.scratch_bytes)
                .ok_or(PlanDecodeError::ResourceLimitExceeded)?;
            workspace_bytes = workspace_bytes.max(workspace);
        }
        if failure_entries > limits.max_contract_entries {
            return Err(PlanDecodeError::LimitExceeded);
        }
        if used_invocations.len() != plan.invocation_ports.len()
            || used_invocations
                .iter()
                .copied()
                .ne(0..plan.invocation_ports.len() as u32)
        {
            return Err(PlanDecodeError::InvalidPlan);
        }
        let mut produced_buffers = alloc::vec![0u8; plan.buffers.len()];
        for binding in plan
            .nodes
            .iter()
            .flat_map(|node| node.output_bindings.iter())
        {
            if let OutputBinding::Buffer(buffer) = binding {
                produced_buffers[buffer.0 as usize] = produced_buffers[buffer.0 as usize]
                    .checked_add(1)
                    .ok_or(PlanDecodeError::InvalidBuffer)?;
            }
        }
        if produced_buffers.iter().any(|count| *count != 1) {
            return Err(PlanDecodeError::InvalidBuffer);
        }
        let expected_peak = buffer_bytes
            .checked_add(workspace_bytes)
            .ok_or(PlanDecodeError::ResourceLimitExceeded)?;
        if plan.peak_bytes != expected_peak {
            return Err(PlanDecodeError::ResourceLimitExceeded);
        }
        let expected = PlanId(hash_plan(&plan));
        if plan.plan_id != expected {
            return Err(PlanDecodeError::IdentityMismatch);
        }
        Ok(plan)
    }

    /// Decode and authorize an executable AOT plan against identity supplied
    /// by a trusted compiler, signed manifest, or statically linked firmware.
    /// `from_aot_bytes` alone performs structural validation, not authorization.
    pub fn from_authorized_aot_bytes(
        bytes: &[u8],
        limits: PlanLimits,
        authorization: PlanAuthorization,
    ) -> Result<Self, PlanDecodeError> {
        let plan = Self::from_aot_bytes(bytes, limits)?;
        if plan.realm != authorization.expected_realm {
            return Err(PlanDecodeError::RealmMismatch);
        }
        if plan.plan_id != authorization.expected_plan_id {
            return Err(PlanDecodeError::UnauthorizedPlan);
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
    fn plan_round_trip_and_tamper_rejection() {
        let mut plan = CompiledPlan {
            schema_version: 2,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::McuAot,
            order: vec![crate::NodeId(7)],
            nodes: vec![crate::CompiledNode {
                id: crate::StepId(0),
                semantic_nodes: vec![crate::NodeId(7)],
                semantic_types: vec![crate::NodeTypeRef {
                    type_name: "test".into(),
                    version: 1,
                }],
                semantic_configs: vec![alloc::collections::BTreeMap::new()],
                kernel: crate::KernelId(11),
                implementation_id: crate::ImplementationId([11; 32]),
                resources: crate::ResourceEnvelope::bounded(0, 0, 1),
                determinism: crate::Determinism::BitExact,
                lowering: "test".into(),
                conversion: None,
                input_ports: vec![],
                output_ports: vec!["out".into()],
                input_bindings: vec![],
                output_bindings: vec![crate::OutputBinding::Terminal],
                partiality: crate::Partiality::Atomic,
                failure: crate::FailureContract { domains: vec![] },
                effect: crate::Effect::Pure,
                retry_limit: 0,
                checkpointable: false,
            }],
            buffers: vec![],
            invocation_ports: vec![],
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
    fn manifest_authorization_binds_realm_and_plan_identity() {
        let mut plan = CompiledPlan {
            schema_version: 2,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::McuAot,
            order: vec![crate::NodeId(7)],
            nodes: vec![crate::CompiledNode {
                id: crate::StepId(0),
                semantic_nodes: vec![crate::NodeId(7)],
                semantic_types: vec![crate::NodeTypeRef {
                    type_name: "test".into(),
                    version: 1,
                }],
                semantic_configs: vec![alloc::collections::BTreeMap::new()],
                kernel: crate::KernelId(11),
                implementation_id: crate::ImplementationId([11; 32]),
                resources: crate::ResourceEnvelope::bounded(0, 0, 1),
                determinism: crate::Determinism::BitExact,
                lowering: "test".into(),
                conversion: None,
                input_ports: vec![],
                output_ports: vec!["out".into()],
                input_bindings: vec![],
                output_bindings: vec![crate::OutputBinding::Terminal],
                partiality: crate::Partiality::Atomic,
                failure: crate::FailureContract { domains: vec![] },
                effect: crate::Effect::Pure,
                retry_limit: 0,
                checkpointable: false,
            }],
            buffers: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 0,
        };
        plan.plan_id = PlanId(hash_plan(&plan));
        let bytes = plan.to_aot_bytes().unwrap();
        assert_eq!(
            CompiledPlan::from_authorized_aot_bytes(
                &bytes,
                PlanLimits::default(),
                PlanAuthorization {
                    expected_realm: ExecutionRealm::HostStream,
                    expected_plan_id: plan.plan_id,
                },
            ),
            Err(PlanDecodeError::RealmMismatch)
        );
        assert_eq!(
            CompiledPlan::from_authorized_aot_bytes(
                &bytes,
                PlanLimits::default(),
                PlanAuthorization {
                    expected_realm: ExecutionRealm::McuAot,
                    expected_plan_id: PlanId([9; 32]),
                },
            ),
            Err(PlanDecodeError::UnauthorizedPlan)
        );
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

    #[test]
    fn declared_peak_memory_is_bounded_before_execution() {
        let mut plan = CompiledPlan {
            schema_version: 2,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::McuAot,
            order: vec![crate::NodeId(7)],
            nodes: vec![crate::CompiledNode {
                id: crate::StepId(0),
                semantic_nodes: vec![crate::NodeId(7)],
                semantic_types: vec![crate::NodeTypeRef {
                    type_name: "test".into(),
                    version: 1,
                }],
                semantic_configs: vec![alloc::collections::BTreeMap::new()],
                kernel: crate::KernelId(11),
                implementation_id: crate::ImplementationId([11; 32]),
                resources: crate::ResourceEnvelope::bounded(0, 0, 1),
                determinism: crate::Determinism::BitExact,
                lowering: "test".into(),
                conversion: None,
                input_ports: vec![],
                output_ports: vec!["out".into()],
                input_bindings: vec![],
                output_bindings: vec![crate::OutputBinding::Terminal],
                partiality: crate::Partiality::Atomic,
                failure: crate::FailureContract { domains: vec![] },
                effect: crate::Effect::Pure,
                retry_limit: 0,
                checkpointable: false,
            }],
            buffers: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 4096,
        };
        plan.plan_id = PlanId(hash_plan(&plan));
        let bytes = plan.to_aot_bytes().unwrap();
        assert_eq!(
            CompiledPlan::from_aot_bytes(
                &bytes,
                PlanLimits {
                    max_peak_bytes: 1024,
                    ..PlanLimits::default()
                }
            ),
            Err(PlanDecodeError::ResourceLimitExceeded)
        );
    }
}
