// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

use crate::compile::hash_plan;
use crate::model::{CompiledPlan, Effect, ExecutionRealm, InputBinding, OutputBinding, PlanId};

const MAGIC: &[u8; 4] = b"BGP3";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanLimits {
    pub max_bytes: usize,
    pub max_nodes: usize,
    pub max_buffers: usize,
    pub max_contract_entries: usize,
    pub max_peak_bytes: u64,
    pub max_persistent_state_bytes: u64,
    pub max_feedback_edges: usize,
    pub max_subgraph_depth: usize,
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
            max_persistent_state_bytes: 64 * 1024 * 1024,
            max_feedback_edges: 65_536,
            max_subgraph_depth: 16,
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

fn valid_compiled_port(port: &crate::CompiledPortContract) -> bool {
    let descriptor = crate::PortDescriptor {
        name: port.name.clone(),
        semantic_type: port.semantic_type.clone(),
        optional: port.optional,
        layouts: alloc::vec![port.layout],
        max_bytes: port.max_bytes,
        abir: port.abir.clone(),
        proof: port.proof.clone(),
        policy: port.policy.clone(),
        fidelity: port.fidelity.clone(),
        extent: port.extent.clone(),
        lease: port.lease.clone(),
    };
    crate::compile::valid_port_contract(&descriptor)
        && port.proof.requires.windows(2).all(|pair| pair[0] < pair[1])
        && port.proof.provides.windows(2).all(|pair| pair[0] < pair[1])
        && port
            .proof
            .invalidates
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        && port
            .policy
            .requires
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        && port.policy.adds.windows(2).all(|pair| pair[0] < pair[1])
}

fn contract_entry_count(port: &crate::CompiledPortContract) -> Option<usize> {
    0usize
        .checked_add(port.proof.requires.len())?
        .checked_add(port.proof.provides.len())?
        .checked_add(port.proof.invalidates.len())?
        .checked_add(port.policy.requires.len())?
        .checked_add(port.policy.adds.len())?
        .checked_add(port.extent.maximum_shape.len())
}

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
        if plan.schema_version != 3 {
            return Err(PlanDecodeError::UnsupportedSchema(plan.schema_version));
        }
        if plan.nodes.len() > limits.max_nodes
            || plan.order.len() > limits.max_nodes
            || plan.buffers.len() > limits.max_buffers
            || plan.invocation_ports.len() > limits.max_contract_entries
            || plan.propagated_proofs.len() > limits.max_contract_entries
            || plan.propagated_policy.len() > limits.max_contract_entries
            || plan.feedback.len() > limits.max_feedback_edges
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
            || plan.propagated_proofs.iter().any(|entry| entry.is_empty())
            || plan.propagated_policy.iter().any(|entry| entry.is_empty())
        {
            return Err(PlanDecodeError::InvalidPlan);
        }
        if plan.peak_bytes > limits.max_peak_bytes
            || plan.persistent_state_bytes > limits.max_persistent_state_bytes
        {
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
        let mut used_feedback = BTreeSet::new();
        let mut state_bytes = 0u64;
        for (index, node) in plan.nodes.iter().enumerate() {
            failure_entries = failure_entries
                .checked_add(node.failure.domains.len())
                .ok_or(PlanDecodeError::LimitExceeded)?;
            failure_entries = node
                .semantic_configs
                .iter()
                .try_fold(failure_entries, |count, config| {
                    count.checked_add(config.len())
                })
                .ok_or(PlanDecodeError::LimitExceeded)?;
            failure_entries = node
                .input_contracts
                .iter()
                .chain(&node.output_contracts)
                .try_fold(failure_entries, |count, port| {
                    count.checked_add(contract_entry_count(port)?)
                })
                .and_then(|count| count.checked_add(node.subgraph_path.len()))
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
            for feedback in node
                .input_bindings
                .iter()
                .filter_map(|binding| match binding {
                    InputBinding::Feedback(feedback) => Some(*feedback),
                    _ => None,
                })
            {
                if !used_feedback.insert(feedback) {
                    return Err(PlanDecodeError::InvalidPlan);
                }
            }
            if node.id.0 as usize != index
                || conversion != node.semantic_nodes.is_empty()
                || node.semantic_nodes.len() != node.semantic_types.len()
                || node.semantic_nodes.len() != node.semantic_configs.len()
                || node.input_ports.len() != node.input_bindings.len()
                || node.output_ports.len() != node.output_bindings.len()
                || node.input_contracts.len() != node.input_bindings.len()
                || node.output_contracts.len() != node.output_bindings.len()
                || node
                    .input_ports
                    .iter()
                    .zip(&node.input_contracts)
                    .any(|(name, contract)| name != &contract.name || !valid_compiled_port(contract))
                || node
                    .output_ports
                    .iter()
                    .zip(&node.output_contracts)
                    .any(|(name, contract)| name != &contract.name || !valid_compiled_port(contract))
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
                || node.input_bindings.iter().any(|binding| {
                    matches!(binding, InputBinding::Feedback(feedback) if feedback.0 as usize >= plan.feedback.len())
                })
                || node
                    .input_bindings
                    .iter()
                    .zip(&node.input_contracts)
                    .any(|(binding, contract)| match binding {
                        InputBinding::Buffer(buffer) => {
                            let buffer = &plan.buffers[buffer.0 as usize];
                            contract.layout != buffer.layout
                                || buffer.capacity_bytes > contract.max_bytes
                        }
                        _ => false,
                    })
                || node.output_bindings.iter().any(|binding| {
                    matches!(binding, OutputBinding::Buffer(buffer) if buffer.0 as usize >= plan.buffers.len())
                })
                || node
                    .output_bindings
                    .iter()
                    .zip(&node.output_contracts)
                    .any(|(binding, contract)| match binding {
                        OutputBinding::Buffer(buffer) => {
                            let buffer = &plan.buffers[buffer.0 as usize];
                            contract.layout != buffer.layout
                                || buffer.capacity_bytes != contract.max_bytes
                        }
                        OutputBinding::Terminal => false,
                    })
                || node.resources.threads == 0
                || node.failure.domains.iter().any(|domain| domain.is_empty())
                || node.failure.domains.windows(2).any(|pair| pair[0] >= pair[1])
                || (node.partiality == crate::Partiality::ExplicitGaps
                    && node.failure.domains.is_empty())
                || (node.effect == Effect::AtMostOnce && node.retry_limit > 0)
                || !crate::compile::valid_state_contract(&node.state)
                || node.subgraph_path.len() > limits.max_subgraph_depth
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
                        || node.input_bindings.iter().any(|binding| matches!(binding, InputBinding::Feedback(_)))
                        || node.output_bindings.contains(&OutputBinding::Terminal)
                        || node.partiality != crate::Partiality::Atomic
                        || !node.failure.domains.is_empty()
                        || node.effect != Effect::Pure
                        || node.retry_limit != 0
                        || node.state != crate::StateContract::stateless()
                        || !node.subgraph_path.is_empty()))
            {
                return Err(PlanDecodeError::InvalidPlan);
            }
            if matches!(
                node.state.scope,
                crate::StateScope::Session | crate::StateScope::Durable
            ) {
                state_bytes = state_bytes
                    .checked_add(node.state.max_bytes)
                    .ok_or(PlanDecodeError::ResourceLimitExceeded)?;
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
                    || node.input_contracts[0].semantic_type != conversion.semantic_type
                    || node.output_contracts[0].semantic_type != conversion.semantic_type
                    || node.input_contracts[0].layout != conversion.from
                    || node.output_contracts[0].layout != conversion.to
                    || node.input_contracts[0].max_bytes != input.capacity_bytes
                    || node.output_contracts[0].max_bytes != output.capacity_bytes
                {
                    return Err(PlanDecodeError::InvalidPlan);
                }
            }
            let mut workspace = node
                .resources
                .peak_bytes
                .checked_add(node.resources.scratch_bytes)
                .ok_or(PlanDecodeError::ResourceLimitExceeded)?;
            if node.state.scope == crate::StateScope::Invocation {
                workspace = workspace
                    .checked_add(node.state.max_bytes)
                    .ok_or(PlanDecodeError::ResourceLimitExceeded)?;
            }
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
        if used_feedback.len() != plan.feedback.len()
            || used_feedback
                .iter()
                .copied()
                .ne((0..plan.feedback.len() as u32).map(crate::FeedbackId))
        {
            return Err(PlanDecodeError::InvalidPlan);
        }
        for buffer in &plan.buffers {
            let producer = &plan.nodes[buffer.producer.0 as usize];
            let producer_contract = producer
                .output_bindings
                .iter()
                .position(|binding| *binding == OutputBinding::Buffer(buffer.id))
                .and_then(|port| producer.output_contracts.get(port))
                .ok_or(PlanDecodeError::InvalidPlan)?;
            for consumer in &buffer.consumers {
                let consumer = &plan.nodes[consumer.0 as usize];
                let mut matched = false;
                for (port, binding) in consumer.input_bindings.iter().enumerate() {
                    if *binding == InputBinding::Buffer(buffer.id) {
                        matched = true;
                        if !crate::compile::compiled_port_contract_satisfies(
                            producer_contract,
                            &consumer.input_contracts[port],
                        ) {
                            return Err(PlanDecodeError::InvalidPlan);
                        }
                    }
                }
                if !matched {
                    return Err(PlanDecodeError::InvalidPlan);
                }
            }
        }
        for (index, feedback) in plan.feedback.iter().enumerate() {
            let output = plan
                .nodes
                .get(feedback.from_step.0 as usize)
                .and_then(|node| node.output_contracts.get(feedback.from_port as usize));
            let input = plan
                .nodes
                .get(feedback.to_step.0 as usize)
                .and_then(|node| node.input_contracts.get(feedback.to_port as usize));
            let expected_state_bytes = output.and_then(|contract| {
                contract
                    .max_bytes
                    .checked_mul(u64::from(feedback.delay.invocations))
            });
            if feedback.id.0 as usize != index
                || feedback.delay.invocations == 0
                || feedback.state_bytes == 0
                || feedback.from_step.0 as usize >= plan.nodes.len()
                || feedback.to_step.0 as usize >= plan.nodes.len()
                || output.is_none()
                || input.is_none()
                || (matches!(&feedback.delay.initial, crate::DelayInitial::Absent)
                    && input.is_some_and(|contract| !contract.optional))
                || plan.nodes[feedback.to_step.0 as usize].input_bindings[feedback.to_port as usize]
                    != InputBinding::Feedback(feedback.id)
                || expected_state_bytes != Some(feedback.state_bytes)
                || !crate::compile::compiled_port_contract_satisfies(
                    output.expect("checked above"),
                    input.expect("checked above"),
                )
            {
                return Err(PlanDecodeError::InvalidPlan);
            }
            state_bytes = state_bytes
                .checked_add(feedback.state_bytes)
                .ok_or(PlanDecodeError::ResourceLimitExceeded)?;
        }
        if state_bytes != plan.persistent_state_bytes
            || (!plan.feedback.is_empty() && plan.session.is_none())
            || plan.session.as_ref().is_some_and(|session| {
                session.namespace.is_empty()
                    || session.max_concurrent_sessions == 0
                    || session.max_idle_millis == 0
            })
            || plan.nodes.iter().any(|node| {
                matches!(
                    node.state.scope,
                    crate::StateScope::Session | crate::StateScope::Durable
                ) && plan.session.is_none()
            })
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

    fn minimal_plan() -> CompiledPlan {
        let mut plan = CompiledPlan {
            schema_version: 3,
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
                input_contracts: vec![],
                output_contracts: vec![crate::CompiledPortContract::opaque(
                    "out",
                    "test",
                    crate::Layout::Canonical,
                    1,
                )],
                input_bindings: vec![],
                output_bindings: vec![crate::OutputBinding::Terminal],
                partiality: crate::Partiality::Atomic,
                failure: crate::FailureContract { domains: vec![] },
                effect: crate::Effect::Pure,
                retry_limit: 0,
                state: crate::StateContract::stateless(),
                subgraph_path: vec![],
            }],
            buffers: vec![],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 0,
            persistent_state_bytes: 0,
            session: None,
        };
        plan.plan_id = PlanId(hash_plan(&plan));
        plan
    }

    #[test]
    fn bgp2_magic_is_never_reinterpreted_as_bgp3() {
        let mut bytes = minimal_plan().to_aot_bytes().unwrap();
        bytes[..4].copy_from_slice(b"BGP2");
        assert_eq!(
            CompiledPlan::from_aot_bytes(&bytes, PlanLimits::default()),
            Err(PlanDecodeError::BadMagic)
        );
    }

    #[test]
    fn self_consistent_port_and_state_forgery_is_structurally_rejected() {
        let mut port_forgery = minimal_plan();
        port_forgery.nodes[0].output_contracts[0].name = "different".into();
        port_forgery.plan_id = PlanId(hash_plan(&port_forgery));
        assert_eq!(
            CompiledPlan::from_aot_bytes(
                &port_forgery.to_aot_bytes().unwrap(),
                PlanLimits::default(),
            ),
            Err(PlanDecodeError::InvalidPlan)
        );

        let mut state_forgery = minimal_plan();
        state_forgery.persistent_state_bytes = 1;
        state_forgery.plan_id = PlanId(hash_plan(&state_forgery));
        assert_eq!(
            CompiledPlan::from_aot_bytes(
                &state_forgery.to_aot_bytes().unwrap(),
                PlanLimits::default(),
            ),
            Err(PlanDecodeError::InvalidPlan)
        );

        let mut rewire_forgery = minimal_plan();
        rewire_forgery.order.push(crate::NodeId(8));
        rewire_forgery.nodes[0].output_bindings[0] =
            crate::OutputBinding::Buffer(crate::BufferId(0));
        let mut sink = rewire_forgery.nodes[0].clone();
        sink.id = crate::StepId(1);
        sink.semantic_nodes = vec![crate::NodeId(8)];
        sink.input_ports = vec!["in".into()];
        sink.output_ports = vec!["out".into()];
        sink.input_contracts = vec![crate::CompiledPortContract::opaque(
            "in",
            "test",
            crate::Layout::Canonical,
            1,
        )];
        sink.output_contracts = vec![crate::CompiledPortContract::opaque(
            "out",
            "test",
            crate::Layout::Canonical,
            1,
        )];
        sink.input_bindings = vec![InputBinding::Buffer(crate::BufferId(0))];
        sink.output_bindings = vec![OutputBinding::Terminal];
        rewire_forgery.nodes.push(sink);
        rewire_forgery.buffers.push(crate::BufferPlan {
            id: crate::BufferId(0),
            layout: crate::Layout::Canonical,
            capacity_bytes: 1,
            producer: crate::StepId(0),
            consumers: vec![crate::StepId(1)],
            last_consumer: crate::StepId(1),
            aliases: None,
        });
        rewire_forgery.peak_bytes = 1;
        rewire_forgery.plan_id = PlanId(hash_plan(&rewire_forgery));
        assert!(
            CompiledPlan::from_aot_bytes(
                &rewire_forgery.to_aot_bytes().unwrap(),
                PlanLimits::default(),
            )
            .is_ok()
        );

        rewire_forgery.nodes[1].input_contracts[0].semantic_type = "different".into();
        rewire_forgery.plan_id = PlanId(hash_plan(&rewire_forgery));
        assert_eq!(
            CompiledPlan::from_aot_bytes(
                &rewire_forgery.to_aot_bytes().unwrap(),
                PlanLimits::default(),
            ),
            Err(PlanDecodeError::InvalidPlan)
        );
    }

    #[test]
    fn empty_compiled_contract_names_are_rejected() {
        let mut port_contract = minimal_plan();
        port_contract.nodes[0].output_contracts[0]
            .policy
            .adds
            .push(String::new());
        port_contract.plan_id = PlanId(hash_plan(&port_contract));
        assert_eq!(
            CompiledPlan::from_aot_bytes(
                &port_contract.to_aot_bytes().unwrap(),
                PlanLimits::default(),
            ),
            Err(PlanDecodeError::InvalidPlan)
        );

        let mut propagated_contract = minimal_plan();
        propagated_contract.propagated_proofs.push(String::new());
        propagated_contract.plan_id = PlanId(hash_plan(&propagated_contract));
        assert_eq!(
            CompiledPlan::from_aot_bytes(
                &propagated_contract.to_aot_bytes().unwrap(),
                PlanLimits::default(),
            ),
            Err(PlanDecodeError::InvalidPlan)
        );
    }

    #[test]
    fn subgraph_depth_limit_is_enforced_during_structural_decode() {
        let mut plan = minimal_plan();
        plan.nodes[0].subgraph_path = vec![crate::SubgraphId([1; 32]), crate::SubgraphId([2; 32])];
        plan.plan_id = PlanId(hash_plan(&plan));
        assert_eq!(
            CompiledPlan::from_authorized_aot_bytes(
                &plan.to_aot_bytes().unwrap(),
                PlanLimits {
                    max_subgraph_depth: 1,
                    ..PlanLimits::default()
                },
                PlanAuthorization {
                    expected_realm: plan.realm,
                    expected_plan_id: plan.plan_id,
                },
            ),
            Err(PlanDecodeError::InvalidPlan)
        );
    }

    #[test]
    fn plan_round_trip_and_tamper_rejection() {
        let mut plan = CompiledPlan {
            schema_version: 3,
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
                input_contracts: vec![],
                output_contracts: vec![crate::CompiledPortContract::opaque(
                    "out",
                    "test",
                    crate::Layout::Canonical,
                    1,
                )],
                input_bindings: vec![],
                output_bindings: vec![crate::OutputBinding::Terminal],
                partiality: crate::Partiality::Atomic,
                failure: crate::FailureContract { domains: vec![] },
                effect: crate::Effect::Pure,
                retry_limit: 0,
                state: crate::StateContract::stateless(),
                subgraph_path: vec![],
            }],
            buffers: vec![],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 0,
            persistent_state_bytes: 0,
            session: None,
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
            schema_version: 3,
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
                input_contracts: vec![],
                output_contracts: vec![crate::CompiledPortContract::opaque(
                    "out",
                    "test",
                    crate::Layout::Canonical,
                    1,
                )],
                input_bindings: vec![],
                output_bindings: vec![crate::OutputBinding::Terminal],
                partiality: crate::Partiality::Atomic,
                failure: crate::FailureContract { domains: vec![] },
                effect: crate::Effect::Pure,
                retry_limit: 0,
                state: crate::StateContract::stateless(),
                subgraph_path: vec![],
            }],
            buffers: vec![],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 0,
            persistent_state_bytes: 0,
            session: None,
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
            schema_version: 3,
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
                input_contracts: vec![],
                output_contracts: vec![crate::CompiledPortContract::opaque(
                    "out",
                    "test",
                    crate::Layout::Canonical,
                    1,
                )],
                input_bindings: vec![],
                output_bindings: vec![crate::OutputBinding::Terminal],
                partiality: crate::Partiality::Atomic,
                failure: crate::FailureContract { domains: vec![] },
                effect: crate::Effect::Pure,
                retry_limit: 0,
                state: crate::StateContract::stateless(),
                subgraph_path: vec![],
            }],
            buffers: vec![],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 4096,
            persistent_state_bytes: 0,
            session: None,
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
