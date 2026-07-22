// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::model::{
    AuthorizedPlan, BufferId, BufferPlan, CompiledNode, CompiledPlan, CompiledPortContract, Edge,
    ExecutionRealm, Graph, GraphId, KernelDescriptor, KernelId, Layout, NodeDescriptor, NodeId,
    NodeTypeRef, OutputBinding, PlanId, PortDescriptor, StateContract, StateScope, StepId, Target,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    UnsupportedGraphVersion(u32),
    DuplicateNode(NodeId),
    UnknownNode(NodeId),
    UnknownDescriptor(String, u32),
    DuplicateDescriptor(String, u32),
    InvalidDescriptor(String, u32),
    InvalidConfig(NodeId, crate::ConfigError),
    DuplicateKernel(KernelId),
    InvalidKernelContract(KernelId),
    UnknownPort(NodeId, String),
    InvalidPortSize(NodeId, String),
    TypeMismatch(String, String),
    PortCapacityMismatch(NodeId, String, NodeId, String),
    LayoutUnavailable(NodeId, String),
    MissingInput(NodeId, String),
    DuplicateInput(NodeId, String),
    DuplicateInvocation(NodeId, String),
    Cycle,
    CapabilityMissing(String),
    CapabilityUnsupported(String),
    TargetUnsupported(NodeId, Target),
    KernelUnavailable(NodeId, Target),
    ProofMissing(NodeId, String),
    PolicyMissing(NodeId, String),
    FidelityInsufficient(NodeId),
    UnsafeRetry(NodeId),
    SearchLimitExceeded,
    CompileLimitExceeded,
    ResourceOverflow,
    EmptyGraph,
    InvalidGraphContract,
    InvalidState(NodeId),
    InvalidSession,
    InvalidFeedback(NodeId, String),
    PortContractMismatch(NodeId, String, NodeId, String),
    UnknownSubgraph(crate::SubgraphId),
    InvalidSubgraph(crate::SubgraphId),
    SubgraphDepthExceeded,
    SubgraphEntryLimitExceeded,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CompileError {}

#[derive(Clone, Debug, Default)]
pub struct KernelRegistry {
    descriptors: BTreeMap<(String, u32), NodeDescriptor>,
    kernels: BTreeMap<KernelId, KernelDescriptor>,
    subgraphs: BTreeMap<crate::SubgraphId, crate::SubgraphSchema>,
}

impl KernelRegistry {
    pub fn register_descriptor(
        &mut self,
        mut descriptor: NodeDescriptor,
    ) -> Result<(), CompileError> {
        for port in descriptor
            .inputs
            .iter_mut()
            .chain(descriptor.outputs.iter_mut())
        {
            port.layouts.sort_unstable();
            port.layouts.dedup();
            normalize_contract(&mut port.proof, &mut port.policy);
        }
        descriptor.capabilities.sort_unstable();
        descriptor.capabilities.dedup();
        descriptor.targets.sort_unstable();
        descriptor.targets.dedup();
        descriptor.proof.requires.sort_unstable();
        descriptor.proof.requires.dedup();
        descriptor.proof.provides.sort_unstable();
        descriptor.proof.provides.dedup();
        descriptor.proof.invalidates.sort_unstable();
        descriptor.proof.invalidates.dedup();
        descriptor.policy.requires.sort_unstable();
        descriptor.policy.requires.dedup();
        descriptor.policy.adds.sort_unstable();
        descriptor.policy.adds.dedup();
        descriptor.failure.domains.sort_unstable();
        descriptor.failure.domains.dedup();
        if let Some(lowering) = &mut descriptor.subgraph {
            lowering.input_map.sort_unstable();
            lowering.input_map.dedup();
            lowering.output_map.sort_unstable();
            lowering.output_map.dedup();
        }
        descriptor.config.normalize().map_err(|_| {
            CompileError::InvalidDescriptor(descriptor.type_name.clone(), descriptor.version)
        })?;
        let key = (descriptor.type_name.clone(), descriptor.version);
        if self.descriptors.contains_key(&key) {
            return Err(CompileError::DuplicateDescriptor(key.0, key.1));
        }
        fn invalid_ports(ports: &[PortDescriptor]) -> bool {
            let mut names = BTreeSet::new();
            ports.iter().any(|port| {
                port.name.is_empty()
                    || port.semantic_type.is_empty()
                    || port.max_bytes == 0
                    || port.layouts.is_empty()
                    || !valid_port_contract(port)
                    || !names.insert(port.name.as_str())
            })
        }
        if descriptor.type_name.is_empty()
            || descriptor.version == 0
            || descriptor.resources.threads == 0
            || !valid_contract_names(&descriptor.proof, &descriptor.policy)
            || invalid_ports(&descriptor.inputs)
            || invalid_ports(&descriptor.outputs)
            || (descriptor.effect == crate::model::Effect::AtMostOnce && descriptor.retry_limit > 0)
            || descriptor
                .failure
                .domains
                .iter()
                .any(|domain| domain.is_empty())
            || (descriptor.partiality == crate::model::Partiality::ExplicitGaps
                && descriptor.failure.domains.is_empty())
            || !valid_state_contract(&descriptor.state)
        {
            return Err(CompileError::InvalidDescriptor(key.0, key.1));
        }
        self.descriptors.insert(key, descriptor);
        Ok(())
    }

    pub fn register_subgraph(
        &mut self,
        mut schema: crate::SubgraphSchema,
    ) -> Result<(), CompileError> {
        schema.nodes.sort_by_key(|node| node.id);
        schema
            .edges
            .sort_by_key(|edge| (edge.from.clone(), edge.to.clone()));
        schema.inputs.sort_unstable();
        schema.outputs.sort_unstable();
        let duplicate_nodes = schema.nodes.windows(2).any(|pair| pair[0].id == pair[1].id);
        let duplicate_edges = schema.edges.windows(2).any(|pair| pair[0] == pair[1]);
        let duplicate_interface = |ports: &[crate::SubgraphInterfacePort]| {
            ports.windows(2).any(|pair| pair[0].name == pair[1].name)
                || ports
                    .iter()
                    .map(|port| &port.inner)
                    .collect::<BTreeSet<_>>()
                    .len()
                    != ports.len()
        };
        if schema.version == 0
            || schema.nodes.is_empty()
            || duplicate_nodes
            || duplicate_edges
            || duplicate_interface(&schema.inputs)
            || duplicate_interface(&schema.outputs)
            || schema.id != subgraph_identity(&schema)
            || self.subgraphs.contains_key(&schema.id)
        {
            return Err(CompileError::InvalidSubgraph(schema.id));
        }
        let mut descriptors = BTreeMap::new();
        for node in &schema.nodes {
            let descriptor = self
                .descriptors
                .get(&(node.node_type.type_name.clone(), node.node_type.version))
                .ok_or(CompileError::InvalidSubgraph(schema.id))?;
            let invalid_config = match descriptor.config.canonicalize(&node.config) {
                Ok(canonical) => canonical != node.config,
                Err(_) => true,
            };
            let invalid_child = match node.child {
                Some(child) if child == schema.id => true,
                Some(child) => self.subgraphs.get(&child).is_none_or(|child| {
                    !subgraph_implements_descriptor(child, descriptor, &self.descriptors)
                }),
                None => false,
            };
            if invalid_config || invalid_child {
                return Err(CompileError::InvalidSubgraph(schema.id));
            }
            descriptors.insert(node.id, descriptor);
        }
        let mut bound_inputs = BTreeSet::new();
        for edge in &schema.edges {
            let from = descriptors
                .get(&edge.from.node)
                .ok_or(CompileError::InvalidSubgraph(schema.id))?;
            let to = descriptors
                .get(&edge.to.node)
                .ok_or(CompileError::InvalidSubgraph(schema.id))?;
            let output = from
                .outputs
                .iter()
                .find(|port| port.name == edge.from.port)
                .ok_or(CompileError::InvalidSubgraph(schema.id))?;
            let input = to
                .inputs
                .iter()
                .find(|port| port.name == edge.to.port)
                .ok_or(CompileError::InvalidSubgraph(schema.id))?;
            if !port_contract_satisfies(output, input) || !bound_inputs.insert(edge.to.clone()) {
                return Err(CompileError::InvalidSubgraph(schema.id));
            }
        }
        for port in &schema.inputs {
            let descriptor = descriptors
                .get(&port.inner.node)
                .ok_or(CompileError::InvalidSubgraph(schema.id))?;
            if port.name.is_empty()
                || !descriptor
                    .inputs
                    .iter()
                    .any(|input| input.name == port.inner.port)
                || !bound_inputs.insert(port.inner.clone())
            {
                return Err(CompileError::InvalidSubgraph(schema.id));
            }
        }
        for port in &schema.outputs {
            let descriptor = descriptors
                .get(&port.inner.node)
                .ok_or(CompileError::InvalidSubgraph(schema.id))?;
            if port.name.is_empty()
                || !descriptor
                    .outputs
                    .iter()
                    .any(|output| output.name == port.inner.port)
            {
                return Err(CompileError::InvalidSubgraph(schema.id));
            }
        }
        if descriptors.iter().any(|(node, descriptor)| {
            descriptor.inputs.iter().any(|input| {
                !input.optional
                    && !bound_inputs.contains(&crate::PortRef {
                        node: *node,
                        port: input.name.clone(),
                    })
            })
        }) || !subgraph_is_acyclic(&schema)
        {
            return Err(CompileError::InvalidSubgraph(schema.id));
        }
        self.subgraphs.insert(schema.id, schema);
        Ok(())
    }

    pub fn register_kernel(&mut self, mut kernel: KernelDescriptor) -> Result<(), CompileError> {
        kernel.input_layouts.sort_unstable();
        kernel.input_layouts.dedup();
        kernel.output_layouts.sort_unstable();
        kernel.output_layouts.dedup();
        let id = kernel.id;
        if self.kernels.contains_key(&id) {
            return Err(CompileError::DuplicateKernel(id));
        }
        let conversion_role = kernel.conversion.is_some();
        if conversion_role != kernel.implements.is_empty()
            || kernel.resources.threads == 0
            || kernel.conversion.as_ref().is_some_and(|conversion| {
                conversion.max_input_bytes == 0
                    || conversion.max_output_bytes == 0
                    || conversion.semantic_type.is_empty()
                    || conversion.from == conversion.to
                    || !kernel.input_layouts.contains(&conversion.from)
                    || !kernel.output_layouts.contains(&conversion.to)
                    || kernel.determinism != crate::model::Determinism::BitExact
            })
        {
            return Err(CompileError::InvalidKernelContract(id));
        }
        self.kernels.insert(id, kernel);
        Ok(())
    }

    /// Decode an untrusted physical plan, bind it to a trusted realm/PlanId,
    /// and verify every executable step against this registry before use.
    pub fn decode_authorized_plan(
        &self,
        bytes: &[u8],
        limits: crate::PlanLimits,
        authorization: crate::PlanAuthorization,
    ) -> Result<AuthorizedPlan, crate::PlanDecodeError> {
        let plan = CompiledPlan::from_authorized_aot_bytes(bytes, limits, authorization)?;
        let target = plan.realm.target();
        for node in &plan.nodes {
            let kernel = self
                .kernels
                .get(&node.kernel)
                .ok_or(crate::PlanDecodeError::UnauthorizedPlan)?;
            if kernel.target != target
                || kernel.implements != node.semantic_types
                || kernel.implementation_id != node.implementation_id
                || kernel.conversion != node.conversion
                || kernel.resources != node.resources
                || kernel.determinism != node.determinism
                || kernel.lowering != node.lowering
            {
                return Err(crate::PlanDecodeError::UnauthorizedPlan);
            }
            if let Some(conversion) = &node.conversion {
                if !kernel.implements.is_empty()
                    || node.input_ports.as_slice() != ["input"]
                    || node.output_ports.as_slice() != ["output"]
                    || node.input_contracts.len() != 1
                    || node.output_contracts.len() != 1
                    || !conversion_contracts_match(
                        &node.input_contracts[0],
                        &node.output_contracts[0],
                        conversion,
                    )
                    || node.partiality != crate::model::Partiality::Atomic
                    || !node.failure.domains.is_empty()
                    || node.state != StateContract::stateless()
                    || !node.subgraph_path.is_empty()
                {
                    return Err(crate::PlanDecodeError::UnauthorizedPlan);
                }
                continue;
            }
            let mut semantic_descriptors = Vec::with_capacity(node.semantic_types.len());
            for semantic_type in &node.semantic_types {
                semantic_descriptors.push(
                    self.descriptors
                        .get(&(semantic_type.type_name.clone(), semantic_type.version))
                        .ok_or(crate::PlanDecodeError::UnauthorizedPlan)?,
                );
            }
            let first = semantic_descriptors
                .first()
                .ok_or(crate::PlanDecodeError::UnauthorizedPlan)?;
            let last = semantic_descriptors
                .last()
                .ok_or(crate::PlanDecodeError::UnauthorizedPlan)?;
            if semantic_descriptors.iter().zip(&node.semantic_configs).any(
                |(descriptor, config)| match descriptor.config.canonicalize(config) {
                    Ok(canonical) => canonical != *config,
                    Err(_) => true,
                },
            ) {
                return Err(crate::PlanDecodeError::UnauthorizedPlan);
            }
            if semantic_descriptors
                .iter()
                .any(|descriptor| kernel.determinism > descriptor.determinism)
                || !fused_layouts_compatible(first, last, kernel)
                || node.input_bindings.len() != first.inputs.len()
                || node.output_bindings.len() != last.outputs.len()
                || node.input_ports
                    != first
                        .inputs
                        .iter()
                        .map(|port| port.name.clone())
                        .collect::<Vec<_>>()
                || node.input_contracts.len() != first.inputs.len()
                || node.output_contracts.len() != last.outputs.len()
                || node
                    .input_contracts
                    .iter()
                    .any(|contract| !kernel.input_layouts.contains(&contract.layout))
                || node
                    .output_contracts
                    .iter()
                    .any(|contract| !kernel.output_layouts.contains(&contract.layout))
                || node
                    .input_contracts
                    .iter()
                    .zip(&first.inputs)
                    .any(|(compiled, port)| !compiled_port_matches(compiled, port))
                || node
                    .output_contracts
                    .iter()
                    .zip(&last.outputs)
                    .any(|(compiled, port)| !compiled_port_matches(compiled, port))
                || node.output_ports
                    != last
                        .outputs
                        .iter()
                        .map(|port| port.name.clone())
                        .collect::<Vec<_>>()
                || first
                    .inputs
                    .iter()
                    .zip(&node.input_bindings)
                    .enumerate()
                    .any(|(input_index, (port, binding))| {
                        (!port.optional && matches!(binding, crate::model::InputBinding::Absent))
                            || match binding {
                                crate::model::InputBinding::Invocation(invocation) => plan
                                    .invocation_ports
                                    .get(*invocation as usize)
                                    .is_none_or(|invocation_port| {
                                        invocation_port.node != node.semantic_nodes[0]
                                            || invocation_port.port
                                                != first.inputs[input_index].name
                                    }),
                                _ => false,
                            }
                    })
            {
                return Err(crate::PlanDecodeError::UnauthorizedPlan);
            }
            if semantic_descriptors.len() == 1 {
                if node.effect != first.effect
                    || node.retry_limit != first.retry_limit
                    || node.state != first.state
                    || node.subgraph_path
                        != first
                            .subgraph
                            .iter()
                            .map(|lowering| lowering.subgraph)
                            .collect::<Vec<_>>()
                    || node.partiality != first.partiality
                    || node.failure != first.failure
                {
                    return Err(crate::PlanDecodeError::UnauthorizedPlan);
                }
                if let Some(lowering) = &first.subgraph {
                    validate_subgraph_path(
                        &self.subgraphs,
                        lowering.subgraph,
                        limits.max_subgraph_depth,
                        limits.max_contract_entries,
                    )
                    .map_err(|_| crate::PlanDecodeError::UnauthorizedPlan)?;
                    validate_port_map(first, lowering, &self.subgraphs)
                        .map_err(|_| crate::PlanDecodeError::UnauthorizedPlan)?;
                }
            } else if node.effect != crate::model::Effect::Pure
                || node.retry_limit != 0
                || node.state != StateContract::stateless()
                || !node.subgraph_path.is_empty()
                || node.partiality != crate::model::Partiality::Atomic
                || !node.failure.domains.is_empty()
                || semantic_descriptors.iter().any(|descriptor| {
                    descriptor.effect != crate::model::Effect::Pure
                        || descriptor.state.scope != StateScope::Stateless
                        || descriptor.retry_limit != 0
                        || descriptor.state.checkpointable()
                        || descriptor.partiality != crate::model::Partiality::Atomic
                        || !descriptor.failure.domains.is_empty()
                })
            {
                return Err(crate::PlanDecodeError::UnauthorizedPlan);
            }
        }
        Ok(AuthorizedPlan::new(plan))
    }
}

fn normalize_contract(proof: &mut crate::ProofContract, policy: &mut crate::PolicyContract) {
    proof.requires.sort_unstable();
    proof.requires.dedup();
    proof.provides.sort_unstable();
    proof.provides.dedup();
    proof.invalidates.sort_unstable();
    proof.invalidates.dedup();
    policy.requires.sort_unstable();
    policy.requires.dedup();
    policy.adds.sort_unstable();
    policy.adds.dedup();
}

fn valid_contract_names(proof: &crate::ProofContract, policy: &crate::PolicyContract) -> bool {
    proof
        .requires
        .iter()
        .chain(&proof.provides)
        .chain(&proof.invalidates)
        .chain(&policy.requires)
        .chain(&policy.adds)
        .all(|name| !name.is_empty())
}

fn subgraph_implements_descriptor(
    schema: &crate::SubgraphSchema,
    descriptor: &NodeDescriptor,
    descriptors: &BTreeMap<(String, u32), NodeDescriptor>,
) -> bool {
    fn interface_matches(
        schema: &crate::SubgraphSchema,
        interface: &crate::SubgraphInterfacePort,
        expected: &PortDescriptor,
        descriptors: &BTreeMap<(String, u32), NodeDescriptor>,
        input: bool,
    ) -> bool {
        let Some(node) = schema
            .nodes
            .iter()
            .find(|node| node.id == interface.inner.node)
        else {
            return false;
        };
        let Some(descriptor) =
            descriptors.get(&(node.node_type.type_name.clone(), node.node_type.version))
        else {
            return false;
        };
        let ports = if input {
            &descriptor.inputs
        } else {
            &descriptor.outputs
        };
        let Some(inner) = ports.iter().find(|port| port.name == interface.inner.port) else {
            return false;
        };
        let mut inner = inner.clone();
        inner.name = interface.name.clone();
        &inner == expected
    }

    schema.inputs.len() == descriptor.inputs.len()
        && schema.outputs.len() == descriptor.outputs.len()
        && descriptor.inputs.iter().all(|expected| {
            schema
                .inputs
                .iter()
                .find(|interface| interface.name == expected.name)
                .is_some_and(|interface| {
                    interface_matches(schema, interface, expected, descriptors, true)
                })
        })
        && descriptor.outputs.iter().all(|expected| {
            schema
                .outputs
                .iter()
                .find(|interface| interface.name == expected.name)
                .is_some_and(|interface| {
                    interface_matches(schema, interface, expected, descriptors, false)
                })
        })
}

pub(crate) fn valid_port_contract(port: &PortDescriptor) -> bool {
    let extent = &port.extent;
    if extent.maximum_shape.len() != usize::from(extent.rank)
        || extent.max_elements == 0
        || extent.maximum_shape.contains(&0)
    {
        return false;
    }
    let shape_product = extent
        .maximum_shape
        .iter()
        .try_fold(1u64, |product, size| product.checked_mul(*size));
    if shape_product.is_none_or(|product| product > extent.max_elements) {
        return false;
    }
    if port.lease.access == crate::LeaseAccess::ExclusiveWrite
        && port.lease.lifetime == crate::LeaseLifetime::Session
    {
        return false;
    }
    valid_contract_names(&port.proof, &port.policy)
        && !matches!(&port.abir.root, crate::AbirRootType::Unknown(name) if name.is_empty())
        && !matches!(&port.abir.view, crate::AbirViewType::Unknown(name) if name.is_empty())
}

pub(crate) fn valid_state_contract(state: &StateContract) -> bool {
    match state.scope {
        StateScope::Stateless => {
            state.max_bytes == 0
                && state.checkpoint.mode == crate::CheckpointMode::Disabled
                && state.checkpoint.max_snapshot_bytes == 0
                && state.checkpoint.max_interval_invocations == 0
        }
        StateScope::Invocation => {
            state.max_bytes > 0
                && state.checkpoint.mode == crate::CheckpointMode::Disabled
                && state.checkpoint.max_snapshot_bytes == 0
                && state.checkpoint.max_interval_invocations == 0
        }
        StateScope::Session => {
            state.max_bytes > 0
                && match state.checkpoint.mode {
                    crate::CheckpointMode::Disabled => {
                        state.checkpoint.max_snapshot_bytes == 0
                            && state.checkpoint.max_interval_invocations == 0
                    }
                    crate::CheckpointMode::Optional | crate::CheckpointMode::Required => {
                        state.checkpoint.max_snapshot_bytes > 0
                            && state.checkpoint.max_snapshot_bytes <= state.max_bytes
                            && state.checkpoint.max_interval_invocations > 0
                    }
                }
        }
        StateScope::Durable => {
            state.max_bytes > 0
                && state.checkpoint.mode == crate::CheckpointMode::Required
                && state.checkpoint.max_snapshot_bytes > 0
                && state.checkpoint.max_snapshot_bytes <= state.max_bytes
                && state.checkpoint.max_interval_invocations > 0
        }
    }
}

fn port_contract_satisfies(output: &PortDescriptor, input: &PortDescriptor) -> bool {
    output.semantic_type == input.semantic_type
        && output.abir == input.abir
        && (!output.optional || input.optional)
        && output.max_bytes <= input.max_bytes
        && output.extent.rank == input.extent.rank
        && output.extent.max_elements <= input.extent.max_elements
        && output
            .extent
            .maximum_shape
            .iter()
            .zip(&input.extent.maximum_shape)
            .all(|(actual, maximum)| actual <= maximum)
        && (!output.extent.ragged || input.extent.ragged)
        && (!output.extent.sparse || input.extent.sparse)
        && input
            .proof
            .requires
            .iter()
            .all(|required| output.proof.provides.contains(required))
        && input
            .policy
            .requires
            .iter()
            .all(|required| output.policy.adds.contains(required))
        && output.fidelity.maximum_loss <= input.fidelity.maximum_loss
        && output.fidelity.minimum_input >= input.fidelity.minimum_input
        && output.lease == input.lease
}

pub(crate) fn compiled_port_contract_satisfies(
    output: &CompiledPortContract,
    input: &CompiledPortContract,
) -> bool {
    output.layout == input.layout
        && output.semantic_type == input.semantic_type
        && output.abir == input.abir
        && (!output.optional || input.optional)
        && output.max_bytes <= input.max_bytes
        && output.extent.rank == input.extent.rank
        && output.extent.max_elements <= input.extent.max_elements
        && output
            .extent
            .maximum_shape
            .iter()
            .zip(&input.extent.maximum_shape)
            .all(|(actual, maximum)| actual <= maximum)
        && (!output.extent.ragged || input.extent.ragged)
        && (!output.extent.sparse || input.extent.sparse)
        && input
            .proof
            .requires
            .iter()
            .all(|required| output.proof.provides.contains(required))
        && input
            .policy
            .requires
            .iter()
            .all(|required| output.policy.adds.contains(required))
        && output.fidelity.maximum_loss <= input.fidelity.maximum_loss
        && output.fidelity.minimum_input >= input.fidelity.minimum_input
        && output.lease == input.lease
}

fn select_layout(port: &PortDescriptor, kernel_layouts: &[Layout]) -> Layout {
    port.layouts
        .iter()
        .filter(|layout| kernel_layouts.contains(layout))
        .copied()
        .min()
        .expect("kernel compatibility was checked")
}

fn compiled_port_contract(port: &PortDescriptor, layout: Layout) -> CompiledPortContract {
    CompiledPortContract {
        name: port.name.clone(),
        semantic_type: port.semantic_type.clone(),
        optional: port.optional,
        layout,
        max_bytes: port.max_bytes,
        abir: port.abir.clone(),
        proof: port.proof.clone(),
        policy: port.policy.clone(),
        fidelity: port.fidelity.clone(),
        extent: port.extent.clone(),
        lease: port.lease.clone(),
    }
}

fn conversion_port_contract(
    port: &PortDescriptor,
    layout: Layout,
    name: &str,
    max_bytes: u64,
) -> CompiledPortContract {
    let mut contract = compiled_port_contract(port, layout);
    contract.name = name.to_string();
    contract.optional = false;
    contract.max_bytes = max_bytes;
    contract
}

fn compiled_port_matches(compiled: &CompiledPortContract, descriptor: &PortDescriptor) -> bool {
    compiled.name == descriptor.name
        && compiled.semantic_type == descriptor.semantic_type
        && compiled.optional == descriptor.optional
        && descriptor.layouts.contains(&compiled.layout)
        && compiled.max_bytes == descriptor.max_bytes
        && compiled.abir == descriptor.abir
        && compiled.proof == descriptor.proof
        && compiled.policy == descriptor.policy
        && compiled.fidelity == descriptor.fidelity
        && compiled.extent == descriptor.extent
        && compiled.lease == descriptor.lease
}

fn conversion_contracts_match(
    input: &CompiledPortContract,
    output: &CompiledPortContract,
    conversion: &crate::LayoutConversion,
) -> bool {
    input.name == "input"
        && output.name == "output"
        && !input.optional
        && !output.optional
        && input.semantic_type == conversion.semantic_type
        && output.semantic_type == conversion.semantic_type
        && input.layout == conversion.from
        && output.layout == conversion.to
        && input.max_bytes <= conversion.max_input_bytes
        && output.max_bytes == conversion.max_output_bytes
        && input.abir == output.abir
        && input.proof == output.proof
        && input.policy == output.policy
        && input.fidelity == output.fidelity
        && input.extent == output.extent
        && input.lease == output.lease
}

fn synchronize_port_layouts(nodes: &mut [CompiledNode], buffers: &[BufferPlan]) {
    for node in nodes {
        for (contract, binding) in node.input_contracts.iter_mut().zip(&node.input_bindings) {
            if let crate::InputBinding::Buffer(buffer) = binding {
                contract.layout = buffers[buffer.0 as usize].layout;
            }
        }
        for (contract, binding) in node.output_contracts.iter_mut().zip(&node.output_bindings) {
            if let OutputBinding::Buffer(buffer) = binding {
                contract.layout = buffers[buffer.0 as usize].layout;
            }
        }
    }
}

/// Domain-separated semantic identity for a normalized hierarchical schema.
pub fn subgraph_identity(schema: &crate::SubgraphSchema) -> crate::SubgraphId {
    let mut hasher = blake3::Hasher::new_derive_key("blut.subgraph.v1");
    put_u32(&mut hasher, schema.version);
    let mut nodes = schema.nodes.clone();
    nodes.sort_by_key(|node| node.id);
    put_u32(&mut hasher, nodes.len() as u32);
    for node in &nodes {
        put_u32(&mut hasher, node.id.0);
        put_str(&mut hasher, &node.node_type.type_name);
        put_u32(&mut hasher, node.node_type.version);
        put_u32(&mut hasher, node.config.len() as u32);
        for (key, value) in &node.config {
            put_str(&mut hasher, key);
            hash_config_value(&mut hasher, value);
        }
        match node.child {
            Some(child) => {
                hasher.update(&[1]);
                hasher.update(&child.0);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    let mut edges = schema.edges.clone();
    edges.sort_by_key(|edge| (edge.from.clone(), edge.to.clone()));
    put_u32(&mut hasher, edges.len() as u32);
    for edge in &edges {
        put_port_ref(&mut hasher, &edge.from);
        put_port_ref(&mut hasher, &edge.to);
    }
    for ports in [&schema.inputs, &schema.outputs] {
        let mut ports = ports.clone();
        ports.sort_unstable();
        put_u32(&mut hasher, ports.len() as u32);
        for port in &ports {
            put_str(&mut hasher, &port.name);
            put_port_ref(&mut hasher, &port.inner);
        }
    }
    crate::SubgraphId(*hasher.finalize().as_bytes())
}

fn subgraph_is_acyclic(schema: &crate::SubgraphSchema) -> bool {
    let mut indegree: BTreeMap<_, usize> = schema.nodes.iter().map(|node| (node.id, 0)).collect();
    let mut outgoing: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for edge in &schema.edges {
        let Some(degree) = indegree.get_mut(&edge.to.node) else {
            return false;
        };
        *degree += 1;
        outgoing
            .entry(edge.from.node)
            .or_default()
            .push(edge.to.node);
    }
    let mut ready: Vec<_> = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(*node))
        .collect();
    let mut visited = 0usize;
    while let Some(node) = ready.pop() {
        visited += 1;
        for target in outgoing.get(&node).into_iter().flatten() {
            let Some(degree) = indegree.get_mut(target) else {
                return false;
            };
            *degree -= 1;
            if *degree == 0 {
                ready.push(*target);
            }
        }
    }
    visited == schema.nodes.len()
}

fn validate_subgraph_path(
    schemas: &BTreeMap<crate::SubgraphId, crate::SubgraphSchema>,
    root: crate::SubgraphId,
    max_depth: usize,
    max_entries: usize,
) -> Result<(), CompileError> {
    let mut pending = alloc::vec![(root, 1usize, BTreeSet::new())];
    let mut searched = 0usize;
    while let Some((id, depth, mut ancestors)) = pending.pop() {
        let schema = schemas.get(&id).ok_or(CompileError::UnknownSubgraph(id))?;
        searched = searched
            .checked_add(schema.nodes.len())
            .and_then(|count| count.checked_add(schema.edges.len()))
            .and_then(|count| count.checked_add(schema.inputs.len()))
            .and_then(|count| count.checked_add(schema.outputs.len()))
            .ok_or(CompileError::SubgraphEntryLimitExceeded)?;
        if searched > max_entries {
            return Err(CompileError::SubgraphEntryLimitExceeded);
        }
        if depth > max_depth {
            return Err(CompileError::SubgraphDepthExceeded);
        }
        if !ancestors.insert(id) {
            return Err(CompileError::InvalidSubgraph(id));
        }
        for child in schema.nodes.iter().filter_map(|node| node.child) {
            pending.push((child, depth + 1, ancestors.clone()));
        }
    }
    Ok(())
}

fn validate_port_map(
    descriptor: &NodeDescriptor,
    lowering: &crate::SubgraphLowering,
    schemas: &BTreeMap<crate::SubgraphId, crate::SubgraphSchema>,
) -> Result<(), CompileError> {
    let schema = schemas
        .get(&lowering.subgraph)
        .ok_or(CompileError::UnknownSubgraph(lowering.subgraph))?;
    let input_names: BTreeSet<_> = descriptor
        .inputs
        .iter()
        .map(|port| port.name.as_str())
        .collect();
    let output_names: BTreeSet<_> = descriptor
        .outputs
        .iter()
        .map(|port| port.name.as_str())
        .collect();
    let mapped_inputs: BTreeSet<_> = lowering
        .input_map
        .iter()
        .map(|map| map.outer.as_str())
        .collect();
    let mapped_outputs: BTreeSet<_> = lowering
        .output_map
        .iter()
        .map(|map| map.outer.as_str())
        .collect();
    let inner_inputs: BTreeSet<_> = schema
        .inputs
        .iter()
        .map(|port| port.name.as_str())
        .collect();
    let inner_outputs: BTreeSet<_> = schema
        .outputs
        .iter()
        .map(|port| port.name.as_str())
        .collect();
    if input_names != mapped_inputs
        || output_names != mapped_outputs
        || lowering.input_map.len() != input_names.len()
        || lowering.output_map.len() != output_names.len()
        || lowering
            .input_map
            .iter()
            .any(|map| !inner_inputs.contains(map.inner.as_str()))
        || lowering
            .output_map
            .iter()
            .any(|map| !inner_outputs.contains(map.inner.as_str()))
        || lowering
            .input_map
            .iter()
            .map(|map| map.inner.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != lowering.input_map.len()
        || lowering
            .output_map
            .iter()
            .map(|map| map.inner.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != lowering.output_map.len()
    {
        return Err(CompileError::InvalidSubgraph(lowering.subgraph));
    }
    Ok(())
}

pub struct Compiler<'a> {
    registry: &'a KernelRegistry,
    realm: ExecutionRealm,
    max_peak_bytes: u64,
    fuse: bool,
    limits: CompileLimits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileLimits {
    pub max_search_states: usize,
    pub max_conversion_states: usize,
    pub max_semantic_nodes: usize,
    pub max_steps: usize,
    pub max_buffers: usize,
    pub max_subgraph_depth: usize,
    pub max_subgraph_entries: usize,
    pub max_feedback_edges: usize,
    pub max_persistent_state_bytes: u64,
}

type LoweredCandidate = (
    u64,
    usize,
    Vec<KernelId>,
    Vec<CompiledNode>,
    Vec<BufferPlan>,
);

impl Default for CompileLimits {
    fn default() -> Self {
        Self {
            max_search_states: 65_536,
            max_conversion_states: 65_536,
            max_semantic_nodes: 65_536,
            max_steps: 65_536,
            max_buffers: 262_144,
            max_subgraph_depth: 16,
            max_subgraph_entries: 65_536,
            max_feedback_edges: 65_536,
            max_persistent_state_bytes: 64 * 1024 * 1024,
        }
    }
}

impl<'a> Compiler<'a> {
    pub const fn new(registry: &'a KernelRegistry, realm: ExecutionRealm) -> Self {
        Self {
            registry,
            realm,
            max_peak_bytes: u64::MAX,
            fuse: true,
            limits: CompileLimits {
                max_search_states: 65_536,
                max_conversion_states: 65_536,
                max_semantic_nodes: 65_536,
                max_steps: 65_536,
                max_buffers: 262_144,
                max_subgraph_depth: 16,
                max_subgraph_entries: 65_536,
                max_feedback_edges: 65_536,
                max_persistent_state_bytes: 64 * 1024 * 1024,
            },
        }
    }

    pub const fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.max_peak_bytes = bytes;
        self
    }

    pub const fn with_fusion(mut self, enabled: bool) -> Self {
        self.fuse = enabled;
        self
    }

    pub const fn with_limits(mut self, limits: CompileLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn compile(&self, graph: &Graph) -> Result<AuthorizedPlan, CompileError> {
        if graph.version != 3 {
            return Err(CompileError::UnsupportedGraphVersion(graph.version));
        }
        if graph.nodes.is_empty() {
            return Err(CompileError::EmptyGraph);
        }
        if graph
            .required_proofs
            .iter()
            .chain(&graph.policy)
            .any(|name| name.is_empty())
        {
            return Err(CompileError::InvalidGraphContract);
        }
        if graph.nodes.len() > self.limits.max_semantic_nodes {
            return Err(CompileError::CompileLimitExceeded);
        }

        let mut normalized_graph = graph.clone();
        let mut seen_nodes = BTreeSet::new();
        for node in &normalized_graph.nodes {
            if !seen_nodes.insert(node.id) {
                return Err(CompileError::DuplicateNode(node.id));
            }
        }

        let mut descriptors = BTreeMap::new();
        let target = self.realm.target();
        let required_caps: BTreeSet<_> = graph.required_capabilities.iter().collect();
        for node in &normalized_graph.nodes {
            let key = (node.descriptor.clone(), node.descriptor_version);
            let descriptor = self
                .registry
                .descriptors
                .get(&key)
                .ok_or_else(|| CompileError::UnknownDescriptor(key.0.clone(), key.1))?;
            if !descriptor.targets.contains(&target) {
                return Err(CompileError::TargetUnsupported(node.id, target));
            }
            if descriptor.retry_limit > 0
                && matches!(descriptor.effect, crate::model::Effect::AtMostOnce)
            {
                return Err(CompileError::UnsafeRetry(node.id));
            }
            if matches!(
                descriptor.state.scope,
                StateScope::Session | StateScope::Durable
            ) && normalized_graph.session.is_none()
            {
                return Err(CompileError::InvalidState(node.id));
            }
            if let Some(lowering) = &descriptor.subgraph {
                validate_subgraph_path(
                    &self.registry.subgraphs,
                    lowering.subgraph,
                    self.limits.max_subgraph_depth,
                    self.limits.max_subgraph_entries,
                )?;
                validate_port_map(descriptor, lowering, &self.registry.subgraphs)?;
            }
            for capability in &descriptor.capabilities {
                if !required_caps.contains(capability) {
                    return Err(CompileError::CapabilityMissing(capability.0.clone()));
                }
            }
            descriptors.insert(node.id, descriptor);
        }
        for node in &mut normalized_graph.nodes {
            node.config = descriptors[&node.id]
                .config
                .canonicalize(&node.config)
                .map_err(|error| CompileError::InvalidConfig(node.id, error))?;
        }
        if normalized_graph.session.as_ref().is_some_and(|session| {
            session.namespace.is_empty()
                || session.max_concurrent_sessions == 0
                || session.max_idle_millis == 0
        }) {
            return Err(CompileError::InvalidSession);
        }
        if !normalized_graph.feedback.is_empty() && normalized_graph.session.is_none() {
            return Err(CompileError::InvalidSession);
        }
        let mut nodes = BTreeMap::new();
        for node in &normalized_graph.nodes {
            nodes.insert(node.id, node);
        }
        let supplied_caps: BTreeSet<_> = descriptors
            .values()
            .flat_map(|descriptor| descriptor.capabilities.iter())
            .collect();
        if let Some(extra) = required_caps.difference(&supplied_caps).next() {
            return Err(CompileError::CapabilityUnsupported(extra.0.clone()));
        }

        let invocation_ports = self.verify_edges(&normalized_graph, &descriptors)?;
        let order = topological_order(&normalized_graph)?;
        let (proofs, policy, fidelity) =
            propagate_contracts(&normalized_graph, &order, &descriptors)?;
        let (mut compiled_nodes, buffers, peak_bytes) =
            self.select_and_lower(&normalized_graph, &order, &nodes, &descriptors, target)?;
        let (feedback, feedback_bytes) = lower_feedback(
            &normalized_graph,
            &mut compiled_nodes,
            self.limits.max_feedback_edges,
        )?;
        let node_state_bytes = compiled_nodes
            .iter()
            .filter(|node| matches!(node.state.scope, StateScope::Session | StateScope::Durable))
            .try_fold(0u64, |total, node| {
                total
                    .checked_add(node.state.max_bytes)
                    .ok_or(CompileError::ResourceOverflow)
            })?;
        let persistent_state_bytes = node_state_bytes
            .checked_add(feedback_bytes)
            .ok_or(CompileError::ResourceOverflow)?;
        if persistent_state_bytes > self.limits.max_persistent_state_bytes {
            return Err(CompileError::ResourceOverflow);
        }
        let graph_id = GraphId(hash_graph(&normalized_graph, &descriptors));
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id,
            plan_id: PlanId([0; 32]),
            realm: self.realm,
            order,
            nodes: compiled_nodes,
            buffers,
            feedback,
            invocation_ports,
            propagated_proofs: proofs,
            propagated_policy: policy,
            resulting_fidelity: fidelity,
            peak_bytes,
            persistent_state_bytes,
            session: normalized_graph.session.clone(),
        };
        plan.plan_id = PlanId(hash_plan(&plan));
        Ok(AuthorizedPlan::new(plan))
    }

    fn verify_edges(
        &self,
        graph: &Graph,
        descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
    ) -> Result<Vec<crate::model::PortRef>, CompileError> {
        let mut bound = BTreeSet::new();
        for edge in &graph.edges {
            let from = descriptors
                .get(&edge.from.node)
                .ok_or_else(|| CompileError::UnknownPort(edge.from.node, edge.from.port.clone()))?;
            let to = descriptors
                .get(&edge.to.node)
                .ok_or_else(|| CompileError::UnknownPort(edge.to.node, edge.to.port.clone()))?;
            let output = find_port(&from.outputs, edge.from.node, &edge.from.port)?;
            let input = find_port(&to.inputs, edge.to.node, &edge.to.port)?;
            if output.max_bytes == 0 {
                return Err(CompileError::InvalidPortSize(
                    edge.from.node,
                    edge.from.port.clone(),
                ));
            }
            if output.semantic_type != input.semantic_type {
                return Err(CompileError::TypeMismatch(
                    output.semantic_type.clone(),
                    input.semantic_type.clone(),
                ));
            }
            if !port_contract_satisfies(output, input) {
                return Err(CompileError::PortContractMismatch(
                    edge.from.node,
                    edge.from.port.clone(),
                    edge.to.node,
                    edge.to.port.clone(),
                ));
            }
            if !bound.insert(edge.to.clone()) {
                return Err(CompileError::DuplicateInput(
                    edge.to.node,
                    edge.to.port.clone(),
                ));
            }
        }
        for feedback in &graph.feedback {
            let from = descriptors.get(&feedback.from.node).ok_or_else(|| {
                CompileError::UnknownPort(feedback.from.node, feedback.from.port.clone())
            })?;
            let to = descriptors.get(&feedback.to.node).ok_or_else(|| {
                CompileError::UnknownPort(feedback.to.node, feedback.to.port.clone())
            })?;
            let output = find_port(&from.outputs, feedback.from.node, &feedback.from.port)?;
            let input = find_port(&to.inputs, feedback.to.node, &feedback.to.port)?;
            if feedback.delay.invocations == 0
                || (matches!(&feedback.delay.initial, crate::DelayInitial::Absent)
                    && !input.optional)
                || !port_contract_satisfies(output, input)
                || output.max_bytes > input.max_bytes
                || !bound.insert(feedback.to.clone())
            {
                return Err(CompileError::InvalidFeedback(
                    feedback.to.node,
                    feedback.to.port.clone(),
                ));
            }
        }
        let mut invocation_ports = graph.invocation_inputs.clone();
        invocation_ports.sort_unstable();
        for pair in invocation_ports.windows(2) {
            if pair[0] == pair[1] {
                return Err(CompileError::DuplicateInvocation(
                    pair[0].node,
                    pair[0].port.clone(),
                ));
            }
        }
        for invocation in &invocation_ports {
            let descriptor = descriptors.get(&invocation.node).ok_or_else(|| {
                CompileError::UnknownPort(invocation.node, invocation.port.clone())
            })?;
            find_port(&descriptor.inputs, invocation.node, &invocation.port)?;
            if bound.contains(invocation) {
                return Err(CompileError::DuplicateInput(
                    invocation.node,
                    invocation.port.clone(),
                ));
            }
        }
        let invocation_set: BTreeSet<_> = invocation_ports.iter().cloned().collect();
        for (node_id, descriptor) in descriptors {
            for input in descriptor.inputs.iter().filter(|port| !port.optional) {
                let port_ref = crate::model::PortRef {
                    node: *node_id,
                    port: input.name.clone(),
                };
                if !bound.contains(&port_ref) && !invocation_set.contains(&port_ref) {
                    return Err(CompileError::MissingInput(*node_id, input.name.clone()));
                }
            }
        }
        Ok(invocation_ports)
    }

    fn kernel_candidates(
        &self,
        order: &[NodeId],
        nodes: &BTreeMap<NodeId, &crate::model::NodeInstance>,
        target: Target,
    ) -> Result<BTreeMap<NodeId, Vec<&KernelDescriptor>>, CompileError> {
        let mut candidates = BTreeMap::new();
        for node_id in order {
            let node = nodes[node_id];
            let mut node_candidates: Vec<_> = self
                .registry
                .kernels
                .values()
                .filter(|kernel| {
                    kernel.implements.as_slice()
                        == [NodeTypeRef {
                            type_name: node.descriptor.clone(),
                            version: node.descriptor_version,
                        }]
                        && kernel.target == target
                        && kernel.determinism
                            <= self.registry.descriptors
                                [&(node.descriptor.clone(), node.descriptor_version)]
                                .determinism
                        && descriptor_layouts_compatible(
                            self.registry
                                .descriptors
                                .get(&(node.descriptor.clone(), node.descriptor_version))
                                .expect("descriptor was verified"),
                            kernel,
                        )
                })
                .collect();
            node_candidates.sort_by_key(|kernel| {
                (
                    kernel.resources.peak_bytes,
                    kernel.resources.scratch_bytes,
                    kernel.implementation_id,
                    kernel.id,
                )
            });
            if node_candidates.is_empty() {
                return Err(CompileError::KernelUnavailable(*node_id, target));
            }
            candidates.insert(*node_id, node_candidates);
        }
        Ok(candidates)
    }

    fn select_and_lower(
        &self,
        graph: &Graph,
        order: &[NodeId],
        instances: &BTreeMap<NodeId, &crate::model::NodeInstance>,
        descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
        target: Target,
    ) -> Result<(Vec<CompiledNode>, Vec<BufferPlan>, u64), CompileError> {
        let candidates = self.kernel_candidates(order, instances, target)?;
        let assignment_count = order.iter().try_fold(1usize, |count, node| {
            let count = count
                .checked_mul(candidates[node].len())
                .ok_or(CompileError::SearchLimitExceeded)?;
            if count > self.limits.max_search_states {
                return Err(CompileError::SearchLimitExceeded);
            }
            Ok(count)
        })?;
        let region_limit = self
            .limits
            .max_search_states
            .checked_div(assignment_count)
            .filter(|limit| *limit > 0)
            .ok_or(CompileError::SearchLimitExceeded)?;

        let mut best: Option<LoweredCandidate> = None;
        let mut saw_layout_failure = false;
        let mut saw_feedback_failure = false;
        let mut saw_resource_failure = false;
        for ordinal in 0..assignment_count {
            let mut remainder = ordinal;
            let mut selected = alloc::vec![0usize; order.len()];
            for (index, node) in order.iter().enumerate().rev() {
                selected[index] = remainder % candidates[node].len();
                remainder /= candidates[node].len();
            }
            let assignment: BTreeMap<_, _> = order
                .iter()
                .enumerate()
                .map(|(index, node)| (*node, candidates[node][selected[index]]))
                .collect();
            match lower_physical_plan(
                PhysicalLowering {
                    graph,
                    order,
                    descriptors,
                    kernels: &assignment,
                    registry: self.registry,
                    target,
                },
                self.fuse,
                PhysicalSearchLimits {
                    region_candidates: region_limit,
                    conversion_states: self.limits.max_conversion_states,
                    max_peak_bytes: self.max_peak_bytes,
                    max_steps: self.limits.max_steps,
                    max_buffers: self.limits.max_buffers,
                },
            ) {
                Ok((mut nodes, buffers, peak))
                    if peak <= self.max_peak_bytes
                        && nodes.len() <= self.limits.max_steps
                        && buffers.len() <= self.limits.max_buffers =>
                {
                    if !align_feedback_layouts(graph, descriptors, self.registry, &mut nodes)? {
                        saw_feedback_failure = true;
                        continue;
                    }
                    if !feedback_physical_compatible(graph, &nodes)? {
                        saw_feedback_failure = true;
                        continue;
                    }
                    let implementation_order = nodes.iter().map(|node| node.kernel).collect();
                    let score = (peak, nodes.len(), implementation_order);
                    if best
                        .as_ref()
                        .is_none_or(|current| score < (current.0, current.1, current.2.clone()))
                    {
                        best = Some((score.0, score.1, score.2, nodes, buffers));
                    }
                }
                Ok((nodes, buffers, _))
                    if nodes.len() > self.limits.max_steps
                        || buffers.len() > self.limits.max_buffers =>
                {
                    return Err(CompileError::CompileLimitExceeded);
                }
                Ok(_) | Err(CompileError::ResourceOverflow) => saw_resource_failure = true,
                Err(CompileError::LayoutUnavailable(..)) => saw_layout_failure = true,
                Err(error) => return Err(error),
            }
        }
        match best {
            Some((peak, _, _, nodes, buffers)) => Ok((nodes, buffers, peak)),
            None if saw_feedback_failure => {
                let feedback = graph.feedback.first().expect("failure requires feedback");
                Err(CompileError::InvalidFeedback(
                    feedback.to.node,
                    feedback.to.port.clone(),
                ))
            }
            None if saw_resource_failure => Err(CompileError::ResourceOverflow),
            None if saw_layout_failure => Err(CompileError::LayoutUnavailable(
                *order.first().expect("non-empty graph"),
                "physical-lowering".to_string(),
            )),
            None => Err(CompileError::KernelUnavailable(
                *order.first().expect("non-empty graph"),
                target,
            )),
        }
    }
}

fn descriptor_layouts_compatible(descriptor: &NodeDescriptor, kernel: &KernelDescriptor) -> bool {
    descriptor.inputs.iter().all(|port| {
        port.layouts
            .iter()
            .any(|layout| kernel.input_layouts.contains(layout))
    }) && descriptor.outputs.iter().all(|port| {
        port.layouts
            .iter()
            .any(|layout| kernel.output_layouts.contains(layout))
    })
}

fn align_feedback_layouts(
    graph: &Graph,
    descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
    registry: &KernelRegistry,
    nodes: &mut [CompiledNode],
) -> Result<bool, CompileError> {
    let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for edge in &graph.feedback {
        groups.entry(edge.from.clone()).or_default().push(edge);
    }
    for (source, edges) in groups {
        let from_index = nodes
            .iter()
            .position(|node| node.semantic_nodes.contains(&source.node))
            .ok_or(CompileError::UnknownNode(source.node))?;
        let from_port = nodes[from_index]
            .output_ports
            .iter()
            .position(|port| port == &source.port)
            .ok_or_else(|| CompileError::UnknownPort(source.node, source.port.clone()))?;
        let from_descriptor = descriptors[&source.node]
            .outputs
            .iter()
            .find(|port| port.name == source.port)
            .ok_or_else(|| CompileError::UnknownPort(source.node, source.port.clone()))?;
        let from_kernel = registry
            .kernels
            .get(&nodes[from_index].kernel)
            .ok_or_else(|| CompileError::InvalidFeedback(source.node, source.port.clone()))?;
        let producer_fixed = matches!(
            nodes[from_index].output_bindings[from_port],
            OutputBinding::Buffer(_)
        );
        let mut layouts: Vec<_> = from_descriptor
            .layouts
            .iter()
            .filter(|layout| from_kernel.output_layouts.contains(layout))
            .copied()
            .collect();
        let mut consumers = Vec::with_capacity(edges.len());
        for edge in edges {
            let to_index = nodes
                .iter()
                .position(|node| node.semantic_nodes.contains(&edge.to.node))
                .ok_or(CompileError::UnknownNode(edge.to.node))?;
            let to_port = nodes[to_index]
                .input_ports
                .iter()
                .position(|port| port == &edge.to.port)
                .ok_or_else(|| CompileError::UnknownPort(edge.to.node, edge.to.port.clone()))?;
            let to_descriptor = descriptors[&edge.to.node]
                .inputs
                .iter()
                .find(|port| port.name == edge.to.port)
                .ok_or_else(|| CompileError::UnknownPort(edge.to.node, edge.to.port.clone()))?;
            let to_kernel = registry
                .kernels
                .get(&nodes[to_index].kernel)
                .ok_or_else(|| CompileError::InvalidFeedback(edge.to.node, edge.to.port.clone()))?;
            layouts.retain(|layout| {
                to_descriptor.layouts.contains(layout) && to_kernel.input_layouts.contains(layout)
            });
            consumers.push((to_index, to_port));
        }
        layouts.sort_unstable();
        layouts.dedup();
        let selected = if producer_fixed {
            let current = nodes[from_index].output_contracts[from_port].layout;
            layouts.contains(&current).then_some(current)
        } else {
            layouts.first().copied()
        };
        let Some(selected) = selected else {
            return Ok(false);
        };
        nodes[from_index].output_contracts[from_port].layout = selected;
        for (to_index, to_port) in consumers {
            nodes[to_index].input_contracts[to_port].layout = selected;
        }
    }
    Ok(true)
}

fn feedback_physical_compatible(
    graph: &Graph,
    nodes: &[CompiledNode],
) -> Result<bool, CompileError> {
    for edge in &graph.feedback {
        let from = nodes
            .iter()
            .find(|node| node.semantic_nodes.contains(&edge.from.node))
            .ok_or(CompileError::UnknownNode(edge.from.node))?;
        let from_port = from
            .output_ports
            .iter()
            .position(|port| port == &edge.from.port)
            .ok_or_else(|| CompileError::UnknownPort(edge.from.node, edge.from.port.clone()))?;
        let to = nodes
            .iter()
            .find(|node| node.semantic_nodes.contains(&edge.to.node))
            .ok_or(CompileError::UnknownNode(edge.to.node))?;
        let to_port = to
            .input_ports
            .iter()
            .position(|port| port == &edge.to.port)
            .ok_or_else(|| CompileError::UnknownPort(edge.to.node, edge.to.port.clone()))?;
        if !compiled_port_contract_satisfies(
            &from.output_contracts[from_port],
            &to.input_contracts[to_port],
        ) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn find_port<'a>(
    ports: &'a [PortDescriptor],
    node: NodeId,
    name: &str,
) -> Result<&'a PortDescriptor, CompileError> {
    ports
        .iter()
        .find(|port| port.name == name)
        .ok_or_else(|| CompileError::UnknownPort(node, name.to_string()))
}

fn topological_order(graph: &Graph) -> Result<Vec<NodeId>, CompileError> {
    let mut indegree: BTreeMap<NodeId, usize> =
        graph.nodes.iter().map(|node| (node.id, 0)).collect();
    let mut outgoing: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
    for Edge { from, to } in &graph.edges {
        if !indegree.contains_key(&from.node) || !indegree.contains_key(&to.node) {
            return Err(CompileError::UnknownNode(
                if !indegree.contains_key(&from.node) {
                    from.node
                } else {
                    to.node
                },
            ));
        }
        *indegree.get_mut(&to.node).expect("checked") += 1;
        outgoing.entry(from.node).or_default().push(to.node);
    }
    for values in outgoing.values_mut() {
        values.sort();
    }
    let mut ready: BTreeSet<NodeId> = indegree
        .iter()
        .filter_map(|(id, count)| (*count == 0).then_some(*id))
        .collect();
    let mut result = Vec::with_capacity(indegree.len());
    while let Some(id) = ready.pop_first() {
        result.push(id);
        if let Some(next) = outgoing.get(&id) {
            for target in next {
                let degree = indegree.get_mut(target).expect("edge target checked");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*target);
                }
            }
        }
    }
    if result.len() != indegree.len() {
        return Err(CompileError::Cycle);
    }
    Ok(result)
}

fn propagate_contracts(
    graph: &Graph,
    order: &[NodeId],
    descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
) -> Result<(Vec<String>, Vec<String>, u16), CompileError> {
    #[derive(Clone)]
    struct State {
        proofs: BTreeSet<String>,
        policy: BTreeSet<String>,
        fidelity: u16,
    }

    let initial = State {
        proofs: graph.required_proofs.iter().cloned().collect(),
        policy: graph.policy.iter().cloned().collect(),
        fidelity: u16::MAX,
    };
    let mut predecessors: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
    let mut has_successor = BTreeSet::new();
    for edge in &graph.edges {
        predecessors
            .entry(edge.to.node)
            .or_default()
            .push(edge.from.node);
        has_successor.insert(edge.from.node);
    }
    for values in predecessors.values_mut() {
        values.sort_unstable();
        values.dedup();
    }
    let mut states: BTreeMap<NodeId, State> = BTreeMap::new();
    for id in order {
        let descriptor = descriptors[id];
        let mut state = match predecessors.get(id).map(Vec::as_slice).unwrap_or_default() {
            [] => initial.clone(),
            [first, rest @ ..] => {
                let mut joined = states[first].clone();
                for predecessor in rest {
                    joined.proofs = joined
                        .proofs
                        .intersection(&states[predecessor].proofs)
                        .cloned()
                        .collect();
                    joined
                        .policy
                        .extend(states[predecessor].policy.iter().cloned());
                    joined.fidelity = joined.fidelity.min(states[predecessor].fidelity);
                }
                joined
            }
        };
        for required in &descriptor.proof.requires {
            if !state.proofs.contains(required) {
                return Err(CompileError::ProofMissing(*id, required.clone()));
            }
        }
        for required in &descriptor.policy.requires {
            if !state.policy.contains(required) {
                return Err(CompileError::PolicyMissing(*id, required.clone()));
            }
        }
        if state.fidelity < descriptor.fidelity.minimum_input {
            return Err(CompileError::FidelityInsufficient(*id));
        }
        state.fidelity = state
            .fidelity
            .saturating_sub(descriptor.fidelity.maximum_loss);
        for invalidated in &descriptor.proof.invalidates {
            state.proofs.remove(invalidated);
        }
        state
            .proofs
            .extend(descriptor.proof.provides.iter().cloned());
        state.policy.extend(descriptor.policy.adds.iter().cloned());
        states.insert(*id, state);
    }
    let mut terminals = order.iter().filter(|id| !has_successor.contains(id));
    let first = *terminals.next().expect("non-empty graph has a terminal");
    let mut result = states[&first].clone();
    for terminal in terminals {
        result.proofs = result
            .proofs
            .intersection(&states[terminal].proofs)
            .cloned()
            .collect();
        result
            .policy
            .extend(states[terminal].policy.iter().cloned());
        result.fidelity = result.fidelity.min(states[terminal].fidelity);
    }
    if result.fidelity < graph.minimum_fidelity {
        return Err(CompileError::FidelityInsufficient(
            *order.last().expect("non-empty graph"),
        ));
    }
    Ok((
        result.proofs.into_iter().collect(),
        result.policy.into_iter().collect(),
        result.fidelity,
    ))
}

struct PhysicalLowering<'a> {
    graph: &'a Graph,
    order: &'a [NodeId],
    descriptors: &'a BTreeMap<NodeId, &'a NodeDescriptor>,
    kernels: &'a BTreeMap<NodeId, &'a KernelDescriptor>,
    registry: &'a KernelRegistry,
    target: Target,
}

#[derive(Clone, Copy)]
struct PhysicalSearchLimits {
    region_candidates: usize,
    conversion_states: usize,
    max_peak_bytes: u64,
    max_steps: usize,
    max_buffers: usize,
}

fn build_semantic_region_candidates(
    context: &PhysicalLowering<'_>,
    fuse: bool,
    max_candidates: usize,
) -> Result<Vec<Vec<CompiledNode>>, CompileError> {
    let graph = context.graph;
    let order = context.order;
    let descriptors = context.descriptors;
    let kernels = context.kernels;
    let registry = context.registry;
    let target = context.target;
    let instances: BTreeMap<_, _> = graph.nodes.iter().map(|node| (node.id, node)).collect();
    let mut pending = alloc::vec![(0usize, Vec::<(usize, &KernelDescriptor)>::new())];
    let mut complete = Vec::new();
    while let Some((position, fused_choices)) = pending.pop() {
        if position == order.len() {
            let mut regions = Vec::with_capacity(order.len());
            let mut cursor = 0usize;
            let mut choice = 0usize;
            while cursor < order.len() {
                if fused_choices
                    .get(choice)
                    .is_some_and(|(start, _)| *start == cursor)
                {
                    let (_, kernel) = fused_choices[choice];
                    let end = cursor + kernel.implements.len();
                    regions.push(semantic_region(
                        &order[cursor..end],
                        kernel,
                        descriptors,
                        &instances,
                    ));
                    cursor = end;
                    choice += 1;
                } else {
                    let id = order[cursor];
                    regions.push(semantic_region(
                        &order[cursor..cursor + 1],
                        kernels[&id],
                        descriptors,
                        &instances,
                    ));
                    cursor += 1;
                }
            }
            complete.push(regions);
            if complete.len() > max_candidates {
                return Err(CompileError::SearchLimitExceeded);
            }
            continue;
        }

        let mut alternatives = Vec::new();
        if fuse {
            for kernel in registry.kernels.values().filter(|kernel| {
                kernel.target == target
                    && kernel.implements.len() >= 2
                    && position + kernel.implements.len() <= order.len()
            }) {
                let ids = &order[position..position + kernel.implements.len()];
                if kernel
                    .implements
                    .iter()
                    .zip(ids)
                    .all(|(implemented, node)| {
                        implemented.type_name == instances[node].descriptor
                            && implemented.version == instances[node].descriptor_version
                    })
                    && ids
                        .iter()
                        .all(|node| kernel.determinism <= descriptors[node].determinism)
                    && fused_layouts_compatible(
                        descriptors[&ids[0]],
                        descriptors[ids.last().expect("non-empty")],
                        kernel,
                    )
                    && linear_fusion_is_safe(graph, ids, descriptors)
                {
                    alternatives.push(kernel);
                }
            }
        }
        alternatives.sort_by_key(|kernel| {
            (
                core::cmp::Reverse(kernel.implements.len()),
                kernel.resources.peak_bytes,
                kernel.resources.scratch_bytes,
                kernel.implementation_id,
                kernel.id,
            )
        });
        for kernel in alternatives.into_iter().rev() {
            let mut next = fused_choices.clone();
            next.push((position, kernel));
            pending.push((position + kernel.implements.len(), next));
            if pending.len().saturating_add(complete.len()) > max_candidates {
                return Err(CompileError::SearchLimitExceeded);
            }
        }
        pending.push((position + 1, fused_choices));
        if pending.len().saturating_add(complete.len()) > max_candidates {
            return Err(CompileError::SearchLimitExceeded);
        }
    }
    Ok(complete)
}

fn semantic_region(
    ids: &[NodeId],
    kernel: &KernelDescriptor,
    descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
    instances: &BTreeMap<NodeId, &crate::model::NodeInstance>,
) -> CompiledNode {
    let first = descriptors[&ids[0]];
    let last = descriptors[ids.last().expect("semantic region is non-empty")];
    let fused = ids.len() > 1;
    CompiledNode {
        id: StepId(0),
        semantic_nodes: ids.to_vec(),
        semantic_types: ids
            .iter()
            .map(|id| NodeTypeRef {
                type_name: instances[id].descriptor.clone(),
                version: instances[id].descriptor_version,
            })
            .collect(),
        semantic_configs: ids.iter().map(|id| instances[id].config.clone()).collect(),
        kernel: kernel.id,
        implementation_id: kernel.implementation_id,
        resources: kernel.resources.clone(),
        determinism: kernel.determinism,
        lowering: kernel.lowering.clone(),
        conversion: None,
        input_ports: first.inputs.iter().map(|port| port.name.clone()).collect(),
        output_ports: last.outputs.iter().map(|port| port.name.clone()).collect(),
        input_contracts: first
            .inputs
            .iter()
            .map(|port| compiled_port_contract(port, select_layout(port, &kernel.input_layouts)))
            .collect(),
        output_contracts: last
            .outputs
            .iter()
            .map(|port| compiled_port_contract(port, select_layout(port, &kernel.output_layouts)))
            .collect(),
        input_bindings: alloc::vec![crate::model::InputBinding::Absent; first.inputs.len()],
        output_bindings: alloc::vec![OutputBinding::Terminal; last.outputs.len()],
        partiality: if fused {
            crate::model::Partiality::Atomic
        } else {
            first.partiality
        },
        failure: if fused {
            crate::model::FailureContract {
                domains: Vec::new(),
            }
        } else {
            first.failure.clone()
        },
        effect: if fused {
            crate::model::Effect::Pure
        } else {
            first.effect
        },
        retry_limit: if fused { 0 } else { first.retry_limit },
        state: if fused {
            StateContract::stateless()
        } else {
            first.state.clone()
        },
        subgraph_path: ids
            .iter()
            .filter_map(|id| descriptors[id].subgraph.as_ref().map(|item| item.subgraph))
            .collect(),
    }
}

fn linear_fusion_is_safe(
    graph: &Graph,
    ids: &[NodeId],
    descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
) -> bool {
    if graph
        .feedback
        .iter()
        .any(|edge| ids.contains(&edge.from.node) || ids.contains(&edge.to.node))
    {
        return false;
    }
    if ids.iter().any(|id| {
        let descriptor = descriptors[id];
        descriptor.effect != crate::model::Effect::Pure
            || descriptor.partiality != crate::model::Partiality::Atomic
            || descriptor.state.scope != StateScope::Stateless
            || descriptor.retry_limit != 0
            || descriptor.state.checkpointable()
            || !descriptor.failure.domains.is_empty()
            || descriptor.subgraph.is_some()
    }) {
        return false;
    }
    ids.windows(2).all(|pair| {
        let from = pair[0];
        let to = pair[1];
        let outgoing: Vec<_> = graph
            .edges
            .iter()
            .filter(|edge| edge.from.node == from)
            .collect();
        let incoming: Vec<_> = graph
            .edges
            .iter()
            .filter(|edge| edge.to.node == to)
            .collect();
        descriptors[&from].outputs.len() == 1
            && descriptors[&to].inputs.len() == 1
            && outgoing.len() == 1
            && incoming.len() == 1
            && outgoing[0] == incoming[0]
            && outgoing[0].from.port == descriptors[&from].outputs[0].name
            && incoming[0].to.port == descriptors[&to].inputs[0].name
    })
}

fn fused_layouts_compatible(
    first: &NodeDescriptor,
    last: &NodeDescriptor,
    kernel: &KernelDescriptor,
) -> bool {
    first.inputs.iter().all(|port| {
        port.layouts
            .iter()
            .any(|layout| kernel.input_layouts.contains(layout))
    }) && last.outputs.iter().all(|port| {
        port.layouts
            .iter()
            .any(|layout| kernel.output_layouts.contains(layout))
    })
}

#[derive(Clone)]
struct Route<'a> {
    consumer_region: usize,
    input_index: usize,
    path: Vec<&'a KernelDescriptor>,
}

struct PortGroup<'a> {
    producer_region: usize,
    output_index: usize,
    layout: Layout,
    capacity_bytes: u64,
    contract: PortDescriptor,
    routes: Vec<Route<'a>>,
}

fn lower_physical_plan(
    context: PhysicalLowering<'_>,
    fuse: bool,
    limits: PhysicalSearchLimits,
) -> Result<(Vec<CompiledNode>, Vec<BufferPlan>, u64), CompileError> {
    let region_candidates =
        build_semantic_region_candidates(&context, fuse, limits.region_candidates)?;
    let mut best: Option<LoweredCandidate> = None;
    let mut first_error = None;
    let mut saw_resource_limit = false;
    let mut saw_compile_limit = false;
    for semantic_regions in region_candidates {
        match lower_semantic_regions(&context, semantic_regions, limits.conversion_states) {
            Ok((nodes, buffers, _))
                if nodes.len() > limits.max_steps || buffers.len() > limits.max_buffers =>
            {
                saw_compile_limit = true;
            }
            Ok((_, _, peak)) if peak > limits.max_peak_bytes => {
                saw_resource_limit = true;
            }
            Ok((nodes, buffers, peak)) => {
                let kernels = nodes.iter().map(|node| node.kernel).collect::<Vec<_>>();
                let score = (peak, nodes.len(), kernels);
                if best
                    .as_ref()
                    .is_none_or(|current| score < (current.0, current.1, current.2.clone()))
                {
                    best = Some((score.0, score.1, score.2, nodes, buffers));
                }
            }
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    best.map(|(peak, _, _, nodes, buffers)| (nodes, buffers, peak))
        .ok_or_else(|| {
            if saw_resource_limit {
                CompileError::ResourceOverflow
            } else if saw_compile_limit {
                CompileError::CompileLimitExceeded
            } else {
                first_error.unwrap_or(CompileError::ResourceOverflow)
            }
        })
}

fn lower_semantic_regions(
    context: &PhysicalLowering<'_>,
    semantic_regions: Vec<CompiledNode>,
    max_conversion_states: usize,
) -> Result<(Vec<CompiledNode>, Vec<BufferPlan>, u64), CompileError> {
    let graph = context.graph;
    let descriptors = context.descriptors;
    let registry = context.registry;
    let target = context.target;
    let owner: BTreeMap<_, _> = semantic_regions
        .iter()
        .enumerate()
        .flat_map(|(index, node)| node.semantic_nodes.iter().map(move |id| (*id, index)))
        .collect();
    let mut grouped: BTreeMap<crate::model::PortRef, Vec<&Edge>> = BTreeMap::new();
    for edge in &graph.edges {
        if owner[&edge.from.node] != owner[&edge.to.node] {
            grouped.entry(edge.from.clone()).or_default().push(edge);
        }
    }
    let mut groups = Vec::new();
    let mut conversion_states = 0usize;
    for (source, mut edges) in grouped {
        edges.sort_by_key(|edge| edge.to.clone());
        let producer_descriptor = descriptors[&source.node];
        let output_index = producer_descriptor
            .outputs
            .iter()
            .position(|port| port.name == source.port)
            .ok_or_else(|| CompileError::UnknownPort(source.node, source.port.clone()))?;
        let output = &producer_descriptor.outputs[output_index];
        if output.max_bytes == 0 {
            return Err(CompileError::InvalidPortSize(source.node, source.port));
        }
        let producer_region = owner[&source.node];
        let producer_kernel = registry
            .kernels
            .get(&semantic_regions[producer_region].kernel)
            .expect("compiled kernel is registered");
        let mut source_layouts: Vec<_> = output
            .layouts
            .iter()
            .filter(|layout| producer_kernel.output_layouts.contains(layout))
            .copied()
            .collect();
        source_layouts.sort_unstable();
        source_layouts.dedup();

        let mut best: Option<(usize, u64, Layout, Vec<Route<'_>>)> = None;
        for source_layout in source_layouts {
            let mut routes = Vec::new();
            let mut conversion_count = 0usize;
            let mut conversion_workspace = 0u64;
            let mut valid = true;
            for edge in &edges {
                let consumer_descriptor = descriptors[&edge.to.node];
                let input_index = consumer_descriptor
                    .inputs
                    .iter()
                    .position(|port| port.name == edge.to.port)
                    .ok_or_else(|| CompileError::UnknownPort(edge.to.node, edge.to.port.clone()))?;
                let input = &consumer_descriptor.inputs[input_index];
                let consumer_region = owner[&edge.to.node];
                let consumer_kernel = registry
                    .kernels
                    .get(&semantic_regions[consumer_region].kernel)
                    .expect("compiled kernel is registered");
                let mut targets: Vec<_> = input
                    .layouts
                    .iter()
                    .filter(|layout| consumer_kernel.input_layouts.contains(layout))
                    .copied()
                    .collect();
                targets.sort_unstable();
                targets.dedup();
                let path =
                    if targets.contains(&source_layout) && output.max_bytes <= input.max_bytes {
                        Some(Vec::new())
                    } else {
                        conversion_path(
                            registry,
                            ConversionRequest {
                                target,
                                semantic_type: &output.semantic_type,
                                max_bytes: output.max_bytes,
                                from: source_layout,
                                targets: &targets,
                                target_max_bytes: input.max_bytes,
                            },
                            &mut ConversionBudget {
                                searched: &mut conversion_states,
                                max_states: max_conversion_states,
                            },
                        )?
                    };
                let Some(path) = path else {
                    valid = false;
                    break;
                };
                conversion_count = conversion_count
                    .checked_add(path.len())
                    .ok_or(CompileError::SearchLimitExceeded)?;
                for kernel in &path {
                    conversion_workspace = conversion_workspace.max(
                        kernel
                            .resources
                            .peak_bytes
                            .checked_add(kernel.resources.scratch_bytes)
                            .ok_or(CompileError::ResourceOverflow)?,
                    );
                }
                routes.push(Route {
                    consumer_region,
                    input_index,
                    path,
                });
            }
            if valid {
                let score = (conversion_count, conversion_workspace, source_layout);
                if best
                    .as_ref()
                    .is_none_or(|current| score < (current.0, current.1, current.2))
                {
                    best = Some((score.0, score.1, score.2, routes));
                }
            }
        }
        let Some((_, _, layout, routes)) = best else {
            return Err(CompileError::LayoutUnavailable(source.node, source.port));
        };
        groups.push(PortGroup {
            producer_region,
            output_index,
            layout,
            capacity_bytes: output.max_bytes,
            contract: output.clone(),
            routes,
        });
    }

    let mut nodes = Vec::new();
    let mut semantic_steps = alloc::vec![StepId(0); semantic_regions.len()];
    let mut conversion_steps: BTreeMap<(usize, usize), Vec<StepId>> = BTreeMap::new();
    for region_index in 0..semantic_regions.len() {
        for (group_index, group) in groups.iter().enumerate() {
            for (route_index, route) in group.routes.iter().enumerate() {
                if route.consumer_region != region_index {
                    continue;
                }
                let mut steps = Vec::new();
                for (path_index, kernel) in route.path.iter().enumerate() {
                    let conversion = kernel
                        .conversion
                        .clone()
                        .expect("conversion path contains conversion kernels");
                    let id = StepId(nodes.len() as u32);
                    steps.push(id);
                    nodes.push(CompiledNode {
                        id,
                        semantic_nodes: Vec::new(),
                        semantic_types: Vec::new(),
                        semantic_configs: Vec::new(),
                        kernel: kernel.id,
                        implementation_id: kernel.implementation_id,
                        resources: kernel.resources.clone(),
                        determinism: kernel.determinism,
                        lowering: kernel.lowering.clone(),
                        conversion: Some(conversion.clone()),
                        input_ports: alloc::vec!["input".to_string()],
                        output_ports: alloc::vec!["output".to_string()],
                        input_contracts: alloc::vec![conversion_port_contract(
                            &group.contract,
                            conversion.from,
                            "input",
                            if path_index == 0 {
                                group.capacity_bytes
                            } else {
                                route.path[path_index - 1]
                                    .conversion
                                    .as_ref()
                                    .expect("conversion path")
                                    .max_output_bytes
                            },
                        )],
                        output_contracts: alloc::vec![conversion_port_contract(
                            &group.contract,
                            conversion.to,
                            "output",
                            conversion.max_output_bytes,
                        )],
                        input_bindings: alloc::vec![crate::model::InputBinding::Absent],
                        output_bindings: alloc::vec![OutputBinding::Terminal],
                        partiality: crate::model::Partiality::Atomic,
                        failure: crate::model::FailureContract {
                            domains: Vec::new(),
                        },
                        effect: crate::model::Effect::Pure,
                        retry_limit: 0,
                        state: StateContract::stateless(),
                        subgraph_path: Vec::new(),
                    });
                }
                conversion_steps.insert((group_index, route_index), steps);
            }
        }
        let mut semantic = semantic_regions[region_index].clone();
        semantic.id = StepId(nodes.len() as u32);
        semantic_steps[region_index] = semantic.id;
        nodes.push(semantic);
    }

    let mut invocation_ports = graph.invocation_inputs.clone();
    invocation_ports.sort_unstable();
    invocation_ports.dedup();
    for (invocation_index, invocation) in invocation_ports.iter().enumerate() {
        let region = owner[&invocation.node];
        let semantic = &semantic_regions[region];
        if semantic.semantic_nodes.first() != Some(&invocation.node) {
            return Err(CompileError::DuplicateInput(
                invocation.node,
                invocation.port.clone(),
            ));
        }
        let input_index = descriptors[&invocation.node]
            .inputs
            .iter()
            .position(|port| port.name == invocation.port)
            .ok_or_else(|| CompileError::UnknownPort(invocation.node, invocation.port.clone()))?;
        set_invocation(
            &mut nodes,
            semantic_steps[region],
            input_index,
            invocation_index as u32,
        )?;
    }

    let mut buffers = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        let producer = semantic_steps[group.producer_region];
        let mut consumers = Vec::new();
        for (route_index, route) in group.routes.iter().enumerate() {
            let path = &conversion_steps[&(group_index, route_index)];
            consumers.push(
                path.first()
                    .copied()
                    .unwrap_or(semantic_steps[route.consumer_region]),
            );
        }
        consumers.sort_unstable();
        consumers.dedup();
        let source_buffer = push_buffer(
            &mut buffers,
            group.layout,
            group.capacity_bytes,
            producer,
            consumers,
        );
        set_output(&mut nodes, producer, group.output_index, source_buffer)?;
        for (route_index, route) in group.routes.iter().enumerate() {
            let path = &conversion_steps[&(group_index, route_index)];
            if path.is_empty() {
                set_input(
                    &mut nodes,
                    semantic_steps[route.consumer_region],
                    route.input_index,
                    source_buffer,
                )?;
                continue;
            }
            set_input(&mut nodes, path[0], 0, source_buffer)?;
            for (index, step) in path.iter().copied().enumerate() {
                let next = path
                    .get(index + 1)
                    .copied()
                    .unwrap_or(semantic_steps[route.consumer_region]);
                let conversion = nodes[step.0 as usize]
                    .conversion
                    .as_ref()
                    .expect("conversion step");
                let output = push_buffer(
                    &mut buffers,
                    conversion.to,
                    conversion.max_output_bytes,
                    step,
                    alloc::vec![next],
                );
                set_output(&mut nodes, step, 0, output)?;
                if index + 1 < path.len() {
                    set_input(&mut nodes, next, 0, output)?;
                } else {
                    set_input(&mut nodes, next, route.input_index, output)?;
                }
            }
        }
    }
    canonicalize_buffers(&mut nodes, &mut buffers);
    synchronize_port_layouts(&mut nodes, &buffers);
    assign_aliases(&mut buffers);
    let arena_bytes = buffers
        .iter()
        .filter(|buffer| buffer.aliases.is_none())
        .try_fold(0u64, |sum, buffer| {
            sum.checked_add(buffer.capacity_bytes)
                .ok_or(CompileError::ResourceOverflow)
        })?;
    let workspace = nodes.iter().try_fold(0u64, |peak, node| {
        let mut bytes = node
            .resources
            .peak_bytes
            .checked_add(node.resources.scratch_bytes)
            .ok_or(CompileError::ResourceOverflow)?;
        if node.state.scope == StateScope::Invocation {
            bytes = bytes
                .checked_add(node.state.max_bytes)
                .ok_or(CompileError::ResourceOverflow)?;
        }
        Ok::<_, CompileError>(peak.max(bytes))
    })?;
    let peak = arena_bytes
        .checked_add(workspace)
        .ok_or(CompileError::ResourceOverflow)?;
    Ok((nodes, buffers, peak))
}

fn lower_feedback(
    graph: &Graph,
    nodes: &mut [CompiledNode],
    max_feedback_edges: usize,
) -> Result<(Vec<crate::FeedbackPlan>, u64), CompileError> {
    if graph.feedback.len() > max_feedback_edges {
        return Err(CompileError::CompileLimitExceeded);
    }
    let mut feedback_edges = graph.feedback.clone();
    feedback_edges.sort_by_key(|edge| (edge.from.clone(), edge.to.clone()));
    if feedback_edges
        .windows(2)
        .any(|pair| pair[0].from == pair[1].from && pair[0].to == pair[1].to)
    {
        return Err(CompileError::InvalidFeedback(
            feedback_edges[0].to.node,
            feedback_edges[0].to.port.clone(),
        ));
    }
    let mut plans = Vec::with_capacity(feedback_edges.len());
    let mut state_bytes = 0u64;
    for edge in feedback_edges {
        let from_step_index = nodes
            .iter()
            .position(|node| node.semantic_nodes.contains(&edge.from.node))
            .ok_or(CompileError::UnknownNode(edge.from.node))?;
        let from_port = nodes[from_step_index]
            .output_ports
            .iter()
            .position(|port| port == &edge.from.port)
            .ok_or_else(|| CompileError::UnknownPort(edge.from.node, edge.from.port.clone()))?;
        let to_step_index = nodes
            .iter()
            .position(|node| node.semantic_nodes.contains(&edge.to.node))
            .ok_or(CompileError::UnknownNode(edge.to.node))?;
        let to_port = nodes[to_step_index]
            .input_ports
            .iter()
            .position(|port| port == &edge.to.port)
            .ok_or_else(|| CompileError::UnknownPort(edge.to.node, edge.to.port.clone()))?;
        let output_contract = &nodes[from_step_index].output_contracts[from_port];
        let input_contract = &nodes[to_step_index].input_contracts[to_port];
        if !compiled_port_contract_satisfies(output_contract, input_contract) {
            return Err(CompileError::InvalidFeedback(
                edge.to.node,
                edge.to.port.clone(),
            ));
        }
        let value_bytes = output_contract.max_bytes;
        let bytes = value_bytes
            .checked_mul(u64::from(edge.delay.invocations))
            .ok_or(CompileError::ResourceOverflow)?;
        let id = feedback_id(plans.len())?;
        nodes[to_step_index].input_bindings[to_port] = crate::InputBinding::Feedback(id);
        plans.push(crate::FeedbackPlan {
            id,
            from_step: nodes[from_step_index].id,
            from_port: from_port as u32,
            to_step: nodes[to_step_index].id,
            to_port: to_port as u32,
            delay: edge.delay,
            state_bytes: bytes,
        });
        state_bytes = state_bytes
            .checked_add(bytes)
            .ok_or(CompileError::ResourceOverflow)?;
    }
    Ok((plans, state_bytes))
}

fn feedback_id(index: usize) -> Result<crate::FeedbackId, CompileError> {
    u32::try_from(index)
        .map(crate::FeedbackId)
        .map_err(|_| CompileError::CompileLimitExceeded)
}

struct ConversionRequest<'a> {
    target: Target,
    semantic_type: &'a str,
    max_bytes: u64,
    from: Layout,
    targets: &'a [Layout],
    target_max_bytes: u64,
}

struct ConversionBudget<'a> {
    searched: &'a mut usize,
    max_states: usize,
}

fn conversion_path<'a>(
    registry: &'a KernelRegistry,
    request: ConversionRequest<'_>,
    budget: &mut ConversionBudget<'_>,
) -> Result<Option<Vec<&'a KernelDescriptor>>, CompileError> {
    let mut kernels: Vec<_> = registry
        .kernels
        .values()
        .filter(|kernel| {
            kernel.target == request.target
                && kernel.implements.is_empty()
                && kernel.determinism == crate::model::Determinism::BitExact
                && kernel.conversion.as_ref().is_some_and(|conversion| {
                    conversion.semantic_type == request.semantic_type
                        && kernel.input_layouts.contains(&conversion.from)
                        && kernel.output_layouts.contains(&conversion.to)
                })
        })
        .collect();
    kernels.sort_by_key(|kernel| (kernel.implementation_id, kernel.id));
    let mut frontier = alloc::vec![(
        request.from,
        request.max_bytes,
        Vec::new(),
        alloc::vec![request.from],
    )];
    let mut solutions = Vec::new();
    while let Some((layout, capacity_bytes, path, visited)) = frontier.pop() {
        *budget.searched = budget
            .searched
            .checked_add(1)
            .ok_or(CompileError::SearchLimitExceeded)?;
        if *budget.searched > budget.max_states {
            return Err(CompileError::SearchLimitExceeded);
        }
        if request.targets.contains(&layout)
            && capacity_bytes <= request.target_max_bytes
            && !path.is_empty()
        {
            solutions.push(path);
            continue;
        }
        if visited.len() >= 5 {
            continue;
        }
        for kernel in kernels.iter().rev() {
            let conversion = kernel.conversion.as_ref().expect("filtered");
            if conversion.from == layout
                && conversion.max_input_bytes >= capacity_bytes
                && !visited.contains(&conversion.to)
            {
                let mut next_path = path.clone();
                next_path.push(*kernel);
                let mut next_visited = visited.clone();
                next_visited.push(conversion.to);
                frontier.push((
                    conversion.to,
                    conversion.max_output_bytes,
                    next_path,
                    next_visited,
                ));
            }
        }
    }
    Ok(solutions.into_iter().min_by_key(|path| {
        let workspace = path
            .iter()
            .map(|kernel| {
                kernel
                    .resources
                    .peak_bytes
                    .saturating_add(kernel.resources.scratch_bytes)
            })
            .max()
            .unwrap_or_default();
        let implementations: Vec<_> = path.iter().map(|kernel| kernel.implementation_id).collect();
        let ids: Vec<_> = path.iter().map(|kernel| kernel.id).collect();
        (path.len(), workspace, implementations, ids)
    }))
}

fn push_buffer(
    buffers: &mut Vec<BufferPlan>,
    layout: Layout,
    capacity_bytes: u64,
    producer: StepId,
    mut consumers: Vec<StepId>,
) -> BufferId {
    consumers.sort_unstable();
    consumers.dedup();
    let last_consumer = *consumers.last().expect("physical buffer has a consumer");
    let id = BufferId(buffers.len() as u32);
    buffers.push(BufferPlan {
        id,
        layout,
        capacity_bytes,
        producer,
        consumers,
        last_consumer,
        aliases: None,
    });
    id
}

fn set_input(
    nodes: &mut [CompiledNode],
    step: StepId,
    index: usize,
    buffer: BufferId,
) -> Result<(), CompileError> {
    let binding = nodes
        .get_mut(step.0 as usize)
        .and_then(|node| node.input_bindings.get_mut(index))
        .ok_or(CompileError::ResourceOverflow)?;
    *binding = crate::model::InputBinding::Buffer(buffer);
    Ok(())
}

fn set_invocation(
    nodes: &mut [CompiledNode],
    step: StepId,
    index: usize,
    invocation: u32,
) -> Result<(), CompileError> {
    let binding = nodes
        .get_mut(step.0 as usize)
        .and_then(|node| node.input_bindings.get_mut(index))
        .ok_or(CompileError::ResourceOverflow)?;
    if !matches!(binding, crate::model::InputBinding::Absent) {
        return Err(CompileError::ResourceOverflow);
    }
    *binding = crate::model::InputBinding::Invocation(invocation);
    Ok(())
}

fn set_output(
    nodes: &mut [CompiledNode],
    step: StepId,
    index: usize,
    buffer: BufferId,
) -> Result<(), CompileError> {
    let binding = nodes
        .get_mut(step.0 as usize)
        .and_then(|node| node.output_bindings.get_mut(index))
        .ok_or(CompileError::ResourceOverflow)?;
    *binding = OutputBinding::Buffer(buffer);
    Ok(())
}

fn canonicalize_buffers(nodes: &mut [CompiledNode], buffers: &mut [BufferPlan]) {
    let mut order: Vec<_> = (0..buffers.len()).collect();
    order.sort_by_key(|index| {
        let buffer = &buffers[*index];
        (buffer.producer, buffer.last_consumer, buffer.id)
    });
    let mut remap = BTreeMap::new();
    for (new, old) in order.iter().copied().enumerate() {
        remap.insert(buffers[old].id, BufferId(new as u32));
    }
    let old = buffers.to_vec();
    for (new, old_index) in order.into_iter().enumerate() {
        buffers[new] = BufferPlan {
            id: BufferId(new as u32),
            aliases: None,
            ..old[old_index].clone()
        };
    }
    for node in nodes {
        for binding in &mut node.input_bindings {
            if let crate::model::InputBinding::Buffer(buffer) = binding {
                *buffer = remap[buffer];
            }
        }
        for binding in &mut node.output_bindings {
            if let OutputBinding::Buffer(buffer) = binding {
                *buffer = remap[buffer];
            }
        }
    }
}

fn assign_aliases(buffers: &mut [BufferPlan]) {
    struct Slot {
        root: BufferId,
        layout: Layout,
        capacity_bytes: u64,
        available_after: StepId,
    }
    let mut slots: Vec<Slot> = Vec::new();
    for buffer in buffers {
        let candidate = slots
            .iter_mut()
            .filter(|slot| slot.available_after < buffer.producer)
            .filter(|slot| {
                slot.layout == buffer.layout && slot.capacity_bytes >= buffer.capacity_bytes
            })
            .min_by_key(|slot| (slot.capacity_bytes, slot.root));
        if let Some(slot) = candidate {
            buffer.aliases = Some(slot.root);
            slot.available_after = buffer.last_consumer;
        } else {
            slots.push(Slot {
                root: buffer.id,
                layout: buffer.layout,
                capacity_bytes: buffer.capacity_bytes,
                available_after: buffer.last_consumer,
            });
        }
    }
}

fn hash_graph(graph: &Graph, descriptors: &BTreeMap<NodeId, &NodeDescriptor>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("blut.graph.v3");
    put_u32(&mut hasher, graph.version);
    let mut nodes = graph.nodes.clone();
    nodes.sort_by_key(|node| node.id);
    put_u32(&mut hasher, nodes.len() as u32);
    for node in nodes {
        put_u32(&mut hasher, node.id.0);
        put_str(&mut hasher, &node.descriptor);
        put_u32(&mut hasher, node.descriptor_version);
        hash_descriptor(&mut hasher, descriptors[&node.id]);
        put_u32(&mut hasher, node.config.len() as u32);
        for (key, value) in node.config {
            put_str(&mut hasher, &key);
            hash_config_value(&mut hasher, &value);
        }
    }
    let mut edges = graph.edges.clone();
    edges.sort_by_key(|edge| (edge.from.clone(), edge.to.clone()));
    put_u32(&mut hasher, edges.len() as u32);
    for edge in edges {
        put_u32(&mut hasher, edge.from.node.0);
        put_str(&mut hasher, &edge.from.port);
        put_u32(&mut hasher, edge.to.node.0);
        put_str(&mut hasher, &edge.to.port);
    }
    let mut feedback = graph.feedback.clone();
    feedback.sort_by_key(|edge| (edge.from.clone(), edge.to.clone()));
    put_u32(&mut hasher, feedback.len() as u32);
    for edge in feedback {
        put_port_ref(&mut hasher, &edge.from);
        put_port_ref(&mut hasher, &edge.to);
        put_u32(&mut hasher, edge.delay.invocations);
        hash_delay_initial(&mut hasher, &edge.delay.initial);
    }
    let mut invocation_ports = graph.invocation_inputs.clone();
    invocation_ports.sort_unstable();
    invocation_ports.dedup();
    put_u32(&mut hasher, invocation_ports.len() as u32);
    for port in invocation_ports {
        put_u32(&mut hasher, port.node.0);
        put_str(&mut hasher, &port.port);
    }
    let mut capabilities: Vec<_> = graph
        .required_capabilities
        .iter()
        .map(|capability| capability.0.as_str())
        .collect();
    capabilities.sort_unstable();
    capabilities.dedup();
    put_str_set(&mut hasher, &capabilities);
    let mut proofs: Vec<_> = graph.required_proofs.iter().map(String::as_str).collect();
    proofs.sort_unstable();
    proofs.dedup();
    put_str_set(&mut hasher, &proofs);
    let mut policy: Vec<_> = graph.policy.iter().map(String::as_str).collect();
    policy.sort_unstable();
    policy.dedup();
    put_str_set(&mut hasher, &policy);
    put_u32(&mut hasher, u32::from(graph.minimum_fidelity));
    hash_session(&mut hasher, graph.session.as_ref());
    *hasher.finalize().as_bytes()
}

fn hash_descriptor(hasher: &mut blake3::Hasher, descriptor: &NodeDescriptor) {
    fn hash_ports(hasher: &mut blake3::Hasher, ports: &[PortDescriptor]) {
        let mut ports = ports.to_vec();
        ports.sort_by(|a, b| a.name.cmp(&b.name));
        put_u32(hasher, ports.len() as u32);
        for port in ports {
            put_str(hasher, &port.name);
            put_str(hasher, &port.semantic_type);
            put_u32(hasher, u32::from(port.optional));
            let mut layouts = port.layouts;
            layouts.sort_unstable();
            layouts.dedup();
            put_u32(hasher, layouts.len() as u32);
            for layout in layouts {
                put_u32(hasher, layout as u32);
            }
            hasher.update(&port.max_bytes.to_le_bytes());
            hash_abir_type(hasher, &port.abir);
            hash_proof(hasher, &port.proof);
            hash_policy(hasher, &port.policy);
            hash_fidelity(hasher, &port.fidelity);
            hash_extent(hasher, &port.extent);
            hash_lease(hasher, &port.lease);
        }
    }
    hash_ports(hasher, &descriptor.inputs);
    hash_ports(hasher, &descriptor.outputs);
    let mut capabilities: Vec<_> = descriptor
        .capabilities
        .iter()
        .map(|capability| capability.0.as_str())
        .collect();
    capabilities.sort_unstable();
    capabilities.dedup();
    put_str_set(hasher, &capabilities);
    let mut targets = descriptor.targets.clone();
    targets.sort_unstable();
    targets.dedup();
    put_u32(hasher, targets.len() as u32);
    for target in targets {
        put_u32(hasher, target as u32);
    }
    hasher.update(&descriptor.resources.peak_bytes.to_le_bytes());
    hasher.update(&descriptor.resources.scratch_bytes.to_le_bytes());
    put_u32(hasher, u32::from(descriptor.resources.threads));
    match &descriptor.resources.device {
        Some(device) => {
            hasher.update(&[1]);
            put_str(hasher, device);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    put_u32(hasher, descriptor.determinism as u32);
    hash_config_schema(hasher, &descriptor.config);
    hash_state(hasher, &descriptor.state);
    match &descriptor.subgraph {
        Some(lowering) => {
            hasher.update(&[1]);
            hasher.update(&lowering.subgraph.0);
            hash_port_maps(hasher, &lowering.input_map);
            hash_port_maps(hasher, &lowering.output_map);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    let mut proof_requires: Vec<_> = descriptor
        .proof
        .requires
        .iter()
        .map(String::as_str)
        .collect();
    proof_requires.sort_unstable();
    proof_requires.dedup();
    put_str_set(hasher, &proof_requires);
    let mut proof_provides: Vec<_> = descriptor
        .proof
        .provides
        .iter()
        .map(String::as_str)
        .collect();
    proof_provides.sort_unstable();
    proof_provides.dedup();
    put_str_set(hasher, &proof_provides);
    let mut proof_invalidates: Vec<_> = descriptor
        .proof
        .invalidates
        .iter()
        .map(String::as_str)
        .collect();
    proof_invalidates.sort_unstable();
    proof_invalidates.dedup();
    put_str_set(hasher, &proof_invalidates);
    let mut policy_requires: Vec<_> = descriptor
        .policy
        .requires
        .iter()
        .map(String::as_str)
        .collect();
    policy_requires.sort_unstable();
    policy_requires.dedup();
    put_str_set(hasher, &policy_requires);
    let mut policy_adds: Vec<_> = descriptor.policy.adds.iter().map(String::as_str).collect();
    policy_adds.sort_unstable();
    policy_adds.dedup();
    put_str_set(hasher, &policy_adds);
    put_u32(hasher, u32::from(descriptor.fidelity.minimum_input));
    put_u32(hasher, u32::from(descriptor.fidelity.maximum_loss));
    put_u32(hasher, descriptor.partiality as u32);
    let mut failure_domains: Vec<_> = descriptor
        .failure
        .domains
        .iter()
        .map(String::as_str)
        .collect();
    failure_domains.sort_unstable();
    failure_domains.dedup();
    put_str_set(hasher, &failure_domains);
    put_u32(hasher, descriptor.effect as u32);
    put_u32(hasher, u32::from(descriptor.retry_limit));
}

fn put_str_set(hasher: &mut blake3::Hasher, values: &[&str]) {
    put_u32(hasher, values.len() as u32);
    for value in values {
        put_str(hasher, value);
    }
}

fn hash_config_value(hasher: &mut blake3::Hasher, value: &crate::ConfigValue) {
    match value {
        crate::ConfigValue::Bool(value) => {
            hasher.update(&[0, u8::from(*value)]);
        }
        crate::ConfigValue::I64(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_le_bytes());
        }
        crate::ConfigValue::U64(value) => {
            hasher.update(&[2]);
            hasher.update(&value.to_le_bytes());
        }
        crate::ConfigValue::Text(value) => {
            hasher.update(&[3]);
            put_str(hasher, value);
        }
        crate::ConfigValue::Bytes(value) => {
            hasher.update(&[4]);
            put_u32(hasher, value.len() as u32);
            hasher.update(value);
        }
    }
}

fn hash_config_schema(hasher: &mut blake3::Hasher, schema: &crate::ConfigSchema) {
    put_u32(hasher, schema.fields.len() as u32);
    for field in &schema.fields {
        put_str(hasher, &field.name);
        match &field.value_type {
            crate::ConfigType::Bool => {
                hasher.update(&[0]);
            }
            crate::ConfigType::I64 { minimum, maximum } => {
                hasher.update(&[1]);
                hasher.update(&minimum.to_le_bytes());
                hasher.update(&maximum.to_le_bytes());
            }
            crate::ConfigType::U64 { minimum, maximum } => {
                hasher.update(&[2]);
                hasher.update(&minimum.to_le_bytes());
                hasher.update(&maximum.to_le_bytes());
            }
            crate::ConfigType::Text { max_bytes } => {
                hasher.update(&[3]);
                put_u32(hasher, *max_bytes);
            }
            crate::ConfigType::Choice { values } => {
                hasher.update(&[4]);
                put_u32(hasher, values.len() as u32);
                for value in values {
                    put_str(hasher, value);
                }
            }
            crate::ConfigType::Bytes { max_bytes } => {
                hasher.update(&[5]);
                put_u32(hasher, *max_bytes);
            }
        };
        hasher.update(&[u8::from(field.required)]);
        match &field.default {
            Some(value) => {
                hasher.update(&[1]);
                hash_config_value(hasher, value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
}

fn hash_abir_type(hasher: &mut blake3::Hasher, abir: &crate::AbirSemanticType) {
    fn hash_root(hasher: &mut blake3::Hasher, root: &crate::AbirRootType) {
        match root {
            crate::AbirRootType::Dataset => {
                hasher.update(&[0]);
            }
            crate::AbirRootType::Recording => {
                hasher.update(&[1]);
            }
            crate::AbirRootType::Stream => {
                hasher.update(&[2]);
            }
            crate::AbirRootType::SignalBlock => {
                hasher.update(&[3]);
            }
            crate::AbirRootType::TemporalTable => {
                hasher.update(&[4]);
            }
            crate::AbirRootType::Table => {
                hasher.update(&[5]);
            }
            crate::AbirRootType::Tensor => {
                hasher.update(&[6]);
            }
            crate::AbirRootType::EncodedBlock => {
                hasher.update(&[7]);
            }
            crate::AbirRootType::BlobRef => {
                hasher.update(&[8]);
            }
            crate::AbirRootType::Unknown(value) => {
                hasher.update(&[9]);
                put_str(hasher, value);
            }
        };
    }
    fn hash_view(hasher: &mut blake3::Hasher, view: &crate::AbirViewType) {
        match view {
            crate::AbirViewType::Root => {
                hasher.update(&[0]);
            }
            crate::AbirViewType::Recording => {
                hasher.update(&[1]);
            }
            crate::AbirViewType::Stream => {
                hasher.update(&[2]);
            }
            crate::AbirViewType::Block => {
                hasher.update(&[3]);
            }
            crate::AbirViewType::Tensor => {
                hasher.update(&[4]);
            }
            crate::AbirViewType::Atom => {
                hasher.update(&[5]);
            }
            crate::AbirViewType::Unknown(value) => {
                hasher.update(&[6]);
                put_str(hasher, value);
            }
        };
    }
    hash_root(hasher, &abir.root);
    hash_view(hasher, &abir.view);
}

fn hash_proof(hasher: &mut blake3::Hasher, proof: &crate::ProofContract) {
    let requires = proof
        .requires
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let provides = proof
        .provides
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let invalidates = proof
        .invalidates
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    put_str_set(hasher, &requires);
    put_str_set(hasher, &provides);
    put_str_set(hasher, &invalidates);
}

fn hash_policy(hasher: &mut blake3::Hasher, policy: &crate::PolicyContract) {
    let requires = policy
        .requires
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let adds = policy.adds.iter().map(String::as_str).collect::<Vec<_>>();
    put_str_set(hasher, &requires);
    put_str_set(hasher, &adds);
}

fn hash_fidelity(hasher: &mut blake3::Hasher, fidelity: &crate::FidelityContract) {
    put_u32(hasher, u32::from(fidelity.minimum_input));
    put_u32(hasher, u32::from(fidelity.maximum_loss));
}

fn hash_extent(hasher: &mut blake3::Hasher, extent: &crate::ExtentContract) {
    hasher.update(&[extent.rank]);
    put_u32(hasher, extent.maximum_shape.len() as u32);
    for size in &extent.maximum_shape {
        hasher.update(&size.to_le_bytes());
    }
    hasher.update(&extent.max_elements.to_le_bytes());
    hasher.update(&[u8::from(extent.ragged), u8::from(extent.sparse)]);
}

fn hash_lease(hasher: &mut blake3::Hasher, lease: &crate::LeaseContract) {
    put_u32(hasher, lease.access as u32);
    put_u32(hasher, lease.lifetime as u32);
    hasher.update(&[
        u8::from(lease.zero_copy_permitted),
        u8::from(lease.contiguous_required),
    ]);
}

fn hash_state(hasher: &mut blake3::Hasher, state: &StateContract) {
    put_u32(hasher, state.scope as u32);
    hasher.update(&state.max_bytes.to_le_bytes());
    put_u32(hasher, state.checkpoint.mode as u32);
    hasher.update(&state.checkpoint.max_snapshot_bytes.to_le_bytes());
    put_u32(hasher, state.checkpoint.max_interval_invocations);
}

fn hash_port_maps(hasher: &mut blake3::Hasher, maps: &[crate::PortMap]) {
    put_u32(hasher, maps.len() as u32);
    for map in maps {
        put_str(hasher, &map.outer);
        put_str(hasher, &map.inner);
    }
}

fn hash_delay_initial(hasher: &mut blake3::Hasher, initial: &crate::DelayInitial) {
    match initial {
        crate::DelayInitial::Absent => {
            hasher.update(&[0]);
        }
        crate::DelayInitial::Zeroed => {
            hasher.update(&[1]);
        }
        crate::DelayInitial::ContentId(id) => {
            hasher.update(&[2]);
            hasher.update(id);
        }
    };
}

fn hash_session(hasher: &mut blake3::Hasher, session: Option<&crate::SessionContract>) {
    match session {
        Some(session) => {
            hasher.update(&[1]);
            put_str(hasher, &session.namespace);
            put_u32(hasher, session.max_concurrent_sessions);
            hasher.update(&session.max_idle_millis.to_le_bytes());
            hasher.update(&[u8::from(session.reset_on_plan_change)]);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_compiled_ports(hasher: &mut blake3::Hasher, ports: &[CompiledPortContract]) {
    put_u32(hasher, ports.len() as u32);
    for port in ports {
        put_str(hasher, &port.name);
        put_str(hasher, &port.semantic_type);
        hasher.update(&[u8::from(port.optional)]);
        put_u32(hasher, port.layout as u32);
        hasher.update(&port.max_bytes.to_le_bytes());
        hash_abir_type(hasher, &port.abir);
        hash_proof(hasher, &port.proof);
        hash_policy(hasher, &port.policy);
        hash_fidelity(hasher, &port.fidelity);
        hash_extent(hasher, &port.extent);
        hash_lease(hasher, &port.lease);
    }
}

fn put_port_ref(hasher: &mut blake3::Hasher, port: &crate::PortRef) {
    put_u32(hasher, port.node.0);
    put_str(hasher, &port.port);
}

pub(crate) fn hash_plan(plan: &CompiledPlan) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("blut.compiled-plan.v3");
    put_u32(&mut hasher, plan.schema_version);
    hasher.update(&plan.graph_id.0);
    put_u32(&mut hasher, plan.realm as u32);
    put_u32(&mut hasher, plan.order.len() as u32);
    for node in &plan.order {
        put_u32(&mut hasher, node.0);
    }
    put_u32(&mut hasher, plan.nodes.len() as u32);
    for node in &plan.nodes {
        put_u32(&mut hasher, node.id.0);
        put_u32(&mut hasher, node.semantic_nodes.len() as u32);
        for semantic in &node.semantic_nodes {
            put_u32(&mut hasher, semantic.0);
        }
        put_u32(&mut hasher, node.semantic_types.len() as u32);
        for semantic_type in &node.semantic_types {
            put_str(&mut hasher, &semantic_type.type_name);
            put_u32(&mut hasher, semantic_type.version);
        }
        put_u32(&mut hasher, node.semantic_configs.len() as u32);
        for config in &node.semantic_configs {
            put_u32(&mut hasher, config.len() as u32);
            for (key, value) in config {
                put_str(&mut hasher, key);
                hash_config_value(&mut hasher, value);
            }
        }
        put_u32(&mut hasher, node.kernel.0);
        hasher.update(&node.implementation_id.0);
        hasher.update(&node.resources.peak_bytes.to_le_bytes());
        hasher.update(&node.resources.scratch_bytes.to_le_bytes());
        put_u32(&mut hasher, u32::from(node.resources.threads));
        match &node.resources.device {
            Some(device) => {
                hasher.update(&[1]);
                put_str(&mut hasher, device);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        put_u32(&mut hasher, node.determinism as u32);
        put_str(&mut hasher, &node.lowering);
        match &node.conversion {
            Some(conversion) => {
                hasher.update(&[1]);
                put_str(&mut hasher, &conversion.semantic_type);
                put_u32(&mut hasher, conversion.from as u32);
                put_u32(&mut hasher, conversion.to as u32);
                hasher.update(&conversion.max_input_bytes.to_le_bytes());
                hasher.update(&conversion.max_output_bytes.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        put_u32(&mut hasher, node.input_ports.len() as u32);
        for port in &node.input_ports {
            put_str(&mut hasher, port);
        }
        put_u32(&mut hasher, node.output_ports.len() as u32);
        for port in &node.output_ports {
            put_str(&mut hasher, port);
        }
        hash_compiled_ports(&mut hasher, &node.input_contracts);
        hash_compiled_ports(&mut hasher, &node.output_contracts);
        put_u32(&mut hasher, node.input_bindings.len() as u32);
        for binding in &node.input_bindings {
            match binding {
                crate::model::InputBinding::Buffer(buffer) => {
                    hasher.update(&[1]);
                    put_u32(&mut hasher, buffer.0);
                }
                crate::model::InputBinding::Invocation(invocation) => {
                    hasher.update(&[2]);
                    put_u32(&mut hasher, *invocation);
                }
                crate::model::InputBinding::Feedback(feedback) => {
                    hasher.update(&[3]);
                    put_u32(&mut hasher, feedback.0);
                }
                crate::model::InputBinding::Absent => {
                    hasher.update(&[0]);
                }
            }
        }
        put_u32(&mut hasher, node.output_bindings.len() as u32);
        for binding in &node.output_bindings {
            match binding {
                OutputBinding::Buffer(buffer) => {
                    hasher.update(&[1]);
                    put_u32(&mut hasher, buffer.0);
                }
                OutputBinding::Terminal => {
                    hasher.update(&[0]);
                }
            }
        }
        put_u32(&mut hasher, node.partiality as u32);
        let mut failure_domains = node.failure.domains.clone();
        failure_domains.sort_unstable();
        failure_domains.dedup();
        put_u32(&mut hasher, failure_domains.len() as u32);
        for domain in failure_domains {
            put_str(&mut hasher, &domain);
        }
        put_u32(&mut hasher, node.effect as u32);
        put_u32(&mut hasher, u32::from(node.retry_limit));
        hash_state(&mut hasher, &node.state);
        put_u32(&mut hasher, node.subgraph_path.len() as u32);
        for subgraph in &node.subgraph_path {
            hasher.update(&subgraph.0);
        }
    }
    put_u32(&mut hasher, plan.invocation_ports.len() as u32);
    for port in &plan.invocation_ports {
        put_u32(&mut hasher, port.node.0);
        put_str(&mut hasher, &port.port);
    }
    put_u32(&mut hasher, plan.buffers.len() as u32);
    for buffer in &plan.buffers {
        put_u32(&mut hasher, buffer.id.0);
        put_u32(&mut hasher, buffer.layout as u32);
        hasher.update(&buffer.capacity_bytes.to_le_bytes());
        put_u32(&mut hasher, buffer.producer.0);
        put_u32(&mut hasher, buffer.consumers.len() as u32);
        for consumer in &buffer.consumers {
            put_u32(&mut hasher, consumer.0);
        }
        put_u32(&mut hasher, buffer.last_consumer.0);
        match buffer.aliases {
            Some(alias) => {
                hasher.update(&[1]);
                put_u32(&mut hasher, alias.0);
            }
            None => {
                hasher.update(&[0]);
            }
        };
    }
    put_u32(&mut hasher, plan.feedback.len() as u32);
    for feedback in &plan.feedback {
        put_u32(&mut hasher, feedback.id.0);
        put_u32(&mut hasher, feedback.from_step.0);
        put_u32(&mut hasher, feedback.from_port);
        put_u32(&mut hasher, feedback.to_step.0);
        put_u32(&mut hasher, feedback.to_port);
        put_u32(&mut hasher, feedback.delay.invocations);
        hash_delay_initial(&mut hasher, &feedback.delay.initial);
        hasher.update(&feedback.state_bytes.to_le_bytes());
    }
    put_u32(&mut hasher, plan.propagated_proofs.len() as u32);
    for proof in &plan.propagated_proofs {
        put_str(&mut hasher, proof);
    }
    put_u32(&mut hasher, plan.propagated_policy.len() as u32);
    for policy in &plan.propagated_policy {
        put_str(&mut hasher, policy);
    }
    put_u32(&mut hasher, u32::from(plan.resulting_fidelity));
    hasher.update(&plan.peak_bytes.to_le_bytes());
    hasher.update(&plan.persistent_state_bytes.to_le_bytes());
    hash_session(&mut hasher, plan.session.as_ref());
    *hasher.finalize().as_bytes()
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    put_u32(hasher, value.len() as u32);
    hasher.update(value.as_bytes());
}

fn put_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use alloc::vec;

    use super::*;
    use crate::model::{
        Capability, Determinism, Effect, FidelityContract, ImplementationId, KernelDescriptor,
        Layout, NodeInstance, NodeTypeRef, PolicyContract, PortRef, ProofContract,
        ResourceEnvelope,
    };

    struct SemanticKernels;

    impl crate::KernelExecutor for SemanticKernels {
        type Value = u32;

        fn execute(
            &mut self,
            node: &CompiledNode,
            inputs: &[Option<&Self::Value>],
        ) -> Result<Vec<Self::Value>, crate::ExecutionError> {
            let input = inputs
                .first()
                .and_then(|value| *value)
                .copied()
                .unwrap_or_default();
            let value = match node.implementation_id.0[0] {
                1 => input + 10,
                4 => input * 2,
                7 => input.saturating_sub(3),
                100 => (input + 10) * 2,
                other => panic!("unexpected test implementation {other}"),
            };
            Ok(vec![value; node.output_bindings.len()])
        }
    }

    struct NoTransactions;

    impl crate::TransactionalSink for NoTransactions {
        fn prepare(&mut self, _idempotency_key: &str) -> Result<(), crate::ExecutionError> {
            Ok(())
        }

        fn commit(&mut self, _idempotency_key: &str) -> Result<String, crate::ExecutionError> {
            unreachable!("fixture nodes are pure")
        }

        fn abort(&mut self, _idempotency_key: &str) {}
    }

    fn descriptor(name: &str, input: bool) -> NodeDescriptor {
        NodeDescriptor {
            type_name: name.to_string(),
            version: 1,
            inputs: if input {
                vec![PortDescriptor {
                    name: "in".to_string(),
                    semantic_type: "abir.block".to_string(),
                    optional: false,
                    layouts: vec![Layout::Canonical],
                    max_bytes: 64,
                    ..PortDescriptor::opaque("in", "abir.block", 64)
                }]
            } else {
                vec![]
            },
            outputs: vec![PortDescriptor {
                name: "out".to_string(),
                semantic_type: "abir.block".to_string(),
                optional: false,
                layouts: vec![Layout::Canonical],
                max_bytes: 64,
                ..PortDescriptor::opaque("out", "abir.block", 64)
            }],
            capabilities: vec![Capability("abir".to_string())],
            targets: vec![Target::Host, Target::McuAot, Target::BlutDurable],
            resources: ResourceEnvelope::bounded(64, 0, 1),
            determinism: Determinism::BitExact,
            config: crate::ConfigSchema {
                fields: vec![crate::ConfigField {
                    name: "gain".into(),
                    value_type: crate::ConfigType::Text { max_bytes: 16 },
                    required: false,
                    default: None,
                }],
            },
            state: StateContract::stateless(),
            subgraph: None,
            proof: ProofContract {
                requires: vec![],
                provides: vec![format!("{name}.verified")],
                invalidates: vec![],
            },
            policy: PolicyContract {
                requires: vec![],
                adds: vec![],
            },
            fidelity: FidelityContract {
                minimum_input: 0,
                maximum_loss: 0,
            },
            partiality: crate::Partiality::Atomic,
            failure: crate::FailureContract { domains: vec![] },
            effect: Effect::Pure,
            retry_limit: 0,
        }
    }

    fn fixture(reverse: bool) -> (KernelRegistry, Graph) {
        let mut registry = KernelRegistry::default();
        for (index, name) in ["source", "process", "sink"].into_iter().enumerate() {
            registry
                .register_descriptor(descriptor(name, index != 0))
                .unwrap();
            for (target_index, target) in [Target::Host, Target::McuAot, Target::BlutDurable]
                .into_iter()
                .enumerate()
            {
                registry
                    .register_kernel(KernelDescriptor {
                        id: KernelId((index * 3 + target_index) as u32),
                        implements: vec![NodeTypeRef {
                            type_name: name.to_string(),
                            version: 1,
                        }],
                        implementation_id: ImplementationId(
                            [(index * 3 + target_index + 1) as u8; 32],
                        ),
                        conversion: None,
                        target,
                        input_layouts: vec![Layout::Canonical],
                        output_layouts: vec![Layout::Canonical],
                        resources: ResourceEnvelope::bounded(64, 0, 1),
                        determinism: Determinism::BitExact,
                        lowering: "test".to_string(),
                    })
                    .unwrap();
            }
        }
        for (target_index, target) in [Target::Host, Target::McuAot, Target::BlutDurable]
            .into_iter()
            .enumerate()
        {
            registry
                .register_kernel(KernelDescriptor {
                    id: KernelId(100 + target_index as u32),
                    implements: vec![
                        NodeTypeRef {
                            type_name: "source".to_string(),
                            version: 1,
                        },
                        NodeTypeRef {
                            type_name: "process".to_string(),
                            version: 1,
                        },
                    ],
                    implementation_id: ImplementationId([100 + target_index as u8; 32]),
                    conversion: None,
                    target,
                    input_layouts: vec![Layout::Canonical],
                    output_layouts: vec![Layout::Canonical],
                    resources: ResourceEnvelope::bounded(64, 0, 1),
                    determinism: Determinism::BitExact,
                    lowering: "test.fused.source-process".to_string(),
                })
                .unwrap();
        }
        let mut nodes = vec![
            NodeInstance {
                id: NodeId(0),
                descriptor: "source".to_string(),
                descriptor_version: 1,
                config: BTreeMap::new(),
            },
            NodeInstance {
                id: NodeId(1),
                descriptor: "process".to_string(),
                descriptor_version: 1,
                config: BTreeMap::new(),
            },
            NodeInstance {
                id: NodeId(2),
                descriptor: "sink".to_string(),
                descriptor_version: 1,
                config: BTreeMap::new(),
            },
        ];
        if reverse {
            nodes.reverse();
        }
        let graph = Graph {
            version: 3,
            nodes,
            edges: vec![
                Edge {
                    from: PortRef {
                        node: NodeId(0),
                        port: "out".to_string(),
                    },
                    to: PortRef {
                        node: NodeId(1),
                        port: "in".to_string(),
                    },
                },
                Edge {
                    from: PortRef {
                        node: NodeId(1),
                        port: "out".to_string(),
                    },
                    to: PortRef {
                        node: NodeId(2),
                        port: "in".to_string(),
                    },
                },
            ],
            feedback: vec![],
            invocation_inputs: vec![],
            required_capabilities: vec![Capability("abir".to_string())],
            required_proofs: vec![],
            policy: vec![],
            minimum_fidelity: u16::MAX,
            session: None,
        };
        (registry, graph)
    }

    #[test]
    fn deterministic_across_insertion_order() {
        let (registry_a, graph_a) = fixture(false);
        let (registry_template, graph_b) = fixture(true);
        let mut registry_b = KernelRegistry::default();
        for descriptor in registry_template.descriptors.values().rev() {
            registry_b.register_descriptor(descriptor.clone()).unwrap();
        }
        for kernel in registry_template.kernels.values().rev() {
            registry_b.register_kernel(kernel.clone()).unwrap();
        }
        let a = Compiler::new(&registry_a, ExecutionRealm::HostStream)
            .compile(&graph_a)
            .unwrap();
        let b = Compiler::new(&registry_b, ExecutionRealm::HostStream)
            .compile(&graph_b)
            .unwrap();
        assert_eq!(a.graph_id, b.graph_id);
        assert_eq!(a.plan_id, b.plan_id);
        assert_eq!(a.order, vec![NodeId(0), NodeId(1), NodeId(2)]);
        assert_eq!(a.nodes.len(), 2, "source and process must fuse");
    }

    #[test]
    fn physical_lowering_is_independent_of_layout_declaration_order() {
        let (mut registry_a, graph) = fixture(false);
        for descriptor in registry_a.descriptors.values_mut() {
            for port in descriptor.inputs.iter_mut().chain(&mut descriptor.outputs) {
                port.layouts.push(Layout::TimeMajor);
            }
        }
        for kernel in registry_a.kernels.values_mut() {
            kernel.input_layouts.push(Layout::TimeMajor);
            kernel.output_layouts.push(Layout::TimeMajor);
        }
        let mut registry_b = registry_a.clone();
        for kernel in registry_b.kernels.values_mut() {
            kernel.input_layouts.reverse();
            kernel.output_layouts.reverse();
        }

        let a = Compiler::new(&registry_a, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        let b = Compiler::new(&registry_b, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(a.plan_id, b.plan_id);
        assert!(
            a.buffers
                .iter()
                .all(|buffer| buffer.layout == Layout::Canonical)
        );
    }

    #[test]
    fn semantic_graph_identity_covers_graph_contracts() {
        let (registry, graph) = fixture(false);
        let baseline = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap()
            .graph_id;

        let mut variants = Vec::new();
        let mut policy = graph.clone();
        policy.policy.push("export-controlled".to_string());
        variants.push(policy);
        let mut proofs = graph.clone();
        proofs.required_proofs.push("calibrated".to_string());
        variants.push(proofs);
        let mut fidelity = graph.clone();
        fidelity.minimum_fidelity -= 1;
        variants.push(fidelity);

        for variant in variants {
            let identity = Compiler::new(&registry, ExecutionRealm::HostStream)
                .compile(&variant)
                .unwrap()
                .graph_id;
            assert_ne!(identity, baseline);
        }

        let mut unsupported = graph.clone();
        unsupported
            .required_capabilities
            .push(Capability("accelerator".to_string()));
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&unsupported),
            Err(CompileError::CapabilityUnsupported(
                "accelerator".to_string()
            ))
        );
    }

    #[test]
    fn semantic_graph_rejects_empty_proof_and_policy_names() {
        let (registry, mut graph) = fixture(false);
        graph.required_proofs.push(String::new());
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::InvalidGraphContract)
        );

        graph.required_proofs.clear();
        graph.policy.push(String::new());
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::InvalidGraphContract)
        );
    }

    #[test]
    fn instance_configuration_reaches_the_selected_physical_step() {
        let (registry, mut graph) = fixture(false);
        graph.nodes[1].config.insert(
            "gain".to_string(),
            crate::ConfigValue::Text("2".to_string()),
        );
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes[0].semantic_nodes, vec![NodeId(0), NodeId(1)]);
        assert_eq!(
            plan.nodes[0].semantic_configs[1]["gain"],
            crate::ConfigValue::Text("2".into())
        );
    }

    #[test]
    fn semantic_set_order_and_duplicates_do_not_change_identity() {
        let (registry, mut graph) = fixture(false);
        graph
            .policy
            .extend(["alpha".to_string(), "beta".to_string()]);
        graph
            .required_proofs
            .extend(["proof-a".to_string(), "proof-b".to_string()]);
        graph
            .required_capabilities
            .push(Capability("abir".to_string()));
        let canonical = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();

        let mut reordered = graph.clone();
        reordered.policy.reverse();
        reordered.policy.push("alpha".to_string());
        reordered.required_proofs.reverse();
        reordered.required_proofs.push("proof-a".to_string());
        reordered.required_capabilities.reverse();
        reordered
            .required_capabilities
            .push(Capability("abir".to_string()));
        let duplicate = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&reordered)
            .unwrap();
        assert_eq!(canonical.graph_id, duplicate.graph_id);
        assert_eq!(canonical.plan_id, duplicate.plan_id);
    }

    #[test]
    fn descriptor_and_implementation_contracts_are_identity_bound() {
        let (registry, graph) = fixture(false);
        let baseline = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();

        let mut descriptor_changed = registry.clone();
        descriptor_changed
            .descriptors
            .get_mut(&("source".to_string(), 1))
            .unwrap()
            .policy
            .adds
            .push("regulated".to_string());
        let semantic_change = Compiler::new(&descriptor_changed, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_ne!(baseline.graph_id, semantic_change.graph_id);

        let mut implementation_changed = registry.clone();
        implementation_changed
            .kernels
            .get_mut(&KernelId(100))
            .unwrap()
            .implementation_id = ImplementationId([222; 32]);
        let physical_change = Compiler::new(&implementation_changed, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(baseline.graph_id, physical_change.graph_id);
        assert_ne!(baseline.plan_id, physical_change.plan_id);
    }

    #[test]
    fn weaker_kernel_determinism_cannot_satisfy_a_bit_exact_node() {
        let (mut registry, graph) = fixture(false);
        registry.kernels.get_mut(&KernelId(0)).unwrap().determinism =
            Determinism::NumericallyEquivalent;
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::KernelUnavailable(NodeId(0), Target::Host))
        );
    }

    #[test]
    fn duplicate_registry_identities_do_not_replace_the_first_definition() {
        let (mut registry, graph) = fixture(false);
        let descriptor_key = ("source".to_string(), 1);
        let original_descriptor = registry.descriptors[&descriptor_key].clone();
        let mut replacement_descriptor = original_descriptor.clone();
        replacement_descriptor.targets.clear();
        assert_eq!(
            registry.register_descriptor(replacement_descriptor),
            Err(CompileError::DuplicateDescriptor("source".to_string(), 1))
        );
        assert_eq!(registry.descriptors[&descriptor_key], original_descriptor);

        let original_kernel = registry.kernels[&KernelId(0)].clone();
        let mut replacement_kernel = original_kernel.clone();
        replacement_kernel.implements[0].type_name = "replacement".to_string();
        assert_eq!(
            registry.register_kernel(replacement_kernel),
            Err(CompileError::DuplicateKernel(KernelId(0)))
        );
        assert_eq!(registry.kernels[&KernelId(0)], original_kernel);
        Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
    }

    #[test]
    fn descriptor_registration_rejects_empty_contract_names() {
        let mut descriptor_contract = descriptor("empty-descriptor-contract", false);
        descriptor_contract.policy.adds.push(String::new());
        let mut registry = KernelRegistry::default();
        assert_eq!(
            registry.register_descriptor(descriptor_contract),
            Err(CompileError::InvalidDescriptor(
                "empty-descriptor-contract".to_string(),
                1
            ))
        );

        let mut port_contract = descriptor("empty-port-contract", false);
        port_contract.outputs[0].proof.provides.push(String::new());
        assert_eq!(
            registry.register_descriptor(port_contract),
            Err(CompileError::InvalidDescriptor(
                "empty-port-contract".to_string(),
                1
            ))
        );
    }

    #[test]
    fn rejects_cycle_before_lowering() {
        let (registry, mut graph) = fixture(false);
        graph.edges.push(Edge {
            from: PortRef {
                node: NodeId(2),
                port: "out".to_string(),
            },
            to: PortRef {
                node: NodeId(1),
                port: "in".to_string(),
            },
        });
        let error = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap_err();
        assert!(matches!(
            error,
            CompileError::DuplicateInput(..) | CompileError::Cycle
        ));
    }

    #[test]
    fn fusion_preserves_semantic_identity() {
        let (registry, graph) = fixture(false);
        let fused = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        let plain = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_fusion(false)
            .compile(&graph)
            .unwrap();
        assert_eq!(fused.graph_id, plain.graph_id);
        assert_ne!(fused.plan_id, plain.plan_id);
        assert_eq!(fused.order, plain.order);
        assert_eq!(fused.nodes[0].kernel, KernelId(100));

        let mut fused_kernels = SemanticKernels;
        let mut fused_transactions = NoTransactions;
        let roots = BTreeMap::new();
        let fused_result = crate::PlanExecutor::new(&mut fused_kernels, &mut fused_transactions)
            .execute(&fused, [1; 32], roots.clone())
            .unwrap();
        let mut plain_kernels = SemanticKernels;
        let mut plain_transactions = NoTransactions;
        let plain_result = crate::PlanExecutor::new(&mut plain_kernels, &mut plain_transactions)
            .execute(&plain, [1; 32], roots)
            .unwrap();
        assert_eq!(fused_result.terminal_values, plain_result.terminal_values);
        assert_eq!(
            fused_result.receipt.completed_nodes,
            plain_result.receipt.completed_nodes
        );
    }

    #[test]
    fn arbitrary_length_registered_fusion_is_selected_globally() {
        let (mut registry, graph) = fixture(false);
        registry
            .register_kernel(KernelDescriptor {
                id: KernelId(150),
                implements: ["source", "process", "sink"]
                    .into_iter()
                    .map(|type_name| NodeTypeRef {
                        type_name: type_name.to_string(),
                        version: 1,
                    })
                    .collect(),
                implementation_id: ImplementationId([150; 32]),
                conversion: None,
                target: Target::Host,
                input_layouts: vec![Layout::Canonical],
                output_layouts: vec![Layout::Canonical],
                resources: ResourceEnvelope::bounded(1, 0, 1),
                determinism: Determinism::BitExact,
                lowering: "test.fused.all".to_string(),
            })
            .unwrap();
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes.len(), 1);
        assert_eq!(
            plan.nodes[0].semantic_nodes,
            vec![NodeId(0), NodeId(1), NodeId(2)]
        );
        assert_eq!(plan.nodes[0].kernel, KernelId(150));
    }

    #[test]
    fn infeasible_cheapest_fused_kernel_does_not_mask_feasible_alternative() {
        let (mut registry, _graph) = fixture(false);
        registry
            .descriptors
            .get_mut(&("process".to_string(), 1))
            .unwrap()
            .outputs[0]
            .layouts = vec![Layout::Canonical, Layout::TimeMajor];
        registry
            .kernels
            .get_mut(&KernelId(100))
            .unwrap()
            .output_layouts = vec![Layout::TimeMajor];
        registry.kernels.get_mut(&KernelId(100)).unwrap().resources =
            ResourceEnvelope::bounded(1, 0, 1);
        registry
            .register_kernel(KernelDescriptor {
                id: KernelId(110),
                implements: vec![
                    NodeTypeRef {
                        type_name: "source".to_string(),
                        version: 1,
                    },
                    NodeTypeRef {
                        type_name: "process".to_string(),
                        version: 1,
                    },
                ],
                implementation_id: ImplementationId([110; 32]),
                conversion: None,
                target: Target::Host,
                input_layouts: vec![Layout::Canonical],
                output_layouts: vec![Layout::Canonical],
                resources: ResourceEnvelope::bounded(2, 0, 1),
                determinism: Determinism::BitExact,
                lowering: "test.fused.feasible".to_string(),
            })
            .unwrap();
        let graph = fixture(false).1;
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes[0].kernel, KernelId(110));
    }

    #[test]
    fn invocation_values_bind_only_declared_typed_ports() {
        let (mut registry, mut graph) = fixture(false);
        registry
            .descriptors
            .get_mut(&("source".to_string(), 1))
            .unwrap()
            .inputs = vec![PortDescriptor {
            name: "seed".to_string(),
            semantic_type: "abir.block".to_string(),
            optional: false,
            layouts: vec![Layout::Canonical],
            max_bytes: 64,
            ..PortDescriptor::opaque("seed", "abir.block", 64)
        }];
        let seed = PortRef {
            node: NodeId(0),
            port: "seed".to_string(),
        };
        graph.invocation_inputs = vec![seed.clone()];
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.invocation_ports, vec![seed.clone()]);
        assert_eq!(plan.nodes[0].input_ports, vec!["seed"]);
        assert_eq!(
            plan.nodes[0].input_bindings,
            vec![crate::InputBinding::Invocation(0)]
        );

        let mut kernels = SemanticKernels;
        let mut transactions = NoTransactions;
        let mut values = BTreeMap::new();
        values.insert(seed, 5);
        let result = crate::PlanExecutor::new(&mut kernels, &mut transactions)
            .execute(&plan, [4; 32], values)
            .unwrap();
        assert_eq!(result.terminal_values[&NodeId(2)], vec![27]);
    }

    #[test]
    fn infeasible_fused_implementation_falls_back_to_unfused_steps() {
        let (mut registry, graph) = fixture(false);
        registry.kernels.get_mut(&KernelId(100)).unwrap().resources =
            ResourceEnvelope::bounded(1024, 0, 1);
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_memory_limit(192)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes.len(), 3);
        assert!(plan.nodes.iter().all(|node| node.kernel != KernelId(100)));
        assert_eq!(plan.peak_bytes, 192);
    }

    #[test]
    fn hard_step_limit_is_applied_during_global_plan_search() {
        let (mut registry, graph) = fixture(false);
        registry.kernels.get_mut(&KernelId(100)).unwrap().resources =
            ResourceEnvelope::bounded(1024, 0, 1);
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_limits(CompileLimits {
                max_steps: 2,
                ..CompileLimits::default()
            })
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes.len(), 2);
        assert_eq!(plan.nodes[0].kernel, KernelId(100));
    }

    #[test]
    fn fusion_never_erases_an_unconnected_observable_output() {
        let (mut registry, graph) = fixture(false);
        registry
            .descriptors
            .get_mut(&("source".to_string(), 1))
            .unwrap()
            .outputs
            .push(PortDescriptor {
                name: "audit".to_string(),
                semantic_type: "abir.block".to_string(),
                optional: false,
                layouts: vec![Layout::Canonical],
                max_bytes: 64,
                ..PortDescriptor::opaque("audit", "abir.block", 64)
            });
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes.len(), 3);
        assert_eq!(
            plan.nodes[0].output_bindings,
            vec![OutputBinding::Buffer(BufferId(0)), OutputBinding::Terminal]
        );
    }

    #[test]
    fn fanout_shares_one_sized_buffer_and_tracks_every_consumer() {
        let (registry, mut graph) = fixture(false);
        graph.edges.push(Edge {
            from: PortRef {
                node: NodeId(0),
                port: "out".to_string(),
            },
            to: PortRef {
                node: NodeId(2),
                port: "in".to_string(),
            },
        });
        graph
            .edges
            .retain(|edge| !(edge.from.node == NodeId(1) && edge.to.node == NodeId(2)));
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.buffers.len(), 1);
        assert_eq!(plan.buffers[0].capacity_bytes, 64);
        assert_eq!(plan.buffers[0].consumers, vec![StepId(1), StepId(2)]);
        assert_eq!(plan.peak_bytes, 128);
    }

    #[test]
    fn connected_zero_sized_output_is_rejected() {
        let (mut registry, graph) = fixture(false);
        registry
            .descriptors
            .get_mut(&("source".to_string(), 1))
            .unwrap()
            .outputs[0]
            .max_bytes = 0;
        assert!(matches!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::InvalidPortSize(NodeId(0), _))
        ));
    }

    #[test]
    fn all_realms_compile_the_same_semantic_graph() {
        let (registry, graph) = fixture(false);
        let host = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        let mcu = Compiler::new(&registry, ExecutionRealm::McuAot)
            .compile(&graph)
            .unwrap();
        let durable = Compiler::new(&registry, ExecutionRealm::BlutDurable)
            .compile(&graph)
            .unwrap();
        assert_eq!(host.graph_id, mcu.graph_id);
        assert_eq!(host.graph_id, durable.graph_id);
        assert_eq!(host.order, mcu.order);
        assert_eq!(host.order, durable.order);
        assert_ne!(host.plan_id, mcu.plan_id);
        assert_ne!(host.plan_id, durable.plan_id);
        for plan in [&host, &mcu, &durable] {
            assert_eq!(
                crate::CompiledPlan::from_aot_bytes(
                    &plan.to_aot_bytes().unwrap(),
                    crate::PlanLimits::default()
                )
                .unwrap(),
                *plan.as_plan()
            );
        }
    }

    #[test]
    fn kernel_resources_are_included_in_peak_memory_admission() {
        let (registry, graph) = fixture(false);
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_memory_limit(128)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.peak_bytes, 128);
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_memory_limit(127)
                .compile(&graph),
            Err(CompileError::ResourceOverflow)
        );
    }

    fn fanout_graph(graph: &mut Graph) {
        graph.edges.retain(|edge| edge.from.node == NodeId(0));
        graph.edges.push(Edge {
            from: PortRef {
                node: NodeId(0),
                port: "out".to_string(),
            },
            to: PortRef {
                node: NodeId(2),
                port: "in".to_string(),
            },
        });
    }

    #[test]
    fn proofs_do_not_leak_between_parallel_branches() {
        let (mut registry, mut graph) = fixture(false);
        fanout_graph(&mut graph);
        registry
            .descriptors
            .get_mut(&("process".to_string(), 1))
            .unwrap()
            .proof
            .provides
            .push("branch-only".to_string());
        registry
            .descriptors
            .get_mut(&("sink".to_string(), 1))
            .unwrap()
            .proof
            .requires
            .push("branch-only".to_string());
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::ProofMissing(
                NodeId(2),
                "branch-only".to_string()
            ))
        );
    }

    #[test]
    fn invalidated_proof_cannot_satisfy_a_downstream_requirement() {
        let (mut registry, graph) = fixture(false);
        registry
            .descriptors
            .get_mut(&("source".to_string(), 1))
            .unwrap()
            .proof
            .provides
            .push("calibrated".to_string());
        registry
            .descriptors
            .get_mut(&("process".to_string(), 1))
            .unwrap()
            .proof
            .invalidates
            .push("calibrated".to_string());
        registry
            .descriptors
            .get_mut(&("sink".to_string(), 1))
            .unwrap()
            .proof
            .requires
            .push("calibrated".to_string());
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::ProofMissing(
                NodeId(2),
                "calibrated".to_string()
            ))
        );
    }

    #[test]
    fn parallel_fidelity_uses_the_worst_branch_not_the_sum() {
        let (mut registry, mut graph) = fixture(false);
        fanout_graph(&mut graph);
        for name in ["process", "sink"] {
            registry
                .descriptors
                .get_mut(&(name.to_string(), 1))
                .unwrap()
                .fidelity
                .maximum_loss = 100;
        }
        graph.minimum_fidelity = u16::MAX - 100;
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.resulting_fidelity, u16::MAX - 100);
    }

    #[test]
    fn physical_search_chooses_a_feasible_non_greedy_kernel_assignment() {
        let (mut registry, graph) = fixture(false);
        let source_descriptor = registry
            .descriptors
            .get_mut(&("source".to_string(), 1))
            .unwrap();
        source_descriptor.outputs[0].layouts = vec![Layout::Canonical, Layout::TimeMajor];
        let process_descriptor = registry
            .descriptors
            .get_mut(&("process".to_string(), 1))
            .unwrap();
        process_descriptor.inputs[0].layouts = vec![Layout::Canonical, Layout::TimeMajor];
        registry
            .kernels
            .get_mut(&KernelId(0))
            .unwrap()
            .output_layouts = vec![Layout::Canonical];
        registry
            .kernels
            .get_mut(&KernelId(3))
            .unwrap()
            .input_layouts = vec![Layout::TimeMajor];
        registry
            .register_kernel(KernelDescriptor {
                id: KernelId(200),
                implements: vec![NodeTypeRef {
                    type_name: "source".to_string(),
                    version: 1,
                }],
                implementation_id: ImplementationId([200; 32]),
                conversion: None,
                target: Target::Host,
                input_layouts: vec![Layout::Canonical],
                output_layouts: vec![Layout::TimeMajor],
                resources: ResourceEnvelope::bounded(65, 0, 1),
                determinism: Determinism::BitExact,
                lowering: "feasible-source".to_string(),
            })
            .unwrap();

        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_fusion(false)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes[0].kernel, KernelId(200));
        assert_eq!(plan.buffers[0].layout, Layout::TimeMajor);
    }

    #[test]
    fn physical_search_limit_fails_closed() {
        let (registry, graph) = fixture(false);
        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_limits(CompileLimits {
                    max_search_states: 1,
                    ..CompileLimits::default()
                })
                .compile(&graph),
            Err(CompileError::SearchLimitExceeded)
        );
    }

    #[test]
    fn feedback_identity_conversion_is_checked() {
        assert_eq!(
            feedback_id(u32::MAX as usize),
            Ok(crate::FeedbackId(u32::MAX))
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            feedback_id(u32::MAX as usize + 1),
            Err(CompileError::CompileLimitExceeded)
        );
    }

    fn conversion_kernel(id: u32, from: Layout, to: Layout) -> KernelDescriptor {
        KernelDescriptor {
            id: KernelId(id),
            implements: vec![],
            implementation_id: ImplementationId([id as u8; 32]),
            conversion: Some(crate::LayoutConversion {
                semantic_type: "abir.block".to_string(),
                from,
                to,
                max_input_bytes: 64,
                max_output_bytes: 64,
            }),
            target: Target::Host,
            input_layouts: vec![from],
            output_layouts: vec![to],
            resources: ResourceEnvelope::bounded(8, 0, 1),
            determinism: Determinism::BitExact,
            lowering: format!("convert-{from:?}-{to:?}"),
        }
    }

    #[test]
    fn layout_conversion_is_explicit_and_preserves_semantic_receipts() {
        struct ConversionExecutor;
        impl crate::KernelExecutor for ConversionExecutor {
            type Value = u32;

            fn execute(
                &mut self,
                node: &CompiledNode,
                inputs: &[Option<&Self::Value>],
            ) -> Result<Vec<Self::Value>, crate::ExecutionError> {
                let input = inputs
                    .iter()
                    .flatten()
                    .next()
                    .map(|value| **value)
                    .unwrap_or_default();
                let value = if node.conversion.is_some() {
                    input
                } else {
                    input + 1
                };
                Ok(vec![value; node.output_bindings.len()])
            }
        }

        let (mut registry, mut graph) = fixture(false);
        graph
            .edges
            .retain(|edge| !(edge.from.node == NodeId(1) && edge.to.node == NodeId(2)));
        graph.edges.push(Edge {
            from: PortRef {
                node: NodeId(0),
                port: "out".to_string(),
            },
            to: PortRef {
                node: NodeId(2),
                port: "in".to_string(),
            },
        });
        registry
            .descriptors
            .get_mut(&("source".to_string(), 1))
            .unwrap()
            .outputs[0]
            .layouts = vec![Layout::ChannelMajor];
        registry
            .descriptors
            .get_mut(&("process".to_string(), 1))
            .unwrap()
            .inputs[0]
            .layouts = vec![Layout::TimeMajor];
        registry
            .descriptors
            .get_mut(&("sink".to_string(), 1))
            .unwrap()
            .inputs[0]
            .layouts = vec![Layout::ChannelMajor];
        registry
            .kernels
            .get_mut(&KernelId(0))
            .unwrap()
            .output_layouts = vec![Layout::ChannelMajor];
        registry
            .kernels
            .get_mut(&KernelId(3))
            .unwrap()
            .input_layouts = vec![Layout::TimeMajor];
        registry
            .kernels
            .get_mut(&KernelId(6))
            .unwrap()
            .input_layouts = vec![Layout::ChannelMajor];

        assert!(matches!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_fusion(false)
                .compile(&graph),
            Err(CompileError::LayoutUnavailable(..))
        ));
        let mut expanding = conversion_kernel(210, Layout::ChannelMajor, Layout::TimeMajor);
        expanding.conversion.as_mut().unwrap().max_output_bytes = 128;
        registry.register_kernel(expanding).unwrap();
        assert!(matches!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_fusion(false)
                .compile(&graph),
            Err(CompileError::LayoutUnavailable(..))
        ));
        registry
            .descriptors
            .get_mut(&("process".to_string(), 1))
            .unwrap()
            .inputs[0]
            .max_bytes = 128;
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_fusion(false)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes.len(), 4);
        assert_eq!(plan.nodes[1].semantic_nodes, Vec::<NodeId>::new());
        assert!(plan.nodes[1].conversion.is_some());
        let conversion_output = plan.nodes[1].output_bindings[0];
        let OutputBinding::Buffer(conversion_output) = conversion_output else {
            panic!("conversion output must be buffered");
        };
        assert_eq!(
            plan.buffers[conversion_output.0 as usize].capacity_bytes,
            128
        );

        let roots = BTreeMap::new();
        let mut executor = ConversionExecutor;
        let mut transactions = NoTransactions;
        let result = crate::PlanExecutor::new(&mut executor, &mut transactions)
            .execute(&plan, [7; 32], roots)
            .unwrap();
        assert_eq!(result.terminal_values[&NodeId(1)], vec![2]);
        assert_eq!(result.terminal_values[&NodeId(2)], vec![2]);
        assert_eq!(result.receipt.completed_nodes, plan.order);
        assert_eq!(result.receipt.attempts.len(), 4);
        assert!(result.receipt.attempts[1].semantic_nodes.is_empty());
    }

    #[test]
    fn alias_slot_lifetime_advances_after_every_reuse() {
        let mut buffers = vec![
            BufferPlan {
                id: BufferId(0),
                layout: Layout::Canonical,
                capacity_bytes: 64,
                producer: StepId(0),
                consumers: vec![StepId(1)],
                last_consumer: StepId(1),
                aliases: None,
            },
            BufferPlan {
                id: BufferId(1),
                layout: Layout::Canonical,
                capacity_bytes: 64,
                producer: StepId(2),
                consumers: vec![StepId(4)],
                last_consumer: StepId(4),
                aliases: None,
            },
            BufferPlan {
                id: BufferId(2),
                layout: Layout::Canonical,
                capacity_bytes: 64,
                producer: StepId(3),
                consumers: vec![StepId(5)],
                last_consumer: StepId(5),
                aliases: None,
            },
        ];
        assign_aliases(&mut buffers);
        assert_eq!(buffers[1].aliases, Some(BufferId(0)));
        assert_eq!(
            buffers[2].aliases, None,
            "overlapping lifetime needs a new slot"
        );
    }

    #[test]
    fn registry_authorization_rejects_self_consistent_forged_resources() {
        let (registry, graph) = fixture(false);
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        let authorization = crate::PlanAuthorization {
            expected_realm: plan.realm,
            expected_plan_id: plan.plan_id,
        };
        assert_eq!(
            registry
                .decode_authorized_plan(
                    &plan.to_aot_bytes().unwrap(),
                    crate::PlanLimits::default(),
                    authorization,
                )
                .unwrap(),
            plan
        );

        let mut forged = plan.clone().into_plan();
        forged.nodes[0].resources.peak_bytes += 1;
        forged.peak_bytes += 1;
        forged.plan_id = PlanId(hash_plan(&forged));
        let bytes = forged.to_aot_bytes().unwrap();
        let forged_authorization = crate::PlanAuthorization {
            expected_realm: forged.realm,
            expected_plan_id: forged.plan_id,
        };
        assert!(
            CompiledPlan::from_authorized_aot_bytes(
                &bytes,
                crate::PlanLimits::default(),
                forged_authorization,
            )
            .is_ok()
        );
        assert_eq!(
            registry.decode_authorized_plan(
                &bytes,
                crate::PlanLimits::default(),
                forged_authorization,
            ),
            Err(crate::PlanDecodeError::UnauthorizedPlan)
        );
    }

    #[test]
    fn typed_configuration_defaults_are_identity_canonical_and_invalid_values_fail() {
        let (mut registry, graph) = fixture(false);
        registry
            .descriptors
            .get_mut(&("process".into(), 1))
            .unwrap()
            .config
            .fields[0]
            .default = Some(crate::ConfigValue::Text("1".into()));

        let implicit = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        let mut explicit_graph = graph.clone();
        explicit_graph.nodes[1]
            .config
            .insert("gain".into(), crate::ConfigValue::Text("1".into()));
        let explicit = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&explicit_graph)
            .unwrap();
        assert_eq!(implicit.graph_id, explicit.graph_id);
        assert_eq!(implicit.plan_id, explicit.plan_id);

        explicit_graph.nodes[1]
            .config
            .insert("gain".into(), crate::ConfigValue::Text("x".repeat(17)));
        assert!(matches!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&explicit_graph),
            Err(CompileError::InvalidConfig(NodeId(1), _))
        ));
    }

    #[test]
    fn per_port_abir_contract_mismatch_fails_before_kernel_selection() {
        let (mut registry, graph) = fixture(false);
        registry
            .descriptors
            .get_mut(&("process".into(), 1))
            .unwrap()
            .inputs[0]
            .abir
            .root = crate::AbirRootType::Recording;
        assert!(matches!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::PortContractMismatch(
                NodeId(0),
                _,
                NodeId(1),
                _
            ))
        ));
    }

    #[test]
    fn session_feedback_has_explicit_binding_and_exact_bounded_state() {
        let (mut registry, mut graph) = fixture(false);
        let source = registry.descriptors.get_mut(&("source".into(), 1)).unwrap();
        let mut history = PortDescriptor::opaque("history", "abir.block", 64);
        history.optional = true;
        source.inputs.push(history);
        registry
            .descriptors
            .get_mut(&("process".into(), 1))
            .unwrap()
            .state = StateContract {
            scope: StateScope::Session,
            max_bytes: 128,
            checkpoint: crate::CheckpointContract {
                mode: crate::CheckpointMode::Required,
                max_snapshot_bytes: 64,
                max_interval_invocations: 8,
            },
        };
        graph.feedback.push(crate::FeedbackEdge {
            from: PortRef {
                node: NodeId(1),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId(0),
                port: "history".into(),
            },
            delay: crate::DelayContract {
                invocations: 1,
                initial: crate::DelayInitial::Absent,
            },
        });
        graph.session = Some(crate::SessionContract {
            namespace: "patient-session".into(),
            max_concurrent_sessions: 16,
            max_idle_millis: 60_000,
            reset_on_plan_change: true,
        });

        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.feedback.len(), 1);
        assert_eq!(plan.persistent_state_bytes, 192);
        assert!(plan.nodes.iter().any(|step| {
            step.input_bindings
                .contains(&crate::InputBinding::Feedback(crate::FeedbackId(0)))
        }));
        let bytes = plan.to_aot_bytes().unwrap();
        assert!(CompiledPlan::from_aot_bytes(&bytes, crate::PlanLimits::default()).is_ok());

        graph.feedback[0].delay.invocations = 3;
        let longer_delay = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(longer_delay.feedback[0].state_bytes, 192);
        assert_eq!(longer_delay.persistent_state_bytes, 320);

        let mut overlapping_layouts = registry.clone();
        let mut overlapping_graph = graph.clone();
        overlapping_graph.feedback[0].from.node = NodeId(2);
        overlapping_graph.feedback.push(crate::FeedbackEdge {
            from: PortRef {
                node: NodeId(2),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId(1),
                port: "history-2".into(),
            },
            delay: crate::DelayContract {
                invocations: 3,
                initial: crate::DelayInitial::Absent,
            },
        });
        overlapping_layouts
            .descriptors
            .get_mut(&("source".into(), 1))
            .unwrap()
            .inputs[0]
            .layouts = vec![Layout::ChannelMajor, Layout::TimeMajor];
        overlapping_layouts
            .descriptors
            .get_mut(&("sink".into(), 1))
            .unwrap()
            .outputs[0]
            .layouts = vec![Layout::Canonical, Layout::TimeMajor];
        let mut history_2 = PortDescriptor::opaque("history-2", "abir.block", 64);
        history_2.optional = true;
        history_2.layouts = vec![Layout::ChannelMajor, Layout::TimeMajor];
        overlapping_layouts
            .descriptors
            .get_mut(&("process".into(), 1))
            .unwrap()
            .inputs
            .push(history_2);
        for kernel in overlapping_layouts.kernels.values_mut() {
            if kernel.implements.as_slice()
                == [NodeTypeRef {
                    type_name: "source".into(),
                    version: 1,
                }]
            {
                kernel.input_layouts = vec![Layout::ChannelMajor, Layout::TimeMajor];
            }
            if kernel.implements.as_slice()
                == [NodeTypeRef {
                    type_name: "sink".into(),
                    version: 1,
                }]
            {
                kernel.output_layouts = vec![Layout::Canonical, Layout::TimeMajor];
            }
            if kernel.implements.as_slice()
                == [NodeTypeRef {
                    type_name: "process".into(),
                    version: 1,
                }]
            {
                kernel.input_layouts =
                    vec![Layout::Canonical, Layout::ChannelMajor, Layout::TimeMajor];
            }
        }
        let overlapping = Compiler::new(&overlapping_layouts, ExecutionRealm::HostStream)
            .compile(&overlapping_graph)
            .unwrap();
        for feedback in &overlapping.feedback {
            assert_eq!(
                overlapping.nodes[feedback.from_step.0 as usize].output_contracts
                    [feedback.from_port as usize]
                    .layout,
                Layout::TimeMajor
            );
            assert_eq!(
                overlapping.nodes[feedback.to_step.0 as usize].input_contracts
                    [feedback.to_port as usize]
                    .layout,
                Layout::TimeMajor
            );
        }

        let mut incompatible_layouts = registry.clone();
        incompatible_layouts
            .descriptors
            .get_mut(&("source".into(), 1))
            .unwrap()
            .inputs[0]
            .layouts = vec![Layout::TimeMajor];
        incompatible_layouts
            .descriptors
            .get_mut(&("process".into(), 1))
            .unwrap()
            .outputs[0]
            .layouts = vec![Layout::ChannelMajor];
        incompatible_layouts
            .descriptors
            .get_mut(&("sink".into(), 1))
            .unwrap()
            .inputs[0]
            .layouts = vec![Layout::ChannelMajor];
        for kernel in incompatible_layouts.kernels.values_mut() {
            if kernel.implements.as_slice()
                == [NodeTypeRef {
                    type_name: "source".into(),
                    version: 1,
                }]
            {
                kernel.input_layouts = vec![Layout::TimeMajor];
            }
            if kernel.implements.as_slice()
                == [NodeTypeRef {
                    type_name: "process".into(),
                    version: 1,
                }]
            {
                kernel.output_layouts = vec![Layout::ChannelMajor];
            }
            if kernel.implements.as_slice()
                == [NodeTypeRef {
                    type_name: "sink".into(),
                    version: 1,
                }]
            {
                kernel.input_layouts = vec![Layout::ChannelMajor];
            }
        }
        assert_eq!(
            Compiler::new(&incompatible_layouts, ExecutionRealm::HostStream)
                .compile(&graph)
                .unwrap_err(),
            CompileError::InvalidFeedback(NodeId(0), "history".into())
        );

        registry
            .descriptors
            .get_mut(&("source".into(), 1))
            .unwrap()
            .inputs[0]
            .optional = false;
        assert!(matches!(
            Compiler::new(&registry, ExecutionRealm::HostStream).compile(&graph),
            Err(CompileError::InvalidFeedback(NodeId(0), _))
        ));
    }

    #[test]
    fn hierarchical_lowering_identity_and_depth_are_bounded() {
        let (mut registry, graph) = fixture(false);
        let mut leaf = crate::SubgraphSchema {
            id: crate::SubgraphId([0; 32]),
            version: 1,
            nodes: vec![crate::SubgraphNode {
                id: NodeId(0),
                node_type: NodeTypeRef {
                    type_name: "process".into(),
                    version: 1,
                },
                config: BTreeMap::from([("gain".into(), crate::ConfigValue::Text("1".into()))]),
                child: None,
            }],
            edges: vec![],
            inputs: vec![crate::SubgraphInterfacePort {
                name: "in".into(),
                inner: PortRef {
                    node: NodeId(0),
                    port: "in".into(),
                },
            }],
            outputs: vec![crate::SubgraphInterfacePort {
                name: "out".into(),
                inner: PortRef {
                    node: NodeId(0),
                    port: "out".into(),
                },
            }],
        };
        leaf.id = subgraph_identity(&leaf);
        registry.register_subgraph(leaf.clone()).unwrap();
        let mut repeated = leaf.clone();
        repeated.nodes.push(crate::SubgraphNode {
            id: NodeId(1),
            node_type: NodeTypeRef {
                type_name: "process".into(),
                version: 1,
            },
            config: BTreeMap::from([("gain".into(), crate::ConfigValue::Text("1".into()))]),
            child: None,
        });
        repeated.id = subgraph_identity(&repeated);
        assert_ne!(
            leaf.id, repeated.id,
            "repeated instances are identity-bearing"
        );
        let mut parent = crate::SubgraphSchema {
            id: crate::SubgraphId([0; 32]),
            version: 1,
            nodes: vec![crate::SubgraphNode {
                id: NodeId(0),
                node_type: NodeTypeRef {
                    type_name: "process".into(),
                    version: 1,
                },
                config: BTreeMap::from([("gain".into(), crate::ConfigValue::Text("1".into()))]),
                child: Some(leaf.id),
            }],
            edges: vec![],
            inputs: vec![crate::SubgraphInterfacePort {
                name: "in".into(),
                inner: PortRef {
                    node: NodeId(0),
                    port: "in".into(),
                },
            }],
            outputs: vec![crate::SubgraphInterfacePort {
                name: "out".into(),
                inner: PortRef {
                    node: NodeId(0),
                    port: "out".into(),
                },
            }],
        };
        parent.id = subgraph_identity(&parent);
        registry.register_subgraph(parent.clone()).unwrap();
        registry
            .descriptors
            .get_mut(&("process".into(), 1))
            .unwrap()
            .subgraph = Some(crate::SubgraphLowering {
            subgraph: parent.id,
            input_map: vec![crate::PortMap {
                outer: "in".into(),
                inner: "in".into(),
            }],
            output_map: vec![crate::PortMap {
                outer: "out".into(),
                inner: "out".into(),
            }],
        });

        assert_eq!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_limits(CompileLimits {
                    max_subgraph_entries: 5,
                    ..CompileLimits::default()
                })
                .compile(&graph),
            Err(CompileError::SubgraphEntryLimitExceeded)
        );
        assert!(matches!(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_limits(CompileLimits {
                    max_subgraph_depth: 1,
                    ..CompileLimits::default()
                })
                .compile(&graph),
            Err(CompileError::SubgraphDepthExceeded)
        ));
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .with_limits(CompileLimits {
                max_subgraph_depth: 2,
                ..CompileLimits::default()
            })
            .compile(&graph)
            .unwrap();
        assert!(
            plan.nodes
                .iter()
                .any(|step| step.subgraph_path == [parent.id])
        );
    }
}
