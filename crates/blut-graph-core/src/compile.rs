// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::model::{
    AuthorizedPlan, BufferId, BufferPlan, CompiledNode, CompiledPlan, Edge, ExecutionRealm, Graph,
    GraphId, KernelDescriptor, KernelId, Layout, NodeDescriptor, NodeId, NodeTypeRef,
    OutputBinding, PlanId, PortDescriptor, StepId, Target,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    UnsupportedGraphVersion(u32),
    DuplicateNode(NodeId),
    UnknownNode(NodeId),
    UnknownDescriptor(String, u32),
    DuplicateDescriptor(String, u32),
    InvalidDescriptor(String, u32),
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
                    || !names.insert(port.name.as_str())
            })
        }
        if descriptor.type_name.is_empty()
            || descriptor.version == 0
            || descriptor.resources.threads == 0
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
        {
            return Err(CompileError::InvalidDescriptor(key.0, key.1));
        }
        self.descriptors.insert(key, descriptor);
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
            if node.conversion.is_some() {
                if !kernel.implements.is_empty()
                    || node.input_ports.as_slice() != ["input"]
                    || node.output_ports.as_slice() != ["output"]
                    || node.partiality != crate::model::Partiality::Atomic
                    || !node.failure.domains.is_empty()
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
                    || node.checkpointable != first.checkpointable
                    || node.partiality != first.partiality
                    || node.failure != first.failure
                {
                    return Err(crate::PlanDecodeError::UnauthorizedPlan);
                }
            } else if node.effect != crate::model::Effect::Pure
                || node.retry_limit != 0
                || node.checkpointable
                || node.partiality != crate::model::Partiality::Atomic
                || !node.failure.domains.is_empty()
                || semantic_descriptors.iter().any(|descriptor| {
                    descriptor.effect != crate::model::Effect::Pure
                        || descriptor.stateful
                        || descriptor.retry_limit != 0
                        || descriptor.checkpointable
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
        if graph.version != 2 {
            return Err(CompileError::UnsupportedGraphVersion(graph.version));
        }
        if graph.nodes.is_empty() {
            return Err(CompileError::EmptyGraph);
        }
        if graph.nodes.len() > self.limits.max_semantic_nodes {
            return Err(CompileError::CompileLimitExceeded);
        }

        let mut nodes = BTreeMap::new();
        for node in &graph.nodes {
            if nodes.insert(node.id, node).is_some() {
                return Err(CompileError::DuplicateNode(node.id));
            }
        }

        let mut descriptors = BTreeMap::new();
        let target = self.realm.target();
        let required_caps: BTreeSet<_> = graph.required_capabilities.iter().collect();
        for node in &graph.nodes {
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
            for capability in &descriptor.capabilities {
                if !required_caps.contains(capability) {
                    return Err(CompileError::CapabilityMissing(capability.0.clone()));
                }
            }
            descriptors.insert(node.id, descriptor);
        }
        let supplied_caps: BTreeSet<_> = descriptors
            .values()
            .flat_map(|descriptor| descriptor.capabilities.iter())
            .collect();
        if let Some(extra) = required_caps.difference(&supplied_caps).next() {
            return Err(CompileError::CapabilityUnsupported(extra.0.clone()));
        }

        let invocation_ports = self.verify_edges(graph, &descriptors)?;
        let order = topological_order(graph)?;
        let (proofs, policy, fidelity) = propagate_contracts(graph, &order, &descriptors)?;
        let (compiled_nodes, buffers, peak_bytes) =
            self.select_and_lower(graph, &order, &nodes, &descriptors, target)?;
        let graph_id = GraphId(hash_graph(graph, &descriptors));
        let mut plan = CompiledPlan {
            schema_version: 2,
            graph_id,
            plan_id: PlanId([0; 32]),
            realm: self.realm,
            order,
            nodes: compiled_nodes,
            buffers,
            invocation_ports,
            propagated_proofs: proofs,
            propagated_policy: policy,
            resulting_fidelity: fidelity,
            peak_bytes,
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
            if !bound.insert(edge.to.clone()) {
                return Err(CompileError::DuplicateInput(
                    edge.to.node,
                    edge.to.port.clone(),
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
                Ok((nodes, buffers, peak))
                    if peak <= self.max_peak_bytes
                        && nodes.len() <= self.limits.max_steps
                        && buffers.len() <= self.limits.max_buffers =>
                {
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
        checkpointable: if fused { false } else { first.checkpointable },
    }
}

fn linear_fusion_is_safe(
    graph: &Graph,
    ids: &[NodeId],
    descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
) -> bool {
    if ids.iter().any(|id| {
        let descriptor = descriptors[id];
        descriptor.effect != crate::model::Effect::Pure
            || descriptor.partiality != crate::model::Partiality::Atomic
            || descriptor.stateful
            || descriptor.retry_limit != 0
            || descriptor.checkpointable
            || !descriptor.failure.domains.is_empty()
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
                for kernel in &route.path {
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
                        conversion: Some(conversion),
                        input_ports: alloc::vec!["input".to_string()],
                        output_ports: alloc::vec!["output".to_string()],
                        input_bindings: alloc::vec![crate::model::InputBinding::Absent],
                        output_bindings: alloc::vec![OutputBinding::Terminal],
                        partiality: crate::model::Partiality::Atomic,
                        failure: crate::model::FailureContract {
                            domains: Vec::new(),
                        },
                        effect: crate::model::Effect::Pure,
                        retry_limit: 0,
                        checkpointable: false,
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
    assign_aliases(&mut buffers);
    let arena_bytes = buffers
        .iter()
        .filter(|buffer| buffer.aliases.is_none())
        .try_fold(0u64, |sum, buffer| {
            sum.checked_add(buffer.capacity_bytes)
                .ok_or(CompileError::ResourceOverflow)
        })?;
    let workspace = nodes.iter().try_fold(0u64, |peak, node| {
        let bytes = node
            .resources
            .peak_bytes
            .checked_add(node.resources.scratch_bytes)
            .ok_or(CompileError::ResourceOverflow)?;
        Ok::<_, CompileError>(peak.max(bytes))
    })?;
    let peak = arena_bytes
        .checked_add(workspace)
        .ok_or(CompileError::ResourceOverflow)?;
    Ok((nodes, buffers, peak))
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
    let mut hasher = blake3::Hasher::new_derive_key("blut.graph.v2");
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
            put_str(&mut hasher, &value);
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
    put_u32(hasher, u32::from(descriptor.stateful));
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
    put_u32(hasher, u32::from(descriptor.checkpointable));
}

fn put_str_set(hasher: &mut blake3::Hasher, values: &[&str]) {
    put_u32(hasher, values.len() as u32);
    for value in values {
        put_str(hasher, value);
    }
}

pub(crate) fn hash_plan(plan: &CompiledPlan) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("blut.compiled-plan.v2");
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
                put_str(&mut hasher, value);
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
        put_u32(&mut hasher, u32::from(node.checkpointable));
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
            }],
            capabilities: vec![Capability("abir".to_string())],
            targets: vec![Target::Host, Target::McuAot, Target::BlutDurable],
            resources: ResourceEnvelope::bounded(64, 0, 1),
            determinism: Determinism::BitExact,
            stateful: false,
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
            checkpointable: false,
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
            version: 2,
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
            invocation_inputs: vec![],
            required_capabilities: vec![Capability("abir".to_string())],
            required_proofs: vec![],
            policy: vec![],
            minimum_fidelity: u16::MAX,
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
    fn instance_configuration_reaches_the_selected_physical_step() {
        let (registry, mut graph) = fixture(false);
        graph.nodes[1]
            .config
            .insert("gain".to_string(), "2".to_string());
        let plan = Compiler::new(&registry, ExecutionRealm::HostStream)
            .compile(&graph)
            .unwrap();
        assert_eq!(plan.nodes[0].semantic_nodes, vec![NodeId(0), NodeId(1)]);
        assert_eq!(plan.nodes[0].semantic_configs[1]["gain"], "2");
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
}
