// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::model::{
    BufferId, BufferPlan, CompiledNode, CompiledPlan, Edge, ExecutionRealm, Graph, GraphId,
    KernelDescriptor, KernelId, Layout, NodeDescriptor, NodeId, PlanId, PortDescriptor, Target,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    UnsupportedGraphVersion(u32),
    DuplicateNode(NodeId),
    UnknownNode(NodeId),
    UnknownDescriptor(String, u32),
    DuplicateKernel(KernelId),
    UnknownPort(NodeId, String),
    InvalidPortSize(NodeId, String),
    TypeMismatch(String, String),
    MissingInput(NodeId, String),
    DuplicateInput(NodeId, String),
    Cycle,
    CapabilityMissing(String),
    TargetUnsupported(NodeId, Target),
    KernelUnavailable(NodeId, Target),
    ProofMissing(NodeId, String),
    PolicyMissing(NodeId, String),
    FidelityInsufficient(NodeId),
    UnsafeRetry(NodeId),
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
    pub fn register_descriptor(&mut self, descriptor: NodeDescriptor) {
        self.descriptors.insert(
            (descriptor.type_name.clone(), descriptor.version),
            descriptor,
        );
    }

    pub fn register_kernel(&mut self, kernel: KernelDescriptor) -> Result<(), CompileError> {
        let id = kernel.id;
        if self.kernels.insert(id, kernel).is_some() {
            return Err(CompileError::DuplicateKernel(id));
        }
        Ok(())
    }
}

pub struct Compiler<'a> {
    registry: &'a KernelRegistry,
    realm: ExecutionRealm,
    max_peak_bytes: u64,
    fuse: bool,
}

impl<'a> Compiler<'a> {
    pub const fn new(registry: &'a KernelRegistry, realm: ExecutionRealm) -> Self {
        Self {
            registry,
            realm,
            max_peak_bytes: u64::MAX,
            fuse: true,
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

    pub fn compile(&self, graph: &Graph) -> Result<CompiledPlan, CompileError> {
        if graph.version != 1 {
            return Err(CompileError::UnsupportedGraphVersion(graph.version));
        }
        if graph.nodes.is_empty() {
            return Err(CompileError::EmptyGraph);
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

        self.verify_edges(graph, &descriptors)?;
        let order = topological_order(graph)?;
        let kernels = self.select_kernels(&order, &nodes, target)?;
        let (proofs, policy, fidelity) = propagate_contracts(graph, &order, &descriptors)?;
        let buffers = allocate_buffers(graph, &order, &descriptors, &kernels)?;
        let peak_bytes = buffers.iter().try_fold(0u64, |sum, buffer| {
            sum.checked_add(buffer.capacity_bytes)
                .ok_or(CompileError::ResourceOverflow)
        })?;
        if peak_bytes > self.max_peak_bytes {
            return Err(CompileError::ResourceOverflow);
        }
        let compiled_nodes =
            build_compiled_nodes(graph, &order, &descriptors, &kernels, &buffers, self.fuse);
        let graph_id = GraphId(hash_graph(graph));
        let mut plan = CompiledPlan {
            schema_version: 1,
            graph_id,
            plan_id: PlanId([0; 32]),
            realm: self.realm,
            order,
            nodes: compiled_nodes,
            buffers,
            propagated_proofs: proofs,
            propagated_policy: policy,
            resulting_fidelity: fidelity,
            peak_bytes,
        };
        plan.plan_id = PlanId(hash_plan(&plan));
        Ok(plan)
    }

    fn verify_edges(
        &self,
        graph: &Graph,
        descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
    ) -> Result<(), CompileError> {
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
        for (node_id, descriptor) in descriptors {
            for input in descriptor.inputs.iter().filter(|port| !port.optional) {
                if !bound.contains(&crate::model::PortRef {
                    node: *node_id,
                    port: input.name.clone(),
                }) {
                    return Err(CompileError::MissingInput(*node_id, input.name.clone()));
                }
            }
        }
        Ok(())
    }

    fn select_kernels(
        &self,
        order: &[NodeId],
        nodes: &BTreeMap<NodeId, &crate::model::NodeInstance>,
        target: Target,
    ) -> Result<BTreeMap<NodeId, &KernelDescriptor>, CompileError> {
        let mut selected = BTreeMap::new();
        for node_id in order {
            let node = nodes[node_id];
            let kernel = self
                .registry
                .kernels
                .values()
                .filter(|kernel| {
                    kernel.node_type == node.descriptor
                        && kernel.node_version == node.descriptor_version
                        && kernel.target == target
                })
                .min_by_key(|kernel| {
                    (
                        kernel.resources.peak_bytes,
                        kernel.resources.scratch_bytes,
                        kernel.id,
                    )
                })
                .ok_or(CompileError::KernelUnavailable(*node_id, target))?;
            selected.insert(*node_id, kernel);
        }
        Ok(selected)
    }
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
    let mut proofs: BTreeSet<String> = graph.required_proofs.iter().cloned().collect();
    let mut policy: BTreeSet<String> = graph.policy.iter().cloned().collect();
    let mut fidelity = u16::MAX;
    for id in order {
        let descriptor = descriptors[id];
        for required in &descriptor.proof.requires {
            if !proofs.contains(required) {
                return Err(CompileError::ProofMissing(*id, required.clone()));
            }
        }
        for required in &descriptor.policy.requires {
            if !policy.contains(required) {
                return Err(CompileError::PolicyMissing(*id, required.clone()));
            }
        }
        if fidelity < descriptor.fidelity.minimum_input {
            return Err(CompileError::FidelityInsufficient(*id));
        }
        fidelity = fidelity.saturating_sub(descriptor.fidelity.maximum_loss);
        proofs.extend(descriptor.proof.provides.iter().cloned());
        policy.extend(descriptor.policy.adds.iter().cloned());
    }
    if fidelity < graph.minimum_fidelity {
        return Err(CompileError::FidelityInsufficient(
            *order.last().expect("non-empty graph"),
        ));
    }
    Ok((
        proofs.into_iter().collect(),
        policy.into_iter().collect(),
        fidelity,
    ))
}

fn allocate_buffers(
    graph: &Graph,
    order: &[NodeId],
    descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
    kernels: &BTreeMap<NodeId, &KernelDescriptor>,
) -> Result<Vec<BufferPlan>, CompileError> {
    let position: BTreeMap<NodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();
    let mut grouped: BTreeMap<crate::model::PortRef, Vec<NodeId>> = BTreeMap::new();
    for edge in &graph.edges {
        grouped
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.node);
    }
    let mut buffers = Vec::new();
    for (source, mut consumers) in grouped {
        consumers.sort();
        consumers.dedup();
        let last_consumer = *consumers
            .iter()
            .max_by_key(|consumer| position[consumer])
            .expect("edge group is non-empty");
        let producer = descriptors[&source.node];
        let output = find_port(&producer.outputs, source.node, &source.port)?;
        if output.max_bytes == 0 {
            return Err(CompileError::InvalidPortSize(source.node, source.port));
        }
        let kernel = kernels[&source.node];
        let layout = kernel
            .output_layouts
            .iter()
            .find(|layout| output.layouts.contains(layout))
            .copied()
            .unwrap_or(Layout::Canonical);
        buffers.push(BufferPlan {
            id: BufferId(buffers.len() as u32),
            layout,
            capacity_bytes: output.max_bytes,
            producer: source.node,
            consumers,
            last_consumer,
            aliases: None,
        });
    }
    buffers.sort_by_key(|buffer| {
        (
            position[&buffer.producer],
            position[&buffer.last_consumer],
            buffer.id,
        )
    });
    for (index, buffer) in buffers.iter_mut().enumerate() {
        buffer.id = BufferId(index as u32);
    }
    Ok(buffers)
}

fn build_compiled_nodes(
    _graph: &Graph,
    order: &[NodeId],
    descriptors: &BTreeMap<NodeId, &NodeDescriptor>,
    kernels: &BTreeMap<NodeId, &KernelDescriptor>,
    buffers: &[BufferPlan],
    fuse: bool,
) -> Vec<CompiledNode> {
    let mut result: Vec<CompiledNode> = Vec::new();
    for id in order {
        let descriptor = descriptors[id];
        let kernel = kernels[id];
        let input_buffers = buffers
            .iter()
            .filter(|buffer| buffer.consumers.contains(id))
            .map(|buffer| buffer.id)
            .collect();
        let output_buffers = buffers
            .iter()
            .filter(|buffer| buffer.producer == *id)
            .map(|buffer| buffer.id)
            .collect();
        if fuse && let Some(previous) = result.last_mut() {
            let previous_last = *previous.semantic_nodes.last().expect("compiled node");
            let previous_kernel = kernels[&previous_last];
            if previous_kernel
                .fuses_with_next
                .contains(&descriptor.type_name)
                && descriptor.effect == crate::model::Effect::Pure
                && descriptors[&previous_last].effect == crate::model::Effect::Pure
                && previous.output_buffers.len() == 1
                && input_buffers == previous.output_buffers
                && previous.output_buffers.iter().all(|buffer_id| {
                    buffers
                        .iter()
                        .find(|buffer| buffer.id == *buffer_id)
                        .is_some_and(|buffer| buffer.consumers.as_slice() == [*id])
                })
            {
                previous.semantic_nodes.push(*id);
                previous.kernel = kernel.id;
                previous.output_buffers = output_buffers;
                continue;
            }
        }
        result.push(CompiledNode {
            semantic_nodes: alloc::vec![*id],
            kernel: kernel.id,
            input_buffers,
            output_buffers,
            effect: descriptor.effect,
            retry_limit: descriptor.retry_limit,
            checkpointable: descriptor.checkpointable,
        });
    }
    result
}

fn hash_graph(graph: &Graph) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("blut.graph.v1");
    put_u32(&mut hasher, graph.version);
    let mut nodes = graph.nodes.clone();
    nodes.sort_by_key(|node| node.id);
    for node in nodes {
        put_u32(&mut hasher, node.id.0);
        put_str(&mut hasher, &node.descriptor);
        put_u32(&mut hasher, node.descriptor_version);
        for (key, value) in node.config {
            put_str(&mut hasher, &key);
            put_str(&mut hasher, &value);
        }
    }
    let mut edges = graph.edges.clone();
    edges.sort_by_key(|edge| edge.to.clone());
    for edge in edges {
        put_u32(&mut hasher, edge.from.node.0);
        put_str(&mut hasher, &edge.from.port);
        put_u32(&mut hasher, edge.to.node.0);
        put_str(&mut hasher, &edge.to.port);
    }
    *hasher.finalize().as_bytes()
}

pub(crate) fn hash_plan(plan: &CompiledPlan) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("blut.compiled-plan.v1");
    put_u32(&mut hasher, plan.schema_version);
    hasher.update(&plan.graph_id.0);
    put_u32(&mut hasher, plan.realm as u32);
    put_u32(&mut hasher, plan.order.len() as u32);
    for node in &plan.order {
        put_u32(&mut hasher, node.0);
    }
    put_u32(&mut hasher, plan.nodes.len() as u32);
    for node in &plan.nodes {
        put_u32(&mut hasher, node.semantic_nodes.len() as u32);
        for semantic in &node.semantic_nodes {
            put_u32(&mut hasher, semantic.0);
        }
        put_u32(&mut hasher, node.kernel.0);
        put_u32(&mut hasher, node.input_buffers.len() as u32);
        for buffer in &node.input_buffers {
            put_u32(&mut hasher, buffer.0);
        }
        put_u32(&mut hasher, node.output_buffers.len() as u32);
        for buffer in &node.output_buffers {
            put_u32(&mut hasher, buffer.0);
        }
        put_u32(&mut hasher, node.effect as u32);
        put_u32(&mut hasher, u32::from(node.retry_limit));
        put_u32(&mut hasher, u32::from(node.checkpointable));
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
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::model::{
        Capability, Determinism, Effect, FidelityContract, KernelDescriptor, Layout, NodeInstance,
        PolicyContract, PortRef, ProofContract, ResourceEnvelope,
    };

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
            },
            policy: PolicyContract {
                requires: vec![],
                adds: vec![],
            },
            fidelity: FidelityContract {
                minimum_input: 0,
                maximum_loss: 0,
            },
            effect: Effect::Pure,
            retry_limit: 0,
            checkpointable: false,
        }
    }

    fn fixture(reverse: bool) -> (KernelRegistry, Graph) {
        let mut registry = KernelRegistry::default();
        for (index, name) in ["source", "process", "sink"].into_iter().enumerate() {
            registry.register_descriptor(descriptor(name, index != 0));
            for (target_index, target) in [Target::Host, Target::McuAot, Target::BlutDurable]
                .into_iter()
                .enumerate()
            {
                registry
                    .register_kernel(KernelDescriptor {
                        id: KernelId((index * 3 + target_index) as u32),
                        node_type: name.to_string(),
                        node_version: 1,
                        target,
                        input_layouts: vec![Layout::Canonical],
                        output_layouts: vec![Layout::Canonical],
                        resources: ResourceEnvelope::bounded(64, 0, 1),
                        determinism: Determinism::BitExact,
                        lowering: "test".to_string(),
                        fuses_with_next: if name == "source" {
                            vec!["process".to_string()]
                        } else {
                            vec![]
                        },
                    })
                    .unwrap();
            }
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
            version: 1,
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
        let (registry_b, graph_b) = fixture(true);
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
        assert_eq!(plan.buffers[0].consumers, vec![NodeId(1), NodeId(2)]);
        assert_eq!(plan.peak_bytes, 64);
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
                *plan
            );
        }
    }
}
