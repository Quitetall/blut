// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u32);
    };
}

id_type!(NodeId);
id_type!(KernelId);
id_type!(BufferId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GraphId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlanId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Capability(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    McuAot,
    Host,
    BlutDurable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionRealm {
    McuAot,
    HostStream,
    BlutDurable,
}

impl ExecutionRealm {
    pub const fn target(self) -> Target {
        match self {
            Self::McuAot => Target::McuAot,
            Self::HostStream => Target::Host,
            Self::BlutDurable => Target::BlutDurable,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Determinism {
    BitExact,
    NumericallyEquivalent,
    Seeded,
    Nondeterministic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Effect {
    Pure,
    Idempotent,
    Transactional,
    AtMostOnce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Layout {
    Canonical,
    ChannelMajor,
    TimeMajor,
    Packed,
    Opaque,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceEnvelope {
    pub peak_bytes: u64,
    pub scratch_bytes: u64,
    pub threads: u16,
    pub device: Option<String>,
}

impl ResourceEnvelope {
    pub const fn bounded(peak_bytes: u64, scratch_bytes: u64, threads: u16) -> Self {
        Self {
            peak_bytes,
            scratch_bytes,
            threads,
            device: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortDescriptor {
    pub name: String,
    pub semantic_type: String,
    pub optional: bool,
    pub layouts: Vec<Layout>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofContract {
    pub requires: Vec<String>,
    pub provides: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyContract {
    pub requires: Vec<String>,
    pub adds: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidelityContract {
    pub minimum_input: u16,
    pub maximum_loss: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeDescriptor {
    pub type_name: String,
    pub version: u32,
    pub inputs: Vec<PortDescriptor>,
    pub outputs: Vec<PortDescriptor>,
    pub capabilities: Vec<Capability>,
    pub targets: Vec<Target>,
    pub resources: ResourceEnvelope,
    pub determinism: Determinism,
    pub stateful: bool,
    pub proof: ProofContract,
    pub policy: PolicyContract,
    pub fidelity: FidelityContract,
    pub effect: Effect,
    pub retry_limit: u16,
    pub checkpointable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInstance {
    pub id: NodeId,
    pub descriptor: String,
    pub descriptor_version: u32,
    pub config: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PortRef {
    pub node: NodeId,
    pub port: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub from: PortRef,
    pub to: PortRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Graph {
    pub version: u32,
    pub nodes: Vec<NodeInstance>,
    pub edges: Vec<Edge>,
    pub required_capabilities: Vec<Capability>,
    pub required_proofs: Vec<String>,
    pub policy: Vec<String>,
    pub minimum_fidelity: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelDescriptor {
    pub id: KernelId,
    pub node_type: String,
    pub node_version: u32,
    pub target: Target,
    pub input_layouts: Vec<Layout>,
    pub output_layouts: Vec<Layout>,
    pub resources: ResourceEnvelope,
    pub determinism: Determinism,
    pub lowering: String,
    pub fuses_with_next: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledNode {
    pub semantic_nodes: Vec<NodeId>,
    pub kernel: KernelId,
    pub input_buffers: Vec<BufferId>,
    pub output_buffers: Vec<BufferId>,
    pub effect: Effect,
    pub retry_limit: u16,
    pub checkpointable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferPlan {
    pub id: BufferId,
    pub layout: Layout,
    pub capacity_bytes: u64,
    pub producer: NodeId,
    pub last_consumer: NodeId,
    pub aliases: Option<BufferId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledPlan {
    pub schema_version: u32,
    pub graph_id: GraphId,
    pub plan_id: PlanId,
    pub realm: ExecutionRealm,
    pub order: Vec<NodeId>,
    pub nodes: Vec<CompiledNode>,
    pub buffers: Vec<BufferPlan>,
    pub propagated_proofs: Vec<String>,
    pub propagated_policy: Vec<String>,
    pub resulting_fidelity: u16,
    pub peak_bytes: u64,
}
