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
id_type!(StepId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GraphId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlanId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImplementationId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeTypeRef {
    pub type_name: String,
    pub version: u32,
}

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
    AtLeastOnce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Partiality {
    /// Either all declared outputs are produced or the node fails.
    Atomic,
    /// A successful attempt may include explicit, machine-readable gaps.
    ExplicitGaps,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureContract {
    /// Stable namespaced failure domains a kernel is allowed to report.
    pub domains: Vec<String>,
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
    pub max_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofContract {
    pub requires: Vec<String>,
    pub provides: Vec<String>,
    pub invalidates: Vec<String>,
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
    pub partiality: Partiality,
    pub failure: FailureContract,
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
    /// External values accepted by this graph invocation. Every entry names a
    /// concrete descriptor input port; declarations are canonicalized by the
    /// compiler and may not overlap an edge binding.
    #[serde(default)]
    pub invocation_inputs: Vec<PortRef>,
    pub required_capabilities: Vec<Capability>,
    pub required_proofs: Vec<String>,
    pub policy: Vec<String>,
    pub minimum_fidelity: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelDescriptor {
    pub id: KernelId,
    /// Exact semantic chain implemented by this kernel. A chain longer than
    /// one is an explicit fused implementation, never an optimizer guess.
    pub implements: Vec<NodeTypeRef>,
    /// Content identity of implementation code/build inputs. It must change
    /// whenever executable behavior changes, even if the local KernelId does not.
    pub implementation_id: ImplementationId,
    /// A physical layout conversion implemented by this kernel. Conversion
    /// kernels have an empty `implements` chain and execute as explicit plan
    /// steps with no semantic-node identity.
    pub conversion: Option<LayoutConversion>,
    pub target: Target,
    pub input_layouts: Vec<Layout>,
    pub output_layouts: Vec<Layout>,
    pub resources: ResourceEnvelope,
    pub determinism: Determinism,
    pub lowering: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayoutConversion {
    pub semantic_type: String,
    pub from: Layout,
    pub to: Layout,
    pub max_input_bytes: u64,
    pub max_output_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OutputBinding {
    Buffer(BufferId),
    Terminal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InputBinding {
    Buffer(BufferId),
    Invocation(u32),
    Absent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledNode {
    /// Dense physical execution identity. Unlike `NodeId`, this includes
    /// compiler-inserted conversion steps.
    pub id: StepId,
    /// One or more semantic nodes implemented by this step. This is empty only
    /// for an explicit compiler-inserted physical conversion.
    pub semantic_nodes: Vec<NodeId>,
    /// Exact registered semantic types aligned one-to-one with
    /// `semantic_nodes`; empty for conversion steps.
    pub semantic_types: Vec<NodeTypeRef>,
    /// Normalized instance configurations aligned one-to-one with
    /// `semantic_nodes`; empty for conversion steps.
    pub semantic_configs: Vec<BTreeMap<String, String>>,
    pub kernel: KernelId,
    pub implementation_id: ImplementationId,
    pub resources: ResourceEnvelope,
    pub determinism: Determinism,
    pub lowering: String,
    pub conversion: Option<LayoutConversion>,
    /// Stable physical port names aligned with the ordered bindings below.
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    /// One binding per physical kernel input, in descriptor port order.
    pub input_bindings: Vec<InputBinding>,
    /// One binding per physical kernel output, in descriptor port order.
    /// Unconnected semantic outputs are explicit terminal invocation results.
    pub output_bindings: Vec<OutputBinding>,
    pub partiality: Partiality,
    pub failure: FailureContract,
    pub effect: Effect,
    pub retry_limit: u16,
    pub checkpointable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferPlan {
    pub id: BufferId,
    pub layout: Layout,
    pub capacity_bytes: u64,
    pub producer: StepId,
    pub consumers: Vec<StepId>,
    /// Cached final consumer in topological order for constant-time liveness release.
    pub last_consumer: StepId,
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
    /// Canonical port table addressed by `InputBinding::Invocation`.
    pub invocation_ports: Vec<PortRef>,
    pub propagated_proofs: Vec<String>,
    pub propagated_policy: Vec<String>,
    pub resulting_fidelity: u16,
    pub peak_bytes: u64,
}

/// A compiled plan whose physical steps have been selected from, or checked
/// against, a trusted kernel registry. Structural AOT decoding deliberately
/// returns `CompiledPlan`; only local compilation or registry-bound decoding
/// can construct this executable wrapper.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizedPlan {
    plan: CompiledPlan,
}

impl AuthorizedPlan {
    pub(crate) const fn new(plan: CompiledPlan) -> Self {
        Self { plan }
    }

    pub const fn as_plan(&self) -> &CompiledPlan {
        &self.plan
    }

    pub fn into_plan(self) -> CompiledPlan {
        self.plan
    }
}

impl core::ops::Deref for AuthorizedPlan {
    type Target = CompiledPlan;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}
