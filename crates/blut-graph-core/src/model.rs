// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::config::{ConfigSchema, ConfigValue};

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
id_type!(FeedbackId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GraphId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlanId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SubgraphId(pub [u8; 32]);

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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AbirRootType {
    Dataset,
    Recording,
    Stream,
    SignalBlock,
    TemporalTable,
    Table,
    Tensor,
    EncodedBlock,
    BlobRef,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AbirViewType {
    Root,
    Recording,
    Stream,
    Block,
    Tensor,
    Atom,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AbirSemanticType {
    pub root: AbirRootType,
    pub view: AbirViewType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtentContract {
    /// Number of logical dimensions. Zero denotes an opaque scalar/blob atom.
    pub rank: u8,
    /// Per-dimension upper bounds; exactly `rank` entries.
    pub maximum_shape: Vec<u64>,
    pub max_elements: u64,
    pub ragged: bool,
    pub sparse: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LeaseAccess {
    ReadOnly,
    ExclusiveWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LeaseLifetime {
    Step,
    Invocation,
    Session,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseContract {
    pub access: LeaseAccess,
    pub lifetime: LeaseLifetime,
    pub zero_copy_permitted: bool,
    pub contiguous_required: bool,
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
    pub abir: AbirSemanticType,
    pub proof: ProofContract,
    pub policy: PolicyContract,
    pub fidelity: FidelityContract,
    pub extent: ExtentContract,
    pub lease: LeaseContract,
}

impl PortDescriptor {
    /// Conservative bounded atom contract useful for non-ABIR control values.
    pub fn opaque(
        name: impl Into<String>,
        semantic_type: impl Into<String>,
        max_bytes: u64,
    ) -> Self {
        Self {
            name: name.into(),
            semantic_type: semantic_type.into(),
            optional: false,
            layouts: alloc::vec![Layout::Canonical],
            max_bytes,
            abir: AbirSemanticType {
                root: AbirRootType::BlobRef,
                view: AbirViewType::Atom,
            },
            proof: ProofContract {
                requires: Vec::new(),
                provides: Vec::new(),
                invalidates: Vec::new(),
            },
            policy: PolicyContract {
                requires: Vec::new(),
                adds: Vec::new(),
            },
            fidelity: FidelityContract {
                minimum_input: 0,
                maximum_loss: 0,
            },
            extent: ExtentContract {
                rank: 0,
                maximum_shape: Vec::new(),
                max_elements: 1,
                ragged: false,
                sparse: false,
            },
            lease: LeaseContract {
                access: LeaseAccess::ReadOnly,
                lifetime: LeaseLifetime::Invocation,
                zero_copy_permitted: false,
                contiguous_required: false,
            },
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StateScope {
    Stateless,
    Invocation,
    Session,
    Durable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckpointMode {
    Disabled,
    Optional,
    Required,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointContract {
    pub mode: CheckpointMode,
    pub max_snapshot_bytes: u64,
    pub max_interval_invocations: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateContract {
    pub scope: StateScope,
    pub max_bytes: u64,
    pub checkpoint: CheckpointContract,
}

impl StateContract {
    pub const fn stateless() -> Self {
        Self {
            scope: StateScope::Stateless,
            max_bytes: 0,
            checkpoint: CheckpointContract {
                mode: CheckpointMode::Disabled,
                max_snapshot_bytes: 0,
                max_interval_invocations: 0,
            },
        }
    }

    pub const fn checkpointable(&self) -> bool {
        !matches!(self.checkpoint.mode, CheckpointMode::Disabled)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionContract {
    pub namespace: String,
    pub max_concurrent_sessions: u32,
    pub max_idle_millis: u64,
    pub reset_on_plan_change: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelayContract {
    /// Number of completed invocations between write and visibility.
    pub invocations: u32,
    pub initial: DelayInitial,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DelayInitial {
    Absent,
    Zeroed,
    ContentId([u8; 32]),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackEdge {
    pub from: PortRef,
    pub to: PortRef,
    pub delay: DelayContract,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PortMap {
    pub outer: String,
    pub inner: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SubgraphConfigMap {
    /// Field on the outer descriptor.
    pub outer: String,
    /// Local node receiving the bound value.
    pub node: NodeId,
    /// Field on the inner node descriptor.
    pub inner: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubgraphLowering {
    pub subgraph: SubgraphId,
    pub input_map: Vec<PortMap>,
    pub output_map: Vec<PortMap>,
    /// Exact outer-instance configuration bindings into the inner DAG.
    pub config_map: Vec<SubgraphConfigMap>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubgraphNode {
    /// Identity local to the subgraph; repeated node types remain distinct.
    pub id: NodeId,
    pub node_type: NodeTypeRef,
    pub config: BTreeMap<String, ConfigValue>,
    /// Optional nested decomposition invoked by this local node.
    pub child: Option<SubgraphId>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SubgraphInterfacePort {
    pub name: String,
    pub inner: PortRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubgraphSchema {
    pub id: SubgraphId,
    pub version: u32,
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<Edge>,
    pub inputs: Vec<SubgraphInterfacePort>,
    pub outputs: Vec<SubgraphInterfacePort>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializedSubgraph {
    pub graph: Graph,
    /// Outer input names bound to concrete inner ports.
    pub inputs: Vec<SubgraphInterfacePort>,
    /// Outer output names bound to concrete inner ports.
    pub outputs: Vec<SubgraphInterfacePort>,
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
    pub config: ConfigSchema,
    pub state: StateContract,
    pub subgraph: Option<SubgraphLowering>,
    pub proof: ProofContract,
    pub policy: PolicyContract,
    pub fidelity: FidelityContract,
    pub partiality: Partiality,
    pub failure: FailureContract,
    pub effect: Effect,
    pub retry_limit: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInstance {
    pub id: NodeId,
    pub descriptor: String,
    pub descriptor_version: u32,
    pub config: BTreeMap<String, ConfigValue>,
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
    /// Cross-invocation edges are explicit and never participate in same-call
    /// topological ordering.
    #[serde(default)]
    pub feedback: Vec<FeedbackEdge>,
    /// External values accepted by this graph invocation. Every entry names a
    /// concrete descriptor input port; declarations are canonicalized by the
    /// compiler and may not overlap an edge binding.
    #[serde(default)]
    pub invocation_inputs: Vec<PortRef>,
    pub required_capabilities: Vec<Capability>,
    pub required_proofs: Vec<String>,
    pub policy: Vec<String>,
    pub minimum_fidelity: u16,
    pub session: Option<SessionContract>,
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
    Feedback(FeedbackId),
    Absent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledPortContract {
    pub name: String,
    pub semantic_type: String,
    pub optional: bool,
    pub layout: Layout,
    pub max_bytes: u64,
    pub abir: AbirSemanticType,
    pub proof: ProofContract,
    pub policy: PolicyContract,
    pub fidelity: FidelityContract,
    pub extent: ExtentContract,
    pub lease: LeaseContract,
}

impl CompiledPortContract {
    pub fn opaque(
        name: impl Into<String>,
        semantic_type: impl Into<String>,
        layout: Layout,
        max_bytes: u64,
    ) -> Self {
        let port = PortDescriptor::opaque(name, semantic_type, max_bytes);
        Self {
            name: port.name,
            semantic_type: port.semantic_type,
            optional: port.optional,
            layout,
            max_bytes: port.max_bytes,
            abir: port.abir,
            proof: port.proof,
            policy: port.policy,
            fidelity: port.fidelity,
            extent: port.extent,
            lease: port.lease,
        }
    }
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
    pub semantic_configs: Vec<BTreeMap<String, ConfigValue>>,
    pub kernel: KernelId,
    pub implementation_id: ImplementationId,
    pub resources: ResourceEnvelope,
    pub determinism: Determinism,
    pub lowering: String,
    pub conversion: Option<LayoutConversion>,
    /// Stable physical port names aligned with the ordered bindings below.
    pub input_ports: Vec<String>,
    pub output_ports: Vec<String>,
    pub input_contracts: Vec<CompiledPortContract>,
    pub output_contracts: Vec<CompiledPortContract>,
    /// One binding per physical kernel input, in descriptor port order.
    pub input_bindings: Vec<InputBinding>,
    /// One binding per physical kernel output, in descriptor port order.
    /// Unconnected semantic outputs are explicit terminal invocation results.
    pub output_bindings: Vec<OutputBinding>,
    pub partiality: Partiality,
    pub failure: FailureContract,
    pub effect: Effect,
    pub retry_limit: u16,
    pub state: StateContract,
    /// Identity lineage of semantic decompositions used to reach this step.
    pub subgraph_path: Vec<SubgraphId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackPlan {
    pub id: FeedbackId,
    pub from_step: StepId,
    pub from_port: u32,
    pub to_step: StepId,
    pub to_port: u32,
    pub delay: DelayContract,
    pub state_bytes: u64,
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
    pub feedback: Vec<FeedbackPlan>,
    /// Canonical port table addressed by `InputBinding::Invocation`.
    pub invocation_ports: Vec<PortRef>,
    pub propagated_proofs: Vec<String>,
    pub propagated_policy: Vec<String>,
    pub resulting_fidelity: u16,
    /// Peak invocation memory: live buffers, kernel workspaces/scratch, and
    /// invocation-scoped state. Session/durable state and feedback history are
    /// excluded and accounted by `persistent_state_bytes`.
    pub peak_bytes: u64,
    /// Session/durable state plus feedback history retained across invocations.
    pub persistent_state_bytes: u64,
    pub session: Option<SessionContract>,
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
