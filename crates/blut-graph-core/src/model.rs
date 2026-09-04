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

/// An opaque execution target token.
///
/// This was an enum until 0.3.0, and the change is deliberate: the compiler
/// has no business knowing that a target is called "host". It compares tokens,
/// orders them, and folds them into the plan hash — nothing more. The domain
/// layer names them (ADR 0034), exactly as it already names `DomainToken`.
///
/// THE ORDINALS ARE WIRE. They are folded into `graph_id` and `plan_id` as
/// little-endian `u32`, and `#[serde(transparent)]` makes the postcard encoding
/// byte-identical to the variant indices the enum emitted. The values below are
/// therefore historically assigned and may never be renumbered.
///
/// `Debug` is wire too, and that is less obvious: `KernelDescriptor::lowering`
/// is conventionally built as `format!("{target:?}")`, and `lowering` is a
/// hashed field. The hand-written `Debug` below reproduces the enum's output
/// exactly for that reason; a derived one would print `Target(1)` and move every
/// plan id in the fleet without touching an ordinal.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Target(u32);

#[allow(non_upper_case_globals)]
impl Target {
    /// Historically assigned 0.
    pub const McuAot: Self = Self(0);
    /// Historically assigned 1.
    pub const Host: Self = Self(1);
    /// Historically assigned 2.
    pub const BlutDurable: Self = Self(2);

    /// Every token this version of the crate knows, in ordinal order.
    pub const KNOWN: [Self; 3] = [Self::McuAot, Self::Host, Self::BlutDurable];

    /// The wire value. Named `token` rather than `as u32` so that the cast
    /// sites are greppable and cannot be written by accident.
    pub const fn token(self) -> u32 {
        self.0
    }

    /// Build a token from a wire value, WITHOUT range checking — decoding
    /// untrusted bytes must call `is_known` as well. See `is_known`.
    pub const fn from_token(token: u32) -> Self {
        Self(token)
    }

    /// Whether this token is one this version assigns a meaning to.
    ///
    /// The enum's derived `Deserialize` used to reject an out-of-range variant
    /// index for free. A transparent newtype accepts any `u32`, so the check
    /// that was implicit is explicit here, and `from_aot_bytes` calls it.
    pub const fn is_known(self) -> bool {
        self.0 <= 2
    }

    const fn name(self) -> Option<&'static str> {
        match self.0 {
            0 => Some("McuAot"),
            1 => Some("Host"),
            2 => Some("BlutDurable"),
            _ => None,
        }
    }
}

impl core::fmt::Debug for Target {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.name() {
            Some(name) => formatter.write_str(name),
            None => write!(formatter, "Target({})", self.0),
        }
    }
}

/// An opaque execution realm token. See [`Target`] for why this is a newtype,
/// why the ordinals are wire, and why `Debug` is hand-written.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExecutionRealm(u32);

#[allow(non_upper_case_globals)]
impl ExecutionRealm {
    /// Historically assigned 0.
    pub const McuAot: Self = Self(0);
    /// Historically assigned 1.
    pub const HostStream: Self = Self(1);
    /// Historically assigned 2.
    pub const BlutDurable: Self = Self(2);

    /// Every token this version of the crate knows, in ordinal order.
    pub const KNOWN: [Self; 3] = [Self::McuAot, Self::HostStream, Self::BlutDurable];

    pub const fn token(self) -> u32 {
        self.0
    }

    pub const fn from_token(token: u32) -> Self {
        Self(token)
    }

    pub const fn is_known(self) -> bool {
        self.0 <= 2
    }

    /// The target a realm lowers to.
    ///
    /// Kept as a total function on the realm because the compile-side selection
    /// and the decode-side authorization must not be able to disagree; an
    /// unknown realm maps to an unknown target rather than to a default, so a
    /// forged plan cannot borrow the host's target by being out of range.
    pub const fn target(self) -> Target {
        match self.0 {
            0 => Target::McuAot,
            1 => Target::Host,
            2 => Target::BlutDurable,
            other => Target::from_token(other),
        }
    }

    const fn name(self) -> Option<&'static str> {
        match self.0 {
            0 => Some("McuAot"),
            1 => Some("HostStream"),
            2 => Some("BlutDurable"),
            _ => None,
        }
    }
}

impl core::fmt::Debug for ExecutionRealm {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.name() {
            Some(name) => formatter.write_str(name),
            None => write!(formatter, "ExecutionRealm({})", self.0),
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

/// An opaque buffer-layout token. See [`Target`] for why this is a newtype,
/// why the ordinals are wire, and why `Debug` is hand-written.
///
/// `Ord` is load-bearing beyond hashing here: `select_layout` resolves a port's
/// admissible layouts with `.min()`, so the ordinal order IS the selection
/// rule, and two further sites sort or compare layouts to break routing ties.
/// Derived `Ord` over the `u32` preserves all three exactly.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Layout(u32);

#[allow(non_upper_case_globals)]
impl Layout {
    /// Historically assigned 0.
    pub const Canonical: Self = Self(0);
    /// Historically assigned 1.
    pub const ChannelMajor: Self = Self(1);
    /// Historically assigned 2.
    pub const TimeMajor: Self = Self(2);
    /// Historically assigned 3.
    pub const Packed: Self = Self(3);
    /// Historically assigned 4.
    pub const Opaque: Self = Self(4);

    /// Every token this version of the crate knows, in ordinal order.
    pub const KNOWN: [Self; 5] = [
        Self::Canonical,
        Self::ChannelMajor,
        Self::TimeMajor,
        Self::Packed,
        Self::Opaque,
    ];

    pub const fn token(self) -> u32 {
        self.0
    }

    pub const fn from_token(token: u32) -> Self {
        Self(token)
    }

    pub const fn is_known(self) -> bool {
        self.0 <= 4
    }

    const fn name(self) -> Option<&'static str> {
        match self.0 {
            0 => Some("Canonical"),
            1 => Some("ChannelMajor"),
            2 => Some("TimeMajor"),
            3 => Some("Packed"),
            4 => Some("Opaque"),
            _ => None,
        }
    }
}

impl core::fmt::Debug for Layout {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.name() {
            Some(name) => formatter.write_str(name),
            None => write!(formatter, "Layout({})", self.0),
        }
    }
}

/// An opaque, domain-supplied classification token.
///
/// `blut-graph-core` NEVER interprets these. The compiler does exactly three
/// things with a token: compares it to another for edge compatibility, rejects
/// it when empty, and folds its bytes into the plan hash. It has no opinion
/// about what any particular token *means*.
///
/// That is the point. The vocabulary belongs to the domain layer (ADR 0034) —
/// a biosignal domain names recordings and signal blocks, a vision domain names
/// frames and tensors, and the compiler stays ignorant of both. Before the 2026-08-26 domain-token migration this
/// slot was a pair of enums (`AbirRootType`/`AbirViewType`) that hard-coded one
/// domain's taxonomy into the compiler; consumers were already escaping it
/// through an `Unknown(String)` variant in 10 of 22 call sites, which is the
/// shape below with extra steps.
///
/// Construction is deliberately permissive — an empty token is representable and
/// is rejected by [`crate::compile`]'s port-contract validation, exactly as the
/// empty `Unknown("")` was. Validity is the compiler's judgement, not the
/// constructor's.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DomainToken(String);

impl DomainToken {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<&str> for DomainToken {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl From<String> for DomainToken {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// A port's domain classification: the artifact, and the projection of it.
///
/// `root` names the thing that exists; `view` names the way this port looks at
/// it. The distinction is real and worth keeping structured — a dataset read as
/// a stream is not the same contract as a dataset read whole — but both sides
/// are domain vocabulary, so both are opaque [`DomainToken`]s.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DomainType {
    pub root: DomainToken,
    pub view: DomainToken,
}

impl DomainType {
    pub fn new(root: impl Into<DomainToken>, view: impl Into<DomainToken>) -> Self {
        Self {
            root: root.into(),
            view: view.into(),
        }
    }
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
    pub domain: DomainType,
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
            domain: DomainType {
                root: DomainToken::new("blob-ref"),
                view: DomainToken::new("atom"),
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
    pub domain: DomainType,
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
            domain: port.domain,
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
