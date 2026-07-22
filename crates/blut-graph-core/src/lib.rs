// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic semantic graph compilation for ABIR-compatible module seams.
//!
//! This crate deliberately contains no filesystem, network, async-runtime, or
//! biosignal-format dependency. A graph describes semantic nodes; compilation
//! selects compatible kernels, validates all contracts, allocates bounded
//! buffers, and emits one canonical plan for MCU, host, or durable execution.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod compile;
mod config;
mod execute;
mod mcu;
mod model;
mod plugin;
mod wire;

pub use compile::{CompileError, CompileLimits, Compiler, KernelRegistry, subgraph_identity};
pub use config::{ConfigError, ConfigField, ConfigSchema, ConfigType, ConfigValue};
pub use execute::{
    ExecutionAttempt, ExecutionError, ExecutionFailure, ExecutionReceipt, ExecutionResult,
    GapReceipt, KernelExecution, KernelExecutor, KernelGap, PlanExecutor, StructuredFailure,
    TransactionalSink,
};
pub use mcu::{McuArenaRequirements, McuPlanError};
pub use model::{
    AbirRootType, AbirSemanticType, AbirViewType, AuthorizedPlan, BufferId, BufferPlan, Capability,
    CheckpointContract, CheckpointMode, CompiledNode, CompiledPlan, CompiledPortContract,
    DelayContract, DelayInitial, Determinism, Edge, Effect, ExecutionRealm, ExtentContract,
    FailureContract, FeedbackEdge, FeedbackId, FeedbackPlan, FidelityContract, Graph, GraphId,
    ImplementationId, InputBinding, KernelDescriptor, KernelId, Layout, LayoutConversion,
    LeaseAccess, LeaseContract, LeaseLifetime, NodeDescriptor, NodeId, NodeInstance, NodeTypeRef,
    OutputBinding, Partiality, PlanId, PolicyContract, PortDescriptor, PortMap, PortRef,
    ProofContract, ResourceEnvelope, SessionContract, StateContract, StateScope, StepId,
    SubgraphId, SubgraphInterfacePort, SubgraphLowering, SubgraphNode, SubgraphSchema, Target,
};
pub use plugin::{
    ExecutableDigestAlgorithm, PLUGIN_PROTOCOL_VERSION, PluginControlFrame, PluginControlLimits,
    PluginError, PluginFailure, PluginHost, PluginLifecycle, PluginManifest, PluginRequest,
    PluginResponse, ProcessContract, TeardownPolicy, executable_digest,
};
pub use wire::{PlanAuthorization, PlanDecodeError, PlanLimits};
