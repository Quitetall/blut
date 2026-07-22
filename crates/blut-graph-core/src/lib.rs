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
mod execute;
mod mcu;
mod model;
mod plugin;
mod wire;

pub use compile::{CompileError, CompileLimits, Compiler, KernelRegistry};
pub use execute::{
    ExecutionAttempt, ExecutionError, ExecutionFailure, ExecutionReceipt, ExecutionResult,
    GapReceipt, KernelExecution, KernelExecutor, KernelGap, PlanExecutor, StructuredFailure,
    TransactionalSink,
};
pub use mcu::{McuArenaRequirements, McuPlanError};
pub use model::{
    AuthorizedPlan, BufferId, BufferPlan, Capability, CompiledNode, CompiledPlan, Determinism,
    Edge, Effect, ExecutionRealm, FailureContract, FidelityContract, Graph, GraphId,
    ImplementationId, InputBinding, KernelDescriptor, KernelId, Layout, LayoutConversion,
    NodeDescriptor, NodeId, NodeInstance, NodeTypeRef, OutputBinding, Partiality, PlanId,
    PolicyContract, PortDescriptor, PortRef, ProofContract, ResourceEnvelope, StepId, Target,
};
pub use plugin::{PluginError, PluginHost, PluginManifest, PluginRequest, PluginResponse};
pub use wire::{PlanAuthorization, PlanDecodeError, PlanLimits};
