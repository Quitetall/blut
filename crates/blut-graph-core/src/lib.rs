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
mod model;
mod plugin;
mod wire;

pub use compile::{CompileError, Compiler, KernelRegistry};
pub use execute::{
    ExecutionError, ExecutionReceipt, KernelExecutor, PlanExecutor, TransactionalSink,
};
pub use model::{
    BufferId, BufferPlan, Capability, CompiledNode, CompiledPlan, Determinism, Edge, Effect,
    ExecutionRealm, FidelityContract, Graph, GraphId, KernelDescriptor, KernelId, Layout,
    NodeDescriptor, NodeId, NodeInstance, PlanId, PolicyContract, PortDescriptor, PortRef,
    ProofContract, ResourceEnvelope, Target,
};
pub use plugin::{PluginError, PluginHost, PluginManifest, PluginRequest, PluginResponse};
pub use wire::{PlanDecodeError, PlanLimits};
