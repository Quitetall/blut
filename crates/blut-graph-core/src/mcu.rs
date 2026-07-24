// SPDX-License-Identifier: AGPL-3.0-or-later
//! Allocation-free execution sizing for statically linked MCU executors.
//!
//! Graph compilation and AOT authorization may allocate on a host. Once an
//! authorized MCU plan is installed, firmware uses these exact requirements to
//! provision caller-owned arenas; the execution loop never needs to grow a
//! collection or discover an undeclared bound.

use core::fmt;

use crate::model::{InputBinding, OutputBinding};
use crate::{AuthorizedPlan, Effect, ExecutionRealm, GraphId, NodeId, Partiality, PlanId, StepId};

/// Maximum physical fan-in a statically linked MCU step may declare. The
/// firmware executor gathers input references on the stack, so this bound keeps
/// the gather buffer alloc-free. Compilation already caps fan-in far below this.
pub const MAX_STATIC_STEP_INPUTS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct McuArenaRequirements {
    pub byte_arena: u64,
    pub value_slots: usize,
    pub invocation_slots: usize,
    pub max_step_inputs: usize,
    pub max_step_outputs: usize,
    pub attempt_slots: usize,
    pub terminal_slots: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McuPlanError {
    WrongRealm(ExecutionRealm),
    UnsupportedEffect(crate::StepId, Effect),
    UnboundedPartialOutput(crate::StepId),
    HostResource(crate::StepId),
    StatefulPlan,
    HierarchicalPlan(crate::StepId),
    SizeOverflow,
}

impl fmt::Display for McuPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for McuPlanError {}

impl AuthorizedPlan {
    /// Validate the firmware execution subset and return exact fixed-arena
    /// dimensions. This method performs no allocation.
    pub fn mcu_arena_requirements(&self) -> Result<McuArenaRequirements, McuPlanError> {
        if self.realm != ExecutionRealm::McuAot {
            return Err(McuPlanError::WrongRealm(self.realm));
        }
        if self.persistent_state_bytes != 0 || !self.feedback.is_empty() || self.session.is_some() {
            return Err(McuPlanError::StatefulPlan);
        }
        let mut max_step_inputs = 0usize;
        let mut max_step_outputs = 0usize;
        let mut terminal_slots = 0usize;
        for step in &self.nodes {
            if !matches!(step.effect, Effect::Pure | Effect::Idempotent) {
                return Err(McuPlanError::UnsupportedEffect(step.id, step.effect));
            }
            if step.partiality != Partiality::Atomic {
                return Err(McuPlanError::UnboundedPartialOutput(step.id));
            }
            if step.resources.threads != 1 || step.resources.device.is_some() {
                return Err(McuPlanError::HostResource(step.id));
            }
            if step.state.scope != crate::StateScope::Stateless {
                return Err(McuPlanError::StatefulPlan);
            }
            if !step.subgraph_path.is_empty() {
                return Err(McuPlanError::HierarchicalPlan(step.id));
            }
            max_step_inputs = max_step_inputs.max(step.input_bindings.len());
            max_step_outputs = max_step_outputs.max(step.output_bindings.len());
            terminal_slots = terminal_slots
                .checked_add(
                    step.output_bindings
                        .iter()
                        .filter(|binding| matches!(binding, crate::OutputBinding::Terminal))
                        .count(),
                )
                .ok_or(McuPlanError::SizeOverflow)?;
        }
        Ok(McuArenaRequirements {
            byte_arena: self.peak_bytes,
            value_slots: self.buffers.len(),
            invocation_slots: self.invocation_ports.len(),
            max_step_inputs,
            max_step_outputs,
            attempt_slots: self.nodes.len(),
            terminal_slots,
        })
    }
}

/// A structured fault raised by the statically linked MCU executor. Every
/// variant is a bounded-arena or contract violation; the executor never
/// allocates and never panics on well-formed firmware plans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticExecutionError {
    /// The plan was authorized for a different realm than `McuAot`.
    WrongRealm(ExecutionRealm),
    /// The plan is not a valid firmware subset (stateful, hierarchical, …).
    NotFirmwareSubset(McuPlanError),
    /// A caller-owned arena was smaller than the plan's exact requirement.
    ArenaTooSmall,
    /// A step declares more physical inputs than [`MAX_STATIC_STEP_INPUTS`].
    FanInTooWide(StepId),
    /// Buffer identities are not the dense `0..value_slots` the executor indexes.
    NonDenseBuffers,
    /// A step read a buffer that no prior step in topological order produced.
    MissingBuffer(StepId),
    /// A step read an invocation input the caller did not supply.
    MissingInvocation(StepId),
    /// The kernel wrote a different output count than the step declares.
    OutputArity(StepId),
    /// The bound kernel reported a fault for this step.
    KernelFault(StepId),
}

impl fmt::Display for StaticExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StaticExecutionError {}

/// A compact, `Copy`, allocation-free execution receipt. Firmware records only
/// bounded scalars; the host reconstructs full attempt detail from the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticReceipt {
    pub invocation_id: [u8; 32],
    pub graph_id: GraphId,
    pub plan_id: PlanId,
    pub realm: ExecutionRealm,
    pub completed_steps: u32,
    pub terminal_values: usize,
    pub last_step: Option<NodeId>,
}

/// Caller-owned execution arenas for [`StaticExecutor::execute`]. Firmware
/// provisions each slice once from [`McuArenaRequirements`]; the executor grows
/// none of them.
pub struct StaticArenas<'a, V> {
    /// One live-buffer slot per plan buffer (`value_slots`).
    pub values: &'a mut [Option<V>],
    /// Collected unconnected outputs (`terminal_slots`).
    pub terminals: &'a mut [Option<V>],
    /// Per-step output workspace, reused each step (`max_step_outputs`).
    pub output_scratch: &'a mut [V],
    /// One value per invocation port (`invocation_slots`).
    pub invocation: &'a [Option<V>],
}

/// A statically linked firmware kernel. Unlike [`crate::KernelExecutor`], it
/// writes into a caller-owned output slice and never allocates.
pub trait StaticKernel {
    type Value: Clone;

    /// Execute one step from immutable input references, writing exactly
    /// `outputs.len()` values into `outputs` (already sized to the step's
    /// physical output arity). Returning `Err` aborts the plan.
    fn execute(
        &mut self,
        node: &crate::CompiledNode,
        inputs: &[Option<&Self::Value>],
        outputs: &mut [Self::Value],
    ) -> Result<(), StaticExecutionError>;
}

/// The distinct, allocation-free execution engine for authorized firmware
/// plans. It consumes the exact [`McuArenaRequirements`] and executes over
/// caller-owned arenas, growing no collection and touching no allocator. This
/// is the firmware counterpart to the host [`crate::PlanExecutor`]; both drive
/// the same canonical [`AuthorizedPlan`] to an identity-stable receipt.
pub struct StaticExecutor;

impl StaticExecutor {
    /// Execute `plan` on `McuAot` over caller-owned arenas.
    ///
    /// * `values` holds one live-buffer slot per plan buffer (`value_slots`).
    /// * `terminals` collects unconnected outputs (`terminal_slots`).
    /// * `output_scratch` is reused per step (`max_step_outputs`).
    /// * `invocation` supplies one value per invocation port (`invocation_slots`).
    ///
    /// All four are sized by [`AuthorizedPlan::mcu_arena_requirements`]. The
    /// method performs no allocation.
    pub fn execute<V, K>(
        plan: &AuthorizedPlan,
        requirements: &McuArenaRequirements,
        invocation_id: [u8; 32],
        arenas: &mut StaticArenas<'_, V>,
        kernel: &mut K,
    ) -> Result<StaticReceipt, StaticExecutionError>
    where
        V: Clone + Default,
        K: StaticKernel<Value = V>,
    {
        // Disjoint field reborrows keep the executor body allocation- and
        // alias-free while presenting one bundled arena argument.
        let values = &mut *arenas.values;
        let terminals = &mut *arenas.terminals;
        let output_scratch = &mut *arenas.output_scratch;
        let invocation: &[Option<V>] = arenas.invocation;
        if plan.realm != ExecutionRealm::McuAot {
            return Err(StaticExecutionError::WrongRealm(plan.realm));
        }
        // Re-validate the firmware subset from the plan itself; never trust the
        // caller-supplied requirements without binding them to this plan.
        let checked = plan
            .mcu_arena_requirements()
            .map_err(StaticExecutionError::NotFirmwareSubset)?;
        if &checked != requirements {
            return Err(StaticExecutionError::NotFirmwareSubset(
                McuPlanError::SizeOverflow,
            ));
        }
        if values.len() < requirements.value_slots
            || terminals.len() < requirements.terminal_slots
            || output_scratch.len() < requirements.max_step_outputs
            || invocation.len() < requirements.invocation_slots
        {
            return Err(StaticExecutionError::ArenaTooSmall);
        }
        if requirements.max_step_inputs > MAX_STATIC_STEP_INPUTS {
            // The plan's widest step exceeds the stack gather bound.
            return Err(StaticExecutionError::FanInTooWide(
                plan.nodes
                    .iter()
                    .find(|step| step.input_bindings.len() > MAX_STATIC_STEP_INPUTS)
                    .map_or(StepId(0), |step| step.id),
            ));
        }
        // The executor indexes buffers by position; require dense identities so
        // no lookup map (and thus no allocation) is ever needed.
        for (index, buffer) in plan.buffers.iter().enumerate() {
            if buffer.id.0 as usize != index {
                return Err(StaticExecutionError::NonDenseBuffers);
            }
        }
        // A produced-buffer bitmap over the fixed value arena, tracked without
        // allocation by reusing `Option::is_some` on the slots themselves.
        for slot in values.iter_mut().take(requirements.value_slots) {
            *slot = None;
        }
        let mut terminal_cursor = 0usize;
        let mut completed_steps = 0u32;
        let mut last_step = None;

        for step in &plan.nodes {
            // Gather immutable input references on the stack. Scoped so the
            // borrow of `values` ends before outputs are written back.
            let mut gathered: [Option<&V>; MAX_STATIC_STEP_INPUTS] =
                [const { None }; MAX_STATIC_STEP_INPUTS];
            let input_count = step.input_bindings.len();
            {
                for (slot, binding) in gathered.iter_mut().zip(&step.input_bindings) {
                    *slot = match binding {
                        InputBinding::Absent => None,
                        InputBinding::Buffer(buffer) => {
                            let value = values
                                .get(buffer.0 as usize)
                                .and_then(Option::as_ref)
                                .ok_or(StaticExecutionError::MissingBuffer(step.id))?;
                            Some(value)
                        }
                        InputBinding::Invocation(port) => {
                            let value = invocation
                                .get(*port as usize)
                                .and_then(Option::as_ref)
                                .ok_or(StaticExecutionError::MissingInvocation(step.id))?;
                            Some(value)
                        }
                        InputBinding::Feedback(_) => {
                            // `mcu_arena_requirements` already rejects stateful
                            // plans; feedback can never reach a firmware step.
                            return Err(StaticExecutionError::NotFirmwareSubset(
                                McuPlanError::StatefulPlan,
                            ));
                        }
                    };
                }
                let output_count = step.output_bindings.len();
                let outputs = &mut output_scratch[..output_count];
                kernel
                    .execute(step, &gathered[..input_count], outputs)
                    .map_err(|_| StaticExecutionError::KernelFault(step.id))?;
            }
            // Write results back into the fixed arenas. `output_scratch` is a
            // separate slice, so this mutable borrow of `values` does not alias
            // the input gather above.
            for (index, binding) in step.output_bindings.iter().enumerate() {
                let produced = output_scratch
                    .get(index)
                    .ok_or(StaticExecutionError::OutputArity(step.id))?
                    .clone();
                match binding {
                    OutputBinding::Buffer(buffer) => {
                        let slot = values
                            .get_mut(buffer.0 as usize)
                            .ok_or(StaticExecutionError::MissingBuffer(step.id))?;
                        *slot = Some(produced);
                    }
                    OutputBinding::Terminal => {
                        let slot = terminals
                            .get_mut(terminal_cursor)
                            .ok_or(StaticExecutionError::ArenaTooSmall)?;
                        *slot = Some(produced);
                        terminal_cursor += 1;
                    }
                }
            }
            // Release buffers whose last consumer is this step, mirroring the
            // host executor's liveness rule so peak occupancy stays bounded.
            for binding in &step.input_bindings {
                if let InputBinding::Buffer(buffer) = binding
                    && plan
                        .buffers
                        .get(buffer.0 as usize)
                        .is_some_and(|plan_buffer| plan_buffer.last_consumer == step.id)
                    && let Some(slot) = values.get_mut(buffer.0 as usize)
                {
                    *slot = None;
                }
            }
            completed_steps += 1;
            last_step = step.semantic_nodes.last().copied().or(last_step);
        }

        Ok(StaticReceipt {
            invocation_id,
            graph_id: plan.graph_id,
            plan_id: plan.plan_id,
            realm: plan.realm,
            completed_steps,
            terminal_values: terminal_cursor,
            last_step,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;

    use crate::{
        AuthorizedPlan, CompiledNode, CompiledPlan, Determinism, FailureContract, GraphId,
        ImplementationId, KernelId, NodeId, NodeTypeRef, OutputBinding, Partiality, PlanId,
        ResourceEnvelope, StepId,
    };

    use super::*;

    #[test]
    fn authorized_mcu_plan_exposes_exact_fixed_arena_shape() {
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::McuAot,
            order: vec![NodeId(0)],
            nodes: vec![CompiledNode {
                id: StepId(0),
                semantic_nodes: vec![NodeId(0)],
                semantic_types: vec![NodeTypeRef {
                    type_name: "test".into(),
                    version: 1,
                }],
                semantic_configs: vec![BTreeMap::new()],
                kernel: KernelId(0),
                implementation_id: ImplementationId([2; 32]),
                resources: ResourceEnvelope::bounded(0, 0, 1),
                determinism: Determinism::BitExact,
                lowering: "static".into(),
                conversion: None,
                input_ports: vec![],
                output_ports: vec!["out".into()],
                input_contracts: vec![],
                output_contracts: vec![crate::CompiledPortContract::opaque(
                    "out",
                    "test",
                    crate::Layout::Canonical,
                    1,
                )],
                input_bindings: vec![],
                output_bindings: vec![OutputBinding::Terminal],
                partiality: Partiality::Atomic,
                failure: FailureContract { domains: vec![] },
                effect: Effect::Pure,
                retry_limit: 0,
                state: crate::StateContract::stateless(),
                subgraph_path: vec![],
            }],
            buffers: vec![],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 128,
            persistent_state_bytes: 0,
            session: None,
        };
        plan.plan_id = PlanId(crate::compile::hash_plan(&plan));
        let requirements = AuthorizedPlan::new(plan).mcu_arena_requirements().unwrap();
        assert_eq!(requirements.byte_arena, 128);
        assert_eq!(requirements.attempt_slots, 1);
        assert_eq!(requirements.terminal_slots, 1);
    }

    fn firmware_node(
        id: u32,
        inputs: vec::Vec<crate::BufferId>,
        outputs: vec::Vec<crate::BufferId>,
    ) -> CompiledNode {
        let input_count = inputs.len();
        let output_count = outputs.len().max(1);
        CompiledNode {
            id: StepId(id),
            semantic_nodes: vec![NodeId(id)],
            semantic_types: vec![NodeTypeRef {
                type_name: "test".into(),
                version: 1,
            }],
            semantic_configs: vec![BTreeMap::new()],
            kernel: KernelId(id),
            implementation_id: ImplementationId([id as u8 + 1; 32]),
            resources: ResourceEnvelope::bounded(0, 0, 1),
            determinism: Determinism::BitExact,
            lowering: "static".into(),
            conversion: None,
            input_ports: (0..input_count).map(|i| format!("in-{i}")).collect(),
            output_ports: if outputs.is_empty() {
                vec!["out".into()]
            } else {
                (0..outputs.len()).map(|i| format!("out-{i}")).collect()
            },
            input_contracts: (0..input_count)
                .map(|i| {
                    crate::CompiledPortContract::opaque(
                        format!("in-{i}"),
                        "test",
                        crate::Layout::Canonical,
                        4,
                    )
                })
                .collect(),
            output_contracts: (0..output_count)
                .map(|i| {
                    crate::CompiledPortContract::opaque(
                        if output_count == 1 {
                            "out".into()
                        } else {
                            format!("out-{i}")
                        },
                        "test",
                        crate::Layout::Canonical,
                        4,
                    )
                })
                .collect(),
            input_bindings: inputs
                .into_iter()
                .map(crate::model::InputBinding::Buffer)
                .collect(),
            output_bindings: if outputs.is_empty() {
                vec![OutputBinding::Terminal]
            } else {
                outputs.into_iter().map(OutputBinding::Buffer).collect()
            },
            partiality: Partiality::Atomic,
            failure: FailureContract { domains: vec![] },
            effect: Effect::Pure,
            retry_limit: 0,
            state: crate::StateContract::stateless(),
            subgraph_path: vec![],
        }
    }

    struct CountingKernel;

    impl super::StaticKernel for CountingKernel {
        type Value = u32;

        fn execute(
            &mut self,
            _node: &CompiledNode,
            inputs: &[Option<&u32>],
            outputs: &mut [u32],
        ) -> Result<(), super::StaticExecutionError> {
            let value = inputs.iter().flatten().map(|value| **value).sum::<u32>() + 1;
            for slot in outputs.iter_mut() {
                *slot = value;
            }
            Ok(())
        }
    }

    fn firmware_buffer(id: u32, producer: u32, last_consumer: u32) -> crate::BufferPlan {
        crate::BufferPlan {
            id: crate::BufferId(id),
            layout: crate::Layout::Canonical,
            capacity_bytes: 4,
            producer: StepId(producer),
            consumers: vec![StepId(last_consumer)],
            last_consumer: StepId(last_consumer),
            aliases: None,
        }
    }

    #[test]
    fn static_executor_runs_the_firmware_subset_over_caller_owned_arenas() {
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id: GraphId([7; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::McuAot,
            order: vec![NodeId(0), NodeId(1), NodeId(2)],
            nodes: vec![
                firmware_node(0, vec![], vec![crate::BufferId(0)]),
                firmware_node(1, vec![crate::BufferId(0)], vec![crate::BufferId(1)]),
                firmware_node(2, vec![crate::BufferId(1)], vec![]),
            ],
            buffers: vec![firmware_buffer(0, 0, 1), firmware_buffer(1, 1, 2)],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 8,
            persistent_state_bytes: 0,
            session: None,
        };
        plan.plan_id = PlanId(crate::compile::hash_plan(&plan));
        let plan = AuthorizedPlan::new(plan);
        let requirements = plan.mcu_arena_requirements().unwrap();

        let mut values: vec::Vec<Option<u32>> = vec![None; requirements.value_slots];
        let mut terminals: vec::Vec<Option<u32>> = vec![None; requirements.terminal_slots];
        let mut output_scratch: vec::Vec<u32> = vec![0; requirements.max_step_outputs.max(1)];
        let invocation: vec::Vec<Option<u32>> = vec![None; requirements.invocation_slots];
        let mut arenas = super::StaticArenas {
            values: &mut values,
            terminals: &mut terminals,
            output_scratch: &mut output_scratch,
            invocation: &invocation,
        };

        let receipt = super::StaticExecutor::execute(
            &plan,
            &requirements,
            [9; 32],
            &mut arenas,
            &mut CountingKernel,
        )
        .unwrap();

        // source=1 -> process=2 -> sink=3 (terminal), identical to the host
        // reference executor over the same canonical plan.
        assert_eq!(terminals[0], Some(3));
        assert_eq!(receipt.completed_steps, 3);
        assert_eq!(receipt.terminal_values, 1);
        assert_eq!(receipt.graph_id, plan.graph_id);
        assert_eq!(receipt.plan_id, plan.plan_id);
        assert_eq!(receipt.realm, ExecutionRealm::McuAot);
        assert_eq!(receipt.last_step, Some(NodeId(2)));
        // Liveness release: no buffer slot remains occupied after the run.
        assert!(values.iter().all(Option::is_none));
    }

    #[test]
    fn static_executor_rejects_a_host_realm_plan() {
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id: GraphId([7; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::HostStream,
            order: vec![NodeId(0)],
            nodes: vec![firmware_node(0, vec![], vec![])],
            buffers: vec![],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 0,
            persistent_state_bytes: 0,
            session: None,
        };
        plan.plan_id = PlanId(crate::compile::hash_plan(&plan));
        let plan = AuthorizedPlan::new(plan);
        let requirements = McuArenaRequirements {
            byte_arena: 0,
            value_slots: 0,
            invocation_slots: 0,
            max_step_inputs: 0,
            max_step_outputs: 1,
            attempt_slots: 1,
            terminal_slots: 1,
        };
        let mut values: vec::Vec<Option<u32>> = vec![];
        let mut terminals: vec::Vec<Option<u32>> = vec![None];
        let mut output_scratch: vec::Vec<u32> = vec![0];
        let invocation: vec::Vec<Option<u32>> = vec![];
        let mut arenas = super::StaticArenas {
            values: &mut values,
            terminals: &mut terminals,
            output_scratch: &mut output_scratch,
            invocation: &invocation,
        };
        let error = super::StaticExecutor::execute(
            &plan,
            &requirements,
            [0; 32],
            &mut arenas,
            &mut CountingKernel,
        )
        .unwrap_err();
        assert_eq!(
            error,
            super::StaticExecutionError::WrongRealm(ExecutionRealm::HostStream)
        );
    }
}
