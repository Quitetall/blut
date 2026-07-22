// SPDX-License-Identifier: AGPL-3.0-or-later
//! Allocation-free execution sizing for statically linked MCU executors.
//!
//! Graph compilation and AOT authorization may allocate on a host. Once an
//! authorized MCU plan is installed, firmware uses these exact requirements to
//! provision caller-owned arenas; the execution loop never needs to grow a
//! collection or discover an undeclared bound.

use core::fmt;

use crate::{AuthorizedPlan, Effect, ExecutionRealm, Partiality};

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
            schema_version: 2,
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
                input_bindings: vec![],
                output_bindings: vec![OutputBinding::Terminal],
                partiality: Partiality::Atomic,
                failure: FailureContract { domains: vec![] },
                effect: Effect::Pure,
                retry_limit: 0,
                checkpointable: false,
            }],
            buffers: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 128,
        };
        plan.plan_id = PlanId(crate::compile::hash_plan(&plan));
        let requirements = AuthorizedPlan::new(plan).mcu_arena_requirements().unwrap();
        assert_eq!(requirements.byte_arena, 128);
        assert_eq!(requirements.attempt_slots, 1);
        assert_eq!(requirements.terminal_slots, 1);
    }
}
