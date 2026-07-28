// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::model::{
    AuthorizedPlan, BufferId, CompiledNode, CompiledPlan, Effect, ExecutionRealm, GraphId,
    ImplementationId, InputBinding, KernelId, NodeId, OutputBinding, PlanId, StepId,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    UnknownKernel(KernelId),
    MissingBuffer(BufferId),
    MissingInvocation(crate::PortRef),
    UnexpectedInvocation(crate::PortRef),
    OutputArity {
        kernel: KernelId,
        expected: usize,
        actual: usize,
    },
    KernelFailed {
        kernel: KernelId,
        failure: StructuredFailure,
    },
    UndeclaredFailure(KernelId, String),
    UnsafeRetry(KernelId),
    TransactionPrepare(String),
    TransactionCommit(String),
    InvalidGap(KernelId),
    StatefulPlanUnsupported,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailureEvidence {
    pub semantic_type: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredFailure {
    pub domain: String,
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub evidence: Vec<FailureEvidence>,
}

impl ExecutionError {
    const fn retryable(&self) -> bool {
        matches!(
            self,
            Self::KernelFailed {
                failure: StructuredFailure {
                    retryable: true,
                    ..
                },
                ..
            }
        )
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ExecutionError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionAttempt {
    pub step: StepId,
    pub semantic_nodes: Vec<NodeId>,
    pub kernel: KernelId,
    pub implementation_id: ImplementationId,
    pub attempts: u32,
    pub kernel_succeeded: bool,
    pub completed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionReceipt {
    pub invocation_id: [u8; 32],
    pub graph_id: GraphId,
    pub plan_id: PlanId,
    pub realm: ExecutionRealm,
    pub completed_nodes: Vec<NodeId>,
    pub attempts: Vec<ExecutionAttempt>,
    pub committed_transactions: Vec<String>,
    pub gaps: Vec<GapReceipt>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelGap {
    pub output_index: u32,
    pub offset: u64,
    /// Known missing extent. `None` preserves unknown cardinality.
    pub length: Option<u64>,
    pub domain: String,
    pub code: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GapReceipt {
    pub step: StepId,
    pub semantic_nodes: Vec<NodeId>,
    pub gap: KernelGap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelExecution<V> {
    pub outputs: Vec<V>,
    pub gaps: Vec<KernelGap>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionFailure {
    pub error: ExecutionError,
    pub receipt: ExecutionReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionResult<V> {
    pub terminal_values: BTreeMap<NodeId, Vec<V>>,
    pub receipt: ExecutionReceipt,
}

pub trait KernelExecutor {
    type Value;

    /// Execute one attempt from immutable inputs and return one value per
    /// physical output buffer. When a node has no connected output buffer,
    /// returned values are terminal invocation outputs.
    fn execute(
        &mut self,
        node: &CompiledNode,
        inputs: &[Option<&Self::Value>],
    ) -> Result<Vec<Self::Value>, ExecutionError>;

    /// Gap-aware execution hook. Existing atomic kernels implement only
    /// `execute`; explicitly partial kernels override this method and return
    /// structured gaps alongside their ordinary output records.
    fn execute_with_gaps(
        &mut self,
        node: &CompiledNode,
        inputs: &[Option<&Self::Value>],
    ) -> Result<KernelExecution<Self::Value>, ExecutionError> {
        self.execute(node, inputs).map(|outputs| KernelExecution {
            outputs,
            gaps: Vec::new(),
        })
    }
}

pub trait TransactionalSink {
    fn prepare(&mut self, idempotency_key: &str) -> Result<(), ExecutionError>;
    fn commit(&mut self, idempotency_key: &str) -> Result<String, ExecutionError>;
    fn abort(&mut self, idempotency_key: &str);
}

pub struct PlanExecutor<'a, K, S> {
    kernels: &'a mut K,
    sink: &'a mut S,
}

impl<'a, K, S> PlanExecutor<'a, K, S>
where
    K: KernelExecutor,
    S: TransactionalSink,
{
    pub fn new(kernels: &'a mut K, sink: &'a mut S) -> Self {
        Self { kernels, sink }
    }

    pub fn execute(
        &mut self,
        authorized: &AuthorizedPlan,
        invocation_id: [u8; 32],
        invocation_inputs: BTreeMap<crate::PortRef, K::Value>,
    ) -> Result<ExecutionResult<K::Value>, Box<ExecutionFailure>> {
        let plan = authorized.as_plan();
        let mut receipt = ExecutionReceipt {
            invocation_id,
            graph_id: plan.graph_id,
            plan_id: plan.plan_id,
            realm: plan.realm,
            completed_nodes: Vec::new(),
            attempts: Vec::new(),
            committed_transactions: Vec::new(),
            gaps: Vec::new(),
        };
        let mut buffers: BTreeMap<BufferId, K::Value> = BTreeMap::new();
        let mut terminal_values = BTreeMap::new();

        if !plan.feedback.is_empty()
            || plan
                .nodes
                .iter()
                .any(|node| node.state.scope != crate::StateScope::Stateless)
        {
            return Err(Box::new(ExecutionFailure {
                error: ExecutionError::StatefulPlanUnsupported,
                receipt,
            }));
        }

        if let Some(unexpected) = invocation_inputs
            .keys()
            .find(|port| plan.invocation_ports.binary_search(port).is_err())
        {
            return Err(Box::new(ExecutionFailure {
                error: ExecutionError::UnexpectedInvocation(unexpected.clone()),
                receipt,
            }));
        }
        if let Some(missing) = plan
            .invocation_ports
            .iter()
            .find(|port| !invocation_inputs.contains_key(*port))
        {
            return Err(Box::new(ExecutionFailure {
                error: ExecutionError::MissingInvocation(missing.clone()),
                receipt,
            }));
        }

        for node in &plan.nodes {
            let key = idempotency_key(plan, &invocation_id, node.id, node.implementation_id);
            let attempt_index = receipt.attempts.len();
            receipt.attempts.push(ExecutionAttempt {
                step: node.id,
                semantic_nodes: node.semantic_nodes.clone(),
                kernel: node.kernel,
                implementation_id: node.implementation_id,
                attempts: 0,
                kernel_succeeded: false,
                completed: false,
            });
            if node.effect == Effect::Transactional
                && let Err(error) = self.sink.prepare(&key)
            {
                return Err(Box::new(ExecutionFailure { error, receipt }));
            }

            let mut input_values = Vec::with_capacity(node.input_bindings.len());
            for binding in &node.input_bindings {
                match binding {
                    InputBinding::Absent => input_values.push(None),
                    InputBinding::Invocation(invocation) => {
                        let port = &plan.invocation_ports[*invocation as usize];
                        match invocation_inputs.get(port) {
                            Some(value) => input_values.push(Some(value)),
                            None => {
                                if node.effect == Effect::Transactional {
                                    self.sink.abort(&key);
                                }
                                return Err(Box::new(ExecutionFailure {
                                    error: ExecutionError::MissingInvocation(port.clone()),
                                    receipt,
                                }));
                            }
                        }
                    }
                    InputBinding::Buffer(buffer) => match buffers.get(buffer) {
                        Some(value) => input_values.push(Some(value)),
                        None => {
                            if node.effect == Effect::Transactional {
                                self.sink.abort(&key);
                            }
                            return Err(Box::new(ExecutionFailure {
                                error: ExecutionError::MissingBuffer(*buffer),
                                receipt,
                            }));
                        }
                    },
                    InputBinding::Feedback(_) => {
                        unreachable!("stateful plans fail before execution")
                    }
                }
            }

            let outputs = loop {
                receipt.attempts[attempt_index].attempts += 1;
                match self.kernels.execute_with_gaps(node, &input_values) {
                    Ok(outputs) => break outputs,
                    Err(error) => {
                        if let ExecutionError::KernelFailed { failure, .. } = &error
                            && !node.failure.domains.contains(&failure.domain)
                        {
                            if node.effect == Effect::Transactional {
                                self.sink.abort(&key);
                            }
                            return Err(Box::new(ExecutionFailure {
                                error: ExecutionError::UndeclaredFailure(
                                    node.kernel,
                                    failure.domain.clone(),
                                ),
                                receipt,
                            }));
                        }
                        let failures = receipt.attempts[attempt_index].attempts - 1;
                        if failures >= u32::from(node.retry_limit) || !error.retryable() {
                            if node.effect == Effect::Transactional {
                                self.sink.abort(&key);
                            }
                            return Err(Box::new(ExecutionFailure { error, receipt }));
                        }
                        if !matches!(
                            node.effect,
                            Effect::Pure
                                | Effect::Idempotent
                                | Effect::Transactional
                                | Effect::AtLeastOnce
                        ) {
                            return Err(Box::new(ExecutionFailure {
                                error: ExecutionError::UnsafeRetry(node.kernel),
                                receipt,
                            }));
                        }
                    }
                }
            };

            if outputs.outputs.len() != node.output_bindings.len() {
                if node.effect == Effect::Transactional {
                    self.sink.abort(&key);
                }
                return Err(Box::new(ExecutionFailure {
                    error: ExecutionError::OutputArity {
                        kernel: node.kernel,
                        expected: node.output_bindings.len(),
                        actual: outputs.outputs.len(),
                    },
                    receipt,
                }));
            }
            if outputs.gaps.iter().any(|gap| {
                node.partiality != crate::Partiality::ExplicitGaps
                    || gap.output_index as usize >= outputs.outputs.len()
                    || gap.length == Some(0)
                    || !node.failure.domains.contains(&gap.domain)
            }) {
                if node.effect == Effect::Transactional {
                    self.sink.abort(&key);
                }
                return Err(Box::new(ExecutionFailure {
                    error: ExecutionError::InvalidGap(node.kernel),
                    receipt,
                }));
            }
            receipt
                .gaps
                .extend(outputs.gaps.into_iter().map(|gap| GapReceipt {
                    step: node.id,
                    semantic_nodes: node.semantic_nodes.clone(),
                    gap,
                }));
            let mut terminals = Vec::new();
            for (binding, output) in node.output_bindings.iter().copied().zip(outputs.outputs) {
                match binding {
                    OutputBinding::Buffer(buffer) => {
                        buffers.insert(buffer, output);
                    }
                    OutputBinding::Terminal => terminals.push(output),
                }
            }
            if !terminals.is_empty()
                && let Some(terminal) = node.semantic_nodes.last()
            {
                terminal_values.insert(*terminal, terminals);
            }

            receipt.attempts[attempt_index].kernel_succeeded = true;
            if node.effect == Effect::Transactional {
                match self.sink.commit(&key) {
                    Ok(transaction) => receipt.committed_transactions.push(transaction),
                    Err(error) => return Err(Box::new(ExecutionFailure { error, receipt })),
                }
            }
            receipt.attempts[attempt_index].completed = true;
            receipt
                .completed_nodes
                .extend(node.semantic_nodes.iter().copied());

            for buffer in node
                .input_bindings
                .iter()
                .filter_map(|binding| match binding {
                    InputBinding::Buffer(buffer) => Some(buffer),
                    InputBinding::Invocation(_)
                    | InputBinding::Feedback(_)
                    | InputBinding::Absent => None,
                })
            {
                if plan
                    .buffers
                    .get(buffer.0 as usize)
                    .is_some_and(|buffer_plan| node.id == buffer_plan.last_consumer)
                {
                    buffers.remove(buffer);
                }
            }
        }

        Ok(ExecutionResult {
            terminal_values,
            receipt,
        })
    }
}

fn idempotency_key(
    plan: &CompiledPlan,
    invocation_id: &[u8; 32],
    step: StepId,
    implementation_id: ImplementationId,
) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("blut.transaction.v2");
    hasher.update(&plan.plan_id.0);
    hasher.update(invocation_id);
    hasher.update(&step.0.to_le_bytes());
    hasher.update(&implementation_id.0);
    hasher.finalize().to_hex().as_str().to_string()
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;

    use super::*;
    use crate::model::{
        CompiledNode, CompiledPlan, Determinism, ExecutionRealm, GraphId, ImplementationId,
        KernelId, NodeId, PlanId, ResourceEnvelope,
    };

    struct Kernels {
        failures_remaining: u32,
    }

    impl KernelExecutor for Kernels {
        type Value = u32;

        fn execute(
            &mut self,
            node: &CompiledNode,
            inputs: &[Option<&Self::Value>],
        ) -> Result<Vec<Self::Value>, ExecutionError> {
            if self.failures_remaining > 0 {
                self.failures_remaining -= 1;
                return Err(ExecutionError::KernelFailed {
                    kernel: node.kernel,
                    failure: StructuredFailure {
                        domain: "test.kernel".into(),
                        code: "retry".into(),
                        message: "retry".into(),
                        retryable: true,
                        evidence: Vec::new(),
                    },
                });
            }
            let value = inputs.iter().flatten().map(|value| **value).sum::<u32>() + 1;
            Ok(vec![value; node.output_bindings.len()])
        }
    }

    #[derive(Default)]
    struct Sink {
        prepared: Vec<String>,
        committed: Vec<String>,
        fail_commit: bool,
    }

    impl TransactionalSink for Sink {
        fn prepare(&mut self, key: &str) -> Result<(), ExecutionError> {
            self.prepared.push(key.into());
            Ok(())
        }

        fn commit(&mut self, key: &str) -> Result<String, ExecutionError> {
            if self.fail_commit {
                return Err(ExecutionError::TransactionCommit("injected".into()));
            }
            self.committed.push(key.into());
            Ok(key.into())
        }

        fn abort(&mut self, _key: &str) {}
    }

    fn node(
        id: u32,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        effect: Effect,
    ) -> CompiledNode {
        let input_count = inputs.len();
        let output_count = outputs.len().max(1);
        CompiledNode {
            id: StepId(id),
            semantic_nodes: vec![NodeId(id)],
            semantic_types: vec![crate::NodeTypeRef {
                type_name: "test".into(),
                version: 1,
            }],
            semantic_configs: vec![BTreeMap::new()],
            kernel: KernelId(id),
            implementation_id: ImplementationId([id as u8 + 1; 32]),
            resources: ResourceEnvelope::bounded(0, 0, 1),
            determinism: Determinism::BitExact,
            lowering: "test".into(),
            conversion: None,
            input_ports: (0..inputs.len())
                .map(|index| format!("in-{index}"))
                .collect(),
            output_ports: if outputs.is_empty() {
                vec!["out".into()]
            } else {
                (0..outputs.len())
                    .map(|index| format!("out-{index}"))
                    .collect()
            },
            input_contracts: (0..input_count)
                .map(|index| {
                    crate::CompiledPortContract::opaque(
                        format!("in-{index}"),
                        "test",
                        crate::Layout::Canonical,
                        4,
                    )
                })
                .collect(),
            output_contracts: (0..output_count)
                .map(|index| {
                    crate::CompiledPortContract::opaque(
                        if output_count == 1 {
                            "out".into()
                        } else {
                            format!("out-{index}")
                        },
                        "test",
                        crate::Layout::Canonical,
                        4,
                    )
                })
                .collect(),
            input_bindings: inputs.into_iter().map(InputBinding::Buffer).collect(),
            output_bindings: if outputs.is_empty() {
                vec![OutputBinding::Terminal]
            } else {
                outputs.into_iter().map(OutputBinding::Buffer).collect()
            },
            partiality: crate::Partiality::Atomic,
            failure: crate::FailureContract {
                domains: vec!["test.kernel".into()],
            },
            effect,
            retry_limit: 0,
            state: crate::StateContract::stateless(),
            subgraph_path: vec![],
        }
    }

    #[test]
    fn fanout_and_join_follow_buffers_not_node_iteration() {
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([2; 32]),
            realm: ExecutionRealm::HostStream,
            order: vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3)],
            nodes: vec![
                node(0, vec![], vec![BufferId(0)], Effect::Pure),
                node(1, vec![BufferId(0)], vec![BufferId(1)], Effect::Pure),
                node(2, vec![BufferId(0)], vec![BufferId(2)], Effect::Pure),
                node(3, vec![BufferId(1), BufferId(2)], vec![], Effect::Pure),
            ],
            buffers: vec![
                crate::BufferPlan {
                    id: BufferId(0),
                    layout: crate::Layout::Canonical,
                    capacity_bytes: 4,
                    producer: StepId(0),
                    consumers: vec![StepId(1), StepId(2)],
                    last_consumer: StepId(2),
                    aliases: None,
                },
                crate::BufferPlan {
                    id: BufferId(1),
                    layout: crate::Layout::Canonical,
                    capacity_bytes: 4,
                    producer: StepId(1),
                    consumers: vec![StepId(3)],
                    last_consumer: StepId(3),
                    aliases: None,
                },
                crate::BufferPlan {
                    id: BufferId(2),
                    layout: crate::Layout::Canonical,
                    capacity_bytes: 4,
                    producer: StepId(2),
                    consumers: vec![StepId(3)],
                    last_consumer: StepId(3),
                    aliases: None,
                },
            ],
            feedback: vec![],
            invocation_ports: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 12,
            persistent_state_bytes: 0,
            session: None,
        };
        plan.plan_id = PlanId(crate::compile::hash_plan(&plan));
        let plan = AuthorizedPlan::new(plan);
        let roots = BTreeMap::new();
        let mut kernels = Kernels {
            failures_remaining: 0,
        };
        let mut sink = Sink::default();
        let result = PlanExecutor::new(&mut kernels, &mut sink)
            .execute(&plan, [9; 32], roots)
            .unwrap();
        assert_eq!(result.terminal_values[&NodeId(3)], vec![5]);
        assert_eq!(result.receipt.completed_nodes, plan.order);
    }

    #[test]
    fn failure_returns_attempt_receipt_and_invocations_have_distinct_keys() {
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([2; 32]),
            realm: ExecutionRealm::BlutDurable,
            order: vec![NodeId(0)],
            nodes: vec![node(0, vec![], vec![], Effect::Transactional)],
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
        plan.nodes[0].retry_limit = 1;
        plan.plan_id = PlanId(crate::compile::hash_plan(&plan));
        let plan = AuthorizedPlan::new(plan);
        let roots = BTreeMap::new();
        let mut kernels = Kernels {
            failures_remaining: 2,
        };
        let mut sink = Sink::default();
        let failure = PlanExecutor::new(&mut kernels, &mut sink)
            .execute(&plan, [1; 32], roots)
            .unwrap_err();
        assert_eq!(failure.receipt.attempts[0].attempts, 2);
        assert!(!failure.receipt.attempts[0].completed);

        let key_a = idempotency_key(&plan, &[1; 32], StepId(0), plan.nodes[0].implementation_id);
        let key_b = idempotency_key(&plan, &[2; 32], StepId(0), plan.nodes[0].implementation_id);
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn commit_failure_returns_the_completed_attempt_without_a_commit_receipt() {
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([2; 32]),
            realm: ExecutionRealm::BlutDurable,
            order: vec![NodeId(0)],
            nodes: vec![node(0, vec![], vec![], Effect::Transactional)],
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
        let roots = BTreeMap::new();
        let mut kernels = Kernels {
            failures_remaining: 0,
        };
        let mut sink = Sink {
            fail_commit: true,
            ..Sink::default()
        };
        let failure = PlanExecutor::new(&mut kernels, &mut sink)
            .execute(&plan, [3; 32], roots)
            .unwrap_err();
        assert!(matches!(
            failure.error,
            ExecutionError::TransactionCommit(_)
        ));
        assert!(failure.receipt.completed_nodes.is_empty());
        assert!(failure.receipt.attempts[0].kernel_succeeded);
        assert!(!failure.receipt.attempts[0].completed);
        assert!(failure.receipt.committed_transactions.is_empty());
    }

    #[test]
    fn explicit_partial_node_emits_a_structured_gap_receipt() {
        struct GapKernels;
        impl KernelExecutor for GapKernels {
            type Value = u32;

            fn execute(
                &mut self,
                _node: &CompiledNode,
                _inputs: &[Option<&Self::Value>],
            ) -> Result<Vec<Self::Value>, ExecutionError> {
                unreachable!("gap-aware hook is used")
            }

            fn execute_with_gaps(
                &mut self,
                _node: &CompiledNode,
                _inputs: &[Option<&Self::Value>],
            ) -> Result<KernelExecution<Self::Value>, ExecutionError> {
                Ok(KernelExecution {
                    outputs: vec![7],
                    gaps: vec![KernelGap {
                        output_index: 0,
                        offset: 12,
                        length: Some(4),
                        domain: "biosignal.missing".into(),
                        code: "packet-loss".into(),
                    }],
                })
            }
        }

        let mut partial = node(0, vec![], vec![], Effect::Pure);
        partial.partiality = crate::Partiality::ExplicitGaps;
        partial.failure.domains = vec!["biosignal.missing".into()];
        let mut plan = CompiledPlan {
            schema_version: 3,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([0; 32]),
            realm: ExecutionRealm::HostStream,
            order: vec![NodeId(0)],
            nodes: vec![partial],
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
        let mut kernels = GapKernels;
        let mut sink = Sink::default();
        let result = PlanExecutor::new(&mut kernels, &mut sink)
            .execute(&plan, [8; 32], BTreeMap::new())
            .unwrap();
        assert_eq!(result.receipt.gaps.len(), 1);
        assert_eq!(result.receipt.gaps[0].gap.code, "packet-loss");
        assert_eq!(result.terminal_values[&NodeId(0)], vec![7]);
    }
}
