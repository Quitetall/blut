// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::model::{CompiledNode, CompiledPlan, Effect, KernelId, PlanId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    UnknownKernel(KernelId),
    KernelFailed { kernel: KernelId, message: String },
    UnsafeRetry(KernelId),
    TransactionPrepare(String),
    TransactionCommit(String),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ExecutionError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionReceipt {
    pub plan_id: PlanId,
    pub completed_kernels: Vec<KernelId>,
    pub committed_transactions: Vec<String>,
}

pub trait KernelExecutor {
    type Value;

    fn execute(
        &mut self,
        node: &CompiledNode,
        value: Option<Self::Value>,
    ) -> Result<Option<Self::Value>, ExecutionError>;
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
    K::Value: Clone,
    S: TransactionalSink,
{
    pub fn new(kernels: &'a mut K, sink: &'a mut S) -> Self {
        Self { kernels, sink }
    }

    pub fn execute(
        &mut self,
        plan: &CompiledPlan,
        mut value: Option<K::Value>,
    ) -> Result<(Option<K::Value>, ExecutionReceipt), ExecutionError> {
        let mut receipt = ExecutionReceipt {
            plan_id: plan.plan_id,
            completed_kernels: Vec::new(),
            committed_transactions: Vec::new(),
        };
        for (index, node) in plan.nodes.iter().enumerate() {
            let key = idempotency_key(plan, index, node.kernel);
            if node.effect == Effect::Transactional {
                self.sink.prepare(&key)?;
            }
            let mut attempts = 0u16;
            loop {
                match self.kernels.execute(node, value.clone()) {
                    Ok(next) => {
                        value = next;
                        receipt.completed_kernels.push(node.kernel);
                        break;
                    }
                    Err(error) => {
                        if attempts >= node.retry_limit {
                            if node.effect == Effect::Transactional {
                                self.sink.abort(&key);
                            }
                            return Err(error);
                        }
                        if !matches!(
                            node.effect,
                            Effect::Pure | Effect::Idempotent | Effect::Transactional
                        ) {
                            return Err(ExecutionError::UnsafeRetry(node.kernel));
                        }
                        attempts += 1;
                    }
                }
            }
            if node.effect == Effect::Transactional {
                receipt.committed_transactions.push(self.sink.commit(&key)?);
            }
        }
        Ok((value, receipt))
    }
}

fn idempotency_key(plan: &CompiledPlan, index: usize, kernel: KernelId) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("blut.transaction.v1");
    hasher.update(&plan.plan_id.0);
    hasher.update(&(index as u64).to_le_bytes());
    hasher.update(&kernel.0.to_le_bytes());
    hasher.finalize().to_hex().as_str().to_string()
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;

    use super::*;
    use crate::model::{
        CompiledNode, CompiledPlan, ExecutionRealm, GraphId, KernelId, NodeId, PlanId,
    };

    struct Kernels {
        calls: usize,
    }

    impl KernelExecutor for Kernels {
        type Value = u32;

        fn execute(
            &mut self,
            _node: &CompiledNode,
            value: Option<Self::Value>,
        ) -> Result<Option<Self::Value>, ExecutionError> {
            self.calls += 1;
            Ok(Some(value.unwrap_or_default() + 1))
        }
    }

    #[derive(Default)]
    struct Sink {
        prepared: Vec<String>,
        committed: Vec<String>,
    }

    impl TransactionalSink for Sink {
        fn prepare(&mut self, key: &str) -> Result<(), ExecutionError> {
            self.prepared.push(key.into());
            Ok(())
        }

        fn commit(&mut self, key: &str) -> Result<String, ExecutionError> {
            self.committed.push(key.into());
            Ok(key.into())
        }

        fn abort(&mut self, _key: &str) {}
    }

    #[test]
    fn transaction_is_prepared_and_receipted() {
        let plan = CompiledPlan {
            schema_version: 1,
            graph_id: GraphId([1; 32]),
            plan_id: PlanId([2; 32]),
            realm: ExecutionRealm::HostStream,
            order: vec![NodeId(0)],
            nodes: vec![CompiledNode {
                semantic_nodes: vec![NodeId(0)],
                kernel: KernelId(7),
                input_buffers: vec![],
                output_buffers: vec![],
                effect: Effect::Transactional,
                retry_limit: 0,
                checkpointable: true,
            }],
            buffers: vec![],
            propagated_proofs: vec![],
            propagated_policy: vec![],
            resulting_fidelity: u16::MAX,
            peak_bytes: 0,
        };
        let mut kernels = Kernels { calls: 0 };
        let mut sink = Sink::default();
        let (value, receipt) = PlanExecutor::new(&mut kernels, &mut sink)
            .execute(&plan, None)
            .unwrap();
        assert_eq!(value, Some(1));
        assert_eq!(sink.prepared, sink.committed);
        assert_eq!(receipt.committed_transactions, sink.committed);
    }
}
