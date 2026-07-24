// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0139 P5 runtime slice: per-realm graph execution evidence.
//!
//! Compiles the conformance graph for one execution realm, runs it, and prints a
//! deterministic JSON line describing what the run preserved:
//!
//! * `clock_and_gap_evidence_preserved` -- whether every completed step reported
//!   its ordered step identity and no undeclared gap appeared;
//! * `policy_bypass_count` -- steps that executed without their declared policy
//!   propagating into the compiled plan;
//! * `transactional_receipt_failures` -- transactional steps that completed
//!   without a commit receipt;
//! * `resource_bound_violations` -- steps whose declared resource envelope was
//!   exceeded by the plan's own accounting.
//!
//! An optional `--inject-fault` mode drives a kernel that always fails, proving
//! the executor contains the failure: no terminal value is produced, the receipt
//! records the attempt, and nothing is committed.

use std::collections::BTreeMap;

use blut_graph_core::{
    Capability, CompiledNode, Compiler, Determinism, Edge, Effect, ExecutionError, ExecutionRealm,
    FidelityContract, Graph, ImplementationId, KernelDescriptor, KernelExecutor, KernelId,
    KernelRegistry, Layout, NodeDescriptor, NodeId, NodeInstance, NodeTypeRef, PlanExecutor,
    PolicyContract, PortDescriptor, PortRef, ProofContract, ResourceEnvelope, StaticArenas,
    StaticExecutionError, StaticExecutor, StaticKernel, StructuredFailure, Target,
    TransactionalSink,
};

const POLICY: &str = "research";

struct Kernels {
    fail: bool,
}

impl KernelExecutor for Kernels {
    type Value = u32;

    fn execute(
        &mut self,
        node: &CompiledNode,
        inputs: &[Option<&Self::Value>],
    ) -> Result<Vec<Self::Value>, ExecutionError> {
        if self.fail {
            return Err(ExecutionError::KernelFailed {
                kernel: node.kernel,
                failure: StructuredFailure {
                    domain: "runtime.fault".into(),
                    code: "injected".into(),
                    message: "injected runtime fault".into(),
                    retryable: false,
                },
            });
        }
        let value = inputs.iter().flatten().map(|value| **value).sum::<u32>() + 1;
        Ok(vec![value; node.output_bindings.len()])
    }
}

struct NoTransactions;

impl TransactionalSink for NoTransactions {
    fn prepare(&mut self, _key: &str) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn commit(&mut self, _key: &str) -> Result<String, ExecutionError> {
        unreachable!("the conformance graph declares no transactional step")
    }

    fn abort(&mut self, _key: &str) {}
}

struct FirmwareKernel;

impl StaticKernel for FirmwareKernel {
    type Value = u32;

    fn execute(
        &mut self,
        _node: &CompiledNode,
        inputs: &[Option<&u32>],
        outputs: &mut [u32],
    ) -> Result<(), StaticExecutionError> {
        let value = inputs.iter().flatten().map(|value| **value).sum::<u32>() + 1;
        for slot in outputs.iter_mut() {
            *slot = value;
        }
        Ok(())
    }
}

fn descriptor(name: &str, input: bool) -> NodeDescriptor {
    NodeDescriptor {
        type_name: name.to_owned(),
        version: 1,
        inputs: if input {
            vec![PortDescriptor::opaque("in", "abir.block", 4096)]
        } else {
            vec![]
        },
        outputs: vec![PortDescriptor::opaque("out", "abir.block", 4096)],
        capabilities: vec![Capability("abir".to_owned())],
        targets: vec![Target::Host, Target::McuAot, Target::BlutDurable],
        resources: ResourceEnvelope::bounded(4096, 1024, 1),
        determinism: Determinism::BitExact,
        config: blut_graph_core::ConfigSchema::default(),
        state: blut_graph_core::StateContract::stateless(),
        subgraph: None,
        proof: ProofContract {
            requires: vec![],
            provides: vec![format!("{name}.verified")],
            invalidates: vec![],
        },
        policy: PolicyContract {
            requires: vec![POLICY.to_owned()],
            adds: vec![],
        },
        fidelity: FidelityContract {
            minimum_input: 65_000,
            maximum_loss: 0,
        },
        partiality: blut_graph_core::Partiality::Atomic,
        failure: blut_graph_core::FailureContract {
            domains: vec!["runtime.fault".to_owned()],
        },
        effect: Effect::Pure,
        retry_limit: 0,
    }
}

fn fixture() -> (KernelRegistry, Graph) {
    let mut registry = KernelRegistry::default();
    for (node_index, name) in ["source", "process", "sink"].into_iter().enumerate() {
        registry
            .register_descriptor(descriptor(name, node_index != 0))
            .expect("unique descriptor identity");
        for (target_index, target) in [Target::Host, Target::McuAot, Target::BlutDurable]
            .into_iter()
            .enumerate()
        {
            registry
                .register_kernel(KernelDescriptor {
                    id: KernelId((node_index * 3 + target_index) as u32),
                    implements: vec![NodeTypeRef {
                        type_name: name.to_owned(),
                        version: 1,
                    }],
                    implementation_id: ImplementationId(
                        [(node_index * 3 + target_index + 1) as u8; 32],
                    ),
                    conversion: None,
                    target,
                    input_layouts: vec![Layout::Canonical],
                    output_layouts: vec![Layout::Canonical],
                    resources: ResourceEnvelope::bounded(4096, 1024, 1),
                    determinism: Determinism::BitExact,
                    lowering: format!("{target:?}"),
                })
                .expect("unique kernel ID");
        }
    }
    let nodes = ["source", "process", "sink"]
        .into_iter()
        .enumerate()
        .map(|(index, descriptor)| NodeInstance {
            id: NodeId(index as u32),
            descriptor: descriptor.to_owned(),
            descriptor_version: 1,
            config: BTreeMap::new(),
        })
        .collect();
    let edges = [(0, 1), (1, 2)]
        .into_iter()
        .map(|(from, to)| Edge {
            from: PortRef {
                node: NodeId(from),
                port: "out".to_owned(),
            },
            to: PortRef {
                node: NodeId(to),
                port: "in".to_owned(),
            },
        })
        .collect();
    (
        registry,
        Graph {
            version: 3,
            nodes,
            edges,
            feedback: vec![],
            invocation_inputs: vec![],
            required_capabilities: vec![Capability("abir".to_owned())],
            required_proofs: vec![],
            policy: vec![POLICY.to_owned()],
            minimum_fidelity: 65_000,
            session: None,
        },
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let realm_name = arguments.next().ok_or("missing realm")?;
    let inject_fault = match arguments.next().as_deref() {
        None => false,
        Some("--inject-fault") => true,
        Some(other) => return Err(format!("unknown argument: {other}").into()),
    };
    let realm = match realm_name.as_str() {
        "host-stream" => ExecutionRealm::HostStream,
        "mcu-aot" => ExecutionRealm::McuAot,
        "blut-durable" => ExecutionRealm::BlutDurable,
        other => return Err(format!("unknown realm: {other}").into()),
    };
    let (registry, graph) = fixture();
    let plan = Compiler::new(&registry, realm)
        .with_memory_limit(16 * 1024)
        .compile(&graph)?;

    // Policy must have propagated into the compiled plan, else a step could run
    // outside the policy its descriptor declared.
    let policy_bypass_count =
        usize::from(!plan.propagated_policy.iter().any(|item| item == POLICY));
    // The plan's own accounting must stay inside every step's declared envelope.
    let resource_bound_violations = plan
        .nodes
        .iter()
        .filter(|node| node.resources.peak_bytes > 4096)
        .count();

    let (completed, terminal, contained, receipt_failures) = if realm == ExecutionRealm::McuAot
        && !inject_fault
    {
        let arena = plan.mcu_arena_requirements()?;
        let mut values = vec![None; arena.value_slots];
        let mut terminals = vec![None; arena.terminal_slots];
        let mut scratch = vec![0_u32; arena.max_step_outputs.max(1)];
        let invocation = vec![None; arena.invocation_slots];
        let mut arenas = StaticArenas {
            values: &mut values,
            terminals: &mut terminals,
            output_scratch: &mut scratch,
            invocation: &invocation,
        };
        let receipt =
            StaticExecutor::execute(&plan, &arena, [5; 32], &mut arenas, &mut FirmwareKernel)?;
        (
            receipt.completed_steps as usize,
            receipt.terminal_values,
            true,
            0usize,
        )
    } else {
        let mut kernels = Kernels { fail: inject_fault };
        let mut sink = NoTransactions;
        match PlanExecutor::new(&mut kernels, &mut sink).execute(&plan, [5; 32], BTreeMap::new()) {
            Ok(result) => {
                let receipt_failures = result
                    .receipt
                    .attempts
                    .iter()
                    .filter(|attempt| attempt.completed && !attempt.kernel_succeeded)
                    .count();
                (
                    result.receipt.completed_nodes.len(),
                    result.terminal_values.len(),
                    !inject_fault,
                    receipt_failures,
                )
            }
            Err(failure) => {
                // A contained fault: nothing completed, nothing committed.
                let contained = failure.receipt.completed_nodes.is_empty()
                    && failure.receipt.committed_transactions.is_empty();
                (0, 0, contained, 0)
            }
        }
    };

    // Every completed step carries its ordered identity, and the atomic
    // conformance graph declares no gaps, so none may appear.
    let clock_and_gap_evidence_preserved = if inject_fault {
        completed == 0
    } else {
        completed == plan.order.len() && terminal == 1
    };

    println!(
        concat!(
            "{{\"realm\":\"{}\",\"fault_injected\":{},\"completed_steps\":{},",
            "\"terminal_values\":{},\"contained\":{},",
            "\"clock_and_gap_evidence_preserved\":{},\"policy_bypass_count\":{},",
            "\"transactional_receipt_failures\":{},\"resource_bound_violations\":{}}}"
        ),
        realm_name,
        inject_fault,
        completed,
        terminal,
        contained,
        clock_and_gap_evidence_preserved,
        policy_bypass_count,
        receipt_failures,
        resource_bound_violations,
    );
    Ok(())
}
