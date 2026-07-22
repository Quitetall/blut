// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use blut_graph_core::{
    Capability, CompiledNode, Compiler, Determinism, Edge, Effect, ExecutionError, ExecutionRealm,
    FidelityContract, Graph, ImplementationId, KernelDescriptor, KernelExecutor, KernelId,
    KernelRegistry, Layout, NodeDescriptor, NodeId, NodeInstance, NodeTypeRef, PlanExecutor,
    PlanLimits, PolicyContract, PortDescriptor, PortRef, ProofContract, ResourceEnvelope, Target,
    TransactionalSink,
};

const ITERATIONS: usize = 10_000;

struct SyntheticKernels;

impl KernelExecutor for SyntheticKernels {
    type Value = u32;

    fn execute(
        &mut self,
        node: &CompiledNode,
        inputs: &[Option<&Self::Value>],
    ) -> Result<Vec<Self::Value>, ExecutionError> {
        let value = inputs
            .first()
            .and_then(|value| *value)
            .copied()
            .unwrap_or_default()
            + 1;
        Ok(vec![value; node.output_bindings.len()])
    }
}

struct NoTransactions;

impl TransactionalSink for NoTransactions {
    fn prepare(&mut self, _idempotency_key: &str) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn commit(&mut self, _idempotency_key: &str) -> Result<String, ExecutionError> {
        unreachable!("the evidence graph contains no transactional nodes")
    }

    fn abort(&mut self, _idempotency_key: &str) {}
}

fn descriptor(name: &str, input: bool) -> NodeDescriptor {
    NodeDescriptor {
        type_name: name.to_owned(),
        version: 1,
        inputs: if input {
            vec![PortDescriptor {
                name: "in".to_owned(),
                semantic_type: "abir.block".to_owned(),
                optional: false,
                layouts: vec![Layout::Canonical],
                max_bytes: 4096,
                ..PortDescriptor::opaque("in", "abir.block", 4096)
            }]
        } else {
            vec![]
        },
        outputs: vec![PortDescriptor {
            name: "out".to_owned(),
            semantic_type: "abir.block".to_owned(),
            optional: false,
            layouts: vec![Layout::Canonical],
            max_bytes: 4096,
            ..PortDescriptor::opaque("out", "abir.block", 4096)
        }],
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
            requires: vec!["research".to_owned()],
            adds: vec![],
        },
        fidelity: FidelityContract {
            minimum_input: 65_000,
            maximum_loss: 0,
        },
        partiality: blut_graph_core::Partiality::Atomic,
        failure: blut_graph_core::FailureContract { domains: vec![] },
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
            policy: vec!["research".to_owned()],
            minimum_fidelity: 65_000,
            session: None,
        },
    )
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("String writes cannot fail");
    }
    output
}

fn rate(iterations: usize, elapsed: std::time::Duration) -> f64 {
    iterations as f64 / elapsed.as_secs_f64()
}

fn main() {
    let mut args = env::args().skip(1);
    let mut output = None;
    let mut revision = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().expect("--output requires a path"),
                ));
            }
            "--revision" => {
                revision = Some(args.next().expect("--revision requires a value"));
            }
            _ => panic!("unknown argument: {arg}"),
        }
    }
    let output = output.expect("--output is required");
    let revision = revision.expect("--revision is required");
    let (registry, graph) = fixture();

    let started = Instant::now();
    let mut plan = None;
    for _ in 0..ITERATIONS {
        plan = Some(
            Compiler::new(&registry, ExecutionRealm::HostStream)
                .with_memory_limit(16 * 1024)
                .compile(&graph)
                .expect("fixture compiles"),
        );
    }
    let compile_ops_s = rate(ITERATIONS, started.elapsed());
    let plan = plan.expect("iterations are non-zero");
    let mcu_plan = Compiler::new(&registry, ExecutionRealm::McuAot)
        .with_memory_limit(16 * 1024)
        .compile(&graph)
        .expect("MCU fixture compiles");
    let durable_plan = Compiler::new(&registry, ExecutionRealm::BlutDurable)
        .with_memory_limit(16 * 1024)
        .compile(&graph)
        .expect("durable fixture compiles");
    assert_eq!(plan.graph_id, mcu_plan.graph_id);
    assert_eq!(plan.graph_id, durable_plan.graph_id);
    let mcu_arena = mcu_plan
        .mcu_arena_requirements()
        .expect("MCU plan exposes a fixed-arena contract");
    for realm_plan in [&mcu_plan, &plan, &durable_plan] {
        let mut kernels = SyntheticKernels;
        let mut transactions = NoTransactions;
        let roots = BTreeMap::new();
        let result = PlanExecutor::new(&mut kernels, &mut transactions)
            .execute(realm_plan, [1; 32], roots)
            .expect("synthetic realm execution succeeds");
        assert_eq!(result.terminal_values[&NodeId(2)], vec![3]);
        assert_eq!(result.receipt.graph_id, plan.graph_id);
        assert_eq!(result.receipt.plan_id, realm_plan.plan_id);
        assert_eq!(result.receipt.realm, realm_plan.realm);
        assert_eq!(result.receipt.completed_nodes, realm_plan.order);
        assert!(
            result
                .receipt
                .attempts
                .iter()
                .all(|attempt| attempt.attempts == 1)
        );
    }

    let started = Instant::now();
    let mut bytes = Vec::new();
    for _ in 0..ITERATIONS {
        bytes = plan.to_aot_bytes().expect("fixture encodes");
    }
    let encode_ops_s = rate(ITERATIONS, started.elapsed());

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let decoded = blut_graph_core::CompiledPlan::from_aot_bytes(&bytes, PlanLimits::default())
            .expect("fixture decodes");
        std::hint::black_box(decoded);
    }
    let decode_ops_s = rate(ITERATIONS, started.elapsed());

    let evidence = serde_json::json!({
        "schema": "blut.graph-runtime-evidence/v1",
        "stage": "graph-runtime",
        "status": "FAIL",
        "completion_eligible": false,
        "revision": revision,
        "iterations": ITERATIONS,
        "compiled_plan_bytes": bytes.len(),
        "wire_magic": "BGP3",
        "schema_version": plan.schema_version,
        "graph_id": hex(&plan.graph_id.0),
        "plan_id": hex(&plan.plan_id.0),
        "peak_bytes": plan.peak_bytes,
        "persistent_state_bytes": plan.persistent_state_bytes,
        "compile_ops_s": compile_ops_s,
        "encode_ops_s": encode_ops_s,
        "decode_ops_s": decode_ops_s,
        "compile_benchmark_realm": "host-stream",
        "identity_checked_realms": ["mcu-aot", "host-stream", "blut-durable"],
        "synthetic_executor_checked_realms": ["mcu-aot", "host-stream", "blut-durable"],
        "mcu_fixed_arena_bytes": mcu_arena.byte_arena,
        "realm_implementation_blockers": [
            "MCU has an authorized fixed-arena sizing contract but no distinct static executor evidence",
            "host-stream execution evidence still uses the synthetic generic executor",
            "BLUT has a fail-closed adapter contract but no end-to-end durable execution receipt in this artifact",
            "stateful session and feedback realm-store execution evidence is absent",
            "hierarchical schemas are identity-bound but inner DAGs are not yet inline-expanded",
            "checkpoint bounds are declarative; explicit runtime barrier evidence is absent",
            "BPC2 supervised process-plugin lifecycle implementation evidence is absent"
        ],
        "durable_adapter_validation": "cargo test -p blut semantic_plan::tests --lib",
        "synthetic_output": 3
    });
    fs::write(
        output,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&evidence).expect("evidence serializes")
        ),
    )
    .expect("evidence writes");
}
