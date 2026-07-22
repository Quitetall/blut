// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use blut_graph_core::{
    Capability, Compiler, Determinism, Edge, Effect, ExecutionRealm, FidelityContract, Graph,
    KernelDescriptor, KernelId, KernelRegistry, Layout, NodeDescriptor, NodeId, NodeInstance,
    PlanLimits, PolicyContract, PortDescriptor, PortRef, ProofContract, ResourceEnvelope, Target,
};

const ITERATIONS: usize = 10_000;

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
        }],
        capabilities: vec![Capability("abir".to_owned())],
        targets: vec![Target::Host, Target::McuAot, Target::BlutDurable],
        resources: ResourceEnvelope::bounded(4096, 1024, 1),
        determinism: Determinism::BitExact,
        stateful: false,
        proof: ProofContract {
            requires: vec![],
            provides: vec![format!("{name}.verified")],
        },
        policy: PolicyContract {
            requires: vec!["research".to_owned()],
            adds: vec![],
        },
        fidelity: FidelityContract {
            minimum_input: 65_000,
            maximum_loss: 0,
        },
        effect: Effect::Pure,
        retry_limit: 0,
        checkpointable: false,
    }
}

fn fixture() -> (KernelRegistry, Graph) {
    let mut registry = KernelRegistry::default();
    for (node_index, name) in ["source", "process", "sink"].into_iter().enumerate() {
        registry.register_descriptor(descriptor(name, node_index != 0));
        for (target_index, target) in [Target::Host, Target::McuAot, Target::BlutDurable]
            .into_iter()
            .enumerate()
        {
            registry
                .register_kernel(KernelDescriptor {
                    id: KernelId((node_index * 3 + target_index) as u32),
                    node_type: name.to_owned(),
                    node_version: 1,
                    target,
                    input_layouts: vec![Layout::Canonical],
                    output_layouts: vec![Layout::Canonical],
                    resources: ResourceEnvelope::bounded(4096, 1024, 1),
                    determinism: Determinism::BitExact,
                    lowering: format!("{target:?}"),
                    fuses_with_next: vec![],
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
            version: 1,
            nodes,
            edges,
            required_capabilities: vec![Capability("abir".to_owned())],
            required_proofs: vec![],
            policy: vec!["research".to_owned()],
            minimum_fidelity: 65_000,
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
        "status": "PASS",
        "revision": revision,
        "iterations": ITERATIONS,
        "compiled_plan_bytes": bytes.len(),
        "graph_id": hex(&plan.graph_id.0),
        "plan_id": hex(&plan.plan_id.0),
        "peak_bytes": plan.peak_bytes,
        "compile_ops_s": compile_ops_s,
        "encode_ops_s": encode_ops_s,
        "decode_ops_s": decode_ops_s,
        "compile_benchmark_realm": "host-stream",
        "identity_checked_realms": ["mcu-aot", "host-stream", "blut-durable"]
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
