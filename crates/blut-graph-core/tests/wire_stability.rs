// SPDX-License-Identifier: AGPL-3.0-or-later

//! The wire, pinned to literals.
//!
//! Every other assertion in this crate is RELATIONAL, and `compile.rs` says so
//! of itself: "nothing pins one (every assertion in this crate is relational)".
//! A relational test survives any change that moves every identity together,
//! which is exactly what a change to the realm / target / layout vocabulary
//! does. Those three types' ordinals are not an implementation detail. They are
//! folded into `graph_id` and `plan_id` through seven little-endian `put_u32`
//! sites in `compile.rs`, and they reach the BGP3 bytes as postcard variant
//! indices. `Debug` is on the same footing, because `KernelDescriptor::lowering`
//! is conventionally built as `format!("{target:?}")` and that STRING is hashed.
//!
//! So this file pins absolute values for one fixed graph: each realm's
//! `graph_id`, `plan_id` and complete AOT byte string, and the `Debug` form of
//! every variant of all three types. A refactor that preserves the wire leaves
//! every literal below untouched. One that does not fails here, in bytes rather
//! than in argument.
//!
//! The fixture is a verbatim copy of the one in `examples/graph_evidence.rs`, on
//! purpose: a golden that reached into an example would move when the example
//! moved.

use std::collections::BTreeMap;

use blut_graph_core::{
    Capability, CompiledPlan, Compiler, Determinism, Edge, Effect, ExecutionRealm,
    FidelityContract, Graph, ImplementationId, KernelDescriptor, KernelId, KernelRegistry, Layout,
    NodeDescriptor, NodeId, NodeInstance, NodeTypeRef, PlanLimits, PolicyContract, PortDescriptor,
    PortRef, ProofContract, ResourceEnvelope, Target,
};

/// `ExecutionRealm::McuAot`, compiled from `fixture()` at 0.2.0-alpha.1.
const MCU_AOT_AOT: &str = concat!(
    "42475033031bdfdfd98f0018601e2aef4b70e30e545524c9e49d5302bf27d2a52b760fb5e3cf0dae673d55b36c3f",
    "60ec94d9f0bc9356fc98840bffb50902480b959ad052b20003000102030001000106736f75726365010100010202",
    "02020202020202020202020202020202020202020202020202020202020280208008010000064d6375416f740000",
    "01036f75740001036f75740c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d00000000",
    "0000000000010000000100000001000000000000000000000000010101010770726f636573730101000405050505",
    "0505050505050505050505050505050505050505050505050505050580208008010000064d6375416f7400010269",
    "6e01036f75740102696e0c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d0000000000",
    "000000000100000001000001036f75740c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f",
    "6d0000000000000000000100000001000001000001000100000000000000000000020102010473696e6b01010007",
    "080808080808080808080808080808080808080808080808080808080808080880208008010000064d6375416f74",
    "000102696e01036f75740102696e0c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d00",
    "00000000000000000100000001000001036f75740c73616d706c652e626c6f636b0000802008626c6f622d726566",
    "0461746f6d0000000000000000000100000001000001000101010000000000000000000002000080200001010100",
    "0100802001010202000000031070726f636573732e76657269666965640d73696e6b2e76657269666965640f736f",
    "757263652e766572696669656401087265736561726368ffff0380680000",
);

/// `ExecutionRealm::HostStream`, compiled from `fixture()` at 0.2.0-alpha.1.
const HOST_STREAM_AOT: &str = concat!(
    "42475033031bdfdfd98f0018601e2aef4b70e30e545524c9e49d5302bf27d2a52b760fb5e3d4e11ab2c4676b2a37",
    "27ac5e5bea9248acf1e122c2997dc9108226c25b9c7cf60103000102030001000106736f75726365010100000101",
    "0101010101010101010101010101010101010101010101010101010101018020800801000004486f737400000103",
    "6f75740001036f75740c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d000000000000",
    "000000010000000100000001000000000000000000000000010101010770726f6365737301010003040404040404",
    "04040404040404040404040404040404040404040404040404048020800801000004486f7374000102696e01036f",
    "75740102696e0c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d000000000000000000",
    "0100000001000001036f75740c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d000000",
    "0000000000000100000001000001000001000100000000000000000000020102010473696e6b0101000607070707",
    "070707070707070707070707070707070707070707070707070707078020800801000004486f7374000102696e01",
    "036f75740102696e0c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d00000000000000",
    "00000100000001000001036f75740c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f6d00",
    "00000000000000000100000001000001000101010000000000000000000002000080200001010100010080200101",
    "0202000000031070726f636573732e76657269666965640d73696e6b2e76657269666965640f736f757263652e76",
    "6572696669656401087265736561726368ffff0380680000",
);

/// `ExecutionRealm::BlutDurable`, compiled from `fixture()` at 0.2.0-alpha.1.
const BLUT_DURABLE_AOT: &str = concat!(
    "42475033031bdfdfd98f0018601e2aef4b70e30e545524c9e49d5302bf27d2a52b760fb5e30b5f81ff4163a2c65e",
    "693b38ab7efc135bb4604c4374503804e8639bcac3e3a50203000102030001000106736f75726365010100020303",
    "030303030303030303030303030303030303030303030303030303030303802080080100000b426c757444757261",
    "626c65000001036f75740001036f75740c73616d706c652e626c6f636b0000802008626c6f622d7265660461746f",
    "6d000000000000000000010000000100000001000000000000000000000000010101010770726f63657373010100",
    "050606060606060606060606060606060606060606060606060606060606060606802080080100000b426c757444",
    "757261626c65000102696e01036f75740102696e0c73616d706c652e626c6f636b0000802008626c6f622d726566",
    "0461746f6d0000000000000000000100000001000001036f75740c73616d706c652e626c6f636b0000802008626c",
    "6f622d7265660461746f6d0000000000000000000100000001000001000001000100000000000000000000020102",
    "010473696e6b01010008090909090909090909090909090909090909090909090909090909090909090980208008",
    "0100000b426c757444757261626c65000102696e01036f75740102696e0c73616d706c652e626c6f636b00008020",
    "08626c6f622d7265660461746f6d0000000000000000000100000001000001036f75740c73616d706c652e626c6f",
    "636b0000802008626c6f622d7265660461746f6d0000000000000000000100000001000001000101010000000000",
    "0000000000020000802000010101000100802001010202000000031070726f636573732e76657269666965640d73",
    "696e6b2e76657269666965640f736f757263652e766572696669656401087265736561726368ffff0380680000",
);

fn descriptor(name: &str, input: bool) -> NodeDescriptor {
    NodeDescriptor {
        type_name: name.to_owned(),
        version: 1,
        inputs: if input {
            vec![PortDescriptor {
                name: "in".to_owned(),
                semantic_type: "sample.block".to_owned(),
                optional: false,
                layouts: vec![Layout::Canonical],
                max_bytes: 4096,
                ..PortDescriptor::opaque("in", "sample.block", 4096)
            }]
        } else {
            vec![]
        },
        outputs: vec![PortDescriptor {
            name: "out".to_owned(),
            semantic_type: "sample.block".to_owned(),
            optional: false,
            layouts: vec![Layout::Canonical],
            max_bytes: 4096,
            ..PortDescriptor::opaque("out", "sample.block", 4096)
        }],
        capabilities: vec![Capability("sample".to_owned())],
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
            required_capabilities: vec![Capability("sample".to_owned())],
            required_proofs: vec![],
            policy: vec!["research".to_owned()],
            minimum_fidelity: 65_000,
            session: None,
        },
    )
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut out, "{byte:02x}").expect("string write");
    }
    out
}

fn assert_wire(realm: ExecutionRealm, graph_id: &str, plan_id: &str, aot: &str) {
    let (registry, graph) = fixture();
    let plan = Compiler::new(&registry, realm)
        .compile(&graph)
        .expect("the fixture graph compiles in every realm");
    let compiled = plan.as_plan();
    assert_eq!(hex(&compiled.graph_id.0), graph_id, "graph_id moved");
    assert_eq!(hex(&compiled.plan_id.0), plan_id, "plan_id moved");
    let bytes = compiled.to_aot_bytes().expect("the compiled plan encodes");
    assert_eq!(hex(&bytes), aot, "BGP3 bytes moved");

    // The bytes must also still DECODE, to the same identity. An encode-only
    // golden would pass while `from_aot_bytes` rejected its own output.
    let decoded = CompiledPlan::from_aot_bytes(&bytes, PlanLimits::default())
        .expect("the golden bytes decode under default limits");
    assert_eq!(decoded.plan_id, compiled.plan_id, "decode changed plan_id");
    assert_eq!(
        decoded.graph_id, compiled.graph_id,
        "decode changed graph_id"
    );
}

#[test]
fn mcu_aot_plan_is_byte_stable() {
    assert_wire(
        ExecutionRealm::McuAot,
        "1bdfdfd98f0018601e2aef4b70e30e545524c9e49d5302bf27d2a52b760fb5e3",
        "cf0dae673d55b36c3f60ec94d9f0bc9356fc98840bffb50902480b959ad052b2",
        MCU_AOT_AOT,
    );
}

#[test]
fn host_stream_plan_is_byte_stable() {
    assert_wire(
        ExecutionRealm::HostStream,
        "1bdfdfd98f0018601e2aef4b70e30e545524c9e49d5302bf27d2a52b760fb5e3",
        "d4e11ab2c4676b2a3727ac5e5bea9248acf1e122c2997dc9108226c25b9c7cf6",
        HOST_STREAM_AOT,
    );
}

#[test]
fn blut_durable_plan_is_byte_stable() {
    assert_wire(
        ExecutionRealm::BlutDurable,
        "1bdfdfd98f0018601e2aef4b70e30e545524c9e49d5302bf27d2a52b760fb5e3",
        "0b5f81ff4163a2c65e693b38ab7efc135bb4604c4374503804e8639bcac3e3a5",
        BLUT_DURABLE_AOT,
    );
}

/// One graph, three realms: the realm moves the plan identity and not the graph
/// identity. Stated here so a reader sees the invariant the three hex strings
/// above are protecting, rather than inferring it from them.
#[test]
fn the_realm_moves_the_plan_id_and_not_the_graph_id() {
    let (registry, graph) = fixture();
    let plans: Vec<_> = [
        ExecutionRealm::McuAot,
        ExecutionRealm::HostStream,
        ExecutionRealm::BlutDurable,
    ]
    .into_iter()
    .map(|realm| {
        Compiler::new(&registry, realm)
            .compile(&graph)
            .expect("the fixture graph compiles in every realm")
    })
    .collect();
    for pair in plans.windows(2) {
        assert_eq!(
            pair[0].as_plan().graph_id,
            pair[1].as_plan().graph_id,
            "the graph identity must not depend on the realm"
        );
        assert_ne!(
            pair[0].as_plan().plan_id,
            pair[1].as_plan().plan_id,
            "the plan identity must depend on the realm"
        );
    }
}

/// `Debug` is wire here, not diagnostics.
///
/// `KernelDescriptor::lowering` is built as `format!("{target:?}")` by this
/// crate's own examples and by consumers, and `lowering` is hashed into the
/// plan. A newtype whose derived `Debug` prints `Target(1)` instead of `Host`
/// would move every plan id in the fleet without touching a single ordinal.
#[test]
fn debug_spellings_are_frozen() {
    assert_eq!(
        format!("{:?}", [Target::McuAot, Target::Host, Target::BlutDurable]),
        "[McuAot, Host, BlutDurable]"
    );
    assert_eq!(
        format!(
            "{:?}",
            [
                Layout::Canonical,
                Layout::ChannelMajor,
                Layout::TimeMajor,
                Layout::Packed,
                Layout::Opaque
            ]
        ),
        "[Canonical, ChannelMajor, TimeMajor, Packed, Opaque]"
    );
    assert_eq!(
        format!(
            "{:?}",
            [
                ExecutionRealm::McuAot,
                ExecutionRealm::HostStream,
                ExecutionRealm::BlutDurable
            ]
        ),
        "[McuAot, HostStream, BlutDurable]"
    );
}
