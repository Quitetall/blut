use std::collections::BTreeMap;

use blut_graph_core::{
    Capability, CompileError, Compiler, ConfigField, ConfigSchema, ConfigType, ConfigValue,
    Determinism, Edge, Effect, ExecutionRealm, FailureContract, FidelityContract, Graph,
    ImplementationId, KernelDescriptor, KernelId, KernelRegistry, Layout, NodeDescriptor, NodeId,
    NodeInstance, NodeTypeRef, Partiality, PolicyContract, PortDescriptor, PortMap, PortRef,
    ProofContract, ResourceEnvelope, StateContract, SubgraphConfigMap, SubgraphInterfacePort,
    SubgraphLowering, SubgraphNode, SubgraphSchema, Target, subgraph_identity,
};

fn config_schema(name: &str) -> ConfigSchema {
    ConfigSchema {
        fields: vec![ConfigField {
            name: name.into(),
            value_type: ConfigType::U64 {
                minimum: 1,
                maximum: 64,
            },
            required: true,
            default: None,
        }],
    }
}

fn descriptor(
    name: &str,
    input_type: &str,
    output_type: &str,
    config: ConfigSchema,
    failures: &[&str],
) -> NodeDescriptor {
    NodeDescriptor {
        type_name: name.into(),
        version: 1,
        inputs: vec![PortDescriptor::opaque("input", input_type, 64)],
        outputs: vec![PortDescriptor::opaque("output", output_type, 64)],
        capabilities: vec![Capability("test.capability".into())],
        targets: vec![Target::Host],
        resources: ResourceEnvelope::bounded(64, 0, 1),
        determinism: Determinism::BitExact,
        config,
        state: StateContract::stateless(),
        subgraph: None,
        proof: ProofContract {
            requires: vec![],
            provides: vec![],
            invalidates: vec![],
        },
        policy: PolicyContract {
            requires: vec![],
            adds: vec![],
        },
        fidelity: FidelityContract {
            minimum_input: 0,
            maximum_loss: 0,
        },
        partiality: Partiality::Atomic,
        failure: FailureContract {
            domains: failures.iter().map(|value| (*value).into()).collect(),
        },
        effect: Effect::Pure,
        retry_limit: 0,
    }
}

fn kernel(id: u32, implements: &[&str], input: Layout, output: Layout) -> KernelDescriptor {
    KernelDescriptor {
        id: KernelId(id),
        implements: implements
            .iter()
            .map(|name| NodeTypeRef {
                type_name: (*name).into(),
                version: 1,
            })
            .collect(),
        implementation_id: ImplementationId([id as u8; 32]),
        conversion: None,
        target: Target::Host,
        input_layouts: vec![input],
        output_layouts: vec![output],
        resources: ResourceEnvelope::bounded(64, 0, 1),
        determinism: Determinism::BitExact,
        lowering: format!("test:{id}"),
    }
}

#[test]
fn outer_config_materializes_into_inner_subgraph_and_changes_graph_identity() {
    let mut registry = KernelRegistry::default();
    registry
        .register_descriptor(descriptor(
            "test.stage",
            "test.signal",
            "test.packet",
            config_schema("order"),
            &["test.codec"],
        ))
        .unwrap();
    registry
        .register_kernel(kernel(
            1,
            &["test.stage"],
            Layout::Canonical,
            Layout::Canonical,
        ))
        .unwrap();

    let mut schema = SubgraphSchema {
        id: blut_graph_core::SubgraphId([0; 32]),
        version: 1,
        nodes: vec![SubgraphNode {
            id: NodeId(0),
            node_type: NodeTypeRef {
                type_name: "test.stage".into(),
                version: 1,
            },
            config: BTreeMap::from([("order".into(), ConfigValue::U64(1))]),
            child: None,
        }],
        edges: vec![],
        inputs: vec![SubgraphInterfacePort {
            name: "input".into(),
            inner: PortRef {
                node: NodeId(0),
                port: "input".into(),
            },
        }],
        outputs: vec![SubgraphInterfacePort {
            name: "output".into(),
            inner: PortRef {
                node: NodeId(0),
                port: "output".into(),
            },
        }],
    };
    schema.id = subgraph_identity(&schema);
    registry.register_subgraph(schema.clone()).unwrap();

    let mut outer = descriptor(
        "test.outer",
        "test.signal",
        "test.packet",
        config_schema("max_order"),
        &["test.codec"],
    );
    outer.subgraph = Some(SubgraphLowering {
        subgraph: schema.id,
        input_map: vec![PortMap {
            outer: "input".into(),
            inner: "input".into(),
        }],
        output_map: vec![PortMap {
            outer: "output".into(),
            inner: "output".into(),
        }],
        config_map: vec![SubgraphConfigMap {
            outer: "max_order".into(),
            node: NodeId(0),
            inner: "order".into(),
        }],
    });
    registry.register_descriptor(outer).unwrap();

    let materialize = |value| {
        registry
            .materialize_subgraph(&NodeInstance {
                id: NodeId(9),
                descriptor: "test.outer".into(),
                descriptor_version: 1,
                config: BTreeMap::from([("max_order".into(), ConfigValue::U64(value))]),
            })
            .unwrap()
    };
    let order_7 = materialize(7);
    let order_8 = materialize(8);
    assert_eq!(order_7.graph.nodes[0].config["order"], ConfigValue::U64(7));
    assert_eq!(order_7.inputs[0].name, "input");
    assert_eq!(order_7.outputs[0].name, "output");

    let plan_7 = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&order_7.graph)
        .unwrap();
    let plan_8 = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&order_8.graph)
        .unwrap();
    assert_ne!(plan_7.as_plan().graph_id, plan_8.as_plan().graph_id);
}

#[test]
fn fused_fallible_chain_preserves_declared_failure_domain_union() {
    let mut registry = KernelRegistry::default();
    registry
        .register_descriptor(descriptor(
            "test.first",
            "test.signal",
            "test.middle",
            ConfigSchema::default(),
            &["test.decode", "test.io"],
        ))
        .unwrap();
    registry
        .register_descriptor(descriptor(
            "test.second",
            "test.middle",
            "test.packet",
            ConfigSchema::default(),
            &["test.codec", "test.io"],
        ))
        .unwrap();
    registry
        .register_kernel(kernel(
            1,
            &["test.first"],
            Layout::Canonical,
            Layout::Canonical,
        ))
        .unwrap();
    registry
        .register_kernel(kernel(
            2,
            &["test.second"],
            Layout::Canonical,
            Layout::Canonical,
        ))
        .unwrap();
    registry
        .register_kernel(kernel(
            3,
            &["test.first", "test.second"],
            Layout::Canonical,
            Layout::Canonical,
        ))
        .unwrap();

    let graph = Graph {
        version: 3,
        nodes: vec![
            NodeInstance {
                id: NodeId(0),
                descriptor: "test.first".into(),
                descriptor_version: 1,
                config: BTreeMap::new(),
            },
            NodeInstance {
                id: NodeId(1),
                descriptor: "test.second".into(),
                descriptor_version: 1,
                config: BTreeMap::new(),
            },
        ],
        edges: vec![Edge {
            from: PortRef {
                node: NodeId(0),
                port: "output".into(),
            },
            to: PortRef {
                node: NodeId(1),
                port: "input".into(),
            },
        }],
        feedback: vec![],
        invocation_inputs: vec![PortRef {
            node: NodeId(0),
            port: "input".into(),
        }],
        required_capabilities: vec![Capability("test.capability".into())],
        required_proofs: vec![],
        policy: vec![],
        minimum_fidelity: 0,
        session: None,
    };

    let reference = Compiler::new(&registry, ExecutionRealm::HostStream)
        .with_fusion(false)
        .compile(&graph)
        .unwrap();
    let fused = Compiler::new(&registry, ExecutionRealm::HostStream)
        .compile(&graph)
        .unwrap();
    assert_eq!(reference.as_plan().nodes.len(), 2);
    assert_eq!(fused.as_plan().nodes.len(), 1);
    assert_eq!(
        fused.as_plan().nodes[0].failure.domains,
        vec!["test.codec", "test.decode", "test.io"]
    );
}

#[test]
fn config_binding_rejects_unknown_or_incompatible_fields() {
    let mut registry = KernelRegistry::default();
    registry
        .register_descriptor(descriptor(
            "test.stage",
            "test.signal",
            "test.packet",
            config_schema("order"),
            &[],
        ))
        .unwrap();
    let mut schema = SubgraphSchema {
        id: blut_graph_core::SubgraphId([0; 32]),
        version: 1,
        nodes: vec![SubgraphNode {
            id: NodeId(0),
            node_type: NodeTypeRef {
                type_name: "test.stage".into(),
                version: 1,
            },
            config: BTreeMap::from([("order".into(), ConfigValue::U64(1))]),
            child: None,
        }],
        edges: vec![],
        inputs: vec![SubgraphInterfacePort {
            name: "input".into(),
            inner: PortRef {
                node: NodeId(0),
                port: "input".into(),
            },
        }],
        outputs: vec![SubgraphInterfacePort {
            name: "output".into(),
            inner: PortRef {
                node: NodeId(0),
                port: "output".into(),
            },
        }],
    };
    schema.id = subgraph_identity(&schema);
    registry.register_subgraph(schema.clone()).unwrap();

    let mut outer = descriptor(
        "test.outer",
        "test.signal",
        "test.packet",
        config_schema("max_order"),
        &[],
    );
    outer.subgraph = Some(SubgraphLowering {
        subgraph: schema.id,
        input_map: vec![PortMap {
            outer: "input".into(),
            inner: "input".into(),
        }],
        output_map: vec![PortMap {
            outer: "output".into(),
            inner: "output".into(),
        }],
        config_map: vec![SubgraphConfigMap {
            outer: "missing".into(),
            node: NodeId(0),
            inner: "order".into(),
        }],
    });
    assert!(matches!(
        registry.register_descriptor(outer),
        Err(CompileError::InvalidDescriptor(name, 1)) if name == "test.outer"
    ));

    let mut unmapped = descriptor(
        "test.unmapped",
        "test.signal",
        "test.packet",
        ConfigSchema {
            fields: vec![
                ConfigField {
                    name: "max_order".into(),
                    value_type: ConfigType::U64 {
                        minimum: 1,
                        maximum: 64,
                    },
                    required: true,
                    default: None,
                },
                ConfigField {
                    name: "window_size".into(),
                    value_type: ConfigType::U64 {
                        minimum: 1,
                        maximum: 64,
                    },
                    required: true,
                    default: None,
                },
            ],
        },
        &[],
    );
    unmapped.subgraph = Some(SubgraphLowering {
        subgraph: schema.id,
        input_map: vec![PortMap {
            outer: "input".into(),
            inner: "input".into(),
        }],
        output_map: vec![PortMap {
            outer: "output".into(),
            inner: "output".into(),
        }],
        config_map: vec![SubgraphConfigMap {
            outer: "max_order".into(),
            node: NodeId(0),
            inner: "order".into(),
        }],
    });
    assert!(matches!(
        registry.register_descriptor(unmapped),
        Err(CompileError::InvalidDescriptor(name, 1)) if name == "test.unmapped"
    ));
}
