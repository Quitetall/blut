// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed bridge from an authorized semantic physical plan into BLUT's
//! durable DAG executor.

use std::collections::BTreeSet;
use std::sync::Arc;

use blut_graph_core::{
    AuthorizedPlan, Determinism, Effect, ExecutionRealm, ImplementationId, Partiality, StepId,
};

use crate::framework::plan::CompiledPlan;
use crate::framework::stage::StageDyn;

#[derive(Debug, thiserror::Error)]
pub enum DurablePlanError {
    #[error("semantic plan targets {0:?}, not the BLUT durable realm")]
    WrongRealm(ExecutionRealm),
    #[error("durable resolver rejected physical step {step:?}: {message}")]
    Resolve { step: StepId, message: String },
    #[error(
        "step {step:?} has a deterministic semantic contract but a nondeterministic BLUT stage"
    )]
    Determinism { step: StepId },
    #[error(
        "step {step:?} retry contract differs: semantic={semantic_attempts}, BLUT={blut_attempts}"
    )]
    Retry {
        step: StepId,
        semantic_attempts: u32,
        blut_attempts: u32,
    },
    #[error("step {step:?} declares {required} bytes but BLUT reserves only {reserved} bytes")]
    Memory {
        step: StepId,
        required: u64,
        reserved: u64,
    },
    #[error("multiple physical buffers connect the same BLUT stages {from:?}->{to:?}")]
    AmbiguousPorts { from: StepId, to: StepId },
    #[error("BLUT's artifact seam cannot represent {outputs} distinct outputs from step {step:?}")]
    MultipleOutputs { step: StepId, outputs: usize },
    #[error("BLUT's tuple seam cannot represent an absent optional input at step {step:?}")]
    AbsentInput { step: StepId },
    #[error("BLUT's artifact seam cannot bind invocation input {input} at step {step:?}")]
    InvocationInput { step: StepId, input: u32 },
    #[error("BLUT adapter cannot preserve {effect:?} effects for step {step:?}")]
    UnsupportedEffect { step: StepId, effect: Effect },
    #[error("BLUT adapter cannot represent explicit gaps from step {step:?}")]
    PartialOutput { step: StepId },
    #[error("BLUT's legacy stage seam cannot provide {scope:?} state for step {step:?}")]
    UnsupportedState {
        step: StepId,
        scope: blut_graph_core::StateScope,
    },
    #[error("durable implementation identity differs for step {step:?}")]
    Implementation { step: StepId },
    #[error("durable checkpoint contract differs for step {step:?}")]
    Checkpoint { step: StepId },
    #[error("BLUT's tuple seam cannot bind feedback state {feedback:?} at step {step:?}")]
    FeedbackInput {
        step: StepId,
        feedback: blut_graph_core::FeedbackId,
    },
    #[error("durable policy recheck differs for step {step:?}")]
    Policy { step: StepId },
    #[error(transparent)]
    Plan(#[from] crate::framework::error::PlanError),
}

pub struct ResolvedDurableStep {
    pub stage: Arc<dyn StageDyn>,
    pub args: serde_json::Value,
    pub contract: DurableStepContract,
}

/// Trusted binding asserted by the application registry at adaptation time.
/// BLUT's legacy `StageDyn` cannot express these semantic contracts itself, so
/// missing or mismatched declarations fail closed instead of being inferred.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DurableStepContract {
    pub implementation_id: ImplementationId,
    pub effect: Effect,
    pub state: blut_graph_core::StateContract,
    pub rechecked_policy: Vec<String>,
}

/// Trusted application registry that binds each already-authorized physical
/// implementation to a compiled-in BLUT stage.
pub trait DurableStepResolver {
    fn resolve(&self, step: &blut_graph_core::CompiledNode) -> Result<ResolvedDurableStep, String>;
}

/// Keeps semantic identity and physical-step lineage beside the ordinary BLUT
/// plan. Callers execute `durable_plan()` through the existing durable
/// executor and retain this wrapper for audit/receipt correlation.
pub struct DurableAdaptedPlan {
    semantic: AuthorizedPlan,
    durable: CompiledPlan,
}

impl DurableAdaptedPlan {
    pub fn semantic_plan(&self) -> &AuthorizedPlan {
        &self.semantic
    }

    pub fn durable_plan(&self) -> &CompiledPlan {
        &self.durable
    }

    pub fn into_parts(self) -> (AuthorizedPlan, CompiledPlan) {
        (self.semantic, self.durable)
    }
}

pub fn adapt_durable_plan(
    name: impl Into<String>,
    recipe_args: serde_json::Value,
    semantic: AuthorizedPlan,
    resolver: &dyn DurableStepResolver,
) -> Result<DurableAdaptedPlan, DurablePlanError> {
    if semantic.realm != ExecutionRealm::BlutDurable {
        return Err(DurablePlanError::WrongRealm(semantic.realm));
    }
    let mut nodes = Vec::with_capacity(semantic.nodes.len());
    for step in &semantic.nodes {
        if step.output_bindings.len() > 1 {
            return Err(DurablePlanError::MultipleOutputs {
                step: step.id,
                outputs: step.output_bindings.len(),
            });
        }
        if step
            .input_bindings
            .contains(&blut_graph_core::InputBinding::Absent)
        {
            return Err(DurablePlanError::AbsentInput { step: step.id });
        }
        if let Some(input) = step
            .input_bindings
            .iter()
            .find_map(|binding| match binding {
                blut_graph_core::InputBinding::Invocation(input) => Some(*input),
                _ => None,
            })
        {
            return Err(DurablePlanError::InvocationInput {
                step: step.id,
                input,
            });
        }
        if let Some(feedback) = step
            .input_bindings
            .iter()
            .find_map(|binding| match binding {
                blut_graph_core::InputBinding::Feedback(feedback) => Some(*feedback),
                _ => None,
            })
        {
            return Err(DurablePlanError::FeedbackInput {
                step: step.id,
                feedback,
            });
        }
        if step.effect != Effect::Pure {
            return Err(DurablePlanError::UnsupportedEffect {
                step: step.id,
                effect: step.effect,
            });
        }
        if step.partiality != Partiality::Atomic {
            return Err(DurablePlanError::PartialOutput { step: step.id });
        }
        if step.state.scope != blut_graph_core::StateScope::Stateless {
            return Err(DurablePlanError::UnsupportedState {
                step: step.id,
                scope: step.state.scope,
            });
        }
        let resolved = resolver
            .resolve(step)
            .map_err(|message| DurablePlanError::Resolve {
                step: step.id,
                message,
            })?;
        if resolved.contract.implementation_id != step.implementation_id
            || resolved.contract.effect != step.effect
        {
            return Err(DurablePlanError::Implementation { step: step.id });
        }
        if resolved.contract.state != step.state {
            return Err(DurablePlanError::Checkpoint { step: step.id });
        }
        let mut expected_policy = semantic.propagated_policy.clone();
        expected_policy.sort_unstable();
        expected_policy.dedup();
        let mut rechecked_policy = resolved.contract.rechecked_policy.clone();
        rechecked_policy.sort_unstable();
        rechecked_policy.dedup();
        if rechecked_policy != expected_policy {
            return Err(DurablePlanError::Policy { step: step.id });
        }
        if step.determinism != Determinism::Nondeterministic && !resolved.stage.deterministic() {
            return Err(DurablePlanError::Determinism { step: step.id });
        }
        let semantic_attempts = u32::from(step.retry_limit) + 1;
        let blut_attempts = resolved.stage.retry().max_attempts;
        if semantic_attempts != blut_attempts {
            return Err(DurablePlanError::Retry {
                step: step.id,
                semantic_attempts,
                blut_attempts,
            });
        }
        let required = step
            .resources
            .peak_bytes
            .saturating_add(step.resources.scratch_bytes);
        let reserved = u64::from(resolved.stage.memory_gib_for(&resolved.args))
            .saturating_mul(crate::broker::footprint::GIB);
        if reserved < required {
            return Err(DurablePlanError::Memory {
                step: step.id,
                required,
                reserved,
            });
        }
        nodes.push((resolved.stage, resolved.args));
    }

    let mut seen = BTreeSet::new();
    let mut edges = Vec::new();
    for consumer in &semantic.nodes {
        for binding in &consumer.input_bindings {
            let blut_graph_core::InputBinding::Buffer(buffer) = binding else {
                continue;
            };
            let producer = semantic.buffers[buffer.0 as usize].producer;
            if !seen.insert((producer, consumer.id)) {
                return Err(DurablePlanError::AmbiguousPorts {
                    from: producer,
                    to: consumer.id,
                });
            }
            // Insertion order is the consumer's declared input-binding order;
            // BLUT preserves this order when assembling tuple<N> merge inputs.
            edges.push((producer.0, consumer.id.0));
        }
    }
    let recipe_args = serde_json::json!({
        "user": recipe_args,
        "semantic_graph": {
            "graph_id": hex(&semantic.graph_id.0),
            "plan_id": hex(&semantic.plan_id.0),
            "realm": "blut-durable",
            "steps": semantic.nodes.iter().map(|step| serde_json::json!({
                "step": step.id.0,
                "semantic_nodes": step.semantic_nodes.iter().map(|node| node.0).collect::<Vec<_>>(),
                "semantic_configs": step.semantic_configs,
                "implementation_id": hex(&step.implementation_id.0),
                "ordered_inputs": step.input_ports.iter().zip(&step.input_bindings).map(|(port, binding)| serde_json::json!({
                    "port": port,
                    "binding": match binding {
                        blut_graph_core::InputBinding::Buffer(buffer) => format!("buffer:{}", buffer.0),
                        blut_graph_core::InputBinding::Invocation(input) => format!("invocation:{input}"),
                        blut_graph_core::InputBinding::Feedback(feedback) => format!("feedback:{}", feedback.0),
                        blut_graph_core::InputBinding::Absent => "absent".to_string(),
                    },
                })).collect::<Vec<_>>(),
                "ordered_outputs": step.output_ports.iter().zip(&step.output_bindings).map(|(port, binding)| serde_json::json!({
                    "port": port,
                    "binding": match binding {
                        blut_graph_core::OutputBinding::Buffer(buffer) => format!("buffer:{}", buffer.0),
                        blut_graph_core::OutputBinding::Terminal => "terminal".to_string(),
                    },
                })).collect::<Vec<_>>(),
                "input_contracts": step.input_contracts,
                "output_contracts": step.output_contracts,
                "state": step.state,
                "subgraph_path": step.subgraph_path,
            })).collect::<Vec<_>>(),
            "persistent_state_bytes": semantic.persistent_state_bytes,
            "session": semantic.session,
        }
    });
    let durable = CompiledPlan::from_erased_graph(name, recipe_args, nodes, edges)?;
    Ok(DurableAdaptedPlan { semantic, durable })
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("String writes cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use blut_graph_core::{
        Capability, Compiler, Determinism, Edge, Effect, FidelityContract, Graph, ImplementationId,
        KernelDescriptor, KernelId, KernelRegistry, Layout, NodeDescriptor, NodeId, NodeInstance,
        NodeTypeRef, PolicyContract, PortDescriptor, PortRef, ProofContract, ResourceEnvelope,
        Target,
    };

    use super::*;
    use crate::framework::{Resource, Stage, StageContext, StageError};

    struct UnitStage;

    #[async_trait::async_trait]
    impl Stage for UnitStage {
        const NAME: &'static str = "semantic_unit";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[];
        const MEMORY_GIB: u32 = 1;
        type Input = ();
        type Output = ();
        type Args = ();

        async fn run(
            &self,
            _ctx: &StageContext,
            _input: Self::Input,
            _args: &Self::Args,
        ) -> Result<Self::Output, StageError> {
            Ok(())
        }
    }

    struct Resolver;

    impl DurableStepResolver for Resolver {
        fn resolve(
            &self,
            _step: &blut_graph_core::CompiledNode,
        ) -> Result<ResolvedDurableStep, String> {
            Ok(ResolvedDurableStep {
                stage: Arc::new(UnitStage),
                args: serde_json::json!(null),
                contract: DurableStepContract {
                    implementation_id: _step.implementation_id,
                    effect: _step.effect,
                    state: _step.state.clone(),
                    rechecked_policy: vec![],
                },
            })
        }
    }

    struct WrongPolicyResolver;

    impl DurableStepResolver for WrongPolicyResolver {
        fn resolve(
            &self,
            step: &blut_graph_core::CompiledNode,
        ) -> Result<ResolvedDurableStep, String> {
            Ok(ResolvedDurableStep {
                stage: Arc::new(UnitStage),
                args: serde_json::json!(null),
                contract: DurableStepContract {
                    implementation_id: step.implementation_id,
                    effect: step.effect,
                    state: step.state.clone(),
                    rechecked_policy: vec!["untrusted-extra-policy".into()],
                },
            })
        }
    }

    struct WrongStateResolver;

    impl DurableStepResolver for WrongStateResolver {
        fn resolve(
            &self,
            step: &blut_graph_core::CompiledNode,
        ) -> Result<ResolvedDurableStep, String> {
            Ok(ResolvedDurableStep {
                stage: Arc::new(UnitStage),
                args: serde_json::json!(null),
                contract: DurableStepContract {
                    implementation_id: step.implementation_id,
                    effect: step.effect,
                    state: blut_graph_core::StateContract {
                        scope: blut_graph_core::StateScope::Session,
                        max_bytes: 1,
                        checkpoint: blut_graph_core::CheckpointContract {
                            mode: blut_graph_core::CheckpointMode::Disabled,
                            max_snapshot_bytes: 0,
                            max_interval_invocations: 0,
                        },
                    },
                    rechecked_policy: vec![],
                },
            })
        }
    }

    fn descriptor(name: &str, input: bool) -> NodeDescriptor {
        let port = PortDescriptor {
            name: if input { "in" } else { "out" }.into(),
            semantic_type: "abir.block".into(),
            optional: false,
            layouts: vec![Layout::Canonical],
            max_bytes: 64,
            ..PortDescriptor::opaque(if input { "in" } else { "out" }, "abir.block", 64)
        };
        NodeDescriptor {
            type_name: name.into(),
            version: 1,
            inputs: if input { vec![port] } else { vec![] },
            outputs: vec![PortDescriptor {
                name: "out".into(),
                semantic_type: "abir.block".into(),
                optional: false,
                layouts: vec![Layout::Canonical],
                max_bytes: 64,
                ..PortDescriptor::opaque("out", "abir.block", 64)
            }],
            capabilities: vec![Capability("abir".into())],
            targets: vec![Target::BlutDurable, Target::Host],
            resources: ResourceEnvelope::bounded(0, 0, 1),
            determinism: Determinism::BitExact,
            config: blut_graph_core::ConfigSchema::default(),
            state: blut_graph_core::StateContract::stateless(),
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
            partiality: blut_graph_core::Partiality::Atomic,
            failure: blut_graph_core::FailureContract { domains: vec![] },
            effect: Effect::Pure,
            retry_limit: 0,
        }
    }

    fn plan_with_feedback(realm: ExecutionRealm, feedback: bool) -> AuthorizedPlan {
        let mut registry = KernelRegistry::default();
        for (index, name) in ["source", "sink"].into_iter().enumerate() {
            let mut descriptor = descriptor(name, index != 0);
            if feedback && index == 0 {
                let mut history = PortDescriptor::opaque("history", "abir.block", 64);
                history.optional = true;
                descriptor.inputs.push(history);
            }
            registry.register_descriptor(descriptor).unwrap();
            let target = realm.target();
            registry
                .register_kernel(KernelDescriptor {
                    id: KernelId(index as u32),
                    implements: vec![NodeTypeRef {
                        type_name: name.into(),
                        version: 1,
                    }],
                    implementation_id: ImplementationId([index as u8 + 1; 32]),
                    conversion: None,
                    target,
                    input_layouts: vec![Layout::Canonical],
                    output_layouts: vec![Layout::Canonical],
                    resources: ResourceEnvelope::bounded(0, 0, 1),
                    determinism: Determinism::BitExact,
                    lowering: "durable-test".into(),
                })
                .unwrap();
        }
        Compiler::new(&registry, realm)
            .compile(&Graph {
                version: 3,
                nodes: vec![
                    NodeInstance {
                        id: NodeId(0),
                        descriptor: "source".into(),
                        descriptor_version: 1,
                        config: BTreeMap::new(),
                    },
                    NodeInstance {
                        id: NodeId(1),
                        descriptor: "sink".into(),
                        descriptor_version: 1,
                        config: BTreeMap::new(),
                    },
                ],
                edges: vec![Edge {
                    from: PortRef {
                        node: NodeId(0),
                        port: "out".into(),
                    },
                    to: PortRef {
                        node: NodeId(1),
                        port: "in".into(),
                    },
                }],
                feedback: if feedback {
                    vec![blut_graph_core::FeedbackEdge {
                        from: PortRef {
                            node: NodeId(1),
                            port: "out".into(),
                        },
                        to: PortRef {
                            node: NodeId(0),
                            port: "history".into(),
                        },
                        delay: blut_graph_core::DelayContract {
                            invocations: 1,
                            initial: blut_graph_core::DelayInitial::Absent,
                        },
                    }]
                } else {
                    vec![]
                },
                invocation_inputs: vec![],
                required_capabilities: vec![Capability("abir".into())],
                required_proofs: vec![],
                policy: vec![],
                minimum_fidelity: 0,
                session: feedback.then(|| blut_graph_core::SessionContract {
                    namespace: "durable-test".into(),
                    max_concurrent_sessions: 2,
                    max_idle_millis: 1_000,
                    reset_on_plan_change: true,
                }),
            })
            .unwrap()
    }

    fn plan(realm: ExecutionRealm) -> AuthorizedPlan {
        plan_with_feedback(realm, false)
    }

    #[test]
    fn authorized_durable_plan_maps_to_the_existing_blut_dag() {
        let semantic = plan(ExecutionRealm::BlutDurable);
        let graph_id = hex(&semantic.graph_id.0);
        let adapted =
            adapt_durable_plan("semantic", serde_json::json!({}), semantic, &Resolver).unwrap();
        assert_eq!(adapted.durable_plan().n_nodes(), 2);
        assert_eq!(adapted.durable_plan().n_edges(), 1);
        assert_eq!(
            adapted.durable_plan().recipe_args()["semantic_graph"]["graph_id"],
            graph_id
        );
        assert_eq!(
            adapted.durable_plan().recipe_args()["semantic_graph"]["steps"][1]["ordered_inputs"][0]
                ["port"],
            "in"
        );
        assert_eq!(
            adapted.durable_plan().recipe_args()["semantic_graph"]["steps"][1]["input_contracts"]
                [0]["semantic_type"],
            "abir.block"
        );
    }

    #[test]
    fn host_plan_cannot_enter_the_durable_adapter() {
        assert!(matches!(
            adapt_durable_plan(
                "semantic",
                serde_json::json!({}),
                plan(ExecutionRealm::HostStream),
                &Resolver,
            ),
            Err(DurablePlanError::WrongRealm(ExecutionRealm::HostStream))
        ));
    }

    #[test]
    fn durable_adapter_requires_an_exact_policy_recheck() {
        assert!(matches!(
            adapt_durable_plan(
                "semantic",
                serde_json::json!({}),
                plan(ExecutionRealm::BlutDurable),
                &WrongPolicyResolver,
            ),
            Err(DurablePlanError::Policy { .. })
        ));
    }

    #[test]
    fn durable_adapter_requires_exact_state_and_rejects_feedback_until_supported() {
        assert!(matches!(
            adapt_durable_plan(
                "semantic",
                serde_json::json!({}),
                plan(ExecutionRealm::BlutDurable),
                &WrongStateResolver,
            ),
            Err(DurablePlanError::Checkpoint { .. })
        ));
        assert!(matches!(
            adapt_durable_plan(
                "semantic",
                serde_json::json!({}),
                plan_with_feedback(ExecutionRealm::BlutDurable, true),
                &Resolver,
            ),
            Err(DurablePlanError::FeedbackInput { .. })
        ));
    }
}
