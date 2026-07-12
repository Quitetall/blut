// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Named progress gate for ADR 0102's landed advanced-optimizer slice.
//!
//! Only user-priority scheduling has landed. Fusion, speculation, pipeline
//! parallelism, and real cache-warm ready-queue ordering remain later,
//! independently gated increments.

use std::sync::Arc;

use async_trait::async_trait;
use blut::framework::StageError;
use blut::framework::artifact::{Artifact, ContentHash};
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::dag_opt::DagOptimizer;
use blut::framework::executor::{ExecCtx, ParallelExecutor};
use blut::framework::plan::CompiledPlan;
use blut::framework::plan_spec::{PLAN_SPEC_VERSION, PlanSpec, SpecNode};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::recipes::recipe::RecipeDef;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

static EXECUTION_ORDER: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct OrderArgs {
    label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OrderArtifact {
    path: std::path::PathBuf,
    content_hash: ContentHash,
}

impl Artifact for OrderArtifact {
    const KIND: &'static str = "test.order-artifact";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    fn primary_path(&self) -> &std::path::Path {
        &self.path
    }
}

struct RecordOrder;

#[async_trait]
impl Stage for RecordOrder {
    const NAME: &'static str = "record_order";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        EXECUTION_ORDER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(args.label.clone());
        std::fs::create_dir_all(&ctx.stage_dir)
            .map_err(|error| StageError::Backend(error.into()))?;
        let path = ctx.stage_dir.join(format!("{}.txt", args.label));
        std::fs::write(&path, args.label.as_bytes())
            .map_err(|error| StageError::Backend(error.into()))?;
        Ok(OrderArtifact {
            content_hash: ContentHash::of_bytes(args.label.as_bytes()),
            path,
        })
    }
}

struct GateCookbook;

impl Cookbook for GateCookbook {
    fn name(&self) -> &'static str {
        "dag-opt-gate"
    }

    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }

    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static STAGES: &[(&str, ErasedStageCtor)] = &[("record_order", || Arc::new(RecordOrder))];
        STAGES
    }
}

fn compiled(priority: Option<i32>) -> CompiledPlan {
    compiled_nodes(&[("only", priority)])
}

fn compiled_nodes(nodes: &[(&str, Option<i32>)]) -> CompiledPlan {
    let mut registry = Registry::new();
    registry.register(Box::new(GateCookbook));
    PlanSpec {
        name: "priority-gate".into(),
        nodes: nodes
            .iter()
            .map(|(label, priority)| SpecNode {
                stage: "record_order".into(),
                args: serde_json::json!({ "label": label }),
                retry: None,
                timeout: None,
                priority: *priority,
            })
            .collect(),
        edges: Vec::new(),
        expansions: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(&registry)
    .expect("compile gate plan")
}

fn priority_only(enabled: bool) -> DagOptimizer {
    DagOptimizer {
        eliminate_dead_code: false,
        critical_path: false,
        cache_aware: false,
        memory_aware: false,
        priority_aware: enabled,
    }
}

#[test]
fn dag_opt_advanced_gate_covers_priority_pass_and_default_off_parity() {
    let (_, enabled) = priority_only(true).optimize(compiled(Some(42)));
    assert_eq!(enabled[&0].user_priority, 42);

    let (_, disabled) = priority_only(false).optimize(compiled(Some(42)));
    assert!(
        disabled.values().all(|hint| hint.user_priority == 0),
        "default-off priority pass must not change scheduling hints"
    );

    let (_, unset) = priority_only(true).optimize(compiled(None));
    assert!(
        unset.values().all(|hint| hint.user_priority == 0),
        "an omitted priority remains neutral"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_executes_higher_priority_ready_node_first() {
    EXECUTION_ORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    let plan = compiled_nodes(&[("bulk", Some(1)), ("urgent", Some(50))]);
    let temp = tempfile::tempdir().expect("executor tempdir");
    let mut ctx = ExecCtx::new(temp.path().join("job")).with_max_in_flight(1);
    ctx.dag_optimizer = Some(priority_only(true));

    ParallelExecutor::execute(plan, ctx)
        .await
        .expect("execute priority fixture");
    assert_eq!(
        *EXECUTION_ORDER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        ["urgent", "bulk"],
        "enabled priority pass must affect executor ready-node order"
    );
}
