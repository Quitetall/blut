// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0102 executable condition-gate acceptance tests.
//!
//! These tests stay on the public cookbook/PlanSpec/executor seams. A condition
//! gate is control metadata: it may delay or prune a data-ready node, but must
//! not become a typed predecessor or change that node's cache identity.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;
use blut::framework::artifact::{Artifact, BranchDecision, ContentHash};
use blut::framework::cache::CacheProof;
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::error::StageError;
use blut::framework::executor::{ExecCtx, execute_plan};
use blut::framework::plan_spec::{ConditionGateSpec, PLAN_SPEC_VERSION, PlanSpec, SpecNode};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::framework::status::{HostedEvent, StageEvent};
use blut::recipes::recipe::RecipeDef;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const DECISION_STAGE: &str = "conditional_test_decision";
const SOURCE_STAGE: &str = "conditional_test_source";
const GUARDED_STAGE: &str = "conditional_test_guarded";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct TestValue {
    value: u32,
}

impl Artifact for TestValue {
    const KIND: &'static str = "conditional-test.value";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.value.to_le_bytes())
    }

    fn primary_path(&self) -> &Path {
        Path::new("")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct DecisionArgs {
    value: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SourceArgs {
    value: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct GuardArgs {
    add: u32,
}

struct DecisionStage;

#[async_trait]
impl Stage for DecisionStage {
    const NAME: &'static str = DECISION_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = BranchDecision;
    type Args = DecisionArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &DecisionArgs,
    ) -> Result<BranchDecision, StageError> {
        Ok(BranchDecision { value: args.value })
    }
}

struct SourceStage;

#[async_trait]
impl Stage for SourceStage {
    const NAME: &'static str = SOURCE_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = TestValue;
    type Args = SourceArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &SourceArgs,
    ) -> Result<TestValue, StageError> {
        Ok(TestValue { value: args.value })
    }
}

struct GuardedStage;

#[async_trait]
impl Stage for GuardedStage {
    const NAME: &'static str = GUARDED_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = TestValue;
    type Output = TestValue;
    type Args = GuardArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: TestValue,
        args: &GuardArgs,
    ) -> Result<TestValue, StageError> {
        let marker = ctx.stage_dir.join("guard-ran");
        std::fs::write(&marker, b"ran").map_err(|source| StageError::Io {
            path: marker,
            source,
        })?;
        Ok(TestValue {
            value: input.value + args.add,
        })
    }
}

struct ConditionalTestCookbook;

impl Cookbook for ConditionalTestCookbook {
    fn name(&self) -> &'static str {
        "conditional-execution-test"
    }

    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }

    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static STAGES: &[(&str, ErasedStageCtor)] = &[
            (DECISION_STAGE, || Arc::new(DecisionStage)),
            (SOURCE_STAGE, || Arc::new(SourceStage)),
            (GUARDED_STAGE, || Arc::new(GuardedStage)),
        ];
        STAGES
    }
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register(Box::new(ConditionalTestCookbook));
    registry
}

fn node(stage: &str, args: serde_json::Value) -> SpecNode {
    SpecNode {
        stage: stage.into(),
        args,
        retry: None,
        timeout: None,
        priority: None,
        pure: false,
    }
}

fn conditional_spec(decision: bool) -> PlanSpec {
    PlanSpec {
        name: format!("conditional-{decision}"),
        nodes: vec![
            node(DECISION_STAGE, serde_json::json!({ "value": decision })),
            node(SOURCE_STAGE, serde_json::json!({ "value": 41 })),
            node(GUARDED_STAGE, serde_json::json!({ "add": 1 })),
        ],
        edges: vec![(1, 2)],
        expansions: Vec::new(),
        condition_gates: vec![ConditionGateSpec {
            condition: 0,
            target: 2,
            when: true,
        }],
        version: PLAN_SPEC_VERSION,
    }
}

fn ungated_spec() -> PlanSpec {
    PlanSpec {
        name: "ungated-equivalent".into(),
        nodes: vec![
            node(SOURCE_STAGE, serde_json::json!({ "value": 41 })),
            node(GUARDED_STAGE, serde_json::json!({ "add": 1 })),
        ],
        edges: vec![(0, 1)],
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
}

fn conditional_root_spec(decision: bool) -> PlanSpec {
    PlanSpec {
        name: format!("conditional-root-{decision}"),
        nodes: vec![
            node(DECISION_STAGE, serde_json::json!({ "value": decision })),
            node(SOURCE_STAGE, serde_json::json!({ "value": 41 })),
        ],
        edges: Vec::new(),
        expansions: Vec::new(),
        condition_gates: vec![ConditionGateSpec {
            condition: 0,
            target: 1,
            when: true,
        }],
        version: PLAN_SPEC_VERSION,
    }
}

fn status_events(job_dir: &Path) -> Vec<StageEvent> {
    let body = std::fs::read_to_string(job_dir.join("status.jsonl"))
        .expect("conditional execution writes status.jsonl");
    body.lines()
        .map(|line| {
            serde_json::from_str::<HostedEvent>(line)
                .expect("status line is a HostedEvent")
                .event
        })
        .collect()
}

fn materializes_stage(event: &StageEvent, expected: &str) -> bool {
    match event {
        StageEvent::StageBegin { stage_name, .. }
        | StageEvent::StageEnd { stage_name, .. }
        | StageEvent::StageSkipped { stage_name, .. }
        | StageEvent::StageFailed { stage_name, .. }
        | StageEvent::StageBlocked { stage_name, .. }
        | StageEvent::StageStep { stage_name, .. }
        | StageEvent::StageRetrying { stage_name, .. } => stage_name == expected,
        StageEvent::StagePruned { .. } | StageEvent::StepGap { .. } => false,
    }
}

fn is_pruned_stage(event: &StageEvent, expected: &str) -> bool {
    matches!(
        event,
        StageEvent::StagePruned { stage_name, .. } if stage_name == expected
    )
}

fn cache_entry_count(job_dir: &Path) -> usize {
    std::fs::read_dir(job_dir.join("_cache"))
        .expect("job-local cache exists")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .count()
}

#[tokio::test(flavor = "current_thread")]
async fn true_decision_runs_guarded_branch_and_returns_its_output() {
    let temp = tempfile::tempdir().expect("conditional true tempdir");
    let job_dir = temp.path().join("job");
    let plan = conditional_spec(true)
        .compile(&registry())
        .expect("compile true condition");

    let result = execute_plan(plan, ExecCtx::new(job_dir.clone()))
        .await
        .expect("execute selected branch");

    let output = result
        .final_output
        .expect("selected guarded terminal produces final output")
        .into_typed::<TestValue>()
        .expect("decode guarded output");
    assert_eq!(output, TestValue { value: 42 });
    assert!(
        job_dir
            .join(format!("stages/2-{GUARDED_STAGE}/guard-ran"))
            .is_file(),
        "the selected guarded stage must run and promote its private output"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn false_decision_prunes_without_lifecycle_or_cache_materialization() {
    let temp = tempfile::tempdir().expect("conditional false tempdir");
    let job_dir = temp.path().join("job");
    let plan = conditional_spec(false)
        .compile(&registry())
        .expect("compile false condition");

    let result = execute_plan(plan, ExecCtx::new(job_dir.clone()))
        .await
        .expect("pruned branch is a successful plan outcome");

    assert!(
        result.final_output.is_none(),
        "pruned terminal has no output"
    );
    assert_eq!(result.n_cache_misses, 2, "only selector and source execute");
    assert_eq!(cache_entry_count(&job_dir), 2);
    assert!(
        !job_dir.join(format!("stages/2-{GUARDED_STAGE}")).exists(),
        "an unselected branch must not acquire a real stage directory"
    );
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, GUARDED_STAGE)),
        "an unselected branch must emit no run/cache lifecycle record"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| is_pruned_stage(event, GUARDED_STAGE))
            .count(),
        1,
        "the audit trail distinguishes an intentional prune from a missing stage"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn false_decision_does_not_return_a_guarded_roots_seed_input() {
    let temp = tempfile::tempdir().expect("conditional root tempdir");
    let job_dir = temp.path().join("job");

    let result = execute_plan(
        conditional_root_spec(false)
            .compile(&registry())
            .expect("compile guarded graph-input root"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("prune guarded graph-input root");

    assert!(
        result.final_output.is_none(),
        "the root's pre-seeded unit input is not a produced final artifact"
    );
    assert_eq!((result.n_cache_hits, result.n_cache_misses), (0, 1));
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, SOURCE_STAGE))
    );
    assert!(
        events
            .iter()
            .any(|event| is_pruned_stage(event, SOURCE_STAGE))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn condition_relation_is_cache_neutral_for_the_selected_data_path() {
    let temp = tempfile::tempdir().expect("cache-neutral condition tempdir");
    let job_dir = temp.path().join("same-job");
    let registry = registry();

    let selected = execute_plan(
        conditional_spec(true)
            .compile(&registry)
            .expect("compile selected conditional plan"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("run selected conditional plan");
    assert_eq!((selected.n_cache_hits, selected.n_cache_misses), (0, 3));
    let selected_proof =
        CacheProof::read_from(&job_dir.join(format!("stages/2-{GUARDED_STAGE}/cache-proof.json")))
            .expect("selected guarded cache proof");

    let ungated = execute_plan(
        ungated_spec()
            .compile(&registry)
            .expect("compile equivalent ungated plan"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("run equivalent ungated plan in the same job/cache");
    assert_eq!(
        (ungated.n_cache_hits, ungated.n_cache_misses),
        (2, 0),
        "source and guarded node must reuse the selected plan's exact entries"
    );
    let ungated_proof =
        CacheProof::read_from(&job_dir.join(format!("stages/1-{GUARDED_STAGE}/cache-proof.json")))
            .expect("ungated guarded cache proof");
    assert_eq!(
        selected_proof.key, ungated_proof.key,
        "a condition relation must not enter the guarded node's cache key"
    );
}

#[test]
fn execute_plan_forces_parallel_with_no_executor_environment_override() {
    let current_exe = std::env::current_exe().expect("locate this integration-test binary");
    let output = Command::new(current_exe)
        .arg("--ignored")
        .arg("--exact")
        .arg("conditional_parallel_subprocess_entry")
        .arg("--nocapture")
        .env_remove("BLUT_EXECUTOR")
        .env_remove("BLUT_KILL_ON_NAN")
        .output()
        .expect("launch isolated conditional executor check");
    assert!(
        output.status.success(),
        "isolated conditional execution failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "subprocess entry used by execute_plan_forces_parallel_with_no_executor_environment_override"]
fn conditional_parallel_subprocess_entry() {
    assert!(std::env::var_os("BLUT_EXECUTOR").is_none());
    assert!(std::env::var_os("BLUT_KILL_ON_NAN").is_none());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build subprocess runtime");
    runtime.block_on(async {
        let temp = tempfile::tempdir().expect("parallel dispatch tempdir");
        let job_dir = temp.path().join("job");
        let result = execute_plan(
            conditional_spec(false)
                .compile(&registry())
                .expect("compile subprocess conditional plan"),
            ExecCtx::new(job_dir.clone()),
        )
        .await
        .expect("condition gate forces the parallel executor");

        assert!(result.final_output.is_none());
        let events = status_events(&job_dir);
        assert!(
            events
                .iter()
                .all(|event| !materializes_stage(event, GUARDED_STAGE)),
            "false condition must still prune when no executor env override exists"
        );
        assert!(
            events
                .iter()
                .any(|event| is_pruned_stage(event, GUARDED_STAGE)),
            "the forced-parallel path must record the control prune"
        );
    });
}
