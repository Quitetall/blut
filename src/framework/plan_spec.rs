// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `PlanSpec` — the serde plan IR (ADR 0078).
//!
//! `PlanSpec` is the single, plain-data contract between plan *authors* and
//! the engine. Three front-ends emit it:
//!   * Starlark scripts (`.star`, behind the `dsl` feature) — hermetic eval;
//!   * raw `.json` — the future Python SDK's entry point (this module is that
//!     door: `serde_json::from_str::<PlanSpec>` then [`PlanSpec::compile`]);
//!   * (unchanged) declarative TOML, via the existing linear
//!     [`crate::recipes::declarative`] path.
//!
//! A `PlanSpec` is a graph of `{stage-name, args}` nodes plus `(from, to)`
//! edges. [`PlanSpec::compile`] resolves each stage name against a
//! [`Registry`] (registered stages ONLY — no dynamic code loading) and builds
//! a fully kind-checked [`CompiledPlan`] via
//! [`CompiledPlan::from_erased_graph`]. Every wiring break is a typed error
//! before anything executes; the produced plan runs through the SAME executor
//! as any compiled recipe.
//!
//! The struct is a STABLE, versioned wire contract — evolve it additive-only
//! (new fields `#[serde(default)]`) so an older `PlanSpec` JSON keeps parsing.

use serde::{Deserialize, Serialize};

use crate::framework::Registry;
use crate::framework::cache::CacheHandle;
use crate::framework::error::PlanError;
use crate::framework::plan::CompiledPlan;

/// IR version stamped into every `PlanSpec`. Bump only on a
/// backward-incompatible change (additive fields do NOT bump it).
pub const PLAN_SPEC_VERSION: u32 = 1;

fn default_version() -> u32 {
    PLAN_SPEC_VERSION
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Registry-resolved erased nodes: `(stage, args)` pairs ready for
/// `CompiledPlan::from_erased_graph`/`from_erased_template`.
type ResolvedNodes = Vec<(
    std::sync::Arc<dyn crate::framework::stage::StageDyn>,
    serde_json::Value,
)>;

/// One node of a [`PlanSpec`]: a registered stage name + its args JSON, plus the
/// optional per-node execution-control knobs added in **PlanSpec v1.1**
/// (ADR 0088): `retry` and `timeout`. Both are `#[serde(default)]` +
/// `skip_serializing_if` absent, so a v1 plan (neither field) round-trips + hashes
/// identically to a v1.1 plan whose fields are unset. Crucially they are
/// execution-control, NOT computation: they are NOT part of a node's cache key
/// (which is over stage code + args + input), so adding them never invalidates a
/// cached artifact. Omitted `args` default to JSON `null` (the declarative path).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecNode {
    /// Stage name — must resolve via [`Registry::find_erased_stage`].
    pub stage: String,
    /// Per-stage args (validated against the stage's schema at run time).
    #[serde(default)]
    pub args: serde_json::Value,
    /// PlanSpec v1.1 (ADR 0088): per-node retry policy. `None` = use the stage's
    /// own `RETRY` default (v1 behaviour). NOT in the cache key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<crate::framework::retry::RetryPolicy>,
    /// PlanSpec v1.1 (ADR 0088): per-node soft/hard timeout. `None` = the stage's
    /// own `TIMEOUT` default (v1 behaviour). NOT in the cache key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<crate::framework::retry::StageTimeout>,
    /// ADR 0102 pass #4: optional user scheduling priority (higher runs earlier
    /// among ready nodes). `None` = neutral (0), byte-identical to a v1 plan.
    /// Scheduling metadata only — NOT in the cache key (like retry/timeout), and
    /// `skip_serializing_if` keeps an unset plan's canonical bytes unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    /// ADR 0102: author request for speculative execution. Default false and
    /// omitted from the wire for byte-identical legacy plans. Compile accepts
    /// `true` only when the registered stage independently declares
    /// `Stage::SPECULATION_SAFE`; neither side can authorize speculation alone.
    #[serde(default, skip_serializing_if = "is_false")]
    pub pure: bool,
}

/// A control dependency distinct from typed data edges (ADR 0102).
///
/// `condition` must produce [`BranchDecision`](crate::framework::artifact::BranchDecision).
/// The `target` keeps its ordinary data predecessors and cache identity; this
/// relation only decides whether that data-ready node may run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConditionGateSpec {
    pub condition: u32,
    pub target: u32,
    pub when: bool,
}

/// A typed runtime fan-out (ADR 0078 `map_output`): when the node at index
/// `parent` completes with a `list` output, the executor runs `template` once
/// per element, seeding the template's single root with that element.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapSpec {
    /// Index into the enclosing plan's `nodes` — the list-producing parent.
    pub parent: u32,
    /// The sub-plan instantiated per element (its single root consumes the
    /// element; nested maps are not allowed in v1).
    pub template: PlanSpec,
    #[serde(default)]
    pub label: Option<String>,
}

/// The plan IR. `nodes` are dense (indices `0..nodes.len()` are node ids);
/// `edges` are `(producer_index, consumer_index)`, and their **order is the
/// tuple element order** a merge node's `gather_input` will assemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanSpec {
    pub name: String,
    pub nodes: Vec<SpecNode>,
    #[serde(default)]
    pub edges: Vec<(u32, u32)>,
    /// Typed runtime fan-outs (ADR 0078). Empty for a plain DAG.
    #[serde(default)]
    pub expansions: Vec<MapSpec>,
    /// Boolean control gates. Empty/omitted preserves the legacy plan bytes.
    /// The first implementation slice accepts one exclusive, non-reconvergent
    /// gated branch per plan; richer phi/select semantics require a later IR.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub condition_gates: Vec<ConditionGateSpec>,
    /// IR version (see [`PLAN_SPEC_VERSION`]). Defaulted so pre-versioned
    /// JSON still parses.
    #[serde(default = "default_version")]
    pub version: u32,
}

/// Failure compiling a [`PlanSpec`] into a runnable plan.
#[derive(Debug, thiserror::Error)]
pub enum PlanSpecError {
    /// A named stage is in no registered cookbook (dynamic loading is
    /// forbidden — the author must name a compiled-in stage).
    #[error(
        "plan spec '{name}': stage '{stage}' is not in any registered cookbook (see `blut stage list`)"
    )]
    UnknownStage { name: String, stage: String },
    /// The graph failed structural/kind validation (delegated to
    /// [`CompiledPlan::from_erased_graph`]).
    #[error("plan spec '{name}': {source}")]
    Plan {
        name: String,
        #[source]
        source: PlanError,
    },
    /// A `map_output` expansion is malformed (bad parent index, a parent whose
    /// output isn't a `list`, a nested map, or a template kind-check failure).
    #[error("plan spec '{name}': map over node {parent}: {detail}")]
    BadMap {
        name: String,
        parent: u32,
        detail: String,
    },
    /// A condition gate is malformed or outside the deliberately narrow v1
    /// non-reconvergent branch contract.
    #[error("plan spec '{name}': condition {condition} -> target {target}: {detail}")]
    BadCondition {
        name: String,
        condition: u32,
        target: u32,
        detail: String,
    },
    /// A raw plan author requested purity for a stage whose implementation did
    /// not independently certify deterministic speculation safety.
    #[error(
        "plan spec '{name}': node {node} stage '{stage}' requests pure=true but the stage is not deterministic and speculation-safe"
    )]
    UnsafePure {
        name: String,
        node: u32,
        stage: String,
    },
}

impl PlanSpec {
    /// Canonical byte encoding of this spec — key order normalized so two
    /// specs that differ only in JSON key ordering hash identically. Used for
    /// the plan provenance fingerprint (ADR 0078) and golden snapshots.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        // A PlanSpec always serializes (all fields are plain data); the
        // canonicalizer is the same one the cache uses for stage args.
        let v = serde_json::to_value(self).expect("PlanSpec serializes to JSON");
        CacheHandle::canonical_json_bytes(&v)
    }

    /// Provenance fingerprint (ADR 0078): a content hash over the authoring
    /// `source` (e.g. a `.star` script), the canonical `args`, and this spec
    /// — so lineage records exactly what produced a plan, and a change to any
    /// of the three moves the id. Length-prefixed so the concatenation is
    /// unambiguous. Computed in the ENGINE (not the DSL tool) so the engine
    /// owns provenance regardless of how the spec was authored.
    pub fn provenance_fingerprint(
        &self,
        source: &str,
        args: &serde_json::Value,
    ) -> crate::framework::artifact::ContentHash {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"blut-star-v1");
        buf.extend_from_slice(&(source.len() as u64).to_le_bytes());
        buf.extend_from_slice(source.as_bytes());
        let canon_args = CacheHandle::canonical_json_bytes(args);
        buf.extend_from_slice(&(canon_args.len() as u64).to_le_bytes());
        buf.extend_from_slice(&canon_args);
        let spec_bytes = self.canonical_bytes();
        buf.extend_from_slice(&(spec_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(&spec_bytes);
        crate::framework::artifact::ContentHash::of_bytes(&buf)
    }

    /// Resolve this spec's node stage-names against `reg` (registered stages
    /// only — no dynamic loading). Returns the erased `(stage, args)` pairs.
    fn resolve_nodes(&self, reg: &Registry) -> Result<ResolvedNodes, PlanSpecError> {
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for sn in &self.nodes {
            let ctor =
                reg.find_erased_stage(&sn.stage)
                    .ok_or_else(|| PlanSpecError::UnknownStage {
                        name: self.name.clone(),
                        stage: sn.stage.clone(),
                    })?;
            nodes.push((ctor(), sn.args.clone()));
        }
        Ok(nodes)
    }

    fn compile_condition_gates(
        &self,
        nodes: &ResolvedNodes,
    ) -> Result<Vec<crate::framework::plan::CompiledConditionGate>, PlanSpecError> {
        for (index, (spec, (stage, _))) in self.nodes.iter().zip(nodes).enumerate() {
            if spec.pure && (!stage.speculation_safe() || !stage.deterministic()) {
                return Err(PlanSpecError::UnsafePure {
                    name: self.name.clone(),
                    node: index as u32,
                    stage: stage.name().to_string(),
                });
            }
        }

        let Some(&gate) = self.condition_gates.first() else {
            return Ok(Vec::new());
        };
        let bad = |detail: String| PlanSpecError::BadCondition {
            name: self.name.clone(),
            condition: gate.condition,
            target: gate.target,
            detail,
        };
        if self.condition_gates.len() != 1 {
            return Err(bad(
                "v1 supports one exclusive condition gate per plan; use a later phi/select IR for multiple arms"
                    .into(),
            ));
        }
        if !self.expansions.is_empty() {
            return Err(bad(
                "condition gates cannot be composed with map_output expansions in v1".into(),
            ));
        }
        let n = nodes.len();
        if gate.condition as usize >= n || gate.target as usize >= n {
            return Err(bad(format!("node index out of range (plan has {n} nodes)")));
        }
        if gate.condition == gate.target {
            return Err(bad("a condition cannot gate itself".into()));
        }
        let decision_kind = <crate::framework::artifact::BranchDecision as crate::framework::artifact::Artifact>::KIND;
        let got_kind = nodes[gate.condition as usize].0.output_kind();
        if got_kind != decision_kind {
            return Err(bad(format!(
                "condition stage '{}' outputs '{got_kind}', expected '{decision_kind}'",
                nodes[gate.condition as usize].0.name()
            )));
        }
        if nodes[gate.condition as usize].0.is_advisory() {
            return Err(bad(
                "the condition stage cannot be advisory because its decision is load-bearing"
                    .into(),
            ));
        }
        if self
            .edges
            .iter()
            .any(|&(from, to)| from == gate.condition && to == gate.target)
        {
            return Err(bad(
                "the condition relation must not also be a typed data edge".into(),
            ));
        }

        let mut successors = vec![Vec::<u32>::new(); n];
        let mut predecessors = vec![Vec::<u32>::new(); n];
        for &(from, to) in &self.edges {
            if from as usize >= n || to as usize >= n {
                return Err(bad(format!(
                    "data edge {from}->{to} is out of range (plan has {n} nodes)"
                )));
            }
            successors[from as usize].push(to);
            predecessors[to as usize].push(from);
        }

        // The losing arm is pruned as the target plus its data descendants.
        // Reject a descendant with an incoming edge from outside that set: it
        // would be a reconvergent/phi node whose missing input has no v1 meaning.
        let mut branch = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::from([gate.target]);
        while let Some(id) = queue.pop_front() {
            if !branch.insert(id) {
                continue;
            }
            queue.extend(successors[id as usize].iter().copied());
        }
        for &id in &branch {
            if id == gate.target {
                continue;
            }
            if let Some(&outside) = predecessors[id as usize]
                .iter()
                .find(|pred| !branch.contains(pred))
            {
                return Err(bad(format!(
                    "reconvergent node {id} also depends on outside node {outside}; v1 has no phi/select semantics"
                )));
            }
        }

        // Existing PlanResult selects one terminal output. Keep that contract
        // unambiguous: aside from the decision node, the gated branch owns the
        // plan's sole data terminal. A false decision therefore yields None;
        // a true decision yields the same terminal artifact as an ungated run.
        let terminals: Vec<u32> = (0..n as u32)
            .filter(|id| *id != gate.condition && successors[*id as usize].is_empty())
            .collect();
        if terminals.len() != 1 || !branch.contains(&terminals[0]) {
            return Err(bad(format!(
                "v1 requires one gated terminal output, found terminals {terminals:?}"
            )));
        }

        Ok(vec![crate::framework::plan::CompiledConditionGate {
            condition: gate.condition,
            target: gate.target,
            when: gate.when,
        }])
    }

    /// Resolve every node's stage by name against `reg`, then build a
    /// fully kind-checked [`CompiledPlan`] — including any runtime `map_output`
    /// expansions (ADR 0078). An unknown stage, a wiring break, or a malformed
    /// map is a precise [`PlanSpecError`]; nothing executes until this returns
    /// `Ok`.
    pub fn compile(&self, reg: &Registry) -> Result<CompiledPlan, PlanSpecError> {
        let nodes = self.resolve_nodes(reg)?;
        let condition_gates = self.compile_condition_gates(&nodes)?;

        // Compile every map expansion BEFORE the main plan is consumed — each
        // needs its parent node's declared element kind for the template
        // kind-check.
        let mut expansions = Vec::with_capacity(self.expansions.len());
        for m in &self.expansions {
            let bad = |detail: String| PlanSpecError::BadMap {
                name: self.name.clone(),
                parent: m.parent,
                detail,
            };
            let parent = m.parent as usize;
            if parent >= nodes.len() {
                return Err(bad(format!(
                    "parent index out of range (plan has {} nodes)",
                    nodes.len()
                )));
            }
            // The parent must produce a `list`; its element kind drives the
            // template root's type check.
            let elem_kind = nodes[parent].0.output_element_kind().ok_or_else(|| {
                bad(format!(
                    "parent stage '{}' does not output a list (its output is '{}')",
                    nodes[parent].0.name(),
                    nodes[parent].0.output_kind()
                ))
            })?;
            // v1: no nested maps, conditional templates, or speculative map
            // members. Reject purity rather than silently dropping it when the
            // template is lowered to `CompiledTemplate`.
            if !m.template.expansions.is_empty()
                || !m.template.condition_gates.is_empty()
                || m.template.nodes.iter().any(|node| node.pure)
            {
                return Err(bad(
                    "nested maps, condition gates, and pure template nodes are not supported (v1)"
                        .into(),
                ));
            }
            let template_nodes = m.template.resolve_nodes(reg)?;
            let template = CompiledPlan::from_erased_template(
                elem_kind,
                template_nodes,
                m.template.edges.clone(),
            )
            .map_err(|e| bad(e.to_string()))?;
            expansions.push(crate::framework::plan::MapExpansion {
                parent: m.parent,
                template: std::sync::Arc::new(template),
                label: m.label.clone(),
                pipeline: false,
            });
        }

        // recipe_args = provenance for lineage/audit (the executor uses each
        // node's own args). Parallel to the declarative path's blob.
        let recipe_nodes = self
            .nodes
            .iter()
            .map(|s| {
                let mut node = serde_json::json!({ "stage": s.stage, "args": s.args });
                if s.pure {
                    node["pure"] = serde_json::Value::Bool(true);
                }
                node
            })
            .collect::<Vec<_>>();
        let mut recipe_args = serde_json::json!({
            "plan_spec": true,
            "version": self.version,
            "nodes": recipe_nodes,
        });
        if !self.condition_gates.is_empty() {
            recipe_args["condition_gates"] =
                serde_json::to_value(&self.condition_gates).expect("condition gates serialize");
        }
        let mut plan = CompiledPlan::from_erased_graph(
            self.name.clone(),
            recipe_args,
            nodes,
            self.edges.clone(),
        )
        .map_err(|source| PlanSpecError::Plan {
            name: self.name.clone(),
            source,
        })?;
        // PlanSpec v1.1 (ADR 0088): thread per-node retry/timeout onto the
        // compiled nodes. Execution-control only — node cache keys unaffected.
        let overrides: Vec<_> = self
            .nodes
            .iter()
            .map(|n| crate::framework::plan::ExecutionOverrides {
                retry: n.retry,
                timeout: n.timeout,
                priority: n.priority,
                pure: n.pure,
            })
            .collect();
        plan.apply_execution_overrides(&overrides);
        plan = plan
            .with_condition_gates(condition_gates)
            .map_err(|source| PlanSpecError::Plan {
                name: self.name.clone(),
                source,
            })?;
        Ok(plan.with_expansions(expansions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::LamuTrainerBackend;
    use crate::framework::Compatible;
    use crate::framework::artifact::{Artifact, BranchDecision, ContentHash};
    use crate::framework::cookbook::Cookbook;
    use crate::framework::error::StageError;
    use crate::framework::resource::Resource;
    use crate::framework::stage::{ErasedStageCtor, Stage, StageContext};
    use crate::recipes::recipe::RecipeDef;
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Toy artifacts + stages: () -> A -> B, plus a join (A,B) -> C.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct A;
    impl Artifact for A {
        const KIND: &'static str = "spec.a";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"a")
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct B;
    impl Artifact for B {
        const KIND: &'static str = "spec.b";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"b")
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    // Empty-braces (not a unit struct) so it deserializes from `{}` — the
    // natural args a PlanSpec author writes for a no-required-args stage.
    #[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
    struct E {}

    struct MakeA;
    impl Compatible<LamuTrainerBackend> for MakeA {}
    #[async_trait]
    impl Stage for MakeA {
        const NAME: &'static str = "spec_make_a";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = A;
        type Args = E;
        async fn run(&self, _c: &StageContext, _i: (), _a: &E) -> Result<A, StageError> {
            Ok(A)
        }
    }

    struct SafeMakeA;
    impl Compatible<LamuTrainerBackend> for SafeMakeA {}
    #[async_trait]
    impl Stage for SafeMakeA {
        const NAME: &'static str = "spec_safe_make_a";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        const SPECULATION_SAFE: bool = true;
        type Input = ();
        type Output = A;
        type Args = E;
        async fn run(&self, _c: &StageContext, _i: (), _a: &E) -> Result<A, StageError> {
            Ok(A)
        }
    }

    struct Decide;
    impl Compatible<LamuTrainerBackend> for Decide {}
    #[async_trait]
    impl Stage for Decide {
        const NAME: &'static str = "spec_decide";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = BranchDecision;
        type Args = E;
        async fn run(
            &self,
            _c: &StageContext,
            _i: (),
            _a: &E,
        ) -> Result<BranchDecision, StageError> {
            Ok(BranchDecision { value: true })
        }
    }

    struct NondeterministicSafeMakeA;
    impl Compatible<LamuTrainerBackend> for NondeterministicSafeMakeA {}
    #[async_trait]
    impl Stage for NondeterministicSafeMakeA {
        const NAME: &'static str = "spec_nondeterministic_safe_make_a";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        const DETERMINISTIC: bool = false;
        const SPECULATION_SAFE: bool = true;
        type Input = ();
        type Output = A;
        type Args = E;
        async fn run(&self, _c: &StageContext, _i: (), _a: &E) -> Result<A, StageError> {
            Ok(A)
        }
    }

    struct AToB;
    impl Compatible<LamuTrainerBackend> for AToB {}
    #[async_trait]
    impl Stage for AToB {
        const NAME: &'static str = "spec_a_to_b";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = A;
        type Output = B;
        type Args = E;
        async fn run(&self, _c: &StageContext, _i: A, _a: &E) -> Result<B, StageError> {
            Ok(B)
        }
    }

    // Item + a list producer/consumer for map_output tests.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Item;
    impl Artifact for Item {
        const KIND: &'static str = "spec.item";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"item")
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    fn default_width() -> usize {
        2
    }
    #[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
    struct ShardArgs {
        #[serde(default = "default_width")]
        width: usize,
    }

    struct Sharder;
    impl Compatible<LamuTrainerBackend> for Sharder {}
    #[async_trait]
    impl Stage for Sharder {
        const NAME: &'static str = "spec_sharder";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = crate::framework::artifact::ListOf<Item>;
        type Args = ShardArgs;
        async fn run(
            &self,
            _c: &StageContext,
            _i: (),
            a: &ShardArgs,
        ) -> Result<crate::framework::artifact::ListOf<Item>, StageError> {
            Ok(crate::framework::artifact::ListOf(vec![Item; a.width]))
        }
    }

    struct ItemToA;
    impl Compatible<LamuTrainerBackend> for ItemToA {}
    #[async_trait]
    impl Stage for ItemToA {
        const NAME: &'static str = "spec_item_to_a";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = Item;
        type Output = A;
        type Args = E;
        async fn run(&self, _c: &StageContext, _i: Item, _a: &E) -> Result<A, StageError> {
            Ok(A)
        }
    }

    static OVERSIZED_MAP_SHARD_RAN: AtomicU32 = AtomicU32::new(0);
    struct OversizedItemToA;
    impl Compatible<LamuTrainerBackend> for OversizedItemToA {}
    #[async_trait]
    impl Stage for OversizedItemToA {
        const NAME: &'static str = "spec_oversized_item_to_a";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        const MEMORY_GIB: u32 = 8;
        type Input = Item;
        type Output = A;
        type Args = E;
        async fn run(&self, _c: &StageContext, _i: Item, _a: &E) -> Result<A, StageError> {
            OVERSIZED_MAP_SHARD_RAN.fetch_add(1, Ordering::SeqCst);
            Ok(A)
        }
    }

    static ERASED: &[(&str, ErasedStageCtor)] = &[
        ("spec_make_a", || Arc::new(MakeA)),
        ("spec_safe_make_a", || Arc::new(SafeMakeA)),
        ("spec_nondeterministic_safe_make_a", || {
            Arc::new(NondeterministicSafeMakeA)
        }),
        ("spec_decide", || Arc::new(Decide)),
        ("spec_a_to_b", || Arc::new(AToB)),
        ("spec_sharder", || Arc::new(Sharder)),
        ("spec_item_to_a", || Arc::new(ItemToA)),
        ("spec_oversized_item_to_a", || Arc::new(OversizedItemToA)),
    ];
    static NO_RECIPES: &[&RecipeDef] = &[];
    struct ToyCookbook;
    impl Cookbook for ToyCookbook {
        fn name(&self) -> &'static str {
            "spec_toy"
        }
        fn recipes(&self) -> &'static [&'static RecipeDef] {
            NO_RECIPES
        }
        fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
            ERASED
        }
    }
    fn toy_registry() -> Registry {
        let mut reg = Registry::new();
        reg.register(Box::new(ToyCookbook));
        reg
    }

    fn linear_spec() -> PlanSpec {
        PlanSpec {
            name: "chain".into(),
            nodes: vec![
                SpecNode {
                    stage: "spec_make_a".into(),
                    args: serde_json::json!({}),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                },
                SpecNode {
                    stage: "spec_a_to_b".into(),
                    args: serde_json::json!({}),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                },
            ],
            edges: vec![(0, 1)],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: PLAN_SPEC_VERSION,
        }
    }

    fn condition_spec(condition: &str, target: &str) -> PlanSpec {
        PlanSpec {
            name: "conditional".into(),
            nodes: vec![
                SpecNode {
                    stage: condition.into(),
                    args: serde_json::json!({}),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                },
                SpecNode {
                    stage: target.into(),
                    args: serde_json::json!({}),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                },
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

    #[test]
    fn json_round_trip_and_deny_unknown_fields() {
        let spec = linear_spec();
        let json = serde_json::to_string(&spec).unwrap();
        let back: PlanSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back.nodes.len(), 2);
        assert_eq!(back.edges, vec![(0, 1)]);
        assert_eq!(back.version, PLAN_SPEC_VERSION);
        // An unknown field is rejected (the wire contract is closed).
        let bad = r#"{"name":"x","nodes":[],"surprise":true}"#;
        assert!(serde_json::from_str::<PlanSpec>(bad).is_err());
        // A pre-versioned spec (no `version`) still parses (defaulted).
        let noversion = r#"{"name":"x","nodes":[{"stage":"spec_make_a"}]}"#;
        let p: PlanSpec = serde_json::from_str(noversion).unwrap();
        assert_eq!(p.version, PLAN_SPEC_VERSION);
        assert!(p.nodes[0].args.is_null()); // omitted args -> null
    }

    #[test]
    fn purity_and_condition_fields_are_additive_for_legacy_json() {
        let legacy = r#"{"name":"legacy","nodes":[{"stage":"spec_make_a"}],"edges":[],"expansions":[],"version":1}"#;
        let parsed: PlanSpec = serde_json::from_str(legacy).unwrap();

        assert!(!parsed.nodes[0].pure);
        assert!(parsed.condition_gates.is_empty());
        assert_eq!(
            String::from_utf8(parsed.canonical_bytes()).unwrap(),
            r#"{"edges":[],"expansions":[],"name":"legacy","nodes":[{"args":null,"stage":"spec_make_a"}],"version":1}"#,
            "default purity/control fields must not perturb legacy canonical bytes"
        );

        let encoded = serde_json::to_string(&parsed).unwrap();
        assert!(
            !encoded.contains("pure"),
            "default pure=false omitted: {encoded}"
        );
        assert!(
            !encoded.contains("condition_gates"),
            "empty condition gates omitted: {encoded}"
        );
    }

    #[test]
    fn pure_request_requires_independent_stage_capability() {
        let mut unsafe_spec = condition_spec("spec_decide", "spec_make_a");
        unsafe_spec.condition_gates.clear();
        unsafe_spec.nodes.remove(0);
        unsafe_spec.nodes[0].pure = true;
        match unsafe_spec.compile(&toy_registry()) {
            Err(PlanSpecError::UnsafePure { node, stage, .. }) => {
                assert_eq!(node, 0);
                assert_eq!(stage, "spec_make_a");
            }
            Err(e) => panic!("expected UnsafePure, got {e:?}"),
            Ok(_) => panic!("expected UnsafePure, got Ok"),
        }

        let mut safe_spec = unsafe_spec;
        safe_spec.nodes[0].stage = "spec_safe_make_a".into();
        let compiled = safe_spec.compile(&toy_registry()).expect("certified stage");
        assert!(compiled.nodes[0].pure);

        let mut nondeterministic = linear_spec();
        nondeterministic.nodes.truncate(1);
        nondeterministic.edges.clear();
        nondeterministic.nodes[0].stage = "spec_nondeterministic_safe_make_a".into();
        nondeterministic.nodes[0].pure = true;
        assert!(matches!(
            nondeterministic.compile(&toy_registry()),
            Err(PlanSpecError::UnsafePure { .. })
        ));
    }

    #[test]
    fn typed_condition_gate_compiles_separately_from_data_edges() {
        let plan = condition_spec("spec_decide", "spec_make_a")
            .compile(&toy_registry())
            .expect("typed condition gate");

        assert_eq!(plan.n_edges(), 0, "control relation is not a data edge");
        assert_eq!(
            plan.condition_gates().collect::<Vec<_>>(),
            vec![(0, 1, true)]
        );
        let graph = plan.graph_structure().unwrap();
        assert!(graph.edges.is_empty());
        assert_eq!(graph.condition_gates.len(), 1);
        assert_eq!(graph.condition_gates[0].condition, 0);
        assert_eq!(graph.condition_gates[0].target, 1);
        assert!(graph.condition_gates[0].when);
    }

    #[test]
    fn condition_gate_rejects_non_decision_stage() {
        match condition_spec("spec_make_a", "spec_safe_make_a").compile(&toy_registry()) {
            Err(PlanSpecError::BadCondition { detail, .. }) => {
                assert!(
                    detail.contains("expected 'blut.branch-decision'"),
                    "{detail}"
                );
            }
            Err(e) => panic!("expected BadCondition, got {e:?}"),
            Ok(_) => panic!("expected BadCondition, got Ok"),
        }
    }

    #[test]
    fn condition_gate_rejects_out_of_range_node() {
        let mut spec = condition_spec("spec_decide", "spec_make_a");
        spec.condition_gates[0].target = 9;
        match spec.compile(&toy_registry()) {
            Err(PlanSpecError::BadCondition { detail, .. }) => {
                assert!(detail.contains("out of range"), "{detail}");
            }
            Err(e) => panic!("expected BadCondition, got {e:?}"),
            Ok(_) => panic!("expected BadCondition, got Ok"),
        }
    }

    #[test]
    fn condition_gate_rejects_reconvergent_branch() {
        let mut spec = condition_spec("spec_decide", "spec_make_a");
        spec.nodes.push(SpecNode {
            stage: "spec_make_a".into(),
            args: serde_json::json!({}),
            retry: None,
            timeout: None,
            priority: None,
            pure: false,
        });
        spec.nodes.push(SpecNode {
            stage: "spec_a_to_b".into(),
            args: serde_json::json!({}),
            retry: None,
            timeout: None,
            priority: None,
            pure: false,
        });
        spec.edges = vec![(1, 3), (2, 3)];

        match spec.compile(&toy_registry()) {
            Err(PlanSpecError::BadCondition { detail, .. }) => {
                assert!(detail.contains("reconvergent node 3"), "{detail}");
            }
            Err(e) => panic!("expected BadCondition, got {e:?}"),
            Ok(_) => panic!("expected BadCondition, got Ok"),
        }
    }

    #[test]
    fn planspec_v1_1_additive() {
        use crate::framework::retry::{Backoff, RetryOn, RetryPolicy, StageTimeout};
        use std::time::Duration;
        let reg = toy_registry();

        // (1) A v1 JSON (no retry/timeout) parses; the new fields default None.
        let v1: PlanSpec =
            serde_json::from_str(r#"{"name":"c","nodes":[{"stage":"spec_make_a"}],"edges":[]}"#)
                .unwrap();
        assert!(v1.nodes[0].retry.is_none() && v1.nodes[0].timeout.is_none());

        // (2) v1.1 with UNSET retry/timeout omits them from the wire
        // (skip_serializing_if) and hashes byte-identically to the v1 plan.
        let base = PlanSpec {
            name: "c".into(),
            nodes: vec![SpecNode {
                stage: "spec_make_a".into(),
                args: serde_json::Value::Null,
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: PLAN_SPEC_VERSION,
        };
        assert_eq!(
            v1.canonical_bytes(),
            base.canonical_bytes(),
            "absent retry/timeout hashes identically to v1"
        );
        let json = serde_json::to_string(&base).unwrap();
        assert!(
            !json.contains("retry") && !json.contains("timeout"),
            "unset fields are omitted from the wire: {json}"
        );

        // (3) Setting retry/timeout leaves the node's CACHE-KEY inputs (stage +
        // args) byte-identical — only the plan fingerprint moves.
        let mut with = base.clone();
        with.nodes[0].retry = Some(RetryPolicy {
            max_attempts: 3,
            backoff: Backoff::Fixed(Duration::from_secs(1)),
            retry_on: RetryOn::Transient,
        });
        with.nodes[0].timeout = Some(StageTimeout {
            soft: None,
            hard: Some(Duration::from_secs(60)),
        });
        assert_eq!(with.nodes[0].stage, base.nodes[0].stage);
        assert_eq!(with.nodes[0].args, base.nodes[0].args);
        assert_ne!(
            with.canonical_bytes(),
            base.canonical_bytes(),
            "the plan fingerprint reflects retry/timeout"
        );

        // (4) compile threads them onto the executor's PlanNode; the default
        // plan leaves them None (the stage's own RETRY/TIMEOUT default).
        let compiled = with.compile(&reg).unwrap();
        assert_eq!(compiled.nodes[0].retry.unwrap().max_attempts, 3);
        assert_eq!(
            compiled.nodes[0].timeout.unwrap().hard,
            Some(Duration::from_secs(60))
        );
        let compiled_default = base.compile(&reg).unwrap();
        assert!(compiled_default.nodes[0].retry.is_none());
        assert!(compiled_default.nodes[0].timeout.is_none());
    }

    #[test]
    fn dag_opt_priority_planspec_additive() {
        // ADR 0102 pass #4: `priority` is an additive PlanSpec field with the same
        // cache-safety contract as retry/timeout.
        let reg = toy_registry();

        // (1) A plan without `priority` parses; the field defaults None.
        let v1: PlanSpec =
            serde_json::from_str(r#"{"name":"p","nodes":[{"stage":"spec_make_a"}],"edges":[]}"#)
                .unwrap();
        assert!(v1.nodes[0].priority.is_none());

        // (2) UNSET priority is omitted from the wire (skip_serializing_if) and
        // hashes byte-identically to the priority-absent plan.
        let base = PlanSpec {
            name: "p".into(),
            nodes: vec![SpecNode {
                stage: "spec_make_a".into(),
                args: serde_json::Value::Null,
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: PLAN_SPEC_VERSION,
        };
        assert_eq!(v1.canonical_bytes(), base.canonical_bytes());
        let json = serde_json::to_string(&base).unwrap();
        assert!(!json.contains("priority"), "unset priority omitted: {json}");

        // (3) Setting priority leaves the node's cache-key inputs (stage + args)
        // byte-identical; only the plan fingerprint moves.
        let mut with = base.clone();
        with.nodes[0].priority = Some(42);
        assert_eq!(with.nodes[0].stage, base.nodes[0].stage);
        assert_eq!(with.nodes[0].args, base.nodes[0].args);
        assert_ne!(with.canonical_bytes(), base.canonical_bytes());

        // (4) compile threads priority onto the executor's PlanNode; the default
        // plan leaves it None (neutral).
        let compiled = with.compile(&reg).unwrap();
        assert_eq!(compiled.nodes[0].priority, Some(42));
        assert!(base.compile(&reg).unwrap().nodes[0].priority.is_none());
    }

    #[test]
    fn canonical_bytes_are_key_order_stable() {
        // Same graph, args keys in different source order -> identical bytes.
        let a = PlanSpec {
            name: "n".into(),
            nodes: vec![SpecNode {
                stage: "s".into(),
                args: serde_json::json!({ "x": 1, "y": 2 }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: 1,
        };
        let b = PlanSpec {
            name: "n".into(),
            nodes: vec![SpecNode {
                stage: "s".into(),
                args: serde_json::json!({ "y": 2, "x": 1 }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: 1,
        };
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        // A real difference DOES change the bytes.
        let c = PlanSpec {
            name: "n".into(),
            nodes: vec![SpecNode {
                stage: "s".into(),
                args: serde_json::json!({ "x": 9, "y": 2 }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: 1,
        };
        assert_ne!(a.canonical_bytes(), c.canonical_bytes());
    }

    #[test]
    fn compile_unknown_stage_names_it() {
        let spec = PlanSpec {
            name: "x".into(),
            nodes: vec![SpecNode {
                stage: "no_such".into(),
                args: serde_json::json!({}),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: 1,
        };
        match spec.compile(&Registry::new()) {
            Err(PlanSpecError::UnknownStage { stage, .. }) => assert_eq!(stage, "no_such"),
            Err(e) => panic!("expected UnknownStage, got {e:?}"),
            Ok(_) => panic!("expected UnknownStage, got Ok"),
        }
    }

    #[test]
    fn compile_kind_break_is_reported() {
        // make_a -> make_a: the second is a graph-input stage fed a
        // predecessor's output; kind break (A != ()).
        let spec = PlanSpec {
            name: "x".into(),
            nodes: vec![
                SpecNode {
                    stage: "spec_make_a".into(),
                    args: serde_json::json!({}),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                },
                SpecNode {
                    stage: "spec_make_a".into(),
                    args: serde_json::json!({}),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                },
            ],
            edges: vec![(0, 1)],
            expansions: Vec::new(),
            condition_gates: Vec::new(),
            version: 1,
        };
        match spec.compile(&toy_registry()) {
            Err(PlanSpecError::Plan {
                source: PlanError::KindBreak { in_kind, .. },
                ..
            }) => assert_eq!(in_kind, "()"),
            Err(e) => panic!("expected KindBreak, got {e:?}"),
            Ok(_) => panic!("expected KindBreak, got Ok"),
        }
    }

    #[test]
    fn compile_success_shape() {
        let plan = linear_spec().compile(&toy_registry()).unwrap();
        assert_eq!(plan.n_nodes(), 2);
        assert_eq!(plan.n_edges(), 1);
        assert_eq!(plan.topo_order().unwrap(), vec![0, 1]);
        // Provenance blob is stamped.
        assert_eq!(plan.recipe_args()["plan_spec"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn compiled_spec_executes_and_second_run_is_cached() {
        use crate::framework::executor::{ExecCtx, execute_plan};
        let reg = toy_registry();
        let td = tempfile::tempdir().unwrap();

        let plan1 = linear_spec().compile(&reg).unwrap();
        let r1 = execute_plan(plan1, ExecCtx::new(td.path().to_path_buf()))
            .await
            .expect("first run");
        assert_eq!(r1.n_stages, 2);

        // Second run in the SAME job dir: both stages hit the content cache
        // (topology was never part of a stage cache key — a graph-built plan
        // caches exactly like a chain-built one).
        let plan2 = linear_spec().compile(&reg).unwrap();
        let r2 = execute_plan(plan2, ExecCtx::new(td.path().to_path_buf()))
            .await
            .expect("second run");
        assert_eq!(r2.n_cache_hits, 2, "both stages should hit on re-run");
    }

    // ---- map_output expansions (ADR 0078) ----

    /// A spec: sharder (() -> list<item>) with a map over it whose template
    /// consumes an item (item -> A).
    fn map_spec() -> PlanSpec {
        PlanSpec {
            name: "m".into(),
            nodes: vec![SpecNode {
                stage: "spec_sharder".into(),
                args: serde_json::json!({}),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: vec![MapSpec {
                parent: 0,
                template: PlanSpec {
                    name: "tmpl".into(),
                    nodes: vec![SpecNode {
                        stage: "spec_item_to_a".into(),
                        args: serde_json::json!({}),
                        retry: None,
                        timeout: None,
                        priority: None,
                        pure: false,
                    }],
                    edges: vec![],
                    expansions: Vec::new(),
                    condition_gates: Vec::new(),
                    version: PLAN_SPEC_VERSION,
                },
                label: Some("shard".into()),
            }],
            condition_gates: Vec::new(),
            version: PLAN_SPEC_VERSION,
        }
    }

    #[test]
    fn map_compiles_and_attaches_one_expansion() {
        let plan = map_spec().compile(&toy_registry()).unwrap();
        assert_eq!(plan.n_nodes(), 1);
        assert_eq!(plan.expansions().len(), 1);
        let exp = &plan.expansions()[0];
        assert_eq!(exp.parent, 0);
        assert_eq!(exp.label.as_deref(), Some("shard"));
        assert_eq!(exp.template.elem_kind, "spec.item");
        assert_eq!(exp.template.root, 0);
    }

    #[test]
    fn map_over_non_list_parent_is_rejected() {
        // Parent make_a outputs A (not a list).
        let spec = PlanSpec {
            name: "x".into(),
            nodes: vec![SpecNode {
                stage: "spec_make_a".into(),
                args: serde_json::json!({}),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: vec![MapSpec {
                parent: 0,
                template: PlanSpec {
                    name: "t".into(),
                    nodes: vec![SpecNode {
                        stage: "spec_item_to_a".into(),
                        args: serde_json::json!({}),
                        retry: None,
                        timeout: None,
                        priority: None,
                        pure: false,
                    }],
                    edges: vec![],
                    expansions: Vec::new(),
                    condition_gates: Vec::new(),
                    version: PLAN_SPEC_VERSION,
                },
                label: None,
            }],
            condition_gates: Vec::new(),
            version: PLAN_SPEC_VERSION,
        };
        match spec.compile(&toy_registry()) {
            Err(PlanSpecError::BadMap { detail, .. }) => {
                assert!(detail.contains("does not output a list"), "{detail}");
            }
            Err(e) => panic!("expected BadMap, got {e:?}"),
            Ok(_) => panic!("expected BadMap, got Ok"),
        }
    }

    #[test]
    fn map_template_root_wrong_kind_is_rejected() {
        // Template root make_a takes `()`, not the element kind `spec.item`.
        let spec = PlanSpec {
            name: "x".into(),
            nodes: vec![SpecNode {
                stage: "spec_sharder".into(),
                args: serde_json::json!({}),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            }],
            edges: vec![],
            expansions: vec![MapSpec {
                parent: 0,
                template: PlanSpec {
                    name: "t".into(),
                    nodes: vec![SpecNode {
                        stage: "spec_make_a".into(),
                        args: serde_json::json!({}),
                        retry: None,
                        timeout: None,
                        priority: None,
                        pure: false,
                    }],
                    edges: vec![],
                    expansions: Vec::new(),
                    condition_gates: Vec::new(),
                    version: PLAN_SPEC_VERSION,
                },
                label: None,
            }],
            condition_gates: Vec::new(),
            version: PLAN_SPEC_VERSION,
        };
        assert!(matches!(
            spec.compile(&toy_registry()),
            Err(PlanSpecError::BadMap { .. })
        ));
    }

    #[test]
    fn map_parent_out_of_range_is_rejected() {
        let mut spec = map_spec();
        spec.expansions[0].parent = 9;
        match spec.compile(&toy_registry()) {
            Err(PlanSpecError::BadMap { detail, .. }) => {
                assert!(detail.contains("out of range"), "{detail}")
            }
            Err(e) => panic!("expected BadMap, got {e:?}"),
            Ok(_) => panic!("expected BadMap, got Ok"),
        }
    }

    #[test]
    fn nested_map_template_is_rejected() {
        let mut spec = map_spec();
        // Give the template its own (empty-parent) expansion → nested map.
        spec.expansions[0].template.expansions = vec![MapSpec {
            parent: 0,
            template: PlanSpec {
                name: "inner".into(),
                nodes: vec![SpecNode {
                    stage: "spec_item_to_a".into(),
                    args: serde_json::json!({}),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                }],
                edges: vec![],
                expansions: Vec::new(),
                condition_gates: Vec::new(),
                version: PLAN_SPEC_VERSION,
            },
            label: None,
        }];
        match spec.compile(&toy_registry()) {
            Err(PlanSpecError::BadMap { detail, .. }) => {
                assert!(detail.contains("nested"), "{detail}")
            }
            Err(e) => panic!("expected BadMap, got {e:?}"),
            Ok(_) => panic!("expected BadMap, got Ok"),
        }
    }

    #[test]
    fn pure_map_template_node_is_rejected_in_v1() {
        let mut spec = map_spec();
        spec.expansions[0].template.nodes[0].pure = true;

        match spec.compile(&toy_registry()) {
            Err(PlanSpecError::BadMap { detail, .. }) => {
                assert!(detail.contains("pure template nodes"), "{detail}")
            }
            Err(error) => panic!("expected BadMap, got {error:?}"),
            Ok(_) => panic!("expected BadMap, got Ok"),
        }
    }

    // ---- map_output runtime execution (ADR 0078, executor 3/n) ----

    #[tokio::test]
    async fn map_output_fans_out_one_child_per_element() {
        use crate::framework::executor::{ExecCtx, execute_plan};
        let reg = toy_registry();
        let td = tempfile::tempdir().unwrap();
        // sharder (width-2 list) + a map(item -> A): 1 + 2 spawned children.
        let plan = map_spec().compile(&reg).unwrap();
        let r = execute_plan(plan, ExecCtx::new(td.path().to_path_buf()))
            .await
            .expect("run");
        assert_eq!(r.n_stages, 3, "sharder + one child per element");
        assert!(r.warnings.is_empty());
    }

    #[tokio::test]
    async fn map_output_empty_list_spawns_nothing() {
        use crate::framework::executor::{ExecCtx, execute_plan};
        let mut spec = map_spec();
        spec.nodes[0].args = serde_json::json!({ "width": 0 });
        let reg = toy_registry();
        let td = tempfile::tempdir().unwrap();
        let plan = spec.compile(&reg).unwrap();
        let r = execute_plan(plan, ExecCtx::new(td.path().to_path_buf()))
            .await
            .expect("run");
        assert_eq!(r.n_stages, 1, "empty list → only the sharder runs");
    }

    #[tokio::test]
    async fn oversized_required_map_shard_fails_instead_of_being_dropped() {
        use crate::framework::error::PlanError;
        use crate::framework::executor::{ExecCtx, execute_plan};
        OVERSIZED_MAP_SHARD_RAN.store(0, Ordering::SeqCst);
        let mut spec = map_spec();
        spec.expansions[0].template.nodes[0].stage = "spec_oversized_item_to_a".into();
        let reg = toy_registry();
        let td = tempfile::tempdir().unwrap();
        let error = execute_plan(
            spec.compile(&reg).unwrap(),
            ExecCtx::new(td.path().to_path_buf()).with_memory_budget(4),
        )
        .await
        .expect_err("dropping a required map shard would produce a wrong answer");
        match error {
            PlanError::Other(message) => {
                assert!(
                    message.contains("runtime spawn requires 8 GiB"),
                    "{message}"
                );
                assert!(message.contains("reserved only 4 GiB"), "{message}");
            }
            other => panic!("expected an admission PlanError::Other, got {other:?}"),
        }
        assert_eq!(OVERSIZED_MAP_SHARD_RAN.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn map_output_children_cache_hit_on_rerun() {
        use crate::framework::executor::{ExecCtx, execute_plan};
        let reg = toy_registry();
        let td = tempfile::tempdir().unwrap();
        let r1 = execute_plan(
            map_spec().compile(&reg).unwrap(),
            ExecCtx::new(td.path().to_path_buf()),
        )
        .await
        .expect("run1");
        assert_eq!(r1.n_stages, 3);
        // Second run in the SAME job dir: the sharder AND both shards hit the
        // cache. The shards' keys are stable because each element's logical
        // hash derives from the (deterministic) parent's logical hash + index.
        let r2 = execute_plan(
            map_spec().compile(&reg).unwrap(),
            ExecCtx::new(td.path().to_path_buf()),
        )
        .await
        .expect("run2");
        assert_eq!(r2.n_cache_hits, 3, "sharder + both shards hit on re-run");
    }
}
