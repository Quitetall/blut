// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The Starlark globals a `.star` script may call (ADR 0078).
//!
//! Mutation surface: `add()` composes stages (fork = two `add`s sharing
//! `after`, merge = a list `after`, compile-time map = a plain `for` loop);
//! `map_output()` declares a RUNTIME fan-out over a list-producing node. Node
//! handles are plain Starlark integers (a node's dense index), so there is no
//! custom value type. The drafts live in [`DslStore`] as a STACK (reached
//! through `eval.extra`, the documented starlark-rust state-sharing pattern):
//! the bottom is the main plan; `map_output` pushes a template scope, runs its
//! body, then pops it and attaches the result as a `MapSpec`.

// The `#[starlark_module]` macro and the `ProvidesStaticType` derive both emit
// `unsafe impl` blocks (the traits are unsafe by design — they assert ABI /
// static-type-id contracts the macro/derive uphold). The crate denies
// `unsafe_code` globally; scope the allow to this glue module.
#![allow(unsafe_code)]

use std::cell::RefCell;

use anyhow::{Context, anyhow};
use blut::framework::plan_spec::MapSpec;
use starlark::any::ProvidesStaticType;
use starlark::environment::GlobalsBuilder;
use starlark::eval::Evaluator;
use starlark::starlark_module;
use starlark::values::Value;
use starlark::values::none::NoneType;

use crate::builder::PlanDraft;

/// Evaluator-`extra` state: a STACK of plan drafts. The bottom is the main
/// plan; `map_output` pushes a template scope so `add()` calls inside its body
/// build the template, then pops it. Invariant: always ≥1 draft.
#[derive(Debug, ProvidesStaticType)]
pub(crate) struct DslStore(pub RefCell<Vec<PlanDraft>>);

impl Default for DslStore {
    fn default() -> Self {
        DslStore(RefCell::new(vec![PlanDraft::default()]))
    }
}

/// Interpret an `after=` argument as a list of predecessor node ids:
/// `None` → no predecessors (a graph source); an int → one predecessor; a
/// list/iterable of ints → a merge in that order. A negative or non-int
/// value is a script error (handles are always the non-negative ints
/// `add()` returned).
fn parse_after<'v>(
    after: Option<Value<'v>>,
    eval: &mut Evaluator<'v, '_, '_>,
) -> anyhow::Result<Vec<u32>> {
    let Some(v) = after else {
        return Ok(Vec::new());
    };
    if let Some(i) = v.unpack_i32() {
        let id = u32::try_from(i)
            .map_err(|_| anyhow!("after: node handle must be non-negative, got {i}"))?;
        return Ok(vec![id]);
    }
    // Otherwise it must be an iterable of ints.
    let mut ids = Vec::new();
    let it = v
        .iterate(eval.heap())
        .map_err(|e| anyhow!("after: expected a node handle (int) or a list of handles: {e}"))?;
    for el in it {
        let i = el
            .unpack_i32()
            .ok_or_else(|| anyhow!("after: every element must be a node handle (int)"))?;
        ids.push(
            u32::try_from(i)
                .map_err(|_| anyhow!("after: node handle must be non-negative, got {i}"))?,
        );
    }
    Ok(ids)
}

#[starlark_module]
pub(crate) fn dsl_globals(builder: &mut GlobalsBuilder) {
    /// add(stage, args=None, *, after=None) -> int
    ///
    /// Append a stage node and return its handle (an int). `args` is the
    /// stage's args (a dict, or omitted for none — matching a declarative
    /// recipe's null default). `after` wires predecessors: omit for a graph
    /// source, pass one handle for a linear step, or a list of handles for a
    /// merge (list order = the merge's tuple element order).
    fn add<'v>(
        #[starlark(require = pos)] stage: String,
        args: Option<Value<'v>>,
        #[starlark(require = named)] after: Option<Value<'v>>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<i32> {
        let after_ids = parse_after(after, eval)?;
        // Convert args to JSON now (Starlark values don't outlive eval).
        let args_json = match args {
            Some(v) => v
                .to_json_value()
                .context("add: `args` must be JSON-serializable (dict/list/scalar)")?,
            None => serde_json::Value::Null,
        };
        let store = eval
            .extra
            .and_then(|e| e.downcast_ref::<DslStore>())
            .ok_or_else(|| anyhow!("internal: DSL builder state missing from evaluator"))?;
        // Append to the CURRENT scope (main plan, or a map template if inside
        // a map_output body).
        let id = store
            .0
            .borrow_mut()
            .last_mut()
            .ok_or_else(|| anyhow!("internal: no active plan scope"))?
            .add(stage, args_json, &after_ids);
        // Node count is bounded by the tick limit; a plan with > i32::MAX
        // nodes is impossible in practice, but guard the cast anyway.
        i32::try_from(id).map_err(|_| anyhow!("plan too large: node id {id} overflows i32"))
    }

    /// map_output(parent, body, *, label=None) -> None
    ///
    /// Declare a RUNTIME fan-out (ADR 0078): when node `parent` completes with
    /// a `list` output, the engine runs `body`'s sub-plan once per element,
    /// seeding its single root with the element. `body` is a zero-arg function
    /// whose `add()` calls build the template (its first stage — the one with
    /// no `after` — consumes the element).
    fn map_output<'v>(
        #[starlark(require = pos)] parent: i32,
        #[starlark(require = pos)] body: Value<'v>,
        #[starlark(require = named)] label: Option<String>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let parent = u32::try_from(parent)
            .map_err(|_| anyhow!("map_output: parent handle must be non-negative, got {parent}"))?;
        // `store` is a reference into the OUTER store (in `evaluate_script`),
        // not into `eval` — so it stays valid across `body.invoke_pos(.., eval)`
        // even though that needs `&mut eval`. (`eval.extra` is `Copy`.)
        let store = eval
            .extra
            .and_then(|e| e.downcast_ref::<DslStore>())
            .ok_or_else(|| anyhow!("internal: DSL builder state missing from evaluator"))?;
        // Push a template scope; `add()` inside the body fills it.
        store.0.borrow_mut().push(PlanDraft::default());
        let invoke = eval.eval_function(body, &[], &[]);
        // ALWAYS pop the scope (a failing body must not corrupt the stack).
        let template_draft = store
            .0
            .borrow_mut()
            .pop()
            .ok_or_else(|| anyhow!("internal: map_output template scope vanished"))?;
        invoke.map_err(|e| anyhow!("map_output body failed: {e}"))?;
        let template = template_draft.into_spec("<map-template>".into());
        store
            .0
            .borrow_mut()
            .last_mut()
            .ok_or_else(|| anyhow!("internal: no enclosing plan scope for map_output"))?
            .expansions
            .push(MapSpec {
                parent,
                template,
                label,
            });
        Ok(NoneType)
    }

    /// fed_round(shard, local_train, aggregate, *, shard_args=None,
    ///           train_args=None, aggregate_args=None) -> int
    ///
    /// One federated round (ADR 0080 C4): a convenience over `add` +
    /// `map_output` that wires the canonical shape —
    ///
    ///   shard (→ list of participants) → map_output(local_train × N) → aggregate
    ///
    /// `shard` produces the participant list; the engine fans `local_train` out
    /// once per participant (client-side DP-SGD); `aggregate` runs on the
    /// initiator after the fan-out to FedAvg the deltas + emit the round
    /// artifact. Returns the aggregate node's handle so the caller can chain the
    /// next round (the round LOOP is host-driven dynamic-DAG generation). Every
    /// gradient dispatch is still gated fail-closed at runtime (C1).
    fn fed_round<'v>(
        #[starlark(require = pos)] shard: String,
        #[starlark(require = pos)] local_train: String,
        #[starlark(require = pos)] aggregate: String,
        #[starlark(require = named)] shard_args: Option<Value<'v>>,
        #[starlark(require = named)] train_args: Option<Value<'v>>,
        #[starlark(require = named)] aggregate_args: Option<Value<'v>>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<i32> {
        let to_json = |v: Option<Value<'v>>, what: &str| -> anyhow::Result<serde_json::Value> {
            match v {
                Some(v) => v
                    .to_json_value()
                    .with_context(|| format!("fed_round: `{what}` must be JSON-serializable")),
                None => Ok(serde_json::Value::Null),
            }
        };
        let shard_json = to_json(shard_args, "shard_args")?;
        let train_json = to_json(train_args, "train_args")?;
        let agg_json = to_json(aggregate_args, "aggregate_args")?;

        let store = eval
            .extra
            .and_then(|e| e.downcast_ref::<DslStore>())
            .ok_or_else(|| anyhow!("internal: DSL builder state missing from evaluator"))?;
        let mut drafts = store.0.borrow_mut();
        let scope = drafts
            .last_mut()
            .ok_or_else(|| anyhow!("internal: no active plan scope for fed_round"))?;

        // 1. shard: a graph source producing the participant list.
        let shard_id = scope.add(shard, shard_json, &[]);
        // 2. the fan-out template: one local-train per participant element.
        let mut template = PlanDraft::default();
        template.add(local_train, train_json, &[]);
        scope.expansions.push(MapSpec {
            parent: shard_id,
            template: template.into_spec("<fed-local-train>".into()),
            label: Some("fed-local-train".into()),
        });
        // 3. aggregate on the initiator, after the shard (+ its fan-out).
        let agg_id = scope.add(aggregate, agg_json, &[shard_id]);

        i32::try_from(agg_id).map_err(|_| anyhow!("plan too large: node id {agg_id} overflows i32"))
    }
}
