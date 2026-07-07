// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The Starlark globals a `.star` script may call (ADR 0078).
//!
//! The ONLY mutation surface is the `add()` builtin; fork = two `add`s
//! sharing `after`, merge = a list `after`, compile-time map = a plain `for`
//! loop. Node handles are plain Starlark integers (a node's dense index), so
//! there is no custom value type. The draft lives in [`DslStore`], reached
//! through `eval.extra` (the documented starlark-rust state-sharing pattern).

// The `#[starlark_module]` macro and the `ProvidesStaticType` derive both emit
// `unsafe impl` blocks (the traits are unsafe by design — they assert ABI /
// static-type-id contracts the macro/derive uphold). The crate denies
// `unsafe_code` globally; scope the allow to this glue module.
#![allow(unsafe_code)]

use std::cell::RefCell;

use anyhow::{Context, anyhow};
use starlark::any::ProvidesStaticType;
use starlark::environment::GlobalsBuilder;
use starlark::eval::Evaluator;
use starlark::starlark_module;
use starlark::values::Value;

use super::builder::PlanDraft;

/// Evaluator-`extra` state: the plan draft the `add()` builtin appends to.
#[derive(Debug, Default, ProvidesStaticType)]
pub(crate) struct DslStore(pub RefCell<PlanDraft>);

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
        let id = store.0.borrow_mut().add(stage, args_json, &after_ids);
        // Node count is bounded by the tick limit; a plan with > i32::MAX
        // nodes is impossible in practice, but guard the cast anyway.
        i32::try_from(id).map_err(|_| anyhow!("plan too large: node id {id} overflows i32"))
    }
}
