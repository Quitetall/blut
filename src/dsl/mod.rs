// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Starlark front-end for authoring [`PlanSpec`]s (ADR 0078, behind the
//! `dsl` feature).
//!
//! A `.star` script defines `build(args)` and calls the `add()` builtin to
//! compose REGISTERED stages into a graph. Evaluation is **hermetic** — no
//! `load()`, no file/network/clock/random access (the Starlark standard
//! library has none, and we disable `load` in the dialect) — so a script's
//! emitted `PlanSpec` is a pure function of `(source, args)` and therefore
//! content-hashable. A step/heap cap turns a pathological script into an
//! error instead of a hang. The interpreter runs ONCE here, before launch;
//! the executor never re-enters it.
//!
//! Example (`demo.star`):
//! ```python
//! def build(args):
//!     root = add("prepare_data", {"corpus": args["corpus"]})
//!     heads = []
//!     for tier in args["tiers"]:
//!         heads.append(add("train_model", {"tier": tier}, after=root))
//!     add("compare_report", after=heads)   # list after -> merge
//! ```

mod builder;
mod globals;

use starlark::environment::{GlobalsBuilder, Module};
use starlark::eval::Evaluator;
use starlark::syntax::{AstModule, Dialect};
use starlark::values::Value;
use starlark::values::dict::AllocDict;

use crate::framework::artifact::ContentHash;
use crate::framework::cache::CacheHandle;
use crate::framework::plan_spec::PlanSpec;
use globals::{DslStore, dsl_globals};

/// Ceiling on evaluator "ticks" (roughly, executed statements) — turns an
/// accidental infinite loop into an error rather than a hang. Generous: a
/// real plan-authoring script does thousands of ops at most.
const MAX_TICKS: u64 = 10_000_000;
/// Ceiling on the script's heap (bytes). Bounds a runaway allocation.
const MAX_HEAP_BYTES: usize = 256 * 1024 * 1024;

/// Failure evaluating a `.star` script into a [`PlanSpec`].
#[derive(Debug, thiserror::Error)]
pub enum DslError {
    /// The script did not parse (syntax error, or a forbidden `load()`).
    #[error("starlark parse error in {path}: {msg}")]
    Parse { path: String, msg: String },
    /// The script raised at evaluation time (an `add()` misuse, a runtime
    /// error, or the tick/heap cap tripping).
    #[error("starlark evaluation error in {path}: {msg}")]
    Eval { path: String, msg: String },
    /// The script has no `build(args)` entry point.
    #[error("script {path} defines no `build(args)` function")]
    MissingBuild { path: String },
}

/// Evaluate a `.star` script into a [`PlanSpec`]. `path_label` is used for
/// error messages and to derive the plan name (the file stem); `args` is
/// passed to the script's `build(args)`.
pub fn evaluate_script(
    source: &str,
    path_label: &str,
    args: &serde_json::Value,
) -> Result<PlanSpec, DslError> {
    // Hermetic dialect: `load()` is a parse error (no cross-file imports).
    let dialect = Dialect {
        enable_load: false,
        ..Dialect::Standard
    };
    let ast =
        AstModule::parse(path_label, source.to_owned(), &dialect).map_err(|e| DslError::Parse {
            path: path_label.to_string(),
            msg: e.to_string(),
        })?;
    // Standard globals (pure: len/range/dict/sorted/… — no I/O) + our add().
    let starlark_globals = GlobalsBuilder::standard().with(dsl_globals).build();
    let store = DslStore::default();
    let name = plan_name_from(path_label);

    let run = Module::with_temp_heap(|module| -> Result<(), DslError> {
        let mut eval = Evaluator::new(&module);
        // Fail-safe caps (ignore the Result: setting a cap can't fail here).
        let _ = eval.set_max_tick_count(MAX_TICKS);
        let _ = eval.set_max_heap_size(MAX_HEAP_BYTES);
        eval.extra = Some(&store);
        // Run top level (defines `build`).
        eval.eval_module(ast, &starlark_globals)
            .map_err(|e| DslError::Eval {
                path: path_label.to_string(),
                msg: e.to_string(),
            })?;
        // Fetch and invoke build(args).
        let build = module.get("build").ok_or_else(|| DslError::MissingBuild {
            path: path_label.to_string(),
        })?;
        let args_val = json_to_value(module.heap(), args);
        eval.eval_function(build, &[args_val], &[])
            .map_err(|e| DslError::Eval {
                path: path_label.to_string(),
                msg: e.to_string(),
            })?;
        Ok(())
    });
    run?;

    Ok(store.0.into_inner().into_spec(name))
}

/// Provenance fingerprint (ADR 0078): a content hash over the script source,
/// the canonical args, and the emitted spec — so lineage records exactly
/// what produced a plan, and a change to any of the three changes the id.
pub fn script_fingerprint(source: &str, args: &serde_json::Value, spec: &PlanSpec) -> ContentHash {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"blut-star-v1");
    // Length-prefix each variable-length part so concatenation is unambiguous.
    buf.extend_from_slice(&(source.len() as u64).to_le_bytes());
    buf.extend_from_slice(source.as_bytes());
    let canon_args = CacheHandle::canonical_json_bytes(args);
    buf.extend_from_slice(&(canon_args.len() as u64).to_le_bytes());
    buf.extend_from_slice(&canon_args);
    buf.extend_from_slice(&spec.canonical_bytes());
    ContentHash::of_bytes(&buf)
}

/// Plan name from a script path: the file stem (`recipes/train.star` →
/// `train`), falling back to the whole label.
fn plan_name_from(path_label: &str) -> String {
    std::path::Path::new(path_label)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path_label)
        .to_string()
}

/// Recursively allocate a `serde_json::Value` as a Starlark value on `heap`
/// (so the script's `build(args)` sees native dicts/lists/scalars).
fn json_to_value<'v>(heap: starlark::values::Heap<'v>, v: &serde_json::Value) -> Value<'v> {
    match v {
        serde_json::Value::Null => Value::new_none(),
        serde_json::Value::Bool(b) => Value::new_bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                heap.alloc(i)
            } else {
                // Non-integer (or > i64) number → f64 (JSON's only other numeric).
                heap.alloc(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(s) => heap.alloc(s.as_str()),
        serde_json::Value::Array(a) => {
            heap.alloc(a.iter().map(|x| json_to_value(heap, x)).collect::<Vec<_>>())
        }
        serde_json::Value::Object(o) => {
            let pairs: Vec<(Value<'v>, Value<'v>)> = o
                .iter()
                .map(|(k, val)| (heap.alloc(k.as_str()), json_to_value(heap, val)))
                .collect();
            heap.alloc(AllocDict(pairs))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn linear_chain_script_builds_expected_spec() {
        let src = r#"
def build(args):
    a = add("make")
    b = add("step", {"n": args["n"]}, after=a)
    add("finish", after=b)
"#;
        let spec = evaluate_script(src, "chain.star", &json!({ "n": 3 })).unwrap();
        assert_eq!(spec.name, "chain");
        assert_eq!(spec.nodes.len(), 3);
        assert_eq!(spec.nodes[0].stage, "make");
        assert_eq!(spec.nodes[1].args, json!({ "n": 3 }));
        assert_eq!(spec.edges, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn loop_fans_out_and_list_after_merges() {
        let src = r#"
def build(args):
    root = add("prep")
    heads = []
    for t in args["tiers"]:
        heads.append(add("train", {"tier": t}, after=root))
    add("merge", after=heads)
"#;
        let spec = evaluate_script(src, "fan.star", &json!({ "tiers": [1, 2, 3] })).unwrap();
        // prep + 3 trains + merge = 5 nodes.
        assert_eq!(spec.nodes.len(), 5);
        // Each train hangs off root; merge takes all three trains in order.
        assert_eq!(
            spec.edges,
            vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
        );
        assert_eq!(spec.nodes[2].args, json!({ "tier": 2 }));
    }

    #[test]
    fn topology_is_deterministic_across_runs() {
        let src = r#"
def build(args):
    a = add("a")
    add("b", after=a)
"#;
        let s1 = evaluate_script(src, "d.star", &json!({})).unwrap();
        let s2 = evaluate_script(src, "d.star", &json!({})).unwrap();
        assert_eq!(s1.canonical_bytes(), s2.canonical_bytes());
        // Same source+args -> same fingerprint; a real arg change moves it.
        let f1 = script_fingerprint(src, &json!({ "x": 1 }), &s1);
        let f2 = script_fingerprint(src, &json!({ "x": 1 }), &s2);
        assert_eq!(f1, f2);
        let f3 = script_fingerprint(src, &json!({ "x": 2 }), &s1);
        assert_ne!(f1, f3);
    }

    #[test]
    fn load_statement_is_rejected() {
        let src = r#"
load("other.star", "thing")
def build(args):
    add("a")
"#;
        match evaluate_script(src, "bad.star", &json!({})) {
            Err(DslError::Parse { .. }) => {}
            other => panic!("expected a parse error rejecting load(), got {other:?}"),
        }
    }

    #[test]
    fn missing_build_is_a_clear_error() {
        let src = "x = 1\n";
        match evaluate_script(src, "nobuild.star", &json!({})) {
            Err(DslError::MissingBuild { .. }) => {}
            other => panic!("expected MissingBuild, got {other:?}"),
        }
    }

    #[test]
    fn runtime_error_surfaces_as_eval_error() {
        // Referencing an undefined name is a runtime evaluation error.
        let src = r#"
def build(args):
    add(nonexistent_variable)
"#;
        match evaluate_script(src, "err.star", &json!({})) {
            Err(DslError::Eval { .. }) => {}
            other => panic!("expected Eval error, got {other:?}"),
        }
    }

    #[test]
    fn demo_fixture_builds_the_expected_topology() {
        // The committed example recipe, evaluated with representative args.
        let src = include_str!("../../examples/recipes/demo.star");
        let spec = evaluate_script(
            src,
            "demo.star",
            &json!({
                "corpus": "tuh",
                "tiers": [5, 7],
            }),
        )
        .unwrap();
        assert_eq!(spec.name, "demo");
        // prepare + 2 train + compare = 4 nodes.
        assert_eq!(spec.nodes.len(), 4);
        assert_eq!(spec.nodes[0].stage, "prepare_data");
        assert_eq!(spec.nodes[0].args, json!({ "corpus": "tuh" }));
        assert_eq!(spec.nodes[3].stage, "compare_report");
        // train nodes fan off prepare; compare merges both trains.
        assert_eq!(spec.edges, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    }

    proptest::proptest! {
        // Any list of tier ints drives the fan-out: N tiers -> N+2 nodes, and
        // evaluation never panics for arbitrary arg values.
        #[test]
        fn fan_out_width_tracks_args(tiers in proptest::collection::vec(0i64..1000, 0..20)) {
            let src = include_str!("../../examples/recipes/demo.star");
            let spec = evaluate_script(
                src,
                "demo.star",
                &json!({ "corpus": "c", "tiers": tiers.clone() }),
            )
            .expect("demo.star evaluates for any tier list");
            proptest::prop_assert_eq!(spec.nodes.len(), tiers.len() + 2);
        }
    }

    #[test]
    fn infinite_loop_hits_the_tick_cap_instead_of_hanging() {
        let src = r#"
def build(args):
    i = 0
    for _ in range(100000000000):
        i = i + 1
    add("never")
"#;
        // Must terminate with an error, not hang, thanks to MAX_TICKS.
        match evaluate_script(src, "loop.star", &json!({})) {
            Err(DslError::Eval { .. }) => {}
            other => panic!("expected Eval error from the tick cap, got {other:?}"),
        }
    }
}
