// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Starlark front-end for authoring [`blut::framework::plan_spec::PlanSpec`]s
//! (ADR 0078).
//!
//! This lives in a SEPARATE crate/binary, not inside the engine, on purpose:
//! `starlark` hard-depends on `serde_json` with the `arbitrary_precision`
//! feature, and Cargo feature unification would turn that on for the WHOLE
//! engine binary — which silently breaks deserialization of the engine's
//! internally-tagged enums (`#[serde(tag = "kind")]`: `StatusUpdate`,
//! `StageEvent`, `TrainSpec`). Keeping Starlark out-of-process means the
//! engine binary never links it, so its serde behavior is untouched. The
//! engine consumes this tool's output through the plain `.json` PlanSpec
//! path (`blut recipe declare foo.json`).
//!
//! A `.star` script defines `build(args)` and calls the `add()` builtin to
//! compose stages by NAME (resolution + kind-checking happen later, in the
//! engine, against a registry). Evaluation is hermetic — no `load()`, no
//! file/network/clock/random access — so a script's emitted `PlanSpec` is a
//! pure function of `(source, args)` and therefore content-hashable. A
//! step/heap cap turns a pathological script into an error, not a hang.

mod builder;
mod globals;

use blut::framework::plan_spec::PlanSpec;
use starlark::environment::{GlobalsBuilder, Module};
use starlark::eval::Evaluator;
use starlark::syntax::{AstModule, Dialect};
use starlark::values::Value;
use starlark::values::dict::AllocDict;

use globals::{DslStore, dsl_globals};

/// Ceiling on evaluator "ticks" (roughly, executed statements) — turns an
/// accidental infinite loop into an error rather than a hang.
const MAX_TICKS: u64 = 10_000_000;
/// Ceiling on the script's heap (bytes) — bounds a runaway allocation.
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
        // Fail-safe caps. Propagate rather than discard: if a cap can't be
        // set, the termination/hermeticity guarantee is void, so refuse.
        let set_caps = eval
            .set_max_tick_count(MAX_TICKS)
            .and_then(|()| eval.set_max_heap_size(MAX_HEAP_BYTES));
        set_caps.map_err(|e| DslError::Eval {
            path: path_label.to_string(),
            msg: format!("failed to set evaluation limits: {e}"),
        })?;
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
        let args_val = json_to_value(module.heap(), args, path_label)?;
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
/// (so the script's `build(args)` sees native dicts/lists/scalars). Errors on
/// a numerically unrepresentable JSON number rather than silently coercing it
/// to NaN.
fn json_to_value<'v>(
    heap: starlark::values::Heap<'v>,
    v: &serde_json::Value,
    path_label: &str,
) -> Result<Value<'v>, DslError> {
    let val = match v {
        serde_json::Value::Null => Value::new_none(),
        serde_json::Value::Bool(b) => Value::new_bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                heap.alloc(i)
            } else if let Some(f) = n.as_f64() {
                heap.alloc(f)
            } else {
                return Err(DslError::Eval {
                    path: path_label.to_string(),
                    msg: format!("arg value {n} is not representable as a Starlark number"),
                });
            }
        }
        serde_json::Value::String(s) => heap.alloc(s.as_str()),
        serde_json::Value::Array(a) => {
            let items = a
                .iter()
                .map(|x| json_to_value(heap, x, path_label))
                .collect::<Result<Vec<_>, _>>()?;
            heap.alloc(items)
        }
        serde_json::Value::Object(o) => {
            let mut pairs: Vec<(Value<'v>, Value<'v>)> = Vec::with_capacity(o.len());
            for (k, val) in o {
                pairs.push((
                    heap.alloc(k.as_str()),
                    json_to_value(heap, val, path_label)?,
                ));
            }
            heap.alloc(AllocDict(pairs))
        }
    };
    Ok(val)
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
        assert_eq!(spec.nodes.len(), 5);
        assert_eq!(
            spec.edges,
            vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)]
        );
        assert_eq!(spec.nodes[2].args, json!({ "tier": 2 }));
    }

    #[test]
    fn topology_is_deterministic_across_runs() {
        let src = "def build(args):\n    a = add(\"a\")\n    add(\"b\", after=a)\n";
        let s1 = evaluate_script(src, "d.star", &json!({})).unwrap();
        let s2 = evaluate_script(src, "d.star", &json!({})).unwrap();
        assert_eq!(s1.canonical_bytes(), s2.canonical_bytes());
    }

    #[test]
    fn load_statement_is_rejected() {
        let src = "load(\"other.star\", \"thing\")\ndef build(args):\n    add(\"a\")\n";
        assert!(matches!(
            evaluate_script(src, "bad.star", &json!({})),
            Err(DslError::Parse { .. })
        ));
    }

    #[test]
    fn missing_build_is_a_clear_error() {
        assert!(matches!(
            evaluate_script("x = 1\n", "nobuild.star", &json!({})),
            Err(DslError::MissingBuild { .. })
        ));
    }

    #[test]
    fn runtime_error_surfaces_as_eval_error() {
        let src = "def build(args):\n    add(nonexistent_variable)\n";
        assert!(matches!(
            evaluate_script(src, "err.star", &json!({})),
            Err(DslError::Eval { .. })
        ));
    }

    #[test]
    fn infinite_loop_hits_the_tick_cap_instead_of_hanging() {
        let src = "def build(args):\n    for _ in range(100000000000):\n        pass\n    add(\"never\")\n";
        assert!(matches!(
            evaluate_script(src, "loop.star", &json!({})),
            Err(DslError::Eval { .. })
        ));
    }

    #[test]
    fn demo_fixture_builds_the_expected_topology() {
        let src = include_str!("../examples/demo.star");
        let spec = evaluate_script(
            src,
            "demo.star",
            &json!({ "corpus": "tuh", "tiers": [5, 7] }),
        )
        .unwrap();
        assert_eq!(spec.name, "demo");
        assert_eq!(spec.nodes.len(), 4);
        assert_eq!(spec.edges, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    }

    proptest::proptest! {
        // Any list of tier ints drives the demo fan-out: N tiers -> N+2 nodes,
        // and evaluation never panics for arbitrary arg values.
        #[test]
        fn fan_out_width_tracks_args(tiers in proptest::collection::vec(0i64..1000, 0..20)) {
            let src = include_str!("../examples/demo.star");
            let spec = evaluate_script(
                src,
                "demo.star",
                &json!({ "corpus": "c", "tiers": tiers.clone() }),
            )
            .expect("demo.star evaluates for any tier list");
            proptest::prop_assert_eq!(spec.nodes.len(), tiers.len() + 2);
        }
    }
}
