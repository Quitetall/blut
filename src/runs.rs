//! `blut runs diff` — compare two jobs' recipe/args PROVENANCE plus a
//! one-line outcome each. Provenance (what was launched) is BLUT's noun;
//! metric-curve comparison stays in wandb (ADR 0034). All data is already
//! on disk per job dir: `recipe.json` (recipe + args) or `spec.json`
//! (legacy TrainSpec), and the job state / last metric in the summary.

use std::collections::BTreeMap;

use crate::error::{Result, TrainError};
use crate::{jobs, paths};

/// Flatten a JSON value into dot-path → scalar-string pairs so two runs'
/// args can be diffed key-by-key. Arrays index by position
/// (`extra_args.0`); objects recurse; scalars stringify.
fn flatten(prefix: &str, v: &serde_json::Value, out: &mut BTreeMap<String, String>) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(&key, val, out);
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                flatten(&format!("{prefix}.{i}"), val, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), scalar(other));
        }
    }
}

fn scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// One run's provenance: where it came from + the flattened args.
struct Provenance {
    source: String, // "recipe.json" | "spec.json" | "(none)"
    /// `recipe` is `Some(name)` for a recipe run, `None` for legacy spec.
    recipe: Option<String>,
    flat: BTreeMap<String, String>,
}

fn load_provenance(job_id: &str) -> Result<Provenance> {
    let dir = paths::job_dir(job_id)?;
    // Prefer the recipe marker (the modern path).
    let recipe_json = dir.join("recipe.json");
    if let Ok(body) = std::fs::read(&recipe_json) {
        let v: serde_json::Value = serde_json::from_slice(&body)
            .map_err(|e| TrainError::other(format!("parse recipe.json: {e}")))?;
        let recipe = v.get("name").and_then(|n| n.as_str()).map(String::from);
        let mut flat = BTreeMap::new();
        if let Some(args) = v.get("args") {
            flatten("", args, &mut flat);
        }
        return Ok(Provenance {
            source: "recipe.json".into(),
            recipe,
            flat,
        });
    }
    // Fall back to the legacy spec.
    let spec_json = dir.join("spec.json");
    if let Ok(body) = std::fs::read(&spec_json) {
        let v: serde_json::Value = serde_json::from_slice(&body)
            .map_err(|e| TrainError::other(format!("parse spec.json: {e}")))?;
        let mut flat = BTreeMap::new();
        flatten("", &v, &mut flat);
        return Ok(Provenance {
            source: "spec.json".into(),
            recipe: None,
            flat,
        });
    }
    Ok(Provenance {
        source: "(none)".into(),
        recipe: None,
        flat: BTreeMap::new(),
    })
}

fn outcome_line(job_id: &str) -> String {
    // Find the summary among all jobs (single-box scale: cheap scan).
    let summary = jobs::list_jobs()
        .ok()
        .and_then(|all| all.into_iter().find(|j| j.id == job_id));
    match summary {
        Some(j) => {
            let metric = match (j.final_loss, j.last_loss, j.last_step) {
                (Some(fl), _, _) => format!("final_loss={fl:.4}"),
                (_, Some(loss), Some(step)) => format!("step={step} loss={loss:.4}"),
                _ => "-".into(),
            };
            format!("{:<10} {metric}", j.state.as_str())
        }
        None => "(no summary)".into(),
    }
}

/// Run `blut runs diff`. `all` shows identical keys too; `json` emits a
/// machine-readable diff instead of the human table.
pub fn diff(id1_query: &str, id2_query: &str, all: bool, json: bool) -> Result<()> {
    let id1 = jobs::resolve_job_id(id1_query)?;
    let id2 = jobs::resolve_job_id(id2_query)?;
    let p1 = load_provenance(&id1)?;
    let p2 = load_provenance(&id2)?;

    // Union of keys, sorted (BTreeMap iteration is already sorted; merge).
    let mut keys: Vec<&String> = p1.flat.keys().chain(p2.flat.keys()).collect();
    keys.sort();
    keys.dedup();

    #[derive(serde::Serialize)]
    struct KeyDiff {
        key: String,
        left: Option<String>,
        right: Option<String>,
        changed: bool,
    }
    let diffs: Vec<KeyDiff> = keys
        .iter()
        .map(|k| {
            let l = p1.flat.get(*k).cloned();
            let r = p2.flat.get(*k).cloned();
            let changed = l != r;
            KeyDiff {
                key: (*k).clone(),
                left: l,
                right: r,
                changed,
            }
        })
        .collect();

    if json {
        let payload = serde_json::json!({
            "left":  { "job": id1, "source": p1.source, "recipe": p1.recipe, "outcome": outcome_line(&id1) },
            "right": { "job": id2, "source": p2.source, "recipe": p2.recipe, "outcome": outcome_line(&id2) },
            "diff": diffs.iter().filter(|d| all || d.changed).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload)
                .map_err(|e| TrainError::other(format!("serialize diff: {e}")))?
        );
        return Ok(());
    }

    // Human table.
    let r1 = p1.recipe.as_deref().unwrap_or("(spec)");
    let r2 = p2.recipe.as_deref().unwrap_or("(spec)");
    if r1 == r2 {
        println!("recipe   {r1} == {r2}");
    } else {
        println!("recipe   {r1}  →  {r2}");
    }
    let mut elided = 0usize;
    for d in &diffs {
        if !d.changed {
            if all {
                println!("  {:<24} {}  ==", d.key, d.left.as_deref().unwrap_or("-"));
            } else {
                elided += 1;
            }
            continue;
        }
        let l = d.left.as_deref().unwrap_or("(absent)");
        let r = d.right.as_deref().unwrap_or("(absent)");
        println!("  {:<24} {}  →  {}", d.key, l, r);
    }
    if elided > 0 {
        println!("  ({elided} identical key(s) elided; --all to show)");
    }
    println!("─ outcome ─");
    println!("  {id1}   {}", outcome_line(&id1));
    println!("  {id2}   {}", outcome_line(&id2));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{flatten, scalar};
    use std::collections::BTreeMap;

    #[test]
    fn flatten_dot_paths_and_arrays() {
        let v = serde_json::json!({
            "tier": 6,
            "preset": "fast",
            "extra_args": ["--encoder-width", "256"],
        });
        let mut out = BTreeMap::new();
        flatten("", &v, &mut out);
        assert_eq!(out.get("tier").unwrap(), "6");
        assert_eq!(out.get("preset").unwrap(), "fast");
        assert_eq!(out.get("extra_args.0").unwrap(), "--encoder-width");
        assert_eq!(out.get("extra_args.1").unwrap(), "256");
    }

    #[test]
    fn scalar_strings_unquoted() {
        assert_eq!(scalar(&serde_json::json!("hi")), "hi");
        assert_eq!(scalar(&serde_json::json!(3)), "3");
        assert_eq!(scalar(&serde_json::json!(true)), "true");
    }
}
