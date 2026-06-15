//! HPO search space (BLUT v0.20).
//!
//! A typed space of named hyperparameter dimensions, each with a distribution.
//! A *sample* is an [`Overlay`] — a list of `(dotted-arg-path, JSON value)`
//! pairs that [`apply_overlay`] deep-merges onto the base recipe args. Because a
//! trial's overlaid args change `canon_args`, each trial gets a DISTINCT cache
//! key (and therefore a distinct durable-resume dir) for free — no special
//! per-trial bookkeeping in the executor.
//!
//! Two specification surfaces (both land here): a `--space <file>` YAML doc and
//! repeatable inline `--param 'lr=loguniform(1e-5,1e-2)'` flags.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use rand::Rng;

/// An args overlay: `(dotted-path, value)` pairs deep-merged onto base args.
pub type Overlay = Vec<(String, Value)>;

/// A completed trial's overlay + its objective value — fed to model-based
/// samplers (TPE) so they can condition the next suggestion on results. Random
/// search ignores it.
#[derive(Clone, Debug)]
pub struct TrialResult {
    pub overlay: Overlay,
    pub objective: f64,
}

/// One hyperparameter's distribution. Internally tagged so a YAML dim reads
/// `{ dist: log_uniform, low: 1e-5, high: 1e-2 }`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "dist", rename_all = "snake_case")]
pub enum Dist {
    /// Continuous uniform over `[low, high)`.
    Uniform { low: f64, high: f64 },
    /// Log-uniform over `[low, high)` (both > 0) — uniform in log-space, the
    /// right prior for learning rates / weight decays spanning decades.
    LogUniform { low: f64, high: f64 },
    /// Integer uniform over `[low, high]` inclusive.
    IntUniform { low: i64, high: i64 },
    /// Uniform over `[low, high)` rounded to the nearest multiple of `q`.
    QUniform { low: f64, high: f64, q: f64 },
    /// Categorical: one of `choices`.
    Choice { choices: Vec<Value> },
}

impl Dist {
    /// Reject a degenerate / misconfigured distribution (fail-fast — HPO
    /// configs are user-supplied; a silent clamp would sample the wrong space).
    pub fn validate(&self, name: &str) -> Result<(), String> {
        let bad = |m: String| Err(format!("dim '{name}': {m}"));
        match self {
            Dist::Uniform { low, high } => {
                if high <= low {
                    return bad(format!("high ({high}) must be > low ({low})"));
                }
            }
            Dist::LogUniform { low, high } => {
                if *low <= 0.0 {
                    return bad(format!("log_uniform low ({low}) must be > 0"));
                }
                if high <= low {
                    return bad(format!("high ({high}) must be > low ({low})"));
                }
            }
            Dist::IntUniform { low, high } => {
                if high < low {
                    return bad(format!("high ({high}) must be >= low ({low})"));
                }
            }
            Dist::QUniform { low, high, q } => {
                if high <= low {
                    return bad(format!("high ({high}) must be > low ({low})"));
                }
                if *q <= 0.0 {
                    return bad(format!("q ({q}) must be > 0"));
                }
            }
            Dist::Choice { choices } => {
                if choices.is_empty() {
                    return bad("choice list is empty".into());
                }
            }
        }
        Ok(())
    }

    /// Draw one value from this distribution. Assumes [`validate`](Dist::validate)
    /// passed; the `.max(..)` guards below are belt-and-suspenders (no-ops on
    /// validated input). NOTE: `QUniform` rounding can yield `high` for an edge
    /// draw near the top of the range.
    pub fn sample(&self, rng: &mut impl Rng) -> Value {
        match self {
            Dist::Uniform { low, high } => json_f64(rng.gen_range(*low..high.max(*low + f64::EPSILON))),
            Dist::LogUniform { low, high } => {
                let (ll, lh) = (low.max(f64::MIN_POSITIVE).ln(), high.max(f64::MIN_POSITIVE).ln());
                json_f64(rng.gen_range(ll..lh.max(ll + f64::EPSILON)).exp())
            }
            Dist::IntUniform { low, high } => Value::from(rng.gen_range(*low..=(*high).max(*low))),
            Dist::QUniform { low, high, q } => {
                let raw = rng.gen_range(*low..high.max(*low + f64::EPSILON));
                let qq = if *q == 0.0 { 1.0 } else { *q };
                json_f64((raw / qq).round() * qq)
            }
            Dist::Choice { choices } => {
                if choices.is_empty() {
                    Value::Null
                } else {
                    choices[rng.gen_range(0..choices.len())].clone()
                }
            }
        }
    }
}

impl Dist {
    /// PBT "explore": perturb a CURRENT value within this dim's support. Classic
    /// PBT scales a continuous value by 0.8 or 1.2 (then clamps to `[low, high]`);
    /// an integer moves at least one step in the scaled direction; a categorical
    /// resamples. Assumes [`validate`](Dist::validate) passed.
    pub fn perturb(&self, current: &Value, rng: &mut impl Rng) -> Value {
        let factor = if rng.gen_bool(0.5) { 0.8 } else { 1.2 };
        match self {
            Dist::Uniform { low, high } => {
                let c = current.as_f64().unwrap_or((low + high) / 2.0);
                json_f64((c * factor).clamp(*low, *high))
            }
            Dist::LogUniform { low, high } => {
                let c = current.as_f64().unwrap_or((low * high).sqrt());
                json_f64((c * factor).clamp(*low, *high))
            }
            Dist::IntUniform { low, high } => {
                let c = current.as_i64().unwrap_or((low + high) / 2);
                let mut v = ((c as f64) * factor).round() as i64;
                // Ensure the value actually MOVES even when rounding pins it back
                // to `c` (e.g. small magnitudes), so explore makes progress.
                if v == c {
                    v = if factor > 1.0 { c + 1 } else { c - 1 };
                }
                Value::from(v.clamp(*low, *high))
            }
            Dist::QUniform { low, high, q } => {
                let c = current.as_f64().unwrap_or((low + high) / 2.0);
                let qq = if *q == 0.0 { 1.0 } else { *q };
                json_f64((((c * factor) / qq).round() * qq).clamp(*low, *high))
            }
            Dist::Choice { choices } => {
                if choices.is_empty() {
                    current.clone()
                } else {
                    choices[rng.gen_range(0..choices.len())].clone()
                }
            }
        }
    }
}

/// `f64` → a JSON number (falls back to `Null` for NaN/Inf, which `serde_json`
/// cannot represent — never produced by the samplers above on finite inputs).
fn json_f64(x: f64) -> Value {
    serde_json::Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}

/// The search space: an ordered (BTreeMap = deterministic) map of dotted
/// arg-path → distribution. Deterministic key order makes seeded sampling
/// reproducible.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct SearchSpace {
    #[serde(default)]
    pub dims: BTreeMap<String, Dist>,
}

impl SearchSpace {
    /// Parse a `--space` YAML doc: a top-level `dims:` map of name → dist, OR a
    /// bare map of name → dist (the `dims:` wrapper is optional).
    pub fn from_yaml(text: &str) -> Result<SearchSpace, String> {
        // Try the wrapped form first, then the bare map.
        if let Ok(s) = serde_yaml::from_str::<SearchSpace>(text) {
            if !s.dims.is_empty() {
                s.validate()?;
                return Ok(s);
            }
        }
        let space = serde_yaml::from_str::<BTreeMap<String, Dist>>(text)
            .map(|dims| SearchSpace { dims })
            .map_err(|e| format!("search-space YAML parse error: {e}"))?;
        space.validate()?;
        Ok(space)
    }

    /// Reject an empty or degenerate space. Call after assembling the full space
    /// (file + inline `--param` flags merged) before launching trials.
    pub fn validate(&self) -> Result<(), String> {
        if self.dims.is_empty() {
            return Err("search space has no dimensions".into());
        }
        for (name, dist) in &self.dims {
            dist.validate(name)?;
        }
        Ok(())
    }

    /// Parse one inline `--param` spec: `name=fn(args)` where `fn` is
    /// `uniform|loguniform|int|quniform|choice`. Merged into the space by the
    /// caller (later flags override earlier same-name dims). Args split on `,`,
    /// so a `choice` value cannot itself contain a comma — use the `--space`
    /// YAML for complex / comma-bearing choice values.
    pub fn parse_param(spec: &str) -> Result<(String, Dist), String> {
        let (name, rest) = spec
            .split_once('=')
            .ok_or_else(|| format!("--param '{spec}' must be name=fn(...)"))?;
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("--param '{spec}' has an empty name"));
        }
        let rest = rest.trim();
        let open = rest.find('(').ok_or_else(|| format!("--param '{spec}': missing '('"))?;
        if !rest.ends_with(')') {
            return Err(format!("--param '{spec}': missing closing ')'"));
        }
        let func = rest[..open].trim();
        let inner = &rest[open + 1..rest.len() - 1];
        let args: Vec<&str> = inner.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        let f = |i: usize| -> Result<f64, String> {
            args.get(i)
                .and_then(|s| s.parse::<f64>().ok())
                .ok_or_else(|| format!("--param '{spec}': arg {i} not a number"))
        };
        // Strict integer parse — reject `int(1.9,3)` rather than truncating it.
        let int = |i: usize| -> Result<i64, String> {
            args.get(i)
                .and_then(|s| s.parse::<i64>().ok())
                .ok_or_else(|| format!("--param '{spec}': arg {i} not an integer"))
        };
        // Reject the wrong arity (`uniform(1,2,3)` silently dropping `3`, or
        // `quniform(1,10)` missing `q`) rather than guessing.
        let want = |n: usize| -> Result<(), String> {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!("--param '{spec}': {func} expects {n} args, got {}", args.len()))
            }
        };
        let dist = match func {
            "uniform" => {
                want(2)?;
                Dist::Uniform { low: f(0)?, high: f(1)? }
            }
            "loguniform" | "log_uniform" => {
                want(2)?;
                Dist::LogUniform { low: f(0)?, high: f(1)? }
            }
            "int" | "int_uniform" => {
                want(2)?;
                Dist::IntUniform { low: int(0)?, high: int(1)? }
            }
            "quniform" | "q_uniform" => {
                want(3)?;
                Dist::QUniform { low: f(0)?, high: f(1)?, q: f(2)? }
            }
            "choice" if args.is_empty() => {
                return Err(format!("--param '{spec}': choice needs >= 1 value"));
            }
            "choice" => Dist::Choice {
                // Each choice: parse as a JSON scalar (number/bool), else a string.
                choices: args
                    .iter()
                    .map(|s| serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String((*s).to_string())))
                    .collect(),
            },
            other => return Err(format!("--param '{spec}': unknown dist '{other}'")),
        };
        Ok((name.to_string(), dist))
    }

    /// Draw one overlay (in deterministic dim order).
    pub fn sample(&self, rng: &mut impl Rng) -> Overlay {
        self.dims
            .iter()
            .map(|(name, dist)| (name.clone(), dist.sample(rng)))
            .collect()
    }
}

/// Deep-merge an overlay into `base` args: each `"a.b.c" -> v` writes `v` at the
/// nested path, creating intermediate objects as needed. A non-object blocking
/// an intermediate path is overwritten with an object (the overlay wins — it is
/// the operator's explicit per-trial choice).
pub fn apply_overlay(base: &mut Value, overlay: &Overlay) {
    for (path, val) in overlay {
        insert_path(base, path, val.clone());
    }
}

/// Set `val` at dotted `path` within `node`, creating intermediate objects (and
/// coercing a non-object `node` to an object — the overlay is the operator's
/// explicit per-trial choice, so it wins).
fn insert_path(node: &mut Value, path: &str, val: Value) {
    if !node.is_object() {
        *node = Value::Object(serde_json::Map::new());
    }
    let m = node.as_object_mut().expect("coerced to object above");
    match path.split_once('.') {
        None => {
            m.insert(path.to_string(), val);
        }
        Some((head, rest)) => {
            let child = m
                .entry(head.to_string())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            insert_path(child, rest, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn yaml_wrapped_and_bare_parse() {
        let wrapped = "dims:\n  lr: { dist: log_uniform, low: 1.0e-5, high: 1.0e-2 }\n  bs: { dist: int_uniform, low: 8, high: 64 }\n";
        let s = SearchSpace::from_yaml(wrapped).unwrap();
        assert_eq!(s.dims.len(), 2);
        assert_eq!(s.dims["lr"], Dist::LogUniform { low: 1e-5, high: 1e-2 });
        let bare = "lr: { dist: uniform, low: 0.0, high: 1.0 }\n";
        let s2 = SearchSpace::from_yaml(bare).unwrap();
        assert_eq!(s2.dims["lr"], Dist::Uniform { low: 0.0, high: 1.0 });
    }

    #[test]
    fn inline_param_parse() {
        assert_eq!(
            SearchSpace::parse_param("lr=loguniform(1e-5,1e-2)").unwrap(),
            ("lr".into(), Dist::LogUniform { low: 1e-5, high: 1e-2 })
        );
        assert_eq!(
            SearchSpace::parse_param("bs=int(8,64)").unwrap(),
            ("bs".into(), Dist::IntUniform { low: 8, high: 64 })
        );
        assert_eq!(
            SearchSpace::parse_param("opt=choice(soap,adamw)").unwrap(),
            (
                "opt".into(),
                Dist::Choice { choices: vec![Value::from("soap"), Value::from("adamw")] }
            )
        );
        // choice with numbers parses them as numbers, not strings.
        assert_eq!(
            SearchSpace::parse_param("tier=choice(2,3)").unwrap().1,
            Dist::Choice { choices: vec![Value::from(2), Value::from(3)] }
        );
        assert!(SearchSpace::parse_param("bad").is_err());
        assert!(SearchSpace::parse_param("x=nope(1,2)").is_err());
        // Wrong arity rejected (not silently dropped / guessed).
        assert!(SearchSpace::parse_param("x=uniform(1,2,3)").is_err(), "extra arg rejected");
        assert!(SearchSpace::parse_param("x=quniform(1,10)").is_err(), "missing q rejected");
        assert!(SearchSpace::parse_param("x=choice()").is_err(), "empty choice rejected");
    }

    #[test]
    fn sampling_is_in_range_and_seed_reproducible() {
        let mut s = SearchSpace::default();
        s.dims.insert("lr".into(), Dist::LogUniform { low: 1e-4, high: 1e-1 });
        s.dims.insert("bs".into(), Dist::IntUniform { low: 8, high: 16 });
        s.dims.insert("q".into(), Dist::QUniform { low: 0.0, high: 1.0, q: 0.25 });
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        let sa = s.sample(&mut a);
        let sb = s.sample(&mut b);
        assert_eq!(sa, sb, "same seed → same sample");
        for (name, v) in &sa {
            match name.as_str() {
                "lr" => {
                    let x = v.as_f64().unwrap();
                    assert!((1e-4..1e-1).contains(&x), "lr {x} in range");
                }
                "bs" => {
                    let x = v.as_i64().unwrap();
                    assert!((8..=16).contains(&x));
                }
                "q" => {
                    let x = v.as_f64().unwrap();
                    assert!((x / 0.25).fract().abs() < 1e-9, "q quantized: {x}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn overlay_deep_merges_dotted_paths() {
        let mut base = serde_json::json!({ "tier": 3, "extra": { "keep": true } });
        let overlay = vec![
            ("lr".to_string(), Value::from(0.01)),
            ("extra.lr".to_string(), Value::from(0.02)),
            ("a.b.c".to_string(), Value::from(7)),
        ];
        apply_overlay(&mut base, &overlay);
        assert_eq!(base["lr"], Value::from(0.01));
        assert_eq!(base["tier"], Value::from(3), "untouched key preserved");
        assert_eq!(base["extra"]["keep"], Value::from(true), "sibling preserved");
        assert_eq!(base["extra"]["lr"], Value::from(0.02));
        assert_eq!(base["a"]["b"]["c"], Value::from(7), "nested path created");
    }

    #[test]
    fn validate_rejects_degenerate() {
        let mut s = SearchSpace::default();
        assert!(s.validate().is_err(), "empty space rejected");
        s.dims.insert("lr".into(), Dist::LogUniform { low: 0.0, high: 1.0 });
        assert!(s.validate().is_err(), "log_uniform low<=0 rejected");
        s.dims.insert("lr".into(), Dist::Uniform { low: 1.0, high: 0.0 });
        assert!(s.validate().is_err(), "high<=low rejected");
        s.dims.insert("lr".into(), Dist::QUniform { low: 0.0, high: 1.0, q: 0.0 });
        assert!(s.validate().is_err(), "q<=0 rejected");
        s.dims.insert("lr".into(), Dist::Uniform { low: 0.0, high: 1.0 });
        s.dims.insert("c".into(), Dist::Choice { choices: vec![] });
        assert!(s.validate().is_err(), "empty choice rejected");
    }

    #[test]
    fn from_yaml_rejects_empty_and_bad() {
        assert!(SearchSpace::from_yaml("dims:\n").is_err(), "empty dims rejected");
        assert!(
            SearchSpace::from_yaml("lr: { dist: log_uniform, low: 0.0, high: 1.0 }\n").is_err(),
            "log_uniform low<=0 rejected at parse (bare form)"
        );
        // Wrapped `dims:` form must validate too (the early-return path).
        assert!(
            SearchSpace::from_yaml("dims:\n  lr: { dist: uniform, low: 1.0, high: 0.0 }\n").is_err(),
            "high<=low rejected at parse (wrapped form)"
        );
    }

    #[test]
    fn int_param_rejects_float() {
        assert!(SearchSpace::parse_param("x=int(1.9,3)").is_err(), "float arg to int() rejected");
        assert!(SearchSpace::parse_param("x=int(1,3)").is_ok());
    }
}
