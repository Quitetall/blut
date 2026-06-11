//! Runtime DAG control — the policy hook the parallel executor consults
//! against the live `StageStep` metric stream to mutate the running graph
//! (#4 dynamic runtime DAG mutation).
//!
//! The executor owns ALL mutable scheduler state and only mutates it
//! between `join_next().await`s; a [`ControlPolicy`] is evaluated on that
//! same single-threaded seam, so a `KillBranch` decision can never race the
//! FW-2 promote / cache insert. The policy is PURE (`&self -> Control`) and
//! sees one [`StepMetrics`] at a time; it holds any cross-step state behind
//! its own interior mutability.
//!
//! Scope of this slice: `KillBranch` (prune a doomed branch — e.g. a
//! diverged trainer) is wired. `Spawn` (append a node at runtime — PBT /
//! ask-tell) is the next slice; the [`Control`] enum is intentionally small
//! so adding it later is additive, not a breaking reshape.
//!
//! Default policy: [`KillOnNaN`] — kill the emitting node's branch the
//! instant its step metrics report a non-finite value (NaN / Inf). Because
//! JSON numbers cannot carry NaN (serde rejects it at parse), a trainer
//! signals divergence as a STRING (`"loss": "nan"`) or a BOOL flag
//! (`"diverged": true`); the scan below recognizes both. Cookbooks needing
//! a schema-aware or PBT policy implement [`ControlPolicy`] themselves.

use serde_json::Value;

/// What the executor should do with the running graph after a step.
///
/// `KillBranch` targets the node that EMITTED the step (the executor maps
/// the step's `node_idx` → the in-flight node's cancel token). Kept minimal
/// on purpose — `Spawn(delta)` lands in the next slice without reshaping the
/// callers that match `Continue` / `KillBranch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// Leave the graph as-is.
    Continue,
    /// Cancel the emitting node and prune its descendants (their input can
    /// never materialize). FW-2 tmp cleanup discards the partial; no cache
    /// entry is written, so a later re-run is unaffected.
    KillBranch,
}

/// One step's metrics, borrowed from the live `StageEvent::StageStep`. The
/// `update` is the raw trainer payload (loss / val_r / grad_norm / …); a
/// policy reads whatever fields it understands.
#[derive(Clone, Copy, Debug)]
pub struct StepMetrics<'a> {
    /// Topo position of the emitting node — the executor's stable status key.
    pub node_idx: u32,
    pub stage_name: &'a str,
    pub update: &'a Value,
}

impl StepMetrics<'_> {
    /// Does any value in the step payload report a non-finite number?
    ///
    /// serde_json `Number` is always finite (NaN/Inf are rejected at parse),
    /// so divergence can only arrive as:
    ///   * a string token — `"nan"`, `"inf"`, `"-inf"`, `"infinity"`
    ///     (case-insensitive), OR a numeric string that parses to a
    ///     non-finite `f64` (e.g. `"NaN"`); OR
    ///   * an explicit boolean flag on a divergence-named key
    ///     (`nan` / `is_nan` / `diverged` / `non_finite` = `true`).
    ///
    /// The scan recurses through nested objects/arrays so a metric buried
    /// under `{"metrics": {"loss": "nan"}}` is still caught.
    pub fn has_non_finite(&self) -> bool {
        scan_non_finite(self.update)
    }
}

/// Recognize a string token that denotes a non-finite float. `eq_ignore_
/// ascii_case` avoids allocating a lowercased copy per metric.
fn is_non_finite_token(s: &str) -> bool {
    let t = s.trim();
    const TOKENS: [&str; 7] = [
        "nan",
        "inf",
        "-inf",
        "+inf",
        "infinity",
        "-infinity",
        "+infinity",
    ];
    if TOKENS.iter().any(|tok| t.eq_ignore_ascii_case(tok)) {
        return true;
    }
    // A numeric string that parses to a non-finite f64 (e.g. "1e999").
    if let Ok(x) = t.parse::<f64>() {
        return !x.is_finite();
    }
    false
}

/// Keys whose `true` value flags divergence even without a numeric payload.
fn is_divergence_flag_key(key: &str) -> bool {
    const KEYS: [&str; 7] = [
        "nan",
        "is_nan",
        "isnan",
        "has_nan",
        "diverged",
        "divergence",
        "non_finite",
    ];
    KEYS.iter().any(|k| key.eq_ignore_ascii_case(k))
}

/// Bound the recursion: the step payload crosses a process boundary (the
/// trainer subprocess), so a pathologically-nested object must not be able to
/// overflow the stack. Real trainer metrics are flat; 16 levels is generous.
const MAX_SCAN_DEPTH: usize = 16;

fn scan_non_finite(v: &Value) -> bool {
    scan_non_finite_at(v, 0)
}

fn scan_non_finite_at(v: &Value, depth: usize) -> bool {
    if depth >= MAX_SCAN_DEPTH {
        // Past the cap a payload is treated as finite (no false kill); a
        // metric that deep is malformed, not a real divergence signal.
        return false;
    }
    match v {
        Value::String(s) => is_non_finite_token(s),
        // A bare number is always finite (serde guarantees it); but a float
        // that somehow round-tripped as f64 is checked defensively.
        Value::Number(n) => n.as_f64().map(|x| !x.is_finite()).unwrap_or(false),
        Value::Array(items) => items.iter().any(|x| scan_non_finite_at(x, depth + 1)),
        Value::Object(map) => map.iter().any(|(k, val)| {
            // A divergence-named boolean flag set true, or any nested value
            // that is itself non-finite.
            (is_divergence_flag_key(k) && val.as_bool() == Some(true))
                || scan_non_finite_at(val, depth + 1)
        }),
        _ => false,
    }
}

/// The runtime DAG control policy: consulted per live step on the executor's
/// single-threaded coordinator seam. Implementors hold cross-step state
/// (e.g. a divergence counter, a PBT population) behind their own interior
/// mutability — `on_step` takes `&self`.
pub trait ControlPolicy: Send + Sync {
    /// Decide what to do with the graph given this step. MUST be cheap +
    /// non-blocking — it runs inline in the coordinator between joins.
    fn on_step(&self, metrics: &StepMetrics) -> Control;
}

/// Kill a node's branch the moment its step metrics report a non-finite
/// value. The simplest, highest-value runtime policy: a diverged trainer
/// stops burning the GPU immediately instead of running to its epoch budget.
///
/// Relies on the fact that divergence PERSISTS — once a loss goes NaN every
/// subsequent step is NaN too. So even if the live broadcast lags and drops
/// the first NaN step under a flood, the next step re-signals it; the kill is
/// not a single-shot edge. (A one-shot control signal would need a dedicated
/// lossless channel — that is the `Spawn`/PBT slice, not this one.)
#[derive(Clone, Copy, Debug, Default)]
pub struct KillOnNaN;

impl ControlPolicy for KillOnNaN {
    fn on_step(&self, metrics: &StepMetrics) -> Control {
        if metrics.has_non_finite() {
            Control::KillBranch
        } else {
            Control::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn metrics(update: &Value) -> StepMetrics<'_> {
        StepMetrics {
            node_idx: 3,
            stage_name: "train_joint",
            update,
        }
    }

    #[test]
    fn finite_metrics_continue() {
        let u = json!({ "loss": 0.42, "val_r": 0.55, "step": 1200 });
        assert_eq!(KillOnNaN.on_step(&metrics(&u)), Control::Continue);
        assert!(!metrics(&u).has_non_finite());
    }

    #[test]
    fn nan_string_token_kills() {
        for tok in ["nan", "NaN", "inf", "-inf", "Infinity"] {
            let u = json!({ "loss": tok });
            assert_eq!(
                KillOnNaN.on_step(&metrics(&u)),
                Control::KillBranch,
                "token {tok:?} must trigger a kill"
            );
        }
    }

    #[test]
    fn divergence_bool_flag_kills() {
        let u = json!({ "loss": 0.1, "diverged": true });
        assert_eq!(KillOnNaN.on_step(&metrics(&u)), Control::KillBranch);
        // A false flag does NOT kill.
        let ok = json!({ "loss": 0.1, "diverged": false });
        assert_eq!(KillOnNaN.on_step(&metrics(&ok)), Control::Continue);
    }

    #[test]
    fn nested_non_finite_is_caught() {
        let u = json!({ "metrics": { "train": { "loss": "nan" } } });
        assert!(metrics(&u).has_non_finite());
        let arr = json!({ "grads": [0.1, 0.2, "inf"] });
        assert!(metrics(&arr).has_non_finite());
    }

    #[test]
    fn ordinary_strings_do_not_kill() {
        // A non-numeric string field (e.g. a phase label) must not trip it.
        let u = json!({ "phase": "warmup", "note": "information", "loss": 0.3 });
        assert_eq!(KillOnNaN.on_step(&metrics(&u)), Control::Continue);
    }

    #[test]
    fn numeric_string_that_is_finite_continues() {
        // Some trainers emit numbers as strings; a finite one must not kill.
        let u = json!({ "loss": "0.0421" });
        assert_eq!(KillOnNaN.on_step(&metrics(&u)), Control::Continue);
    }
}
