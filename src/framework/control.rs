// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
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
/// the step's `node_idx` → the in-flight node's cancel token). `Spawn`
/// appends a fresh sub-plan to the RUNNING graph at runtime — PBT / TPE
/// ask-tell — drained on the coordinator's single-threaded seam between
/// joins, never inside the `select!`.
#[derive(Debug)]
pub enum Control {
    /// Leave the graph as-is.
    Continue,
    /// Cancel the emitting node and prune its descendants (their input can
    /// never materialize). FW-2 tmp cleanup discards the partial; no cache
    /// entry is written, so a later re-run is unaffected.
    KillBranch,
    /// Inject a new sub-plan into the running graph (its nodes become new
    /// graph nodes; its graph-inputs seed the roots). For PBT a perturbed
    /// clone's `--resume-from` is baked into the sub-plan's args by the
    /// policy's factory, so the executor stays oblivious to resume. Boxed so
    /// the common `Continue`/`KillBranch` returns aren't a big-`SpawnDelta`
    /// wide value.
    Spawn(Box<SpawnDelta>),
}

/// PartialEq for `Control` compares the control INTENT: the unit variants by
/// discriminant, and `Spawn` by its label only (the heavy `CompiledPlan` is
/// never value-compared — equality is used only in tests, which never assert on
/// a `Spawn`'s sub-plan). Reflexive/symmetric/transitive on that projection.
impl PartialEq for Control {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Control::Continue, Control::Continue) | (Control::KillBranch, Control::KillBranch) => {
                true
            }
            (Control::Spawn(a), Control::Spawn(b)) => a.label == b.label,
            _ => false,
        }
    }
}

/// A runtime graph mutation: a self-contained sub-plan to inject. The policy
/// builds the `subplan` via a factory it captured at construction (it owns the
/// `Registry` + recipe + any resume-from wiring); the executor only performs the
/// structural injection — id-offset relabel, seed roots, extend the schedule.
pub struct SpawnDelta {
    pub subplan: crate::framework::plan::CompiledPlan,
    /// Optional provenance label (e.g. a PBT child trial id) for logging.
    pub label: Option<String>,
    /// Root seeds (ADR 0078 `map_output`): `(local_root_id, artifact,
    /// logical_hash)` triples that seed the sub-plan's roots with a SPECIFIC
    /// input instead of the unit graph input. Empty for a disconnected spawn
    /// (PBT/TPE) — then the sub-plan's own `initial` seeds its roots, exactly
    /// as before. Non-empty for a map element: the element artifact.
    ///
    /// `pub(crate)`: set ONLY by the engine's map path (external control
    /// policies construct spawns via [`SpawnDelta::new`], which leaves these
    /// empty/`None`). This keeps `provenance_parent` a trustworthy "this is a
    /// map shard" flag — the executor uses it to fail (not warn-drop) at the
    /// spawn cap.
    pub(crate) root_seeds: Vec<(
        crate::framework::plan::NodeId,
        crate::framework::stage::ErasedArtifact,
        crate::framework::artifact::ContentHash,
    )>,
    /// The global node id whose `list` output produced this spawn (map
    /// provenance); `None` for PBT/TPE spawns. `pub(crate)` — see `root_seeds`.
    pub(crate) provenance_parent: Option<crate::framework::plan::NodeId>,
}

impl SpawnDelta {
    /// A plain disconnected sub-plan spawn (PBT/TPE): no root seeds, no map
    /// provenance. The sub-plan's own graph-inputs seed its roots.
    pub fn new(subplan: crate::framework::plan::CompiledPlan, label: Option<String>) -> Self {
        Self {
            subplan,
            label,
            root_seeds: Vec::new(),
            provenance_parent: None,
        }
    }
}

impl std::fmt::Debug for SpawnDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnDelta")
            .field("subplan", &self.subplan.name())
            .field("n_nodes", &self.subplan.n_nodes())
            .field("label", &self.label)
            .finish()
    }
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

/// Compose several policies into one, consulted in order. The FIRST policy to
/// return a non-`Continue` decision wins and short-circuits the rest; if every
/// policy is satisfied the composite returns `Continue`.
///
/// The canonical use is layering the broad [`KillOnNaN`] safety net UNDER an
/// HPO policy (TPE / PBT / ASHA): `[KillOnNaN, hpo]`. Order is load-bearing.
///
/// * **Safety is never dropped.** Today `ExecCtx::with_control` REPLACES the
///   default `KillOnNaN`, so enabling an HPO policy would otherwise lose the
///   payload-wide non-finite scan (an HPO policy typically only watches its
///   single objective key — `TpePolicy` doesn't kill on divergence at all).
/// * **No spawn-slot leak.** An HPO policy's `on_step` may pop its trial queue
///   and bump its spawn counter BEFORE returning `Spawn`. Running `KillOnNaN`
///   first and short-circuiting means the HPO policy is never consulted on a
///   step that is already being killed, so no queued spawn is silently dropped.
pub struct CompositePolicy {
    policies: Vec<std::sync::Arc<dyn ControlPolicy>>,
}

impl CompositePolicy {
    /// Build a composite from policies in consult order (first wins on a tie).
    pub fn new(policies: Vec<std::sync::Arc<dyn ControlPolicy>>) -> Self {
        Self { policies }
    }
}

impl ControlPolicy for CompositePolicy {
    fn on_step(&self, metrics: &StepMetrics) -> Control {
        for p in &self.policies {
            match p.on_step(metrics) {
                // Satisfied by this policy — consult the next one.
                Control::Continue => continue,
                // KillBranch or Spawn — the first decisive policy wins.
                decisive => return decisive,
            }
        }
        Control::Continue
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

    // --- CompositePolicy (B2) ---------------------------------------------

    /// Always `Continue`, but counts how many times it was consulted — lets a
    /// test prove the composite short-circuited (counter stays 0) instead of
    /// falling through to a downstream policy.
    #[derive(Default)]
    struct CountingContinue {
        seen: std::sync::atomic::AtomicU32,
    }
    impl ControlPolicy for CountingContinue {
        fn on_step(&self, _m: &StepMetrics) -> Control {
            self.seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Control::Continue
        }
    }

    /// Always `KillBranch` — stands in for a decisive downstream policy.
    struct AlwaysKill;
    impl ControlPolicy for AlwaysKill {
        fn on_step(&self, _m: &StepMetrics) -> Control {
            Control::KillBranch
        }
    }

    #[test]
    fn composite_safety_kills_even_if_inner_continues() {
        // [KillOnNaN, AlwaysContinue] on a NaN payload → KillBranch from the
        // safety net, even though the inner policy would have continued.
        let inner = std::sync::Arc::new(CountingContinue::default());
        let comp = CompositePolicy::new(vec![std::sync::Arc::new(KillOnNaN), inner.clone()]);
        let u = json!({ "loss": "nan" });
        assert_eq!(comp.on_step(&metrics(&u)), Control::KillBranch);
        // Short-circuited: the inner policy was never consulted (no spawn-slot
        // leak — an HPO policy would not have popped its trial queue here).
        assert_eq!(inner.seen.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn composite_inner_decides_when_safe() {
        // [KillOnNaN, AlwaysKill] on a FINITE payload → the inner policy is
        // consulted (KillOnNaN continues) and its decision wins.
        let comp = CompositePolicy::new(vec![
            std::sync::Arc::new(KillOnNaN),
            std::sync::Arc::new(AlwaysKill),
        ]);
        let u = json!({ "loss": 0.42 });
        assert_eq!(comp.on_step(&metrics(&u)), Control::KillBranch);
    }

    #[test]
    fn composite_empty_and_single() {
        // Empty list → Continue. Single element behaves exactly like it alone.
        let empty = CompositePolicy::new(vec![]);
        let u = json!({ "loss": 0.1 });
        assert_eq!(empty.on_step(&metrics(&u)), Control::Continue);

        let single = CompositePolicy::new(vec![std::sync::Arc::new(KillOnNaN)]);
        let nan = json!({ "loss": "inf" });
        assert_eq!(single.on_step(&metrics(&nan)), Control::KillBranch);
        assert_eq!(single.on_step(&metrics(&u)), Control::Continue);
    }

    #[test]
    fn composite_first_decisive_wins() {
        // A kill from policy 0 returns without consulting policy 1.
        let inner = std::sync::Arc::new(CountingContinue::default());
        let comp = CompositePolicy::new(vec![std::sync::Arc::new(AlwaysKill), inner.clone()]);
        let u = json!({ "loss": 0.3 });
        assert_eq!(comp.on_step(&metrics(&u)), Control::KillBranch);
        assert_eq!(inner.seen.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
