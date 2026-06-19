//! Named SENSORS (Phase G / G2): observe external state, yield a typed
//! outcome.
//!
//! A [`Sensor`] is the READ/observe sibling of [`framework::control`]
//! (which MUTATES a running plan). It reconciles the scattered
//! "should I proceed right now?" checks — the cross-process GPU
//! scheduler lock, and the auto-train [`policy`](crate::policy) gates —
//! as NAMED, listable, evaluable observations (`blut sensor list|eval`),
//! so a daemon/cron, the TUI, or an operator can ask a sensor the same
//! question the same way.
//!
//! This is intentionally MINIMAL + additive: it reframes existing
//! observations (it does NOT invent a scheduler). The on-disk policy
//! file is still mutated only by `policy enable|disable`; a sensor never
//! writes.

/// Sentinel new-turn count for [`AutoTrainPolicySensor`]: the conversation
/// turn count lives in the lamu store (out of BLUT's reach), so we pass a
/// value that always clears `policy::decide`'s `threshold_new_turns` gate —
/// the sensor then observes only the TIME + lock gates it CAN evaluate. If
/// `decide` ever grows a turns-CEILING check, revisit this coupling.
const ASSUME_ENOUGH_TURNS: i64 = i64::MAX;

/// What a sensor observed.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum SensorOutcome {
    /// The observed state permits proceeding now.
    Ready,
    /// Not ready — a (typically longer-lived) condition blocks it. The
    /// reason is human-facing.
    Skip { reason: String },
    /// Ready in principle but backing off — retry after the hint. Unlike
    /// `Skip` (a standing condition), `Wait` is transient.
    Wait {
        reason: String,
        retry_after_secs: u64,
    },
}

impl SensorOutcome {
    /// One-line status tag for the CLI.
    pub fn tag(&self) -> &'static str {
        match self {
            SensorOutcome::Ready => "READY",
            SensorOutcome::Skip { .. } => "SKIP",
            SensorOutcome::Wait { .. } => "WAIT",
        }
    }
    pub fn reason(&self) -> &str {
        match self {
            SensorOutcome::Ready => "",
            SensorOutcome::Skip { reason } | SensorOutcome::Wait { reason, .. } => reason,
        }
    }
}

/// A named, evaluable observation of external state.
pub trait Sensor: Send + Sync {
    /// Stable identifier (used by `blut sensor eval <name>`).
    fn name(&self) -> &'static str;
    /// One-line human description for `blut sensor list`.
    fn description(&self) -> &'static str;
    /// Observe + decide. Pure read — a sensor NEVER mutates on-disk state.
    fn evaluate(&self) -> SensorOutcome;
}

/// The cross-process GPU scheduler lock as a sensor: `Ready` when the
/// lock is free (or stale), `Wait` when a LIVE process holds it (it will
/// free up — retry). This is exactly the observation
/// [`policy::decide`](crate::policy::decide)'s last gate consults.
pub struct SchedulerLockSensor;

impl Sensor for SchedulerLockSensor {
    fn name(&self) -> &'static str {
        "scheduler-lock"
    }
    fn description(&self) -> &'static str {
        "GPU scheduler lock — Ready when free, Wait when a live process holds it"
    }
    fn evaluate(&self) -> SensorOutcome {
        match crate::scheduler_lock::check_unlocked() {
            Ok(()) => SensorOutcome::Ready,
            // A held lock is transient (the holder finishes) → Wait, not
            // Skip. The error message already names the holder/pid/kind.
            Err(e) => SensorOutcome::Wait {
                reason: e.to_string(),
                retry_after_secs: 30,
            },
        }
    }
}

/// The auto-train [`policy`](crate::policy) gates as a sensor. Reports
/// whether the policy currently permits an auto-run — evaluating every
/// gate BLUT can observe locally (enabled, quiet-hours, cooldown,
/// failure-backoff, and the GPU lock). The new-turn-count gate is the
/// one input BLUT can't see (it lives in the lamu conversation store),
/// so this sensor assumes "enough turns" and reports the status of the
/// REMAINING gates — i.e. "would the policy fire right now, given new
/// data?". `Ready` = all observable gates pass.
pub struct AutoTrainPolicySensor;

impl Sensor for AutoTrainPolicySensor {
    fn name(&self) -> &'static str {
        "auto-train-policy"
    }
    fn description(&self) -> &'static str {
        "Auto-train policy gates (enabled / quiet-hours / cooldown / backoff / lock), \
         assuming enough new turns"
    }
    fn evaluate(&self) -> SensorOutcome {
        let policy = match crate::policy::load() {
            Ok(p) => p,
            Err(e) => {
                return SensorOutcome::Skip {
                    reason: format!("policy unreadable: {e}"),
                };
            }
        };
        let (now_secs, now_mins) = crate::policy::current_clock();
        let lock_held = crate::scheduler_lock::check_unlocked().is_err();
        // Assume plenty of new data so the threshold gate passes and we
        // observe the TIME + lock gates (the ones BLUT can evaluate without
        // the conversation store).
        match crate::policy::decide(&policy, now_secs, now_mins, ASSUME_ENOUGH_TURNS, lock_held) {
            crate::policy::Decision::Run { .. } => SensorOutcome::Ready,
            crate::policy::Decision::Skip(reason) => SensorOutcome::Skip { reason },
        }
    }
}

/// The named sensors this binary knows about. A static list (like the
/// cookbook registry) — extend by adding a constructor here; cookbooks
/// with domain sensors compose their own list in a later phase.
pub fn registry() -> Vec<Box<dyn Sensor>> {
    vec![
        Box::new(SchedulerLockSensor),
        Box::new(AutoTrainPolicySensor),
    ]
}

/// Look up a sensor by name.
pub fn find(name: &str) -> Option<Box<dyn Sensor>> {
    registry().into_iter().find(|s| s.name() == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_named_sensors_uniquely() {
        let names: Vec<&str> = registry().iter().map(|s| s.name()).collect();
        assert!(names.contains(&"scheduler-lock"));
        assert!(names.contains(&"auto-train-policy"));
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "sensor names must be unique");
    }

    #[test]
    fn find_resolves_known_and_rejects_unknown() {
        assert!(find("scheduler-lock").is_some());
        assert!(find("auto-train-policy").is_some());
        assert!(find("no-such-sensor").is_none());
    }

    #[test]
    fn every_sensor_evaluates_to_a_typed_outcome() {
        // Smoke: each sensor returns SOME outcome without panicking
        // (the real state of the box is irrelevant — we only assert the
        // contract holds + the tag/reason accessors work).
        for s in registry() {
            let o = s.evaluate();
            assert!(!o.tag().is_empty());
            // Ready carries no reason; Skip/Wait carry one.
            match &o {
                SensorOutcome::Ready => assert_eq!(o.reason(), ""),
                SensorOutcome::Skip { .. } | SensorOutcome::Wait { .. } => {
                    assert!(!o.reason().is_empty(), "{} gave empty reason", s.name())
                }
            }
        }
    }

    #[test]
    fn outcome_serializes_with_tag() {
        let ready = serde_json::to_value(SensorOutcome::Ready).unwrap();
        assert_eq!(ready["outcome"], "ready");
        let wait = serde_json::to_value(SensorOutcome::Wait {
            reason: "held".into(),
            retry_after_secs: 30,
        })
        .unwrap();
        assert_eq!(wait["outcome"], "wait");
        assert_eq!(wait["retry_after_secs"], 30);
    }
}
