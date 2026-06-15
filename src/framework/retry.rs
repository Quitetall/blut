//! Per-stage retry policy (D1) + stage/plan timeouts (D2).
//!
//! A `Stage` declares `const RETRY` and `const TIMEOUT`; a plan can
//! override either per-node via `Plan::with_retry` / `Plan::with_timeout`.
//! The executor's `run_node` consults the resolved values: it retries a
//! TRANSIENT failure up to `max_attempts` with a backoff (cancellable),
//! and bounds each attempt by an optional soft/hard timeout.
//!
//! All types are `Copy` + const-constructible so they fit the existing
//! `Stage` associated-const pattern with zero allocation.

use std::sync::Arc;
use std::time::Duration;

use crate::framework::error::StageError;

/// The `ExecCtx.on_retry` hook: invoked on a retryable stage failure
/// before the backoff. The cookbook wires the broker's OOM escalation.
pub type RetryHook = Arc<dyn Fn(&RetryEvent) + Send + Sync>;

/// How many attempts, with what backoff, and which errors to retry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetryPolicy {
    /// Total attempts INCLUDING the first. `1` = no retry.
    pub max_attempts: u32,
    pub backoff: Backoff,
    pub retry_on: RetryOn,
}

impl RetryPolicy {
    /// The default: a single attempt, no retry.
    pub const NONE: RetryPolicy = RetryPolicy {
        max_attempts: 1,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };

    /// `n` attempts with exponential backoff on transient errors. A
    /// convenient constructor for stages that want retries.
    pub const fn transient(max_attempts: u32, base: Duration) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            backoff: Backoff::Exponential {
                base,
                mult_x100: 200, // ×2 per attempt
                cap: Duration::from_secs(300),
            },
            retry_on: RetryOn::Transient,
        }
    }

    /// `n` attempts retrying ONLY `OutOfMemory`, exponential backoff. For
    /// contained train stages: each retry re-resolves the broker cap (escalated
    /// above the OomCorrected bound the prior attempt recorded), so an undersize
    /// cap self-heals within ONE `recipe run` instead of dying.
    pub const fn on_oom(max_attempts: u32, base: Duration) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            backoff: Backoff::Exponential {
                base,
                mult_x100: 200,
                cap: Duration::from_secs(300),
            },
            retry_on: RetryOn::OutOfMemoryOnly,
        }
    }

    /// Backoff `Duration` BEFORE attempt number `attempt` (1-based; the
    /// first attempt has no backoff). `Exponential` grows `base ×
    /// (mult/100)^(attempt-2)`, capped.
    pub fn backoff_before(&self, attempt: u32) -> Duration {
        if attempt <= 1 {
            return Duration::ZERO;
        }
        match self.backoff {
            Backoff::None => Duration::ZERO,
            Backoff::Fixed(d) => d,
            Backoff::Exponential {
                base,
                mult_x100,
                cap,
            } => {
                // base * (mult/100)^(attempt-2), saturating, capped.
                let mut ms = base.as_millis() as u64;
                for _ in 0..attempt.saturating_sub(2) {
                    ms = ms.saturating_mul(mult_x100 as u64) / 100;
                }
                Duration::from_millis(ms).min(cap)
            }
        }
    }
}

/// Backoff schedule between attempts. `mult_x100` is the growth factor
/// ×100 (200 = ×2) so the whole thing stays `const`-friendly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backoff {
    None,
    Fixed(Duration),
    Exponential {
        base: Duration,
        mult_x100: u32,
        cap: Duration,
    },
}

/// Which errors a policy retries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RetryOn {
    /// Only transient failures (backend/io/timeout/OOM). Deterministic
    /// errors (bad input, kind/schema mismatch, deserialize, cancelled)
    /// re-fail identically, so they are NEVER retried regardless.
    Transient,
    /// Any error except `Cancelled` (a cancel always wins).
    AllErrors,
    /// ONLY `OutOfMemory`. For contained train stages where a retry is
    /// meaningful *only* because the broker escalates the cgroup cap on the
    /// next attempt — a non-OOM failure reproduces identically, so retrying it
    /// would just waste the (expensive) train startup.
    OutOfMemoryOnly,
}

/// Is `err` worth retrying under `policy`? `Cancelled` is never
/// retried; deterministic input/contract errors are never retried even
/// under `AllErrors` (they reproduce on every attempt).
pub fn is_retryable(err: &StageError, policy: RetryOn) -> bool {
    // Deterministic / contract errors re-fail identically — never retry.
    let deterministic = matches!(
        err,
        StageError::Cancelled
            | StageError::KindMismatch { .. }
            | StageError::BadInput(_)
            | StageError::InputDeserialize { .. }
            | StageError::ArgsDeserialize { .. }
            | StageError::OutputSerialize { .. }
    );
    if deterministic {
        return false;
    }
    // Transient classes always retry; anything else only under AllErrors
    // (keeps the policy meaningful for future StageError variants). `Diverged`
    // is transient: the retry resumes from the last good checkpoint (S3) and may
    // recover — but NOT under `OutOfMemoryOnly` (a divergence is not an OOM, so a
    // stage wanting divergence-retry uses `Transient`).
    let transient = matches!(
        err,
        StageError::Backend(_)
            | StageError::Io { .. }
            | StageError::ResourceTimeout(_)
            | StageError::Timeout { .. }
            | StageError::OutOfMemory { .. }
            | StageError::Diverged { .. }
    );
    match policy {
        // Self-heal only the OOM (the next attempt's cap is escalated);
        // everything else reproduces identically, so don't waste a retry.
        RetryOn::OutOfMemoryOnly => matches!(err, StageError::OutOfMemory { .. }),
        RetryOn::Transient => transient,
        RetryOn::AllErrors => true,
    }
}

/// Per-stage soft/hard timeout (D2). `soft` fires the stage's
/// (cooperative) cancel token — a Python trainer SIGTERMs its child;
/// `hard` drops the run future outright (kill_on_drop reaps the
/// subprocess). A stage that returns Ok AFTER its soft timeout fired is
/// failed with `StageError::Timeout` and NOT promoted (a possibly-
/// degraded artifact must not poison the cache).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageTimeout {
    /// Cooperative deadline: fires the stage's cancel token. ONLY
    /// effective for stages that observe cancellation (subprocess
    /// stages, or anything that `select!`s on `ctx.cancel`). A purely
    /// CPU-bound stage with no yield/cancel points will NOT stop on a
    /// soft timeout — pair it with `hard` for a guaranteed ceiling.
    pub soft: Option<Duration>,
    /// Hard deadline: drops the run future outright (the only ceiling
    /// that bounds a non-cooperative stage).
    pub hard: Option<Duration>,
}

impl StageTimeout {
    pub const NONE: StageTimeout = StageTimeout {
        soft: None,
        hard: None,
    };

    pub fn is_set(&self) -> bool {
        self.soft.is_some() || self.hard.is_some()
    }
}

/// Context passed to an `ExecCtx.on_retry` hook when a stage attempt
/// fails with a retryable error, BEFORE the backoff sleep. The cookbook
/// (cli) wires a closure that, on `OutOfMemory`, records an
/// `OomCorrected` calibration so the next attempt's memory cap is
/// escalated — the framework stays broker-agnostic.
#[derive(Clone, Debug)]
pub struct RetryEvent {
    pub stage_name: String,
    pub recipe_name: String,
    /// The attempt that just failed (1-based).
    pub attempt: u32,
    pub max_attempts: u32,
    /// `true` if the failure was `StageError::OutOfMemory`.
    pub was_oom: bool,
    /// `Display` of the failing error.
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_single_attempt() {
        assert_eq!(RetryPolicy::NONE.max_attempts, 1);
        assert_eq!(RetryPolicy::NONE.backoff_before(1), Duration::ZERO);
    }

    #[test]
    fn exponential_grows_and_caps() {
        let p = RetryPolicy {
            max_attempts: 10,
            backoff: Backoff::Exponential {
                base: Duration::from_millis(100),
                mult_x100: 200,
                cap: Duration::from_millis(1000),
            },
            retry_on: RetryOn::Transient,
        };
        assert_eq!(p.backoff_before(1), Duration::ZERO); // first attempt
        assert_eq!(p.backoff_before(2), Duration::from_millis(100)); // base
        assert_eq!(p.backoff_before(3), Duration::from_millis(200)); // ×2
        assert_eq!(p.backoff_before(4), Duration::from_millis(400)); // ×4
        assert_eq!(p.backoff_before(10), Duration::from_millis(1000)); // capped
    }

    #[test]
    fn deterministic_errors_never_retry() {
        assert!(!is_retryable(
            &StageError::BadInput("x".into()),
            RetryOn::AllErrors
        ));
        assert!(!is_retryable(&StageError::Cancelled, RetryOn::AllErrors));
    }

    #[test]
    fn transient_errors_retry() {
        use crate::framework::resource::Resource;
        assert!(is_retryable(
            &StageError::ResourceTimeout(Resource::Gpu),
            RetryOn::Transient
        ));
        assert!(is_retryable(
            &StageError::OutOfMemory {
                detail: "cuda".into()
            },
            RetryOn::Transient
        ));
    }

    #[test]
    fn diverged_is_transient_but_not_oom_only() {
        let div = StageError::Diverged { detail: "loss nan".into() };
        // Transient (resume on the next attempt) under Transient + AllErrors…
        assert!(is_retryable(&div, RetryOn::Transient));
        assert!(is_retryable(&div, RetryOn::AllErrors));
        // …but a divergence is NOT an OOM, so OutOfMemoryOnly does not retry it
        // (a stage wanting divergence-retry must use Transient).
        assert!(!is_retryable(&div, RetryOn::OutOfMemoryOnly));
    }

    #[test]
    fn oom_only_retries_oom_not_other_transients() {
        let oom = StageError::OutOfMemory { detail: "cgroup".into() };
        let backend = StageError::Backend(anyhow::anyhow!("crashed"));
        // OutOfMemoryOnly: the OOM self-heals (escalated cap next attempt);
        // a non-OOM backend crash reproduces, so it must NOT retry.
        assert!(is_retryable(&oom, RetryOn::OutOfMemoryOnly));
        assert!(!is_retryable(&backend, RetryOn::OutOfMemoryOnly));
        // Transient still retries both; OutOfMemoryOnly never retries a
        // deterministic error.
        assert!(is_retryable(&backend, RetryOn::Transient));
        assert!(!is_retryable(
            &StageError::Cancelled,
            RetryOn::OutOfMemoryOnly
        ));
    }
}
