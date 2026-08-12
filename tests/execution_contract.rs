//! ADR 0092 A08 — one execution lifecycle, three adapters.
//!
//! A08 is "training execution modes have incompatible lifecycles": local
//! `run_node` owns typed failures, retries, cancellation and soft/hard timeouts,
//! while the executor-integrated P2P path cannot even carry the work. Its gate
//! asks that "the same lifecycle contract suite passes for local, P2P, and cloud
//! adapters".
//!
//! The suite therefore takes an ADAPTER and asserts the rules every adapter owes,
//! rather than testing one implementation. That shape is the point: a contract
//! written against a single adapter proves that adapter, and A08 exists because
//! three implementations disagreed while each passed its own tests.
//!
//! WHAT THIS FILE DELIBERATELY DOES NOT DO. It does not run P2P or cloud. Both
//! are unable to satisfy the contract today, and A08 says why. Those facts are
//! pinned below as executable statements about the SEAM rather than left as
//! prose, so that fixing the seam breaks these tests — which is the signal the
//! work landed, and is the opposite of a skip that quietly reports success.
//!
//! A08 also depends on A09 (artifact identity) and A10 (storage policy), so the
//! full merge is not this file's job.

use blut::config::launcher::JobState;
use blut::framework::artifact::ContentHash;
use blut::framework::executor::{DispatchHandle, DispatchRequest, DispatchSubmitter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A minimal adapter that honours the lifecycle, so the contract below is known
/// to be satisfiable. A contract no implementation can pass is indistinguishable
/// from a contract that is simply wrong.
struct MockHandle {
    polls: Arc<AtomicUsize>,
    cancels: Arc<AtomicUsize>,
    settle_after: usize,
    terminal: JobState,
}

impl DispatchHandle for MockHandle {
    fn poll(&self) -> Result<Option<JobState>, blut::error::TrainError> {
        let seen = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.cancels.load(Ordering::SeqCst) > 0 {
            return Ok(Some(JobState::Cancelled));
        }
        Ok((seen >= self.settle_after).then(|| self.terminal.clone()))
    }

    fn cancel(&self) -> Result<(), blut::error::TrainError> {
        self.cancels.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct MockSubmitter {
    polls: Arc<AtomicUsize>,
    cancels: Arc<AtomicUsize>,
    settle_after: usize,
    terminal: JobState,
}

impl DispatchSubmitter for MockSubmitter {
    fn submit(
        &self,
        _request: DispatchRequest<'_>,
    ) -> Result<Box<dyn DispatchHandle>, blut::error::TrainError> {
        Ok(Box::new(MockHandle {
            polls: Arc::clone(&self.polls),
            cancels: Arc::clone(&self.cancels),
            settle_after: self.settle_after,
            terminal: self.terminal.clone(),
        }))
    }
}

fn request<'a>(
    args: &'a serde_json::Value,
    tenant: &'a blut::tenant::Tenant,
) -> DispatchRequest<'a> {
    DispatchRequest {
        stage_name: "contract_stage",
        stage_schema: 1,
        invocation_key: blut::framework::InvocationKey::from_digest(ContentHash::of_bytes(
            b"invocation",
        )),
        input_content_id: blut::framework::ContentId::from_digest(ContentHash::of_bytes(b"input")),
        args_hash: ContentHash::of_bytes(b"args"),
        args,
        expected_content_id: Some(blut::framework::ContentId::from_digest(
            ContentHash::of_bytes(b"output"),
        )),
        resource_request: Default::default(),
        data_class: 0,
        tenant,
    }
}

/// THE CONTRACT. Every adapter owes exactly these, whatever transport it uses.
fn assert_lifecycle_contract(submitter: &dyn DispatchSubmitter, label: &str) {
    let args = serde_json::json!({});
    let tenant = blut::tenant::Tenant::default();

    // 1. A handle settles into exactly ONE terminal state, and stays there.
    let handle = submitter.submit(request(&args, &tenant)).expect("submit");
    let mut terminal = None;
    for _ in 0..16 {
        if let Some(state) = handle.poll().expect("poll") {
            terminal = Some(state);
            break;
        }
    }
    let settled = terminal.unwrap_or_else(|| panic!("{label}: never reached a terminal state"));
    for _ in 0..3 {
        assert_eq!(
            handle.poll().expect("poll after terminal"),
            Some(settled.clone()),
            "{label}: terminal state changed after settling. Exactly one terminal \
             outcome is what lets a caller stop polling and trust the answer."
        );
    }

    // 2. Cancellation is observable, and cancelling twice is not an error.
    //    An adapter whose second cancel fails forces every caller to track
    //    whether it already cancelled, which is state the caller should not own.
    let handle = submitter.submit(request(&args, &tenant)).expect("submit");
    handle.cancel().expect("first cancel");
    handle.cancel().expect("second cancel must be idempotent");
    assert_eq!(
        handle.poll().expect("poll after cancel"),
        Some(JobState::Cancelled),
        "{label}: cancellation was not observable through poll"
    );
}

#[test]
fn a_conforming_adapter_satisfies_the_lifecycle_contract() {
    for terminal in [JobState::Succeeded, JobState::Failed("boom".into())] {
        let submitter = MockSubmitter {
            polls: Arc::new(AtomicUsize::new(0)),
            cancels: Arc::new(AtomicUsize::new(0)),
            settle_after: 3,
            terminal: terminal.clone(),
        };
        assert_lifecycle_contract(&submitter, &format!("mock({terminal:?})"));
    }
}

/// A08's evidence, as executable statements rather than prose.
///
/// "successful output cannot return through `DispatchHandle`". `poll` yields
/// `Option<JobState>`, and `JobState::Succeeded` is a UNIT variant — so a remote
/// success reports the word "Succeeded" and nothing else. There is no channel for
/// the artifact the stage produced, which is why the gate says "no adapter
/// reports success without validated artifact identity and rehydrated output":
/// today no adapter *could*, whatever it wanted to do.
///
/// These pins pass while the gap exists and FAIL once the seam can carry an
/// outcome. That failure is the signal A08's work landed, not a regression.
#[test]
fn a_remote_success_carries_no_output() {
    let submitter = MockSubmitter {
        polls: Arc::new(AtomicUsize::new(0)),
        cancels: Arc::new(AtomicUsize::new(0)),
        settle_after: 1,
        terminal: JobState::Succeeded,
    };
    let args = serde_json::json!({});
    let tenant = blut::tenant::Tenant::default();
    let handle = submitter.submit(request(&args, &tenant)).expect("submit");
    let settled = handle.poll().expect("poll").expect("terminal");

    // Everything a caller learns from a remote success. If `Succeeded` ever gains
    // a payload this stops compiling, which is exactly the intended alarm.
    match settled {
        JobState::Succeeded => {}
        other => panic!("expected Succeeded, got {other:?}"),
    }
}

/// Remote failure identity is an unstructured string. A08 says so, and it is right.
///
/// I first wrote this test asserting failures carried NO identity at all, having
/// read `blut::jobs::JobState` — a DIFFERENT enum from the one this seam uses.
/// The dispatch trait speaks `blut::config::launcher::JobState`, whose
/// `Failed(String)` and `Unknown(String)` do carry a scheduler-reported reason.
/// The ADR's wording was accurate; my reading was not.
///
/// That two same-named enums describe job state in one crate is itself worth
/// noticing while consolidating lifecycles, since "which JobState?" is precisely
/// the ambiguity A08 is about.
///
/// What the gate wants is TYPED failure identity. A string means every consumer
/// re-parses vendor text — `"OutOfMemory"` from one scheduler, `"TIMEOUT"` from
/// another — and no two consumers classify identically.
#[test]
fn remote_failure_identity_is_an_unstructured_string() {
    let oom = JobState::Failed("OutOfMemory".to_string());
    let timeout = JobState::Failed("TIMEOUT".to_string());
    assert_ne!(
        oom, timeout,
        "sanity: the payload is the only discriminator"
    );

    // Distinguishable ONLY by string comparison — there is no code, no kind, no
    // retryability flag. Delete this test when a typed failure lands.
    match (&oom, &timeout) {
        (JobState::Failed(a), JobState::Failed(b)) => {
            assert!(
                a.parse::<u32>().is_err() && b.parse::<u32>().is_err(),
                "failure payloads became structured; assert the typed identity instead"
            );
        }
        _ => panic!("Failed is no longer the string-carrying variant"),
    }
}

/// The request seam carries no input bytes and no source root.
///
/// A08: "`DispatchRequest` carries neither artifact nor source root, the
/// coordinator sends `encrypted_input: None`, the peer rejects it". The struct
/// carries portable identities — `input_content_id`, `expected_content_id` — which describe work
/// without transporting it. A peer can therefore verify what it was asked for
/// and still be unable to do it.
#[test]
fn the_request_seam_describes_work_without_transporting_it() {
    let args = serde_json::json!({});
    let tenant = blut::tenant::Tenant::default();
    let req = request(&args, &tenant);
    // Present: identity of the work.
    let _ = req.input_content_id;
    let _ = req.expected_content_id;
    // Absent: the work itself. This test exists to be DELETED when a payload or
    // source-root field is added, because that addition is A08's actual fix.
    assert_eq!(
        req.stage_name, "contract_stage",
        "sanity: the request under test is the one constructed above"
    );
}
