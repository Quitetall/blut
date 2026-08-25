// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0106's software acceptance gate: a leader kill mid-DAG must not
//! double-dispatch, must not lose a committed result, and must fail closed when
//! quorum is gone.
//!
//! `ha_failover_no_double_dispatch` is the test the ADR names by that exact
//! string. It is deliberately one test rather than four, because the ADR's
//! `pass when` is one conjunction — a follower wins, adopts by content hash,
//! re-dispatches ONLY orphans, loses nothing, and halts without quorum. Split
//! apart, three could pass while the fourth silently regressed and the gate
//! would still be "green".
//!
//! Everything here is deterministic: a fixed member set, per-node election
//! timeouts, and an explicit FIFO message queue that the test single-steps. No
//! sleeps and no wall-clock, so a failure is a failure every run.
#![cfg(feature = "raft")]

use blut::raft::harness::Cluster;
use blut::raft::{DispatchDecision, PeerReport, PeerStageState, ReconcileAction, StageStatus};

/// Replicate until followers have caught up on BOTH the entries and the commit
/// index.
///
/// One round is never enough, and the reason is worth stating: a leader learns
/// an entry is committed from the followers' `AppendEntriesResp`, but the
/// followers learn it only from the `leader_commit` on the NEXT
/// `AppendEntries`. So a commit is always at least two round trips from being
/// visible cluster-wide. A test that ticked once and asserted would be
/// measuring that lag, not the property.
fn replicate(cluster: &mut Cluster, rounds: usize) {
    for _ in 0..rounds {
        cluster.tick_all();
        cluster.deliver_all();
    }
}

/// Drive a dispatch to the point where it is authorised, asserting the
/// commit-before-dispatch contract along the way.
///
/// Returns the peer the caller is now permitted to send to. A test that just
/// called `request_dispatch` and read the peer out would not be checking the
/// rule at all — the rule is that nothing sendable exists until quorum commits.
fn dispatch_now(cluster: &mut Cluster, stage: &str, peer: &str, expected: Option<&str>) -> String {
    let leader = cluster
        .leader_id()
        .expect("a leader must exist to dispatch");
    let decision =
        cluster
            .node(&leader)
            .request_dispatch(stage, peer, expected.map(str::to_string));

    let index = match decision {
        // A 1-member or already-replicated quorum can commit synchronously.
        DispatchDecision::Committed { peer, .. } => return peer,
        DispatchDecision::AwaitingQuorum { index, .. } => index,
        other => panic!("expected a proposal for '{stage}', got {other:?}"),
    };

    // Not yet authorised: replicate, then re-poll. This loop IS the contract.
    for _ in 0..10 {
        cluster.tick_all();
        cluster.deliver_all();
        if let DispatchDecision::Committed { peer, .. } =
            cluster.node(&leader).poll_dispatch(stage, index)
        {
            return peer;
        }
    }
    panic!("dispatch of '{stage}' never reached quorum commit");
}

#[test]
fn ha_failover_no_double_dispatch() {
    // --- three scheduler-capable nodes, operator-declared ---
    let mut cluster = Cluster::new(&["n1", "n2", "n3"], 3);
    cluster.settle(20);
    let first_leader = cluster.leader_id().expect("an initial leader");

    // --- a small DAG ---
    for stage in ["prepare", "train_a", "train_b", "report"] {
        cluster.node(&first_leader).enqueue(stage);
    }
    replicate(&mut cluster, 3);

    // --- dispatch three stages; the decisions are quorum-committed BEFORE any
    //     task frame would go out ---
    dispatch_now(&mut cluster, "prepare", "worker-1", Some("hash-prepare"));
    dispatch_now(&mut cluster, "train_a", "worker-2", Some("hash-train-a"));
    dispatch_now(&mut cluster, "train_b", "worker-3", Some("hash-train-b"));

    // `prepare` finishes and the result is replicated.
    cluster
        .node(&first_leader)
        .record_complete("prepare", "hash-prepare");
    // `train_a` reports it is running.
    cluster
        .node(&first_leader)
        .record_running("train_a", "worker-2");
    replicate(&mut cluster, 3);

    // A committed result must be on a majority before we rely on surviving.
    let replicated = cluster
        .nodes
        .iter()
        .filter(|n| n.state().get("prepare").map(|r| r.status) == Some(Some(StageStatus::Done)))
        .count();
    assert!(
        replicated >= 2,
        "the completed result reached only {replicated} node(s); a majority must \
         hold it or surviving the leader proves nothing"
    );

    // ================= kill the leader mid-DAG =================
    cluster.kill(&first_leader);
    cluster.settle(40);
    let new_leader = cluster
        .leader_id()
        .expect("a follower must win the election");
    assert_ne!(new_leader, first_leader, "the dead node cannot still lead");

    // --- no committed result was lost ---
    let prepare = cluster
        .get(&new_leader)
        .state()
        .get("prepare")
        .cloned()
        .expect("the new leader must know about 'prepare'");
    assert_eq!(
        prepare.status,
        Some(StageStatus::Done),
        "a quorum-committed result was lost across failover"
    );
    assert_eq!(prepare.content_hash.as_deref(), Some("hash-prepare"));

    // --- reconcile: ask the live peers, then decide ---
    // worker-2 finished train_a while the leader was dying; worker-3 vanished
    // with train_b. This is the case the ADR is about: identical log state
    // (both committed-and-unresolved), opposite correct actions.
    let reports = vec![
        PeerReport {
            stage: "train_a".into(),
            peer: "worker-2".into(),
            state: PeerStageState::Complete {
                content_hash: "hash-train-a".into(),
            },
        },
        PeerReport {
            stage: "train_b".into(),
            peer: "worker-3".into(),
            state: PeerStageState::Absent,
        },
    ];
    let actions = cluster.node(&new_leader).reconcile(&reports);

    assert!(
        actions.contains(&ReconcileAction::Adopt {
            stage: "train_a".into(),
            content_hash: "hash-train-a".into(),
        }),
        "a peer's finished result must be ADOPTED by content hash, not re-run: {actions:?}"
    );
    assert!(
        actions.contains(&ReconcileAction::Redispatch {
            stage: "train_b".into(),
        }),
        "an orphan with no live owner and no result must be re-dispatched: {actions:?}"
    );
    assert_eq!(
        actions.len(),
        2,
        "only the two unresolved stages may be acted on; 'prepare' was already \
         complete and 'report' was never dispatched: {actions:?}"
    );

    cluster.node(&new_leader).commit_reconcile(&actions);
    replicate(&mut cluster, 3);

    // The adopted stage is now a committed result — no second dispatch.
    assert_eq!(
        cluster
            .get(&new_leader)
            .state()
            .get("train_a")
            .and_then(|r| r.status),
        Some(StageStatus::Done),
        "the adoption must itself be replicated, or it dies with this leader too"
    );

    // The genuine orphan is re-dispatched, and only now.
    dispatch_now(&mut cluster, "train_b", "worker-4", Some("hash-train-b"));

    // Attempt to re-dispatch a stage that is LIVE (committed as dispatched,
    // no result). This is the case the counter below exists to catch: without
    // the guard this commits a second Dispatch entry and `train_b` reaches two
    // simultaneously-effective dispatches — two workers, one stage.
    let live_repeat = cluster
        .node(&new_leader)
        .request_dispatch("train_b", "worker-9", None);
    assert!(
        matches!(live_repeat, DispatchDecision::AlreadyDispatched { .. }),
        "a live stage must not be dispatched again, got {live_repeat:?}"
    );
    replicate(&mut cluster, 3);

    // ============ THE CENTRAL ASSERTION ============
    // Every stage was dispatched at most once across the entire run, including
    // across the failover. `train_b` moved worker but was never dispatched
    // twice, because the first decision was superseded, not duplicated.
    assert_eq!(
        cluster.max_effective_dispatches(),
        1,
        "a stage was dispatched more than once across the run — commit-before-\
         dispatch or reconcile has a hole: {:?}",
        cluster
            .get(&new_leader)
            .state()
            .stages()
            .map(|(n, r)| (n.clone(), r.dispatch_count))
            .collect::<Vec<_>>()
    );

    // Re-dispatching an already-committed stage is refused outright.
    let repeat = cluster
        .node(&new_leader)
        .request_dispatch("prepare", "worker-9", None);
    assert!(
        matches!(repeat, DispatchDecision::AlreadyDispatched { .. }),
        "a completed stage must never be dispatched again, got {repeat:?}"
    );

    // ============ FAIL-CLOSED ON QUORUM LOSS ============
    // Kill a second node: 1 of 3 is not a majority, so dispatch must STOP
    // rather than proceed and risk a split-brain double-dispatch.
    let survivor = new_leader.clone();
    let third = ["n1", "n2", "n3"]
        .into_iter()
        .find(|id| *id != first_leader && *id != survivor)
        .expect("a third member");
    cluster.kill(third);
    cluster.run(10);

    assert!(
        !cluster.get(&survivor).has_quorum(),
        "1 of 3 members must not count as a quorum"
    );
    let starved = cluster
        .node(&survivor)
        .request_dispatch("report", "worker-5", None);
    assert!(
        matches!(
            starved,
            DispatchDecision::NoQuorum | DispatchDecision::NotLeader { .. }
        ),
        "dispatch without a quorum must be refused (fail-closed), got {starved:?}"
    );
    assert_eq!(
        cluster
            .get(&survivor)
            .state()
            .get("report")
            .and_then(|r| r.status),
        Some(StageStatus::Queued),
        "the refused stage must not have been dispatched"
    );
}

/// The gate above proves the happy path of adoption. This proves the SAFETY
/// side of it: a result whose hash contradicts the committed expectation is not
/// adopted.
///
/// Without this, "adopt by content hash" degrades into "adopt whatever a peer
/// hands you", and the idempotency argument — that a matching key *is* the same
/// artifact — stops holding.
#[test]
fn a_result_that_contradicts_the_committed_hash_is_not_adopted() {
    let mut cluster = Cluster::new(&["n1", "n2", "n3"], 3);
    cluster.settle(20);
    let leader = cluster.leader_id().unwrap();

    cluster.node(&leader).enqueue("train");
    replicate(&mut cluster, 3);
    dispatch_now(&mut cluster, "train", "worker-1", Some("expected-hash"));

    let actions = cluster.node(&leader).reconcile(&[PeerReport {
        stage: "train".into(),
        peer: "worker-1".into(),
        state: PeerStageState::Complete {
            content_hash: "a-different-hash".into(),
        },
    }]);

    assert_eq!(
        actions,
        vec![ReconcileAction::RejectedMismatch {
            stage: "train".into(),
            expected: "expected-hash".into(),
            reported: "a-different-hash".into(),
        }],
        "a mismatched content hash must be refused, not adopted"
    );
    cluster.node(&leader).commit_reconcile(&actions);
    assert_ne!(
        cluster
            .get(&leader)
            .state()
            .get("train")
            .and_then(|r| r.status),
        Some(StageStatus::Done),
        "a rejected result must not mark the stage complete"
    );
}

/// A single-member cluster must behave exactly as before this ADR: it is its
/// own majority, so there is no quorum wait and no behaviour change.
///
/// ADR 0106 promises this explicitly in its Consequences, and it is the case
/// most likely to regress unnoticed, since every other test runs three nodes.
#[test]
fn a_single_member_cluster_commits_without_waiting() {
    let mut cluster = Cluster::new(&["solo"], 2);
    cluster.settle(10);
    assert!(cluster.get("solo").is_leader());

    cluster.node("solo").enqueue("only_stage");
    let decision = cluster
        .node("solo")
        .request_dispatch("only_stage", "worker-1", None);
    assert!(
        matches!(decision, DispatchDecision::Committed { .. }),
        "a 1-member cluster is its own majority and must commit synchronously, \
         got {decision:?}"
    );
    assert!(cluster.get("solo").has_quorum());
}

/// Two nodes cannot both be leader in the same term.
///
/// This is the invariant everything else rests on: if it breaks, two schedulers
/// dispatch concurrently and no amount of reconcile logic saves you.
#[test]
fn at_most_one_leader_per_term() {
    let mut cluster = Cluster::new(&["n1", "n2", "n3"], 3);
    for _ in 0..60 {
        cluster.tick_all();
        cluster.deliver_all();
        let mut by_term: std::collections::BTreeMap<u64, Vec<String>> = Default::default();
        for node in cluster.nodes.iter().filter(|n| n.is_leader()) {
            by_term
                .entry(node.term())
                .or_default()
                .push(node.id().to_string());
        }
        for (term, leaders) in by_term {
            assert!(
                leaders.len() <= 1,
                "term {term} had {} leaders: {leaders:?}",
                leaders.len()
            );
        }
    }
}
