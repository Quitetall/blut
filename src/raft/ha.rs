// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Commit-before-dispatch and reconcile-not-re-dispatch (ADR 0106).
//!
//! The two rules that make failover safe, and why each is shaped this way:
//!
//! **Commit before dispatch.** A dispatch decision is a quorum-committed log
//! entry BEFORE the task frame leaves. The API therefore cannot hand you
//! something sendable until the entry commits — [`DispatchDecision::Committed`]
//! is the only variant carrying the peer to send to. If the call returned the
//! task and left "wait for commit" to the caller, the one line that matters
//! would be a convention, and a convention is what gets dropped in a refactor.
//!
//! **Reconcile, do not re-dispatch.** A new leader inherits a log full of
//! stages committed as dispatched with no committed result. It cannot know from
//! the log alone whether they ran. So it ASKS the live peers, adopts anything
//! already complete by content hash (idempotent — the key is content-addressed,
//! so a matching result *is* the result regardless of who produced it), and
//! re-dispatches only what has neither a live owner nor a result.
//!
//! **Fail-closed on quorum loss.** Losing quorum stops dispatch. That is a
//! deliberate availability sacrifice from ADR 0106's Consequences: a scheduler
//! that keeps dispatching without consensus is a scheduler that can split-brain
//! and double-dispatch, which is the failure this whole ADR exists to close.

use super::consensus::{LogIndex, MemberId, Outbound, RaftMessage, RaftNode, Role};
use super::state::{SchedulerCommand, SchedulerState, StageStatus};

/// What the scheduler is permitted to do about one stage right now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchDecision {
    /// Quorum-committed. This is the ONLY variant that authorises sending a
    /// task frame, and it carries the peer precisely so that "did we commit
    /// first?" is not a question the caller can get wrong.
    Committed {
        stage: String,
        peer: String,
        index: LogIndex,
    },
    /// Proposed, not yet committed. Nothing may be sent.
    AwaitingQuorum { stage: String, index: LogIndex },
    /// This node is not the leader; dispatch is not its decision to make.
    NotLeader { leader_hint: Option<String> },
    /// Quorum is unreachable. Fail-closed by design.
    NoQuorum,
    /// A dispatch for this stage is already committed. Returning this rather
    /// than committing a second entry is the first line of defence against
    /// double dispatch; the counter in the replicated state is the proof.
    AlreadyDispatched { stage: String, peer: String },
}

/// What a live peer says about a stage during reconciliation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeerStageState {
    /// The peer holds a finished result with this content hash.
    Complete { content_hash: String },
    /// The peer is still working on it.
    Running,
    /// The peer knows nothing about it — it never arrived, or the peer is gone.
    Absent,
}

/// One peer's answer about one stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerReport {
    pub stage: String,
    pub peer: String,
    pub state: PeerStageState,
}

/// What a new leader decided to do about an inherited stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconcileAction {
    /// A peer already produced a result; take it. No re-run, no second bill.
    Adopt { stage: String, content_hash: String },
    /// A live owner is still working; leave it alone.
    KeepRunning { stage: String, peer: String },
    /// No live owner and no result — this one is genuinely orphaned.
    Redispatch { stage: String },
    /// A peer reported a result whose hash contradicts the committed
    /// expectation. NOT adopted: a content-addressed key that does not match is
    /// not the same artifact, and silently taking it would defeat the identity
    /// the adoption argument rests on.
    RejectedMismatch {
        stage: String,
        expected: String,
        reported: String,
    },
}

/// A scheduler whose dispatch decisions are replicated.
pub struct HaScheduler {
    raft: RaftNode<SchedulerCommand>,
    state: SchedulerState,
    /// Members currently believed reachable, including self. Fail-closed
    /// dispatch is decided against this.
    reachable: Vec<MemberId>,
    members: Vec<MemberId>,
}

impl HaScheduler {
    pub fn new(id: impl Into<MemberId>, members: Vec<MemberId>, election_timeout: u64) -> Self {
        let id = id.into();
        let raft = RaftNode::new(id, members.clone(), election_timeout);
        Self {
            raft,
            state: SchedulerState::new(),
            reachable: members.clone(),
            members,
        }
    }

    pub fn id(&self) -> &str {
        self.raft.id()
    }
    pub fn role(&self) -> Role {
        self.raft.role()
    }
    pub fn is_leader(&self) -> bool {
        self.raft.is_leader()
    }
    pub fn term(&self) -> u64 {
        self.raft.term()
    }
    pub fn state(&self) -> &SchedulerState {
        &self.state
    }
    pub fn leader_hint(&self) -> Option<&str> {
        self.raft.leader_hint()
    }

    /// Declare which members are reachable. A node absent from this list is
    /// treated as partitioned for the purpose of the fail-closed check.
    pub fn set_reachable(&mut self, reachable: Vec<MemberId>) {
        self.reachable = reachable;
    }

    /// Strict majority of the DECLARED members — never of the reachable ones.
    fn quorum(&self) -> usize {
        self.members.len() / 2 + 1
    }

    /// Whether a quorum is currently reachable.
    pub fn has_quorum(&self) -> bool {
        let live = self
            .members
            .iter()
            .filter(|m| self.reachable.contains(m))
            .count();
        live >= self.quorum()
    }

    pub fn tick(&mut self) -> Vec<Outbound<SchedulerCommand>> {
        let out = self.raft.tick();
        self.pump();
        out
    }

    pub fn handle(
        &mut self,
        message: RaftMessage<SchedulerCommand>,
    ) -> Vec<Outbound<SchedulerCommand>> {
        let out = self.raft.handle(message);
        self.pump();
        out
    }

    /// Fold any newly committed entries into the state machine.
    fn pump(&mut self) {
        self.state.apply_prefix(self.raft.committed_entries());
    }

    /// Record that a stage exists.
    pub fn enqueue(&mut self, stage: &str) -> Option<LogIndex> {
        let index = self.raft.propose(SchedulerCommand::Enqueue {
            stage: stage.to_string(),
        });
        self.pump();
        index
    }

    /// Ask permission to dispatch `stage` to `peer`.
    ///
    /// Returns [`DispatchDecision::Committed`] only once the decision is
    /// durable on a quorum. Anything else means: do not send.
    pub fn request_dispatch(
        &mut self,
        stage: &str,
        peer: &str,
        expected: Option<String>,
    ) -> DispatchDecision {
        if !self.raft.is_leader() {
            return DispatchDecision::NotLeader {
                leader_hint: self.raft.leader_hint().map(str::to_string),
            };
        }
        // Fail-closed BEFORE proposing: a leader that has lost contact with a
        // majority must not even append, or it accumulates decisions it cannot
        // commit and that a real leader may contradict.
        if !self.has_quorum() {
            return DispatchDecision::NoQuorum;
        }
        if let Some(record) = self.state.get(stage) {
            if matches!(
                record.status,
                Some(StageStatus::Dispatched) | Some(StageStatus::Running)
            ) {
                return DispatchDecision::AlreadyDispatched {
                    stage: stage.to_string(),
                    peer: record.peer.clone().unwrap_or_default(),
                };
            }
            if record.status == Some(StageStatus::Done) {
                return DispatchDecision::AlreadyDispatched {
                    stage: stage.to_string(),
                    peer: record.peer.clone().unwrap_or_default(),
                };
            }
        }

        let Some(index) = self.raft.propose(SchedulerCommand::Dispatch {
            stage: stage.to_string(),
            peer: peer.to_string(),
            expected,
        }) else {
            return DispatchDecision::NotLeader {
                leader_hint: self.raft.leader_hint().map(str::to_string),
            };
        };
        self.pump();
        if self.raft.commit_index() >= index {
            DispatchDecision::Committed {
                stage: stage.to_string(),
                peer: peer.to_string(),
                index,
            }
        } else {
            DispatchDecision::AwaitingQuorum {
                stage: stage.to_string(),
                index,
            }
        }
    }

    /// Re-check a previously proposed dispatch now that replication may have
    /// advanced. Same contract: `Committed` is the only authorisation.
    pub fn poll_dispatch(&mut self, stage: &str, index: LogIndex) -> DispatchDecision {
        self.pump();
        if self.raft.commit_index() >= index
            && let Some(record) = self.state.get(stage)
            && let Some(peer) = &record.peer
        {
            return DispatchDecision::Committed {
                stage: stage.to_string(),
                peer: peer.clone(),
                index,
            };
        }
        DispatchDecision::AwaitingQuorum {
            stage: stage.to_string(),
            index,
        }
    }

    /// Replicate that a peer started a stage.
    pub fn record_running(&mut self, stage: &str, peer: &str) -> Option<LogIndex> {
        let index = self.raft.propose(SchedulerCommand::Running {
            stage: stage.to_string(),
            peer: peer.to_string(),
        });
        self.pump();
        index
    }

    /// Replicate a completed stage and its content hash.
    pub fn record_complete(&mut self, stage: &str, content_hash: &str) -> Option<LogIndex> {
        let index = self.raft.propose(SchedulerCommand::Complete {
            stage: stage.to_string(),
            content_hash: content_hash.to_string(),
        });
        self.pump();
        index
    }

    /// Replicate a cloud-queue job handle (ADR 0082), so a failover does not
    /// submit — and bill — a second job for the same work.
    pub fn record_cloud_job(&mut self, stage: &str, job_id: &str) -> Option<LogIndex> {
        let index = self.raft.propose(SchedulerCommand::CloudJob {
            stage: stage.to_string(),
            job_id: job_id.to_string(),
        });
        self.pump();
        index
    }

    /// Decide what to do about every inherited unresolved stage, given what the
    /// live peers report.
    ///
    /// Takes the reports rather than fetching them so the policy is testable
    /// without a network, and so the "ask the peers" step is visibly the
    /// caller's obligation rather than something buried in here.
    pub fn reconcile(&mut self, reports: &[PeerReport]) -> Vec<ReconcileAction> {
        self.pump();
        let mut actions = Vec::new();
        for stage in self
            .state
            .unresolved()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()
        {
            let record = self.state.get(&stage).cloned().unwrap_or_default();
            let report = reports.iter().find(|r| r.stage == stage);
            match report.map(|r| (&r.peer, &r.state)) {
                Some((_, PeerStageState::Complete { content_hash })) => match &record.expected {
                    Some(expected) if expected != content_hash => {
                        actions.push(ReconcileAction::RejectedMismatch {
                            stage: stage.clone(),
                            expected: expected.clone(),
                            reported: content_hash.clone(),
                        });
                    }
                    _ => actions.push(ReconcileAction::Adopt {
                        stage: stage.clone(),
                        content_hash: content_hash.clone(),
                    }),
                },
                Some((peer, PeerStageState::Running)) => {
                    actions.push(ReconcileAction::KeepRunning {
                        stage: stage.clone(),
                        peer: peer.clone(),
                    });
                }
                // No report at all is the same as an explicit Absent: nobody
                // live claims it. Treating silence as "probably still running"
                // is how a stage gets stranded forever.
                Some((_, PeerStageState::Absent)) | None => {
                    actions.push(ReconcileAction::Redispatch {
                        stage: stage.clone(),
                    });
                }
            }
        }
        actions
    }

    /// Commit a reconcile pass.
    ///
    /// Both outcomes are replicated facts, and BOTH must be, for the same
    /// reason: a decision only this leader knows about dies with this leader.
    /// An unreplicated adoption re-runs finished work; an unreplicated void
    /// leaves the next leader believing a dead peer still owns the stage, and
    /// the stage strands.
    ///
    /// `Redispatch` commits a `VoidDispatch` rather than a new `Dispatch` — the
    /// void records what reconciliation PROVED (that assignment never took
    /// effect) and returns the stage to `Queued`. The actual re-dispatch then
    /// goes through the ordinary `request_dispatch` path, so it is
    /// quorum-committed before it is sent, exactly like a first dispatch. A
    /// shortcut that dispatched directly from here would be the one dispatch in
    /// the system that skipped commit-before-dispatch.
    pub fn commit_reconcile(&mut self, actions: &[ReconcileAction]) {
        for action in actions {
            match action {
                ReconcileAction::Adopt {
                    stage,
                    content_hash,
                } => {
                    self.raft.propose(SchedulerCommand::Complete {
                        stage: stage.clone(),
                        content_hash: content_hash.clone(),
                    });
                }
                ReconcileAction::Redispatch { stage } => {
                    self.raft.propose(SchedulerCommand::VoidDispatch {
                        stage: stage.clone(),
                        reason: "no live owner and no committed result".into(),
                    });
                }
                // A live owner needs no entry: the committed state already says
                // so, and re-asserting it would just grow the log.
                ReconcileAction::KeepRunning { .. } => {}
                // A mismatch is NOT voided. The stage may still be running
                // somewhere, and the safe move is to leave the committed
                // assignment alone and surface it, not to silently re-dispatch
                // work whose identity we cannot account for.
                ReconcileAction::RejectedMismatch { .. } => {}
            }
        }
        self.pump();
    }
}
