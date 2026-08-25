// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The Raft core: leader election, log replication, and commit (ADR 0106).
//!
//! Deliberately **deterministic and tick-driven**. There are no timers, no
//! threads, and no clock reads in here: a node advances only when the caller
//! calls [`RaftNode::tick`] or delivers a message. Consensus bugs are
//! interleaving bugs, and a gate that reproduces one only sometimes is not a
//! gate — ADR 0106's acceptance test drives three nodes through an exact
//! schedule, and it must fail every time the invariant breaks, not one run in
//! twenty.
//!
//! Randomised election timeouts are Raft's liveness mechanism against split
//! votes. Here the timeout is a per-node *constant* supplied by the caller,
//! which is the same thing with the randomness lifted out where a test can
//! choose it. Liveness is then the caller's problem (a real deployment jitters
//! the value); safety — at most one leader per term — does not depend on it.
//!
//! **Scope.** Election, replication, and commit. No snapshotting, no log
//! compaction, no joint-consensus membership change: ADR 0106 specifies an
//! operator-declared, fixed member list, so the one reconfiguration mechanism
//! Raft is subtle about is out of scope by decision rather than by omission.
//! A long-lived deployment will want compaction; that is a later ADR, and
//! leaving it out is recorded here rather than discovered later.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A Raft term. Monotonic; every message carries one.
pub type Term = u64;

/// 1-based index into the replicated log. Index 0 is the empty sentinel
/// "before the first entry", which is what makes the `prev_log_index` check
/// uniform for the first append (no special case).
pub type LogIndex = u64;

/// Operator-declared member id. ADR 0106 forbids open membership: a quorum that
/// admits a gossiped stranger is a quorum a Sybil can capture, so this is never
/// derived from peer discovery.
pub type MemberId = String;

/// One replicated command plus the term it was created in.
///
/// The term is what makes the log matching property work: two logs holding the
/// same (index, term) hold the same command and the same history before it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry<C> {
    pub term: Term,
    pub command: C,
}

/// What a node currently believes it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Follower,
    Candidate,
    Leader,
}

/// Consensus messages. In a deployment these ride the mesh as another
/// authenticated `MeshFrame` variant (ADR 0106: no HTTP, no second transport);
/// the transport is abstracted so the in-process harness can deliver them
/// directly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RaftMessage<C> {
    RequestVote {
        term: Term,
        candidate: MemberId,
        last_log_index: LogIndex,
        last_log_term: Term,
    },
    RequestVoteResp {
        term: Term,
        from: MemberId,
        granted: bool,
    },
    AppendEntries {
        term: Term,
        leader: MemberId,
        prev_log_index: LogIndex,
        prev_log_term: Term,
        entries: Vec<LogEntry<C>>,
        leader_commit: LogIndex,
    },
    AppendEntriesResp {
        term: Term,
        from: MemberId,
        /// Highest index this follower now has, so the leader can advance
        /// `match_index` without a second round trip.
        match_index: LogIndex,
        success: bool,
    },
}

/// A message the caller must deliver to `to`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outbound<C> {
    pub to: MemberId,
    pub message: RaftMessage<C>,
}

/// A single Raft participant.
///
/// `C` is the replicated command type — for ADR 0106 that is a scheduler fact
/// (see [`crate::raft::state`]), never PHI or gradients. The Raft log is
/// metadata only, which is what keeps the clinical hard-block trivially true
/// rather than a property someone has to re-audit.
#[derive(Debug)]
pub struct RaftNode<C> {
    id: MemberId,
    members: Vec<MemberId>,

    // --- persistent state (would be fsynced before responding, in a
    // deployment; the in-process harness keeps it in memory) ---
    current_term: Term,
    voted_for: Option<MemberId>,
    log: Vec<LogEntry<C>>,

    // --- volatile state ---
    role: Role,
    commit_index: LogIndex,
    leader_hint: Option<MemberId>,
    votes_granted: Vec<MemberId>,

    // --- leader-only volatile state ---
    next_index: BTreeMap<MemberId, LogIndex>,
    match_index: BTreeMap<MemberId, LogIndex>,

    // --- election timing, as a tick budget rather than a clock ---
    election_timeout: u64,
    ticks_since_heard: u64,
}

impl<C: Clone + PartialEq> RaftNode<C> {
    /// Create a follower in term 0 with an empty log.
    ///
    /// `members` MUST include `id` and MUST be identical on every node — it is
    /// the operator-declared set, and a node that disagrees about membership
    /// computes a different majority, which is how split-brain gets in.
    pub fn new(id: impl Into<MemberId>, members: Vec<MemberId>, election_timeout: u64) -> Self {
        let id = id.into();
        assert!(
            members.contains(&id),
            "a node must be a member of its own cluster"
        );
        assert!(
            election_timeout > 0,
            "a zero election timeout would make a node a candidate before it \
             could ever hear a leader"
        );
        Self {
            id,
            members,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            role: Role::Follower,
            commit_index: 0,
            leader_hint: None,
            votes_granted: Vec::new(),
            next_index: BTreeMap::new(),
            match_index: BTreeMap::new(),
            election_timeout,
            ticks_since_heard: 0,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn role(&self) -> Role {
        self.role
    }
    pub fn term(&self) -> Term {
        self.current_term
    }
    pub fn commit_index(&self) -> LogIndex {
        self.commit_index
    }
    pub fn is_leader(&self) -> bool {
        self.role == Role::Leader
    }
    pub fn leader_hint(&self) -> Option<&str> {
        self.leader_hint.as_deref()
    }
    pub fn last_log_index(&self) -> LogIndex {
        self.log.len() as LogIndex
    }

    /// Entries the state machine may now apply (1-based, inclusive range).
    pub fn committed_entries(&self) -> &[LogEntry<C>] {
        &self.log[..self.commit_index as usize]
    }

    fn last_log_term(&self) -> Term {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Strict majority of the DECLARED member set — not of the reachable nodes.
    /// Computing it over who happens to be reachable is exactly how a minority
    /// partition elects itself a leader.
    fn quorum(&self) -> usize {
        self.members.len() / 2 + 1
    }

    /// Advance one tick. A follower or candidate that has not heard from a
    /// leader within its timeout starts an election.
    pub fn tick(&mut self) -> Vec<Outbound<C>> {
        if self.role == Role::Leader {
            // A leader keeps its followers from timing out by heartbeating.
            return self.broadcast_append();
        }
        self.ticks_since_heard += 1;
        if self.ticks_since_heard >= self.election_timeout {
            return self.begin_election();
        }
        Vec::new()
    }

    fn begin_election(&mut self) -> Vec<Outbound<C>> {
        self.current_term += 1;
        self.role = Role::Candidate;
        self.voted_for = Some(self.id.clone());
        self.votes_granted = vec![self.id.clone()];
        self.leader_hint = None;
        self.ticks_since_heard = 0;

        // A single-member cluster is its own majority — ADR 0106 requires a
        // 1-member deployment to behave exactly as before, with no quorum wait.
        if self.votes_granted.len() >= self.quorum() {
            self.become_leader();
            return self.broadcast_append();
        }

        let (last_log_index, last_log_term) = (self.last_log_index(), self.last_log_term());
        self.peers()
            .map(|to| Outbound {
                to,
                message: RaftMessage::RequestVote {
                    term: self.current_term,
                    candidate: self.id.clone(),
                    last_log_index,
                    last_log_term,
                },
            })
            .collect()
    }

    fn peers(&self) -> impl Iterator<Item = MemberId> + '_ {
        self.members.iter().filter(|m| **m != self.id).cloned()
    }

    fn become_leader(&mut self) {
        self.role = Role::Leader;
        self.leader_hint = Some(self.id.clone());
        let next = self.last_log_index() + 1;
        self.next_index.clear();
        self.match_index.clear();
        for peer in self.peers().collect::<Vec<_>>() {
            self.next_index.insert(peer.clone(), next);
            self.match_index.insert(peer, 0);
        }
    }

    fn step_down(&mut self, term: Term) {
        self.current_term = term;
        self.role = Role::Follower;
        self.voted_for = None;
        self.votes_granted.clear();
        self.ticks_since_heard = 0;
    }

    /// Append a command as leader. Returns the index it landed at, or `None` if
    /// this node is not the leader.
    ///
    /// The entry is NOT committed yet. ADR 0106's commit-before-dispatch rule
    /// means the caller must wait for [`RaftNode::commit_index`] to reach this
    /// index before the side effect (sending the `Task` frame) is allowed to
    /// happen — see [`crate::raft::ha`].
    pub fn propose(&mut self, command: C) -> Option<LogIndex> {
        if self.role != Role::Leader {
            return None;
        }
        self.log.push(LogEntry {
            term: self.current_term,
            command,
        });
        let index = self.last_log_index();
        // A single-member cluster commits immediately: it is its own majority.
        self.advance_commit();
        Some(index)
    }

    fn broadcast_append(&mut self) -> Vec<Outbound<C>> {
        let peers: Vec<MemberId> = self.peers().collect();
        peers
            .into_iter()
            .map(|peer| {
                let next = *self.next_index.get(&peer).unwrap_or(&1);
                let prev_log_index = next.saturating_sub(1);
                let prev_log_term = if prev_log_index == 0 {
                    0
                } else {
                    self.log[(prev_log_index - 1) as usize].term
                };
                let entries = self.log[prev_log_index as usize..].to_vec();
                Outbound {
                    to: peer,
                    message: RaftMessage::AppendEntries {
                        term: self.current_term,
                        leader: self.id.clone(),
                        prev_log_index,
                        prev_log_term,
                        entries,
                        leader_commit: self.commit_index,
                    },
                }
            })
            .collect()
    }

    /// Handle one message; return anything that must be sent in reply.
    pub fn handle(&mut self, message: RaftMessage<C>) -> Vec<Outbound<C>> {
        // Any message from a higher term makes this node a follower of that
        // term before anything else is considered.
        let incoming_term = match &message {
            RaftMessage::RequestVote { term, .. }
            | RaftMessage::RequestVoteResp { term, .. }
            | RaftMessage::AppendEntries { term, .. }
            | RaftMessage::AppendEntriesResp { term, .. } => *term,
        };
        if incoming_term > self.current_term {
            self.step_down(incoming_term);
        }

        match message {
            RaftMessage::RequestVote {
                term,
                candidate,
                last_log_index,
                last_log_term,
            } => self.on_request_vote(term, candidate, last_log_index, last_log_term),
            RaftMessage::RequestVoteResp {
                term,
                from,
                granted,
            } => self.on_vote_response(term, from, granted),
            RaftMessage::AppendEntries {
                term,
                leader,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit,
            } => self.on_append_entries(
                term,
                leader,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit,
            ),
            RaftMessage::AppendEntriesResp {
                term,
                from,
                match_index,
                success,
            } => self.on_append_response(term, from, match_index, success),
        }
    }

    fn on_request_vote(
        &mut self,
        term: Term,
        candidate: MemberId,
        last_log_index: LogIndex,
        last_log_term: Term,
    ) -> Vec<Outbound<C>> {
        let mut granted = false;
        if term >= self.current_term {
            let free = match &self.voted_for {
                None => true,
                Some(already) => *already == candidate,
            };
            // Leader completeness: only a candidate whose log is at least as
            // up to date as ours may win, so a committed entry can never be
            // lost to a node that never had it. "Up to date" compares the last
            // TERM first, then length — a longer log from an older term loses.
            let up_to_date =
                (last_log_term, last_log_index) >= (self.last_log_term(), self.last_log_index());
            if free && up_to_date {
                granted = true;
                self.voted_for = Some(candidate.clone());
                self.ticks_since_heard = 0;
            }
        }
        vec![Outbound {
            to: candidate,
            message: RaftMessage::RequestVoteResp {
                term: self.current_term,
                from: self.id.clone(),
                granted,
            },
        }]
    }

    fn on_vote_response(&mut self, term: Term, from: MemberId, granted: bool) -> Vec<Outbound<C>> {
        // A vote for an older election says nothing about this one.
        if self.role != Role::Candidate || term != self.current_term || !granted {
            return Vec::new();
        }
        if !self.votes_granted.contains(&from) {
            self.votes_granted.push(from);
        }
        if self.votes_granted.len() >= self.quorum() {
            self.become_leader();
            return self.broadcast_append();
        }
        Vec::new()
    }

    fn on_append_entries(
        &mut self,
        term: Term,
        leader: MemberId,
        prev_log_index: LogIndex,
        prev_log_term: Term,
        entries: Vec<LogEntry<C>>,
        leader_commit: LogIndex,
    ) -> Vec<Outbound<C>> {
        if term < self.current_term {
            // A stale leader. Reply with our term so it steps down.
            return vec![Outbound {
                to: leader,
                message: RaftMessage::AppendEntriesResp {
                    term: self.current_term,
                    from: self.id.clone(),
                    match_index: 0,
                    success: false,
                },
            }];
        }
        // Same term, live leader: accept it and stop counting toward an election.
        self.role = Role::Follower;
        self.leader_hint = Some(leader.clone());
        self.ticks_since_heard = 0;

        // Log matching: refuse unless we hold the leader's previous entry.
        let consistent = prev_log_index == 0
            || (prev_log_index <= self.last_log_index()
                && self.log[(prev_log_index - 1) as usize].term == prev_log_term);
        if !consistent {
            return vec![Outbound {
                to: leader,
                message: RaftMessage::AppendEntriesResp {
                    term: self.current_term,
                    from: self.id.clone(),
                    match_index: 0,
                    success: false,
                },
            }];
        }

        // Truncate only on a real conflict. Blindly truncating on every append
        // would discard entries an in-flight duplicate has already delivered.
        for (offset, entry) in entries.iter().enumerate() {
            let index = prev_log_index + offset as LogIndex + 1;
            let slot = (index - 1) as usize;
            if slot < self.log.len() {
                if self.log[slot].term != entry.term {
                    self.log.truncate(slot);
                    self.log.push(entry.clone());
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        if leader_commit > self.commit_index {
            self.commit_index = leader_commit.min(self.last_log_index());
        }

        vec![Outbound {
            to: leader,
            message: RaftMessage::AppendEntriesResp {
                term: self.current_term,
                from: self.id.clone(),
                match_index: self.last_log_index(),
                success: true,
            },
        }]
    }

    fn on_append_response(
        &mut self,
        term: Term,
        from: MemberId,
        match_index: LogIndex,
        success: bool,
    ) -> Vec<Outbound<C>> {
        if self.role != Role::Leader || term != self.current_term {
            return Vec::new();
        }
        if success {
            self.match_index.insert(from.clone(), match_index);
            self.next_index.insert(from, match_index + 1);
            self.advance_commit();
        } else {
            // Walk back one and retry; the next heartbeat carries the shorter
            // prefix. Linear backoff is fine at this scale and is far easier to
            // convince yourself is correct than the optimised variants.
            let next = self.next_index.entry(from).or_insert(1);
            *next = (*next).saturating_sub(1).max(1);
        }
        Vec::new()
    }

    /// Advance `commit_index` to the highest index replicated on a majority —
    /// but only for an entry from the CURRENT term.
    ///
    /// That restriction is the subtle one (Raft §5.4.2). Counting replicas of a
    /// previous-term entry can commit an entry that a later leader then
    /// overwrites, which loses a committed result — precisely the failure ADR
    /// 0106 exists to prevent. A current-term entry committing carries the
    /// earlier ones with it.
    fn advance_commit(&mut self) {
        let last = self.last_log_index();
        let mut candidate = self.commit_index;
        for index in (self.commit_index + 1)..=last {
            if self.log[(index - 1) as usize].term != self.current_term {
                continue;
            }
            let replicas = 1 + self.match_index.values().filter(|m| **m >= index).count();
            if replicas >= self.quorum() {
                candidate = index;
            }
        }
        self.commit_index = candidate;
    }
}
