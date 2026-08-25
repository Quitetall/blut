// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Highly-available scheduler state via embedded Raft (ADR 0106).
//!
//! ADR 0079 made "scheduler" a capability any node may enable, but a given DAG
//! is driven by one node at a time. Lose that node mid-run and the in-flight
//! dispatch table goes with it: surviving workers keep computing while nobody
//! collects their results, and a restart re-dispatches stages that are already
//! running — double-billing the cloud queue (ADR 0082) and breaking the
//! exactly-once expectation content-addressing is supposed to give.
//!
//! This module replicates the scheduler's durable facts across an
//! operator-declared quorum so a failover can reason instead of guess.
//!
//! - [`consensus`] — the Raft core. Deterministic and tick-driven: no timers,
//!   no threads, no clock reads, because a consensus gate that reproduces a bug
//!   one run in twenty is not a gate.
//! - [`state`] — the replicated state machine. Metadata ONLY (stage status,
//!   peer ids, content hashes, cloud job handles); no artifact bytes and no
//!   gradients, which is what keeps the clinical hard-block true by
//!   construction rather than by audit.
//! - [`ha`] — commit-before-dispatch, reconcile-not-re-dispatch, and
//!   fail-closed-on-quorum-loss.
//!
//! **Transport.** ADR 0106 specifies Raft riding the existing mesh as another
//! authenticated frame — no HTTP, no second service, per the ADR 0034 charter.
//! The core here is transport-agnostic: it consumes and returns messages and
//! never performs I/O, so the mesh carries [`consensus::RaftMessage`] as a
//! `MeshFrame` payload and the in-process harness delivers it directly. That
//! split is what lets ADR 0106's software gate run in CI with no network at
//! all, which is the half of its acceptance gate that is not hardware-gated.
//!
//! **What is NOT here, stated so it is not mistaken for done:** no snapshotting
//! or log compaction (a long-lived cluster will need both), no joint-consensus
//! membership change (ADR 0106 declares a fixed operator-supplied member set,
//! so the subtlest part of Raft is out of scope by decision), and no mesh
//! wiring yet — the frame variant and the hardware chaos test
//! (`tools/scripts/ha_chaos.sh`, ≥3 real nodes) remain the hardware-gated half.

pub mod consensus;
pub mod ha;
pub mod state;

pub use consensus::{LogEntry, MemberId, Outbound, RaftMessage, RaftNode, Role, Term};
pub use ha::{DispatchDecision, HaScheduler, PeerReport, PeerStageState, ReconcileAction};
pub use state::{SchedulerCommand, SchedulerState, StageRecord, StageStatus};

/// An in-process cluster for tests and for the ADR 0106 software gate.
///
/// Lives beside the implementation rather than in the test file because the
/// gate is not the only consumer — anything reasoning about failover wants a
/// deterministic cluster it can single-step, and a harness that only exists
/// inside one `#[test]` gets reinvented slightly differently by the next one.
pub mod harness {
    use super::*;
    use std::collections::VecDeque;

    /// A deterministic multi-node cluster with an explicit message queue.
    ///
    /// Delivery is FIFO and single-stepped, so a test names the exact
    /// interleaving it is asserting about instead of hoping a scheduler
    /// produces it.
    pub struct Cluster {
        pub nodes: Vec<HaScheduler>,
        queue: VecDeque<(String, Outbound<SchedulerCommand>)>,
        /// Members that have been killed. Their messages are dropped, which is
        /// what "the leader crashed" means to everyone else.
        down: Vec<String>,
    }

    impl Cluster {
        pub fn new(ids: &[&str], election_timeout: u64) -> Self {
            let members: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
            // Distinct timeouts, ascending, so elections are decided rather than
            // split — the deterministic stand-in for Raft's randomised timers.
            let nodes = ids
                .iter()
                .enumerate()
                .map(|(i, id)| HaScheduler::new(*id, members.clone(), election_timeout + i as u64))
                .collect();
            Self {
                nodes,
                queue: VecDeque::new(),
                down: Vec::new(),
            }
        }

        pub fn node(&mut self, id: &str) -> &mut HaScheduler {
            self.nodes
                .iter_mut()
                .find(|n| n.id() == id)
                .unwrap_or_else(|| panic!("no such member '{id}'"))
        }

        pub fn get(&self, id: &str) -> &HaScheduler {
            self.nodes
                .iter()
                .find(|n| n.id() == id)
                .unwrap_or_else(|| panic!("no such member '{id}'"))
        }

        pub fn is_down(&self, id: &str) -> bool {
            self.down.iter().any(|d| d == id)
        }

        /// Kill a node: it stops ticking, stops receiving, and its queued
        /// messages are dropped.
        pub fn kill(&mut self, id: &str) {
            self.down.push(id.to_string());
            self.queue.retain(|(from, out)| from != id && out.to != id);
            let live: Vec<String> = self
                .nodes
                .iter()
                .map(|n| n.id().to_string())
                .filter(|n| !self.down.contains(n))
                .collect();
            for node in self.nodes.iter_mut() {
                node.set_reachable(live.clone());
            }
        }

        pub fn enqueue_out(&mut self, from: &str, out: Vec<Outbound<SchedulerCommand>>) {
            for o in out {
                if self.is_down(&o.to) || self.is_down(from) {
                    continue;
                }
                self.queue.push_back((from.to_string(), o));
            }
        }

        /// Tick every live node once.
        pub fn tick_all(&mut self) {
            let ids: Vec<String> = self
                .nodes
                .iter()
                .map(|n| n.id().to_string())
                .filter(|id| !self.down.contains(id))
                .collect();
            for id in ids {
                let out = self.node(&id).tick();
                self.enqueue_out(&id, out);
            }
        }

        /// Deliver every queued message, following cascades, until quiescent.
        ///
        /// Bounded: a consensus bug that ping-pongs forever should fail the test
        /// loudly rather than hang CI until it is killed with no diagnosis.
        pub fn deliver_all(&mut self) {
            let mut budget = 10_000;
            while let Some((_, out)) = self.queue.pop_front() {
                budget -= 1;
                assert!(
                    budget > 0,
                    "message delivery did not converge — likely a consensus loop"
                );
                if self.is_down(&out.to) {
                    continue;
                }
                let to = out.to.clone();
                let replies = self.node(&to).handle(out.message);
                self.enqueue_out(&to, replies);
            }
        }

        /// Tick and deliver until a leader exists, or panic with the state.
        pub fn settle(&mut self, rounds: usize) {
            for _ in 0..rounds {
                self.tick_all();
                self.deliver_all();
                if self.leader().is_some() {
                    return;
                }
            }
            panic!(
                "no leader after {rounds} rounds: {:?}",
                self.nodes
                    .iter()
                    .map(|n| (n.id().to_string(), n.role(), n.term()))
                    .collect::<Vec<_>>()
            );
        }

        /// Run rounds without requiring a leader (used while partitioned).
        pub fn run(&mut self, rounds: usize) {
            for _ in 0..rounds {
                self.tick_all();
                self.deliver_all();
            }
        }

        pub fn leader(&self) -> Option<&HaScheduler> {
            self.nodes
                .iter()
                .find(|n| n.is_leader() && !self.down.iter().any(|d| d == n.id()))
        }

        pub fn leader_id(&self) -> Option<String> {
            self.leader().map(|n| n.id().to_string())
        }

        /// The highest number of simultaneously-EFFECTIVE dispatches any live
        /// node believes in. ADR 0106's core assertion reads this.
        pub fn max_effective_dispatches(&self) -> u32 {
            self.nodes
                .iter()
                .filter(|n| !self.down.iter().any(|d| d == n.id()))
                .map(|n| n.state().max_effective_dispatches())
                .max()
                .unwrap_or(0)
        }
    }
}
