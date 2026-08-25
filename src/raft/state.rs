// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The replicated state machine: the scheduler's durable facts (ADR 0106).
//!
//! What goes in the Raft log is exactly what a new leader needs to avoid
//! re-doing work: per-stage status, the stage→peer assignment, the
//! content-addressed result key, and the cloud-queue job handle (ADR 0082).
//!
//! **Metadata only, and that is load-bearing.** No artifact bytes, no
//! gradients, no `Restricted` payload ever enters an entry — the log holds
//! stage names, peer ids, and hashes. That is what keeps ADR 0106's claim on
//! the clinical hard-block trivially true instead of a property someone has to
//! re-derive every time the log grows a field. A future field carrying anything
//! derived from sample data breaks it, which is why [`SchedulerCommand`] is a
//! closed enum rather than a bag of bytes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Where a stage is in its lifecycle, as the quorum understands it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageStatus {
    /// Known to the plan, not yet assigned.
    Queued,
    /// A dispatch decision is COMMITTED. The task frame may or may not have
    /// reached the peer — that ambiguity is the entire point: committing first
    /// means a failover can never forget that it might have.
    Dispatched,
    /// The assigned peer reported it started.
    Running,
    /// Finished, with a content-addressed result.
    Done,
    /// Finished unsuccessfully.
    Failed,
}

/// One replicated scheduler fact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulerCommand {
    /// A stage entered the plan.
    Enqueue { stage: String },
    /// A dispatch decision. Committed BEFORE the task frame goes out.
    ///
    /// `expected` is the analytically-known output identity when the stage can
    /// supply one. It is what lets a new leader ADOPT a result it did not
    /// dispatch: the key is content-addressed, so a result matching it is the
    /// result, whoever produced it.
    Dispatch {
        stage: String,
        peer: String,
        expected: Option<String>,
    },
    /// The assigned peer acknowledged it is running.
    Running { stage: String, peer: String },
    /// A completed stage and the content hash of its output.
    Complete { stage: String, content_hash: String },
    /// A failed stage.
    Fail { stage: String, reason: String },
    /// A cloud-queue job handle (ADR 0082), replicated so a failover does not
    /// submit a second job for work already billed.
    CloudJob { stage: String, job_id: String },
    /// A committed dispatch that reconciliation PROVED never took effect: no
    /// live peer claims the stage and no result exists.
    ///
    /// This exists because the guard against double dispatch would otherwise
    /// also block the one re-dispatch that is correct. An orphan is still
    /// committed as `Dispatched`, so without an explicit void the new leader
    /// sees "already dispatched" and the stage strands forever — which is a
    /// worse failure than the one the guard prevents, and a quieter one.
    ///
    /// Voiding is itself replicated: a leader that voided a dispatch and then
    /// crashed must not leave the next leader believing the original is live.
    VoidDispatch { stage: String, reason: String },
}

/// One stage's replicated record.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StageRecord {
    pub status: Option<StageStatus>,
    pub peer: Option<String>,
    pub expected: Option<String>,
    pub content_hash: Option<String>,
    pub cloud_job: Option<String>,
    /// How many dispatch decisions were COMMITTED for this stage, ever.
    pub dispatch_count: u32,
    /// How many of those were later PROVEN never to have taken effect.
    ///
    /// Kept as a separate number rather than decrementing `dispatch_count`, so
    /// both facts stay auditable: a stage re-dispatched once after a proven
    /// orphan reads 2/1, not 1/0. Silently decrementing would make a real
    /// double-dispatch indistinguishable from a legitimate recovery.
    pub voided_dispatches: u32,
}

impl StageRecord {
    /// Dispatches that could actually have caused execution.
    ///
    /// This is what ADR 0106's gate asserts on. "At most once" means at most
    /// one LIVE dispatch — re-dispatching a stage whose previous assignment was
    /// proven void is the recovery the ADR asks for, not a violation of it.
    pub fn effective_dispatches(&self) -> u32 {
        self.dispatch_count.saturating_sub(self.voided_dispatches)
    }
}

/// The deterministic state machine every member applies the log into.
#[derive(Clone, Debug, Default)]
pub struct SchedulerState {
    stages: BTreeMap<String, StageRecord>,
    applied: u64,
}

impl SchedulerState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Entries applied so far. Applying is idempotent by index, so a caller may
    /// hand the whole committed prefix every time.
    pub fn applied(&self) -> u64 {
        self.applied
    }

    pub fn get(&self, stage: &str) -> Option<&StageRecord> {
        self.stages.get(stage)
    }

    pub fn stages(&self) -> impl Iterator<Item = (&String, &StageRecord)> {
        self.stages.iter()
    }

    /// Apply the committed prefix, skipping what is already applied.
    ///
    /// Takes the whole prefix rather than a delta because that is what a new
    /// leader has after an election: a log, not a diff. Skipping by count keeps
    /// it O(new) and makes double-application impossible.
    pub fn apply_prefix(&mut self, committed: &[super::consensus::LogEntry<SchedulerCommand>]) {
        for entry in committed.iter().skip(self.applied as usize) {
            self.apply(&entry.command);
            self.applied += 1;
        }
    }

    fn apply(&mut self, command: &SchedulerCommand) {
        match command {
            SchedulerCommand::Enqueue { stage } => {
                let record = self.stages.entry(stage.clone()).or_default();
                if record.status.is_none() {
                    record.status = Some(StageStatus::Queued);
                }
            }
            SchedulerCommand::Dispatch {
                stage,
                peer,
                expected,
            } => {
                let record = self.stages.entry(stage.clone()).or_default();
                record.status = Some(StageStatus::Dispatched);
                record.peer = Some(peer.clone());
                if expected.is_some() {
                    record.expected = expected.clone();
                }
                record.dispatch_count += 1;
            }
            SchedulerCommand::Running { stage, peer } => {
                let record = self.stages.entry(stage.clone()).or_default();
                // A late `Running` must not resurrect a finished stage: the
                // result is the terminal fact, and message order across a
                // failover is not guaranteed.
                if !matches!(record.status, Some(StageStatus::Done | StageStatus::Failed)) {
                    record.status = Some(StageStatus::Running);
                    record.peer = Some(peer.clone());
                }
            }
            SchedulerCommand::Complete {
                stage,
                content_hash,
            } => {
                let record = self.stages.entry(stage.clone()).or_default();
                record.status = Some(StageStatus::Done);
                record.content_hash = Some(content_hash.clone());
            }
            SchedulerCommand::Fail { stage, .. } => {
                let record = self.stages.entry(stage.clone()).or_default();
                if record.status != Some(StageStatus::Done) {
                    record.status = Some(StageStatus::Failed);
                }
            }
            SchedulerCommand::CloudJob { stage, job_id } => {
                let record = self.stages.entry(stage.clone()).or_default();
                record.cloud_job = Some(job_id.clone());
            }
            SchedulerCommand::VoidDispatch { stage, .. } => {
                let record = self.stages.entry(stage.clone()).or_default();
                // Never void a stage that finished. A result is terminal, and a
                // late void arriving after a completion would resurrect work
                // that is already done and paid for.
                if record.status != Some(StageStatus::Done) {
                    record.status = Some(StageStatus::Queued);
                    record.peer = None;
                    record.voided_dispatches += 1;
                }
            }
        }
    }

    /// Stages a new leader must decide about: committed as dispatched or
    /// running, with no committed result.
    ///
    /// Deliberately NOT "stages to re-dispatch" — that is the decision
    /// [`super::ha`] makes after asking the live peers, and naming it that here
    /// would invite a caller to skip the asking.
    pub fn unresolved(&self) -> Vec<&String> {
        self.stages
            .iter()
            .filter(|(_, r)| {
                matches!(
                    r.status,
                    Some(StageStatus::Dispatched) | Some(StageStatus::Running)
                )
            })
            .map(|(name, _)| name)
            .collect()
    }

    /// Stages with a committed result — a new leader adopts these untouched.
    pub fn completed(&self) -> Vec<&String> {
        self.stages
            .iter()
            .filter(|(_, r)| r.status == Some(StageStatus::Done))
            .map(|(name, _)| name)
            .collect()
    }

    /// The highest number of simultaneously-effective dispatches over all
    /// stages. ADR 0106's gate requires this to be 1 across a run that survives
    /// a leader kill.
    pub fn max_effective_dispatches(&self) -> u32 {
        self.stages
            .values()
            .map(|r| r.effective_dispatches())
            .max()
            .unwrap_or(0)
    }

    /// Total committed dispatch decisions, including ones later voided. Useful
    /// for telling "recovered once" apart from "never needed recovery".
    pub fn total_dispatch_decisions(&self) -> u32 {
        self.stages.values().map(|r| r.dispatch_count).sum()
    }
}
