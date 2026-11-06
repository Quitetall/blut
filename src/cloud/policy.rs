// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The dispatch policy of `--launcher cloud`.
//!
//! Asking for the cloud launcher is asking for every stage to run on a worker,
//! so every stage is dispatchable. Custody is decided by the run's tenant: a
//! restricted tenant's run is refused before anything is submitted, and the
//! submitter and every worker refuse a restricted tenant or `Restricted` data
//! again (`custody_allows_off_box`). Every other stage is `Internal`, which a
//! `Registered` cloud worker may run under the default [`DispatchMatrix`].
//! This is the classification the executor already gives a local stage of a
//! non-restricted tenant.

use crate::framework::artifact::ArtifactContentId;
use crate::p2p::dispatch::{DefaultDispatchPolicy, DispatchPolicy, DispatchVerdict};
use crate::p2p::peer::{PeerId, PeerInfo};
use crate::p2p::task::{ResourceRequest, TaskResult};
use crate::p2p::trust::{DataClass, DispatchMatrix};

/// See the [module docs](self).
pub struct CloudDispatchPolicy {
    peers: DefaultDispatchPolicy,
}

impl CloudDispatchPolicy {
    pub fn new(matrix: DispatchMatrix) -> Self {
        Self {
            peers: DefaultDispatchPolicy::new(matrix),
        }
    }
}

impl Default for CloudDispatchPolicy {
    fn default() -> Self {
        Self::new(DispatchMatrix::default())
    }
}

impl DispatchPolicy for CloudDispatchPolicy {
    fn is_dispatchable(&self, _stage_name: &str) -> bool {
        true
    }

    fn classify_stage(&self, _stage_name: &str, _args: &serde_json::Value) -> DataClass {
        DataClass::Internal
    }

    fn select_peer(
        &self,
        stage_name: &str,
        resources: &ResourceRequest,
        data_class: DataClass,
        peers: &[PeerInfo],
    ) -> Option<PeerId> {
        self.peers
            .select_peer(stage_name, resources, data_class, peers)
    }

    fn verify_result(
        &self,
        result: &TaskResult,
        expected: Option<ArtifactContentId>,
        peer_pubkey: &ed25519_dalek::VerifyingKey,
    ) -> DispatchVerdict {
        self.peers.verify_result(result, expected, peer_pubkey)
    }
}
