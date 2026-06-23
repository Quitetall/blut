// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Peer registry — persistent store of known peers.
//!
//! The coordinator maintains a registry of peers that have connected.
//! Each peer's trust level, capabilities, and reputation are tracked.
//! The registry persists to JSON so it survives coordinator restarts.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::TrainError;
use crate::p2p::peer::{PeerId, PeerInfo};
use crate::p2p::trust::TrustLevel;

/// Persistent peer registry. Maps PeerId → PeerInfo.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerRegistry {
    peers: HashMap<PeerId, PeerInfo>,
    #[serde(skip)]
    path: PathBuf,
}

impl PeerRegistry {
    /// Load the registry from a JSON file, or create an empty one if the
    /// file doesn't exist.
    pub fn load(path: &Path) -> Result<Self, TrainError> {
        if path.exists() {
            let data = std::fs::read_to_string(path).map_err(|e| TrainError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            let mut reg: Self =
                serde_json::from_str(&data).map_err(|e| TrainError::other(format!(
                    "corrupt peer registry {}: {e}",
                    path.display()
                )))?;
            reg.path = path.to_path_buf();
            Ok(reg)
        } else {
            Ok(Self {
                peers: HashMap::new(),
                path: path.to_path_buf(),
            })
        }
    }

    /// Save the registry to disk (atomic write via fsync + rename).
    pub fn save(&self) -> Result<(), TrainError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| TrainError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TrainError::other(format!("serialize registry: {e}")))?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, &json).map_err(|e| TrainError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        // fsync before rename to prevent data loss on crash (ext4 data=writeback,
        // XFS can reorder writes vs rename without fsync).
        let f = std::fs::File::open(&tmp).map_err(|e| TrainError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.sync_all().map_err(|e| TrainError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp, &self.path).map_err(|e| TrainError::Io {
            path: self.path.clone(),
            source: e,
        })
    }

    /// Add or update a peer. If the peer already exists, updates its
    /// capabilities and last_seen but preserves trust and reputation.
    pub fn upsert(&mut self, info: PeerInfo) {
        if let Some(existing) = self.peers.get_mut(&info.id) {
            existing.capabilities = info.capabilities;
            existing.last_seen = chrono::Utc::now();
        } else {
            self.peers.insert(info.id.clone(), info);
        }
    }

    /// Add a new peer. Fails if the peer already exists (use `upsert` to
    /// update).
    pub fn add(&mut self, info: PeerInfo) -> Result<(), TrainError> {
        if self.peers.contains_key(&info.id) {
            return Err(TrainError::other(format!(
                "peer {} already registered",
                info.id
            )));
        }
        self.peers.insert(info.id.clone(), info);
        Ok(())
    }

    /// Remove a peer by ID.
    pub fn remove(&mut self, id: &PeerId) -> bool {
        self.peers.remove(id).is_some()
    }

    /// Get a peer by ID.
    pub fn get(&self, id: &PeerId) -> Option<&PeerInfo> {
        self.peers.get(id)
    }

    /// Get a mutable reference to a peer by ID.
    pub fn get_mut(&mut self, id: &PeerId) -> Option<&mut PeerInfo> {
        self.peers.get_mut(id)
    }

    /// List all peers.
    pub fn list(&self) -> Vec<&PeerInfo> {
        self.peers.values().collect()
    }

    /// List peers with at least the given trust level.
    pub fn by_trust(&self, min_trust: TrustLevel) -> Vec<&PeerInfo> {
        self.peers
            .values()
            .filter(|p| p.trust.level() >= min_trust.level())
            .collect()
    }

    /// List peers that can handle the given resource requirements.
    pub fn by_capability(
        &self,
        cpu_cores: u32,
        memory_gib: u32,
        gpu: bool,
        gpu_vram_gib: Option<u32>,
    ) -> Vec<&PeerInfo> {
        self.peers
            .values()
            .filter(|p| {
                p.capabilities.cpu_cores >= cpu_cores
                    && p.capabilities.memory_gib >= memory_gib
                    && (!gpu || p.capabilities.gpu_model.is_some())
                    && gpu_vram_gib
                        .map(|v| p.capabilities.gpu_vram_gib.unwrap_or(0) >= v)
                        .unwrap_or(true)
            })
            .collect()
    }

    /// Update a peer's reputation after a task outcome.
    pub fn update_reputation(&mut self, id: &PeerId, success: bool) {
        if let Some(peer) = self.peers.get_mut(id) {
            peer.record_outcome(success);
        }
    }

    /// Set a peer's trust level.
    pub fn set_trust(&mut self, id: &PeerId, trust: TrustLevel) -> bool {
        if let Some(peer) = self.peers.get_mut(id) {
            peer.trust = trust;
            true
        } else {
            false
        }
    }

    /// Number of registered peers.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::peer::PeerCapabilities;

    fn make_peer(trust: TrustLevel) -> PeerInfo {
        let kp = KeyPair::generate();
        PeerInfo::new(kp.verifying, kp.x25519_public, trust, PeerCapabilities::default())
    }

    #[test]
    fn add_and_get() {
        let mut reg = PeerRegistry {
            peers: HashMap::new(),
            path: PathBuf::new(),
        };
        let peer = make_peer(TrustLevel::Anonymous);
        let id = peer.id.clone();
        reg.add(peer).unwrap();
        assert!(reg.get(&id).is_some());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn add_duplicate_fails() {
        let mut reg = PeerRegistry {
            peers: HashMap::new(),
            path: PathBuf::new(),
        };
        let peer = make_peer(TrustLevel::Anonymous);
        reg.add(peer.clone()).unwrap();
        assert!(reg.add(peer).is_err());
    }

    #[test]
    fn upsert_updates_existing() {
        let mut reg = PeerRegistry {
            peers: HashMap::new(),
            path: PathBuf::new(),
        };
        let mut peer = make_peer(TrustLevel::Anonymous);
        let id = peer.id.clone();
        reg.add(peer.clone()).unwrap();

        // Update capabilities.
        peer.capabilities.cpu_cores = 32;
        reg.upsert(peer);
        assert_eq!(reg.get(&id).unwrap().capabilities.cpu_cores, 32);
        // Trust preserved.
        assert_eq!(reg.get(&id).unwrap().trust, TrustLevel::Anonymous);
    }

    #[test]
    fn remove_peer() {
        let mut reg = PeerRegistry {
            peers: HashMap::new(),
            path: PathBuf::new(),
        };
        let peer = make_peer(TrustLevel::Registered);
        let id = peer.id.clone();
        reg.add(peer).unwrap();
        assert!(reg.remove(&id));
        assert!(reg.get(&id).is_none());
        assert!(!reg.remove(&id)); // already removed
    }

    #[test]
    fn by_trust_filter() {
        let mut reg = PeerRegistry {
            peers: HashMap::new(),
            path: PathBuf::new(),
        };
        reg.add(make_peer(TrustLevel::Anonymous)).unwrap();
        reg.add(make_peer(TrustLevel::Registered)).unwrap();
        reg.add(make_peer(TrustLevel::Trusted)).unwrap();

        assert_eq!(reg.by_trust(TrustLevel::Anonymous).len(), 3);
        assert_eq!(reg.by_trust(TrustLevel::Registered).len(), 2);
        assert_eq!(reg.by_trust(TrustLevel::Trusted).len(), 1);
    }

    #[test]
    fn by_capability_filter() {
        let mut reg = PeerRegistry {
            peers: HashMap::new(),
            path: PathBuf::new(),
        };
        let mut peer = make_peer(TrustLevel::Registered);
        peer.capabilities.cpu_cores = 16;
        peer.capabilities.memory_gib = 64;
        peer.capabilities.gpu_model = Some("RTX 4090".into());
        peer.capabilities.gpu_vram_gib = Some(24);
        reg.add(peer).unwrap();
        reg.add(make_peer(TrustLevel::Anonymous)).unwrap(); // default: 1 core, 4 GiB

        // Need 8 cores, 32 GiB, GPU with 16 GiB VRAM.
        let capable = reg.by_capability(8, 32, true, Some(16));
        assert_eq!(capable.len(), 1);

        // Need 1 core, 1 GiB, no GPU.
        let any = reg.by_capability(1, 1, false, None);
        assert_eq!(any.len(), 2);
    }

    #[test]
    fn set_trust() {
        let mut reg = PeerRegistry {
            peers: HashMap::new(),
            path: PathBuf::new(),
        };
        let peer = make_peer(TrustLevel::Anonymous);
        let id = peer.id.clone();
        reg.add(peer).unwrap();

        assert!(reg.set_trust(&id, TrustLevel::Trusted));
        assert_eq!(reg.get(&id).unwrap().trust, TrustLevel::Trusted);
    }

    #[test]
    fn persist_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("peers.json");

        // Write.
        let mut reg = PeerRegistry::load(&path).unwrap();
        reg.add(make_peer(TrustLevel::Trusted)).unwrap();
        reg.add(make_peer(TrustLevel::Anonymous)).unwrap();
        reg.save().unwrap();

        // Read.
        let reg2 = PeerRegistry::load(&path).unwrap();
        assert_eq!(reg2.len(), 2);
    }

    #[test]
    fn reputation_update_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("peers.json");

        let mut reg = PeerRegistry::load(&path).unwrap();
        let peer = make_peer(TrustLevel::Registered);
        let id = peer.id.clone();
        reg.add(peer).unwrap();

        reg.update_reputation(&id, true);
        reg.update_reputation(&id, true);
        reg.update_reputation(&id, false);
        reg.save().unwrap();

        let reg2 = PeerRegistry::load(&path).unwrap();
        let p = reg2.get(&id).unwrap();
        assert_eq!(p.tasks_completed, 2);
        assert_eq!(p.tasks_failed, 1);
        assert!((p.success_rate() - 2.0 / 3.0).abs() < 1e-10);
    }
}
