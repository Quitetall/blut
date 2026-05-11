//! Resources a stage holds while running.
//!
//! Each `Stage` declares a `&'static [Resource]` constant. The
//! executor (lands commit 6) keeps one semaphore per `Resource` and
//! makes a stage acquire all its declared resources before
//! `Stage::run` is awaited. This is intra-binary scheduling — the
//! cross-process `lamu_core::scheduler_lock` is a separate concern
//! that `Resource::Gpu` stages additionally acquire.
//!
//! Why an enum and not a string: enums make wrong values a compile
//! error. New resource kinds should be deliberate. If a stage
//! genuinely needs something not on this list, add a variant; the
//! executor's semaphore registry is keyed on the enum so adding a
//! variant in one place adds the semaphore everywhere.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    /// Stage holds the GPU. Acquires `lamu_core::scheduler_lock`
    /// (cross-process arbitration) AND the executor's `Gpu`
    /// semaphore (intra-process serialization). On a single-card
    /// box the semaphore permit count is 1; multi-GPU is a future
    /// expansion.
    Gpu,
    /// CPU-bound. Cap is the executor's CPU concurrency setting,
    /// default `num_cpus`.
    Cpu,
    /// Network-heavy: HuggingFace downloads, judge-model API calls.
    /// Cap defaults to a small number (4) so a runaway stage can't
    /// saturate the link.
    Network,
    /// Disk-bound (large reads/writes). Cap defaults to 2; the
    /// usual bottleneck is the SSD's queue depth, not CPU.
    Disk,
}

impl Resource {
    /// Compact tag for logging / status events. Stable — used in
    /// status.jsonl entries, so changing this is a wire-format
    /// break.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
            Self::Network => "network",
            Self::Disk => "disk",
        }
    }

    /// All variants. Drives the executor's semaphore registry —
    /// adding a new variant automatically adds a semaphore.
    pub const fn all() -> &'static [Resource] {
        &[Self::Gpu, Self::Cpu, Self::Network, Self::Disk]
    }
}

impl std::fmt::Display for Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_present() {
        assert_eq!(Resource::all().len(), 4);
        for r in [Resource::Gpu, Resource::Cpu, Resource::Network, Resource::Disk] {
            assert!(Resource::all().contains(&r));
        }
    }

    #[test]
    fn tags_are_stable_lowercase_snake() {
        assert_eq!(Resource::Gpu.tag(), "gpu");
        assert_eq!(Resource::Cpu.tag(), "cpu");
        assert_eq!(Resource::Network.tag(), "network");
        assert_eq!(Resource::Disk.tag(), "disk");
    }

    #[test]
    fn serde_round_trips_via_snake_case() {
        for r in [Resource::Gpu, Resource::Cpu, Resource::Network, Resource::Disk] {
            let s = serde_json::to_string(&r).unwrap();
            assert!(s.starts_with('"'));
            assert!(s.contains(r.tag()), "serde tag for {r:?}: {s}");
            let back: Resource = serde_json::from_str(&s).unwrap();
            assert_eq!(back, r);
        }
    }

    #[test]
    fn display_matches_tag() {
        assert_eq!(format!("{}", Resource::Gpu), "gpu");
    }
}
