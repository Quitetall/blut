//! Compute-cost accounting (ADR 0067 · T3.1e).
//!
//! `cost = f(ResourceRequest, wall_time_ms)`. v1 records cost in abstract **compute
//! units** (a linear model over CPU cores / GPU / memory × seconds), not currency —
//! real per-provider $ pricing is T3.2. The point of v1 is the ledger PLUMBING: every
//! completed job's cost is attributable, so the "billed on compute" property is real
//! end-to-end. The worker emits one entry per job (it holds both the resources and
//! the compute-only `wall_time_ms`).

use parking_lot::Mutex;

use crate::p2p::task::ResourceRequest;

/// Linear cost weights (units per resource-second). GPU dominates by default,
/// reflecting that GPU-seconds are the scarce resource.
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    pub cpu_core_unit: f64,
    pub gpu_unit: f64,
    pub mem_gib_unit: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            cpu_core_unit: 1.0,
            gpu_unit: 100.0,
            mem_gib_unit: 0.5,
        }
    }
}

impl CostModel {
    /// Compute units for `resources` held for `wall_time_ms`. v1 ignores
    /// `gpu_vram_gib` (a 16 GB and an 80 GB GPU cost the same) — VRAM-tiered and
    /// per-provider $ pricing are T3.2.
    pub fn estimate(&self, resources: &ResourceRequest, wall_time_ms: u64) -> f64 {
        let secs = wall_time_ms as f64 / 1000.0;
        let per_sec = self.cpu_core_unit * resources.cpu_cores as f64
            + if resources.gpu { self.gpu_unit } else { 0.0 }
            + self.mem_gib_unit * resources.memory_gib as f64;
        secs * per_sec
    }
}

/// One ledger row: which job cost how much.
#[derive(Clone, Debug)]
pub struct CostEntry {
    pub job_id: String,
    pub units: f64,
}

/// An append-only, in-process cost ledger. A real deployment swaps in a durable
/// sink (DB / metrics); the recording point (the worker, per completed job) is the
/// same.
#[derive(Default)]
pub struct CostLedger {
    model: CostModel,
    entries: Mutex<Vec<CostEntry>>,
}

impl CostLedger {
    pub fn new(model: CostModel) -> Self {
        Self {
            model,
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Record `job_id`'s cost from its resources + compute time. Returns the units.
    pub fn record(&self, job_id: &str, resources: &ResourceRequest, wall_time_ms: u64) -> f64 {
        let units = self.model.estimate(resources, wall_time_ms);
        self.entries.lock().push(CostEntry {
            job_id: job_id.to_string(),
            units,
        });
        units
    }

    /// Total units billed across all recorded jobs. v1 sums `f64` abstract units;
    /// a real currency ledger (T3.2) should use fixed-point to avoid drift.
    pub fn total(&self) -> f64 {
        self.entries.lock().iter().map(|e| e.units).sum()
    }

    /// A snapshot of every ledger row.
    pub fn entries(&self) -> Vec<CostEntry> {
        self.entries.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_costs_more_than_cpu_for_equal_time() {
        let m = CostModel::default();
        let gpu = ResourceRequest {
            cpu_cores: 1,
            memory_gib: 0,
            gpu: true,
            gpu_vram_gib: None,
        };
        let cpu = ResourceRequest {
            cpu_cores: 1,
            memory_gib: 0,
            gpu: false,
            gpu_vram_gib: None,
        };
        assert!(m.estimate(&gpu, 1000) > m.estimate(&cpu, 1000));
    }

    #[test]
    fn cost_scales_with_wall_time() {
        let m = CostModel::default();
        let r = ResourceRequest::default();
        let one = m.estimate(&r, 1000);
        let ten = m.estimate(&r, 10_000);
        assert!((ten - one * 10.0).abs() < 1e-9);
    }

    #[test]
    fn ledger_records_and_totals() {
        let ledger = CostLedger::new(CostModel::default());
        let r = ResourceRequest {
            cpu_cores: 2,
            memory_gib: 4,
            gpu: false,
            gpu_vram_gib: None,
        };
        ledger.record("job-a", &r, 1000);
        ledger.record("job-b", &r, 2000);
        assert_eq!(ledger.entries().len(), 2);
        // 2 jobs, second twice as long → total == 3× the single-second cost.
        let single = CostModel::default().estimate(&r, 1000);
        assert!((ledger.total() - single * 3.0).abs() < 1e-9);
    }
}
