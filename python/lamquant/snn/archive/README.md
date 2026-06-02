# archive/ — old-but-useful, NOT canon

Code here still runs and may be useful (reproducing past results, baselines),
but it is **not the source of truth**. Do not build new work on it.

Convention across the repo:
- **`archive/`** — superseded but still-useful code (e.g. a prior-generation
  trainer kept as a validated baseline).
- **`legacy/`** (repo root) — dead / deprecated code kept only for history.

| File | Status |
|---|---|
| `train_mamba_snn.py` | Legacy seizure-objective SNN trainer. SOT is `../train_4state_controller.py`. Used by all run-3…15 clinical-FPR results; kept as a baseline. |
