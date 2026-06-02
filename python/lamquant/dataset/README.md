# Dataset — manifests, splits, preprocessing (source-of-truth index)

> **NOTE (2026-06):** this index replaces an older procedure that pointed at
> `train_dlif_run.py` (dead — dLIF replaced by Mamba),
> `student/train_student_subband.py` (now a deprecated shim) and
> `oracle/train_teacher.py` (Gen-6). For the current trainers see the per-tool
> READMEs: [`../snn`](../snn/README.md), [`../student`](../student/README.md),
> [`../oracle`](../oracle/README.md).

## ✅ Source of truth

| Purpose | File |
|---|---|
| **Train/val/test split (SOT)** | **`build_seizure_split_manifest.py`** — patient-level, seizure-stratified, cross-site external-test holdout. Emits `split_manifest_vN.json` (consumed by `lamquant.snn.lma_dataset::load_split_manifest`). Run: `python -m lamquant.dataset.build_seizure_split_manifest …` |
| **Full dataset manifest** | `build_manifest.py` — typed `DatasetManifest` (replaces the old `official_split_config.json` + `validation_manifest.json` dual-source) |

## Data prep / preprocessing (support)

- `edf_to_events.py` — EDF → Q31 tensors
- `preprocess.py`, `stream_preprocess_tueg.py` — subband preprocess (LPC + lifting)
- `precompute_fullband_memmap.py`, `ensure_l3_precomputed.py` — training caches
- `channel_resolver.py` — **SOT for EDF channel-name resolution**
- `vet_montage.py`, `audit_dataset.py` — montage vetting / integrity checks

## ⛔ Legacy (Gen-7.1, superseded — do not use for new work)

| File | Superseded by |
|---|---|
| `generate_validation_split.py` + `official_split_config.json` | `build_seizure_split_manifest.py` (split) / `build_manifest.py` (manifest) |
| `validate_subband.py`, `validate_cross_dataset.py` | per-tool eval SOTs (`../student/eval_fullband.py`, `../snn/eval_event_fpr.py`) |
| `manifest_utils.py` legacy loaders | `DatasetManifest.load()` (typed) |

Kept in place for back-compat / reproduction of pre-v3 results; candidates for
relocation to the repo `legacy/` tree in a later quiescent pass.

## Tests
`test_build_split.py` (split builder).
