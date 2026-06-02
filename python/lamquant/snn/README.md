# SNN — training & evaluation (source-of-truth index)

The SNN is the EEG→SNAC **compression-tier controller**: each EEG latent
timestep is mapped to one of four SNAC tiers (QUIET / BASELINE /
INTERESTING / CRITICAL → CR). It is **not** a seizure detector — seizure is
one trigger of the CRITICAL tier. See `decisions/0027` and `decisions/0029`.

## ✅ Source of truth

| Purpose | File |
|---|---|
| **Trainer (SOT)** | **`train_4state_controller.py`** — the canonical SNN trainer (4-state rate–distortion objective, ADR-0029) |
| **Clinical eval (SOT)** | **`eval_event_fpr.py`** — event-level FPR/h, NEDC OVLP (full recordings) |
| Config | `snn_training_config.py` — `SNN_CONFIGS` run registry |

Run training from the SOT trainer. Do not start from anything in `archive/`.

## 🗄️ Archived (not canon — kept for reference)

| File | Why archived |
|---|---|
| `archive/train_mamba_snn.py` | Legacy **seizure-objective** trainer. All run-3…15 clinical-FPR work used it; retained as a validated baseline, superseded by `train_4state_controller.py`. |

## Supporting modules (not entry points)

- **Pretrain stage:** `pretrain_ssl_tueg.py` (optional SSL warmup on TUEG)
- **Eval / diagnostics:** `snn_event_eval.py`, `snn_to_nedc_eval.py` (NEDC CSV export), `event_scoring.py`, `snn_metrics.py`, `leaderboard.py`, `roc_diagnostic.py`, `oracle_ceiling.py`, `head_benchmark.py`
- **Data / labels:** `lma_dataset.py` (loader), `generate_activity_labels.py`, `generate_tueg_quiet_labels.py`, `ingest_new_seizure_corpus.py`, `lma_annotations.py`, `verify_lma_annotations.py`, `lma_subject_id.py`
- **Model / loss libs:** `four_state.py`, `ordinal_loss.py`, `spectral.py`, `spike_augmentation.py`, `distill_teacher.py`
- **Tests:** `tests/`, `test_event_scoring.py`
