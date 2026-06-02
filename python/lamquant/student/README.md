# Student codec — encoder/decoder training (source-of-truth index)

Trains the deployed compression path: the ternary-QAT **encoder**
(`TernaryMobileNetV5_Subband`, ships to MCU) and the **Vocos decoder**
(ships to base station).

> **NOTE (2026-06):** this index supersedes an older README that recommended
> `train_student_subband.py` as the production trainer. That file is now a
> deprecated shim — the SOT is **`train_joint.py`** (joint encoder+decoder).

## ✅ Source of truth

| Purpose | File |
|---|---|
| **Trainer (SOT)** | **`train_joint.py`** — joint encoder + Vocos decoder, end-to-end on the real deployed reconstruction path. Replaces the old encoder-solo / mini-decoder pattern. |
| **Compliance eval (SOT)** | **`eval_fullband.py`** — fullband (250 Hz) PRD / Pearson-R gate |
| Config | `training_config.py`, `training_guard.py` |
| Optimizer (production) | `soap_optimizer.py` (SOAP). A/B candidates: `esoap.py`, `sinksoaph.py`, `muon_optimizer.py`, `cautious_wd.py` |

## ⛔ Deprecated shims — do NOT import (use the target instead)

| Shim | Import from instead |
|---|---|
| `train_ternary.py` | `ternary_encoder.py` |
| `train_student_subband.py` | `training_utils.py` |

Thin re-export shims kept for back-compat only; a canonical copy also lives
under the repo `legacy/training/shims/`. Slated for removal once remaining
cross-repo importers (e.g. Lossless `export_firmware.py`) are repointed.

## Encoder QAT recipe (runs inside `train_joint.py`)

`TernaryMobileNetV5_Subband`: input L3 approx [21, 313] → latent [32, 79];
width 128, 3 focal blocks + GLU bottleneck, stride 2. Ternary QAT:
- Phase 1 — FP32 warm-up (no quantization)
- Phase 2 — ternary LSQ QAT (Tequila deadzone τ=0.1, INT16 activation quant,
  LSQ grad scaling 1/√n, data-driven alpha init, block-WHT activation smoothing)
- Phase 3 — fine-tune (spectral loss, deadzone τ annealed 0.1→0)

The quant primitives (`TernaryConv1d`, the model classes) live in
`ternary_encoder.py`; subband transforms in `subband_preprocess.py`.

## Supporting modules (not entry points)

- **Pretrain stage:** `pretrain_mae.py`
- **Model / quant libs:** `ternary_encoder.py`, `joint_codec.py`, `multiscale_fsq.py`, `progressive_quant.py`, `seizure_head.py`, `_subband_int_helpers.py`
- **Data / preprocess:** `subband_preprocess.py`, `subband_dataset.py`, `precompute_l3_fast.py`, `lma_typed_adapter.py`, `clinical_sampler.py`, `augmentations.py`
- **Train infra:** `training_utils.py`, `checkpoint_manager.py`
- **Runners / ops:** `experiment_runner.py`, `launch_production.py`, `ship_fast_preset.py`, `run_diagnostics.py`, `sweep_noise_bits.py`, `harden_artifacts.py`, `recon_difficulty_probe.py`

## Historical pipelines

Gen-7.0 (`train_student.py`, [21,2500]→[32,312]) and the
progressive-distill / route-B-decoder / strided-teacher scripts have moved to
the repo `legacy/` tree. They are not part of the current joint-training path.
