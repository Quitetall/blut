# Student codec — encoder/decoder training (source-of-truth index)

Trains the deployed compression path: the ternary-QAT **encoder**
(`TernaryMobileNetV5_Subband`, ships to MCU) and the **Vocos decoder**
(ships to base station).

> **NOTE (2026-06):** this index supersedes an older README that recommended
> `train_student_subband.py` as the production trainer. That file (and the
> rest of the per-arch `archive/` trainers) has been **retired** — the SOT is
> **`train_joint.py`** (joint encoder+decoder).

## ✅ Source of truth

| Purpose | File |
|---|---|
| **Trainer (SOT)** | **`train_joint.py`** — joint encoder + Vocos decoder, end-to-end on the real deployed reconstruction path. Replaces the old encoder-solo / mini-decoder pattern. |
| **Compliance eval (SOT)** | **`eval_fullband.py`** — fullband (250 Hz) PRD / Pearson-R gate |
| Config | `training_config.py`, `training_guard.py` |
| Optimizer (production) | `soap_optimizer.py` (SOAP). A/B candidates: `esoap.py`, `sinksoaph.py`, `muon_optimizer.py`, `cautious_wd.py` |

## ⛔ Retired (2026-06) — do NOT resurrect

The per-arch `archive/` trainers — `train_student_subband.py`,
`train_ternary.py`, `ternary_encoder.py`, `training_utils.py` — have been
**deleted**. Their reusable primitives were extracted to `lamquant/common/`
(losses / metrics / augment) and the canonical model classes live in the
`lamquant_neural.models.encoder` package (pulled via `joint_codec.py`). The
SOT trainer is `train_joint.py`; nothing imports the old shims.

## Encoder QAT recipe (runs inside `train_joint.py`)

`TernaryMobileNetV5_Subband`: input L3 approx `[21, 313]` → latent `[32, 79]`;
width 128, 3 focal blocks + GLU bottleneck, stride 2. Ternary QAT:
- Phase 1 — FP32 warm-up (no quantization)
- Phase 2 — ternary LSQ QAT (Tequila deadzone τ=0.1, INT16 activation quant,
  LSQ grad scaling 1/√n, data-driven alpha init, block-WHT activation smoothing)
- Phase 3 — fine-tune (spectral loss, deadzone τ annealed 0.1→0)

The quant primitives (`TernaryConv1d`, the model classes) live in the
`lamquant_neural.models.encoder` package (imported by `joint_codec.py`);
subband transforms in `subband_preprocess.py`.

## Supporting modules (not entry points)

- **Pretrain stage:** `pretrain_mae.py`
- **Model / quant libs:** `joint_codec.py`, `multiscale_fsq.py`, `progressive_quant.py`, `seizure_head.py`, `_subband_int_helpers.py` (encoder/quant model classes are in `lamquant_neural.models.encoder`)
- **Data / preprocess:** `subband_preprocess.py`, `subband_dataset.py`, `precompute_l3_fast.py`, `lma_typed_adapter.py`, `clinical_sampler.py`, `augmentations.py`
- **Train infra:** `checkpoint_manager.py`, `durable_resume.py`, `training_config.py`, `training_guard.py`
- **Runners / ops:** `experiment_runner.py`, `launch_production.py`, `ship_fast_preset.py`, `run_diagnostics.py`, `sweep_noise_bits.py`, `harden_artifacts.py`, `recon_difficulty_probe.py`

## Historical pipelines

Gen-7.0 (`train_student.py`, [21,2500]→[32,312]) and the
progressive-distill / route-B-decoder / strided-teacher scripts have been
**retired** (the meta-repo `legacy/` tree they lived in was removed). They
are not part of the current joint-training path.
