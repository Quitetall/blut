# Student Training Pipeline

## Gen 7.1 Subband Pipeline (recommended)

1. **`subband_preprocess.py`** — Preprocess Q31 dataset with LPC order-8 + 3-level lifting DWT
   - Input: `q31_events/` (21ch x 2500 samples)
   - Output: `q31_subbands/` (L3 approx [21,313] + detail coefficients)

2. **`train_student_subband.py`** — Train `TernaryMobileNetV5_Subband` on L3 approximation
   - Architecture: width 128, 3 focal blocks + GLU bottleneck, stride 2
   - Input: [21, 313] → Latent: [32, 79] | Encoder: 104.8 KB ternary in flash (XIP)
   - Uses `PrecomputedL3Dataset` — loads L3 from RAM (~20 GB), zero disk I/O
   - Phase 1: Warm-up (50 epochs, FP32, no quantization)
   - Phase 2: QAT (200 epochs, ternary LSQ + Tequila deadzone τ=0.1 + INT16 activation quant)
   - Phase 3: Fine-tune (250 epochs, spectral loss, deadzone τ annealed 0.1→0)
   - **QAT features**: LSQ gradient scaling (1/√n), data-driven alpha init, block-WHT activation smoothing, NativeTernary-compatible encoding
   - Output: `student_subband.ckpt`

3. **`validate_subband.py`** — Validate subband pipeline across quality modes
   - Tests Alerting (~150:1), Monitoring (~80:1), Clinical (~40:1)
   - Target: R = 0.96-0.98 at clinical mode

4. **`harden_artifacts.py`** — Align latents with strided teacher for Route B deployment
   - Requires: `student_subband.ckpt` + `teacher_strided_best.ckpt`
   - Updates checkpoint in-place

## Gen 7.0 Legacy Pipeline

1. **`train_student.py`** — Train `TernaryMobileNetV5` (width 96, 4 focal blocks, stride 8)
   - Input: [21, 2500] -> Latent: [32, 312]
   - Encoder size: 42.3 KB (98.4% of 43 KB budget)
   - Output: `student_hardened.ckpt`

2. **`harden_artifacts.py`** — Align latents with strided teacher

## Experimental Scripts

- **`train_progressive_distill.py`** — Progressive distillation: Teacher → INT8 Medium (width=56) → Ternary Student. Reduces capacity gap per step for better final accuracy.

## Optional Steps

- **`finetune_student.py`** — Resume training with low LR. Safe to run multiple times.
- **`train_route_b_decoder.py`** — Train Route B FP32 decoder (alternative to strided teacher).

## Key Files

| File | Purpose |
|---|---|
| `train_ternary.py` | Model architecture + QAT primitives: `TernaryConv1d` (LSQ + Tequila + INT16 quant), `TernaryMobileNetV5` (Gen 7.0), `TernaryMobileNetV5_Subband` (Gen 7.1). Block-WHT activation smoothing. |
| `subband_preprocess.py` | LPC order-8 + 3-level Le Gall 5/3 lifting DWT + WHT32. Both forward and inverse transforms. |
| `subband_dataset.py` | `SubbandDataset` wrapper — applies LPC+lifting on-the-fly in DataLoader workers. |
| `precompute_l3_fast.py` | Precomputes L3 into NPZ files (atomic write, idempotent). |
| `train_student_subband.py` | Production student training with full QAT. |
| `train_progressive_distill.py` | (Experimental) Progressive Teacher→Medium→Student distillation. |
| `validate_subband.py` | Quality-mode validation (Alerting/Monitoring/Clinical). |
