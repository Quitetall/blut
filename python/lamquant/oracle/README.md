# Oracle / teacher — training (source-of-truth index)

Trains the high-capacity FP32 teacher used as the distillation target for
hardening the ternary student encoder.

## ✅ Source of truth

| Purpose | File |
|---|---|
| **Trainer (SOT)** | **`train_l3_teacher.py`** — Gen-7.5 L3-native teacher. FP32 autoencoder on L3 subband [21, 313] → latent [32, 79], matching the student exactly so hardening measures only the cost of ternarization. (~8.3M params, width 512.) |

## 🗄️ Legacy (superseded)

| File | Why |
|---|---|
| `train_teacher.py` | Gen-6 FP32 oracle teacher trainer. Superseded by the L3-native Gen-7.5 teacher above (different latent contract). Kept for reference. |

## Supporting modules (not entry points)

- **Model:** `teacher_arch.py` (teacher architecture, formerly `architectures/teacher.py`)
- **Data:** `streaming_dataset.py`, `dataset_with_manifest_filter.py`
- **Loss:** `freq_weighted_loss.py`
