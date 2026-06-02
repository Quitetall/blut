# Oracle / teacher — training (NO current source of truth)

The oracle trains the high-capacity FP32 teacher used as the distillation
target for hardening the ternary student encoder.

## ⚠ No SOT yet — slated for rewrite

There is **no designated source-of-truth trainer** for the oracle at this
time. The existing trainers below are prior-generation and are **not** canon;
the oracle is to be rewritten. Do not treat either as the standard.

| File | What it is (neither is SOT) |
|---|---|
| `train_l3_teacher.py` | Gen-7.5 L3-native teacher (most recent of the two). FP32 autoencoder on L3 subband [21, 313] → latent [32, 79] matching the student. The least-stale starting point, but not adopted as canon. |
| `train_teacher.py` | Gen-6 FP32 oracle teacher. Older; superseded by the L3-native attempt. |

When the rewrite lands, designate its trainer here as the SOT and move the
above into `archive/`.

## Supporting modules

- **Model:** `teacher_arch.py` (teacher architecture, formerly `architectures/teacher.py`)
- **Data:** `streaming_dataset.py`, `dataset_with_manifest_filter.py`
- **Loss:** `freq_weighted_loss.py`
