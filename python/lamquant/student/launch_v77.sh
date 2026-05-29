#!/bin/bash
# v7.7 Production Launch — infinite LR mode
#
# Validated stack (all multi-seed A/B, 2026-04-17):
#   SOAP optimizer:     +0.0135 R over AdamW (3-seed)
#   Fixed R gradient:   +0.003 R (2-seed)
#   V1 decoder:         +0.0064 R over V2 (3-seed)
#   GAN (MPD+MS-STFT):  +0.014 R (prior ablation)
#
# Infinite LR: stable phase runs forever at peak LR.
# Every checkpoint is shippable. When saturated, run:
#   python ai_models/student/trigger_decay.py --epochs 40
# to start cosine cooldown for final checkpoint.
#
# Prerequisites:
#   1. Manifest rebuilt: python ai_models/dataset_sim/build_manifest.py
#   2. Fullband memmaps: python ai_models/dataset_sim/precompute_fullband_memmap.py
#   3. L3 precompute: python ai_models/student/precompute_l3_fast.py
#
set -e
cd /mnt/4tb/LamQuant

echo "=== v7.7 Production Launch (infinite LR) ==="
echo "$(date)"
echo ""

# Preflight
python -c "
from ai_models.data_types import DatasetManifest
m = DatasetManifest.load('ai_models/dataset_sim/manifest_v3.json')
print(f'Manifest: {m.train_files:,} train, {m.val_files:,} val, {m.total_windows:,} windows')
assert m.train_files > 1000, f'Too few train files: {m.train_files}'
"
echo "Manifest: OK"

for f in ai_models/dataset_sim/fullband_train.dat ai_models/dataset_sim/fullband_val.dat; do
    [ -f "$f" ] || { echo "MISSING: $f — run precompute_fullband_memmap.py"; exit 1; }
done
echo "Memmaps: OK"
echo ""

python -u ai_models/student/train_joint.py \
    --config production \
    --tier 3 \
    --lr-schedule soap \
    --infinite-lr \
    --fullband-mode memmap \
    --no-compile \
    --gan \
    --seizure-head \
    --clinical-sampling \
    --augment moderate \
    --ema \
    --seed 42
