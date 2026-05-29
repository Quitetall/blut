#!/usr/bin/env bash
# ============================================================
# download_datasets.sh — Download all EEG datasets for LamQuant
# ============================================================
# Usage: ./download_datasets.sh [DEST_DIR]
# Default destination: ./datasets/
#
# Requires: aws-cli (pip install awscli) or wget
# No AWS account needed — PhysioNet S3 is public

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$SCRIPT_DIR/datasets}"
mkdir -p "$DEST"

echo "=========================================="
echo " LamQuant EEG Dataset Downloader"
echo " Destination: $DEST"
echo "=========================================="

# ============================================================
# 1. CHB-MIT Scalp EEG Database (TRAINING SET)
#    22 subjects, 664 EDF files, ~40 GB
#    256 Hz, 23 channels, 198 seizures annotated
#    This is your PRIMARY training dataset
# ============================================================
echo ""
echo "[1/5] CHB-MIT Scalp EEG Database (~40 GB)"
echo "  22 subjects, 664 files, 198 seizures"
echo "  Source: PhysioNet S3 (no auth required)"
echo ""

if command -v aws &>/dev/null; then
    echo "  Using aws s3 sync (fastest)..."
    aws s3 sync --no-sign-request \
        s3://physionet-open/chbmit/1.0.0/ \
        "$DEST/chbmit/" \
        --exclude "*.pdf"
else
    echo "  aws-cli not found, using wget (slower)..."
    wget -r -N -c -np -nH --cut-dirs=3 \
        -P "$DEST/chbmit/" \
        "https://physionet.org/files/chbmit/1.0.0/"
fi
echo "  ✓ CHB-MIT download complete"


# ============================================================
# 2. Siena Scalp EEG Database (VALIDATION SET — NOT IN TRAINING)
#    14 subjects, adults (ages 20-71), epilepsy
#    Different hospital, different equipment, different population
#    THIS IS YOUR CROSS-SITE VALIDATION
# ============================================================
echo ""
echo "[2/5] Siena Scalp EEG Database (~5 GB)"
echo "  14 subjects, adults, different site from CHB-MIT"
echo "  Source: PhysioNet S3"
echo ""

if command -v aws &>/dev/null; then
    aws s3 sync --no-sign-request \
        s3://physionet-open/siena-scalp-eeg/1.0.0/ \
        "$DEST/siena/"
else
    wget -r -N -c -np -nH --cut-dirs=3 \
        -P "$DEST/siena/" \
        "https://physionet.org/files/siena-scalp-eeg/1.0.0/"
fi
echo "  ✓ Siena download complete"


# ============================================================
# 3. EEG Motor Movement/Imagery Dataset (VALIDATION SET)
#    109 subjects, 64 channels, 160 Hz
#    Healthy subjects, motor imagery BCI tasks
#    Tests your codec on HEALTHY brains (not just epilepsy)
# ============================================================
echo ""
echo "[3/5] EEG Motor Movement/Imagery Dataset (~3.4 GB)"
echo "  109 subjects, 64 channels, healthy subjects"
echo "  Source: PhysioNet S3"
echo ""

if command -v aws &>/dev/null; then
    aws s3 sync --no-sign-request \
        s3://physionet-open/eegmmidb/1.0.0/ \
        "$DEST/eegmmidb/"
else
    wget -r -N -c -np -nH --cut-dirs=3 \
        -P "$DEST/eegmmidb/" \
        "https://physionet.org/files/eegmmidb/1.0.0/"
fi
echo "  ✓ EEGMMIDB download complete"


# ============================================================
# 4. Mental Arithmetic EEG Dataset (VALIDATION SET)
#    36 subjects, 19 channels, 500 Hz
#    Resting + mental arithmetic tasks
#    Tests codec on cognitive load conditions
# ============================================================
echo ""
echo "[4/5] Mental Arithmetic EEG Dataset (~200 MB)"
echo "  36 subjects, cognitive tasks"
echo "  Source: PhysioNet S3"
echo ""

if command -v aws &>/dev/null; then
    aws s3 sync --no-sign-request \
        s3://physionet-open/eeg-during-mental-arithmetic-tasks/1.0.0/ \
        "$DEST/mental_arithmetic/"
else
    wget -r -N -c -np -nH --cut-dirs=3 \
        -P "$DEST/mental_arithmetic/" \
        "https://physionet.org/files/eeg-during-mental-arithmetic-tasks/1.0.0/"
fi
echo "  ✓ Mental Arithmetic download complete"


# ============================================================
# 5. TUH EEG Corpora (REQUIRES REGISTRATION)
#    Four corpora used for training / SNN:
#      - TUSZ (seizure)    — 592 subjects
#      - TUAR (artifact)   — artifact event annotations
#      - TUEP (epilepsy)   — epilepsy vs non-epilepsy
#      - TUEV (events)     — 6-class event annotations (excluded from SNN)
#    Register once at: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/
#    After approval, download each corpus via rsync.
# ============================================================
echo ""
echo "[5/5] TUH EEG Corpora (TUSZ + TUAR + TUEP + TUEV)"
echo "  ⚠ Requires NEDC registration at:"
echo "  https://isip.piconepress.com/projects/nedc/html/tuh_eeg/"
echo ""
echo "  After approval, run these rsync commands (password supplied by NEDC):"
echo ""
echo "  # TUH Seizure Corpus (~55 GB)"
echo "  rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_seizure/v2.0.3/ $DEST/tuh_seizure/"
echo ""
echo "  # TUH Artifact Corpus (~6 GB)"
echo "  rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_artifact/v3.0.1/ $DEST/tuh_artifact/"
echo ""
echo "  # TUH Epilepsy Corpus (~20 GB)"
echo "  rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/ $DEST/tuh_epilepsy/"
echo ""
echo "  # TUH Events Corpus (~7 GB)"
echo "  rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_events/v2.0.1/ $DEST/tuh_events/"
echo ""
echo "  Skipping automated download."
echo ""


# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo " Download Summary"
echo "=========================================="
echo ""
du -sh "$DEST"/*/ 2>/dev/null || echo "  (calculating...)"
echo ""
echo " Dataset splits for LamQuant (see official_split_config.json):"
echo "   TRAINING:    CHB-MIT (chb01-chb20) + TUSZ/TUAR/TUEP/TUEV (ALL subjects)"
echo "   HOLDOUT:     CHB-MIT (chb21-chb24)"
echo "   VALIDATION:  Siena — cross-site epilepsy"
echo "   VALIDATION:  EEGMMIDB — healthy subjects, motor imagery"
echo "   VALIDATION:  Mental Arithmetic — cognitive load"
echo ""
echo "   SNN training:  chbmit, tuh_seizure, tuh_artifact"
echo "   SNN excluded:  tuh_epilepsy, tuh_events"
echo ""
echo " Next steps:"
echo "   1. bash scripts/convert_tuh_to_npz.sh          # EDF → Q31 NPZ + regen manifest"
echo "   2. python ai_models/dataset_sim/validate_cross_dataset.py"
echo "=========================================="
