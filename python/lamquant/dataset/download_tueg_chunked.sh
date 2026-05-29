#!/bin/bash
# download_tueg_chunked.sh — Stream-and-delete TUEG in patient-prefix batches.
# Downloads ~2-3K patients per batch, preprocesses to NPZ, deletes raw, repeats.
# Fits in available disk by never holding more than one batch of raw EDFs.
set -e
cd /mnt/4tb/LamQuant

TUEG_SRC="nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg/v2.0.1"
SSH_KEY="$HOME/.ssh/id_ed25519"
TUEG_RAW="/mnt/4tb/tueg_raw"
Q31_OUT="ai_models/dataset_sim/q31_events"
LOG_DIR="/tmp/tueg_download_logs"

mkdir -p "$TUEG_RAW" "$LOG_DIR"

# Patient IDs are 8-char lowercase alpha. Batch by 6th char (a-z).
# Each batch is ~500-3000 patients depending on density.
PREFIXES="a b c d e f g h i j k l m n o p q r s t u v w x y z"
BATCH=0
TOTAL=$(echo $PREFIXES | wc -w)

for P in $PREFIXES; do
    BATCH=$((BATCH + 1))
    echo ""
    echo "========================================"
    echo "  BATCH $BATCH/$TOTAL: patients aaaaa${P}*"
    echo "  $(date)"
    echo "========================================"
    
    FREE_GB=$(df -BG /mnt/4tb | tail -1 | awk '{print $4}' | tr -d 'G')
    echo "  Free disk: ${FREE_GB} GB"
    if [ "$FREE_GB" -lt 100 ]; then
        echo "  [!] Less than 100 GB free — stopping"
        break
    fi
    
    # Download this patient prefix batch
    echo "  [1/3] Downloading aaaaa${P}* patients..."
    rsync -auvxL -e "ssh -i $SSH_KEY" \
        --include='*/' \
        --include="edf/*/aaaaa${P}*/**" \
        --exclude='*' \
        "$TUEG_SRC/" "$TUEG_RAW/" \
        > "$LOG_DIR/rsync_batch_${P}.log" 2>&1 || true
    
    N_EDF=$(find "$TUEG_RAW" -name "*.edf" -type f 2>/dev/null | wc -l)
    echo "  Downloaded: $N_EDF EDFs"
    
    if [ "$N_EDF" -eq 0 ]; then
        echo "  [skip] No EDFs for prefix aaaaa${P}"
        rm -rf "$TUEG_RAW"/*
        continue
    fi
    
    # Preprocess to Q31 NPZ
    echo "  [2/3] Preprocessing to Q31 NPZ..."
    python -u ai_models/dataset_sim/edf_to_events.py \
        --input "$TUEG_RAW" \
        --output "$Q31_OUT" \
        --dataset tuh \
        --skip-existing \
        > "$LOG_DIR/preprocess_batch_${P}.log" 2>&1 || true
    
    # Delete raw EDFs
    echo "  [3/3] Deleting raw EDFs..."
    rm -rf "$TUEG_RAW"/*
    
    TOTAL_NPZ=$(ls "$Q31_OUT"/*.npz 2>/dev/null | wc -l)
    echo "  Total NPZs so far: $TOTAL_NPZ"
done

echo ""
echo "========================================"
echo "  DOWNLOAD BATCHES COMPLETE — $(date)"
echo "========================================"

# L3 precomputation on all new files
echo "Running precompute_l3_fast.py..."
python -u ai_models/student/precompute_l3_fast.py \
    --input "$Q31_OUT" \
    > "$LOG_DIR/precompute_l3.log" 2>&1 || true

# Rebuild manifest
echo "Rebuilding manifest_v3.json..."
python -u ai_models/dataset_sim/build_manifest.py \
    --output ai_models/dataset_sim/manifest_v3.json \
    > "$LOG_DIR/rebuild_manifest.log" 2>&1 || true

rm -rf "$TUEG_RAW"

TOTAL_NPZ=$(ls "$Q31_OUT"/*.npz 2>/dev/null | wc -l)
echo ""
echo "========================================"
echo "  ALL DONE — $(date)"
echo "  Total NPZs: $TOTAL_NPZ"
echo "========================================"
