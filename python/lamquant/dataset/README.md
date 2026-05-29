# Dataset Management and Training Splits

## Training Splits

The official release weights were trained using `official_split_config.json`.
To reproduce the exact split:

```bash
python generate_validation_split.py --config official_split_config.json
```

To train on your own data with an automatic split:

```bash
python generate_validation_split.py
```

Never train on data listed in `validation_manifest.json`.

---

## Official Data Split

### Training Data (TNN Encoder + SNN Detector)

| Dataset | Role | Subjects | Hours | Files |
|---------|------|----------|-------|-------|
| CHB-MIT chb01-chb20 | TNN primary | 20 | ~350 | 572 |
| TUH Seizure v2.0.6 | TNN + SNN | 592 | ~200 | 8139 |
| TUH Artifact v3.0.1 | SNN negatives | varies | ~100 | 310 |

### Validation Data (NEVER TRAIN ON THESE)

| Dataset | Purpose | Subjects | Files |
|---------|---------|----------|-------|
| CHB-MIT chb21-chb24 | Within-site generalization | 4 | ~114 |
| TUH Epilepsy v3.0.0 | Diagnostic transparency | 200 | 2821 |
| TUH Events v2.0.1 | Event morphology preservation | varies | 518 |
| Siena Scalp EEG | Cross-site generalization | 14 | 41 |
| EEGMMIDB | Healthy brain generalization | 109 | 1526 |
| Mental Arithmetic | Cognitive state preservation | 36 | ~36 |

### What Validation Proves

- **TUH Epilepsy**: Can a neurologist still diagnose epilepsy from compressed EEG?
- **TUH Events**: Can a neurologist still see spikes, sharp waves, and periodic discharges?
- **TUH Artifact**: Does compression create false spikes from artifacts?
- **Siena**: Does it work on equipment from a different hospital in a different country?
- **EEGMMIDB**: Does it work on healthy brains with different channel counts?

---

## Full Training Procedure

### Step 0: Download datasets

```bash
# PhysioNet (open access)
s5cmd --no-sign-request cp "s3://physionet-open/chb-mit-scalp-eeg-database/1.0.0/*" datasets/chbmit/
s5cmd --no-sign-request cp "s3://physionet-open/siena-scalp-eeg/1.0.0/*" datasets/siena/
s5cmd --no-sign-request cp "s3://physionet-open/eegmmidb/1.0.0/*" datasets/eegmmidb/
s5cmd --no-sign-request cp "s3://physionet-open/eeg-during-mental-arithmetic-tasks/1.0.0/*" datasets/mental_arithmetic/

# TUH (requires EULA — apply at isip.piconepress.com)
# After approval, download into:
#   datasets/tuh_seizure/
#   datasets/tuh_artifact/
#   datasets/tuh_epilepsy/
#   datasets/tuh_events/
```

### Step 1: Convert EDF to Q31 tensors

```bash
python edf_to_events.py --input datasets/chbmit --output q31_events --dataset chbmit
python edf_to_events.py --input datasets/tuh_seizure --output q31_events --dataset tuh
python edf_to_events.py --input datasets/tuh_artifact --output q31_events --dataset tuh
```

### Step 2: Generate validation manifest (official split)

```bash
python generate_validation_split.py --config official_split_config.json
```

This reads all EDF files, applies the locked subject assignments from the config,
and writes `validation_manifest/validation_manifest.json`. The manifest contains
only file paths and window indices — no signal data. Commit it for reproducibility.

### Step 3: Train teacher (FP32 reference)

```bash
python ../oracle/train_teacher.py --headless
# ~25 min on RTX 4090, 800 epochs
# Output: ai_models/oracle/teacher_best.ckpt
```

### Step 4: Train Gen 7.1 subband TNN

```bash
python ../student/train_student_subband.py
# ~8 min on RTX 4090, 500 epochs (3-phase: warmup, QAT, fine-tune)
# Output: weights/student_subband.ckpt
```

### Step 5: Generate SNN activity labels

```bash
python ../snn/generate_activity_labels.py \
  --input datasets/chbmit \
  --output ../snn/labels \
  --manifest validation_manifest/validation_manifest.json \
  --training-only
# ~45 min for 686 files
```

### Step 6: Train SNN activity detector

```bash
python ../snn/train_dlif_run.py \
  --data ../snn/labels \
  --epochs 100 \
  --manifest validation_manifest/validation_manifest.json
# ~12 min on RTX 4090
# Output: weights/snn_subband.pt
# Auto-exports snn_weights.h if sensitivity >= 0.99
```

### Step 7: Validate

```bash
python validate_subband.py \
  --subband-checkpoint ../../weights/student_subband.ckpt \
  --output validation_subband_report.json
```

### Step 8: Export to firmware

```bash
python ../../firmware/export_firmware.py
# Auto-detects weights/student_subband.ckpt
# Output: firmware/firmware_export/focal_net_weights.h
```

### Step 9: Commit weights

```bash
git add ../../weights/
git commit -m "weights: trained Gen 7.1 release weights"
git push
```

---

## Files in this directory

| File | Purpose |
|------|---------|
| `generate_validation_split.py` | Generates deterministic validation manifests. Works with any data. |
| `official_split_config.json` | Locked split for release weights. DO NOT MODIFY. |
| `validation_manifest/validation_manifest.json` | Generated output — committed for reproducibility. |
| `channel_resolver.py` | Single source of truth for EDF channel name resolution. |
| `edf_to_events.py` | Converts EDF files to Q31 NPZ tensors. |
| `validate_cross_dataset.py` | Cross-dataset validation suite (Gen 7.0). |
| `validate_subband.py` | Gen 7.1 subband validation with ablation study. |
| `audit_dataset.py` | Dataset integrity checker. |
| `datasets/` | Downloaded EDF datasets (gitignored). |
| `q31_events/` | Converted Q31 tensors (gitignored). |
