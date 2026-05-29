# INR Experiment Results

## Phase 1.6: Real EEG Validation (TUAR, 100 recordings)
- 5K params SIREN on L4 (4-level DWT, 1 kHz)
- R median: 0.992
- PCA manifold: 63 dims (95% variance) — first run
- Entropy: 0.29 bits/param (13.7x gain over INT4)
- Per-window size: 182 bytes (entropy-coded)

## Phase 1.7A: Cross-Corpus Manifold Consistency (250 recordings, 5 corpora)

| Corpus | N | R median | 95% dims | 99% dims | Entropy |
|--------|---|----------|----------|----------|---------|
| TUAR (artifacts) | 50 | 0.993 | 39 | 46 | 0.30 |
| TUEG (general) | 50 | 0.994 | 39 | 46 | 0.29 |
| TUSZ (seizures) | 50 | 0.994 | 38 | 46 | 0.29 |
| TUAB (abnormal) | 50 | 0.992 | 39 | 46 | 0.29 |
| TUSL (slowing) | 50 | 0.995 | 39 | 46 | 0.28 |

**Verdict: PASS.** Max manifold dim = 39. Universal weight predictor viable.

Key findings:
- Manifold dimension consistent (38-39) across ALL clinical EEG types
- Seizure recordings (TUSZ) show identical structure to normal EEG at L4 level
- Pathology lives in detail subbands, not smooth L4 component
- One universal predictor with 40 basis vectors sufficient

Architecture implications:
- Latent: 40-dim (design for 64 with safety margin)
- Basis vectors: 40 x 4993 = 200K params (200 KB flash)
- Total MCU decode: ~15ms (well under 50ms target)

## Phase 1.7B: Hypernetwork Prototype — FAIL

**Tested:** Can a small encoder (100-500K params) predict SIREN PCA coordinates
from raw L4 signal in one forward pass?

**Result: NO.** Two architectures tested:

1. End-to-end (encoder → basis predictor → SIREN eval → loss): R=0.12
2. Two-stage (supervised pre-train on PCA coords, then fine-tune): R=0.06

Both approaches produce reconstructions indistinguishable from noise.

**Root cause analysis:**
- Stage 1 supervised training shows encoder can't even learn signal→PCA-coord mapping
  (val loss=0.25 = 25% error on normalized coords)
- The mapping from L4 signal VALUES to optimal SIREN WEIGHTS is too complex
  for a small conv encoder — the relationship is highly nonlinear
- SIREN weight space is non-smooth (sin() causes small weight changes → large output changes)
- PCA assumes linear manifold; actual manifold likely nonlinear and folded

**What this means for Mode 4:**
- MCU ENCODING of Mode 4 is NOT VIABLE (can't predict latent in real-time)
- MCU DECODING of Mode 4 IS VIABLE (PCA basis + SIREN eval is cheap)
- Mode 4 repositioned: base-station-encode, universal-decode

**Revised architecture:**
```
ENCODE (base station, offline):
  L4 signal → per-signal SIREN fitting (1500 epochs, ~0.5s/window on GPU)
  → project weights to PCA basis → store 40-64 PCA coordinates (INT8)
  → ~40-60 bytes per window

DECODE (MCU or anywhere):
  40-60 bytes → PCA coords → reconstruct weights via basis multiply
  → SIREN eval → L4 reconstruction (~15ms on RP2350)
```

**What's preserved:**
- Compression result (180 bytes per L4 window at R~0.99)
- MCU-decodable representation (killer feature)
- Universal decoder portability
- Cross-corpus universality of the manifold

**What's lost:**
- Real-time MCU encoding (wearable can't produce Mode 4)
- One-pass amortized inference at encode time

**Open question (Path 2):**
- Would a 5-10M param transformer encoder learn the mapping?
- Worth 1-2 weeks of research. Doesn't change Mode 4 v0 design.

## Phase 1.7B RERUN (Fixed Eval Bug) — Architecture Conclusion

**Bug found:** All previous hypernetwork experiments had transposed weight matrices
in the SIREN eval code (PyTorch Linear stores [out, in], eval assumed [in, out]).
Caught via contradiction: perfect supervised loss + garbage reconstruction.

**After fix — critical finding from PCA reconstruction test:**

| PCA K | SIREN R median |
|-------|----------------|
| 20 | 0.025 |
| 40 | 0.030 |
| 64 | 0.055 |
| 100 | 0.016 |
| 160 (all) | 0.997 |

**PCA-compressed SIREN weights produce invalid SIRENs regardless of K.**
Only lossless (K=all components) works. This kills:
- Latent-SIREN architecture (PCA projection doesn't preserve function)
- Hypernetwork prediction (no valid target representation to predict)
- Any linear dimensionality reduction of SIREN weights

**Root cause:** SIREN with sin() activations is catastrophically sensitive to
STRUCTURED perturbation (PCA truncation) while tolerating UNSTRUCTURED perturbation
(INT4 independent quantization noise). The "manifold" exists by variance metric
but is functionally meaningless — the 5% "unimportant" variance is essential
for sin() to produce coherent output.

**Lesson:** Variance-based manifold analysis can mislead for nonlinear models.
Always validate by function-preservation, not just variance retention.

## FINAL MODE 4 ARCHITECTURE (Research Phase Complete)

```
ENCODE (base station, GPU, ~0.5s/window):
  L4 [21, 625] → per-signal SIREN fit (1500 epochs)
  → INT4 quantize each weight independently (4993 weights)
  → rANS entropy code → ~182 bytes

DECODE (MCU, RP2350, 30ms):
  ~182 bytes → rANS decode → INT4 weights (4993 params, 2.5 KB)
  → SIREN eval (625 timesteps × 21 channels) → L4 reconstruction
  Flash: 200 KB (basis not needed — direct weight storage)
  SRAM: 31 KB working

FULL RECONSTRUCTION (base station):
  L4 (from SIREN) + detail subbands (lossless Golomb-Rice)
  → inverse 4-level lifting DWT → full signal [21, 10000]
  R > 0.99
```

**What works:**
- Per-signal SIREN fit: R=0.992 median on real EEG (100+ recordings, 5 corpora)
- INT4 quantization + rANS: 182 bytes per 10s window (validated)
- MCU decode: 30ms on RP2350 (estimated, hardware test pending)
- Universal across clinical EEG types (TUAR, TUEG, TUSZ, TUAB, TUSL)

**What doesn't work:**
- PCA/latent compression of SIREN weights
- Hypernetwork prediction of weights or latents
- Any dimensionality reduction that removes weight-space directions

**Product positioning:**
Mode 4 = base-station-encode + universal MCU-decode. Value is decode portability
and archival compression of smooth component, not real-time wearable encoding.
