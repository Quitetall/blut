# INR/SIREN Research — Final Conclusion

**Status: CLOSED. Not viable for EEG compression.**

## The End-to-End Number That Killed It

```
Per-signal SIREN fit (FP32): R=0.99, 20 KB per window
Per-signal SIREN fit (FP16): R=0.99, 10 KB per window
Per-signal SIREN fit (INT8): R=0.13 — DESTROYED
Per-signal SIREN fit (INT4): R=0.005 — DESTROYED

Existing lossless (LML Mode 3): R=1.000, ~30 KB per window (4.8:1 CR)
```

SIREN requires FP16 minimum precision. At FP16 it's 10 KB for L4 only (not full signal).
Combined with lossless detail subbands (~30 KB), total Mode 4 = ~40 KB = 3.5:1 CR.
This is WORSE than existing lossless (4.8:1) while being lossy (R=0.99 < R=1.000).

**SIREN-based INR does not beat existing lossless compression for EEG.**

## Why

SIREN with sin() activations is catastrophically sensitive to weight perturbation:
- INT4 quantization: R drops from 0.99 to 0.005 (noise)
- INT8 quantization: R drops from 0.99 to 0.13 (garbage)
- FP16 quantization: R=0.99 preserved (works but too large)
- PCA truncation: R drops to 0.03 regardless of K
- Gaussian noise at INT4 magnitude: R=-0.01

The sin() nonlinearity amplifies small weight errors into completely different
output waveforms. No quantization scheme below FP16 produces valid output.

## What Was Validated vs What Was Assumed

| Claim | Validated? | End-to-end? |
|-------|-----------|-------------|
| SIREN fits EEG at R=0.99 | YES | YES (FP32) |
| INT4 preserves quality | **NO** | **NEVER TESTED** until final day |
| 182 bytes per window | **INVALID** | Entropy of quantized weights, never evaluated |
| 39-dim manifold | Real by variance | **Functionally meaningless** (PCA breaks SIREN) |
| MCU decode at 30ms | Valid for FP16 weights | Requires 10 KB per window (not 182 bytes) |

## Lesson Learned

**Only end-to-end numbers matter.** Intermediate metrics (entropy, PCA variance,
supervised loss, per-signal fit R at FP32) are signals but never final answers.
The final answer is always: compress → store → decompress → evaluate → compare.

Every claim about compression must include the ACTUAL decoded output quality.
"Entropy says 182 bytes" is meaningless without "and those 182 bytes decode to R=0.99."

## Could INR Help Elsewhere in the Pipeline?

Possible residual value (none validated, would need end-to-end testing):

1. **SIREN as decoder initialization/prior (not compression):**
   A pre-trained SIREN for "average EEG" could provide a warm-start for the
   VocosDecoder's iSTFT head. The SIREN output conditions the decoder rather
   than replacing it. Not compression — architecture component. Unvalidated.

2. **SIREN for continuous-time upsampling (not compression):**
   Given a decoded signal at one rate, fit a SIREN at FP32, then evaluate at
   higher rate. Not storage — post-processing tool. Requires per-signal fit
   (0.5s on GPU). Marginal utility vs standard interpolation.

3. **SIREN as data augmentation during training:**
   Fit SIRENs to training data, evaluate at random phase offsets or slightly
   different time coordinates. Generates realistic augmented samples. Unvalidated.

4. **SIREN for adaptive temporal resolution in archival:**
   Store FP16 SIREN weights alongside lossless data. Enables continuous-rate
   access to the smooth component without resampling. Marginal value since
   lossless already stores exact samples.

**None of these are compelling enough to pursue.** The existing pipeline
(lifting DWT + TNN encoder + Vocos decoder + Golomb-Rice lossless) is
superior on all metrics. INR adds complexity without benefit.

## Final Status

- Mode 4 (INR): **CANCELLED**
- SIREN code: kept in `lamquant_neural/models/siren.py` for reference
- Experiments: kept in `blut/python/lamquant/experiments/` for documentation
- No further development planned
