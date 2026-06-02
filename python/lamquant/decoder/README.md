# Decoder — training (source-of-truth index)

The decoder reconstructs fullband EEG from the encoder's quantized latent. It
ships to the **base station** (FP32), not the MCU.

## Shipping decoder is trained jointly

The **deployed** Tier-3 decoder is trained **with the encoder** in
[`../student/train_joint.py`](../student/README.md) (the student SOT) — that is
the source of truth for the decoder that ships alongside the production encoder.
Do not train the shipping decoder standalone.

## Standalone / large-decoder trainers (Route B)

These train **bigger base-station decoders** independently of the encoder. They
are separate products by tier, not competing copies:

| File | Role |
|---|---|
| `run_decoder_tier.py` | **Production Route B** — Tiers 5/6/7 (100M/400M/837M), latent [32,79] → fullband [21,2500] via iSTFT, tokens-only 274:1. Most-referenced; the standard standalone-decoder trainer. |
| `train_vocos_decoder.py` | Gen-7.5 Tier-3 Vocos decoder, 2-phase (reconstruction + adversarial). |
| `train_combined.py` | Trains the L3 teacher + Vocos decoder simultaneously (~40% faster than separately); convenience, not a distinct model. |

> ⚠ **SOT not formally locked.** Which standalone decoder route/tier is canonical
> is a product decision (mirrors the oracle's in-flux state). `run_decoder_tier.py`
> is the de-facto production trainer today; confirm before treating it as THE SOT.

## Supporting modules
`discriminator.py`, `perceptual_losses.py`, `flow_postfilter.py`,
`raw_window_dataset.py`, `federated.py`, `geta_pruning.py`.
