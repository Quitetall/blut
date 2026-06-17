"""Forward ingredients (ADR 0050/0051) — the model forward pass for one batch.

One spec, ``kind="forward"``:

  * ``mae_masked`` — the MAE masked-autoencoder forward (``pretrain_mae``):
                     mask-zero the input, encode visible patches, predict L3.
"""
