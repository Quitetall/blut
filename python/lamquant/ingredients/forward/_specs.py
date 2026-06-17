"""Forward ingredient specs (ADR 0050/0051). Importing this registers them.

A *forward* ingredient encapsulates the model's forward pass for one batch —
the mask-zero → encode → predict sequence — and is built into a callable:
``build_ingredient("forward", "<name>", cfg)`` returns
``forward(encoder, pred_head, l3, mask) -> recon``.

The body is transcribed VERBATIM from the trainer inner loop; the only edit is
turning the loop's surrounding objects (encoder/pred_head, the l3/mask tensors)
into call args so the forward stays byte-identical.

One spec, ``cache_relevant=False`` — the forward pass is a deterministic function
of the model + inputs; it does not, on its own, change the trained artifact's
identity (the model arch + the loss + the mask sampler already carry that).

  * ``mae_masked`` — ``student/pretrain_mae.py``'s masked-autoencoder forward:
                     zero the masked L3 region, encode the visible patches
                     (quantize=False), predict the full L3 from the latent.
"""
from __future__ import annotations

from dataclasses import dataclass

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


@dataclass(frozen=True)
class MaeMaskedForwardConfig:
    pass


def _build_mae_masked_forward(cfg):
    def forward(encoder, pred_head, l3, mask):
        """l3 [B,21,T], mask [B,21,T] (1==masked) -> recon [B,21,T].

        Verbatim from pretrain_mae.run_pretraining (lines 196-202): zero the
        masked regions in the input, encode the visible patches (quantize=False),
        then predict the full L3 from the latent.
        """
        # Zero out masked regions in the input
        l3_masked = l3 * (1.0 - mask)

        # Encode the visible patches
        latent = encoder.encode(l3_masked, quantize=False)

        # Predict full L3 from latent
        l3_pred = pred_head(latent)
        return l3_pred

    return forward


@register_ingredient
def _mae_masked_forward_spec():
    return IngredientSpec(
        name="mae_masked", kind="forward", config_cls=MaeMaskedForwardConfig,
        cache_relevant=False,
        build=_build_mae_masked_forward,
    )
