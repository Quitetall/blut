"""GETA-inspired joint pruning for decoder efficiency.

Structured channel pruning based on importance scoring. After decoder
training converges, prune low-importance channels to reduce inference
cost for deployment on constrained GPU tiers (school PCs, mobile).

GETA (2025) observation: jointly pruning encoder+decoder is better than
pruning each independently. For LamQuant, the encoder is ternary-fixed
(deployed to MCU), so we only prune the decoder.

Usage:
    from geta_pruning import compute_importance, prune_decoder

    # Score channel importance
    importance = compute_importance(decoder, val_loader, device)

    # Prune bottom 20% of channels
    pruned = prune_decoder(decoder, importance, ratio=0.2)
"""

import torch
import torch.nn as nn
import numpy as np


def compute_importance(decoder, dataloader, device, n_batches=50):
    """Compute per-channel importance via gradient-weighted activation magnitude.

    For each ConvNeXt block, the importance of channel c is:
        I(c) = mean(|activation[c]| * |grad[c]|)

    This combines Taylor expansion (grad * activation) with magnitude pruning.

    Returns: dict mapping layer_name -> importance_scores [dim]
    """
    decoder.eval()
    importance = {}

    # Register hooks to capture activations
    activations = {}
    hooks = []

    for name, module in decoder.named_modules():
        if isinstance(module, nn.Conv1d) and 'blocks' in name:
            def hook_fn(mod, inp, out, name=name):
                activations[name] = out.detach()
            hooks.append(module.register_forward_hook(hook_fn))

    # Accumulate importance across batches
    for batch_idx, (x_l3, *_) in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        x_l3 = x_l3.to(device)
        x_l3.requires_grad_(True)

        # Forward + backward to get gradients
        out = decoder(x_l3)
        loss = out.abs().mean()
        loss.backward()

        # Score each captured activation
        for name, act in activations.items():
            if act.grad_fn is not None:
                grad = torch.autograd.grad(loss, act, retain_graph=True)[0]
            else:
                grad = torch.ones_like(act)
            # Per-channel importance: mean over batch and time
            channel_imp = (act.abs() * grad.abs()).mean(dim=(0, 2))  # [dim]
            if name not in importance:
                importance[name] = channel_imp.cpu()
            else:
                importance[name] += channel_imp.cpu()

        activations.clear()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Normalize
    for name in importance:
        importance[name] /= n_batches

    return importance


def prune_decoder(decoder, importance, ratio=0.2):
    """Structured channel pruning: zero out bottom `ratio` fraction of channels.

    This is soft pruning (zeroing, not removing) so the architecture stays
    compatible with existing checkpoints. For hard pruning, export to ONNX
    and use ONNX optimizer to remove dead channels.

    Args:
        decoder: VocosDecoder instance
        importance: dict from compute_importance()
        ratio: fraction of channels to prune (0.2 = prune 20%)

    Returns:
        n_pruned: total channels zeroed
    """
    n_pruned = 0
    for name, module in decoder.named_modules():
        if name in importance and hasattr(module, 'weight'):
            scores = importance[name]
            n_channels = len(scores)
            n_to_prune = int(n_channels * ratio)
            if n_to_prune == 0:
                continue

            # Find lowest-importance channels
            _, indices = scores.sort()
            prune_indices = indices[:n_to_prune]

            # Zero out pruned channels
            with torch.no_grad():
                module.weight.data[prune_indices] = 0
                if module.bias is not None:
                    module.bias.data[prune_indices] = 0

            n_pruned += n_to_prune

    return n_pruned
