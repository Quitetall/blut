#!/usr/bin/env python3
"""Utility helpers for LamQuant models"""
import warnings
import torch


def safe_torch_load(path, map_location='cpu', device=None):
    """Load a checkpoint, preferring weights_only=True for security.

    Falls back to weights_only=False for checkpoints containing objects
    not in PyTorch's safe allowlist (e.g. optimizer state in older
    format, custom schedulers). All checkpoints produced by this
    project are safe, but older PyTorch versions (<2.4) reject dicts
    with non-tensor values under weights_only=True.
    """
    loc = device if device is not None else map_location
    try:
        return torch.load(path, map_location=loc, weights_only=True)
    except (TypeError, RuntimeError):
        warnings.warn(
            f"weights_only=True failed for {path}; falling back to "
            f"weights_only=False. Ensure this checkpoint is from a "
            f"trusted source.",
            stacklevel=2,
        )
        return torch.load(path, map_location=loc, weights_only=False)

def percent_zero_weights(model: torch.nn.Module) -> float:
    """Return the percentage of parameters that are exactly zero in their quantized state.
    Handles both standard layers and TernaryConv1d (LSQ).
    """
    total = 0
    zeros = 0
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and m.weight is not None:
            w = m.weight.data
            # If it's a Ternary Layer with LSQ Alpha
            if hasattr(m, 'lsq_alpha'):
                alpha = m.lsq_alpha.data
                # Count quantized zeros: |w| <= alpha
                zeros += (torch.abs(w) <= alpha).sum().item()
            else:
                zeros += (w == 0).sum().item()
            total += w.numel()
            
    if total == 0:
        return 0.0
    return (zeros / total) * 100.0
