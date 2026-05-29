"""SNAC-inspired multi-scale FSQ for EEG latent quantization.

Instead of flat FSQ across all 79 timesteps at one resolution, quantize at
multiple temporal strides. Captures both slow structure (delta/theta) and
fast detail (alpha/beta) with appropriate resolution at each scale.

For latent [32, 79]:
  Scale 0 (stride 8): [32, 10] — 1 Hz equivalent, captures delta envelope
  Scale 1 (stride 4): [32, 20] — 2 Hz, theta modulation
  Scale 2 (stride 2): [32, 40] — 4 Hz, alpha rhythm
  Scale 3 (stride 1): [32, 79] — 8 Hz, full resolution residual

Each scale gets its own FSQ level assignment: coarser scales get fewer
levels (lower rate), finer scales get more levels (capture detail).

Encoding produces a multi-scale token set. Decoding reconstructs from
coarse to fine (residual refinement).

Usage:
    msfsq = MultiScaleFSQ(dim=32, T=79, strides=[8,4,2,1], levels=[3,3,5,5])
    tokens, scales = msfsq.encode(latent)   # multi-scale tokens
    recon = msfsq.decode(tokens, scales)     # reconstructed latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleFSQ(nn.Module):
    """Multi-scale Finite Scalar Quantization inspired by SNAC.

    Quantizes a latent tensor at multiple temporal resolutions. Coarse
    scales capture slow structure; fine scales capture residuals.

    The forward pass:
    1. Downsample latent to each scale's resolution
    2. Quantize each scale with FSQ at its assigned level
    3. Upsample coarse reconstruction and subtract from original
    4. Next scale quantizes the residual

    This is hierarchical VQ in the temporal dimension with FSQ instead of
    learned codebooks — no codebook collapse, no auxiliary losses.
    """

    def __init__(self, dim: int = 32, T: int = 79,
                 strides: list = None, levels: list = None):
        """
        Args:
            dim: latent channel dimension (32)
            T: latent temporal length (79)
            strides: temporal downsampling factors per scale (default: [8,4,2,1])
            levels: FSQ levels per scale (default: [3,3,5,5])
        """
        super().__init__()
        self.dim = dim
        self.T = T
        self.strides = strides or [8, 4, 2, 1]
        self.levels = levels or [3, 3, 5, 5]
        self.n_scales = len(self.strides)

        assert len(self.strides) == len(self.levels), \
            f"strides ({len(self.strides)}) and levels ({len(self.levels)}) must match"

        # Per-scale learnable gain (helps balance residual magnitudes)
        self.scale_gains = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.n_scales)
        ])

    def _downsample(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """Average-pool temporal dimension by stride."""
        if stride == 1:
            return x
        # [B, D, T] → [B, D, T//stride]
        return F.avg_pool1d(x, kernel_size=stride, stride=stride,
                            count_include_pad=False)

    def _upsample(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
        """Upsample back to target temporal length."""
        if x.shape[-1] == target_T:
            return x
        return F.interpolate(x, size=target_T, mode='linear', align_corners=False)

    def _fsq(self, x: torch.Tensor, L: int) -> tuple:
        """Apply FSQ quantization at level L.

        Maps [-1, 1] → {0, 1, ..., L-1} → [-1, 1] (dequantized).
        Uses STE for gradient flow.

        Returns: (quantized, symbols)
        """
        step = 2.0 / L
        # Clamp to [-1, 1] range
        x_clamped = x.clamp(-1.0, 1.0)
        # Quantize: map to integer indices
        indices = ((x_clamped + 1.0) / step).floor().clamp(0, L - 1).long()
        # Dequantize: map back to centers
        centers = (indices.float() + 0.5) * step - 1.0
        # STE: forward uses quantized, backward uses smooth
        quantized = x_clamped + (centers - x_clamped).detach()
        return quantized, indices

    def encode(self, latent: torch.Tensor) -> tuple:
        """Multi-scale FSQ encoding.

        Args:
            latent: [B, D, T] in [-1, 1] (post-CDF)
        Returns:
            all_tokens: list of [B, D, T_scale] int tensors
            all_quant: list of [B, D, T_scale] float tensors
        """
        residual = latent.clone()
        all_tokens = []
        all_quant = []

        for i, (stride, L) in enumerate(zip(self.strides, self.levels)):
            gain = self.scale_gains[i]
            # Downsample residual to this scale's resolution
            down = self._downsample(residual, stride)
            # Apply gain and quantize
            scaled = (down * gain).clamp(-1.0, 1.0)
            quantized, tokens = self._fsq(scaled, L)
            # Undo gain
            quantized_unscaled = quantized / gain.clamp(min=1e-6)

            all_tokens.append(tokens)
            all_quant.append(quantized_unscaled)

            # Subtract this scale's contribution from residual
            contribution = self._upsample(quantized_unscaled, self.T)
            residual = residual - contribution

        return all_tokens, all_quant

    def decode(self, all_quant: list) -> torch.Tensor:
        """Reconstruct latent from multi-scale quantized representations.

        Args:
            all_quant: list of [B, D, T_scale] quantized tensors
        Returns:
            reconstructed: [B, D, T]
        """
        recon = torch.zeros(all_quant[0].shape[0], self.dim, self.T,
                            device=all_quant[0].device, dtype=all_quant[0].dtype)
        for q in all_quant:
            recon = recon + self._upsample(q, self.T)
        return recon

    def forward(self, latent: torch.Tensor) -> tuple:
        """Full forward: encode + decode.

        Args:
            latent: [B, D, T] in [-1, 1]
        Returns:
            reconstructed: [B, D, T]
            tokens: list of per-scale token tensors
            quant_loss: quantization error (for monitoring)
        """
        tokens, quants = self.encode(latent)
        recon = self.decode(quants)
        quant_loss = F.mse_loss(recon, latent)
        return recon, tokens, quant_loss

    def token_count(self) -> dict:
        """Report token counts per scale and total bits."""
        info = {}
        total_tokens = 0
        total_bits = 0
        for i, (stride, L) in enumerate(zip(self.strides, self.levels)):
            T_scale = max(1, self.T // stride)
            n_tokens = self.dim * T_scale
            bits_per_token = np.log2(L)
            info[f'scale_{i}'] = {
                'stride': stride,
                'T': T_scale,
                'L': L,
                'tokens': n_tokens,
                'bits': n_tokens * bits_per_token,
            }
            total_tokens += n_tokens
            total_bits += n_tokens * bits_per_token
        info['total_tokens'] = total_tokens
        info['total_bits'] = total_bits
        return info

    def estimated_cr(self, raw_bytes: int = 105000) -> float:
        """Estimate compression ratio from token counts."""
        total_bits = self.token_count()['total_bits']
        compressed_bytes = total_bits / 8
        return raw_bytes / compressed_bytes


# ─── Presets ───────────────────────────────────────────────────────

def make_multiscale_fsq(preset: str = 'balanced') -> MultiScaleFSQ:
    """Create MultiScaleFSQ with a named preset.

    Presets:
        'compact':   4 scales, minimal bits (highest CR)
        'balanced':  4 scales, moderate bits (good quality/CR tradeoff)
        'quality':   4 scales, more bits at fine scales (best quality)
        'flat':      1 scale at stride=1 (equivalent to standard FSQ)
    """
    configs = {
        'compact': {
            'strides': [8, 4, 2, 1],
            'levels':  [2, 2, 3, 3],
            # 32*10*1 + 32*20*1 + 32*40*1.58 + 32*79*1.58 = ~6340 bits = 793 bytes → 132:1
        },
        'balanced': {
            'strides': [8, 4, 2, 1],
            'levels':  [3, 3, 5, 5],
            # 32*10*1.58 + 32*20*1.58 + 32*40*2.32 + 32*79*2.32 = ~9600 bits = 1200 bytes → 87:1
        },
        'quality': {
            'strides': [8, 4, 2, 1],
            'levels':  [5, 5, 8, 8],
            # 32*10*2.32 + 32*20*2.32 + 32*40*3 + 32*79*3 = ~13200 bits = 1650 bytes → 63:1
        },
        'flat': {
            'strides': [1],
            'levels':  [5],
            # Standard FSQ: 32*79*2.32 = 5864 bits = 733 bytes → 143:1
        },
    }
    cfg = configs.get(preset, configs['balanced'])
    return MultiScaleFSQ(dim=32, T=79, **cfg)
