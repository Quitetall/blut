"""Conditional Flow Matching postfilter for EEG reconstruction refinement.

Inspired by FlowDec (Meta, ICLR 2025): a non-adversarial generative
postfilter that refines deterministic codec output in 6 DNN evaluations.

Architecture: simplified NCSN++ backbone for 1D EEG signals.
The two-stage approach: train Vocos decoder first with reconstruction
losses, then add CFM postfilter that refines output quality.

Usage:
    postfilter = CFMPostfilter(dim=128, n_blocks=8)
    refined = postfilter.refine(coarse_reconstruction, n_steps=6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embedding for the diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock1D(nn.Module):
    """1D residual block with time conditioning for flow network."""
    def __init__(self, dim, time_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 7, padding=3)
        self.conv2 = nn.Conv1d(dim, dim, 7, padding=3)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)
        self.time_proj = nn.Linear(time_dim, dim)

    def forward(self, x, t_emb):
        h = F.gelu(self.norm1(self.conv1(x)))
        # Add time conditioning
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = F.gelu(self.norm2(self.conv2(h)))
        return x + h


class FlowVelocityNetwork(nn.Module):
    """Predicts the velocity field v(x_t, t) for the flow ODE.

    Input: noisy signal x_t [B, C, T] + conditioning (coarse reconstruction)
    Output: velocity v [B, C, T]
    """
    def __init__(self, in_ch=21, dim=128, n_blocks=8, time_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        # Input: noisy + conditioning = 2 × in_ch
        self.input_proj = nn.Conv1d(in_ch * 2, dim, 1)
        self.blocks = nn.ModuleList([ResBlock1D(dim, time_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Conv1d(dim, in_ch, 1)
        # Zero-init output for stable start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t, t, conditioning):
        """
        x_t: [B, C, T] noisy signal at time t
        t: [B] diffusion timestep ∈ [0, 1]
        conditioning: [B, C, T] coarse reconstruction from Vocos decoder
        """
        t_emb = self.time_embed(t)
        h = self.input_proj(torch.cat([x_t, conditioning], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


class CFMPostfilter(nn.Module):
    """Conditional Flow Matching postfilter for EEG reconstruction.

    Refines coarse Vocos decoder output using a learned velocity field.
    The ODE: dx/dt = v(x_t, t, conditioning)
    Solved with Euler method in n_steps (default 6).

    Training: regress v_θ against the optimal transport vector field
      v*(x_0, x_1, t) = x_1 - x_0 (straight line interpolant)
    where x_0 = coarse reconstruction, x_1 = ground truth.
    """
    def __init__(self, in_ch=21, dim=128, n_blocks=8):
        super().__init__()
        self.velocity_net = FlowVelocityNetwork(in_ch, dim, n_blocks)

    def training_loss(self, coarse, target):
        """Compute CFM training loss.

        Sample t ~ U[0,1], interpolate x_t = (1-t)*coarse + t*target,
        predict velocity, compare to optimal v* = target - coarse.
        """
        B = coarse.shape[0]
        t = torch.rand(B, device=coarse.device)
        # Linear interpolant
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * coarse + t_expand * target
        # Optimal velocity
        v_star = target - coarse
        # Predicted velocity
        v_pred = self.velocity_net(x_t, t, coarse)
        return F.mse_loss(v_pred, v_star)

    @torch.no_grad()
    def refine(self, coarse, n_steps=6):
        """Refine coarse reconstruction via Euler integration of the flow ODE.

        Args:
            coarse: [B, C, T] coarse Vocos decoder output
            n_steps: number of Euler steps (6 = default, 1 = fast)
        Returns:
            refined: [B, C, T] refined signal
        """
        x = coarse.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.velocity_net(x, t, coarse)
            x = x + dt * v
        return x

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
