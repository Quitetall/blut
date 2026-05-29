"""
LamQuant Teacher Model Architectures
=====================================
FP32 oracle teacher models for distillation training.

Contains:
  - FocalModulationBlock: residual conv block with group norm
  - MobileNetV5Focal: original Gen 6 encoder
  - DecoderBlock: original Gen 6 decoder
  - FP32OracleAutoEncoder: Gen 6 encoder+decoder composite
  - ChannelAwareEncoding: spatial attention over EEG channels
  - BottleneckAttention: multi-head self-attention before latent projection
  - StridedFocalBlock: focal block with optional stride-2 downsampling
  - UpsampleFocalBlock: focal block with stride-2 upsampling (transposed conv)
  - L3TeacherEncoder: Gen 7.5 FP32 encoder on L3 approximation
  - L3TeacherDecoder: Gen 7.5 FP32 decoder
  - L3Teacher: Gen 7.5 composite encoder+decoder

Extracted from ai_models/oracle/train_teacher.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Gen 6 Oracle Architecture
# ============================================================

class FocalModulationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        # Clinical Hardening: 4 groups for better spatial feature separation
        self.norm = nn.GroupNorm(4 if out_ch % 4 == 0 else 1, out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.norm(self.conv(x)))
        return out + identity

class MobileNetV5Focal(nn.Module):
    def __init__(self, in_ch=21, latent_dim=32):
        super().__init__()
        # ENCODER: Maps 21 natively mapping continuous chunks
        self.focal1 = FocalModulationBlock(in_ch, 64, kernel_size=7)
        self.focal2 = FocalModulationBlock(64, 128, kernel_size=5)
        self.focal3 = FocalModulationBlock(128, 256, kernel_size=3)
        self.bottleneck = nn.Conv1d(256, latent_dim, 1) # L32-Elite capacity

    def forward(self, x):
        x = self.focal1(x)
        x = self.focal2(x)
        x = self.focal3(x)
        return self.bottleneck(x)

class DecoderBlock(nn.Module):
    def __init__(self, latent_dim=32, out_ch=21):
        super().__init__()
        # Symmetric expansion tracing identical dimensions ensuring Gradient mapping natively
        self.expand1 = FocalModulationBlock(latent_dim, 256, kernel_size=3)
        self.expand2 = FocalModulationBlock(256, 128, kernel_size=5)
        self.expand3 = FocalModulationBlock(128, 64, kernel_size=7)
        self.output = nn.Conv1d(64, out_ch, 1)
    def forward(self, x):
        x = self.expand1(x)
        x = self.expand2(x)
        x = self.expand3(x)
        return self.output(x)

class FP32OracleAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MobileNetV5Focal()
        self.decoder = DecoderBlock()
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================
# L3-Native Teacher (Gen 7.5)
# ============================================================
# Operates on L3 approximation [21, 313] -> latent [32, 79] -- same
# input and latent shape as the student TNN.  FP32 with 5.4M params
# (18x student capacity).  Produces near-perfect L3 reconstruction
# as a distillation target for ternary student hardening.

class ChannelAwareEncoding(nn.Module):
    """Spatial attention over EEG channels exploiting electrode topology.

    Learns a [21, 21] attention matrix that models volume conduction and
    inter-channel correlations. Applied before the main encoder blocks.
    Produces a channel-mixed representation where each output channel is
    a learned weighted combination of all input channels.

    Cost: 21x21 + 21x21 = 882 params (negligible).
    """
    def __init__(self, n_channels=21):
        super().__init__()
        # Learned spatial affinity matrix (initialized near-identity)
        self.spatial_attn = nn.Parameter(torch.eye(n_channels) + 0.01 * torch.randn(n_channels, n_channels))
        self.norm = nn.LayerNorm(n_channels)
        # Gate starts at 0 -- channel mixing initially disabled (pure passthrough)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, T] -- apply spatial attention across channels
        attn = torch.softmax(self.spatial_attn, dim=-1)  # [C, C]
        x_mixed = torch.einsum('ij,bjt->bit', attn, x)   # [B, C, T]
        # Gated residual: gate=0 -> pure passthrough, no disruption
        return x + self.gate * (x_mixed - x)


class BottleneckAttention(nn.Module):
    """Multi-head self-attention before latent projection.

    Operates on [B, W, T] features, attending across the temporal dimension
    within each channel group. Learns which temporal positions and feature
    combinations matter most for the 32-channel latent compression.

    Uses grouped attention (n_heads groups of W/n_heads channels) to keep
    the cost manageable at width=2048.
    """
    def __init__(self, d_model, n_heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # Zero-init output projection: attn starts as identity (x + 0 = x).
        # No LayerNorm -- it disrupts the zero-init passthrough.
        # The attention output is the only new contribution; residual is exact.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x):
        # x: [B, W, T] -- transpose to [B, T, W] for attention over temporal dim
        x_t = x.permute(0, 2, 1)  # [B, T, W]
        attn_out, _ = self.attn(x_t, x_t, x_t)  # self-attention over T
        x_t = x_t + attn_out  # pure residual, no norm
        return x_t.permute(0, 2, 1)  # [B, W, T]


class StridedFocalBlock(nn.Module):
    """FocalModulation block with optional stride-2 downsampling."""
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(8 if out_ch % 8 == 0 else 4, out_ch)
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return F.relu(self.norm(self.conv(x))) + self.shortcut(x)


class UpsampleFocalBlock(nn.Module):
    """FocalModulation block with stride-2 upsampling (transposed conv)."""
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=2):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size,
                                        stride=stride, padding=kernel_size // 2,
                                        output_padding=stride - 1)
        self.norm = nn.GroupNorm(8 if out_ch % 8 == 0 else 4, out_ch)
        self.shortcut = nn.ConvTranspose1d(in_ch, out_ch, 1,
                                            stride=stride, output_padding=stride - 1)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x))) + self.shortcut(x)


class L3TeacherEncoder(nn.Module):
    """FP32 encoder on L3 approximation.  [B, 21, 313] -> [B, 32, 79].

    Configurable depth via strides list. Default [1, 2, 2] matches the
    student's temporal reduction. Extra stride-1 blocks add depth
    (more nonlinear feature extraction) without changing the latent shape.

    Optional channel-aware encoding (spatial attention over electrodes)
    and bottleneck attention (multi-head self-attention before compression).
    """
    def __init__(self, in_ch=21, latent_dim=32, width=512, strides=None,
                 channel_attn=False, bottleneck_attn=False, attn_heads=8):
        super().__init__()
        if strides is None:
            strides = [1, 2, 2]
        self.channel_encoding = ChannelAwareEncoding(in_ch) if channel_attn else None
        self.premix = nn.Conv1d(in_ch, in_ch, 1)
        self.blocks = nn.ModuleList()
        for i, s in enumerate(strides):
            in_c = in_ch if i == 0 else width
            k = 7 if i == 0 else 5
            self.blocks.append(StridedFocalBlock(in_c, width, kernel_size=k, stride=s))
        self.bn_attn = BottleneckAttention(width, n_heads=attn_heads) if bottleneck_attn else None
        self.bottleneck = nn.Conv1d(width, latent_dim, 1)

    def encode_stage1(self, x):
        """Stage 1: premix -> first block (resolution change). For SMoDi."""
        if self.channel_encoding is not None:
            x = self.channel_encoding(x)
        x = self.premix(x)
        x = self.blocks[0](x)
        return x

    def encode_stage2(self, x):
        """Stage 2: middle blocks (constant resolution). For SMoDi."""
        for block in self.blocks[1:-1]:
            x = block(x)
        return x

    def encode_stage3(self, x):
        """Stage 3: final block -> attention -> bottleneck. For SMoDi."""
        x = self.blocks[-1](x)
        if self.bn_attn is not None:
            x = self.bn_attn(x)
        return self.bottleneck(x)

    def forward(self, x):
        x = self.encode_stage1(x)
        x = self.encode_stage2(x)
        return self.encode_stage3(x)


class L3TeacherDecoder(nn.Module):
    """FP32 decoder.  [B, 32, 79] -> [B, 21, 313].

    Symmetric to encoder: reverses the stride pattern.
    """
    def __init__(self, latent_dim=32, out_ch=21, width=512, strides=None):
        super().__init__()
        if strides is None:
            strides = [1, 2, 2]
        self.expand = nn.Conv1d(latent_dim, width, 1)
        # Reverse the strides for decoder
        rev_strides = list(reversed(strides))
        self.blocks = nn.ModuleList()
        for i, s in enumerate(rev_strides):
            k = 7 if i == len(rev_strides) - 1 else 5
            if s > 1:
                self.blocks.append(UpsampleFocalBlock(width, width, kernel_size=k, stride=s))
            else:
                self.blocks.append(StridedFocalBlock(width, width, kernel_size=k, stride=1))
        self.output = nn.Conv1d(width, out_ch, 1)

    def forward(self, x, target_len=313):
        x = self.expand(x)
        for block in self.blocks:
            x = block(x)
        x = self.output(x)
        return x[:, :, :target_len]


class L3Teacher(nn.Module):
    """L3-native FP32 teacher for Gen 7.5 distillation.

    Same input [21, 313] and latent [32, 79] as the student TNN.
    Configurable width, depth, and attention mechanisms.

    Default (width=512, strides=[1,2,2]): 8.3M params -- 18x student.
    Maximum (width=2048, strides=[1,1,1,2,2], attention): ~220M params.
    """
    def __init__(self, width=512, strides=None, channel_attn=False,
                 bottleneck_attn=False, attn_heads=8):
        super().__init__()
        self.encoder = L3TeacherEncoder(
            width=width, strides=strides, channel_attn=channel_attn,
            bottleneck_attn=bottleneck_attn, attn_heads=attn_heads)
        self.decoder = L3TeacherDecoder(width=width, strides=strides)

    def forward(self, x):
        lat = self.encoder(x)
        return self.decoder(lat, target_len=x.shape[2])

    def encode(self, x):
        return self.encoder(x)
