"""
Conditional DCGAN for Synthetic EEG Spike/Seizure Waveform Generation.

Generates synthetic EEG windows conditioned on activity labels
(background, active, seizure) to augment the SNN training set.
Addresses the severe class imbalance problem where seizure events
constitute <1% of the training data.

Usage:
    # Pre-train the GAN
    gan = SpikeGAN(n_channels=21, window_size=2500)
    train_gan(gan, real_data_loader, epochs=50, device='cuda')

    # Generate synthetic seizure windows during SNN training
    synthetic_eeg, synthetic_labels = gan.generate(
        n_samples=32, label=2, device='cuda')

Reference: Conditional DCGANs for pre-ictal EEG generation (2026),
iAAFT augmentation preserving non-stationary properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    """Conditional generator: noise + label → synthetic EEG [21, 2500].

    Architecture: MLP projection → 4-stage ConvTranspose1d upsampling.
    Conditioned on a 3-class label embedding (background/active/seizure).
    """

    def __init__(self, noise_dim=64, label_dim=3, embed_dim=8,
                 n_channels=21, window_size=2500):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_channels = n_channels
        self.window_size = window_size

        self.label_embed = nn.Embedding(label_dim, embed_dim)

        # Project noise + label to spatial feature map
        self.proj = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 256 * 39),
            nn.ReLU(True),
        )

        # Upsample: 39 → 78 → 156 → 312 → 625 → 2500
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),  # 39→78
            nn.BatchNorm1d(128), nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),   # 78→156
            nn.BatchNorm1d(64), nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),    # 156→312
            nn.BatchNorm1d(32), nn.ReLU(True),
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1),    # 312→624
            nn.BatchNorm1d(32), nn.ReLU(True),
            nn.ConvTranspose1d(32, n_channels, 4, stride=4, padding=0),  # 624→2500
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        """
        noise: [B, noise_dim]
        labels: [B] int (0=background, 1=active, 2=seizure)
        returns: [B, 21, 2500] synthetic EEG
        """
        label_emb = self.label_embed(labels)  # [B, embed_dim]
        x = torch.cat([noise, label_emb], dim=1)  # [B, noise_dim + embed_dim]
        x = self.proj(x).reshape(-1, 256, 39)  # [B, 256, 39]
        x = self.upsample(x)  # [B, 21, ~2496]
        # BUG FIX M1: ConvTranspose chain may produce <window_size samples.
        # Pad to exact size instead of silently returning fewer samples.
        if x.shape[-1] < self.window_size:
            x = F.pad(x, (0, self.window_size - x.shape[-1]))
        return x[:, :, :self.window_size]


class Discriminator(nn.Module):
    """Conditional discriminator: EEG + label → real/fake score."""

    def __init__(self, n_channels=21, window_size=2500, label_dim=3):
        super().__init__()
        self.label_embed = nn.Embedding(label_dim, n_channels * 8)

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, stride=4, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 64, 5, stride=4, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 128, 5, stride=4, padding=2),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 256, 3, stride=4, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256 + n_channels * 8, 1)

    def forward(self, x, labels):
        """
        x: [B, 21, 2500]
        labels: [B] int
        returns: [B, 1] logits
        """
        feat = self.conv(x).squeeze(-1)  # [B, 256]
        label_emb = self.label_embed(labels)  # [B, 21*8]
        return self.head(torch.cat([feat, label_emb], dim=1))


class SpikeGAN(nn.Module):
    """Wrapper for conditional GAN training and generation."""

    def __init__(self, n_channels=21, window_size=2500, noise_dim=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_channels = n_channels
        self.window_size = window_size
        self.G = Generator(noise_dim, n_channels=n_channels, window_size=window_size)
        self.D = Discriminator(n_channels=n_channels, window_size=window_size)

    @torch.no_grad()
    def generate(self, n_samples, label, device='cpu'):
        """Generate synthetic EEG windows of a specific class.

        Args:
            n_samples: number of windows to generate
            label: 0=background, 1=active, 2=seizure
            device: torch device

        Returns:
            (eeg [n_samples, 21, 2500], labels [n_samples])
        """
        self.G.eval()
        noise = torch.randn(n_samples, self.noise_dim, device=device)
        labels = torch.full((n_samples,), label, dtype=torch.long, device=device)
        eeg = self.G(noise, labels)
        return eeg, labels


def train_gan(gan, real_loader, epochs=50, device='cpu', lr=2e-4):
    """Pre-train the conditional GAN on real EEG data.

    Args:
        gan: SpikeGAN instance
        real_loader: DataLoader yielding (signal [B, 21, T], labels [B, ...])
        epochs: number of GAN training epochs
        device: torch device
        lr: learning rate for both G and D
    """
    gan = gan.to(device)
    opt_G = torch.optim.Adam(gan.G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(gan.D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        g_losses, d_losses = [], []

        for signal, labels in real_loader:
            B = signal.shape[0]
            signal = signal.to(device)

            # BUG FIX M2: collapse multi-dim labels to [B] (single class per sample).
            # Labels may be [B, 8, T] (spatial groups × time) — need full reduction.
            while labels.dim() > 1:
                labels = labels.amax(dim=-1)
            labels = labels.long().clamp(0, 2).to(device)

            # Ensure signal is [B, 21, 2500]
            if signal.shape[-1] < gan.window_size:
                signal = F.pad(signal, (0, gan.window_size - signal.shape[-1]))
            elif signal.shape[-1] > gan.window_size:
                signal = signal[:, :, :gan.window_size]

            real_labels = torch.ones(B, 1, device=device)
            fake_labels = torch.zeros(B, 1, device=device)

            # --- Train Discriminator ---
            noise = torch.randn(B, gan.noise_dim, device=device)
            fake = gan.G(noise, labels).detach()

            d_real = gan.D(signal, labels)
            d_fake = gan.D(fake, labels)
            d_loss = (F.binary_cross_entropy_with_logits(d_real, real_labels)
                      + F.binary_cross_entropy_with_logits(d_fake, fake_labels)) / 2

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())

            # --- Train Generator ---
            noise = torch.randn(B, gan.noise_dim, device=device)
            fake = gan.G(noise, labels)
            g_pred = gan.D(fake, labels)
            g_loss = F.binary_cross_entropy_with_logits(g_pred, real_labels)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            g_losses.append(g_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"  [GAN] Epoch {epoch+1}/{epochs}  "
                  f"D_loss={np.mean(d_losses):.4f}  G_loss={np.mean(g_losses):.4f}")
