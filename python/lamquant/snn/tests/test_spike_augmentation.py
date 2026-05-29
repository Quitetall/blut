"""Unit tests for ai_models/snn/spike_augmentation.py — Phase 2.

Conditional DCGAN for synthetic EEG generation. Memory says this is
legacy ("No synthetic data allowed" per feedback_mamba_snn_promoted),
but code remains in tree. Cover Generator, Discriminator, SpikeGAN,
and the train_gan loop with a synthetic real-data loader (no real EEG).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from spike_augmentation import (
    Discriminator,
    Generator,
    SpikeGAN,
    train_gan,
)

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class TestGenerator:
    def test_default_construction(self):
        g = Generator()
        assert g.noise_dim == 64
        assert g.n_channels == 21
        assert g.window_size == 2500

    def test_forward_shape(self):
        g = Generator()
        noise = torch.randn(2, 64)
        labels = torch.tensor([0, 2])
        out = g(noise, labels)
        assert out.shape == (2, 21, 2500)

    def test_pad_when_shorter(self):
        g = Generator(window_size=2500)
        noise = torch.randn(1, 64)
        out = g(noise, torch.tensor([0]))
        assert out.shape[-1] == 2500


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class TestDiscriminator:
    def test_forward_shape(self):
        d = Discriminator()
        x = torch.randn(2, 21, 2500)
        labels = torch.tensor([0, 1])
        out = d(x, labels)
        assert out.shape == (2, 1)

    def test_construction_custom_channels(self):
        d = Discriminator(n_channels=8, window_size=1000, label_dim=2)
        x = torch.randn(1, 8, 2500)  # forward needs the matching channels
        # Actually conv first layer is 8 channels in this case
        out = d(x, torch.tensor([0]))
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# SpikeGAN
# ---------------------------------------------------------------------------
class TestSpikeGAN:
    def test_default_construction(self):
        gan = SpikeGAN()
        assert isinstance(gan.G, Generator)
        assert isinstance(gan.D, Discriminator)

    def test_generate(self):
        gan = SpikeGAN()
        eeg, labels = gan.generate(n_samples=4, label=2)
        assert eeg.shape == (4, 21, 2500)
        assert labels.shape == (4,)
        assert (labels == 2).all()


# ---------------------------------------------------------------------------
# train_gan
# ---------------------------------------------------------------------------
class _FakeLoader:
    """Iterable that yields a few (signal, labels) batches."""
    def __init__(self, n_batches=2, B=2, T=2500, C=21):
        self.n = n_batches
        self.B = B
        self.T = T
        self.C = C

    def __iter__(self):
        for _ in range(self.n):
            sig = torch.randn(self.B, self.C, self.T)
            labels = torch.randint(0, 3, (self.B,))
            yield sig, labels


class TestTrainGan:
    def test_runs_one_epoch(self):
        gan = SpikeGAN()
        loader = _FakeLoader(n_batches=2)
        train_gan(gan, loader, epochs=1, device="cpu", lr=1e-4)

    def test_handles_multi_dim_labels(self):
        gan = SpikeGAN()

        class _MultiLabelLoader:
            def __iter__(self):
                for _ in range(1):
                    sig = torch.randn(2, 21, 2500)
                    # Labels [B, 8, T] simulating spatial groups
                    labels = torch.randint(0, 3, (2, 8, 313))
                    yield sig, labels

        train_gan(gan, _MultiLabelLoader(), epochs=1, device="cpu")

    def test_pads_short_signal(self):
        gan = SpikeGAN(window_size=2500)

        class _ShortLoader:
            def __iter__(self):
                for _ in range(1):
                    sig = torch.randn(2, 21, 1000)  # short
                    labels = torch.randint(0, 3, (2,))
                    yield sig, labels

        train_gan(gan, _ShortLoader(), epochs=1, device="cpu")

    def test_trims_long_signal(self):
        gan = SpikeGAN(window_size=2500)

        class _LongLoader:
            def __iter__(self):
                for _ in range(1):
                    sig = torch.randn(2, 21, 5000)
                    labels = torch.randint(0, 3, (2,))
                    yield sig, labels

        train_gan(gan, _LongLoader(), epochs=1, device="cpu")

    def test_progress_print_at_milestone(self, capsys):
        gan = SpikeGAN()
        loader = _FakeLoader(n_batches=1)
        # epochs=10 → prints at (epoch+1) % 10 == 0 i.e. epoch=9
        train_gan(gan, loader, epochs=10, device="cpu")
        out = capsys.readouterr().out
        # The 10th (epoch=9 in zero-indexed) prints; check presence of [GAN] tag
        assert "[GAN]" in out
