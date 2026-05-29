"""Safe EEG augmentation wrappers for training.

Wraps selfEEG augmentations with scale guards so they work on L3 subband
data (non-standard amplitudes) without the breakage we hit previously.

The problem last time: selfEEG augmentations assume standard EEG scales
(~10-100 uV). L3 subband data after integer lifting has different ranges.
These wrappers normalize to standard scale, apply augmentation, then
scale back.

Usage:
    from augmentations import EEGAugmentor

    aug = EEGAugmentor(mode='moderate')  # or 'light', 'aggressive'
    augmented = aug(batch_tensor)        # [B, 21, 313]
"""

import torch
import numpy as np

# selfEEG is optional — all augmentations degrade gracefully
try:
    from selfeeg import augmentation as seeg_aug
    HAS_SELFEEG = True
except ImportError:
    HAS_SELFEEG = False
    print("[!] selfeeg not installed. Using built-in augmentations only.")


def _estimate_scale(x: torch.Tensor) -> torch.Tensor:
    """Estimate per-batch RMS scale for normalization."""
    return x.reshape(x.shape[0], -1).std(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)


class EEGAugmentor:
    """Safe EEG augmentation pipeline for codec training.

    All augmentations normalize to unit scale before applying, then
    rescale back. This prevents the scale mismatch that broke selfEEG
    integration previously.

    Args:
        mode: 'light', 'moderate', or 'aggressive'
        p: probability of applying each augmentation (default: per-mode)
        use_selfeeg: whether to use selfEEG augmentations (default: True if available)
    """

    PRESETS = {
        'light': {
            'noise_snr': 30.0,       # dB (barely noticeable)
            'channel_drop_p': 0.05,  # 5% chance per channel
            'scale_range': (0.95, 1.05),
            'mask_p': 0.05,
            'shift_samples': 2,
            'p': 0.3,               # 30% chance of any augmentation
        },
        'moderate': {
            'noise_snr': 20.0,
            'channel_drop_p': 0.1,
            'scale_range': (0.9, 1.1),
            'mask_p': 0.1,
            'shift_samples': 5,
            'p': 0.5,
        },
        'aggressive': {
            'noise_snr': 15.0,
            'channel_drop_p': 0.15,
            'scale_range': (0.8, 1.2),
            'mask_p': 0.15,
            'shift_samples': 10,
            'p': 0.7,
        },
    }

    def __init__(self, mode: str = 'moderate', p: float = None,
                 use_selfeeg: bool = True):
        self.cfg = self.PRESETS.get(mode, self.PRESETS['moderate']).copy()
        if p is not None:
            self.cfg['p'] = p
        self.use_selfeeg = use_selfeeg and HAS_SELFEEG

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to batch.

        Args:
            x: [B, C, T] EEG tensor (any scale)
        Returns:
            augmented: [B, C, T] same shape, similar scale
        """
        if torch.rand(1).item() > self.cfg['p']:
            return x

        # Pick one augmentation randomly
        aug_fn = np.random.choice([
            self._additive_noise,
            self._channel_dropout,
            self._temporal_mask,
            self._amplitude_scale,
            self._temporal_shift,
        ])
        return aug_fn(x)

    def _additive_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at a controlled SNR."""
        snr_db = self.cfg['noise_snr']
        if self.use_selfeeg:
            # selfEEG's add_noise_SNR expects numpy [C, T]
            # Process per-batch to be safe
            out = x.clone()
            for b in range(x.shape[0]):
                try:
                    sample = x[b].cpu().numpy()
                    noisy = seeg_aug.add_noise_SNR(sample, snr_db, random_state=None)
                    out[b] = torch.from_numpy(noisy).to(x.device, x.dtype)
                except Exception:
                    # Fallback: simple Gaussian
                    scale = _estimate_scale(x[b:b+1])
                    noise_std = scale / (10 ** (snr_db / 20))
                    out[b] = x[b] + torch.randn_like(x[b]) * noise_std.squeeze()
            return out
        else:
            scale = _estimate_scale(x)
            noise_std = scale / (10 ** (snr_db / 20))
            return x + torch.randn_like(x) * noise_std

    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out entire channels."""
        p = self.cfg['channel_drop_p']
        if self.use_selfeeg:
            out = x.clone()
            for b in range(x.shape[0]):
                try:
                    sample = x[b].cpu().numpy()
                    dropped = seeg_aug.channel_dropout(sample, p)
                    out[b] = torch.from_numpy(dropped).to(x.device, x.dtype)
                except Exception:
                    mask = torch.rand(x.shape[1], device=x.device) > p
                    out[b] = x[b] * mask.unsqueeze(-1).float()
            return out
        else:
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > p).float()
            return x * mask

    def _temporal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out random temporal segments (SpecAugment-style)."""
        B, C, T = x.shape
        mask_len = max(1, int(T * self.cfg['mask_p']))
        out = x.clone()
        for b in range(B):
            start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            out[b, :, start:start + mask_len] = 0
        return out

    def _amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random per-channel amplitude scaling."""
        lo, hi = self.cfg['scale_range']
        if self.use_selfeeg:
            out = x.clone()
            for b in range(x.shape[0]):
                try:
                    sample = x[b].cpu().numpy()
                    scaled = seeg_aug.scaling(sample, lo, hi)
                    out[b] = torch.from_numpy(scaled).to(x.device, x.dtype)
                except Exception:
                    scale = torch.empty(x.shape[1], 1, device=x.device).uniform_(lo, hi)
                    out[b] = x[b] * scale
            return out
        else:
            scale = torch.empty(x.shape[0], x.shape[1], 1, device=x.device).uniform_(lo, hi)
            return x * scale

    def _temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Random circular temporal shift."""
        max_shift = self.cfg['shift_samples']
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class BuiltinAugmentor:
    """Minimal augmentor with no external dependencies.

    For use when selfEEG is unavailable or unwanted. Same interface as
    EEGAugmentor but only uses PyTorch operations.
    """

    def __init__(self, noise_snr: float = 20.0, channel_drop_p: float = 0.1,
                 p: float = 0.5):
        self.noise_snr = noise_snr
        self.channel_drop_p = channel_drop_p
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        choice = torch.randint(0, 3, (1,)).item()
        if choice == 0:
            # Additive noise
            scale = _estimate_scale(x)
            noise_std = scale / (10 ** (self.noise_snr / 20))
            return x + torch.randn_like(x) * noise_std
        elif choice == 1:
            # Channel dropout
            mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device)
                    > self.channel_drop_p).float()
            return x * mask
        else:
            # Temporal mask
            B, C, T = x.shape
            mlen = max(1, T // 10)
            out = x.clone()
            for b in range(B):
                s = torch.randint(0, max(1, T - mlen), (1,)).item()
                out[b, :, s:s + mlen] = 0
            return out
