"""Channel-agnostic codec <-> .lmq round-trip (montage-carrying neural wire).

Ties the trained channel-agnostic JointCodec to the LMQC container: encode an
L3 window + its montage to a self-describing .lmq, and decode that .lmq back to
an N-channel reconstruction off-device — WITHOUT the decoder needing any
external montage info (the coords travel in the file). This is the capability
that makes channel-agnostic .lmq files round-trip off-device (spec C5).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from lamquant_neural.wire import write_ca_lmq, read_ca_lmq


def encode_to_lmq(codec, x_l3, coords, channels: Optional[Sequence[str]],
                  path: str, *, sample_rate: int = 250,
                  window_samples: int = 2500) -> str:
    """Encode one window to an LMQC .lmq carrying the montage.

    x_l3:  [N,313] or [1,N,313]   L3 approximation (one window).
    coords: [N,3] or [1,N,3]      electrode positions (meters; NaN = unknown).
    channels: per-channel names (or None).
    """
    codec.train(False)
    if x_l3.dim() == 2:
        x_l3 = x_l3.unsqueeze(0)
    c = coords if coords.dim() == 3 else coords.unsqueeze(0)
    with torch.no_grad():
        latent = codec.encoder.encode(x_l3, quantize=True, coords=c)   # [1,32,79]
    return write_ca_lmq(path, coords=c[0], channels=channels, latent=latent[0],
                        sample_rate=sample_rate, window_samples=window_samples)


def decode_from_lmq(codec, path: str, device: str = "cpu"):
    """Decode an LMQC .lmq back to [1,N,window_samples] using ONLY the file's
    own montage. Returns (recon, meta_dict)."""
    d = read_ca_lmq(path)
    if d["latent"] is None:
        raise NotImplementedError(
            f"payload_kind={d['payload_kind']} (non-fp16) decode not wired yet")
    latent = torch.from_numpy(d["latent"]).unsqueeze(0).to(device)      # [1,32,79]
    coords = (torch.from_numpy(d["coords"]).unsqueeze(0).to(device)
              if d["coords"] is not None else None)
    codec.train(False)
    with torch.no_grad():
        recon = codec.decoder(latent, coords=coords, ch_mask=None)
    return recon, d
