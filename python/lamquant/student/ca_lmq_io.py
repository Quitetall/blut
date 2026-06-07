"""Channel-agnostic codec <-> .lmq round-trip (montage-carrying neural wire).

The wire format (LMQC container) is CANONICAL in Rust — `lamquant_core.write_ca_lmq`
/ `read_ca_lmq` (lamquant-lossless/src/lmqc.rs). This module is the thin Python
marshalling layer: torch latent <-> fp16 payload bytes, coords <-> flat f32 list,
and the codec encode/decode calls. No wire format lives in Python (deprecated).

encode an L3 window + its montage to a self-describing .lmq, and decode that .lmq
back to an N-channel reconstruction off-device WITHOUT the decoder needing any
external montage (the coords travel in the file). This is the capability that
makes channel-agnostic .lmq files round-trip off-device (spec C5).
"""
from __future__ import annotations

import contextlib
from typing import Optional, Sequence

import numpy as np
import torch

import lamquant_core as _lc

PAYLOAD_FP16_LATENT = 0


@contextlib.contextmanager
def _eval_mode(module):
    """Set eval for a deterministic forward, restoring the caller's mode after
    (MiMo review: train(False) is not self-restoring like torch.no_grad)."""
    was_training = module.training
    module.train(False)
    try:
        yield
    finally:
        module.train(was_training)


def encode_to_lmq(codec, x_l3, coords, channels: Optional[Sequence[str]],
                  path: str, *, sample_rate: int = 250,
                  window_samples: int = 2500) -> str:
    """Encode one window to an LMQC .lmq carrying the montage (via Rust).

    x_l3:  [N,313] or [1,N,313]   L3 approximation (one window).
    coords: [N,3] or [1,N,3]      electrode positions (meters; NaN = unknown).
    channels: per-channel names (or None).
    """
    if x_l3.dim() not in (2, 3):
        raise ValueError(f"x_l3 must be [N,313] or [1,N,313], got {tuple(x_l3.shape)}")
    if x_l3.dim() == 2:
        x_l3 = x_l3.unsqueeze(0)
    c = coords if coords.dim() == 3 else coords.unsqueeze(0)
    with _eval_mode(codec), torch.no_grad():
        latent = codec.encoder.encode(x_l3, quantize=True, coords=c)   # [1,32,79]
    lat = latent[0].detach().cpu().numpy()                             # [32,79]
    payload = np.ascontiguousarray(lat, dtype=np.float16).tobytes()
    coords_flat = [float(v) for v in np.ascontiguousarray(
        c[0].detach().cpu().numpy(), dtype=np.float32).reshape(-1)]
    n = c.shape[1]
    chans = list(channels) if channels is not None else None
    _lc.write_ca_lmq(path, n, int(lat.shape[0]), int(lat.shape[1]),
                     int(sample_rate), int(window_samples),
                     PAYLOAD_FP16_LATENT, payload, coords_flat, chans)
    return path


def decode_from_lmq(codec, path: str, device: str = "cpu"):
    """Decode an LMQC .lmq back to [1,N,window_samples] using ONLY the file's
    own montage (Rust reader). Returns (recon, meta_dict)."""
    d = _lc.read_ca_lmq(path)                                          # Rust → dict
    if d["payload_kind"] != PAYLOAD_FP16_LATENT:
        raise NotImplementedError(
            f"payload_kind={d['payload_kind']} (non-fp16) decode not wired yet")
    latent = (torch.from_numpy(
        np.frombuffer(bytes(d["payload"]), dtype="<f2")
          .reshape(d["latent_c"], d["latent_t"]).astype(np.float32))
        .unsqueeze(0).to(device))                                     # [1,32,79]
    coords = None
    if d["coords"] is not None:
        coords = (torch.tensor(d["coords"], dtype=torch.float32)
                  .reshape(d["n_channels"], 3).unsqueeze(0).to(device))
    with _eval_mode(codec), torch.no_grad():
        recon = codec.decoder(latent, coords=coords, ch_mask=None)
    meta = {k: d[k] for k in ("n_channels", "latent_c", "latent_t", "sample_rate",
                              "window_samples", "payload_kind", "channels")}
    meta["coords"] = (np.array(d["coords"], np.float32).reshape(d["n_channels"], 3)
                      if d["coords"] is not None else None)
    return recon, meta
