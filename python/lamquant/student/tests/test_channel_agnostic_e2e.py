"""CA-4 end-to-end smoke — channel_agnostic JointCodec (encoder+decoder wired).

Verifies the full encode→latent→decode path is channel-count-agnostic: any N
in, latent is N-invariant [B,32,79], reconstruction is [B,N,2500]; padded
channels are mask-zeroed and do not leak into real channels; and the legacy
N=21 path is byte-for-shape unchanged.
"""
import pytest
import torch

from lamquant.student.joint_codec import build_default_joint
from lamquant.common.metrics import (
    pearson_r_torch, prd_torch,
    masked_pearson_r_torch, masked_pearson_r_batch, masked_prd_torch,
)
from lamquant.student.training_utils import pearson_r_batch
from lamquant_neural.data import VariableNAdapter, variable_n_collate

TIER = 3                      # smallest iSTFT (fullband) tier — cheap on CPU
torch.manual_seed(0)


def _coords(B, N):
    return torch.randn(B, N, 3) * 0.05


def _build_ca():
    # tiny encoder width keeps the CPU smoke fast; tier-3 decoder is iSTFT
    return build_default_joint(vocos_tier=TIER, encoder_width=32,
                               channel_agnostic=True).train(False)


def test_legacy_path_unchanged():
    codec = build_default_joint(vocos_tier=TIER, encoder_width=32).train(False)
    out = codec(torch.randn(2, 21, 313), quantize=False)
    assert out.shape == (2, 21, 2500) and torch.isfinite(out).all()


def test_ca_default_coords_N21():
    codec = _build_ca()
    out = codec(torch.randn(2, 21, 313), quantize=False)   # coords default → 10-20
    assert out.shape == (2, 21, 2500) and torch.isfinite(out).all()


def test_ca_variable_N_shapes_and_latent_invariance():
    codec = _build_ca()
    for N in (8, 21, 64):
        x = torch.randn(1, N, 313)
        c = _coords(1, N)
        lat = codec.encoder.encode(x, quantize=False, coords=c)
        assert lat.shape == (1, 32, 79), f"latent not N-invariant at N={N}"
        out = codec(x, quantize=False, coords=c)
        assert out.shape == (1, N, 2500) and torch.isfinite(out).all()


def test_ca_padded_batch_masks_and_no_leak():
    codec = _build_ca()
    x8 = torch.randn(2, 8, 313)
    c8 = _coords(2, 8)
    out_a = codec(x8, quantize=False, coords=c8)            # [2,8,2500]
    # pad to N=12 with junk + mask
    x12 = torch.cat([x8, torch.randn(2, 4, 313) * 9], dim=1)
    c12 = torch.cat([c8, _coords(2, 4)], dim=1)
    m12 = torch.tensor([[True] * 8 + [False] * 4] * 2)
    out_b = codec(x12, quantize=False, coords=c12, ch_mask=m12)
    assert out_b.shape == (2, 12, 2500)
    assert (out_b[:, 8:] == 0).all(), "padded channels must be zeroed"
    # real-channel reconstruction must match the unpadded run (no leak via pool)
    torch.testing.assert_close(out_a, out_b[:, :8], atol=1e-4, rtol=1e-3)


def test_ca_requires_coords_for_non21():
    codec = _build_ca()
    with pytest.raises(ValueError):
        codec.encoder.encode(torch.randn(1, 8, 313), quantize=False)


def test_ca_rejects_direct_tier():
    with pytest.raises(ValueError):
        build_default_joint(vocos_tier=1, channel_agnostic=True)   # tier 1 = direct


def test_ca_decoder_legacy_isolation_build():
    """CA encoder + LEGACY decoder head (warm-start parity isolation, ADR-0036).
    The decoder ignores coords (channel_agnostic=False) and emits fixed 21ch."""
    codec = build_default_joint(vocos_tier=TIER, encoder_width=32,
                                channel_agnostic=True, ca_decoder=False).train(False)
    assert codec.encoder.channel_agnostic is True
    assert codec.decoder.channel_agnostic is False
    out = codec(torch.randn(2, 21, 313), quantize=False)   # coords default → canonical-21
    assert out.shape == (2, 21, 2500) and torch.isfinite(out).all()


def test_ca_decoder_defaults_to_encoder_flag():
    """ca_decoder=None (default) follows channel_agnostic -> both CA."""
    codec = build_default_joint(vocos_tier=TIER, encoder_width=32,
                                channel_agnostic=True).train(False)
    assert codec.encoder.channel_agnostic and codec.decoder.channel_agnostic


# ---------------- masked metric anchors (no-op-default proof) ----------------

def test_masked_metrics_equal_unmasked_when_all_real():
    """ch_mask=None / all-True must equal the existing metric bit-for-bit —
    this is the guarantee that the active N=21 run (#236) is untouched."""
    pred = torch.randn(4, 21, 200)
    tgt = torch.randn(4, 21, 200)
    ones = torch.ones(4, 21, dtype=torch.bool)
    # None path == existing functions
    torch.testing.assert_close(masked_pearson_r_torch(pred, tgt),
                               pearson_r_torch(pred, tgt), atol=0, rtol=0)
    torch.testing.assert_close(masked_prd_torch(tgt, pred),
                               prd_torch(tgt, pred), atol=0, rtol=0)
    assert abs(masked_pearson_r_batch(pred, tgt) - pearson_r_batch(pred, tgt)) < 1e-6
    # all-True mask == None path
    torch.testing.assert_close(masked_pearson_r_torch(pred, tgt, ones),
                               masked_pearson_r_torch(pred, tgt), atol=1e-6, rtol=1e-5)


def test_masked_metric_excludes_padded_channels():
    """A flat-zero padded channel pollutes the unmasked metric; masking fixes it."""
    real_pred = torch.randn(2, 8, 200)
    real_tgt = torch.randn(2, 8, 200)
    # pad with 4 zero channels (what the CA head emits for padding)
    pad = torch.zeros(2, 4, 200)
    pred = torch.cat([real_pred, pad], dim=1)
    tgt = torch.cat([real_tgt, pad], dim=1)
    mask = torch.tensor([[True] * 8 + [False] * 4] * 2)
    masked_r = masked_pearson_r_torch(pred, tgt, mask)
    real_only_r = masked_pearson_r_torch(real_pred, real_tgt)   # ground truth
    torch.testing.assert_close(masked_r, real_only_r, atol=1e-6, rtol=1e-5)
    # PRD likewise tracks real-only
    torch.testing.assert_close(masked_prd_torch(tgt, pred, mask),
                               masked_prd_torch(real_tgt, real_pred), atol=1e-4, rtol=1e-4)


# -------- composition: adapter -> collate -> CA codec -> masked loss --------

def test_composition_adapter_codec_masked_loss():
    """The whole variable-N path on CPU: VariableNAdapter subsets a window,
    collate pads + masks, the channel_agnostic codec consumes coords/ch_mask,
    and the masked metric scores ONLY real channels — identical to running the
    real channels unpadded. Proves the MASK path (no pad leak into real
    channels). Coords-routing is proven separately in
    test_composition_coords_routed_per_channel (this test is coords-invariant
    at init — gate=0/FiLM-identity — so it cannot see coords misrouting)."""
    codec = _build_ca()
    ad = VariableNAdapter(n_range=(8, 21))
    # two windows, fixed subsets so the test is deterministic
    l3 = torch.randn(21, 313)
    fb = torch.randn(21, 2500)
    s1 = ad.apply(l3, fb, idx=torch.arange(8))        # N=8
    s2 = ad.apply(l3, fb, idx=torch.arange(12))       # N=12
    batch = variable_n_collate([s1, s2])

    out = codec(batch["l3"], quantize=False,
                coords=batch["coords"], ch_mask=batch["ch_mask"])
    assert out.shape == (2, 12, 2500) and torch.isfinite(out).all()
    assert (out[0, 8:] == 0).all()                    # padded channels zeroed

    # masked metric is finite + scores only real channels
    mr = masked_pearson_r_torch(out, batch["fullband"], batch["ch_mask"])
    assert torch.isfinite(mr)

    # sample 0's masked score must equal running its 8 real channels unpadded
    # through the SAME codec (proves coords/mask wired correctly, no pad leak)
    out_real = codec(s1["l3"].unsqueeze(0), quantize=False,
                     coords=s1["coords"].unsqueeze(0))
    r_padded = masked_pearson_r_torch(out[:1], batch["fullband"][:1],
                                      batch["ch_mask"][:1])
    r_real = masked_pearson_r_torch(out_real, s1["fullband"].unsqueeze(0))
    torch.testing.assert_close(r_padded, r_real, atol=1e-4, rtol=1e-3)


def test_composition_coords_routed_per_channel():
    """coords[i] must condition output channel i through the FULL codec.

    Make the decoder head coords-sensitive (perturb its FiLM pos_mlp out of
    identity init), then permute ONLY channels 0,1's coords with l3 fixed. If
    coords are routed per-channel correctly: output channels 0,1 change and the
    untouched channels are bit-identical. A transposed/misaligned coords wiring
    in JointCodec/decoder.forward fails this. (Encoder gate=0 at init → latent
    is coords-invariant, so this isolates the decoder-head coords path.)"""
    codec = _build_ca()
    with torch.no_grad():
        h = codec.decoder.head                       # PositionConditionedHead
        h.pos_mlp[-1].weight.normal_(std=0.5)
        h.pos_mlp[-1].bias.normal_(std=0.5)
    l3 = torch.randn(1, 8, 313)
    coords = torch.randn(1, 8, 3) * 0.05
    mask = torch.ones(1, 8, dtype=torch.bool)
    out_a = codec(l3, quantize=False, coords=coords, ch_mask=mask)
    coords_p = coords.clone()
    coords_p[0, [0, 1]] = coords[0, [1, 0]]          # swap ch 0 and 1 positions
    out_b = codec(l3, quantize=False, coords=coords_p, ch_mask=mask)
    assert (out_a[0, 0] - out_b[0, 0]).abs().max() > 1e-5, "coords[0] must affect ch 0"
    assert (out_a[0, 1] - out_b[0, 1]).abs().max() > 1e-5, "coords[1] must affect ch 1"
    torch.testing.assert_close(out_a[0, 2:], out_b[0, 2:], atol=1e-6, rtol=1e-5)


# -------- .lmq montage round-trip (encode -> file -> off-device decode) --------

def test_lmq_montage_roundtrip_off_device():
    """A channel-agnostic .lmq round-trips off-device using ONLY its own montage:
    encode(l3, coords) -> .lmq (carries coords+names) -> decode -> [N,2500].
    No external montage handed to decode — coords travel in the file."""
    import tempfile, os
    import numpy as np
    from lamquant.student.ca_lmq_io import encode_to_lmq, decode_from_lmq
    codec = _build_ca()
    for N in (8, 21, 64):
        x = torch.randn(1, N, 313)
        coords = _coords(1, N)
        chans = [f"EEG E{i}-REF" for i in range(N)]
        path = os.path.join(tempfile.mkdtemp(), "w.lmq")
        encode_to_lmq(codec, x, coords, chans, path)
        recon, meta = decode_from_lmq(codec, path)
        assert recon.shape == (1, N, 2500) and torch.isfinite(recon).all()
        assert meta["n_channels"] == N and meta["channels"] == chans
        np.testing.assert_array_equal(meta["coords"], coords[0].numpy())  # coords bit-exact in file
        # decode-from-file matches a direct decode of the same window (fp16-latent tol)
        codec.train(False)
        with torch.no_grad():
            direct = codec(x, quantize=True, coords=coords)
        torch.testing.assert_close(recon, direct, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except BaseException as e:
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
