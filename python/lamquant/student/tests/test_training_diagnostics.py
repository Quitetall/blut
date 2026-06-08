"""Tests for the training diagnostics battery.

Crucially, these prove the probes FIRE on real bugs (a diagnostic that never
fails is useless): NaN data, frozen model (can't overfit), frozen decoder (no
grad), non-finite step, collapsed latent — each must produce a FAIL.
"""
import torch

from lamquant.student.joint_codec import build_default_joint
from lamquant.student.training_diagnostics import (
    TrainingDiagnostics, PASS, WARN, FAIL)

TIER = 3
torch.manual_seed(0)


def _ca_codec():
    return build_default_joint(vocos_tier=TIER, encoder_width=32, channel_agnostic=True)


def _batch(N=8, B=2):
    x = torch.randn(B, N, 313)
    fb = torch.randn(B, N, 2500)
    coords = torch.randn(B, N, 3) * 0.05
    return x, fb, coords


def _structured(N=8, B=2, T=2500, Tl=313):
    """Smooth, fittable batch — overfit_one_batch needs STRUCTURE (white noise is
    incompressible, so the lossy bottleneck can't fit it; real EEG is structured)."""
    import math
    tt = torch.linspace(0, 1, T)
    tl = torch.linspace(0, 1, Tl)
    fb = torch.zeros(B, N, T)
    x = torch.zeros(B, N, Tl)
    for b in range(B):
        for n in range(N):
            f = 2.0 + (n % 5)
            ph = 0.3 * (b * N + n)
            fb[b, n] = torch.sin(2 * math.pi * f * tt + ph)
            x[b, n] = torch.sin(2 * math.pi * f * tl + ph)
    coords = torch.randn(B, N, 3) * 0.05
    return x, fb, coords


def _status(report, name):
    for r in report.results:
        if r.name == name:
            return r.status
    return None


# ---------------- healthy path ----------------
def test_healthy_preflight_core_checks_pass():
    diag = TrainingDiagnostics(_ca_codec(), channel_agnostic=True)
    x, fb, coords = _structured()
    rep = diag.run_preflight(x, fb, coords, expect_out=2500, overfit_steps=120)
    # structural checks must pass on a healthy model
    assert _status(rep, "data.l3.finite") == PASS
    assert _status(rep, "shape.latent") == PASS
    assert _status(rep, "shape.recon") == PASS
    assert _status(rep, "grad.encoder.nonzero") == PASS
    assert _status(rep, "grad.decoder.nonzero") == PASS
    assert _status(rep, "masked_invariant") == PASS
    assert _status(rep, "coords_routing") == PASS
    assert _status(rep, "overfit_one_batch") == PASS, rep.summary()


# ---------------- probes FIRE on real bugs ----------------
def test_data_sanity_catches_nan():
    diag = TrainingDiagnostics(_ca_codec(), channel_agnostic=True)
    x, fb, _ = _batch()
    x[0, 0, 0] = float("nan")
    res = diag.check_data_sanity(x, fb)
    assert any(r.name == "data.l3.finite" and r.status == FAIL for r in res)


def test_data_sanity_flags_dead_channel():
    diag = TrainingDiagnostics(_ca_codec(), channel_agnostic=True)
    x, fb, _ = _batch()
    x[:, 0, :] = 0.0                                   # a dead (zero-variance) channel
    res = diag.check_data_sanity(x, fb)
    dead = [r for r in res if r.name == "data.l3.dead_channels"][0]
    assert dead.status in (WARN, FAIL) and dead.value > 0


def test_overfit_fails_when_frozen():
    codec = _ca_codec()
    for p in codec.parameters():
        p.requires_grad = False
    diag = TrainingDiagnostics(codec, channel_agnostic=True)
    x, fb, coords = _batch()
    r = diag.overfit_one_batch(x, fb, coords, steps=10)
    assert r.status == FAIL


def test_gradient_flow_detects_frozen_decoder():
    codec = _ca_codec()
    for p in codec.decoder.parameters():
        p.requires_grad = False
    diag = TrainingDiagnostics(codec, channel_agnostic=True)
    x, fb, coords = _batch()
    res = diag.check_gradient_flow(x, fb, coords)
    dec = [r for r in res if r.name == "grad.decoder.nonzero"][0]
    assert dec.status == FAIL


def test_step_probe_catches_nonfinite():
    diag = TrainingDiagnostics(_ca_codec(), channel_agnostic=True)
    res = diag.step_probe(loss=float("nan"), grad_norm=float("inf"))
    assert any(r.name == "step.loss.finite" and r.status == FAIL for r in res)
    assert any(r.name == "step.grad_norm" and r.status == FAIL for r in res)


def test_latent_health_catches_collapse():
    diag = TrainingDiagnostics(_ca_codec(), channel_agnostic=True)
    flat = torch.zeros(2, 32, 79)                      # fully collapsed latent
    res = diag.latent_health(flat)
    coll = [r for r in res if r.name == "latent.collapse"][0]
    assert coll.status == FAIL


def test_shape_contract_latent_n_invariant():
    diag = TrainingDiagnostics(_ca_codec(), channel_agnostic=True)
    for N in (8, 21, 64):
        x = torch.randn(1, N, 313)
        coords = torch.randn(1, N, 3) * 0.05
        res = diag.check_shape_contract(x, coords, expect_out=2500)
        assert _status_in(res, "shape.latent") == PASS
        assert _status_in(res, "shape.recon") == PASS


def test_shape_contract_residual_in_neq_out():
    """Full-residual config (E1, --detail-bands all): encoder in_ch=168 but the
    decoder emits the fullband montage out_ch=21. shape.recon must compare recon
    channels to the FULLBAND TARGET (21), not the encoder input (168).
    Regression for the E1 preflight false-FAIL (2026-06-08). A minimal fake codec
    isolates the shape-contract logic from the encoder's width>=in_ch build
    constraint (the real residual encoder uses width=256 >= 168)."""
    class _FakeEnc:
        def encode(self, x, quantize=False, coords=None, ch_mask=None):
            return torch.zeros(x.shape[0], 32, 79)

    class _FakeResidualCodec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _FakeEnc()

        def forward(self, x, quantize=False, coords=None, ch_mask=None):
            return torch.zeros(x.shape[0], 21, 2500)  # fullband montage output

    diag = TrainingDiagnostics(_FakeResidualCodec(), channel_agnostic=False)
    x = torch.randn(2, 168, 313)    # full residual input (168 ch)
    fb = torch.randn(2, 21, 2500)   # fullband target (21-ch montage)
    # WITH the fullband target → expect 21 output channels → PASS
    res = diag.check_shape_contract(x, fullband=fb)
    assert _status_in(res, "shape.recon") == PASS, [r.detail for r in res]
    # WITHOUT it → falls back to input channels (168) → FAIL (proves the fix
    # discriminates, not a blanket pass).
    res_nofb = diag.check_shape_contract(x)
    assert _status_in(res_nofb, "shape.recon") == FAIL


def test_report_ok_and_summary():
    diag = TrainingDiagnostics(_ca_codec(), channel_agnostic=True)
    x, fb, coords = _batch()
    rep = diag.run_preflight(x, fb, coords, expect_out=2500, overfit_steps=15)
    assert isinstance(rep.summary(), str) and "diagnostics:" in rep.summary()
    assert rep.to_dict()["ok"] == rep.ok


def _status_in(res, name):
    for r in res:
        if r.name == name:
            return r.status
    return None


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
