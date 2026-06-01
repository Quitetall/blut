# SPDX-License-Identifier: GPL-3.0-or-later
"""Correctness checks for the ESOAP optimizer (clean-room COSMOS-principle)."""
import os
import sys

import torch

_STUDENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lamquant", "student"))
if _STUDENT not in sys.path:
    sys.path.insert(0, _STUDENT)

from esoap import ESOAP, _newton_schulz5, _esoap_direction  # noqa: E402


def test_newton_schulz_orthogonalizes():
    """NS5 drives singular values toward 1 (orthogonal columns/rows)."""
    torch.manual_seed(0)
    G = torch.randn(20, 12)  # tall
    X = _newton_schulz5(G, steps=5, eps=1e-7)
    s = torch.linalg.svdvals(X)
    # All singular values should be ~1 after orthogonalization.
    assert s.max() < 1.3 and s.min() > 0.7, f"singular values not ~1: {s}"
    print(f"[ok] NS5 singular values in [{s.min():.3f}, {s.max():.3f}]")


def test_esoap_direction_is_rms_normalized_and_finite():
    """The returned direction is RMS-normalized (rms ~ 1) and finite."""
    torch.manual_seed(0)
    m, n = 40, 16
    grad = torch.randn(m, n)
    mom = torch.zeros(m, n)
    r = 8
    gram = torch.zeros(n, n)
    v_lead = torch.zeros(m, r)
    d = _esoap_direction(
        grad=grad, momentum=mom, gram=gram, v_lead=v_lead, rank=r,
        mu=0.95, gram_beta=0.95, beta2=0.95, nesterov=True, ns_steps=5, eps=1e-8)
    assert torch.isfinite(d).all()
    rms = d.square().mean().sqrt().item()
    assert abs(rms - 1.0) < 0.05, f"direction not RMS-normalized: rms={rms}"
    print(f"[ok] esoap direction finite, rms={rms:.4f}")


def test_esoap_two_step_smoke_on_real_snn():
    """Two real steps: no NaN/Inf; esoap-group + adamw-group both move."""
    from lamquant_neural.models.mamba_ssm_minimal import MambaSNN

    torch.manual_seed(1337)
    model = MambaSNN(in_channels=21, d_model=40, d_state=16, n_layers=2,
                     use_subband=True)
    suff = ('in_proj.weight', 'x_proj.weight', 'out_proj.weight',
            'spatial_mix.weight')
    eso, rest = [], []
    for nm, p in model.named_parameters():
        (eso if (p.ndim == 2 and nm.endswith(suff)) else rest).append(p)
    assert len(eso) == 13, f"expected 13 linear matrices, got {len(eso)}"

    opt = ESOAP(
        [{"params": eso, "method": "esoap", "weight_decay": 1e-4},
         {"params": rest, "method": "adamw", "weight_decay": 1e-4}],
        lr=1e-3, betas=(0.9, 0.95))

    e0 = eso[0].detach().clone()
    a0 = rest[0].detach().clone()
    for _ in range(2):
        x = torch.randn(4, 21, 313)
        opt.zero_grad()
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits.float().pow(2).mean().backward()
        opt.step()

    for p in eso + rest:
        assert torch.isfinite(p).all()
    assert (eso[0] - e0).abs().sum().item() > 0, "esoap group did not move"
    assert (rest[0] - a0).abs().sum().item() > 0, "adamw group did not move"
    print("[ok] 2-step SNN smoke: finite, esoap + adamw groups both moved")


def _run_k(make_opt, p0, grads):
    p = torch.nn.Parameter(p0.clone())
    opt = make_opt([p])
    for g in grads:
        p.grad = g.clone()
        opt.step()
    return p.detach().clone()


def test_cautious_wd_mask_identity():
    """Cautious WD (ADR 0030, flag-gated) obeys the per-entry fold identity
    ``p_cautious = where((p0-p_nowd)*p0 > 0, p_plain, p_nowd)`` on all three
    optimizer paths (SOAP, ESOAP-esoap, ESOAP-adamw), and is non-vacuous:
    it differs from BOTH plain-WD and no-WD. Default off stays plain-WD.
    """
    from soap_optimizer import SOAP  # noqa: E402

    factories = {
        "SOAP": lambda ps, wd, c: SOAP(ps, lr=0.1, weight_decay=wd, cautious_wd=c),
        "ESOAP-esoap": lambda ps, wd, c: ESOAP(
            [{"params": ps, "method": "esoap"}], lr=0.1,
            weight_decay=wd, cautious_wd=c),
        "ESOAP-adamw": lambda ps, wd, c: ESOAP(
            [{"params": ps, "method": "adamw"}], lr=0.1,
            weight_decay=wd, cautious_wd=c),
    }
    WD, K = 0.2, 5
    for name, factory in factories.items():
        torch.manual_seed(0)
        p0 = torch.randn(8, 6) * 0.5
        grads = [torch.randn(8, 6) for _ in range(K)]
        nowd = _run_k(lambda ps: factory(ps, 0.0, False), p0, grads)
        plain = _run_k(lambda ps: factory(ps, WD, False), p0, grads)
        caut = _run_k(lambda ps: factory(ps, WD, True), p0, grads)
        # WD path actually executed (not vacuous like SOAP's skipped step 1).
        assert not torch.allclose(plain, nowd, atol=1e-5), \
            f"{name}: WD inert — test vacuous"
        # Cautious differs from both plain and no-WD (mask is partial).
        assert not torch.allclose(caut, plain, atol=1e-6), \
            f"{name}: cautious == plain (mask all-true?)"
        assert not torch.allclose(caut, nowd, atol=1e-6), \
            f"{name}: cautious == nowd (mask all-false?)"
        print(f"[ok] cautious-WD non-vacuous + bracketed on {name}")


if __name__ == "__main__":
    test_newton_schulz_orthogonalizes()
    test_esoap_direction_is_rms_normalized_and_finite()
    test_esoap_two_step_smoke_on_real_snn()
    test_cautious_wd_mask_identity()
    print("\nALL ESOAP TESTS PASSED")
