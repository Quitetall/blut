"""CA-6 wiring tests — train_joint._ca_inputs channel-agnostic input transform.

Bare-import module (matches the student/tests convention, e.g. test_eval_fullband:
`import eval_fullband`). Run with the student dir on PYTHONPATH and WITHOUT the
repo root on sys.path (a `lamquant.py` module there shadows the `lamquant`
package); the BLUT/pytest harness sets this up.
"""
import torch

import train_joint as tj
from lamquant_neural.positions import canonical_21_coords


def test_not_ca_passthrough_is_identity():
    x = torch.randn(2, 21, 313)
    fb = torch.randn(2, 21, 2500)
    xo, fo, co, cm = tj._ca_inputs(x, fb, channel_agnostic=False, variable_n=False)
    assert xo is x and fo is fb and co is None and cm is None


def test_ca_parity_passthrough_no_coords():
    """channel_agnostic + not variable_n -> passthrough; model defaults canonical-21."""
    x = torch.randn(2, 21, 313)
    fb = torch.randn(2, 21, 2500)
    xo, fo, co, cm = tj._ca_inputs(x, fb, channel_agnostic=True, variable_n=False)
    assert xo is x and fo is fb and co is None and cm is None


def test_variable_n_subset_shapes_and_range():
    x = torch.randn(4, 21, 313)
    fb = torch.randn(4, 21, 2500)
    for _ in range(10):
        xs, fs, co, cm = tj._ca_inputs(x, fb, channel_agnostic=True,
                                       variable_n=True, n_range=(8, 21))
        k = co.shape[1]
        assert 8 <= k <= 21
        assert xs.shape == (4, k, 313) and fs.shape == (4, k, 2500)
        assert co.shape == (4, k, 3) and cm is None     # uniform k -> no pad
        assert torch.isfinite(co).all()


def test_variable_n_alignment_invariant():
    """l3, fullband AND coords must all track the SAME electrode per (b,j)."""
    # channel i is constant i in l3, constant i+100 in fullband -> any
    # misalignment is directly visible in the values.
    x = torch.arange(21).float().view(1, 21, 1).expand(4, 21, 313).contiguous()
    fb = (torch.arange(21).float() + 100).view(1, 21, 1).expand(4, 21, 2500).contiguous()
    xs, fs, co, _ = tj._ca_inputs(x, fb, channel_agnostic=True,
                                  variable_n=True, n_range=(8, 21))
    canon = torch.as_tensor(canonical_21_coords())
    k = co.shape[1]
    for b in range(4):
        for j in range(k):
            ch = int(round(xs[b, j, 0].item()))          # electrode id from l3 value
            assert abs(fs[b, j, 0].item() - (ch + 100)) < 1e-4, "l3/fullband misaligned"
            assert torch.allclose(co[b, j], canon[ch], atol=1e-5), "coords misaligned"


def test_variable_n_no_fullband_returns_none():
    x = torch.randn(2, 21, 313)
    xs, fs, co, cm = tj._ca_inputs(x, None, channel_agnostic=True, variable_n=True)
    assert fs is None and co.shape[0] == 2 and xs.shape[1] == co.shape[1]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
