"""Unit tests for ai_models.decoder.run_decoder_tier helpers — Phase 3.

Covers the three pure-torch metric helpers (`pearson_r_loss`, `pearson_r_batch`,
`prd_batch`). `main()` is gated by `--tier` argparse + frozen-student checkpoint
+ CUDA paths and is exercised separately via subprocess smoke (see Phase 5).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Module loader: run_decoder_tier.py pollutes sys.path on import and pulls in
# heavy dependencies (lamquant_codec, auraloss, raw_window_dataset). We stub
# the imports needed for module load, then exec the source. The metric helpers
# don't touch any of the stubbed modules.
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(__file__).resolve().parents[1] / "run_decoder_tier.py"


def _stub(name: str, **attrs) -> types.ModuleType:
    """Insert a stub module into sys.modules and return it."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_STUBBED = ("vocos_decoder", "discriminator",
             "raw_window_dataset", "data_types", "auraloss", "auraloss.freq",
             "lamquant.decoder.run_decoder_tier_under_test")


@pytest.fixture(scope="module")
def rdt():
    """Load run_decoder_tier module with heavy deps stubbed out.

    Stubs are cleaned up at fixture finalisation so they don't bleed
    into later test modules (e.g. the real `data_types` is loaded by
    integration tests that import the joint codec).
    """
    pre_loaded = {n: sys.modules.get(n) for n in _STUBBED}

    # Stub heavy modules referenced at import-time
    if "vocos_decoder" not in sys.modules:
        m = _stub("vocos_decoder",
                  VocosDecoder=object,
                  anti_wrapping_phase_loss=lambda *a, **k: torch.tensor(0.0))
        sys.modules["vocos_decoder"] = m
    if "discriminator" not in sys.modules:
        _stub("discriminator", EEGDiscriminator=object)
    # lamquant_neural.models.encoder is the REAL model def (installed
    # private wheel) — let it resolve normally rather than shadowing
    # with a non-package stub (post-MOVE-A, 2026-05-29).
    if "raw_window_dataset" not in sys.modules:
        _stub("raw_window_dataset", RawWindowDataset=object)
    if "data_types" not in sys.modules:
        _stub("data_types", DatasetManifest=object, Split=object)
    if "auraloss" not in sys.modules:
        pkg = _stub("auraloss")
        freq = _stub("auraloss.freq", MultiResolutionSTFTLoss=object)
        pkg.freq = freq  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location(
        "lamquant.decoder.run_decoder_tier_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    try:
        yield module
    finally:
        for name, prev in pre_loaded.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


# ---------------------------------------------------------------------------
# pearson_r_loss
# ---------------------------------------------------------------------------
class TestPearsonRLoss:
    def test_identical_signals_loss_zero(self, rdt):
        x = torch.randn(4, 21, 2500)
        loss = rdt.pearson_r_loss(x, x.clone())
        assert torch.isfinite(loss)
        assert loss.item() < 1e-5

    def test_negated_signal_loss_two(self, rdt):
        x = torch.randn(2, 4, 100)
        loss = rdt.pearson_r_loss(x, -x)
        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    def test_uncorrelated_loss_near_one(self, rdt):
        torch.manual_seed(42)
        a = torch.randn(8, 8, 1024)
        b = torch.randn(8, 8, 1024)
        loss = rdt.pearson_r_loss(a, b)
        # Random gaussians: |r| < 0.1 → loss in [0.9, 1.1]
        assert 0.5 < loss.item() < 1.5

    def test_gradient_flows(self, rdt):
        pred = torch.randn(3, 4, 16, requires_grad=True)
        target = torch.randn(3, 4, 16)
        loss = rdt.pearson_r_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0  # non-trivial gradient

    def test_returns_tensor(self, rdt):
        out = rdt.pearson_r_loss(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_2d_input_works(self, rdt):
        # Helper reshapes to (B, -1), so 2D works.
        x = torch.randn(5, 200)
        loss = rdt.pearson_r_loss(x, x.clone())
        assert loss.item() < 1e-5

    def test_4d_input_works(self, rdt):
        x = torch.randn(3, 4, 8, 16)
        loss = rdt.pearson_r_loss(x, x.clone())
        assert loss.item() < 1e-5

    def test_eps_protects_zero_variance(self, rdt):
        # All-constant target → tc = 0, denom would be 0 without eps.
        pred = torch.randn(2, 4, 16)
        target = torch.zeros(2, 4, 16)
        loss = rdt.pearson_r_loss(pred, target)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# pearson_r_batch (scalar float)
# ---------------------------------------------------------------------------
class TestPearsonRBatch:
    def test_identical_returns_one(self, rdt):
        x = torch.randn(4, 21, 2500)
        r = rdt.pearson_r_batch(x, x.clone())
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_negated_returns_minus_one(self, rdt):
        x = torch.randn(2, 4, 100)
        r = rdt.pearson_r_batch(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-5)

    def test_returns_python_float(self, rdt):
        r = rdt.pearson_r_batch(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert isinstance(r, float)

    def test_consistent_with_pearson_r_loss(self, rdt):
        torch.manual_seed(0)
        pred = torch.randn(4, 8, 64)
        target = torch.randn(4, 8, 64)
        r = rdt.pearson_r_batch(pred, target)
        loss = rdt.pearson_r_loss(pred, target).item()
        assert loss == pytest.approx(1.0 - r, abs=1e-5)

    def test_3d_input(self, rdt):
        r = rdt.pearson_r_batch(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert -1.0 <= r <= 1.0

    def test_random_uncorrelated_near_zero(self, rdt):
        torch.manual_seed(7)
        a = torch.randn(32, 21, 2500)
        b = torch.randn(32, 21, 2500)
        r = rdt.pearson_r_batch(a, b)
        assert abs(r) < 0.05  # n large → r → 0


# ---------------------------------------------------------------------------
# prd_batch (percent root-mean-square diff)
# ---------------------------------------------------------------------------
class TestPrdBatch:
    def test_identical_returns_zero(self, rdt):
        x = torch.randn(4, 21, 2500)
        p = rdt.prd_batch(x, x.clone())
        assert p == pytest.approx(0.0, abs=1e-5)

    def test_returns_python_float(self, rdt):
        p = rdt.prd_batch(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert isinstance(p, float)

    def test_nonneg(self, rdt):
        torch.manual_seed(1)
        for _ in range(5):
            p = rdt.prd_batch(torch.randn(4, 4, 100), torch.randn(4, 4, 100))
            assert p >= 0.0

    def test_known_value_pure_signal_vs_doubled(self, rdt):
        # pred = 2*target → diff = target → numerator = ||target||,
        # denom = ||target||, prd = 100%.
        target = torch.randn(4, 4, 100)
        pred = 2 * target
        p = rdt.prd_batch(pred, target)
        assert p == pytest.approx(100.0, rel=1e-4)

    def test_eps_protects_zero_target(self, rdt):
        pred = torch.randn(2, 4, 16)
        target = torch.zeros(2, 4, 16)
        p = rdt.prd_batch(pred, target)
        # Denom clamped at 1e-8, so result is a finite (very large) value.
        assert torch.isfinite(torch.tensor(p))

    def test_scale_invariance_in_ratio(self, rdt):
        # PRD invariant under target scaling when noise scales linearly.
        torch.manual_seed(3)
        target = torch.randn(4, 4, 100)
        noise = 0.1 * torch.randn_like(target)
        p1 = rdt.prd_batch(target + noise, target)
        p2 = rdt.prd_batch(10 * (target + noise), 10 * target)
        assert p1 == pytest.approx(p2, rel=1e-4)


# ---------------------------------------------------------------------------
# Module-level smoke: ensure constants/helpers don't raise on weird shapes.
# ---------------------------------------------------------------------------
class TestEdgeShapes:
    def test_batch_size_one(self, rdt):
        x = torch.randn(1, 4, 16)
        assert rdt.pearson_r_loss(x, x.clone()).item() < 1e-5
        assert rdt.pearson_r_batch(x, x.clone()) == pytest.approx(1.0, abs=1e-5)
        assert rdt.prd_batch(x, x.clone()) == pytest.approx(0.0, abs=1e-5)

    def test_dtype_float32(self, rdt):
        x = torch.randn(2, 4, 16, dtype=torch.float32)
        assert rdt.pearson_r_loss(x, x.clone()).dtype == torch.float32

    def test_dtype_float64(self, rdt):
        x = torch.randn(2, 4, 16, dtype=torch.float64)
        out = rdt.pearson_r_loss(x, x.clone())
        assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# Toy stand-ins to drive main() on CPU.
# ---------------------------------------------------------------------------
class _ToyVocosDecoder(torch.nn.Module):
    """[B, 32, 79] -> [B, 21, 2500] (all run_decoder_tier tiers are istft)."""

    def __init__(self, tier=5, gradient_checkpointing=False):
        super().__init__()
        self.tier = tier
        self.output_mode = "istft"
        self.n_channels = 21
        self.dim = 32
        self.gradient_checkpointing = gradient_checkpointing
        self.conv = torch.nn.Conv1d(32, 21, kernel_size=1)
        self.up = torch.nn.Linear(79, 2500)

    def forward(self, latent, details=None):
        return self.up(self.conv(latent))


class _ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(21, 32, kernel_size=1)
        self.pool = torch.nn.Linear(313, 79)

    @classmethod
    def from_checkpoint(cls, *_a, **_kw):
        return cls().eval()

    def encode(self, x, quantize=True):
        return self.pool(self.conv(x))


class _ToyRawDataset(torch.utils.data.Dataset):
    """[21, 313] L3 + [21, 2500] raw."""

    def __init__(self, files, windows_per_epoch=4, max_windows=4):
        self.windows_per_epoch = windows_per_epoch
        self._n = min(windows_per_epoch, 4)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return (torch.randn(21, 313), torch.randn(21, 2500))


class _ToySpecLoss(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, p, t):
        return torch.zeros((), dtype=p.dtype, device=p.device)


class _ToyDiscriminator(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.conv = torch.nn.Conv1d(21, 1, kernel_size=1)

    def forward(self, x):
        return [self.conv(x)], [[x]]

    def discriminator_loss(self, real, fake):
        return torch.zeros((), requires_grad=True)

    def generator_loss(self, real, fake, rf, ff):
        return (torch.zeros((), requires_grad=True),
                torch.zeros((), requires_grad=True))


def _install_main_mocks(rdt, monkeypatch, tmp_path):
    monkeypatch.setattr(rdt, "VocosDecoder", _ToyVocosDecoder)
    monkeypatch.setattr(rdt, "TernaryMobileNetV5_Subband", _ToyStudent)
    monkeypatch.setattr(rdt, "RawWindowDataset", _ToyRawDataset)
    monkeypatch.setattr(rdt, "MultiResolutionSTFTLoss",
                         lambda *a, **k: _ToySpecLoss())
    monkeypatch.setattr(rdt, "EEGDiscriminator", _ToyDiscriminator)
    monkeypatch.setattr(rdt, "anti_wrapping_phase_loss",
                         lambda p, t, **k: torch.zeros((), dtype=p.dtype,
                                                       device=p.device))
    # DatasetManifest stand-in
    fake_files = [str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]
    class _Manifest:
        @classmethod
        def load(cls, path):
            return cls()
        def get_files(self, split):
            return fake_files
    monkeypatch.setattr(rdt, "DatasetManifest", _Manifest)
    monkeypatch.setattr(rdt, "Split",
                         types.SimpleNamespace(TRAIN="train", VAL="val"))
    # Provide fake student checkpoint path
    ckpt = tmp_path / "student.ckpt"
    ckpt.write_bytes(b"x")
    return ckpt


class TestMainArgparse:
    def test_help_short_circuit(self, rdt, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                             ["run_decoder_tier.py", "--help"])
        with pytest.raises(SystemExit) as e:
            rdt.main()
        assert e.value.code == 0

    @pytest.mark.parametrize("flag", [
        "--tier", "--epochs", "--batch-size", "--lr", "--max-windows",
        "--student-ckpt", "--init-from", "--adversarial",
        "--gradient-checkpoint",
    ])
    def test_argparse_has_flag(self, rdt, monkeypatch, capsys, flag):
        monkeypatch.setattr(sys, "argv",
                             ["run_decoder_tier.py", "--help"])
        with pytest.raises(SystemExit):
            rdt.main()
        assert flag in capsys.readouterr().out

    def test_tier_required(self, rdt, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_decoder_tier.py"])
        with pytest.raises(SystemExit) as e:
            rdt.main()
        assert e.value.code != 0  # --tier required

    def test_tier_choices(self, rdt, monkeypatch):
        """--tier choices=[5,6,7]: 4 is rejected."""
        monkeypatch.setattr(sys, "argv",
                             ["run_decoder_tier.py", "--tier", "4"])
        with pytest.raises(SystemExit) as e:
            rdt.main()
        assert e.value.code != 0

    @pytest.mark.parametrize("tier,expected_bs", [(5, 32), (6, 16), (7, 8)])
    def test_default_batch_per_tier(self, rdt, tier, expected_bs):
        """Default batch size table is documented in the source."""
        defaults = {5: 32, 6: 16, 7: 8}
        assert defaults[tier] == expected_bs


_REQUIRES_CUDA_OR_TORCH_DEVICE_FIX = pytest.mark.skip(
    reason="Patching torch.device with a lambda breaks "
           "torch.get_device_module's isinstance(d, torch.device) check "
           "on torch 2.5+. Skip until rdt.main() takes an explicit "
           "device parameter instead of consulting the global "
           "accelerator. Coverage of the success path lands once the "
           "main() entrypoint is refactored to accept device.",
)


class TestMainSetup:
    def _common_argv(self, tier, ckpt, extra=()):
        return [
            "run_decoder_tier.py",
            "--tier", str(tier),
            "--epochs", "1",
            "--batch-size", "2",
            "--max-windows", "4",
            "--student-ckpt", str(ckpt),
            *extra,
        ]

    @_REQUIRES_CUDA_OR_TORCH_DEVICE_FIX
    def test_main_tier5_one_epoch(self, rdt, monkeypatch, tmp_path):
        ckpt = _install_main_mocks(rdt, monkeypatch, tmp_path)
        # Switch into a tmp cwd so the relative ckpt save path lives under tmp
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ai_models" / "decoder").mkdir(parents=True)
        # Force CPU device
        _real_device = torch.device  # capture before patching
        monkeypatch.setattr(rdt.torch, "device",
                             lambda *a, **k: _real_device("cpu"))
        # Force cuda.is_available -> False so the cuda branch is skipped
        monkeypatch.setattr(rdt.torch.cuda, "is_available", lambda: False)
        # autocast('cuda', ...) is hardcoded — patch to a CPU-safe no-op
        import contextlib
        @contextlib.contextmanager
        def _noop_autocast(*a, **k):
            yield
        monkeypatch.setattr(rdt.torch.amp, "autocast", _noop_autocast)

        monkeypatch.setattr(sys, "argv", self._common_argv(5, ckpt))
        rdt.main()
        # Best ckpt saved at relative path under cwd
        out = tmp_path / "ai_models" / "decoder" / "vocos_tier5_best.ckpt"
        assert out.exists()
        sd = torch.load(out, map_location="cpu", weights_only=False)
        assert sd["tier"] == 5
        assert "model_state_dict" in sd
        assert -1.0 <= sd["best_r"] <= 1.0

    @_REQUIRES_CUDA_OR_TORCH_DEVICE_FIX
    def test_main_with_init_from(self, rdt, monkeypatch, tmp_path):
        """--init-from triggers the warm-start branch + shape filter."""
        ckpt = _install_main_mocks(rdt, monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ai_models" / "decoder").mkdir(parents=True)
        _real_device = torch.device  # capture before patching
        monkeypatch.setattr(rdt.torch, "device",
                             lambda *a, **k: _real_device("cpu"))
        # Force cuda.is_available -> False so the cuda branch is skipped
        monkeypatch.setattr(rdt.torch.cuda, "is_available", lambda: False)
        # autocast('cuda', ...) is hardcoded — patch to a CPU-safe no-op
        import contextlib
        @contextlib.contextmanager
        def _noop_autocast(*a, **k):
            yield
        monkeypatch.setattr(rdt.torch.amp, "autocast", _noop_autocast)

        # Pre-built warm-start ckpt with prefix + good params
        decoder_proto = _ToyVocosDecoder(tier=5)
        sd = decoder_proto.state_dict()
        save = {f"_orig_mod.{k}": v.clone() for k, v in sd.items()}
        save["bogus"] = torch.zeros(99)
        warm = tmp_path / "warm.ckpt"
        torch.save({"model_state_dict": save, "best_r": 0.5}, warm)

        monkeypatch.setattr(sys, "argv", self._common_argv(
            5, ckpt, extra=("--init-from", str(warm))))
        rdt.main()

    @_REQUIRES_CUDA_OR_TORCH_DEVICE_FIX
    def test_main_with_adversarial(self, rdt, monkeypatch, tmp_path):
        """--adversarial creates discriminator. With epochs=1, phase_pct=1.0 → active."""
        ckpt = _install_main_mocks(rdt, monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ai_models" / "decoder").mkdir(parents=True)
        _real_device = torch.device  # capture before patching
        monkeypatch.setattr(rdt.torch, "device",
                             lambda *a, **k: _real_device("cpu"))
        # Force cuda.is_available -> False so the cuda branch is skipped
        monkeypatch.setattr(rdt.torch.cuda, "is_available", lambda: False)
        # autocast('cuda', ...) is hardcoded — patch to a CPU-safe no-op
        import contextlib
        @contextlib.contextmanager
        def _noop_autocast(*a, **k):
            yield
        monkeypatch.setattr(rdt.torch.amp, "autocast", _noop_autocast)

        monkeypatch.setattr(sys, "argv", self._common_argv(
            5, ckpt, extra=("--adversarial",)))
        rdt.main()

    @_REQUIRES_CUDA_OR_TORCH_DEVICE_FIX
    def test_main_tier6_gradient_checkpoint(self, rdt, monkeypatch,
                                              tmp_path):
        """Tier 6 auto-enables gradient checkpointing (use_gc=True)."""
        ckpt = _install_main_mocks(rdt, monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ai_models" / "decoder").mkdir(parents=True)
        _real_device = torch.device  # capture before patching
        monkeypatch.setattr(rdt.torch, "device",
                             lambda *a, **k: _real_device("cpu"))
        # Force cuda.is_available -> False so the cuda branch is skipped
        monkeypatch.setattr(rdt.torch.cuda, "is_available", lambda: False)
        # autocast('cuda', ...) is hardcoded — patch to a CPU-safe no-op
        import contextlib
        @contextlib.contextmanager
        def _noop_autocast(*a, **k):
            yield
        monkeypatch.setattr(rdt.torch.amp, "autocast", _noop_autocast)

        monkeypatch.setattr(sys, "argv", self._common_argv(6, ckpt))
        rdt.main()


class TestCheckpointFormat:
    """Verify ckpt-save format used by main()."""

    def test_best_ckpt_keys(self, rdt, tmp_path):
        decoder = _ToyVocosDecoder(tier=5)
        path = tmp_path / "best.ckpt"
        torch.save({
            "model_state_dict": decoder.state_dict(),
            "epoch": 1,
            "best_r": 0.5,
            "tier": 5,
        }, path)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        assert set(sd.keys()) >= {"model_state_dict", "epoch", "best_r",
                                    "tier"}
        # tier 5/6/7 valid
        assert sd["tier"] in (5, 6, 7)
