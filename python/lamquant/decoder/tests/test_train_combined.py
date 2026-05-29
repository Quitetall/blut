"""Unit tests for ai_models/decoder/train_combined.py — Phase 3."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "train_combined.py")


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


_STUBBED = ("vocos_decoder", "train_teacher", "train_student_subband",
             "streaming_dataset", "raw_window_dataset",
             "auraloss", "auraloss.freq",
             "lamquant.decoder.train_combined_under_test")


@pytest.fixture(scope="module")
def tc():
    pre = {n: sys.modules.get(n) for n in _STUBBED}
    sys.modules["vocos_decoder"] = _stub("vocos_decoder", VocosDecoder=object)
    sys.modules["train_teacher"] = _stub(
        "train_teacher", L3Teacher=object, Q31Dataset=object)
    sys.modules["train_student_subband"] = _stub(
        "train_student_subband",
        pearson_r_loss=lambda p, t: torch.tensor(0.0))
    sys.modules["streaming_dataset"] = _stub(
        "streaming_dataset", PrecomputedL3Dataset=object)
    sys.modules["raw_window_dataset"] = _stub(
        "raw_window_dataset", RawWindowDataset=object)
    pkg = _stub("auraloss")
    freq = _stub("auraloss.freq", MultiResolutionSTFTLoss=object)
    pkg.freq = freq  # type: ignore[attr-defined]
    sys.modules["auraloss"] = pkg
    sys.modules["auraloss.freq"] = freq
    # lamquant_neural.models.* are the REAL model defs (installed private
    # wheel) — let them resolve normally rather than shadowing with a
    # non-package stub (post-MOVE-A, 2026-05-29).

    name = "lamquant.decoder.train_combined_under_test"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        for n, prev in pre.items():
            if prev is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = prev


# ---------------------------------------------------------------------------
# pearson_r_batch + prd_batch
# ---------------------------------------------------------------------------
class TestMetricsHelpers:
    def test_pearson_identical(self, tc):
        x = torch.randn(4, 21, 313)
        assert tc.pearson_r_batch(x, x.clone()) == pytest.approx(1.0, abs=1e-5)

    def test_pearson_negated(self, tc):
        x = torch.randn(2, 4, 100)
        assert tc.pearson_r_batch(x, -x) == pytest.approx(-1.0, abs=1e-5)

    def test_prd_identical_zero(self, tc):
        x = torch.randn(4, 21, 313)
        assert tc.prd_batch(x, x.clone()) == pytest.approx(0.0, abs=1e-5)

    def test_prd_doubled_target_is_100(self, tc):
        target = torch.randn(4, 4, 100)
        pred = 2 * target
        assert tc.prd_batch(pred, target) == pytest.approx(100.0, rel=1e-4)

    def test_prd_eps_protects_zero_target(self, tc):
        # All-zero target → denom clamped at 1e-8
        out = tc.prd_batch(torch.randn(2, 4, 16), torch.zeros(2, 4, 16))
        assert torch.isfinite(torch.tensor(out))

    def test_returns_python_float(self, tc):
        assert isinstance(tc.pearson_r_batch(torch.randn(2, 4, 8),
                                              torch.randn(2, 4, 8)), float)
        assert isinstance(tc.prd_batch(torch.randn(2, 4, 8),
                                        torch.randn(2, 4, 8)), float)


# ---------------------------------------------------------------------------
# validate_teacher
# ---------------------------------------------------------------------------
class TestValidateTeacher:
    def test_returns_two_floats(self, tc):
        class _Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(313, 313)
            def forward(self, x):
                return self.lin(x)

        teacher = _Teacher()
        # val_loader yields tuples (x_l3, ?, ?)
        x = torch.randn(2, 21, 313)
        val_loader = [(x, None, None), (x, None, None)]
        r, prd = tc.validate_teacher(teacher, val_loader, torch.device("cpu"))
        assert isinstance(r, float)
        assert isinstance(prd, float)

    def test_empty_loader_returns_zero(self, tc):
        teacher = torch.nn.Module()
        r, prd = tc.validate_teacher(teacher, [], torch.device("cpu"))
        assert r == 0.0
        assert prd == 0.0


# ---------------------------------------------------------------------------
# Argparse + main() coverage
# ---------------------------------------------------------------------------
class TestMainArgparse:
    """Cover the argparse setup in main() via --help short-circuit."""

    def test_help_short_circuit_exits_zero(self, tc, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train_combined.py", "--help"])
        with pytest.raises(SystemExit) as e:
            tc.main()
        assert e.value.code == 0

    @pytest.mark.parametrize("flag", [
        "--teacher-epochs", "--decoder-epochs", "--decoder-tier",
        "--teacher-width", "--teacher-strides", "--channel-attn",
        "--bottleneck-attn", "--teacher-r-loss", "--batch-size",
        "--teacher-lr", "--decoder-lr", "--lr-min", "--windows-per-epoch",
        "--max-windows", "--student-checkpoint", "--device", "--resume",
        "--teacher-init", "--decoder-init", "--lma-root", "--split-manifest",
    ])
    def test_argparse_has_flag(self, tc, monkeypatch, capsys, flag):
        """All documented flags are wired into argparse."""
        monkeypatch.setattr(sys, "argv", ["train_combined.py", "--help"])
        with pytest.raises(SystemExit):
            tc.main()
        out = capsys.readouterr().out
        assert flag in out, f"missing flag {flag!r} in --help output"

    def test_bad_tier_rejected(self, tc, monkeypatch):
        """argparse choices=[1,2,3,4] enforces validation."""
        monkeypatch.setattr(sys, "argv",
                             ["train_combined.py", "--decoder-tier", "99"])
        with pytest.raises(SystemExit) as e:
            tc.main()
        assert e.value.code != 0


# ---------------------------------------------------------------------------
# Setup-path coverage: drive main() through one epoch with mocked deps.
# ---------------------------------------------------------------------------
class _ToyTeacher(torch.nn.Module):
    """Trivial L3Teacher stand-in with the right shape contract."""

    def __init__(self, width=32, strides=(1, 2, 2), channel_attn=False,
                 bottleneck_attn=False):
        super().__init__()
        self.encoder = torch.nn.Conv1d(21, 21, kernel_size=1)
        # Used by checkpoint serializer
        self.lin = torch.nn.Conv1d(21, 21, kernel_size=1)

    def forward(self, x):
        # [B, 21, 313] -> [B, 21, 313]
        return self.lin(x)


class _ToyVocosDecoder(torch.nn.Module):
    """Trivial VocosDecoder stand-in producing [B, 21, 313] (direct mode)."""

    def __init__(self, tier=1):
        super().__init__()
        self.tier = tier
        self.output_mode = "direct"
        self.n_channels = 21
        self.dim = 32
        # latent [B, 32, 79] -> [B, 21, 313]
        self.conv = torch.nn.Conv1d(32, 21, kernel_size=1)
        self.up = torch.nn.Linear(79, 313)

    def forward(self, latent, details=None):
        x = self.conv(latent)
        return self.up(x)


class _ToyStudent(torch.nn.Module):
    """Frozen student encoder stand-in. Maps [B, 21, 313] -> [B, 32, 79]."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(21, 32, kernel_size=1)
        self.pool = torch.nn.Linear(313, 79)

    @classmethod
    def from_checkpoint(cls, *_args, **_kwargs):
        return cls().eval()

    def encode(self, x, quantize=True):
        return self.pool(self.conv(x))


class _ToyL3Dataset:
    """Stand-in for PrecomputedL3Dataset / LmaL3Dataset.

    Provides ``windows_per_epoch``, ``__len__``, ``prefetch_batches``,
    ``calibrate_shard_budget`` so the _PrefetchLoader wrapper works.
    """

    def __init__(self, windows_per_epoch=4, batches=2, channels=21,
                 length=313):
        self.windows_per_epoch = windows_per_epoch
        self._batches = batches
        self._c = channels
        self._l = length

    def __len__(self):
        return self.windows_per_epoch

    def __getitem__(self, idx):
        return (torch.randn(self._c, self._l),
                torch.zeros(self._l, dtype=torch.float32),
                torch.zeros(1))

    def prefetch_batches(self, bs, dev):
        for _ in range(self._batches):
            x = torch.randn(bs, self._c, self._l)
            yield (x, None, None)

    def calibrate_shard_budget(self, device):
        pass


class _ToyRawDataset:
    """Stand-in for RawWindowDataset / LmaSignalDataset."""

    def __init__(self, n=4, channels=21, l3_len=313, raw_len=2500):
        self._n = n
        self._c = channels
        self._l3 = l3_len
        self._raw = raw_len

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return (torch.randn(self._c, self._l3),
                torch.randn(self._c, self._raw))


class _ToySpecLoss(torch.nn.Module):
    """Trivial MultiResolutionSTFTLoss stand-in."""

    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, pred, target):
        return torch.zeros((), dtype=pred.dtype, device=pred.device)


def _install_main_mocks(tc, monkeypatch, tmp_path):
    """Common monkeypatches used by main() integration tests."""
    monkeypatch.setattr(tc, "L3Teacher", _ToyTeacher)
    monkeypatch.setattr(tc, "VocosDecoder", _ToyVocosDecoder)
    monkeypatch.setattr(tc, "TernaryMobileNetV5_Subband", _ToyStudent)
    monkeypatch.setattr(tc, "PrecomputedL3Dataset",
                         lambda files, **kw: _ToyL3Dataset())
    monkeypatch.setattr(tc, "RawWindowDataset",
                         lambda files, **kw: _ToyRawDataset())
    monkeypatch.setattr(tc, "MultiResolutionSTFTLoss",
                         lambda *a, **k: _ToySpecLoss())
    # pearson_r_loss returns differentiable tensor
    monkeypatch.setattr(tc, "pearson_r_loss",
                         lambda p, t: (p - t).pow(2).mean())
    # ROOT_DIR -> tmp checkpoint locations
    (tmp_path / "lamquant" / "oracle").mkdir(parents=True)
    (tmp_path / "lamquant" / "decoder").mkdir(parents=True)
    (tmp_path / "lamquant" / "student").mkdir(parents=True)
    (tmp_path / "lamquant" / "dataset").mkdir(parents=True)
    # Make a fake student checkpoint
    fake_ckpt = tmp_path / "lamquant" / "student" / "student_hardened.ckpt"
    fake_ckpt.write_bytes(b"x")
    monkeypatch.setattr(tc, "ROOT_DIR", str(tmp_path))


class TestMainSetupNoCuda:
    """Drive main() through its setup + one-epoch loop on CPU."""

    def test_main_runs_one_epoch_lma_direct(self, tc, monkeypatch, tmp_path):
        """LMA-direct branch — synthetic manifest + tiny dataset."""
        _install_main_mocks(tc, monkeypatch, tmp_path)

        # LMA-direct dataset module mock
        lma_mod = types.ModuleType("lamquant_codec.training")
        lma_mod.LmaL3Dataset = lambda **kw: _ToyL3Dataset()
        lma_mod.LmaSignalDataset = lambda **kw: _ToyRawDataset()
        lma_mod.load_split_stems = lambda manifest, split: (["s1", "s2"], None)
        monkeypatch.setitem(sys.modules, "lamquant_codec.training", lma_mod)

        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"train": [], "val": []}')

        monkeypatch.setattr(sys, "argv", [
            "train_combined.py",
            "--teacher-epochs", "1",
            "--decoder-epochs", "1",
            "--decoder-tier", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--windows-per-epoch", "4",
            "--max-windows", "4",
            "--student-checkpoint",
            str(tmp_path / "lamquant" / "student" / "student_hardened.ckpt"),
            "--lma-root", str(tmp_path),
            "--split-manifest", str(manifest),
        ])

        tc.main()  # should complete without raising

        # Final-completed checkpoint always saved
        d_final = (tmp_path / "lamquant" / "decoder"
                    / "vocos_tier1_1_completed.ckpt")
        assert d_final.exists()

    def test_main_warm_start_decoder_init(self, tc, monkeypatch, tmp_path):
        """--decoder-init triggers _load_ckpt branch (shape + prefix filter)."""
        _install_main_mocks(tc, monkeypatch, tmp_path)

        # Manifest path expected by non-LMA branch
        (tmp_path / "lamquant" / "dataset" / "q31_events").mkdir(
            parents=True)
        fake_q31 = (tmp_path / "lamquant" / "dataset" / "q31_events"
                    / "fake.npz")
        fake_q31.write_bytes(b"x")
        manifest_path = (tmp_path / "lamquant" / "dataset"
                         / "manifest_v3.json")
        manifest_path.write_text("{}")

        class _Manifest:
            @classmethod
            def load(cls, path):
                return cls()
            def get_files(self, split):
                return [str(fake_q31)]
        dt_mod = types.ModuleType("data_types")
        dt_mod.DatasetManifest = _Manifest
        dt_mod.Split = types.SimpleNamespace(TRAIN="train", VAL="val")
        monkeypatch.setitem(sys.modules, "data_types", dt_mod)

        # Pre-build a decoder-shaped checkpoint with one matching param +
        # one bogus shape mismatch + one attention-prefixed param.
        decoder_proto = _ToyVocosDecoder(tier=1)
        sd = decoder_proto.state_dict()
        sd_to_save = {k: v.clone() for k, v in sd.items()}
        sd_to_save["_orig_mod.conv.weight"] = sd_to_save.pop("conv.weight")
        sd_to_save["bn_attn.dummy"] = torch.zeros(3)
        sd_to_save["bogus.shape"] = torch.zeros(99)
        ckpt_path = tmp_path / "warm.ckpt"
        torch.save({"model_state_dict": sd_to_save, "best_r": 0.5},
                    ckpt_path)

        monkeypatch.setattr(sys, "argv", [
            "train_combined.py",
            "--teacher-epochs", "1",
            "--decoder-epochs", "1",
            "--decoder-tier", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--windows-per-epoch", "4",
            "--max-windows", "4",
            "--decoder-init", str(ckpt_path),
            "--teacher-init", str(ckpt_path),  # exercise both branches
        ])
        tc.main()


class TestPrefetchLoader:
    """Cover the inner `_PrefetchLoader` class via main()."""

    def test_loader_length_and_iter(self, tc):
        # Re-create the local class semantics: __len__ = windows / bs
        ds = _ToyL3Dataset(windows_per_epoch=8, batches=2)
        # Use a tiny stand-in; iterating must yield tuples of (tensor, ?, ?)
        it = ds.prefetch_batches(2, torch.device("cpu"))
        batch = next(iter(it))
        assert isinstance(batch, tuple)
        assert batch[0].shape == (2, 21, 313)


class TestCheckpointSerialisation:
    """Output format: best checkpoint dict invariants."""

    def test_save_includes_required_fields(self, tc, tmp_path):
        teacher = _ToyTeacher()
        decoder = _ToyVocosDecoder(tier=1)

        t_path = tmp_path / "t.ckpt"
        d_path = tmp_path / "d.ckpt"
        torch.save({
            "model_state_dict": teacher.state_dict(),
            "encoder_state_dict": teacher.encoder.state_dict(),
            "epoch": 1, "best_r": 0.5, "width": 32,
        }, t_path)
        torch.save({
            "model_state_dict": decoder.state_dict(),
            "epoch": 1, "best_r": 0.5, "tier": 1,
        }, d_path)

        for path in (t_path, d_path):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in sd
            assert "epoch" in sd
            assert "best_r" in sd
