"""Unit tests for ai_models/decoder/train_vocos_decoder.py — Phase 3.

Covers `pearson_r_batch` + `discover_student_checkpoint`. The main()
training loop needs real data + GPU + heavy deps (auraloss,
streaming_dataset, perceptual_losses); covered indirectly via the
integration suite.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "train_vocos_decoder.py")


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


_STUBBED = ("vocos_decoder", "train_student_subband", "auraloss",
             "auraloss.freq", "streaming_dataset", "flow_postfilter",
             "perceptual_losses",
             "lamquant.decoder.train_vocos_decoder_under_test")


@pytest.fixture(scope="module")
def tvd():
    """Load train_vocos_decoder with heavy deps stubbed out.

    Snapshot/restore sys.modules to avoid leaking stubs into other tests.
    """
    pre = {n: sys.modules.get(n) for n in _STUBBED}

    if "vocos_decoder" not in sys.modules:
        sys.modules["vocos_decoder"] = _stub("vocos_decoder",
                                              VocosDecoder=object)
    if "train_student_subband" not in sys.modules:
        sys.modules["train_student_subband"] = _stub(
            "train_student_subband",
            pearson_r_loss=lambda p, t: torch.tensor(0.0))
    if "auraloss" not in sys.modules:
        pkg = _stub("auraloss")
        freq = _stub("auraloss.freq", MultiResolutionSTFTLoss=object)
        pkg.freq = freq  # type: ignore[attr-defined]
        sys.modules["auraloss"] = pkg
        sys.modules["auraloss.freq"] = freq
    if "streaming_dataset" not in sys.modules:
        sys.modules["streaming_dataset"] = _stub(
            "streaming_dataset", PrecomputedL3Dataset=object)
    if "flow_postfilter" not in sys.modules:
        sys.modules["flow_postfilter"] = _stub(
            "flow_postfilter", CFMPostfilter=object)
    if "perceptual_losses" not in sys.modules:
        sys.modules["perceptual_losses"] = _stub(
            "perceptual_losses", MultiTeacherPerceptualLoss=object)
    # lamquant_neural.models.{vocos_decoder,encoder} are the REAL model
    # defs (installed private wheel) — let them resolve normally rather
    # than shadowing with a non-package stub (post-MOVE-A, 2026-05-29).

    name = "lamquant.decoder.train_vocos_decoder_under_test"
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
# pearson_r_batch
# ---------------------------------------------------------------------------
class TestPearsonRBatch:
    def test_identical_returns_one(self, tvd):
        x = torch.randn(4, 21, 313)
        r = tvd.pearson_r_batch(x, x.clone())
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_negated_returns_minus_one(self, tvd):
        x = torch.randn(2, 4, 100)
        r = tvd.pearson_r_batch(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-5)

    def test_returns_python_float(self, tvd):
        r = tvd.pearson_r_batch(torch.randn(2, 4, 16),
                                 torch.randn(2, 4, 16))
        assert isinstance(r, float)

    def test_random_uncorrelated_near_zero(self, tvd):
        torch.manual_seed(0)
        r = tvd.pearson_r_batch(torch.randn(32, 21, 313),
                                 torch.randn(32, 21, 313))
        assert abs(r) < 0.05


# ---------------------------------------------------------------------------
# discover_student_checkpoint
# ---------------------------------------------------------------------------
class TestDiscoverCheckpoint:
    def test_explicit_path_returned(self, tvd, tmp_path):
        p = tmp_path / "explicit.ckpt"
        p.write_text("x")
        assert tvd.discover_student_checkpoint(str(p)) == str(p)

    def test_explicit_missing_exits_1(self, tvd, tmp_path):
        with pytest.raises(SystemExit) as e:
            tvd.discover_student_checkpoint(str(tmp_path / "nope.ckpt"))
        assert e.value.code == 1

    def test_no_args_no_candidates_exits(self, tvd, monkeypatch, tmp_path):
        # Override ROOT_DIR so no candidate paths exist
        monkeypatch.setattr(tvd, "ROOT_DIR", str(tmp_path))
        with pytest.raises(SystemExit) as e:
            tvd.discover_student_checkpoint()
        assert e.value.code == 1

    def test_finds_first_candidate(self, tvd, monkeypatch, tmp_path):
        monkeypatch.setattr(tvd, "ROOT_DIR", str(tmp_path))
        # Create one candidate
        (tmp_path / "ai_models" / "student").mkdir(parents=True)
        ckpt = tmp_path / "ai_models" / "student" / "student_hardened.ckpt"
        ckpt.write_text("x")
        result = tvd.discover_student_checkpoint()
        assert os.path.basename(result) == "student_hardened.ckpt"


# ---------------------------------------------------------------------------
# Toy modules for main() integration tests (CPU, 1 epoch, tiny tensors).
# ---------------------------------------------------------------------------
class _ToyVocosDecoder(torch.nn.Module):
    """[B, 32, 79] -> [B, 21, expected_out_len] per tier output_mode."""

    def __init__(self, tier=1):
        super().__init__()
        self.tier = tier
        # Tiers 1-2 are direct, 3+ iSTFT. Force tier 1 -> direct
        self.output_mode = "direct" if tier <= 2 else "istft"
        self.n_channels = 21
        self.dim = 32
        self.conv = torch.nn.Conv1d(32, 21, kernel_size=1)
        self.up = torch.nn.Linear(
            79, 313 if self.output_mode == "direct" else 2500)

    def forward(self, latent, details=None):
        return self.up(self.conv(latent))


class _ToyStudent(torch.nn.Module):
    """[B, 21, 313] -> [B, 32, 79] latent."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(21, 32, kernel_size=1)
        self.pool = torch.nn.Linear(313, 79)

    @classmethod
    def from_checkpoint(cls, *_a, **_kw):
        return cls().eval()

    def encode(self, x, quantize=True):
        return self.pool(self.conv(x))


class _ToyL3Dataset(torch.utils.data.Dataset):
    def __init__(self, n=4, channels=21, length=313):
        self._n = n
        self._c = channels
        self._l = length

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return (torch.randn(self._c, self._l),
                torch.zeros(1),
                torch.zeros(1))


class _ToySpecLoss(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, p, t):
        return torch.zeros((), dtype=p.dtype, device=p.device)


class _ToyDiscriminator(torch.nn.Module):
    """EEGDiscriminator stand-in."""

    def __init__(self, *a, **k):
        super().__init__()
        self.conv = torch.nn.Conv1d(21, 1, kernel_size=1)

    def forward(self, x):
        # Returns (scores, feats)
        return [self.conv(x)], [[x]]

    def discriminator_loss(self, real_scores, fake_scores):
        return torch.zeros((), requires_grad=True)

    def generator_loss(self, real, fake, rf, ff):
        return (torch.zeros((), requires_grad=True),
                torch.zeros((), requires_grad=True))


def _install_main_mocks(tvd, monkeypatch, tmp_path):
    monkeypatch.setattr(tvd, "VocosDecoder", _ToyVocosDecoder)
    monkeypatch.setattr(tvd, "TernaryMobileNetV5_Subband", _ToyStudent)
    monkeypatch.setattr(tvd, "PrecomputedL3Dataset",
                         lambda files, **kw: _ToyL3Dataset())
    monkeypatch.setattr(tvd, "MultiResolutionSTFTLoss",
                         lambda *a, **k: _ToySpecLoss())
    monkeypatch.setattr(tvd, "pearson_r_loss",
                         lambda p, t: (p - t).pow(2).mean())
    (tmp_path / "ai_models" / "decoder").mkdir(parents=True)
    (tmp_path / "ai_models" / "student").mkdir(parents=True)
    (tmp_path / "ai_models" / "dataset_sim" / "q31_events").mkdir(
        parents=True)
    fake_ckpt = tmp_path / "ai_models" / "student" / "student_hardened.ckpt"
    fake_ckpt.write_bytes(b"x")
    fake_q31 = tmp_path / "ai_models" / "dataset_sim" / "q31_events" / "a.npz"
    fake_q31.write_bytes(b"x")
    monkeypatch.setattr(tvd, "ROOT_DIR", str(tmp_path))
    return fake_ckpt, fake_q31


# ---------------------------------------------------------------------------
# Argparse coverage
# ---------------------------------------------------------------------------
class TestMainArgparse:
    def test_help_short_circuit(self, tvd, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                             ["train_vocos_decoder.py", "--help"])
        with pytest.raises(SystemExit) as e:
            tvd.main()
        assert e.value.code == 0

    @pytest.mark.parametrize("flag", [
        "--tier", "--epochs", "--batch-size", "--lr", "--lr-min",
        "--windows-per-epoch", "--max-windows", "--student-checkpoint",
        "--device", "--resume", "--adversarial", "--adv-start-epoch",
        "--adv-ramp-epochs", "--disc-lr", "--cfm-postfilter",
        "--cfm-start-epoch", "--perceptual-loss", "--perceptual-weight",
        "--dac-init", "--lma-root", "--split-manifest",
    ])
    def test_argparse_has_flag(self, tvd, monkeypatch, capsys, flag):
        monkeypatch.setattr(sys, "argv",
                             ["train_vocos_decoder.py", "--help"])
        with pytest.raises(SystemExit):
            tvd.main()
        assert flag in capsys.readouterr().out

    def test_tier_choices_enforced(self, tvd, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                             ["train_vocos_decoder.py", "--tier", "99"])
        with pytest.raises(SystemExit) as e:
            tvd.main()
        assert e.value.code != 0


# ---------------------------------------------------------------------------
# main() integration via mocks
# ---------------------------------------------------------------------------
class TestMainSetup:
    def test_main_tier1_one_epoch_legacy(self, tvd, monkeypatch, tmp_path):
        """Legacy NPZ branch: glob finds files, dataset constructs, 1 epoch."""
        _install_main_mocks(tvd, monkeypatch, tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "train_vocos_decoder.py",
            "--tier", "1",
            "--epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
        ])
        tvd.main()
        final = (tmp_path / "ai_models" / "decoder"
                 / "vocos_tier1_1_completed.ckpt")
        assert final.exists()
        sd = torch.load(final, map_location="cpu", weights_only=False)
        assert "model_state_dict" in sd
        assert sd["tier"] == 1
        assert sd["epoch"] == 1

    def test_main_lma_direct_one_epoch(self, tvd, monkeypatch, tmp_path):
        _install_main_mocks(tvd, monkeypatch, tmp_path)
        # LMA-direct dataset module mock
        lma_mod = types.ModuleType("lamquant_codec.training")
        lma_mod.LmaL3Dataset = lambda **kw: _ToyL3Dataset()
        lma_mod.load_split_stems = lambda m, s: (["s1"], None)
        monkeypatch.setitem(sys.modules, "lamquant_codec.training", lma_mod)
        manifest = tmp_path / "m.json"
        manifest.write_text("{}")
        monkeypatch.setattr(sys, "argv", [
            "train_vocos_decoder.py",
            "--tier", "1",
            "--epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--lma-root", str(tmp_path),
            "--split-manifest", str(manifest),
        ])
        tvd.main()

    def test_main_resume_branch(self, tvd, monkeypatch, tmp_path):
        """--resume reads an existing best ckpt and restores epoch/best_r."""
        _install_main_mocks(tvd, monkeypatch, tmp_path)
        # Pre-create best ckpt for resume
        decoder = _ToyVocosDecoder(tier=1)
        opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
        best = tmp_path / "ai_models" / "decoder" / "vocos_tier1_best.ckpt"
        torch.save({
            "model_state_dict": decoder.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "epoch": 0, "best_r": 0.5, "tier": 1,
        }, best)
        monkeypatch.setattr(sys, "argv", [
            "train_vocos_decoder.py",
            "--tier", "1",
            "--epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--resume",
        ])
        tvd.main()

    def test_main_adversarial_branch(self, tvd, monkeypatch, tmp_path):
        """--adversarial creates discriminator + optimizer."""
        _install_main_mocks(tvd, monkeypatch, tmp_path)
        # Stub discriminator module
        disc_mod = types.ModuleType("discriminator")
        disc_mod.EEGDiscriminator = _ToyDiscriminator
        monkeypatch.setitem(sys.modules, "discriminator", disc_mod)
        monkeypatch.setattr(sys, "argv", [
            "train_vocos_decoder.py",
            "--tier", "1",
            "--epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--adversarial",
            "--adv-start-epoch", "0",  # active from epoch 0
            "--adv-ramp-epochs", "1",
        ])
        tvd.main()

    def test_main_perceptual_and_cfm(self, tvd, monkeypatch, tmp_path):
        """--perceptual-loss + --cfm-postfilter flag-setup branches."""
        _install_main_mocks(tvd, monkeypatch, tmp_path)
        # Replace heavy MultiTeacherPerceptualLoss + CFMPostfilter with toys
        monkeypatch.setattr(tvd, "MultiTeacherPerceptualLoss",
                             lambda device: torch.nn.Identity())
        monkeypatch.setattr(tvd, "CFMPostfilter",
                             lambda channels, dim: torch.nn.Identity())
        monkeypatch.setattr(sys, "argv", [
            "train_vocos_decoder.py",
            "--tier", "1",
            "--epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
            "--perceptual-loss",
            "--cfm-postfilter",
        ])
        tvd.main()


class TestCheckpointFormat:
    """Final + best checkpoint dict structure invariants."""

    def test_final_ckpt_keys(self, tvd, tmp_path):
        decoder = _ToyVocosDecoder(tier=1)
        path = tmp_path / "final.ckpt"
        torch.save({
            "model_state_dict": decoder.state_dict(),
            "epoch": 1,
            "best_r": 0.5,
            "tier": 1,
        }, path)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        assert set(sd.keys()) >= {"model_state_dict", "epoch", "best_r", "tier"}
